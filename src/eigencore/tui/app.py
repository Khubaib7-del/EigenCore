"""
Full-screen chat TUI for `eigencore chat` on a TTY (owner decision 2026-11-11:
chat goes full-screen like Gemini CLI; `--plain`/non-TTY keeps the line-mode
`tui/chat.py` ChatUI). Zero new dependencies: Rich Live + stdlib key reader.

Layout (docs/design/TUI.md §3.1, handbook 5.3):

    ◩ EigenCore <version> · chat                          local · ready
    model <name> <quant>   ctx ≈<used> / <set> tok   fit ok +<h> GB est.   tools off
    ──────────────────────────────────────────────────────────────────────
    you
      <message>

    eigencore
      <markdown reply, framed code blocks>
      ◐ Writing… 2.1 s            (agent state line while streaming)
      decode <n> tok/s · prefill <t> s · <n> tokens · measured this turn
    <status>                                   ctx ███░░░░░░░ ≈3% of 4096 tok
    ──────────────────────────────────────────────────────────────────────
    › message   /help /model /context /clear /exit             ctrl-c stops

Narrow (< 100 columns): header collapses to two short lines, no rules, footer
shows `› message · /help`. Overlays (model picker, help, context, confirm,
SSD-paging warning) replace the transcript area.

Safety rules kept from docs/design/TUI.md: model text is sanitised and never
parsed as Rich markup; Ctrl-C stops the stream without losing history; a
mid-stream error never regenerates; every number carries its unit and kind
("measured this turn", "≈" estimate, "est."); deletes remove files from disk
via the registry; the tools label reads the attached policy (none → "off").
"""

from __future__ import annotations

import contextlib
import difflib
import io
import queue
import sys
import threading
import time
from typing import Callable, Optional

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from eigencore import __version__
from eigencore.engine.inference import GenerationConfig
from eigencore.fit import FitVerdict, ModelGeometry, estimate_fit, read_gguf_model_info
from eigencore.hal.profiler import HardwareProfile
from eigencore.models.registry import ModelRegistry, ModelSpec
from eigencore.tui import keys as K
from eigencore.tui import theme as T
from eigencore.tui.chat import ChatUiConfig
from eigencore.tui.components import SsdWarning, tools_label
from eigencore.tui.render import MD_THEME, TurnStats, open_fence, render_markdown, stats_from_run
from eigencore.tui.sanitize import sanitize
from eigencore.tui.theme import TerminalCaps, probe_terminal

sanitize_model_text = sanitize  # backwards-compatible name

SLASH_COMMANDS = (
    "/help",
    "/models",
    "/model",
    "/download",
    "/delete",
    "/context",
    "/clear",
    "/tokens",
    "/temp",
    "/exit",
)

FOOTER_HINTS = ("/help", "/model", "/context", "/clear", "/exit")
PREFILL_VERBS = ("Thinking", "Analyzing", "Reading prompt")
PLACEHOLDER = "message"


def pick_glyphs(ascii: bool | None = None) -> dict:
    """Unicode glyphs when the terminal can show them, else ASCII (theme.py)."""
    return T.pick_glyphs(ascii)


FIT_SHORT = {
    FitVerdict.FITS_WITH_MARGIN: ("ok", T.OK),
    FitVerdict.TIGHT: ("tight", T.CAUTION),
    FitVerdict.DOES_NOT_FIT: ("no", T.DANGER),
    FitVerdict.UNKNOWN: ("unknown", T.MUTED),
}


def _secs(s: float) -> str:
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"


class InputLine:
    """Editable single-line input: cursor moves, backspace/delete, history
    (up/down), Tab completion of slash commands, Enter submits."""

    def __init__(self) -> None:
        self.buf = ""
        self.cursor = 0
        self.history: list[str] = []
        self._hist_idx: Optional[int] = None
        self._stash = ""
        self._comp_matches: list[str] = []
        self._comp_idx = 0

    def handle_key(self, key: str) -> tuple[Optional[str], Optional[str]]:
        """Returns ("submit", text) on Enter, ("cancel", None) on Ctrl-C."""
        if key == K.ENTER:
            text = self.buf
            if text.strip():
                self.history.append(text)
            self._hist_idx = None
            self.buf = ""
            self.cursor = 0
            self._comp_matches = []
            return ("submit", text) if text.strip() else (None, None)
        if key == K.CTRL_C:
            self.buf = ""
            self.cursor = 0
            self._comp_matches = []
            return ("cancel", None)
        if key == K.BACKSPACE:
            if self.cursor > 0:
                self.buf = self.buf[: self.cursor - 1] + self.buf[self.cursor :]
                self.cursor -= 1
            self._comp_matches = []
        elif key == K.DELETE:
            if self.cursor < len(self.buf):
                self.buf = self.buf[: self.cursor] + self.buf[self.cursor + 1 :]
            self._comp_matches = []
        elif key == K.LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif key == K.RIGHT:
            self.cursor = min(len(self.buf), self.cursor + 1)
        elif key == K.HOME:
            self.cursor = 0
        elif key == K.END:
            self.cursor = len(self.buf)
        elif key == K.UP:
            if self.history:
                if self._hist_idx is None:
                    self._stash = self.buf
                    self._hist_idx = len(self.history) - 1
                else:
                    self._hist_idx = max(0, self._hist_idx - 1)
                self.buf = self.history[self._hist_idx]
                self.cursor = len(self.buf)
        elif key == K.DOWN:
            if self._hist_idx is not None:
                self._hist_idx += 1
                if self._hist_idx >= len(self.history):
                    self._hist_idx = None
                    self.buf = self._stash
                else:
                    self.buf = self.history[self._hist_idx]
                self.cursor = len(self.buf)
        elif key == K.TAB:
            self._complete()
        elif len(key) == 1:
            self.buf = self.buf[: self.cursor] + key + self.buf[self.cursor :]
            self.cursor += 1
            self._comp_matches = []
        return (None, None)

    def _complete(self) -> None:
        if self._comp_matches and self.buf.startswith("/"):
            self._comp_idx = (self._comp_idx + 1) % len(self._comp_matches)
            self.buf = self._comp_matches[self._comp_idx]
            self.cursor = len(self.buf)
            return
        if not self.buf.startswith("/"):
            return
        matches = [c for c in SLASH_COMMANDS if c.startswith(self.buf) and c != self.buf]
        if not matches:
            return
        self._comp_matches = matches
        self._comp_idx = 0
        self.buf = matches[0] + (" " if len(matches) == 1 else "")
        self.cursor = len(self.buf)


# ----- overlays -------------------------------------------------------------


def _overlay_panel(
    body, title: str, subtitle: str | None = None, border: str = T.RULE, ascii=False
):
    return Panel(
        body,
        title=Text(f" {title} ", style=f"bold {T.CORE}"),
        title_align="left",
        subtitle=Text(f" {subtitle} ", style=T.MUTED) if subtitle else None,
        subtitle_align="left",
        box=box.ASCII if ascii else box.ROUNDED,
        border_style=border,
        padding=(0, 1),
    )


class Overlay:
    def handle_key(self, key: str, app: "ChatApp") -> None: ...
    def render(self, width: int, height: int): ...


class HelpOverlay(Overlay):
    ROWS = [
        ("/help", "show this help"),
        ("/models", "model picker: ↑/↓ move, Enter switch, d download, x delete, Esc close"),
        ("/model", "show active model and fit estimate"),
        ("/download <name>", "download a model from the registry"),
        ("/delete <name>", "delete a downloaded model from this device"),
        ("/context", "context usage this session"),
        ("/clear", "clear conversation history"),
        ("/tokens <n>", "set max tokens per response"),
        ("/temp <f>", "set sampling temperature"),
        ("/exit", "end session (Ctrl-D on an empty line also exits)"),
    ]

    def __init__(self, ascii: bool = False) -> None:
        self.ascii = ascii

    def handle_key(self, key: str, app: "ChatApp") -> None:
        app.close_overlay()

    def render(self, width: int, height: int):
        table = Table.grid(padding=(0, 3))
        table.add_column(style=T.CORE, no_wrap=True)
        table.add_column()
        for cmd, desc in self.ROWS:
            table.add_row(cmd, desc)
        table.add_row("", "")
        table.add_row("Tab", "complete a /command")
        table.add_row("PgUp / PgDn", "scroll the conversation")
        table.add_row("Ctrl-C", "stop generation (keeps the conversation) / clear input")
        return _overlay_panel(table, "commands", "any key closes", ascii=self.ascii)


class ContextOverlay(Overlay):
    def __init__(self, app: "ChatApp") -> None:
        self.app = app

    def handle_key(self, key: str, app: "ChatApp") -> None:
        app.close_overlay()

    def render(self, width: int, height: int):
        app = self.app
        used = app.estimated_context_tokens()
        limit = app.profile.recommended_context_length
        table = Table.grid(padding=(0, 3))
        table.add_column(style=T.MUTED, no_wrap=True)
        table.add_column()
        table.add_row("messages", f"{len(app.messages)} messages")
        table.add_row("history", f"≈ {used} tokens of {limit} set (estimate: chars / 4)")
        table.add_row("max reply", f"{app.cfg.max_tokens} tokens per response")
        table.add_row("temperature", f"{app.cfg.temperature} (sampling)")
        table.add_row("model limit", f"{app.spec.context_length} tokens (model card)")
        table.add_row("", "")
        table.add_row("trim", "/clear drops the history; /tokens <n> caps reply length")
        return _overlay_panel(table, "context", "any key closes", ascii=app.ascii)


class ConfirmDialog(Overlay):
    def __init__(self, title: str, body: str, on_yes: Callable[[], None]) -> None:
        self.title = title
        self.body = body
        self.on_yes = on_yes

    def handle_key(self, key: str, app: "ChatApp") -> None:
        if key in ("y", "Y"):
            app.close_overlay()
            self.on_yes()
        else:  # n / Esc / Enter / anything else: default deny
            app.close_overlay()
            app.set_status("info", "cancelled")

    def render(self, width: int, height: int):
        body = Text(self.body + "\n\n")
        body.append("[y] ", style="bold")
        body.append("confirm    ")
        body.append("[n] ", style=f"bold {T.CORE}")
        body.append("cancel (default)", style=T.CORE)
        return Panel(
            body,
            title=Text(f" {self.title} ", style=f"bold {T.CAUTION}"),
            title_align="left",
            box=box.ROUNDED,
            border_style=T.CAUTION,
            padding=(0, 1),
        )


class SsdWarningOverlay(Overlay):
    """§3.4: shown before switching to a model whose file exceeds free RAM."""

    def __init__(self, warning: SsdWarning, spec: ModelSpec) -> None:
        self.warning = warning
        self.spec = spec

    def handle_key(self, key: str, app: "ChatApp") -> None:
        choice = SsdWarning.decide(key)
        if choice is None:
            return
        app.close_overlay()
        if choice == "continue":
            app.switch_model(self.spec, confirmed=True)
        elif choice == "pick":
            app.overlay = ModelPicker(app)
        else:
            app.set_status("info", f"not loading {self.spec.name}")

    def render(self, width: int, height: int):
        return Padding(self.warning.render(), (1, 0, 0, 0))


class ModelPicker(Overlay):
    """Registry browser: Enter switches to a downloaded model, d downloads,
    x deletes (asks first). Marks the active model."""

    def __init__(self, app: "ChatApp") -> None:
        self.app = app
        self.items = app.registry.list_models()
        self.idx = 0
        for i, m in enumerate(self.items):
            if m.name == app.spec.name:
                self.idx = i
                break

    def handle_key(self, key: str, app: "ChatApp") -> None:
        if key in (K.ESC, "q"):
            app.close_overlay()
        elif key == K.UP:
            self.idx = max(0, self.idx - 1)
        elif key == K.DOWN:
            self.idx = min(len(self.items) - 1, self.idx + 1)
        elif key == K.ENTER:
            spec = self.items[self.idx]
            if app.registry.is_downloaded(spec):
                app.close_overlay()
                app.switch_model(spec)
            else:
                app.set_status("warn", f"{spec.name} is not downloaded · press d to download")
        elif key == "d":
            app.close_overlay()
            app.request_download(self.items[self.idx])
        elif key == "x":
            spec = self.items[self.idx]
            app.confirm_delete(spec)

    def render(self, width: int, height: int):
        g = self.app.glyphs
        app = self.app
        table = Table(box=None, show_edge=False, pad_edge=False, expand=True, header_style=T.MUTED)
        table.add_column("", width=2, no_wrap=True)
        table.add_column("model", no_wrap=True)
        table.add_column("params", justify="right", no_wrap=True)
        table.add_column("quant", no_wrap=True)
        table.add_column("file", justify="right", no_wrap=True)
        table.add_column("fit (est.)", no_wrap=True)
        table.add_column("state", no_wrap=True)
        for i, m in enumerate(self.items):
            downloaded = app.registry.is_downloaded(m)
            active = m.name == app.spec.name
            sel = i == self.idx
            state = Text()
            if active:
                state.append(f"{g['ok']} active", style=T.OK)
            elif downloaded:
                state.append("downloaded", style=T.OK)
            else:
                state.append("not downloaded", style=T.MUTED)
            name = Text(m.name, style=f"bold {T.CORE}" if sel else "")
            table.add_row(
                Text(g["prompt"] if sel else " ", style=f"bold {T.CORE}"),
                name,
                f"{m.params_b:.1f}B",
                m.quant,
                f"{m.size_gb:.1f} GB",
                self._fit_label(m, downloaded),
                state,
                style=f"on {T.SELECT_BG}" if sel else None,
            )
        hint = "↑/↓ move · Enter switch · d download · x delete · Esc close"
        return _overlay_panel(table, "models", hint, ascii=app.ascii)

    def _fit_label(self, m: ModelSpec, downloaded: bool) -> Text:
        app = self.app
        if not downloaded:
            return Text("unknown until downloaded", style=T.MUTED)
        try:
            est = app.fit_for(m)
            short, color = FIT_SHORT[est.verdict]
            t = Text(short, style=color)
            if est.verdict is not FitVerdict.UNKNOWN:
                t.append(f" {est.headroom_gb:+.1f} GB", style="")
            if not est.resident:
                t.append(" · pages SSD", style=T.CAUTION)
            return t
        except Exception:
            return Text("unknown", style=T.MUTED)


# ----- transcript viewport ------------------------------------------------------------


class _Viewport:
    """Renders transcript blocks and shows the last `height` lines (minus the
    scroll offset). Finished blocks are cached per width, so a long session
    does not re-render every old reply on every frame."""

    def __init__(self, app: "ChatApp") -> None:
        self.app = app

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        app = self.app
        width = options.max_width
        height = options.height or app.console.height
        opts = options.update(width=width, height=None)
        lines: list[list[Segment]] = []
        for block in app.blocks:
            cached = block.get("_cache")
            if cached is not None and cached[0] == width and not block.get("streaming"):
                lines.extend(cached[1])
                continue
            rendered = console.render_lines(app._block_render(block), opts, pad=True)
            if not block.get("streaming"):
                block["_cache"] = (width, rendered)
            lines.extend(rendered)
        if not app.blocks:
            lines = console.render_lines(app._welcome_render(), opts, pad=True)
        if width < 100:  # narrow: no rule under the header, so leave one blank row
            lines = [[Segment(" " * width)]] + lines
        total = len(lines)
        max_scroll = max(0, total - height)
        app.scroll = min(app.scroll, max_scroll)
        app._max_scroll = max_scroll
        end = total - app.scroll
        start = max(0, end - height)
        view = lines[start:end]
        blank = [Segment(" " * width)]
        while len(view) < height:
            view.append(blank)
        for line in view:
            yield from line
            yield Segment.line()


# ----- the app ---------------------------------------------------------------


class ChatApp:
    """Full-screen chat app. Testable headlessly: drive `handle_key` /
    `submit_line`, call `tick()`, render `_frame()` into a recording Console."""

    def __init__(
        self,
        engine_getter: Callable,
        spec: ModelSpec,
        profile: HardwareProfile,
        registry: ModelRegistry,
        geometry: ModelGeometry | None = None,
        cfg: ChatUiConfig | None = None,
        caps: TerminalCaps | None = None,
        on_model_change: Optional[Callable[[ModelSpec], None]] = None,
        console: Console | None = None,
        policy=None,
        tools=None,
        ascii: bool | None = None,
    ) -> None:
        self._engine_getter = engine_getter
        self.spec = spec
        self.profile = profile
        self.registry = registry
        self.geometry = geometry
        self.cfg = cfg or ChatUiConfig()
        self.caps = caps or probe_terminal()
        self.on_model_change = on_model_change
        self.console = console or Console()
        with contextlib.suppress(Exception):
            self.console.push_theme(MD_THEME)
        self.policy = policy
        self.tools = tools
        self.ascii = T.use_ascii(getattr(self.console, "file", None)) if ascii is None else ascii
        self.glyphs = pick_glyphs(self.ascii)
        self.reduced_motion = T.reduced_motion()
        self.input = InputLine()
        self.messages: list[dict[str, str]] = []
        self.blocks: list[dict] = []
        self.overlay: Optional[Overlay] = None
        self.status = ("info", "")
        self.running = True
        self.scroll = 0
        self._max_scroll = 0
        self._q: queue.Queue = queue.Queue()
        self._gen_active = False
        self._cancel = threading.Event()
        self._download: Optional[ModelSpec] = None
        self._last_stats: Optional[TurnStats] = None
        self._last_tps: Optional[float] = None
        self._last_dl_pct = -1
        self._dirty = False
        self._fit_cache: dict[str, object] = {}

    # ----- helpers ----------------------------------------------------------
    def set_status(self, kind: str, text: str) -> None:
        self.status = (kind, sanitize(text))

    def close_overlay(self) -> None:
        self.overlay = None

    def estimated_context_tokens(self) -> int:
        return sum(max(len(m["content"]) // 4, 1) for m in self.messages)

    def _sys(self, text: str, kind: str = "info") -> None:
        self.blocks.append({"kind": "sys", "text": sanitize(text), "level": kind})

    def _find_spec(self, name: str) -> Optional[ModelSpec]:
        for m in self.registry.list_models():
            if m.name == name:
                return m
        return None

    def tools_state(self) -> str:
        return tools_label(self.policy, self.tools)

    def fit_for(self, spec: ModelSpec):
        """Fit estimate for `spec` at the session context, from real GGUF
        geometry when the file is present. Cached per model name."""
        if spec.name in self._fit_cache:
            return self._fit_cache[spec.name]
        geometry = self.geometry if spec.name == self.spec.name else None
        file_size = None
        try:
            if self.registry.is_downloaded(spec):
                info = read_gguf_model_info(spec.local_path(self.registry.cache_dir))
                if info is not None:
                    if info.geometry is not None and info.geometry.is_complete:
                        geometry = info.geometry
                    file_size = info.file_size_bytes
        except Exception:
            pass
        est = estimate_fit(
            spec.params_b,
            spec.quant,
            self.profile.recommended_context_length,
            self.profile.available_ram_gb,
            geometry=geometry,
            file_size_bytes=file_size,
        )
        self._fit_cache[spec.name] = est
        return est

    # ----- commands -----------------------------------------------------------
    def submit_line(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith("/"):
            self.handle_command(text)
            return
        if self._gen_active:
            self.set_status("warn", "still generating · Ctrl-C stops it")
            return
        self.scroll = 0
        self.messages.append({"role": "user", "content": text})
        self.blocks.append({"kind": "user", "text": sanitize(text)})
        self._start_generation()

    def handle_command(self, cmd: str) -> None:
        parts = cmd.split()
        name, args = parts[0].lower(), parts[1:]
        if name in ("/exit", "/quit"):
            self.running = False
        elif name == "/help":
            self.overlay = HelpOverlay(self.ascii)
        elif name == "/models":
            self.overlay = ModelPicker(self)
        elif name == "/model":
            self._sys(self._header_text())
        elif name == "/download":
            if not args:
                self.set_status("warn", "/download <name> · /models lists all")
            else:
                spec = self._find_spec(args[0])
                if spec is None:
                    self.set_status("err", f"unknown model '{args[0]}' · /models lists all")
                else:
                    self.request_download(spec)
        elif name == "/delete":
            if not args:
                self.set_status("warn", "/delete <name> · /models lists all")
            else:
                spec = self._find_spec(args[0])
                if spec is None:
                    self.set_status("err", f"unknown model '{args[0]}'")
                else:
                    self.confirm_delete(spec)
        elif name == "/context":
            self.overlay = ContextOverlay(self)
        elif name == "/clear":
            self.messages.clear()
            self.blocks.clear()
            self.scroll = 0
            self._sys("history cleared")
        elif name == "/tokens":
            try:
                self.cfg.max_tokens = max(1, int(args[0]))
                self._sys(f"max reply length → {self.cfg.max_tokens} tokens")
            except (ValueError, IndexError):
                self.set_status("err", "/tokens <positive int>")
        elif name == "/temp":
            try:
                self.cfg.temperature = max(0.0, float(args[0]))
                self._sys(f"temperature → {self.cfg.temperature}")
            except (ValueError, IndexError):
                self.set_status("err", "/temp <float ≥ 0>")
        else:
            suggestion = difflib.get_close_matches(name, SLASH_COMMANDS, n=1)
            hint = f" · did you mean {suggestion[0]}?" if suggestion else ""
            self.set_status("err", f"unknown command: {name}{hint} (/help lists commands)")

    # ----- model management ---------------------------------------------------
    def request_download(self, spec: ModelSpec) -> None:
        if self.registry.is_downloaded(spec):
            self.set_status("ok", f"{spec.name} already downloaded")
            return
        if self._download is not None:
            self.set_status("warn", f"already downloading {self._download.name}")
            return
        self._download = spec
        self._last_dl_pct = -1
        self.set_status("info", f"downloading {spec.name} ({spec.size_gb:.1f} GB)…")
        self._sys(f"downloading {spec.name} · {spec.size_gb:.1f} GB · from {spec.repo_id}")

        def worker() -> None:
            try:
                # HF/tqdm write progress to stderr; keep it out of the alt screen
                with contextlib.redirect_stderr(io.StringIO()):
                    path = self.registry.download(spec, progress=False)
                self._q.put(("dl_done", (True, f"downloaded {spec.name} → {path}")))
            except Exception as e:
                self._q.put(("dl_done", (False, f"download failed: {type(e).__name__}: {e}")))

        threading.Thread(target=worker, daemon=True).start()

    def confirm_delete(self, spec: ModelSpec) -> None:
        if not self.registry.is_downloaded(spec):
            self.set_status("warn", f"{spec.name} is not downloaded")
            return
        if spec.name == self.spec.name:
            self.set_status("err", f"{spec.name} is the active model · switch with /models first")
            return
        self.overlay = ConfirmDialog(
            f"delete {spec.name}?",
            f"Removes ≈ {spec.size_gb:.1f} GB from {self.registry.cache_dir} (all shards).",
            lambda: self._do_delete(spec),
        )

    def _do_delete(self, spec: ModelSpec) -> None:
        removed = self.registry.delete(spec)
        self._fit_cache.pop(spec.name, None)
        if removed:
            self.set_status("ok", f"deleted {spec.name} from this device")
            self._sys(f"deleted {spec.name} · ≈ {spec.size_gb:.1f} GB freed")
        else:
            self.set_status("warn", f"{spec.name} was not present on disk")

    def switch_model(self, spec: ModelSpec, confirmed: bool = False) -> None:
        if spec.name == self.spec.name:
            self.set_status("info", f"{spec.name} is already active")
            return
        if self._gen_active:
            self.set_status("warn", "still generating · Ctrl-C first")
            return
        if not confirmed:
            # §3.4: never silently load into SSD paging
            try:
                est = self.fit_for(spec)
            except Exception:
                est = None
            if SsdWarning.needed(est):
                self.overlay = SsdWarningOverlay(
                    SsdWarning(spec.name, est.weights_gb, est.available_ram_gb), spec
                )
                return
        self.set_status("info", f"loading {spec.name}…")
        try:
            if self.on_model_change is not None:
                self.on_model_change(spec)
        except Exception as e:
            self.set_status("err", f"could not load {spec.name}: {type(e).__name__}: {e}")
            return
        self.spec = spec
        self.geometry = None
        self._fit_cache.pop(spec.name, None)
        self.messages.clear()
        self.blocks.clear()
        self.scroll = 0
        self._last_stats = None
        self._sys(f"switched to {spec.name} · {spec.params_b:.1f}B {spec.quant}")
        self.set_status("ok", f"active model: {spec.name}")

    # ----- generation -----------------------------------------------------------
    def _start_generation(self) -> None:
        self._cancel.clear()
        block = {
            "kind": "assistant",
            "text": "",
            "streaming": True,
            "t0": time.monotonic(),
            "t_first": None,
        }
        self.blocks.append(block)
        self._gen_active = True
        self.set_status("info", "")
        msgs = list(self.messages)
        getter = self._engine_getter
        cfg = self.cfg
        cancel = self._cancel
        q = self._q

        def worker() -> None:
            t_start = time.perf_counter()
            t_first = None
            chunks = 0
            parts: list[str] = []
            gen = None
            engine = None
            error = None
            try:
                engine = getter()
                gen = engine.stream_messages(
                    msgs,
                    GenerationConfig(
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        stream=True,
                        prompt_lookup=cfg.prompt_lookup,
                    ),
                )
                for tok in gen:
                    if cancel.is_set():
                        break
                    if not tok:
                        continue
                    if t_first is None:
                        t_first = time.perf_counter()
                        q.put(("first", None))
                    q.put(("tok", tok))
                    parts.append(tok)
                    chunks += 1
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                q.put(("error", error))
            finally:
                if gen is not None and hasattr(gen, "close"):
                    with contextlib.suppress(Exception):
                        gen.close()
            t_end = time.perf_counter()
            stats = stats_from_run(
                engine,
                sanitize("".join(parts)),
                chunks,
                t_start,
                t_first,
                t_end,
                interrupted=cancel.is_set(),
                error=error,
            )
            q.put(("done", stats))

        threading.Thread(target=worker, daemon=True).start()

    def handle_key(self, key: str) -> None:
        self._dirty = True  # input buffer / overlay / status changed → repaint
        if key == K.CTRL_C:
            if self._gen_active or self._download is not None:
                self._cancel.set()
                if self._gen_active:
                    self.set_status("warn", "stopping…")
            else:
                self.input.buf = ""
                self.input.cursor = 0
            return
        if key == K.CTRL_D and not self.input.buf and self.overlay is None:
            self.running = False
            return
        if self.overlay is not None:
            self.overlay.handle_key(key, self)
            return
        if key in (K.PGUP, K.PGDN):
            page = max(3, self.console.height - 8)
            if key == K.PGUP:
                self.scroll = min(self.scroll + page, max(self._max_scroll, 0))
            else:
                self.scroll = max(0, self.scroll - page)
            return
        action, value = self.input.handle_key(key)
        if action == "submit" and value:
            self.submit_line(value)

    def tick(self) -> bool:
        """Drain the worker queue; returns True when the frame changed."""
        dirty = False
        # bounded drain: a fast producer must never starve the render loop
        for _ in range(512):
            try:
                kind, val = self._q.get_nowait()
            except queue.Empty:
                break
            block = (
                self.blocks[-1] if self.blocks and self.blocks[-1]["kind"] == "assistant" else None
            )
            if kind == "first":
                if block is not None:
                    block["t_first"] = time.monotonic()
            elif kind == "tok":
                if block is not None:
                    block["text"] += sanitize(val)
                    dirty = True
            elif kind == "error":
                if block is not None:
                    block["error"] = sanitize(val)
                self.set_status("err", "stream error · partial reply kept, not regenerated")
                dirty = True
            elif kind == "done":
                stats: TurnStats = val
                if block is not None:
                    block["streaming"] = False
                    block["stats"] = stats
                    block["meta"] = self._meta_text(block)
                    block.pop("_cache", None)
                    self.messages.append({"role": "assistant", "content": block["text"]})
                if not stats.interrupted and stats.error is None:
                    self._last_stats = stats
                    if stats.unit == "tokens" and stats.decode_rate is not None:
                        self._last_tps = stats.decode_rate
                if stats.interrupted:
                    self.set_status("info", "stopped · conversation kept")
                elif stats.error is None:
                    self.set_status("info", "")
                self._gen_active = False
                dirty = True
            elif kind == "dl_done":
                ok, msg = val
                self._download = None
                self._fit_cache.clear()
                self.set_status("ok" if ok else "err", msg)
                self._sys(msg, "ok" if ok else "err")
                dirty = True
        if self._download is not None:
            pct = self._download_progress()
            if pct != self._last_dl_pct:
                self._last_dl_pct = pct
                spec = self._download
                self.set_status(
                    "info",
                    f"downloading {spec.name} · {pct:.0f}% · {self._downloaded_gb(spec):.2f} of "
                    f"{spec.size_gb:.1f} GB",
                )
                dirty = True
        return dirty

    def _meta_text(self, block: dict) -> str:
        stats: TurnStats = block["stats"]
        sep = self.glyphs["sep"]
        if block.get("error"):
            return f"stream error {sep} {block['error']} {sep} partial reply kept"
        return stats.wide(sep)

    def _downloaded_gb(self, spec: ModelSpec) -> float:
        total = 0
        parent = spec.local_path(self.registry.cache_dir).parent
        if parent.exists():
            prefix = self.registry.shard_prefix(spec)
            pattern = f"{prefix}-*.gguf" if prefix else spec.filename
            for f in parent.glob(pattern):
                total += f.stat().st_size
        return total / (1024**3)

    def _download_progress(self) -> float:
        spec = self._download
        if spec is None:
            return 0.0
        return min(100.0, 100.0 * self._downloaded_gb(spec) / max(spec.size_gb, 1e-9))

    # ----- header ---------------------------------------------------------------
    def _fit_parts(self) -> tuple[str, str, Optional[str]]:
        """(short verdict, colour, headroom text or None)."""
        est = self.fit_for(self.spec)
        short, color = FIT_SHORT[est.verdict]
        if est.verdict is FitVerdict.UNKNOWN:
            return short, color, None
        return short, color, f"{est.headroom_gb:+.1f} GB est."

    def _header_text(self) -> str:
        """One-line plain summary (used by /model)."""
        est = self.fit_for(self.spec)
        used = self.estimated_context_tokens()
        head = (
            f"{self.spec.name} ({self.spec.params_b:.1f}B {self.spec.quant}) | "
            f"ctx ≈ {used} / {self.profile.recommended_context_length} tokens | "
            f"fit: {est.verdict.value}"
        )
        if est.verdict is not FitVerdict.UNKNOWN:
            head += f" (headroom {est.headroom_gb:+.1f} GB, estimate)"
        head += f" | tools: {self.tools_state()}"
        if self._last_stats is not None:
            head += f" | last turn: {self._last_stats.wide('·')}"
        return head

    def _run_state(self) -> tuple[str, str]:
        """Right-hand header state: (text, colour)."""
        if self._gen_active:
            block = self.blocks[-1] if self.blocks else {}
            if block.get("t_first") is None:
                return "thinking", T.CORE
            return "writing", T.CORE
        if self._download is not None:
            return f"downloading {self._last_dl_pct:.0f}%", T.CAUTION
        return "ready", T.OK

    def _ctx_ratio(self) -> float:
        limit = max(self.profile.recommended_context_length, 1)
        return self.estimated_context_tokens() / limit

    def _header_render(self, width: int):
        g = self.glyphs
        sep = g["sep"]
        wide = width >= 100
        # row 1: brand + right-aligned run state
        left = Text()
        left.append(f"{g['brand']} ", style=f"bold {T.CORE}")
        left.append("EigenCore", style="bold")
        left.append(f" {__version__} {sep} chat", style=T.MUTED)
        state, color = self._run_state()
        right = Text(f"local {sep} {state}" if wide else state, style=color)
        row1 = Table.grid(expand=True)
        row1.add_column(no_wrap=True, overflow="ellipsis")
        row1.add_column(justify="right", no_wrap=True)
        row1.add_row(left, right)
        # row 2: model / ctx / fit / tools
        used = self.estimated_context_tokens()
        limit = self.profile.recommended_context_length
        ctx_color = T.CAUTION if self._ctx_ratio() >= 0.85 else ""
        short, fcolor, headroom = self._fit_parts()
        row2 = Text(no_wrap=True, overflow="ellipsis")
        if wide:
            gap = "    "
            row2.append("model ", style=T.MUTED)
            row2.append(f"{self.spec.name} {self.spec.quant}")
            row2.append(gap + "ctx ", style=T.MUTED)
            row2.append(f"{g['approx']}{used} / {limit} tok", style=ctx_color)
            row2.append(gap + "fit ", style=T.MUTED)
            row2.append(short, style=fcolor)
            if headroom:
                row2.append(f" {headroom}")
            row2.append(gap + "tools ", style=T.MUTED)
            row2.append(self.tools_state())
        else:
            row2.append(f"{self.spec.name} {self.spec.quant}")
            row2.append(f" {sep} ctx ", style=T.MUTED)
            row2.append(f"{g['approx']}{used}/{limit}", style=ctx_color)
            row2.append(f" {sep} fit ", style=T.MUTED)
            row2.append(short, style=fcolor)
            row2.append(f" {sep} tools ", style=T.MUTED)
            row2.append(self.tools_state())
        row2.truncate(width, overflow="ellipsis")
        return Group(row1, row2)

    def _rule(self, width: int) -> Text:
        return Text(self.glyphs["rule"] * width, style=T.RULE)

    # ----- footer ---------------------------------------------------------------
    def _ctx_bar(self, width: int) -> Text:
        g = self.glyphs
        ratio = min(self._ctx_ratio(), 1.0)
        bar_w = 10 if width >= 100 else 6
        filled = round(bar_w * ratio)
        if ratio > 0 and filled == 0:
            filled = 1
        color = T.CAUTION if ratio >= 0.85 else T.OK
        t = Text(no_wrap=True)
        t.append("ctx ", style=T.MUTED)
        t.append(g["bar"] * filled, style=color)
        t.append(g["empty"] * (bar_w - filled), style=T.RULE)
        pct = f" {g['approx']}{ratio * 100:.0f}%" if ratio == 0 or ratio >= 0.01 else " <1%"
        if width >= 100:
            pct += f" of {self.profile.recommended_context_length} tok"
        t.append(pct, style=color if ratio >= 0.85 else T.MUTED)
        if ratio >= 0.85:
            t.append(f" {self.glyphs['sep']} /context to trim", style=T.CAUTION)
        return t

    def _status_render(self, width: int):
        g = self.glyphs
        kind, msg = self.status
        color = {
            "ok": T.OK,
            "warn": T.CAUTION,
            "err": T.DANGER,
            "info": T.MUTED,
        }.get(kind, T.MUTED)
        icon = {"ok": g["ok"], "warn": g["warn"], "err": g["err"]}.get(kind, "")
        left = Text(no_wrap=True, overflow="ellipsis")
        if self.scroll > 0:
            left.append(f"scrolled up {self.scroll} lines {g['sep']} PgDn returns", style=T.CAUTION)
        elif msg:
            if icon:
                left.append(f"{icon} ", style=color)
            left.append(msg, style=color)
        grid = Table.grid(expand=True)
        grid.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_row(left, self._ctx_bar(width))
        return grid

    def _input_render(self, width: int):
        g = self.glyphs
        wide = width >= 100
        prompt = f"{g['prompt']} "
        buf = self.input.buf
        cur = self.input.cursor
        right = Text("ctrl-c stops", style=T.CORE if self._gen_active else T.MUTED)
        hints = Text(no_wrap=True)
        if not buf:
            if wide:
                hints.append("   " + " ".join(FOOTER_HINTS), style=T.MUTED)
            else:
                hints.append(f" {g['sep']} /help", style=T.MUTED)
        elif buf.startswith("/") and " " not in buf:
            matches = [c for c in SLASH_COMMANDS if c.startswith(buf)]
            if matches and matches != [buf]:
                hints.append("   ")
                for i, m in enumerate(matches[:6]):
                    hints.append(m, style=f"bold {T.CORE}" if i == 0 else T.MUTED)
                    hints.append("  ")
                hints.append("tab completes", style=T.MUTED)
            elif not matches:
                hints.append("   no such command · /help", style=T.CAUTION)
        right_w = len(right) + 2 if wide else 0
        avail = max(1, width - len(prompt) - right_w - (len(hints) if buf else 0) - 1)
        line = Text(no_wrap=True, overflow="crop")
        line.append(prompt, style=f"bold {T.CORE}")
        if not buf:
            line.append(PLACEHOLDER[0], style="reverse")
            line.append(PLACEHOLDER[1:], style=T.MUTED)
            line.append_text(hints)
        else:
            off = cur - avail + 1 if cur >= avail else 0
            visible = buf[off : off + avail]
            cin = cur - off
            line.append(visible[:cin])
            line.append(visible[cin : cin + 1] or " ", style="reverse")
            line.append(visible[cin + 1 :])
            line.append_text(hints)
        if not wide:
            line.truncate(width, overflow="crop")
            return line
        grid = Table.grid(expand=True)
        grid.add_column(no_wrap=True, overflow="crop", ratio=1)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_row(line, right)
        return grid

    # ----- transcript -------------------------------------------------------------
    def _state_line(self, block: dict) -> Text:
        """Agent state while streaming: orb · verb · elapsed seconds."""
        g = self.glyphs
        now = time.monotonic()
        elapsed = now - block.get("t0", now)
        if self.reduced_motion:
            orb = g["orb_static"]
        else:
            frames = g["orb"]
            orb = frames[int(elapsed * 8) % len(frames)]
        if block.get("t_first") is None:
            verb = PREFILL_VERBS[int(elapsed / 2.5) % len(PREFILL_VERBS)]
        elif open_fence(block["text"]):
            verb = "Writing code"
        else:
            verb = "Writing"
        t = Text(no_wrap=True)
        t.append(f"{orb} ", style=f"bold {T.CORE}")
        t.append(f"{verb}{g['ellipsis']}", style=T.CORE)
        t.append(f" {_secs(elapsed)}", style=T.MUTED)
        if self._cancel.is_set():
            t.append(f" {g['sep']} stopping", style=T.CAUTION)
        return t

    def _block_render(self, block: dict):
        kind = block["kind"]
        indent = (0, 0, 0, 2)
        if kind == "user":
            return Group(
                Text("you", style=f"bold {T.CORE}"),
                Padding(Text(block["text"]), indent),
                Text(""),
            )
        if kind == "assistant":
            parts = [Text("eigencore", style=f"bold {T.LIGHT}")]
            body = block["text"]
            streaming = block.get("streaming")
            if body:
                parts.append(
                    Padding(render_markdown(body, streaming=streaming, ascii=self.ascii), indent)
                )
            if streaming:
                parts.append(Padding(self._state_line(block), indent))
            else:
                meta = block.get("meta")
                if meta:
                    stats = block.get("stats")
                    color = T.DANGER if block.get("error") else T.MUTED
                    if stats is not None and stats.interrupted:
                        color = T.MUTED
                    parts.append(Padding(Text(meta, style=color), indent))
            parts.append(Text(""))
            return Group(*parts)
        level = block.get("level", "info")
        color = {"ok": T.OK, "err": T.DANGER, "warn": T.CAUTION}.get(level, T.MUTED)
        return Group(Text(f"{self.glyphs['info']} {block['text']}", style=color), Text(""))

    def _welcome_render(self):
        """Blank chat with one hint line (handbook 5.5 step 7)."""
        g = self.glyphs
        sep = g["sep"]
        t = Text()
        t.append("  Runs locally on this machine. Type a message to start", style=T.MUTED)
        for cmd, what in (("/models", "switch model"), ("/help", "commands")):
            t.append(f" {sep} ", style=T.MUTED)
            t.append(cmd, style=T.CORE)
            t.append(f" {what}", style=T.MUTED)
        return Group(Text(""), t)

    def _transcript_render(self, width: int, height: int):
        return _Viewport(self)

    def _frame(self):
        width = self.console.width
        height = self.console.height
        wide = width >= 100
        layout = Layout()
        rows = [Layout(name="header", size=2)]
        if wide:
            rows.append(Layout(name="rule1", size=1))
        rows.append(Layout(name="body", ratio=1))
        rows.append(Layout(name="status", size=1))
        if wide:
            rows.append(Layout(name="rule2", size=1))
        rows.append(Layout(name="input", size=1))
        layout.split_column(*rows)
        layout["header"].update(self._header_render(width))
        if wide:
            layout["rule1"].update(self._rule(width))
            layout["rule2"].update(self._rule(width))
        body_h = max(3, height - (6 if wide else 4))
        if self.overlay is not None:
            layout["body"].update(self.overlay.render(width, body_h))
        else:
            layout["body"].update(self._transcript_render(width, body_h))
        layout["status"].update(self._status_render(width))
        layout["input"].update(self._input_render(width))
        return layout

    # ----- main loop --------------------------------------------------------------
    def run(self) -> None:
        """Run the full-screen loop. Requires a TTY; the CLI routes non-TTY
        and --plain to the line-mode ChatUI instead."""
        if not sys.stdin.isatty():
            raise RuntimeError("ChatApp requires an interactive terminal")
        from eigencore.tui.keys import KeyReader

        console = self.console
        console.file.write("\x1b[?25l")  # hide cursor (rich 15 dropped hide_cursor)
        console.file.flush()
        frame_dt = 1 / 8  # orb animation: 8 fps, only while generating
        try:
            with KeyReader() as reader:
                live = Live(self._frame(), console=console, screen=True, auto_refresh=False)
                with live:
                    last_size = (console.width, console.height)
                    last_paint = time.monotonic()
                    while self.running:
                        try:
                            key = reader.poll(0.05)
                            while key is not None:
                                self.handle_key(key)
                                key = reader.poll(0)
                        except KeyboardInterrupt:
                            self._cancel.set()
                        dirty = self.tick() or self._dirty
                        self._dirty = False
                        now = time.monotonic()
                        animate = self._gen_active and not self.reduced_motion
                        if self._gen_active and now - last_paint >= (frame_dt if animate else 1.0):
                            dirty = True  # orb frame / elapsed seconds
                        size = (console.width, console.height)
                        if dirty or size != last_size:
                            live.update(self._frame(), refresh=True)
                            last_size = size
                            last_paint = now
        except KeyboardInterrupt:
            self._cancel.set()
        finally:
            console.file.write("\x1b[?25h")  # show cursor
            console.file.flush()
        self.console.print(f"\n{self.glyphs['info']} session ended", markup=False, highlight=False)
