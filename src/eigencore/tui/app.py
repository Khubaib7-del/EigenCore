"""
Full-screen chat TUI for `eigencore chat` on a TTY (owner decision 2026-11-11:
chat goes full-screen like Gemini CLI; `--plain`/non-TTY keeps the line-mode
`tui/chat.py` ChatUI). Zero new dependencies — Rich Live + stdlib key reader.

Layout: header (brand glyph, model, context bar, fit, last measured tok/s),
scrollback transcript, status line, editable input line with history and
slash-command completion. Overlays: model picker (select / download / delete),
help, context, confirm dialogs.

Safety rules kept from docs/design/TUI.md: model text is sanitized and printed
with markup disabled; Ctrl-C stops the stream without losing history; a
mid-stream error never regenerates; every speed number is labeled "measured
this turn"; deletes remove files from disk via the registry.
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
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from eigencore import __version__
from eigencore.engine.inference import GenerationConfig
from eigencore.fit import ModelGeometry, estimate_fit, read_gguf_model_info
from eigencore.hal.profiler import HardwareProfile
from eigencore.models.registry import ModelRegistry, ModelSpec
from eigencore.tui import keys as K
from eigencore.tui.chat import ChatUiConfig, sanitize_model_text
from eigencore.tui.theme import TOKENS, TerminalCaps, probe_terminal

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


def pick_glyphs() -> dict[str, str]:
    """Unicode glyphs when the output encoding supports them, else ASCII."""
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" in enc:
        return {
            "brand": "◩",
            "prompt": "❯",
            "ok": "●",
            "warn": "▲",
            "err": "✖",
            "info": "•",
            "dl": "⬇",
            "bar": "█",
            "empty": "░",
            "cursor": "▏",
        }
    return {
        "brand": "[#]",
        "prompt": ">",
        "ok": "o",
        "warn": "!",
        "err": "x",
        "info": "*",
        "dl": "v",
        "bar": "#",
        "empty": "-",
        "cursor": "|",
    }


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


class Overlay:
    def handle_key(self, key: str, app: "ChatApp") -> None: ...
    def render(self, width: int, height: int): ...


class HelpOverlay(Overlay):
    ROWS = [
        ("/help", "show this help"),
        ("/models", "model picker: ↑/↓ move, Enter select, d download, x delete, Esc close"),
        ("/model", "show active model and fit estimate"),
        ("/download <name>", "download a model from the registry"),
        ("/delete <name>", "delete a downloaded model from this device"),
        ("/context", "context usage this session"),
        ("/clear", "clear conversation history"),
        ("/tokens <n>", "set max tokens per response"),
        ("/temp <f>", "set sampling temperature"),
        ("/exit", "end session (Ctrl-D on empty line also exits)"),
    ]

    def handle_key(self, key: str, app: "ChatApp") -> None:
        app.close_overlay()

    def render(self, width: int, height: int):
        table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False)
        table.add_column("command", style=TOKENS["brand"])
        table.add_column("action")
        for cmd, desc in self.ROWS:
            table.add_row(cmd, desc)
        table.add_row("Ctrl-C", "stop generation / clear input")
        return Panel(table, title="[bold]Commands[/]", border_style=TOKENS["brand"])


class ContextOverlay(Overlay):
    def __init__(self, app: "ChatApp") -> None:
        self.app = app

    def handle_key(self, key: str, app: "ChatApp") -> None:
        app.close_overlay()

    def render(self, width: int, height: int):
        app = self.app
        used = app.estimated_context_tokens()
        limit = app.profile.recommended_context_length
        table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False)
        table.add_column("metric", style=TOKENS["brand"])
        table.add_column("value")
        table.add_row("messages", str(len(app.messages)))
        table.add_row("est. tokens in history", f"{used} / {limit} (chars/4 heuristic)")
        table.add_row("max tokens / response", str(app.cfg.max_tokens))
        table.add_row("temperature", str(app.cfg.temperature))
        table.add_row("model context limit", str(app.spec.context_length))
        return Panel(table, title="[bold]Context[/]", border_style=TOKENS["brand"])


class ConfirmDialog(Overlay):
    def __init__(self, title: str, body: str, on_yes: Callable[[], None]) -> None:
        self.title = title
        self.body = body
        self.on_yes = on_yes

    def handle_key(self, key: str, app: "ChatApp") -> None:
        if key in ("y", "Y", K.ENTER):
            app.close_overlay()
            self.on_yes()
        else:  # n / Esc / anything else: default deny
            app.close_overlay()
            app.set_status("info", "cancelled")

    def render(self, width: int, height: int):
        body = Text(self.body + "\n\n[y] confirm   [n] cancel (default)", style="")
        return Panel(body, title=f"[bold]{self.title}[/]", border_style=TOKENS["warn"])


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
                app.set_status("warn", f"{spec.name} not downloaded — press d to download")
        elif key == "d":
            app.close_overlay()
            app.request_download(self.items[self.idx])
        elif key == "x":
            spec = self.items[self.idx]
            app.confirm_delete(spec)

    def render(self, width: int, height: int):
        g = pick_glyphs()
        app = self.app
        table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, expand=True)
        table.add_column("", width=2)
        table.add_column("model", style=TOKENS["brand"])
        table.add_column("params", justify="right")
        table.add_column("quant")
        table.add_column("size", justify="right")
        table.add_column("fit")
        table.add_column("state")
        for i, m in enumerate(self.items):
            downloaded = app.registry.is_downloaded(m)
            active = m.name == app.spec.name
            mark = "›" if i == self.idx else " "
            state = []
            if active:
                state.append(f"[{TOKENS['ok']}]{g['ok']} active[/]")
            if downloaded:
                state.append(f"[{TOKENS['ok']}]{g['ok']} downloaded[/]")
            else:
                state.append(f"[{TOKENS['dim']}]not downloaded[/]")
            fit = self._fit_label(m, downloaded)
            table.add_row(
                mark,
                m.name,
                f"{m.params_b:.1f}B",
                m.quant,
                f"{m.size_gb:.1f} GB",
                fit,
                " ".join(state),
            )
        hint = "↑/↓ move · Enter switch · d download · x delete · Esc close"
        return Panel(table, title="[bold]Models[/]", subtitle=hint, border_style=TOKENS["brand"])

    def _fit_label(self, m: ModelSpec, downloaded: bool) -> str:
        app = self.app
        if not downloaded:
            return f"[{TOKENS['dim']}]unknown*[/]"
        try:
            info = read_gguf_model_info(m.local_path(app.registry.cache_dir))
            est = estimate_fit(
                m.params_b,
                m.quant,
                app.profile.recommended_context_length,
                app.profile.available_ram_gb,
                geometry=info.geometry if info else None,
                file_size_bytes=info.file_size_bytes if info else None,
            )
            return est.verdict.value.replace("_", " ")
        except Exception:
            return "unknown"


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
        self.glyphs = pick_glyphs()
        self.input = InputLine()
        self.messages: list[dict[str, str]] = []
        self.blocks: list[dict] = []
        self.overlay: Optional[Overlay] = None
        self.status = ("info", "ready — /help for commands")
        self.running = True
        self._q: queue.Queue = queue.Queue()
        self._gen_active = False
        self._cancel = threading.Event()
        self._download: Optional[ModelSpec] = None
        self._last_tps: Optional[float] = None
        self._last_dl_pct = -1
        self._dirty = False

    # ----- helpers ----------------------------------------------------------
    def set_status(self, kind: str, text: str) -> None:
        self.status = (kind, sanitize_model_text(text))

    def close_overlay(self) -> None:
        self.overlay = None

    def estimated_context_tokens(self) -> int:
        return sum(max(len(m["content"]) // 4, 1) for m in self.messages)

    def _sys(self, text: str) -> None:
        self.blocks.append({"kind": "sys", "text": sanitize_model_text(text)})

    def _find_spec(self, name: str) -> Optional[ModelSpec]:
        for m in self.registry.list_models():
            if m.name == name:
                return m
        return None

    # ----- commands -----------------------------------------------------------
    def submit_line(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith("/"):
            self.handle_command(text)
            return
        if self._gen_active:
            self.set_status("warn", "still generating — Ctrl-C to interrupt")
            return
        self.messages.append({"role": "user", "content": text})
        self.blocks.append({"kind": "user", "text": sanitize_model_text(text)})
        self._start_generation()

    def handle_command(self, cmd: str) -> None:
        parts = cmd.split()
        name, args = parts[0].lower(), parts[1:]
        if name in ("/exit", "/quit"):
            self.running = False
        elif name == "/help":
            self.overlay = HelpOverlay()
        elif name == "/models":
            self.overlay = ModelPicker(self)
        elif name == "/model":
            self._sys(self._header_text())
        elif name == "/download":
            if not args:
                self.set_status("warn", "/download <name> — see /models")
            else:
                spec = self._find_spec(args[0])
                if spec is None:
                    self.set_status("err", f"unknown model '{args[0]}' — /models lists all")
                else:
                    self.request_download(spec)
        elif name == "/delete":
            if not args:
                self.set_status("warn", "/delete <name> — see /models")
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
            self._sys("history cleared")
        elif name == "/tokens":
            try:
                self.cfg.max_tokens = max(1, int(args[0]))
                self._sys(f"max tokens → {self.cfg.max_tokens}")
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
            hint = f" — did you mean {suggestion[0]}?" if suggestion else ""
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
        self.set_status("info", f"downloading {spec.name} ({spec.size_gb:.1f} GB)...")
        self._sys(
            f"⬇ downloading {spec.name} ({spec.size_gb:.1f} GB) — shard-aware, from {spec.repo_id}"
        )

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
            self.set_status("err", f"{spec.name} is the active model — switch with /models first")
            return
        self.overlay = ConfirmDialog(
            f"Delete {spec.name}?",
            f"Removes ~{spec.size_gb:.1f} GB from {self.registry.cache_dir} (all shards).",
            lambda: self._do_delete(spec),
        )

    def _do_delete(self, spec: ModelSpec) -> None:
        removed = self.registry.delete(spec)
        if removed:
            self.set_status("ok", f"deleted {spec.name} from this device")
            self._sys(f"✖ deleted {spec.name} (~{spec.size_gb:.1f} GB freed)")
        else:
            self.set_status("warn", f"{spec.name} was not present on disk")

    def switch_model(self, spec: ModelSpec) -> None:
        if spec.name == self.spec.name:
            self.set_status("info", f"{spec.name} is already active")
            return
        try:
            if self.on_model_change is not None:
                self.on_model_change(spec)
        except Exception as e:
            self.set_status("err", f"could not load {spec.name}: {type(e).__name__}: {e}")
            return
        self.spec = spec
        self.messages.clear()
        self.blocks.clear()
        self._sys(f"switched to {spec.name} ({spec.params_b:.1f}B {spec.quant})")
        self.set_status("ok", f"active model: {spec.name}")

    # ----- generation -----------------------------------------------------------
    def _start_generation(self) -> None:
        self._cancel.clear()
        self.blocks.append({"kind": "assistant", "text": "", "streaming": True})
        self._gen_active = True
        self.set_status("info", "generating… (Ctrl-C to stop)")
        msgs = list(self.messages)
        getter = self._engine_getter
        cfg = self.cfg
        cancel = self._cancel
        q = self._q

        def worker() -> None:
            start = time.perf_counter()
            count = 0
            gen = None
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
                    q.put(("tok", tok))
                    count += 1
            except Exception as e:
                q.put(("error", f"{type(e).__name__}: {e}"))
            finally:
                if gen is not None and hasattr(gen, "close"):
                    with contextlib.suppress(Exception):
                        gen.close()
            elapsed = time.perf_counter() - start
            q.put(("done", (count, elapsed, cancel.is_set())))

        threading.Thread(target=worker, daemon=True).start()

    def handle_key(self, key: str) -> None:
        self._dirty = True  # input buffer / overlay / status changed → repaint
        if key == K.CTRL_C:
            if self._gen_active or self._download is not None:
                self._cancel.set()
                if self._gen_active:
                    self.set_status("warn", "interrupting…")
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
            if kind == "tok":
                if self.blocks and self.blocks[-1]["kind"] == "assistant":
                    self.blocks[-1]["text"] += sanitize_model_text(val)
                    dirty = True
            elif kind == "done":
                count, elapsed, interrupted = val
                if self.blocks and self.blocks[-1]["kind"] == "assistant":
                    block = self.blocks[-1]
                    block["streaming"] = False
                    text = block["text"]
                    self.messages.append({"role": "assistant", "content": text})
                    tps = count / elapsed if elapsed > 0 else 0.0
                    if not interrupted:
                        self._last_tps = tps
                    label = "interrupted" if interrupted else "measured this turn"
                    block["meta"] = f"{count} tokens in {elapsed:.1f}s, {tps:.1f} tok/s ({label})"
                self._gen_active = False
                self.set_status("info", "ready — /help for commands")
                dirty = True
            elif kind == "error":
                if self.blocks and self.blocks[-1]["kind"] == "assistant":
                    self.blocks[-1]["streaming"] = False
                    self.blocks[-1]["meta"] = f"stream error: {val} — partial response kept"
                self._gen_active = False
                self.set_status("err", "stream error — partial response kept")
                dirty = True
            elif kind == "dl_done":
                ok, msg = val
                self._download = None
                self.set_status("ok" if ok else "err", msg)
                self._sys(("✓ " if ok else "✖ ") + sanitize_model_text(msg))
                dirty = True
        if self._download is not None:
            pct = self._download_progress()
            if pct != self._last_dl_pct:
                self._last_dl_pct = pct
                spec = self._download
                self.set_status(
                    "info",
                    f"downloading {spec.name} {pct:5.1f}% ({self._downloaded_gb(spec):.2f} / "
                    f"{spec.size_gb:.1f} GB)",
                )
                dirty = True
        return dirty

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

    # ----- rendering ------------------------------------------------------------
    def _header_text(self) -> str:
        est = estimate_fit(
            self.spec.params_b,
            self.spec.quant,
            self.profile.recommended_context_length,
            self.profile.available_ram_gb,
            self.geometry,
        )
        verdict = est.verdict.value.replace("_", " ")
        used = self.estimated_context_tokens()
        tps = f"{self._last_tps:.1f} tok/s" if self._last_tps else "tok/s —"
        return (
            f"{self.spec.name} ({self.spec.params_b:.1f}B {self.spec.quant}) | "
            f"ctx {used}/{self.profile.recommended_context_length} | "
            f"fit: {verdict} (headroom {est.headroom_gb:.1f} GB) | {tps}"
        )

    def _header_render(self, width: int) -> Text:
        g = self.glyphs
        text = Text()
        text.append(f"{g['brand']} eigencore v{__version__}", style=f"bold {TOKENS['brand']}")
        text.append("  │  ", style=TOKENS["dim"])
        text.append(f"{g['ok']} ", style=TOKENS["ok"])
        header = self._header_text()
        text.append(header, style="")
        # context mini-bar inside the header
        used = self.estimated_context_tokens()
        limit = max(self.profile.recommended_context_length, 1)
        bar_w = max(6, min(16, width // 8))
        filled = int(bar_w * min(used / limit, 1.0))
        bar = g["bar"] * filled + g["empty"] * (bar_w - filled)
        text.append("  │  ctx ", style=TOKENS["dim"])
        text.append(bar, style=TOKENS["ok"] if used < 0.8 * limit else TOKENS["warn"])
        if len(text) > width:
            text.truncate(width, overflow="ellipsis")
        return text

    def _status_render(self, width: int) -> Text:
        g = self.glyphs
        kind, msg = self.status
        icon = {"ok": g["ok"], "warn": g["warn"], "err": g["err"], "info": g["info"]}.get(
            kind, g["info"]
        )
        color = {
            "ok": TOKENS["ok"],
            "warn": TOKENS["warn"],
            "err": TOKENS["deny"],
            "info": TOKENS["dim"],
        }.get(kind, TOKENS["dim"])
        text = Text()
        text.append(f"{icon} ", style=color)
        text.append(msg)
        text.truncate(width)
        return text

    def _input_render(self, width: int) -> Text:
        g = self.glyphs
        prompt = f"{g['prompt']} "
        avail = max(1, width - len(prompt))
        buf = self.input.buf
        cur = self.input.cursor
        # horizontal scroll so the cursor stays visible
        off = 0
        if cur >= avail:
            off = cur - avail + 1
        visible = buf[off : off + avail]
        cur_in_view = cur - off
        text = Text()
        text.append(prompt, style=f"bold {TOKENS['brand']}")
        text.append(visible[:cur_in_view])
        cursor_char = visible[cur_in_view : cur_in_view + 1] or " "
        text.append(cursor_char, style="reverse")
        text.append(visible[cur_in_view + 1 :])
        return text

    def _block_render(self, block: dict):
        g = self.glyphs
        kind = block["kind"]
        if kind == "user":
            return Text.assemble(
                (f"You {g['prompt']} ", f"bold {TOKENS['brand']}"), (block["text"], "")
            )
        if kind == "assistant":
            label = Text(f"{g['brand']} EigenCore", style=f"bold {TOKENS['brand']}")
            body = block["text"]
            if block.get("streaming"):
                content = Text(body)
            else:
                if "md" not in block:
                    block["md"] = Markdown(body) if body else Text("")
                content = block["md"]
            meta = block.get("meta")
            return Group(label, content, Text(meta, style=TOKENS["dim"]) if meta else Text(""))
        # sys
        return Text(f"{g['info']} {block['text']}", style=TOKENS["dim"])

    def _transcript_render(self, width: int, height: int):
        return (
            Group(*(self._block_render(b) for b in self.blocks))
            if self.blocks
            else Text(
                f"{self.glyphs['info']} say something — /models to pick a model, /help for commands",
                style=TOKENS["dim"],
            )
        )

    def _frame(self):
        width = self.console.width
        height = self.console.height
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body", ratio=1),
            Layout(name="status", size=1),
            Layout(name="input", size=1),
        )
        layout["header"].update(self._header_render(width))
        if self.overlay is not None:
            layout["body"].update(self.overlay.render(width, max(3, height - 4)))
        else:
            layout["body"].update(self._transcript_render(width, max(3, height - 4)))
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
        try:
            with KeyReader() as reader:
                live = Live(self._frame(), console=console, screen=True, auto_refresh=False)
                with live:
                    last_size = (console.width, console.height)
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
                        size = (console.width, console.height)
                        if dirty or size != last_size:
                            live.update(self._frame(), refresh=True)
                            last_size = size
        except KeyboardInterrupt:
            self._cancel.set()
        finally:
            console.file.write("\x1b[?25h")  # show cursor
            console.file.flush()
        self.console.print(f"\n{self.glyphs['info']} session ended")
