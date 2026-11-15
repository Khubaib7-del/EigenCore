"""
Line-mode terminal chat UI (`eigencore chat --plain`, non-TTY, NO_COLOR,
TERM=dumb), docs/design/TUI.md §3.2 / handbook 5.3:

    EigenCore <ver> | <model> <quant> | ctx <set> | fit: <verdict> (+<h> GB) | tools: off
    you> Explain this function.
    eigencore> ...
    [decode <n> tok/s measured | prefill <t> s | <n> tokens]

- Same information as the full-screen view, no colour, no box drawing, one
  line per event. Never clears scrollback.
- Streaming token-by-token; measured stats only after the turn, labelled.
  Token counts come from the engine (tokenizer); without one, stream chunks
  are counted and labelled as chunks.
- Slash commands: /help, /model, /context, /clear, /exit; unknown commands
  suggest the closest match.
- Ctrl-C mid-stream stops the reply and keeps the history; a mid-stream error
  keeps the partial reply and never regenerates.
- Chat has no tools: the tools field reads the attached policy ("off").
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass

from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.fit import ModelGeometry, estimate_fit
from eigencore.hal.profiler import HardwareProfile
from eigencore.models.registry import ModelSpec
from eigencore.tui.theme import TerminalCaps, probe_terminal

from eigencore.tui.sanitize import sanitize

SLASH_COMMANDS = ("/help", "/model", "/context", "/clear", "/exit")


def sanitize_model_text(text: str) -> str:
    """Remove terminal control sequences, keep newline/tab (tui/sanitize.py)."""
    return sanitize(text)


@dataclass
class ChatUiConfig:
    plain: bool = False
    max_tokens: int = 512
    temperature: float = 0.7
    prompt_lookup: bool = False  # PLD: off by default until owner measurements


class ChatUI:
    """Interactive line-mode chat loop. Rendering is separated from the engine
    so a dropped/repainted frame can never drop or duplicate model events."""

    def __init__(
        self,
        engine: InferenceEngine,
        spec: ModelSpec,
        profile: HardwareProfile,
        geometry: ModelGeometry | None = None,
        policy_label: str | None = None,
        approval_card=None,  # unused: chat has no tools (kept for API compatibility)
        cfg: ChatUiConfig | None = None,
        caps: TerminalCaps | None = None,
        out=None,
        inp=None,
        policy=None,
        tools=None,
    ):
        from eigencore.tui.components import tools_label

        self.engine = engine
        self.spec = spec
        self.profile = profile
        self.geometry = geometry
        # tools state is read from what is attached, never a promise
        self.policy_label = policy_label or tools_label(policy, tools)
        self.approval_card = approval_card
        self.cfg = cfg or ChatUiConfig()
        self.caps = caps or probe_terminal(force_plain=self.cfg.plain)
        if self.caps.plain:
            self.cfg.plain = True
        self._out = out  # file-like for output (tests); None → stdout / rich console
        self._inp = inp  # callable returning next line (tests); None → input()
        self.messages: list[dict[str, str]] = []
        self._console = None
        if not self.cfg.plain:
            from rich.console import Console

            self._console = Console()

    # ----- output helpers -------------------------------------------------
    def _print(self, text: str = "") -> None:
        if self._console is not None:
            self._console.print(text, markup=False, highlight=False)
        elif self._out is not None:
            self._out.write(text + "\n")
        else:
            print(text, flush=True)

    def _write(self, text: str) -> None:
        text = sanitize(text)
        if self._console is not None:
            self._console.print(text, end="", markup=False, highlight=False)
        elif self._out is not None:
            self._out.write(text)
        else:
            print(text, end="", flush=True)

    def _input(self, prompt: str) -> str:
        if self._inp is not None:
            return self._inp(prompt)
        return input(prompt)

    # ----- header / status -------------------------------------------------
    def _fit(self):
        return estimate_fit(
            self.spec.params_b,
            self.spec.quant,
            self.profile.recommended_context_length,
            self.profile.available_ram_gb,
            self.geometry,
        )

    def header(self) -> str:
        from eigencore import __version__

        est = self._fit()
        fit = f"fit: {est.verdict.value}"
        if est.verdict.value != "unknown":
            fit += f" ({est.headroom_gb:+.1f} GB est.)"
        return (
            f"EigenCore {__version__} | {self.spec.name} {self.spec.quant} | "
            f"ctx {self.profile.recommended_context_length} | {fit} | tools: {self.policy_label}"
        )

    def print_banner(self) -> None:
        self._print(self.header())
        if not self.resident_ok():
            est = self._fit()
            self._print(
                f"[!] model {est.weights_gb:.1f} GB > free RAM {est.available_ram_gb:.1f} GB: "
                "reads from SSD on every token, decode will be much slower"
            )
        self._print("Type a message. /help for commands, /exit to quit.")

    def resident_ok(self) -> bool:
        try:
            return self._fit().resident
        except Exception:
            return True

    # ----- commands ---------------------------------------------------------
    def _handle_command(self, cmd: str) -> bool:
        """Returns True to continue the session, False to exit."""
        cmd = cmd.strip().lower()
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            self._print("commands: " + " ".join(SLASH_COMMANDS))
            self._print("  /model    show active model and fit estimate")
            self._print("  /context  message count and context use (estimate)")
            self._print("  /clear    clear conversation history")
            self._print("  /exit     end session (Ctrl-D also exits)")
            self._print("  Ctrl-C    stop the reply that is streaming; history is kept")
        elif cmd == "/model":
            self._print(self.header())
        elif cmd == "/context":
            used = sum(max(len(m["content"]) // 4, 1) for m in self.messages)
            limit = self.profile.recommended_context_length
            pct = 100 * used / max(limit, 1)
            warn = "[!] " if pct >= 85 else ""
            self._print(
                f"{warn}context: {len(self.messages)} messages | ~{used} of {limit} tokens "
                f"(~{pct:.0f}%, estimate: chars / 4)"
            )
        elif cmd == "/clear":
            self.messages.clear()
            self._print("history cleared")
        elif cmd.startswith("/"):
            suggestion = difflib.get_close_matches(cmd, SLASH_COMMANDS, n=1)
            hint = f" Did you mean {suggestion[0]}?" if suggestion else ""
            self._print(f"[x] unknown command: {cmd}.{hint} /help lists commands.")
        return True

    # ----- main loop ---------------------------------------------------------
    def run(self) -> None:
        from eigencore.tui.render import stats_from_run

        self.print_banner()
        while True:
            try:
                user_input = self._input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._print("")
                self._print("session ended")
                break
            if not user_input:
                continue
            if user_input.startswith("/"):
                if not self._handle_command(user_input):
                    self._print("session ended")
                    break
                continue

            self.messages.append({"role": "user", "content": user_input})
            self._write("eigencore> ")
            t_start = time.perf_counter()
            t_first = None
            chunks: list[str] = []
            interrupted = False
            error = None
            stream_config = GenerationConfig(
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
                prompt_lookup=self.cfg.prompt_lookup,
            )
            stream_fn = getattr(self.engine, "stream_messages", None)
            try:
                if stream_fn is not None:
                    gen = stream_fn(self.messages, stream_config)
                else:
                    # engine without multi-turn streaming: single-prompt stream.
                    # Chosen BEFORE streaming, so an error mid-stream never
                    # falls through to a second generation.
                    gen = self.engine.stream(user_input, stream_config)
                for token in gen:
                    if not token:
                        continue
                    if t_first is None:
                        t_first = time.perf_counter()
                    self._write(token)
                    chunks.append(sanitize(token))
            except KeyboardInterrupt:
                # Ctrl-C mid-stream: keep history, do not regenerate or double-print
                interrupted = True
            except Exception as e:
                # stream error: keep what we have, never regenerate
                error = f"{type(e).__name__}: {e}"
            t_end = time.perf_counter()
            text = "".join(chunks)
            self.messages.append({"role": "assistant", "content": text})
            if not text.endswith("\n"):
                self._print("")
            if error is not None:
                self._print(f"[x] stream error: {sanitize(error)} (partial reply kept)")
                continue
            stats = stats_from_run(
                self.engine, text, len(chunks), t_start, t_first, t_end, interrupted=interrupted
            )
            self._print(stats.plain() if not interrupted else "[stopped]")

    def render_approval_card(self, tool_name: str, args: dict) -> bool:
        """Plain approval prompt (§3.5). Chat has no tools, so nothing calls
        this from `eigencore chat`; kept for API compatibility. Default denies."""
        from eigencore.agent.policy import redact_args
        from eigencore.tui.components import ApprovalCardModel, plain_approval

        safe_args = redact_args(args)
        path = str(next((v for k, v in safe_args.items() if "path" in k), ""))
        model = ApprovalCardModel(
            tool=tool_name,
            path=path or "-",
            action=", ".join(f"{k}={v!r}" for k, v in safe_args.items()),
        )
        return plain_approval(model, ask=self._input, out=self._print)
