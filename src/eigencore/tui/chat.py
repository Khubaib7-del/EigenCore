"""
Terminal chat UI — Phase 5.

A lightweight interactive chat over InferenceEngine. Design rules from
`docs/design/terminal-chat.md` that are implemented here:

- Plain-text fallback (`--plain` or non-TTY stdin): no panels, no color codes
  beyond ANSI-stripped text, works piped.
- Streaming output token-by-token with measured tok/s shown only after the
  turn completes (labeled as measured for this session).
- Header line shows model, quant, context budget, and fit verdict from the
  P2 estimator; tool policy status is shown when a policy is attached.
- Slash commands: /help, /model, /context, /clear, /exit. Unknown commands
  suggest the closest match instead of failing silently.
- Never clears scrollback; Ctrl-C / Ctrl-D end the session cleanly.

NOT implemented (still pending in P4/P5): full-screen adaptive layout,
approval cards, tool invocation from chat, OS process isolation.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass

from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.fit import ModelGeometry, estimate_fit
from eigencore.hal.profiler import HardwareProfile
from eigencore.models.registry import ModelSpec
from eigencore.tui.theme import TerminalCaps, probe_terminal

SLASH_COMMANDS = ("/help", "/model", "/context", "/clear", "/approvals", "/exit")

# Strip ANSI/OSC/CSI control sequences but keep newline/tab — model output is
# untrusted and must never drive the terminal or Rich markup.
_CONTROL_SEQ = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")


def sanitize_model_text(text: str) -> str:
    """Remove terminal control sequences, keep newline/tab."""
    return _CONTROL_SEQ.sub("", text)


@dataclass
class ChatUiConfig:
    plain: bool = False
    max_tokens: int = 512
    temperature: float = 0.7
    prompt_lookup: bool = False  # PLD: off by default until owner measurements


class ChatUI:
    """Interactive chat loop. Rendering is separated from the engine so a
    dropped/repainted frame can never drop or duplicate model events."""

    def __init__(
        self,
        engine: InferenceEngine,
        spec: ModelSpec,
        profile: HardwareProfile,
        geometry: ModelGeometry | None = None,
        policy_label: str = "no tools",
        approval_card=None,  # callable(tool_name, args) -> bool; renders a card and collects the answer
        cfg: ChatUiConfig | None = None,
        caps: TerminalCaps | None = None,
        out=None,
        inp=None,
    ):
        self.engine = engine
        self.spec = spec
        self.profile = profile
        self.geometry = geometry
        self.policy_label = policy_label
        self.approval_card = approval_card
        self.cfg = cfg or ChatUiConfig()
        self.caps = caps or probe_terminal(force_plain=self.cfg.plain)
        if self.caps.plain:
            self.cfg.plain = True
        self._out = out  # file-like for output (tests); None → rich console
        self._inp = inp  # callable returning next line (tests); None → console.input
        self.messages: list[dict[str, str]] = []
        self._console = None
        if not self.cfg.plain:
            from rich.console import Console

            self._console = Console()

    # ----- output helpers -------------------------------------------------
    def _print(self, text: str = "") -> None:
        if self._console is not None:
            self._console.print(text)
        elif self._out is not None:
            self._out.write(text + "\n")
        else:
            print(text)

    def _write(self, text: str) -> None:
        text = sanitize_model_text(text)
        if self._console is not None:
            from rich.markup import escape

            self._console.print(escape(text), end="", markup=True, highlight=False)
        elif self._out is not None:
            self._out.write(text)
        else:
            print(text, end="", flush=True)

    def _input(self, prompt: str) -> str:
        if self._inp is not None:
            return self._inp(prompt)
        if self._console is not None:
            return self._console.input(prompt)
        return input(prompt)

    # ----- header / status -------------------------------------------------
    def header(self) -> str:
        est = estimate_fit(
            self.spec.params_b,
            self.spec.quant,
            self.profile.recommended_context_length,
            self.profile.available_ram_gb,
            self.geometry,
        )
        verdict = est.verdict.value.replace("_", " ")
        prefix = "eigencore chat"
        if self.caps.tier == "narrow":
            return (
                f"{prefix} | {self.spec.name} {self.spec.quant} | ctx {self.profile.recommended_context_length} | "
                f"{verdict}"
            )
        return (
            f"{prefix} | {self.spec.name} ({self.spec.params_b:.1f}B {self.spec.quant}) | "
            f"ctx {self.profile.recommended_context_length} | "
            f"fit: {verdict} (headroom {est.headroom_gb:.1f} GB) | tools: {self.policy_label}"
        )

    def print_banner(self) -> None:
        if self.cfg.plain:
            self._print(self.header())
            self._print("Type a message. /help for commands, /exit to quit.")
        else:
            from rich.panel import Panel

            self._console.print(
                Panel(
                    self.header() + "\nType a message. /help for commands, /exit to quit.",
                    title="[bold]EigenCore Chat[/]",
                    border_style="cyan",
                )
            )

    # ----- commands ---------------------------------------------------------
    def _handle_command(self, cmd: str) -> bool:
        """Returns True to continue the session, False to exit."""
        cmd = cmd.strip().lower()
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            self._print("Commands: " + ", ".join(SLASH_COMMANDS))
            self._print("  /model    show active model and fit estimate")
            self._print("  /context  show message count and context usage estimate")
            self._print("  /clear    clear conversation history")
            self._print("  /exit     end session")
        elif cmd == "/model":
            self._print(self.header())
        elif cmd == "/context":
            used = sum(max(len(m["content"]) // 4, 1) for m in self.messages)
            self._print(
                f"messages: {len(self.messages)} | estimated tokens in history: {used} / "
                f"{self.profile.recommended_context_length}"
            )
        elif cmd == "/clear":
            self.messages.clear()
            self._print("history cleared")
        elif cmd == "/approvals":
            self._print(
                f"approval mode: {getattr(self, 'approval_mode', 'ask')}; "
                "tool calls require approval before running (enforced in the executor, not the UI)"
            )
        elif cmd.startswith("/"):
            suggestion = difflib.get_close_matches(cmd, SLASH_COMMANDS, n=1)
            hint = f" Did you mean {suggestion[0]}?" if suggestion else ""
            self._print(f"unknown command: {cmd}.{hint} /help lists commands.")
        return True

    # ----- main loop ---------------------------------------------------------
    def run(self) -> None:
        self.print_banner()
        while True:
            try:
                user_input = self._input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                self._print("\nsession ended")
                break
            if not user_input:
                continue
            if user_input.startswith("/"):
                if not self._handle_command(user_input):
                    self._print("session ended")
                    break
                continue

            self.messages.append({"role": "user", "content": user_input})
            self._write("\nEigenCore: ")
            start = time.perf_counter()
            chunks: list[str] = []
            interrupted = False
            stream_config = GenerationConfig(
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
                prompt_lookup=self.cfg.prompt_lookup,
            )
            try:
                for token in self.engine.stream_messages(self.messages, stream_config):
                    self._write(token)
                    chunks.append(token)
            except KeyboardInterrupt:
                # Ctrl-C mid-stream: keep history, do not regenerate or double-print
                interrupted = True
                self._print("\n[interrupted — partial response kept]")
            except AttributeError:
                # engine without stream_messages: fall back to single-prompt stream
                text_chunks = list(self.engine.stream(user_input, stream_config))
                for token in text_chunks:
                    self._write(token)
                chunks = text_chunks
            except Exception as e:
                # stream error: keep what we have, never regenerate
                self._print(f"\n[stream error: {type(e).__name__}: {e} — partial response kept]")
            elapsed = time.perf_counter() - start
            self.messages.append({"role": "assistant", "content": "".join(chunks)})
            n = len(chunks)
            tps = n / elapsed if elapsed > 0 else 0.0
            label = "interrupted" if interrupted else "measured this turn"
            self._print(f"\n[{n} tokens in {elapsed:.1f}s, {tps:.1f} tok/s ({label})]")

    def render_approval_card(self, tool_name: str, args: dict) -> bool:
        """Render the approval card and collect y/n. Default denies."""
        from eigencore.agent.policy import redact_args

        safe_args = redact_args(args)
        lines = [
            "┌ Tool request ─────────────────────────",
            f"│ tool: {tool_name}",
            f"│ args: {safe_args}",
            "│ [y] approve once   [s] approve session   [n] deny",
            "└──────────────────────────────────────────",
        ]
        if self.cfg.plain or self._console is None:
            for line in lines:
                self._print(line)
        else:
            from rich.panel import Panel

            self._console.print(
                Panel("\n".join(lines[1:-1]), title="[bold]Tool request[/]", border_style="yellow")
            )
        try:
            answer = self._input("approve? [y/s/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "s")
