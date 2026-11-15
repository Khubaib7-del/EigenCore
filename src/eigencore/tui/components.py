"""
Stand-alone TUI components from docs/design/TUI.md §3:

- `tools_label(policy)`  header tools state, read from the attached policy.
- SSD-paging warning (§3.4): wide renderable, plain lines, and a key handler.
- Tool approval card (§3.5): wide renderable + key state, plain y/N prompt.

The approval card is a renderer only. Chat has no tools today, so nothing in
`eigencore chat` shows it; enforcement lives in the policy layer
(`eigencore.agent.policy`), never in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

from eigencore.tui import keys as K
from eigencore.tui import theme as T
from eigencore.tui.sanitize import sanitize


# ----- header tools label -------------------------------------------------------


def tools_label(policy=None, tools: Optional[Sequence] = None) -> str:
    """Exactly what is attached: 'off' when chat has no tools or no policy."""
    if policy is None or not tools:
        return "off"
    caps = getattr(policy, "allowed_capabilities", None)
    if not caps:
        return "deny-all (policy)"
    names = sorted(getattr(c, "name", str(c)).lower() for c in caps)
    if names == ["read"]:
        return "read-only (policy)"
    return "+".join(names) + " (policy)"


# ----- SSD-paging warning (§3.4) ------------------------------------------------


@dataclass
class SsdWarning:
    model: str
    size_gb: float  # estimated resident size (weights), GiB
    free_gb: float  # free RAM, GiB

    @staticmethod
    def needed(est) -> bool:
        """True when the fit estimate says the file will not be RAM-resident."""
        return est is not None and not getattr(est, "resident", True)

    def headline(self) -> str:
        return f"model {self.size_gb:.1f} GB > free RAM {self.free_gb:.1f} GB"

    BODY = "llama.cpp will memory-map it and read the rest from the SSD on every token."
    NOTE = "Decode will be much slower than the RAM-bandwidth ceiling. A smaller model or quant stays in RAM."

    def render(self, ascii: bool = False):
        head = Text()
        head.append("[!]  ", style=f"bold {T.CAUTION}")
        head.append(f"{self.model} · " if not ascii else f"{self.model} | ", style=T.CAUTION)
        head.append(self.headline(), style=T.CAUTION)
        keys = Text("     ")
        for k, label in (("c", "continue"), ("q", "pick another model"), ("x", "cancel")):
            keys.append(f"[{k}] ", style="bold")
            keys.append(f"{label}      ", style=T.MUTED)
        return Group(
            head,
            Text("     " + self.BODY),
            Text("     " + self.NOTE, style=T.MUTED),
            Text(""),
            keys,
        )

    def plain_lines(self) -> list[str]:
        return [
            f"[!] {self.model}: {self.headline()}",
            f"    {self.BODY}",
            f"    {self.NOTE}",
        ]

    @staticmethod
    def decide(key: str) -> Optional[str]:
        """Map a key to 'continue' | 'pick' | 'cancel' (None = ignore).
        Esc cancels; Enter does nothing (no default 'continue' into paging)."""
        k = key.lower() if len(key) == 1 else key
        return {"c": "continue", "q": "pick", "x": "cancel", K.ESC: "cancel"}.get(k)


def ask_ssd_warning(
    warning: SsdWarning, ask: Callable[[str], str], out: Callable[[str], None]
) -> str:
    """Line-mode version used before load in the CLI. Default is cancel."""
    for line in warning.plain_lines():
        out(line)
    try:
        answer = ask("continue anyway? [c]ontinue / [x] cancel (default x): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "cancel"
    return "continue" if answer in ("c", "continue", "y", "yes") else "cancel"


# ----- tool approval card (§3.5) ----------------------------------------------------


class _Card:
    """Box with a title on the left AND a status on the right of the top
    border (Rich's Panel only has one title per edge)."""

    def __init__(self, body, left: Text, right: Text, width: int, style: str, ascii: bool):
        self.body, self.left, self.right = body, left, right
        self.width, self.style, self.ascii = width, style, ascii

    def __rich_console__(self, console, options):
        from rich.segment import Segment

        w = min(self.width, options.max_width)
        h, v = ("-", "|") if self.ascii else ("─", "│")
        tl, tr, bl, br = ("+", "+", "+", "+") if self.ascii else ("┌", "┐", "└", "┘")
        style = console.get_style(self.style)
        fill = max(1, w - 4 - len(self.left) - len(self.right))
        top = Text(tl + h, style=self.style)
        top.append_text(self.left)
        top.append(h * fill, style=self.style)
        top.append_text(self.right)
        top.append(h + tr, style=self.style)
        yield top
        inner = options.update(width=w - 4, height=None)
        for line in console.render_lines(self.body, inner, pad=True):
            yield Segment(v + " ", style)
            yield from line
            yield Segment(" " + v, style)
            yield Segment.line()
        yield Text(bl + h * (w - 2) + br, style=self.style)


ACTIONS = (("a", "approve once"), ("s", "approve for session"), ("d", "deny"), ("v", "diff"))


@dataclass
class ApprovalCardModel:
    """What the card shows. All strings are sanitised on render."""

    tool: str  # e.g. "write file"
    path: str
    action: str  # e.g. "replace 3 lines"
    inside_workspace: bool = True
    diff: list[str] = field(default_factory=list)  # lines starting with '-', '+' or ' '
    # Sandbox limits as reported by the executor. Empty = nothing to report;
    # the card never invents isolation that does not exist.
    limits: str = ""
    stale: bool = False


class ApprovalCard:
    """Key state for the card. Default focus is deny; Enter on an unfocused
    card does nothing; a stale card cannot be approved."""

    def __init__(self, model: ApprovalCardModel) -> None:
        self.model = model
        self.focus = "d"  # deny
        self.focused = False  # Enter only acts after the user moved focus
        self.show_full_diff = False
        self.result: Optional[str] = None  # "once" | "session" | "deny"

    def handle_key(self, key: str) -> Optional[str]:
        order = [a for a, _ in ACTIONS]
        if key in (K.LEFT, K.RIGHT, K.TAB):
            i = order.index(self.focus)
            i = (i + (-1 if key == K.LEFT else 1)) % len(order)
            self.focus = order[i]
            self.focused = True
            return None
        if key == K.ENTER:
            if not self.focused:
                return None
            key = self.focus
        if key == "v":
            self.show_full_diff = not self.show_full_diff
            return None
        if key in ("a", "s") and self.model.stale:
            return None
        self.result = {"a": "once", "s": "session", "d": "deny", K.ESC: "deny"}.get(key)
        return self.result

    def render(self, ascii: bool = False, width: int = 76):
        m = self.model
        sep = "|" if ascii else "·"
        body = Table.grid(padding=(0, 2))
        body.add_column(style=T.MUTED, no_wrap=True)
        body.add_column()
        where = (
            Text("(inside workspace)", style=T.OK)
            if m.inside_workspace
            else Text("(outside workspace)", style=T.DANGER)
        )
        body.add_row("path", Text.assemble(Text(sanitize(m.path)), "   ", where))
        body.add_row("action", Text(sanitize(m.action)))
        diff_lines = m.diff if self.show_full_diff else m.diff[:8]
        diff = Text()
        for i, line in enumerate(diff_lines):
            line = sanitize(line)
            style = T.DANGER if line.startswith("-") else T.OK if line.startswith("+") else ""
            if i:
                diff.append("\n")
            diff.append(line, style=style)
        if len(m.diff) > len(diff_lines):
            diff.append(f"\n… {len(m.diff) - len(diff_lines)} more lines · [v] diff", style=T.MUTED)
        limits = Text(sanitize(m.limits).replace("·", sep), style=T.MUTED)
        actions = Text()
        if m.stale:
            actions.append(f"stale {sep} re-request", style=f"bold {T.DANGER}")
            actions.append("    ")
        for k, label in ACTIONS:
            disabled = m.stale and k in ("a", "s")
            if k == self.focus:
                style = f"bold {T.CORE}"
            elif disabled:
                style = f"strike {T.MUTED}"
            else:
                style = ""
            actions.append(f"[{k}] {label}", style=style)
            actions.append("   ")
        parts = [body]
        if diff:
            parts.append(diff)
        if m.limits:
            parts.append(limits)
        rule = Text(("-" if ascii else "─") * max(10, width - 4), style=T.CAUTION)
        parts += [rule, actions]
        title = Text(f" tool request {sep} {sanitize(m.tool)} ", style=T.CAUTION)
        status = Text(" stale " if m.stale else " needs approval ", style=T.CAUTION)
        return Group(
            _Card(Group(*parts), title, status, width, T.CAUTION, ascii),
            Text(
                f"  default focus: deny {sep} Enter on an unfocused card does nothing",
                style=T.MUTED,
            ),
        )


def plain_approval(
    model: ApprovalCardModel, ask: Callable[[str], str], out: Callable[[str], None]
) -> bool:
    """Plain mode (§3.5): diff printed, then `approve <tool> <path>? [y/N]`.
    Anything but y/yes denies; a stale request is denied without asking."""
    verb = sanitize(model.tool).split()[0] if model.tool.strip() else "run"
    out(
        f"tool request | {sanitize(model.tool)} | {sanitize(model.path)} | {sanitize(model.action)}"
    )
    for line in model.diff:
        out(sanitize(line))
    if model.stale:
        out("[x] stale request, re-request")
        return False
    try:
        answer = ask(f"approve {verb} {sanitize(model.path)}? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")
