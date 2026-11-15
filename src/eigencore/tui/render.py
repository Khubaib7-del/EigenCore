"""
Rendering helpers for the chat TUI: streaming-safe Markdown, framed code
blocks, the git-style tree component, and the turn-stats formatter.

Markdown: assistant text is rendered with Rich's Markdown (markdown-it), with
two element overrides:
  - fenced / indented code becomes a framed panel with a language label and
    Rich `Syntax` highlighting on the terminal's own background;
  - headings are left-aligned (Rich centres h1 inside a box).
CommonMark treats an unclosed ``` fence as running to the end of the
document, so a reply that is still streaming inside a code block renders as
an (open) framed code panel on every frame: no raw backticks, no reflow jump
when the fence closes. All text here is already sanitised (tui/sanitize.py);
Markdown never interprets Rich markup, so `[/]` is shown literally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from eigencore.tui import theme as T

MD_THEME = Theme(
    {
        "markdown.code": f"bold {T.CORE}",
        "markdown.code_block": "",
        "markdown.block_quote": T.MUTED,
        "markdown.list": T.CORE,
        "markdown.item.bullet": f"bold {T.CORE}",
        "markdown.item.number": T.CORE,
        "markdown.hr": T.RULE,
        "markdown.h1": f"bold {T.CORE}",
        "markdown.h2": f"bold {T.CORE}",
        "markdown.h3": "bold",
        "markdown.h4": "bold italic",
        "markdown.h5": "italic",
        "markdown.h6": f"italic {T.MUTED}",
        "markdown.link": T.CORE,
        "markdown.link_url": f"underline {T.MUTED}",
        "markdown.table.border": T.RULE,
        "markdown.table.header": "bold",
    }
)

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def open_fence(text: str) -> bool:
    """True when `text` ends inside an unclosed ``` / ~~~ fence."""
    opener: Optional[str] = None
    for line in text.split("\n"):
        m = _FENCE.match(line)
        if not m:
            continue
        mark = m.group(1)
        if opener is None:
            opener = mark
        elif mark[0] == opener[0] and len(mark) >= len(opener) and not line.strip()[len(mark) :]:
            opener = None
    return opener is not None


class _LeftHeading(Heading):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"
        if self.tag in ("h1", "h2"):
            yield Text("")
        yield text


class _FramedCode(CodeBlock):
    """Code block as a framed panel: language label top-left, Syntax inside."""

    @classmethod
    def create(cls, markdown: Markdown, token) -> "_FramedCode":
        info = (token.info or "").strip()
        lexer = info.partition(" ")[0]
        block = cls(lexer or "text", markdown.code_theme)
        block.label = lexer or "code"
        block.ascii = getattr(markdown, "ascii", False)
        streaming = getattr(markdown, "streaming_open", False)
        n_lines = getattr(markdown, "n_lines", 0)
        block.open = bool(streaming and token.map and token.map[1] >= n_lines)
        return block

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip("\n")
        if not code.strip():
            code = " "
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            background_color="default",
            word_wrap=True,
            padding=0,
        )
        sub = None
        if getattr(self, "open", False):
            sub = Text("writing" + ("..." if self.ascii else "…"), style=T.MUTED)
        yield Panel(
            syntax,
            title=Text(f" {self.label} ", style=T.MUTED),
            title_align="left",
            subtitle=sub,
            subtitle_align="right",
            box=box.ASCII if self.ascii else box.ROUNDED,
            border_style=T.RULE,
            padding=(0, 1),
            expand=True,
        )


class ChatMarkdown(Markdown):
    elements = dict(Markdown.elements)
    elements["fence"] = _FramedCode
    elements["code_block"] = _FramedCode
    elements["heading_open"] = _LeftHeading


def render_markdown(text: str, streaming: bool = False, ascii: bool = False):
    """Renderable for assistant text. Never raises: falls back to plain Text."""
    if not text:
        return Text("")
    try:
        md = ChatMarkdown(text, code_theme=T.CODE_THEME, hyperlinks=False)
        md.ascii = ascii
        md.streaming_open = streaming and open_fence(text)
        md.n_lines = text.count("\n") + 1
        return md
    except Exception:  # pragma: no cover - defensive
        return Text(text)


# ----- tree component ---------------------------------------------------------

TreeNode = Union[str, tuple]  # "leaf" or ("label", [children])


def render_tree(
    nodes: Sequence[TreeNode],
    ascii: bool = False,
    style: str = "",
    branch_style: str = T.RULE,
) -> Text:
    """Git-style tree (├─ └─ │) for file lists or tool steps.

    `nodes` is a list of strings (leaves) or `(label, children)` tuples.
    A flat list of paths can be turned into nodes with `tree_from_paths`.
    """
    g = T.pick_glyphs(ascii)
    out = Text()

    def walk(items: Sequence[TreeNode], prefix: str) -> None:
        for i, node in enumerate(items):
            last = i == len(items) - 1
            label, children = (node, ()) if isinstance(node, str) else node
            if out:
                out.append("\n")
            out.append(prefix + (g["tree_end"] if last else g["tree_mid"]), style=branch_style)
            out.append(label if isinstance(label, Text) else Text(str(label), style=style))
            if children:
                walk(
                    children,
                    prefix + ("   " if not ascii else "    ") if last else prefix + g["tree_pipe"],
                )

    walk(list(nodes), "")
    return out


def tree_from_paths(paths: Iterable[str]) -> list[TreeNode]:
    """['src/a.py', 'src/b/c.py', 'README.md'] -> nested nodes (dirs end with '/')."""
    root: dict = {}
    for p in paths:
        parts = [x for x in str(p).replace("\\", "/").split("/") if x]
        cur = root
        for part in parts[:-1]:
            cur = cur.setdefault(part + "/", {})
        if parts:
            cur.setdefault(parts[-1], {})

    def build(d: dict) -> list[TreeNode]:
        items = sorted(d.items(), key=lambda kv: (not kv[0].endswith("/"), kv[0]))
        return [(k, build(v)) if v or k.endswith("/") else k for k, v in items]

    return build(root)


# ----- turn stats ----------------------------------------------------------------


@dataclass
class TurnStats:
    """What one assistant turn measured. Only real measurements go in here."""

    units: int  # completion tokens (or stream chunks when no tokenizer count)
    unit: str  # "tokens" | "chunks"
    prefill_s: Optional[float]  # time to first streamed text (prefill + 1st token)
    decode_s: Optional[float]  # first streamed text -> end of stream
    total_s: float
    interrupted: bool = False
    error: Optional[str] = None

    @property
    def decode_rate(self) -> Optional[float]:
        """Units per second after the first one (the first is part of prefill)."""
        if self.decode_s is None or self.decode_s <= 0 or self.units < 2:
            return None
        return (self.units - 1) / self.decode_s

    def wide(self, sep: str = "·") -> str:
        unit_rate = "tok/s" if self.unit == "tokens" else "chunks/s"
        if self.interrupted:
            parts = [f"stopped {sep} {self.units} {self.unit} kept"]
            if self.prefill_s is not None:
                parts.append(f"prefill {self.prefill_s:.2f} s")
            return f" {sep} ".join(parts)
        parts = []
        rate = self.decode_rate
        if rate is not None:
            parts.append(f"decode {rate:.1f} {unit_rate}")
        if self.prefill_s is not None:
            parts.append(f"prefill {self.prefill_s:.2f} s")
        parts.append(f"{self.units} {self.unit}")
        label = "measured this turn"
        if self.unit != "tokens":
            label += " (stream chunks, not tokens)"
        parts.append(label)
        return f" {sep} ".join(parts)

    def plain(self) -> str:
        if self.interrupted:
            return f"[stopped | {self.units} {self.unit} kept]"
        unit_rate = "tok/s" if self.unit == "tokens" else "chunks/s"
        parts = []
        rate = self.decode_rate
        if rate is not None:
            parts.append(f"decode {rate:.1f} {unit_rate} measured")
        if self.prefill_s is not None:
            parts.append(f"prefill {self.prefill_s:.2f} s")
        parts.append(f"{self.units} {self.unit}")
        return "[" + " | ".join(parts) + "]"


def stats_from_run(
    engine,
    text: str,
    chunks: int,
    t_start: float,
    t_first: Optional[float],
    t_end: float,
    interrupted: bool,
    error: Optional[str] = None,
) -> TurnStats:
    """Build TurnStats from what the engine reports, else from UI timing.

    Preference order for the token count: the engine's own per-stream stats
    (`engine.last_stream_stats["completion_tokens"]`), then the engine
    tokenizer (`engine.count_tokens(text)`), else stream chunks, labelled as
    chunks. Times are wall-clock around the stream as seen by the UI.
    """
    units, unit = chunks, "chunks"
    prefill = (t_first - t_start) if t_first is not None else None
    decode = (t_end - t_first) if t_first is not None else None
    stats = getattr(engine, "last_stream_stats", None)
    if not interrupted and error is None and isinstance(stats, dict):
        n = stats.get("completion_tokens")
        if isinstance(n, int) and n >= 0:
            units, unit = n, "tokens"
        if isinstance(stats.get("prefill_s"), (int, float)):
            prefill = float(stats["prefill_s"])
        if isinstance(stats.get("decode_s"), (int, float)):
            decode = float(stats["decode_s"])
    elif callable(getattr(engine, "count_tokens", None)):
        try:
            units, unit = int(engine.count_tokens(text)), "tokens"
        except Exception:
            pass
    return TurnStats(
        units=units,
        unit=unit,
        prefill_s=prefill,
        decode_s=decode,
        total_s=t_end - t_start,
        interrupted=interrupted,
        error=error,
    )
