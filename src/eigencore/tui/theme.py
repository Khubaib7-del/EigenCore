"""
TUI theme + terminal capability probe (docs/design/TUI.md §2, handbook 5.2).

One module owns all colour tokens, glyph sets and the capability decision:
  - plain mode when: --plain flag, non-TTY stdin, NO_COLOR set, TERM=dumb.
  - wide tier at >= 100 columns, narrow below (header collapses, no rules).
  - ASCII glyphs when the output encoding is not UTF-8, on the legacy Windows
    console (conhost without Windows Terminal; its fonts lack ◩ ◐ etc.), or
    when EIGENCORE_ASCII=1.
  - reduced motion (static orb) when EIGENCORE_REDUCED_MOTION=1.
Every colour has a text equivalent; plain mode loses no information.
No hex literal for a UI colour lives outside this module.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

# Handbook 5.2 truecolor values (Rich downgrades to 256/16 colours itself).
CORE = "#8C98FF"  # core-ice: brand / focus / "you" label
OK = "#4FB3A9"  # teal: ok, measured, fits_with_margin, allowed
CAUTION = "#E0A84A"  # amber: tight, needs approval, estimate
DANGER = "#E07A5A"  # clay: does_not_fit, denied, error
MUTED = "#9AA1AB"  # secondary text on dark
LIGHT = "#E8E6E1"  # primary text on dark (assistant label)
RULE = "#3A3F47"  # hairline rules / panel borders
SELECT_BG = "#1F2330"  # selected row background in pickers

TOKENS = {
    "ok": OK,
    "warn": CAUTION,
    "caution": CAUTION,
    "deny": DANGER,
    "danger": DANGER,
    "brand": CORE,
    "core": CORE,
    "dim": MUTED,
    "muted": MUTED,
    "light": LIGHT,
    "rule": RULE,
}

# Pygments theme for code blocks; background is dropped so the user's
# terminal background shows through.
CODE_THEME = "monokai"

# Plain-mode text equivalents of the signal colours (handbook 5.2).
PLAIN_PREFIX = {"ok": "[ok]", "caution": "[!]", "danger": "[x]"}

UNICODE_GLYPHS = {
    "brand": "◩",
    "prompt": "›",
    "ok": "●",
    "warn": "▲",
    "err": "✖",
    "info": "·",
    "dl": "↓",
    "bar": "━",
    "empty": "━",
    "rule": "─",
    "sep": "·",
    "cursor": "▍",
    "orb": ("◐", "◓", "◑", "◒"),
    "orb_static": "●",
    "tree_mid": "├─ ",
    "tree_end": "└─ ",
    "tree_pipe": "│  ",
    "approx": "≈",
    "ellipsis": "…",
}

ASCII_GLYPHS = {
    "brand": "[#]",
    "prompt": ">",
    "ok": "o",
    "warn": "!",
    "err": "x",
    "info": "*",
    "dl": "v",
    "bar": "#",
    "empty": "-",
    "rule": "-",
    "sep": "|",
    "cursor": "_",
    "orb": ("|", "/", "-", "\\"),
    "orb_static": "*",
    "tree_mid": "|-- ",
    "tree_end": "`-- ",
    "tree_pipe": "|   ",
    "approx": "~",
    "ellipsis": "...",
}


def legacy_windows_console() -> bool:
    """conhost without Windows Terminal: limited fonts, avoid exotic glyphs."""
    return os.name == "nt" and not os.environ.get("WT_SESSION")


def use_ascii(stream=None) -> bool:
    if os.environ.get("EIGENCORE_ASCII"):
        return True
    stream = stream if stream is not None else sys.stdout
    enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
    if "utf8" not in enc and "utf" not in enc:
        return True
    return legacy_windows_console()


def pick_glyphs(ascii: bool | None = None) -> dict:
    """Unicode glyphs when the terminal can show them, else ASCII."""
    if ascii is None:
        ascii = use_ascii()
    return dict(ASCII_GLYPHS if ascii else UNICODE_GLYPHS)


def reduced_motion() -> bool:
    return os.environ.get("EIGENCORE_REDUCED_MOTION", "") not in ("", "0")


@dataclass
class TerminalCaps:
    width: int
    height: int
    plain: bool
    color: bool

    @property
    def tier(self) -> str:
        if self.plain:
            return "plain"
        if self.width >= 100:
            return "wide"
        return "narrow"


def probe_terminal(force_plain: bool = False, width: int | None = None) -> TerminalCaps:
    """Detect terminal capabilities. Never raises; unknowns degrade to plain."""
    try:
        interactive = sys.stdin.isatty()
    except Exception:
        interactive = False
    plain = force_plain or not interactive
    if os.environ.get("NO_COLOR"):
        plain = True
    term = os.environ.get("TERM", "")
    if term == "dumb":
        plain = True
    if width is None:
        try:
            size = shutil.get_terminal_size()
            width, height = size.columns, size.lines
        except Exception:
            width, height = 80, 24
    else:
        height = 24
    try:
        out_tty = sys.stdout.isatty()
    except Exception:
        out_tty = False
    color = not plain and out_tty
    return TerminalCaps(width=width, height=height, plain=plain, color=color)


def tier_for_width(width: int) -> str:
    return "wide" if width >= 100 else "narrow"
