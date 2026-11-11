"""
TUI theme + terminal capability probe (docs/design/TUI.md §2, handbook 5.2).

One module owns all colour tokens and the capability decision:
  - plain mode when: --plain flag, non-TTY stdin, NO_COLOR set, TERM=dumb.
  - narrow tier below 80 columns: no box drawing, shorter header.
Every colour has a text equivalent; plain mode loses no information.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

# Design tokens (handbook 5.2): teal = ok/measured/fits, amber = tight/approval/
# estimate, clay = denied/error, indigo = brand/focus.
TOKENS = {
    "ok": "teal",
    "warn": "dark_orange",
    "deny": "indian_red",
    "brand": "medium_purple",
    "dim": "grey50",
}


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
    plain = force_plain or not sys.stdin.isatty()
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
    color = not plain and sys.stdout.isatty()
    return TerminalCaps(width=width, height=height, plain=plain, color=color)
