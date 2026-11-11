"""
EigenCore ASCII banner + reveal animation, shown when `eigencore chat` starts
on a TTY. Hand-drawn art (no pyfiglet dependency); reveal is a short
line-by-line colour ramp from dim to brand. Skips animation when the output
is not a terminal or NO_COLOR/plain is in effect.
"""

from __future__ import annotations

import time

from eigencore import __version__
from eigencore.tui.theme import TOKENS

BANNER_LINES = [
    r" _____ _                   ____ ___  ____  _____ ",
    r"| ____(_) __ _  ___ _ __  / ___/ _ \|  _ \| ____|",
    r"|  _| | |/ _` |/ _ \ '_ \| |  | | | | |_) |  _|  ",
    r"| |___| | (_| |  __/ | | | |__| |_| |  _ <| |___ ",
    r"|_____|_|\__, |\___|_| |_|\____\___/|_| \_\_____|",
    r"         |___/                                     ",
]

TAGLINE = f"local-first LLM runtime  •  v{__version__}"


def banner_text() -> str:
    """Plain banner (no styling) — used by tests and --plain output."""
    return "\n".join(line.rstrip() for line in BANNER_LINES) + "\n" + TAGLINE


def render_banner(console, animate: bool | None = None, delay: float = 0.04) -> None:
    """Print the banner with a reveal animation on interactive terminals."""
    if animate is None:
        animate = console.is_terminal
    brand = TOKENS["brand"]
    dim = TOKENS["dim"]
    for i, line in enumerate(BANNER_LINES):
        # ramp: first lines start dim, later lines land on brand
        frac = i / max(len(BANNER_LINES) - 1, 1)
        style = dim if frac < 0.5 else brand
        console.print(line.rstrip(), style=style, highlight=False)
        if animate:
            time.sleep(delay * (1.0 if frac >= 0.5 else 0.6))
    console.print(TAGLINE, style=dim, highlight=False)
