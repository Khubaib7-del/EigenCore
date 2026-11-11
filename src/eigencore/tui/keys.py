"""
Cross-platform raw key reading for the full-screen TUI.

No third-party dependency: POSIX uses termios cbreak (ISIG off, so Ctrl-C
arrives as a key we handle ourselves), Windows uses msvcrt. `parse_keys` is a
pure function so key parsing is testable without a terminal.
"""

from __future__ import annotations

import os
import sys
from collections import deque

# Named keys are plain strings; printable keys are the character itself.
ENTER = "enter"
BACKSPACE = "backspace"
TAB = "tab"
ESC = "esc"
UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
HOME = "home"
END = "end"
DELETE = "delete"
PGUP = "pgup"
PGDN = "pgdn"
CTRL_C = "ctrl-c"
CTRL_D = "ctrl-d"
UNKNOWN = "unknown"

_ESC_MAP = {
    "[A": UP,
    "[B": DOWN,
    "[C": RIGHT,
    "[D": LEFT,
    "[H": HOME,
    "[F": END,
    "[1~": HOME,
    "[4~": END,
    "[3~": DELETE,
    "[5~": PGUP,
    "[6~": PGDN,
    "OA": UP,
    "OB": DOWN,
    "OC": RIGHT,
    "OD": LEFT,
    "OH": HOME,
    "OF": END,
}


def parse_keys(data: str) -> list[str]:
    """Parse a chunk of input into key tokens. A lone ESC is `esc`."""
    keys: list[str] = []
    i = 0
    while i < len(data):
        ch = data[i]
        if ch == "\x1b":
            if i + 1 >= len(data):
                keys.append(ESC)
                i += 1
                continue
            # Greedily consume a CSI/SS3 sequence: ESC [ ... final byte, or ESC <char>
            j = i + 1
            if j < len(data) and data[j] in "[O":
                j += 1
                while j < len(data) and not ("@" <= data[j] <= "~"):
                    j += 1
                seq = data[i + 1 : j + 1] if j < len(data) else data[i + 1 :]
                keys.append(_ESC_MAP.get(seq, UNKNOWN if seq else ESC))
                i = j + 1
            else:
                keys.append(ESC)
                keys.append(data[j])
                i = j + 1
        elif ch in ("\r", "\n"):
            keys.append(ENTER)
            i += 1
        elif ch == "\t":
            keys.append(TAB)
            i += 1
        elif ch in ("\x7f", "\x08"):
            keys.append(BACKSPACE)
            i += 1
        elif ch == "\x03":
            keys.append(CTRL_C)
            i += 1
        elif ch == "\x04":
            keys.append(CTRL_D)
            i += 1
        elif ch < " " or ch == "\x7f":
            keys.append(UNKNOWN)
            i += 1
        else:
            keys.append(ch)
            i += 1
    return keys


class KeyReader:
    """Context manager reading key tokens from stdin in cbreak mode.

    `poll(timeout)` returns the next key token or None. Restores the
    terminal mode on exit even on exception.
    """

    def __init__(self, stream=None):
        self._stream = stream or sys.stdin
        self._fd = self._stream.fileno()
        self._windows = os.name == "nt"
        self._old_attrs = None
        self._buf: deque[str] = deque()
        self._msvcrt = None
        self._termios = None

    def __enter__(self) -> "KeyReader":
        if self._windows:
            import msvcrt

            self._msvcrt = msvcrt
        else:
            import termios

            self._termios = termios
            self._old_attrs = termios.tcgetattr(self._fd)
            new = list(self._old_attrs)
            new[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
            new[6][termios.VMIN] = 1
            new[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSADRAIN, new)
        return self

    def __exit__(self, *exc) -> None:
        if not self._windows and self._old_attrs is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def _read_raw(self, timeout: float) -> str:
        """Read available input ('' if none within timeout)."""
        if self._windows:
            import time

            deadline = time.monotonic() + timeout
            while True:
                if self._msvcrt.kbhit():
                    ch = self._msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):  # special-key prefix
                        code = self._msvcrt.getwch()
                        mapped = {
                            "H": "[H",
                            "P": "[3~",
                            "K": "[D",
                            "M": "[C",
                            "G": "[1~",
                            "S": "[4~",
                            "Q": "[B",
                            "R": "[A",
                            "I": "[5~",
                            "?": "[6~",
                        }.get(code, "")
                        return "\x1b" + mapped
                    return ch
                if time.monotonic() >= deadline:
                    return ""
                time.sleep(0.005)
        import select

        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return ""
        try:
            data = os.read(self._fd, 64)
        except OSError:
            return ""
        return data.decode(errors="replace")

    def poll(self, timeout: float = 0.0) -> str | None:
        """Next key token, or None if none arrives within `timeout` seconds."""
        if not self._buf:
            raw = self._read_raw(timeout)
            if raw:
                self._buf.extend(parse_keys(raw))
        return self._buf.popleft() if self._buf else None
