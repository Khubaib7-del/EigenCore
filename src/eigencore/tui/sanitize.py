"""
One sanitiser for all untrusted text shown in the terminal (docs/design/TUI.md §2).

Model output (and, later, tool output) must never drive the terminal or the
UI. `sanitize()`:

1. strips 7-bit escape sequences: CSI (`ESC [ ...`), OSC (`ESC ] ... BEL|ST`),
   DCS / SOS / PM / APC strings (`ESC P|X|^|_ ... ST`) and two-byte escapes;
2. strips their 8-bit C1 forms (`\\x9b` CSI, `\\x9d` OSC, `\\x90` DCS, ...);
3. drops every remaining C0 / C1 control character except `\\n` and `\\t`
   (`\\r` is dropped too: a bare carriage return lets text overwrite itself).

Callers then print the result as `rich.text.Text(...)` or with `markup=False`,
so `[bold]` / `[/]` in model text are shown literally and can never crash Rich.
"""

from __future__ import annotations

import re

# String-type sequences: OSC / DCS / SOS / PM / APC, 7-bit and 8-bit introducers,
# terminated by BEL or ST (ESC \ or \x9c). Unterminated strings run to the end.
_STRING_SEQ = re.compile(
    r"(?:\x1b[\]PX^_]|[\x90\x98\x9d\x9e\x9f])[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c|$)"
)
# CSI: ESC [ or \x9b, parameter bytes, intermediate bytes, final byte.
_CSI_SEQ = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]?")
# Remaining two-byte escapes (ESC 7, ESC c, ESC ( B, ...).
_ESC_SEQ = re.compile(r"\x1b[ -/]*[0-~]?")
# Any other control char except \n and \t.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Return `text` with every terminal control sequence removed (keeps \\n, \\t)."""
    if not text:
        return ""
    text = _STRING_SEQ.sub("", text)
    text = _CSI_SEQ.sub("", text)
    text = _ESC_SEQ.sub("", text)
    return _CONTROL.sub("", text)
