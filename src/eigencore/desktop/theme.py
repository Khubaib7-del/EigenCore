"""Desktop theme from docs/design/DESIGN_HANDBOOK.md — dark "graphite" ground,
one brand accent (core indigo), state colours teal/amber/clay always paired
with a word or icon."""

from __future__ import annotations

import flet as ft

# Handbook palette (dark-surface values)
INK = "#1B1E22"
GRAPHITE = "#1F2226"
PANEL = "#26292F"
STONE = "#ECE9E2"
LIGHT = "#E8E6E1"
MUTED_DARK = "#9AA1AB"
CORE = "#5B6CF0"
CORE_ICE = "#8C98FF"
OK = "#4FB3A9"
CAUTION = "#E0A84A"
DANGER = "#E07A5A"

# State → (colour, label word). Every state colour is paired with text.
STATE_STYLES = {
    "ok": (OK, "OK"),
    "tight": (CAUTION, "TIGHT"),
    "no": (DANGER, "DOES NOT FIT"),
    "unknown": (MUTED_DARK, "UNKNOWN"),
}


def build_theme() -> ft.Theme:
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=CORE,
            on_primary=LIGHT,
            secondary=CORE_ICE,
            on_secondary=INK,
            surface=GRAPHITE,
            on_surface=LIGHT,
            error=DANGER,
            on_error=LIGHT,
            outline=MUTED_DARK,
        ),
        use_material3=True,
    )


def chip(text: str, color: str, bgcolor: str | None = None) -> ft.Container:
    """State chip: coloured text + word, optionally filled. Monochrome-safe."""
    return ft.Container(
        content=ft.Text(text, size=11, weight=ft.FontWeight.BOLD, color=color),
        bgcolor=bgcolor,
        border_radius=8,
        padding=ft.Padding.symmetric(horizontal=8, vertical=3),
    )
