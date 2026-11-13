"""EigenCore desktop app entry point (Flet 1.x).

Run with `eigencore-gui` or `python -m eigencore.desktop`. Pack for Windows
with: flet pack <this file> --name EigenCore --icon <assets/eigencore.ico>
"""

from __future__ import annotations

import asyncio

import flet as ft

from eigencore import __version__
from eigencore.desktop import theme as T
from eigencore.desktop.backend import DesktopBackend
from eigencore.desktop.views import ChatView, ModelsView, ProfileView


async def main(page: ft.Page) -> None:
    page.title = f"EigenCore v{__version__}"
    page.theme = T.build_theme()
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    try:
        page.window.width = 1080
        page.window.height = 720
        page.window.min_width = 800
        page.window.min_height = 560
    except AttributeError:
        pass  # web/preview targets without a native window

    backend = DesktopBackend()
    chat = ChatView(backend)
    models = ModelsView(backend)
    profile = ProfileView(backend)
    for v in (chat, models, profile):
        v.attach(page)

    views = {0: chat, 1: models, 2: profile}
    content = ft.Container(expand=True, padding=ft.Padding.all(8))

    def show(i: int) -> None:
        content.content = views[i].build()
        content.update()

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.CHAT_BUBBLE_OUTLINE, selected_icon=ft.Icons.CHAT_BUBBLE, label="Chat"
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.INVENTORY_2_OUTLINED,
                selected_icon=ft.Icons.INVENTORY_2,
                label="Models",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.MONITOR_OUTLINED, selected_icon=ft.Icons.MONITOR, label="Profile"
            ),
        ],
        on_change=lambda e: show(rail.selected_index),
    )

    show(0)
    page.add(ft.Row([rail, ft.VerticalDivider(width=1), content], expand=True))

    # auto-select the first downloaded model so chat is ready
    backend.ensure_active()

    async def pump() -> None:
        while True:
            ev = await asyncio.to_thread(backend.events.get)
            for v in (chat, models, profile):
                v.handle_event(ev)

    asyncio.create_task(pump())


def cli() -> None:
    ft.run(main)


if __name__ == "__main__":
    cli()
