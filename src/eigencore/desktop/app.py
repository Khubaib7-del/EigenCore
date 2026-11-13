"""EigenCore desktop app entry point (Flet 1.x).

Run with `eigencore-gui` or `python -m eigencore.desktop`. Pack for Windows
with: flet pack <this file> --name EigenCore --icon <assets/eigencore.ico>
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import flet as ft

from eigencore import __version__
from eigencore.desktop import theme as T
from eigencore.desktop.backend import EXE_URL, DesktopBackend, is_newer, latest_release
from eigencore.desktop.views import ChatView, ModelsView, ProfileView

ASSETS = Path(__file__).parent / "assets"


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

    update_banner = ft.Container(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.NEW_RELEASES_OUTLINED, color=T.CAUTION),
                ft.Text("", size=12, expand=True),
                ft.TextButton("Download update", on_click=lambda e: page.launch_url(EXE_URL)),
            ]
        ),
        bgcolor=T.PANEL,
        border_radius=8,
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        visible=False,
    )

    def show(i: int) -> None:
        content.content = views[i].build()
        try:
            if content.page is not None:  # page property raises RuntimeError pre-add
                content.update()
        except RuntimeError:
            pass  # not mounted yet — the initial show() runs before page.add

    sidebar_footer = ft.Container(
        content=ft.Column(
            [
                ft.Divider(height=1),
                ft.Text(f"v{__version__}", size=11, color=T.MUTED_DARK),
                ft.FilledButton(
                    "Download",
                    icon=ft.Icons.DOWNLOAD,
                    on_click=lambda e: page.launch_url(EXE_URL),
                    tooltip="Download the latest EigenCore.exe directly",
                    width=160,
                ),
            ],
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.only(bottom=12),
    )

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        leading=ft.Container(
            content=ft.Image(src=str(ASSETS / "logo.svg"), width=36, height=36),
            padding=ft.Padding.only(top=12, bottom=8),
        ),
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
        trailing=sidebar_footer,
        on_change=lambda e: show(rail.selected_index),
    )

    page.add(
        ft.Column(
            [update_banner, ft.Row([rail, ft.VerticalDivider(width=1), content], expand=True)],
            expand=True,
            spacing=0,
        )
    )
    show(0)

    # auto-select the first downloaded model so chat is ready
    backend.ensure_active()

    # one background check for a newer release; show the banner if newer
    def check_update() -> None:
        rel = latest_release()
        if rel and rel.get("tag") and is_newer(rel["tag"], __version__):
            row = update_banner.content
            row.controls[
                1
            ].value = f"v{rel['tag'].lstrip('v')} is available — the Download button below grabs the new .exe"
            update_banner.visible = True
            try:
                if update_banner.page is not None:
                    update_banner.update()
            except RuntimeError:
                pass

    threading.Thread(target=check_update, daemon=True).start()

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
