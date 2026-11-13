"""EigenCore desktop app entry point (Flet 1.x).

Run with `eigencore-gui` or `python -m eigencore.desktop`. Pack for Windows
with: flet pack <this file> --name EigenCore --icon <assets/eigencore.ico>
"""

from __future__ import annotations

import asyncio
import threading
import webbrowser
from pathlib import Path

import flet as ft

from eigencore import __version__
from eigencore.desktop import theme as T
from eigencore.desktop import updater
from eigencore.desktop.backend import DesktopBackend
from eigencore.desktop.views import ChatView, ModelsView, ProfileView

ASSETS = Path(__file__).parent / "assets"


async def main(page: ft.Page) -> None:
    page.title = f"EigenCore v{__version__}"
    page.theme = T.build_theme()
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    try:
        # frameless: the OS title bar (min/max/close) is replaced by the quiet
        # custom title bar below (WindowDragArea + ghost buttons)
        page.window.frameless = True
        page.window.width = 1080
        page.window.height = 720
        page.window.min_width = 800
        page.window.min_height = 560
        page.window.resizable = True
    except AttributeError:
        pass  # web/preview targets without a native window

    def win_btn(icon, tooltip, action, hover=None):
        return ft.IconButton(
            icon=icon,
            tooltip=tooltip,
            icon_size=14,
            icon_color=T.MUTED_DARK,
            width=36,
            height=28,
            on_click=action,
            style=ft.ButtonStyle(
                color=T.MUTED_DARK,
                bgcolor={
                    ft.ControlState.HOVERED: hover or "#33373D",
                    ft.ControlState.DEFAULT: "transparent",
                },
            ),
        )

    # Flet 1.x window actions are coroutines: they must be awaited, otherwise
    # the click does nothing (that was the "close button does nothing" bug).
    async def close_window(e) -> None:
        try:
            await page.window.close()
        except Exception:
            await page.window.destroy()

    def toggle_maximize(e) -> None:
        page.window.maximized = not page.window.maximized
        page.update()

    def minimize(e) -> None:
        page.window.minimized = True
        page.update()

    title_bar = ft.Container(
        content=ft.Row(
            [
                ft.WindowDragArea(
                    content=ft.Row(
                        [
                            ft.Image(src=str(ASSETS / "logo.svg"), width=18, height=18),
                            ft.Text(f"EigenCore v{__version__}", size=12, color=T.MUTED_DARK),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    expand=True,
                    height=32,
                    maximizable=True,  # double-click the title bar to maximize
                ),
                win_btn(ft.Icons.REMOVE, "Minimize", minimize),
                win_btn(ft.Icons.CROP_SQUARE, "Maximize / restore", toggle_maximize),
                win_btn(ft.Icons.CLOSE, "Close", close_window, hover="#C42B1C"),
            ],
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=T.GRAPHITE,
        height=32,
        padding=ft.Padding.only(left=12, right=4),
    )

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
        try:
            if content.page is not None:  # page property raises RuntimeError pre-add
                content.update()
        except RuntimeError:
            pass  # not mounted yet — the initial show() runs before page.add

    update = UpdateButton(page, close_window)
    sidebar_footer = ft.Container(
        content=ft.Column(
            [
                ft.Divider(height=1),
                ft.Text(f"v{__version__}", size=11, color=T.MUTED_DARK),
                update.control,
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
            [
                title_bar,
                ft.Row([rail, ft.VerticalDivider(width=1), content], expand=True),
            ],
            expand=True,
            spacing=0,
        )
    )
    show(0)

    # auto-select the first downloaded model so chat is ready
    backend.ensure_active()

    # one quiet check at startup; the sidebar button shows the result
    update.check_in_background()

    async def pump() -> None:
        while True:
            ev = await asyncio.to_thread(backend.events.get)
            for v in (chat, models, profile):
                v.handle_event(ev)

    asyncio.create_task(pump())


class UpdateButton:
    """Sidebar button: Check for updates -> Update to vX -> Downloading n% -> Installing.

    Installed app (EigenCore-Setup.exe): downloads the new installer and runs it
    silently; it closes this window, updates in place and reopens EigenCore.
    Models in ~/.eigencore stay where they are.
    Portable .exe or pip install: opens the download page instead.
    """

    def __init__(self, page: ft.Page, close_window) -> None:
        self.page = page
        self.close_window = close_window
        self.release: updater.Release | None = None
        self.busy = False
        self.button = ft.FilledButton(
            "Check for updates",
            icon=ft.Icons.SYSTEM_UPDATE_ALT,
            width=160,
            on_click=self.on_click,
        )
        self.note = ft.Text(
            "", size=10, color=T.MUTED_DARK, text_align=ft.TextAlign.CENTER, width=160
        )
        self.control = ft.Column(
            [self.button, self.note],
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _set(self, label: str, note: str = "", icon=None, disabled: bool = False) -> None:
        self.button.content = label
        if icon is not None:
            self.button.icon = icon
        self.button.disabled = disabled
        self.note.value = note
        try:
            if self.control.page is not None:
                self.control.update()
        except RuntimeError:
            pass  # not on the page yet

    def _show_result(self) -> None:
        rel = self.release
        if rel is None:
            self._set("Check for updates", "Could not reach GitHub", ft.Icons.SYSTEM_UPDATE_ALT)
        elif updater.is_newer(rel.version, __version__):
            self._set(f"Update to v{rel.version}", "New version available", ft.Icons.DOWNLOAD)
        else:
            self._set("Up to date", f"v{__version__} is the latest", ft.Icons.CHECK)

    def check_in_background(self) -> None:
        def run() -> None:
            self.release = updater.check()
            self._show_result()

        threading.Thread(target=run, daemon=True).start()

    async def on_click(self, e) -> None:
        if self.busy:
            return
        rel = self.release
        if rel is None or not updater.is_newer(rel.version, __version__):
            self._set("Checking...", "", disabled=True)
            self.release = await asyncio.to_thread(updater.check)
            self._show_result()
            return
        if not updater.is_installed() or not rel.installer_url:
            # portable .exe or pip: nothing to update in place
            webbrowser.open(rel.page_url)
            return
        self.busy = True
        self._set("Downloading 0%", "Keep EigenCore open", disabled=True)

        def progress(pct: float) -> None:
            self._set(f"Downloading {pct:.0f}%", "Keep EigenCore open", disabled=True)

        try:
            path = await asyncio.to_thread(updater.download, rel.installer_url, None, progress)
            self._set("Installing...", "EigenCore will restart", disabled=True)
            updater.install(path)
        except Exception as exc:  # network or disk error: let the user retry
            self.busy = False
            self._set(f"Update to v{rel.version}", f"Update failed: {exc}"[:80], ft.Icons.DOWNLOAD)
            return
        await asyncio.sleep(1.0)
        await self.close_window(None)


def cli() -> None:
    ft.run(main)


if __name__ == "__main__":
    cli()
