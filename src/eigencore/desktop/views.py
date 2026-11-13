"""Flet views for the desktop app. Views are constructible headless (no Page
required to build the control tree); `attach(page)` enables live updates.
All state changes arrive as backend event tuples via `handle_event`."""

from __future__ import annotations

import flet as ft

from eigencore.desktop import theme as T
from eigencore.desktop.backend import DesktopBackend, ModelRow

ASSETS = __import__("pathlib").Path(__file__).parent / "assets"


def _safe_update(control) -> None:
    try:
        control.update()
    except Exception:
        pass  # not mounted yet (headless tests)


def fit_chip(fit: str) -> ft.Control:
    color, word = T.STATE_STYLES.get(fit, T.STATE_STYLES["unknown"])
    return T.chip(word, color)


class ChatView:
    def __init__(self, backend: DesktopBackend) -> None:
        self.backend = backend
        self.model_dropdown = ft.Dropdown(
            label="model",
            options=[ft.DropdownOption(n) for n in backend.downloaded_names()],
            on_select=self._on_model_change,
        )
        self.status = ft.Text("no model loaded — pick one in Models", color=T.MUTED_DARK, size=12)
        self.chat_log = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        self.input = ft.TextField(
            hint_text="Message EigenCore… (Enter to send, Shift+Enter for newline)",
            multiline=True,
            min_lines=1,
            max_lines=6,
            on_submit=lambda e: self.submit(),
            expand=True,
        )
        self.send_btn = ft.IconButton(
            icon=ft.Icons.SEND, tooltip="send", on_click=lambda e: self.submit()
        )
        self.stop_btn = ft.IconButton(
            icon=ft.Icons.STOP_CIRCLE_OUTLINED,
            tooltip="stop",
            on_click=lambda e: self.backend.stop(),
            visible=False,
        )
        self._assistant_md: ft.Markdown | None = None
        self._page = None

    def attach(self, page) -> None:
        self._page = page

    def refresh_models(self) -> None:
        names = self.backend.downloaded_names()
        self.model_dropdown.options = [ft.DropdownOption(n) for n in names]
        if self.backend.active_spec is not None:
            self.model_dropdown.value = self.backend.active_spec.name
        _safe_update(self.model_dropdown)

    def _on_model_change(self, e) -> None:
        if self.model_dropdown.value:
            self.backend.switch_model(self.model_dropdown.value)

    def build(self) -> ft.Control:
        header = ft.Row(
            [
                ft.Image(src=str(ASSETS / "logo.svg"), width=28, height=28),
                ft.Text("Chat", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                self.model_dropdown,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        body = ft.Container(
            content=ft.Column([self.chat_log], expand=True),
            expand=True,
            padding=ft.Padding.all(12),
        )
        composer = ft.Row(
            [self.input, self.send_btn, self.stop_btn],
            vertical_alignment=ft.CrossAxisAlignment.END,
        )
        return ft.Column(
            [header, ft.Divider(height=1), body, self.status, composer],
            expand=True,
            spacing=6,
        )

    # ----- events -------------------------------------------------------------
    def submit(self) -> None:
        text = self.input.value or ""
        if not text.strip():
            return
        self.input.value = ""
        self._add_bubble("You", text, T.CORE_ICE)
        self.backend.send(text)
        self.send_btn.visible = False
        self.stop_btn.visible = True
        _safe_update(self.send_btn)
        _safe_update(self.stop_btn)
        _safe_update(self.input)

    def _add_bubble(self, who: str, text: str, color: str) -> ft.Control:
        bubble = ft.Container(
            content=ft.Column(
                [
                    ft.Text(who, size=11, weight=ft.FontWeight.BOLD, color=color),
                    ft.Markdown(text, selectable=True)
                    if who != "You"
                    else ft.Text(text, selectable=True),
                ],
                spacing=2,
            ),
            bgcolor=T.PANEL,
            border_radius=10,
            padding=ft.Padding.all(10),
        )
        self.chat_log.controls.append(bubble)
        _safe_update(self.chat_log)
        return bubble

    def handle_event(self, ev) -> None:
        kind = ev[0]
        if kind == "tok":
            if self._assistant_md is None:
                bubble = self._add_bubble("EigenCore", "", T.CORE)
                self._assistant_md = bubble.content.controls[1]
                assert isinstance(self._assistant_md, ft.Markdown)
            self._assistant_md.value += ev[1]
            _safe_update(self._assistant_md)
        elif kind == "done":
            stats = ev[1]
            self._assistant_md = None
            if stats.get("error"):
                self.status.value = f"error: {stats['error']} — partial reply kept"
                self.status.color = T.DANGER
            else:
                label = "interrupted" if stats["interrupted"] else "measured this turn"
                self.status.value = (
                    f"{stats['tokens']} tokens in {stats['seconds']:.1f}s, "
                    f"{stats['tok_per_sec']:.1f} tok/s ({label})"
                )
                self.status.color = T.MUTED_DARK
            self.send_btn.visible = True
            self.stop_btn.visible = False
            _safe_update(self.status)
            _safe_update(self.send_btn)
            _safe_update(self.stop_btn)
        elif kind in ("engine", "switched", "dl_done", "deleted"):
            if kind == "engine":
                state, msg = ev[1], ev[2]
                self.status.value = msg
                self.status.color = {"ready": T.OK, "error": T.DANGER}.get(state, T.CAUTION)
                _safe_update(self.status)
            if kind in ("switched", "dl_done", "deleted"):
                self.refresh_models()


class ModelsView:
    def __init__(self, backend: DesktopBackend) -> None:
        self.backend = backend
        self.rows_host = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)
        self.status = ft.Text("", size=12, color=T.MUTED_DARK)
        self._page = None

    def attach(self, page) -> None:
        self._page = page

    def build(self) -> ft.Control:
        self.rebuild()
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("Models", size=20, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        ft.IconButton(
                            icon=ft.Icons.REFRESH,
                            tooltip="refresh",
                            on_click=lambda e: self.rebuild(),
                        ),
                    ]
                ),
                ft.Text(
                    f"cache: {self.backend.registry.cache_dir} — delete frees disk on this device",
                    size=11,
                    color=T.MUTED_DARK,
                ),
                ft.Divider(height=1),
                ft.Container(self.rows_host, expand=True),
                self.status,
            ],
            expand=True,
            spacing=6,
        )

    def rebuild(self) -> None:
        self.rows_host.controls.clear()
        for row in self.backend.model_rows():
            self.rows_host.controls.append(self._row(row))
        _safe_update(self.rows_host)

    def _row(self, row: ModelRow) -> ft.Control:
        chips = ft.Row([fit_chip(row.fit)], spacing=6)
        state = (
            T.chip("DOWNLOADED", T.OK) if row.downloaded else T.chip("not downloaded", T.MUTED_DARK)
        )
        if row.is_active:
            state = T.chip("ACTIVE", T.CORE_ICE)
        actions = []
        if not row.downloaded:
            actions.append(
                ft.FilledButton(
                    f"Download {row.size_gb:.1f} GB",
                    icon=ft.Icons.DOWNLOAD,
                    on_click=lambda e, n=row.name: self.backend.download(n),
                )
            )
        else:
            actions.append(
                ft.OutlinedButton(
                    "Use",
                    icon=ft.Icons.CHECK_CIRCLE_OUTLINE,
                    on_click=lambda e, n=row.name: self.backend.switch_model(n),
                    disabled=row.is_active,
                )
            )
            actions.append(
                ft.OutlinedButton(
                    "Delete",
                    icon=ft.Icons.DELETE_OUTLINE,
                    on_click=lambda e, n=row.name: self._confirm_delete(n),
                    disabled=row.is_active,
                )
            )
        return ft.Container(
            content=ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(row.name, weight=ft.FontWeight.BOLD),
                            ft.Text(
                                f"{row.params_b:.1f}B · {row.quant} · {row.size_gb:.1f} GB · ctx {row.context_length} · {row.task}",
                                size=11,
                                color=T.MUTED_DARK,
                            ),
                            ft.Text(row.description, size=11, color=T.MUTED_DARK, italic=True),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    chips,
                    state,
                    ft.Row(actions, spacing=6),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=T.PANEL,
            border_radius=10,
            padding=ft.Padding.all(10),
        )

    def _confirm_delete(self, name: str) -> None:
        def do_delete(e):
            self.backend.delete(name)
            if self._page is not None:
                self._page.close_dialog()

        def cancel(e):
            if self._page is not None:
                self._page.close_dialog()

        if self._page is None:
            self.backend.delete(name)  # headless/tests: no dialog
            return
        dlg = ft.AlertDialog(
            title=ft.Text(f"Delete {name}?"),
            content=ft.Text("Removes the model files from this device. This cannot be undone."),
            actions=[
                ft.TextButton("Delete", on_click=do_delete),
                ft.TextButton("Cancel", on_click=cancel),
            ],
        )
        self._page.open(dlg)

    def handle_event(self, ev) -> None:
        kind = ev[0]
        if kind in ("dl_progress", "dl_done", "deleted", "switched"):
            if kind == "dl_done":
                _, name, ok, msg = ev
                self.status.value = msg
                self.status.color = T.OK if ok else T.DANGER
                _safe_update(self.status)
            if kind == "deleted":
                self.status.value = f"deleted {ev[1]}" if ev[2] else f"could not delete {ev[1]}"
                self.status.color = T.OK if ev[2] else T.DANGER
                _safe_update(self.status)
            self.rebuild()


class ProfileView:
    def __init__(self, backend: DesktopBackend) -> None:
        self.backend = backend

    def build(self) -> ft.Control:
        p = self.backend.profile
        cards = [
            self._card(
                "CPU", f"{p.cpu_name} ({p.arch})", f"{p.physical_cores}P/{p.logical_cores}L cores"
            ),
            self._card(
                "RAM",
                f"{p.available_ram_gb:.1f} GB free",
                f"of {p.total_ram_gb:.1f} GB · ~{p.estimated_bandwidth_gbps:.0f} GB/s",
            ),
            self._card(
                "GPU",
                p.gpu_name or "none detected",
                f"{p.available_vram_gb:.1f} GB VRAM free"
                if p.available_vram_gb > 0
                else "CPU-only inference",
            ),
            self._card(
                "Recommendation",
                f"ctx {p.recommended_context_length}",
                f"≤ {p.max_model_params_b:.0f}B params · {p.optimal_quantization} · est. {p.estimated_tokens_per_sec:.0f} tok/s (estimate, not measured)",
            ),
        ]
        return ft.Column(
            [
                ft.Text("Hardware profile", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(height=1),
                ft.Row([ft.Container(c, expand=True) for c in cards], spacing=8),
                ft.Text(
                    "Speed figures are estimates until you run eigencore benchmark on this machine.",
                    size=11,
                    color=T.CAUTION,
                    italic=True,
                ),
            ],
            expand=True,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        )

    def _card(self, title: str, big: str, small: str) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(title.upper(), size=11, color=T.MUTED_DARK, weight=ft.FontWeight.BOLD),
                    ft.Text(big, size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(small, size=11, color=T.MUTED_DARK),
                ],
                spacing=2,
            ),
            bgcolor=T.PANEL,
            border_radius=10,
            padding=ft.Padding.all(12),
        )

    def handle_event(self, ev) -> None:
        pass
