"""Desktop host for the web UI (``ui/``) in a native pywebview window.

The window shows the built React app (``ui/dist``) in WebView2 (Edge
Chromium) on Windows. The UI draws its own title bar; the window is
frameless and its minimize / maximize / close / drag / resize go through
:meth:`WebBridge.window_control` and pywebview's drag region.

All engine work is done by :class:`~eigencore.desktop.backend.DesktopBackend`
(the same backend the Flet app uses). :class:`WebBridge` is the ``js_api``
object: JavaScript calls its public methods through ``window.pywebview.api``,
and backend events are pushed back as one ``CustomEvent("eigencore")`` per
batch, whose ``detail`` is a list of event tuples (JSON arrays) in the shape
documented in backend.py. ``ui/src/api/types.ts`` (``fromTuple``) turns them
into typed events.

Nothing in :class:`WebBridge` needs a display, so it is unit-tested headlessly
with a fake backend; only :func:`run` imports pywebview.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import shutil
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from enum import Flag
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

from eigencore import __version__
from eigencore.desktop import updater as _updater

log = logging.getLogger("eigencore.desktop")

EVENT_NAME = "eigencore"
USER_AGENT_MARK = "EigenCoreDesktop"
# Drag the window from the title bar, but not from its buttons. pywebview walks
# up from the clicked element; with DIRECT_TARGET_ONLY the clicked element
# itself must match, so buttons (and anything inside them) never start a drag.
DRAG_SELECTOR = ".pywebview-drag-region, .pywebview-drag-region *:not(button):not(button *)"
# The UI calls ready() once it is listening; if it never does (blank page,
# broken renderer) the window is closed and the launcher falls back to Flet.
LOAD_TIMEOUT_S = 25.0


class WebUIUnavailable(RuntimeError):
    """The web UI cannot start here (no pywebview, no WebView2, no ui/dist...)."""


# ----- paths ------------------------------------------------------------------


def state_dir() -> Path:
    return Path.home() / ".eigencore"


# WebView2 keeps its profile in state_dir()/webview (it outlives updates and
# uninstalls). These caches are dropped whenever the app version changes;
# Local Storage, IndexedDB and cookies are kept.
WEBVIEW_CACHE_DIRS = (
    ("EBWebView", "Default", "Cache"),
    ("EBWebView", "Default", "Code Cache"),
    ("EBWebView", "Default", "GPUCache"),
    ("EBWebView", "Default", "Service Worker"),
)
VERSION_MARKER = "eigencore-ui-version"


def webview_dir() -> Path:
    return state_dir() / "webview"


def clear_stale_webview_cache(store: Path, version: str = __version__) -> bool:
    """Drop WebView2's HTTP/code caches when ``version`` differs from the last run.

    Returns True if caches were cleared. The marker is only written once every
    folder is gone, so a locked folder is retried on the next start.
    """
    marker = store / VERSION_MARKER
    try:
        if marker.read_text(encoding="utf-8").strip() == version:
            return False
    except OSError:
        pass
    ok = True
    for parts in WEBVIEW_CACHE_DIRS:
        d = store.joinpath(*parts)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            if d.exists():
                ok = False
                log.warning("could not clear the WebView cache folder %s", d)
    if ok:
        try:
            store.mkdir(parents=True, exist_ok=True)
            marker.write_text(version, encoding="utf-8")
        except OSError as e:
            log.warning("could not write %s: %s", marker, e)
    if ok:
        log.info("app version is now %s: cleared the WebView cache", version)
    return True


def ui_start_url(dist: Path, version: str = __version__) -> str:
    """Local path pywebview serves, with ``?v=<version>`` so no cache answers it.

    pywebview strips the query when it picks the folder to serve (its common
    path is the dirname of a path that is not a folder) and keeps it in the
    page URL: ``http://127.0.0.1:<port>/index.html?v=<version>``.
    """
    return f"{dist / 'index.html'}?v={quote(version, safe='')}"


def ui_dist_dir() -> Optional[Path]:
    """Folder holding the built UI (index.html), or None.

    Order: ``EIGENCORE_UI_DIST``; the PyInstaller bundle (``sys._MEIPASS/ui``);
    ``eigencore/desktop/ui`` next to this file; the repo's ``ui/dist``
    (a source checkout after ``npm run build``).
    """
    candidates: list[Path] = []
    env = os.environ.get("EIGENCORE_UI_DIST")
    if env:
        candidates.append(Path(env))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "ui")
    here = Path(__file__).resolve().parent
    candidates.append(here / "ui")
    candidates.append(here.parents[2] / "ui" / "dist")  # src/eigencore/desktop -> repo
    for c in candidates:
        if (c / "index.html").is_file():
            return c
    return None


# ----- serialisation ------------------------------------------------------------

_PROVIDERS = (
    ("tinyllama", "tinyllama"),
    ("phi", "microsoft"),
    ("qwen", "qwen"),
    ("llama", "meta"),
    ("mistral", "mistral"),
    ("gemma", "gemma"),
    ("deepseek", "deepseek"),
    ("olmo", "ai2"),
)
_ACTIVE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*B\s+active", re.IGNORECASE)


def provider_for(name: str, repo_id: str = "") -> str:
    """Logo key for a model (UI only). Unknown families get the Hugging Face mark."""
    key = f"{name} {repo_id}".lower()
    for needle, provider in _PROVIDERS:
        if needle in key:
            return provider
    return "huggingface"


def model_row_dict(row, spec=None) -> dict:
    """ModelRow (+ repo_id, provider, active_params_b) as the UI's ModelRow."""
    d = asdict(row)
    repo_id = getattr(spec, "repo_id", "") if spec is not None else ""
    d["repo_id"] = repo_id
    d["provider"] = provider_for(row.name, repo_id)
    m = _ACTIVE_RE.search(row.description or "")
    if m and spec is not None and getattr(spec, "is_moe", False):
        d["active_params_b"] = float(m.group(1))
    return d


def profile_dict(p) -> dict:
    """HardwareProfile as the UI's HardwareProfile (estimates stay estimates)."""
    isa = p.instruction_sets
    if isinstance(isa, Flag):
        names = [f.name for f in type(isa) if f in isa and f.name and f.name != "NONE"]
    else:
        names = list(isa or [])
    return {
        "cpu_name": p.cpu_name,
        "arch": p.arch,
        "physical_cores": p.physical_cores,
        "logical_cores": p.logical_cores,
        "instruction_sets": names,
        "total_ram_gb": p.total_ram_gb,
        "available_ram_gb": p.available_ram_gb,
        "estimated_bandwidth_gbps": p.estimated_bandwidth_gbps,
        "max_model_params_b": p.max_model_params_b,
        "optimal_quantization": p.optimal_quantization,
        "recommended_threads": p.recommended_threads,
        "recommended_context_length": p.recommended_context_length,
        "gpu_name": p.gpu_name,
        "benchmarked": False,
    }


def event_json(events: list) -> str:
    """Event tuples -> JSON array of arrays (non-JSON values become strings)."""
    return json.dumps([list(e) for e in events], default=str)


def dispatch_script(payload: str) -> str:
    return f"window.dispatchEvent(new CustomEvent({json.dumps(EVENT_NAME)},{{detail:{payload}}}))"


# ----- bridge -------------------------------------------------------------------


class WebBridge:
    """``js_api`` object. Public methods are callable from JavaScript.

    Every attribute is private (``_``-prefixed) so pywebview exposes only the
    methods below. Blocking work (downloads, model loads, generation, update
    checks) runs on worker threads; results come back as events.
    """

    def __init__(
        self,
        backend: Any = None,
        backend_factory: Optional[Callable[[], Any]] = None,
        emit: Optional[Callable[[str], None]] = None,
        updater: Any = _updater,
        state_path: Optional[Path] = None,
        open_url: Callable[[str], Any] = webbrowser.open,
        version: str = __version__,
        profiler: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._backend = backend
        self._backend_factory = backend_factory
        self._backend_ready = threading.Event()
        self._backend_error: Optional[str] = None
        if backend is not None:
            self._backend_ready.set()
        self._emit = emit
        self._updater = updater
        self._state_path = state_path or (state_dir() / "desktop-ui.json")
        self._state_lock = threading.Lock()
        self._open_url = open_url
        self._version = version
        self._profiler = profiler
        self._window = None
        self._maximized = True  # the window opens maximized
        self._minimized = False
        self._release = None
        self._update_busy = False
        self._js_ready = threading.Event()
        self._closed = threading.Event()
        self._pump: Optional[threading.Thread] = None

    # ----- lifecycle (Python side) ------------------------------------------------
    def _start_backend(self) -> None:
        """Build the backend on a thread (hardware profiling takes a moment)."""
        if self._backend_ready.is_set() or self._backend_factory is None:
            return

        def build() -> None:
            try:
                self._backend = self._backend_factory()
            except Exception as e:  # surfaced to the UI by _be()
                self._backend_error = f"{type(e).__name__}: {e}"
                log.exception("desktop backend failed to start")
            finally:
                self._backend_ready.set()

        threading.Thread(target=build, daemon=True, name="eigencore-backend").start()

    def _be(self):
        self._start_backend()
        self._backend_ready.wait()
        if self._backend is None:
            raise RuntimeError(f"EigenCore backend failed to start: {self._backend_error}")
        return self._backend

    def _attach(self, window) -> None:
        self._window = window
        if self._emit is None:
            self._emit = lambda payload: window.evaluate_js(dispatch_script(payload))
        try:
            window.events.maximized += lambda *a: self._on_maximized()
            window.events.restored += lambda *a: self._on_restored()
            window.events.closed += lambda *a: self._closed.set()
            window.events.shown += lambda *a: self._on_shown()
            window.events.minimized += lambda *a: setattr(self, "_minimized", True)
        except Exception:  # older pywebview without these events
            pass

    def _on_shown(self) -> None:
        if self._window is not None:
            hwnd = window_hwnd(self._window)
            if hwnd:
                enable_native_minimize(hwnd)

    def _on_maximized(self) -> None:
        self._maximized = True
        self._minimized = False

    def _on_restored(self) -> None:
        # "restored" fires both when a minimized window comes back and when a
        # maximized one is restored. Coming back from the taskbar keeps the
        # maximized state it had (0.4.10 lost it, so the next Maximize click
        # did nothing).
        if self._minimized:
            self._minimized = False
            self._notify_window("restored")
        else:
            self._maximized = False

    def _notify_window(self, what: str) -> None:
        """Tell the page the window was restored, so it can fade back in."""
        if self._window is None or not self._js_ready.is_set():
            return
        with contextlib.suppress(Exception):
            self._window.evaluate_js(
                f"document.dispatchEvent(new CustomEvent('eigencore:window', {{detail: {json.dumps(what)}}}))"
            )

    def _put(self, *event) -> None:
        self._be().events.put(tuple(event))

    def _pump_loop(self) -> None:
        """Forward backend events to JS in small batches (tokens arrive fast)."""
        events: queue.Queue = self._be().events
        while not self._closed.is_set():
            try:
                first = events.get(timeout=0.25)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + 0.02
            while len(batch) < 256 and time.monotonic() < deadline:
                try:
                    batch.append(events.get_nowait())
                except queue.Empty:
                    time.sleep(0.004)
            try:
                if self._emit is not None:
                    self._emit(event_json(batch))
            except Exception:
                if self._closed.is_set():
                    return
                log.exception("could not deliver %d event(s) to the UI", len(batch))

    # ----- called from JavaScript -------------------------------------------------
    def ready(self) -> dict:
        """The UI is listening: start forwarding events. Returns host info."""
        self._be()
        if self._pump is None:
            self._pump = threading.Thread(
                target=self._pump_loop, daemon=True, name="eigencore-events"
            )
            self._pump.start()
        self._js_ready.set()
        return {"version": self._version, "platform": sys.platform}

    def version(self) -> str:
        return self._version

    def list_models(self) -> list:
        be = self._be()
        specs = {m.name: m for m in be.registry.list_models()}
        return [model_row_dict(r, specs.get(r.name)) for r in be.model_rows()]

    def hardware_profile(self, rescan: bool = False) -> dict:
        be = self._be()
        if rescan:
            from eigencore.hal.profiler import profile_hardware

            be.profile = (self._profiler or profile_hardware)()
        return profile_dict(be.profile)

    def get_settings(self) -> dict:
        be = self._be()
        state = self._read_state()
        theme = state.get("theme")
        return {
            "theme": theme if theme in ("dark", "light", "system") else "dark",
            "models_dir": str(be.registry.cache_dir),
            # The desktop backend does not take these yet: the UI shows them
            # read-only ("coming soon") and only theme is saved.
            "threads": "auto",
            # the n_ctx the loaded model opened with; profiler figure before load
            "context_length": getattr(be.engine, "n_ctx", None)
            or be.profile.recommended_context_length,
            "kv_cache_type": "f16",
            "approvals": "ask",
        }

    def save_settings(self, patch: Optional[dict] = None) -> dict:
        patch = patch or {}
        if patch.get("theme") in ("dark", "light", "system"):
            self._write_state({"theme": patch["theme"]})
        return self.get_settings()

    def load_chats(self) -> list:
        chats = self._read_state().get("chats")
        return chats if isinstance(chats, list) else []

    def load_ui_prefs(self) -> dict:
        """Small UI preferences (reply options), kept by the app, not the page."""
        prefs = self._read_state().get("prefs")
        return prefs if isinstance(prefs, dict) else {}

    def save_ui_prefs(self, patch: Optional[dict] = None) -> bool:
        if not isinstance(patch, dict):
            return False
        try:
            if len(json.dumps(patch)) > 16_384:
                return False
        except (TypeError, ValueError):
            return False
        prefs = self.load_ui_prefs()
        prefs.update({str(k): v for k, v in patch.items()})
        self._write_state({"prefs": prefs})
        return True

    def save_chats(self, chats: Optional[list] = None) -> bool:
        if not isinstance(chats, list):
            return False
        self._write_state({"chats": chats[:100]})
        return True

    def download(self, name: str) -> None:
        self._be().download(name)

    def delete(self, name: str) -> bool:
        return bool(self._be().delete(name))

    def switch_model(self, name: str) -> None:
        self._be().switch_model(name)

    def send(self, text: str, opts: Optional[dict] = None) -> None:
        opts = opts or {}
        kwargs: dict = {}
        if opts.get("max_tokens"):
            kwargs["max_tokens"] = int(opts["max_tokens"])
        if opts.get("temperature") is not None:
            kwargs["temperature"] = float(opts["temperature"])
        history = opts.get("history")
        if isinstance(history, list):
            kwargs["history"] = [
                {"role": str(m.get("role")), "content": str(m.get("content", ""))}
                for m in history
                if isinstance(m, dict)
            ]
        self._be().send(text, **kwargs)

    def stop(self) -> None:
        self._be().stop()

    def open_url(self, url: str) -> None:
        if isinstance(url, str) and url.startswith("https://"):
            self._open_url(url)

    # ----- updates ----------------------------------------------------------------
    def _update(self, state: dict) -> None:
        self._put("update", state)

    def check_for_update(self) -> None:
        if self._update_busy:
            return

        def run() -> None:
            self._update({"kind": "checking"})
            rel = self._updater.check()
            self._release = rel
            if rel is None:
                self._update({"kind": "error", "message": "Could not reach GitHub Releases"})
            elif self._updater.is_newer(rel.version, self._version):
                self._update({"kind": "available", "tag": f"v{rel.version}", "url": rel.page_url})
            else:
                self._update({"kind": "up_to_date", "current": self._version})

        threading.Thread(target=run, daemon=True, name="eigencore-update-check").start()

    def install_update(self) -> None:
        if self._update_busy:
            return
        self._update_busy = True
        threading.Thread(target=self._install_worker, daemon=True, name="eigencore-update").start()

    def _install_worker(self) -> None:
        up = self._updater
        try:
            rel = self._release or up.check()
            self._release = rel
            if rel is None:
                self._update({"kind": "error", "message": "Could not reach GitHub Releases"})
                return
            tag = f"v{rel.version}"
            if not up.is_newer(rel.version, self._version):
                self._update({"kind": "up_to_date", "current": self._version})
                return
            if not up.is_installed() or not rel.installer_url:
                # portable .exe or pip install: nothing to update in place
                self._open_url(rel.page_url)
                self._update(
                    {
                        "kind": "manual",
                        "tag": tag,
                        "url": rel.page_url,
                        "message": "This copy was not installed with EigenCore-Setup.exe, "
                        "so the download page was opened.",
                    }
                )
                return
            last = [-1]

            def progress(pct: float) -> None:
                if int(pct) != last[0]:
                    last[0] = int(pct)
                    self._update({"kind": "downloading", "tag": tag, "pct": float(pct)})

            self._update({"kind": "downloading", "tag": tag, "pct": 0.0})
            path = up.download(rel.installer_url, None, progress)
            self._update({"kind": "ready_to_install", "tag": tag})
            up.install(path)
            # The installer closes this app (/CLOSEAPPLICATIONS) and reopens the
            # new version; close ourselves too in case it cannot.
            time.sleep(1.5)
            self.window_control("close")
        except Exception as e:  # network or disk error: let the user retry
            log.exception("update failed")
            self._update({"kind": "error", "message": f"Update failed: {e}"[:160]})
        finally:
            self._update_busy = False

    # ----- window -----------------------------------------------------------------
    def window_control(self, op: str, *args) -> bool:
        w = self._window
        if w is None:
            return False
        if op == "minimize":
            self._minimized = True
            w.minimize()
        elif op == "toggle_maximize":
            self._minimized = False  # the page is visible, so it is not minimized
            if self._maximized:
                w.restore()
                self._maximized = False
                # The window is created at full-screen size (see run()), so a
                # restore alone would not shrink it: go back to the default size.
                with contextlib.suppress(Exception):
                    w.resize(DEFAULT_SIZE[0], DEFAULT_SIZE[1])
                    area = primary_screen_area()
                    if area is not None:
                        x, y, sw, sh = area
                        w.move(
                            x + max(0, (sw - DEFAULT_SIZE[0]) // 2),
                            y + max(0, (sh - DEFAULT_SIZE[1]) // 2),
                        )
            else:
                w.maximize()
                self._maximized = True
        elif op == "close":
            self._closed.set()
            log.info("window close requested by the UI")
            w.destroy()
        elif op == "resize" and len(args) >= 2:
            w.resize(max(860, int(args[0])), max(560, int(args[1])))
        else:
            return False
        return True

    def _shutdown(self) -> None:
        """The window is gone: stop the event pump and any running generation."""
        self._closed.set()
        if self._backend is not None:
            try:
                self._backend.stop()
            except Exception:
                log.debug("backend stop on close failed", exc_info=True)

    def is_maximized(self) -> bool:
        return self._maximized

    # ----- persisted UI state (theme, chats) ----------------------------------------
    def _read_state(self) -> dict:
        with self._state_lock:
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                return {}

    def _write_state(self, patch: dict) -> None:
        with self._state_lock:
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except (OSError, ValueError):
                data = {}
            data.update(patch)
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._state_path)


# ----- window -------------------------------------------------------------------


def _require_webview2() -> None:
    """On Windows, refuse to start unless pywebview will use WebView2.

    Without the WebView2 runtime pywebview silently falls back to the old IE
    engine (MSHTML), which cannot run this UI.
    """
    if sys.platform != "win32":
        return
    try:
        from webview.platforms import winforms
    except Exception as e:
        raise WebUIUnavailable(f"pywebview could not load its Windows backend: {e}") from e
    if not getattr(winforms, "is_chromium", False):
        raise WebUIUnavailable("Microsoft Edge WebView2 Runtime not found")


DEFAULT_SIZE = (1180, 760)  # the restored (not maximized) window size


def primary_screen_area() -> Optional[tuple[int, int, int, int]]:
    """(x, y, width, height) of the primary screen in logical pixels, or None."""
    try:
        import webview

        screens = list(webview.screens)
    except Exception:
        return None
    if not screens:
        return None
    primary = next((sc for sc in screens if sc.x == 0 and sc.y == 0), screens[0])
    try:
        return int(primary.x), int(primary.y), int(primary.width), int(primary.height)
    except (TypeError, ValueError, AttributeError):
        return None


# Win32 window styles a normal (framed) window has. A frameless WinForms
# window drops them, which is why Windows skipped the minimize / restore
# animation and a click on the taskbar button could not minimize it.
_WS_MAXIMIZEBOX = 0x00010000
_WS_MINIMIZEBOX = 0x00020000
_WS_SYSMENU = 0x00080000
_GWL_STYLE = -16
_SWP_FLAGS = (
    0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020
)  # NOSIZE|NOMOVE|NOZORDER|NOACTIVATE|FRAMECHANGED


def window_hwnd(window) -> Optional[int]:
    """The native handle of a pywebview window on Windows, or None."""
    native = getattr(window, "native", None)
    handle = getattr(native, "Handle", None)
    if handle is None:
        return None
    for conv in ("ToInt64", "ToInt32"):
        fn = getattr(handle, conv, None)
        if fn is not None:
            with contextlib.suppress(Exception):
                return int(fn())
    with contextlib.suppress(TypeError, ValueError):
        return int(handle)
    return None


def enable_native_minimize(hwnd: int) -> bool:
    """Give a frameless window the minimize/maximize/system-menu styles.

    Windows then animates minimize and restore like any other app, and the
    taskbar button minimizes the window on a second click. The window keeps
    no caption, so it still looks frameless. Windows only; False elsewhere
    or on failure.
    """
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        get = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
        put = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
        style = get(hwnd, _GWL_STYLE)
        want = style | _WS_MINIMIZEBOX | _WS_MAXIMIZEBOX | _WS_SYSMENU
        if want != style:
            put(hwnd, _GWL_STYLE, want)
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, _SWP_FLAGS)
        return True
    except Exception:
        log.debug("could not add native window styles", exc_info=True)
        return False


def startup_background(state_path: Optional[Path] = None) -> str:
    """The window colour before the page paints: the saved theme's background.

    A light-theme user used to see a dark frame flash on start and restore.
    """
    path = state_path or (state_dir() / "desktop-ui.json")
    try:
        theme = json.loads(path.read_text(encoding="utf-8")).get("theme")
    except (OSError, ValueError, AttributeError):
        theme = None
    return "#FFFFFF" if theme == "light" else "#0A0A0A"


def run(debug: bool = False) -> None:
    """Open the web UI window and block until it closes.

    Raises :class:`WebUIUnavailable` (before or after the window opens) when
    the web UI cannot be shown, so the launcher can fall back to Flet.
    """
    try:
        import webview
    except Exception as e:
        raise WebUIUnavailable(f"pywebview is not installed: {e}") from e

    dev_url = os.environ.get("EIGENCORE_UI_URL")  # e.g. http://localhost:5173 (npm run dev)
    if dev_url:
        url = dev_url
    else:
        dist = ui_dist_dir()
        if dist is None:
            raise WebUIUnavailable("built UI not found (run `npm run build` in ui/)")
        url = ui_start_url(dist)
    _require_webview2()

    from eigencore.desktop.backend import DesktopBackend

    bridge = WebBridge(backend_factory=DesktopBackend)
    bridge._start_backend()

    webview.settings["DRAG_REGION_SELECTOR"] = DRAG_SELECTOR
    webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True
    area = primary_screen_area()
    window = webview.create_window(
        f"EigenCore {__version__}",
        url,
        js_api=bridge,
        # Created at the full size of the primary screen and maximized, so it
        # never flashes at the 1180x760 default (0.4.9 opened offset and then
        # jumped to maximized on Windows).
        x=area[0] if area else None,
        y=area[1] if area else None,
        width=area[2] if area else DEFAULT_SIZE[0],
        height=area[3] if area else DEFAULT_SIZE[1],
        min_size=(860, 560),
        maximized=True,
        frameless=True,
        easy_drag=False,
        text_select=True,
        background_color=startup_background(),
    )
    bridge._attach(window)

    timed_out = threading.Event()

    def watchdog() -> None:
        if not bridge._js_ready.wait(LOAD_TIMEOUT_S) and not bridge._closed.is_set():
            timed_out.set()
            log.error("web UI did not load within %.0fs; closing it", LOAD_TIMEOUT_S)
            try:
                window.destroy()
            except Exception:
                pass

    threading.Thread(target=watchdog, daemon=True, name="eigencore-watchdog").start()

    # Pick the UI server port ourselves: pywebview would take its shared
    # default 42001 even when something else answers there (see instance.py).
    http_port = None
    if not dev_url:
        from eigencore.desktop.instance import pick_port

        http_port = pick_port()
        log.info("serving the UI from %s on http://127.0.0.1:%d", url, http_port)

    store = webview_dir()
    store.mkdir(parents=True, exist_ok=True)
    clear_stale_webview_cache(store)
    icon = Path(__file__).parent / "assets" / "eigencore.ico"
    webview.start(
        gui="edgechromium" if sys.platform == "win32" else None,
        debug=debug or bool(os.environ.get("EIGENCORE_UI_DEBUG")),
        http_server=not dev_url,
        http_port=http_port,
        private_mode=False,
        storage_path=str(store),
        user_agent=f"Mozilla/5.0 (EigenCore) {USER_AGENT_MARK}/{__version__}",
        icon=str(icon) if icon.exists() else None,
    )
    bridge._shutdown()
    log.info("web UI window closed")
    if timed_out.is_set():
        raise WebUIUnavailable(f"the web UI did not load within {LOAD_TIMEOUT_S:.0f}s")
