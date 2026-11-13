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

import json
import logging
import os
import queue
import re
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from enum import Flag
from pathlib import Path
from typing import Any, Callable, Optional

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
        self._maximized = False
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
            window.events.maximized += lambda *a: setattr(self, "_maximized", True)
            window.events.restored += lambda *a: setattr(self, "_maximized", False)
            window.events.closed += lambda *a: self._closed.set()
        except Exception:  # older pywebview without these events
            pass

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
            "context_length": be.profile.recommended_context_length,
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
            w.minimize()
        elif op == "toggle_maximize":
            if self._maximized:
                w.restore()
                self._maximized = False
            else:
                w.maximize()
                self._maximized = True
        elif op == "close":
            self._closed.set()
            w.destroy()
        elif op == "resize" and len(args) >= 2:
            w.resize(max(860, int(args[0])), max(560, int(args[1])))
        else:
            return False
        return True

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
        url = str(dist / "index.html")
    _require_webview2()

    from eigencore.desktop.backend import DesktopBackend

    bridge = WebBridge(backend_factory=DesktopBackend)
    bridge._start_backend()

    webview.settings["DRAG_REGION_SELECTOR"] = DRAG_SELECTOR
    webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True
    window = webview.create_window(
        f"EigenCore {__version__}",
        url,
        js_api=bridge,
        width=1180,
        height=760,
        min_size=(860, 560),
        frameless=True,
        easy_drag=False,
        text_select=True,
        background_color="#0E0F11",
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

    store = state_dir() / "webview"
    store.mkdir(parents=True, exist_ok=True)
    icon = Path(__file__).parent / "assets" / "eigencore.ico"
    webview.start(
        gui="edgechromium" if sys.platform == "win32" else None,
        debug=debug or bool(os.environ.get("EIGENCORE_UI_DEBUG")),
        http_server=not dev_url,
        private_mode=False,
        storage_path=str(store),
        user_agent=f"Mozilla/5.0 (EigenCore) {USER_AGENT_MARK}/{__version__}",
        icon=str(icon) if icon.exists() else None,
    )
    if timed_out.is_set():
        raise WebUIUnavailable(f"the web UI did not load within {LOAD_TIMEOUT_S:.0f}s")
