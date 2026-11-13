"""Tests for the web UI host: WebBridge (the pywebview js_api) and the launcher.

No GUI and no pywebview import: the bridge talks to a real DesktopBackend with
a fake engine (as in test_desktop.py) and to a fake updater, and events are
captured from its emit callback instead of a browser window.
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore import __version__
from eigencore.desktop import launcher, web
from eigencore.desktop.backend import DesktopBackend
from eigencore.desktop.updater import Release
from eigencore.desktop.web import (
    WebBridge,
    WebUIUnavailable,
    dispatch_script,
    event_json,
    profile_dict,
    provider_for,
    ui_dist_dir,
)
from eigencore.engine.inference import GenerationConfig
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelRegistry


def _profile(cpu="Test CPU") -> HardwareProfile:
    return HardwareProfile(
        cpu_name=cpu,
        arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=InstructionSet.AVX2 | InstructionSet.SSE2,
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        estimated_bandwidth_gbps=25.0,
        max_model_params_b=7.0,
        optimal_quantization="Q4_K_M",
        estimated_tokens_per_sec=15.0,
        recommended_threads=4,
        recommended_context_length=4096,
    )


class FakeEngine:
    def __init__(self, reply=("Hello", " there", "!")):
        self.reply = reply
        self.seen = []

    def load(self):
        pass

    def unload(self):
        pass

    def stream_messages(self, messages, config):
        assert isinstance(config, GenerationConfig)
        self.seen.append(list(messages))
        yield from self.reply


class Collector:
    """Stands in for window.evaluate_js: keeps every event the UI would get."""

    def __init__(self):
        self.events = []
        self.cond = threading.Condition()

    def __call__(self, payload: str) -> None:
        batch = json.loads(payload)
        with self.cond:
            self.events.extend(batch)
            self.cond.notify_all()

    def wait_for(self, pred, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                for e in self.events:
                    if pred(e):
                        return e
                left = deadline - time.monotonic()
                if left <= 0:
                    raise AssertionError(f"event not seen; got {self.events}")
                self.cond.wait(left)


class FakeUpdater:
    def __init__(self, release=None, installed=True, fail_download=False):
        self.release = release
        self.installed = installed
        self.fail_download = fail_download
        self.installed_paths = []

    def check(self):
        return self.release

    @staticmethod
    def is_newer(tag, current):
        from eigencore.desktop.updater import is_newer

        return is_newer(tag, current)

    def is_installed(self):
        return self.installed

    def download(self, url, dest=None, on_progress=None):
        if self.fail_download:
            raise OSError("disk full")
        for pct in (10.0, 55.0, 100.0):
            on_progress(pct)
        return Path("/tmp/EigenCore-Setup.exe")

    def install(self, path):
        self.installed_paths.append(path)


class FakeWindow:
    def __init__(self):
        self.calls = []
        self.events = SimpleNamespace(maximized=_Hook(), restored=_Hook(), closed=_Hook())

    def minimize(self):
        self.calls.append("minimize")

    def maximize(self):
        self.calls.append("maximize")

    def restore(self):
        self.calls.append("restore")

    def destroy(self):
        self.calls.append("destroy")

    def resize(self, w, h):
        self.calls.append(("resize", w, h))


class _Hook:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self


def _bridge(tmp_path, engine=None, updater=None, opened=None, downloaded=True):
    registry = ModelRegistry(cache_dir=tmp_path / "models")
    spec = registry.list_models()[0]
    if downloaded:
        p = spec.local_path(registry.cache_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-gguf")
    engine = engine or FakeEngine()
    be = DesktopBackend(
        registry=registry,
        profile=_profile(),
        engine_factory=lambda s: engine,
        events=queue.Queue(),
    )
    col = Collector()
    opened = opened if opened is not None else []
    br = WebBridge(
        backend=be,
        emit=col,
        updater=updater or FakeUpdater(),
        state_path=tmp_path / "ui.json",
        open_url=opened.append,
        version="0.4.6",
        profiler=lambda: _profile("Rescanned CPU"),
    )
    br.ready()
    return br, be, spec, engine, col


def _stop(br):
    br._closed.set()


# ----- serialisation ----------------------------------------------------------


def test_event_json_and_dispatch_script():
    payload = event_json([("tok", "hi"), ("done", {"tokens": 1, "error": None})])
    assert json.loads(payload) == [["tok", "hi"], ["done", {"tokens": 1, "error": None}]]
    js = dispatch_script(payload)
    assert js.startswith('window.dispatchEvent(new CustomEvent("eigencore",{detail:[')


def test_provider_mapping():
    assert provider_for("tinyllama-1.1b") == "tinyllama"
    assert provider_for("phi-3-mini", "bartowski/Phi-3-mini-4k-instruct-GGUF") == "microsoft"
    assert provider_for("qwen2.5-3b") == "qwen"
    assert provider_for("something", "org/unknown") == "huggingface"


def test_profile_dict_lists_instruction_sets():
    d = profile_dict(_profile())
    assert d["cpu_name"] == "Test CPU"
    assert set(d["instruction_sets"]) == {"AVX2", "SSE2"}
    assert d["benchmarked"] is False
    json.dumps(d)  # must be JSON-serialisable for pywebview


def test_bridge_exposes_only_methods():
    """pywebview walks public attributes; every piece of state must be private."""
    br = WebBridge(backend=SimpleNamespace(events=queue.Queue()), emit=lambda p: None)
    public = [n for n in vars(br) if not n.startswith("_")]
    assert public == []


# ----- models / profile / settings -----------------------------------------------


def test_list_models_matches_ui_shape(tmp_path):
    br, be, spec, *_, col = _bridge(tmp_path)
    try:
        rows = br.list_models()
        assert len(rows) == len(be.registry.list_models())
        first = next(r for r in rows if r["name"] == spec.name)
        for key in (
            "name",
            "params_b",
            "quant",
            "size_gb",
            "task",
            "context_length",
            "description",
            "downloaded",
            "fit",
            "is_active",
            "repo_id",
            "provider",
        ):
            assert key in first
        assert first["downloaded"] is True
        assert first["repo_id"] == spec.repo_id
        json.dumps(rows)
    finally:
        _stop(br)


def test_hardware_profile_and_rescan(tmp_path):
    br, *_ = _bridge(tmp_path)
    try:
        assert br.hardware_profile()["cpu_name"] == "Test CPU"
        assert br.hardware_profile(rescan=True)["cpu_name"] == "Rescanned CPU"
    finally:
        _stop(br)


def test_settings_persist_theme_only(tmp_path):
    br, be, *_ = _bridge(tmp_path)
    try:
        s = br.get_settings()
        assert s["theme"] == "dark"
        assert s["models_dir"] == str(be.registry.cache_dir)
        s = br.save_settings({"theme": "light", "threads": 2})
        assert s["theme"] == "light" and s["threads"] == "auto"
        assert json.loads((tmp_path / "ui.json").read_text())["theme"] == "light"
        assert br.save_settings({"theme": "neon"})["theme"] == "light"
    finally:
        _stop(br)


def test_chats_roundtrip(tmp_path):
    br, *_ = _bridge(tmp_path)
    try:
        assert br.load_chats() == []
        chats = [{"id": "a", "title": "Hi", "messages": []}]
        assert br.save_chats(chats) is True
        assert br.load_chats() == chats
        assert br.save_chats("nope") is False
        # theme survives a chats write
        br.save_settings({"theme": "system"})
        br.save_chats([])
        assert br.get_settings()["theme"] == "system"
    finally:
        _stop(br)


# ----- engine / chat ------------------------------------------------------------------


def test_switch_send_streams_tokens_and_stats(tmp_path):
    br, be, spec, engine, col = _bridge(tmp_path)
    try:
        br.switch_model(spec.name)
        col.wait_for(lambda e: e[0] == "switched" and e[1] == spec.name)
        br.send("hi", {"max_tokens": 16, "history": [{"role": "user", "content": "earlier"}]})
        done = col.wait_for(lambda e: e[0] == "done")
        kinds = [e[0] for e in col.events]
        assert kinds.index("prompt_eval") < kinds.index("tok") < kinds.index("done")
        toks = "".join(e[1] for e in col.events if e[0] == "tok")
        assert toks == "Hello there!"
        stats = done[1]
        assert stats["tokens"] == 3 and stats["error"] is None
        assert stats["first_token_s"] is not None and stats["first_token_s"] >= 0
        assert "decode_tok_per_sec" in stats
        # the UI's history replaced the backend conversation
        assert engine.seen[0] == [
            {"role": "user", "content": "earlier"},
            {"role": "user", "content": "hi"},
        ]
        # and the reply was kept for the next turn
        assert be.messages[-1] == {"role": "assistant", "content": "Hello there!"}
        assert next(r for r in br.list_models() if r["name"] == spec.name)["is_active"]
    finally:
        _stop(br)


def test_send_without_model_reports_engine_error(tmp_path):
    br, *_, col = _bridge(tmp_path)
    try:
        br.send("hi")
        e = col.wait_for(lambda e: e[0] == "engine")
        assert e[1] == "error"
    finally:
        _stop(br)


def test_delete_and_unknown_download(tmp_path):
    br, be, spec, *_, col = _bridge(tmp_path)
    try:
        br.download("no-such-model")
        e = col.wait_for(lambda e: e[0] == "dl_done")
        assert e[1] == "no-such-model" and e[2] is False
        assert br.delete(spec.name) is True
        col.wait_for(lambda e: e[0] == "deleted" and e[2] is True)
        assert not be.registry.is_downloaded(spec)
    finally:
        _stop(br)


def test_stop_is_safe_when_idle(tmp_path):
    br, *_ = _bridge(tmp_path)
    try:
        br.stop()
    finally:
        _stop(br)


def test_backend_factory_runs_lazily(tmp_path):
    made = []

    def factory():
        made.append(1)
        return SimpleNamespace(events=queue.Queue(), stop=lambda: None)

    br = WebBridge(backend_factory=factory, emit=lambda p: None, state_path=tmp_path / "s.json")
    br.stop()
    assert made == [1]


def test_backend_factory_failure_is_reported(tmp_path):
    def factory():
        raise RuntimeError("no psutil")

    br = WebBridge(backend_factory=factory, emit=lambda p: None)
    try:
        br.version()  # does not need the backend
        br.stop()
    except RuntimeError as e:
        assert "no psutil" in str(e)
    else:
        raise AssertionError("expected the backend error")


# ----- updates ------------------------------------------------------------------------


def _rel(version="0.4.7", installer=True):
    return Release(
        version=version,
        page_url="https://github.com/Khubaib7-del/EigenCore/releases/tag/v" + version,
        installer_url="https://example.invalid/EigenCore-Setup.exe" if installer else None,
        installer_size=100,
    )


def _updates(col):
    return [e[1] for e in col.events if e[0] == "update"]


def test_check_for_update_states(tmp_path):
    br, *_, col = _bridge(tmp_path, updater=FakeUpdater(_rel("0.4.7")))
    try:
        br.check_for_update()
        e = col.wait_for(lambda e: e[0] == "update" and e[1]["kind"] == "available")
        assert e[1]["tag"] == "v0.4.7"
        assert _updates(col)[0] == {"kind": "checking"}
    finally:
        _stop(br)

    br, *_, col = _bridge(tmp_path, updater=FakeUpdater(_rel("0.4.6")))
    try:
        br.check_for_update()
        e = col.wait_for(lambda e: e[0] == "update" and e[1]["kind"] == "up_to_date")
        assert e[1]["current"] == "0.4.6"
    finally:
        _stop(br)

    br, *_, col = _bridge(tmp_path, updater=FakeUpdater(None))
    try:
        br.check_for_update()
        col.wait_for(lambda e: e[0] == "update" and e[1]["kind"] == "error")
    finally:
        _stop(br)


def test_install_update_downloads_runs_installer_and_closes(tmp_path):
    up = FakeUpdater(_rel("0.4.7"))
    br, *_, col = _bridge(tmp_path, updater=up)
    win = FakeWindow()
    br._attach(win)
    br._emit = col  # _attach installs evaluate_js; keep the collector
    try:
        br.install_update()
        col.wait_for(lambda e: e[0] == "update" and e[1]["kind"] == "ready_to_install")
        deadline = time.monotonic() + 5
        while "destroy" not in win.calls and time.monotonic() < deadline:
            time.sleep(0.05)
        assert up.installed_paths == [Path("/tmp/EigenCore-Setup.exe")]
        assert "destroy" in win.calls
        pcts = [s["pct"] for s in _updates(col) if s["kind"] == "downloading"]
        assert pcts[0] == 0.0 and pcts[-1] == 100.0
    finally:
        _stop(br)


def test_install_update_portable_opens_page(tmp_path):
    opened = []
    up = FakeUpdater(_rel("0.4.7"), installed=False)
    br, *_, col = _bridge(tmp_path, updater=up, opened=opened)
    try:
        br.install_update()
        e = col.wait_for(lambda e: e[0] == "update" and e[1]["kind"] == "manual")
        assert opened == [e[1]["url"]]
        assert up.installed_paths == []
    finally:
        _stop(br)


def test_install_update_failure_is_reported(tmp_path):
    up = FakeUpdater(_rel("0.4.7"), fail_download=True)
    br, *_, col = _bridge(tmp_path, updater=up)
    try:
        br.install_update()
        e = col.wait_for(lambda e: e[0] == "update" and e[1]["kind"] == "error")
        assert "disk full" in e[1]["message"]
        assert br._update_busy is False
    finally:
        _stop(br)


def test_open_url_only_https(tmp_path):
    opened = []
    br = WebBridge(backend=SimpleNamespace(events=queue.Queue()), open_url=opened.append)
    br.open_url("file:///etc/passwd")
    br.open_url("https://github.com/Khubaib7-del/EigenCore")
    assert opened == ["https://github.com/Khubaib7-del/EigenCore"]


# ----- window ---------------------------------------------------------------------------


def test_window_controls():
    br = WebBridge(backend=SimpleNamespace(events=queue.Queue()), emit=lambda p: None)
    assert br.window_control("minimize") is False  # no window yet
    win = FakeWindow()
    br._attach(win)
    assert br.window_control("minimize")
    assert br.window_control("toggle_maximize")
    assert br.is_maximized()
    assert br.window_control("toggle_maximize")
    assert not br.is_maximized()
    assert br.window_control("resize", 100, 2000)
    assert not br.window_control("explode")
    assert br.window_control("close")
    assert win.calls == ["minimize", "maximize", "restore", ("resize", 860, 2000), "destroy"]
    # maximize/restore done with the OS (e.g. Win+Up) keeps the state in sync
    win.events.maximized.handlers[0]()
    assert br.is_maximized()


# ----- paths / run --------------------------------------------------------------------


def test_ui_dist_dir_env_and_meipass(tmp_path, monkeypatch):
    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text("<html></html>")
    monkeypatch.setenv("EIGENCORE_UI_DIST", str(d))
    assert ui_dist_dir() == d

    monkeypatch.delenv("EIGENCORE_UI_DIST")
    bundle = tmp_path / "bundle"
    (bundle / "ui").mkdir(parents=True)
    (bundle / "ui" / "index.html").write_text("<html></html>")
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    assert ui_dist_dir() == bundle / "ui"


def test_run_without_ui_bundle_raises(monkeypatch):
    monkeypatch.setattr(web, "ui_dist_dir", lambda: None)
    monkeypatch.delenv("EIGENCORE_UI_URL", raising=False)
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace())
    try:
        web.run()
    except WebUIUnavailable as e:
        assert "built UI not found" in str(e)
    else:
        raise AssertionError("expected WebUIUnavailable")


def test_run_without_pywebview_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", None)  # import fails
    try:
        web.run()
    except WebUIUnavailable as e:
        assert "pywebview" in str(e)
    else:
        raise AssertionError("expected WebUIUnavailable")


# ----- launcher -------------------------------------------------------------------------


def test_launcher_prefers_web(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "log_path", lambda: tmp_path / "desktop.log")
    ran = []
    which = launcher.main(
        [], run_web=lambda d: ran.append(("web", d)), run_flet=lambda: ran.append("flet")
    )
    assert which == "web" and ran == [("web", False)]


def test_launcher_flet_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "log_path", lambda: tmp_path / "desktop.log")
    ran = []
    which = launcher.main(
        ["--flet"], run_web=lambda d: ran.append("web"), run_flet=lambda: ran.append("flet")
    )
    assert which == "flet" and ran == ["flet"]


def test_launcher_falls_back_to_flet(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "log_path", lambda: tmp_path / "desktop.log")
    ran = []

    def broken(debug):
        raise WebUIUnavailable("Microsoft Edge WebView2 Runtime not found")

    which = launcher.main([], run_web=broken, run_flet=lambda: ran.append("flet"))
    assert which == "flet" and ran == ["flet"]


def test_launcher_ignores_unknown_args():
    args = launcher.parse_args(["--debug", "--multiprocessing-fork"])
    assert args.debug and not args.flet


def test_version_is_package_version(tmp_path):
    br = WebBridge(backend=SimpleNamespace(events=queue.Queue()))
    assert br.version() == __version__


def test_self_test_report(tmp_path, monkeypatch):
    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text("<html></html>")
    monkeypatch.setenv("EIGENCORE_UI_DIST", str(d))
    out = tmp_path / "report.json"
    which = launcher.main(["--self-test", str(out)])
    report = json.loads(out.read_text())
    assert report["version"] == __version__
    assert report["ui_dist"] == str(d)
    assert which == ("self-test-ok" if report["ok"] else "self-test-failed")
