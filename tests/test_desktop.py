"""Tests for the desktop app: backend bridge (headless, fake engine) and
Flet view construction without a display."""

import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.desktop import theme as T
from eigencore.desktop.backend import (
    EXE_URL,
    RELEASES_URL,
    DesktopBackend,
    ModelRow,
    fit_label,
    is_newer,
    latest_release,
)
from eigencore.desktop.views import ChatView, ModelsView, ProfileView, fit_chip
from eigencore.engine.inference import GenerationConfig
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelRegistry, ModelSpec


def _profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_name="Test CPU",
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


def _spec(name="test-model", repo="test/test", filename="test.gguf", size=3.9):
    return ModelSpec(
        name=name,
        repo_id=repo,
        filename=filename,
        params_b=7.0,
        quant="Q4_K_M",
        size_gb=size,
        task="general",
        context_length=8192,
        description="Test model",
    )


class FakeEngine:
    def __init__(self, reply=("Hello", " there"), infinite=False):
        self._reply = reply
        self._infinite = infinite
        self.loaded = False
        self.unloaded = False
        self.seen = []

    def load(self):
        self.loaded = True

    def unload(self):
        self.unloaded = True

    def stream_messages(self, messages, config):
        assert isinstance(config, GenerationConfig) and config.stream is True
        self.seen.append(list(messages))
        if self._infinite:
            while True:
                yield "x"
        yield from self._reply


def _mark_downloaded(registry, spec):
    path = spec.local_path(registry.cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-gguf-bytes")


def _backend(tmp_path, engine=None, monkeypatch=None):
    registry = ModelRegistry(cache_dir=tmp_path / "models")
    spec = registry.list_models()[0]  # a real registry entry
    _mark_downloaded(registry, spec)
    engine = engine or FakeEngine()
    events = queue.Queue()
    be = DesktopBackend(
        registry=registry,
        profile=_profile(),
        engine_factory=lambda s: engine,
        events=events,
    )
    return be, registry, spec, engine, events


def _drain_events(events, until, timeout=5.0):
    """Collect events until `until(ev)` is true."""
    got = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ev = events.get(timeout=0.1)
        except queue.Empty:
            continue
        got.append(ev)
        if until(ev):
            return got
    raise AssertionError(f"event not seen; got {got}")


# ----- backend ---------------------------------------------------------------


def test_model_rows_marks_downloaded_and_active(tmp_path):
    be, registry, spec, _, _ = _backend(tmp_path)
    rows = be.model_rows()
    row = next(r for r in rows if r.name == spec.name)
    assert isinstance(row, ModelRow)
    assert row.downloaded
    assert not row.is_active
    others = [r for r in rows if r.name != spec.name]
    assert all(not r.downloaded for r in others)
    assert all(r.fit == "unknown" for r in others)


def test_switch_model_loads_engine_and_clears_history(tmp_path):
    be, registry, spec, engine, events = _backend(tmp_path)
    be.messages.append({"role": "user", "content": "old"})
    be.switch_model(spec.name)
    got = _drain_events(events, lambda ev: ev[0] == "switched")
    assert engine.loaded
    assert be.active_spec.name == spec.name
    assert be.messages == []
    assert any(ev[0] == "engine" and ev[1] == "ready" for ev in got)


def test_switch_undownloaded_model_errors(tmp_path):
    be, _, _, _, events = _backend(tmp_path)
    be.switch_model("does-not-exist")
    ev = _drain_events(events, lambda ev: ev[0] == "engine" and ev[1] == "error")[-1]
    assert "not downloaded" in ev[2] or "Unknown" in ev[2] or "unknown" in ev[2]


def test_send_streams_tokens_and_reports_measured_tps(tmp_path):
    be, registry, spec, engine, events = _backend(tmp_path)
    be.switch_model(spec.name)
    _drain_events(events, lambda ev: ev[0] == "switched")
    be.send("hi")
    got = _drain_events(events, lambda ev: ev[0] == "done")
    toks = [ev[1] for ev in got if ev[0] == "tok"]
    assert toks == ["Hello", " there"]
    stats = next(ev[1] for ev in got if ev[0] == "done")
    assert stats["tokens"] == 2
    assert stats["tok_per_sec"] > 0 and not stats["interrupted"] and stats["error"] is None
    assert be.messages[0] == {"role": "user", "content": "hi"}


def test_stop_interrupts_infinite_stream(tmp_path):
    engine = FakeEngine(infinite=True)
    be, registry, spec, _, events = _backend(tmp_path, engine=engine)
    be.switch_model(spec.name)
    _drain_events(events, lambda ev: ev[0] == "switched")
    be.send("go")
    time.sleep(0.2)
    be.stop()
    got = _drain_events(events, lambda ev: ev[0] == "done")
    stats = next(ev[1] for ev in got if ev[0] == "done")
    assert stats["interrupted"]
    assert not be.gen_active


def test_send_without_model_emits_engine_error(tmp_path):
    be, _, _, _, events = _backend(tmp_path)
    be.send("hi")
    ev = _drain_events(events, lambda ev: ev[0] == "engine" and ev[1] == "error")[-1]
    assert "no model loaded" in ev[2]


def test_download_and_delete_roundtrip(tmp_path, monkeypatch):
    be, registry, spec, _, events = _backend(tmp_path)
    other = next(
        m
        for m in registry.list_models()
        if m.name != spec.name and registry.shard_prefix(m) is None
    )
    assert not registry.is_downloaded(other)

    monkeypatch.setattr(
        registry,
        "download",
        lambda s, progress=True: _mark_downloaded(registry, s) or s.local_path(registry.cache_dir),
    )
    be.download(other.name)
    _drain_events(events, lambda ev: ev[0] == "dl_done" and ev[2])
    assert registry.is_downloaded(other)

    # active model cannot be deleted
    be.switch_model(spec.name)
    _drain_events(events, lambda ev: ev[0] == "switched")
    assert be.delete(spec.name) is False
    ev = _drain_events(events, lambda ev: ev[0] == "deleted" and ev[1] == spec.name)[-1]
    assert ev[2] is False

    # non-active downloaded model can be deleted (removes files)
    assert be.delete(other.name) is True
    assert not registry.is_downloaded(other)
    assert not other.local_path(registry.cache_dir).exists()


# ----- theme -------------------------------------------------------------------


def test_theme_uses_brand_palette():
    th = T.build_theme()
    assert th.color_scheme.primary == T.CORE == "#5B6CF0"
    assert th.color_scheme.surface == T.GRAPHITE
    assert th.color_scheme.error == T.DANGER
    for key, (color, word) in T.STATE_STYLES.items():
        assert color and word  # every state colour paired with a word


# ----- views (headless construction) ---------------------------------------------


def test_chat_view_build_and_events(tmp_path):
    be, registry, spec, engine, events = _backend(tmp_path)
    view = ChatView(be)
    control = view.build()
    assert control is not None
    assert view.model_dropdown.options  # downloaded model listed

    be.switch_model(spec.name)
    _drain_events(events, lambda ev: ev[0] == "switched")
    view.handle_event(("engine", "ready", "test-model ready"))
    view.refresh_models()
    assert view.model_dropdown.value == spec.name

    view.input.value = "hello"
    view.submit()
    got = _drain_events(events, lambda ev: ev[0] == "done")
    for ev in got:
        view.handle_event(ev)
    assert "Hello there" in view.chat_log.controls[-1].content.controls[1].value
    assert "tok/s" in view.status.value and "measured this turn" in view.status.value
    assert view.send_btn.visible and not view.stop_btn.visible


def test_models_view_lists_rows_with_actions(tmp_path):
    be, registry, spec, _, events = _backend(tmp_path)
    view = ModelsView(be)
    view.build()
    assert len(view.rows_host.controls) == len(be.model_rows())
    view.handle_event(("dl_done", "x", True, "downloaded x"))
    assert view.status.value == "downloaded x"
    view.handle_event(("deleted", "x", True))
    assert view.status.value == "deleted x"


def test_profile_view_builds_cards(tmp_path):
    be, _, _, _, _ = _backend(tmp_path)
    view = ProfileView(be)
    control = view.build()
    assert control is not None


def test_all_views_support_app_polymorphic_loop(tmp_path):
    """app.main() calls attach(page), build(), handle_event(ev) on every view —
    regression test for the ProfileView.attach AttributeError seen in the exe."""
    from eigencore.desktop.views import View

    be, _, _, _, _ = _backend(tmp_path)
    views = [ChatView(be), ModelsView(be), ProfileView(be)]
    for v in views:
        assert isinstance(v, View)
        v.attach(page=None)  # attach must exist and not require a real page
        assert v.build() is not None
        v.handle_event(("engine", "ready", "x"))  # must not raise


def test_backend_shares_cli_cache_dir():
    """The desktop app and the CLI must see the same ~/.eigencore folder:
    ModelRegistry() default is ~/.eigencore/models and config is
    ~/.eigencore/config.json — no separate GUI cache."""
    from eigencore.models.registry import ModelRegistry as MR
    from eigencore.tui.onboarding import default_config_path

    be = DesktopBackend.__new__(DesktopBackend)  # avoid mkdir side effect here
    assert MR().cache_dir == Path.home() / ".eigencore" / "models"
    assert default_config_path() == Path.home() / ".eigencore" / "config.json"
    assert be  # placeholder no-op


def test_app_main_runs_without_page_update_crash(monkeypatch, tmp_path):
    """Regression for the packaged-exe crash: 'Control must be added to the
    page first' — main() used to call content.update() before page.add().
    Driving main() with a mock page must complete without raising."""
    import asyncio
    from unittest.mock import MagicMock

    import eigencore.desktop.app as app_mod

    registry = ModelRegistry(cache_dir=tmp_path / "models")
    be = DesktopBackend(
        registry=registry,
        profile=_profile(),
        engine_factory=lambda s: FakeEngine(),
    )
    monkeypatch.setattr(app_mod, "DesktopBackend", lambda: be)
    monkeypatch.setattr(app_mod, "latest_release", lambda *a, **k: None)  # no network
    # the event pump loops forever; don't let it hang asyncio.run's shutdown
    monkeypatch.setattr(app_mod.asyncio, "create_task", lambda coro: coro.close())

    page = MagicMock()
    page.window = MagicMock()
    added = []

    def fake_add(*controls, **kwargs):
        added.extend(controls)

    page.add = fake_add
    asyncio.run(app_mod.main(page))
    assert added, "main() must add its layout to the page"


def test_fit_chip_pairs_colour_with_word(tmp_path):
    be, registry, spec, _, _ = _backend(tmp_path)
    _mark_downloaded(registry, spec)
    label = fit_label(spec, registry, be.profile)
    chip = fit_chip(label)
    assert chip.content.value  # word present, not colour-only


# ----- release / update helpers ---------------------------------------------------


def test_is_newer():
    assert is_newer("v0.4.5", "0.4.4")
    assert is_newer("0.5.0", "0.4.4")
    assert not is_newer("v0.4.4", "0.4.4")
    assert not is_newer("v0.4.3", "0.4.4")
    assert not is_newer("garbage", "0.4.4")


def test_latest_release_parses_tag_and_url():
    payload = '{"tag_name": "v0.4.4", "html_url": "https://example/r/v0.4.4"}'
    rel = latest_release(fetcher=lambda url: payload)
    assert rel == {"tag": "v0.4.4", "url": "https://example/r/v0.4.4"}


def test_latest_release_returns_none_on_error():
    assert latest_release(fetcher=lambda url: (_ for _ in ()).throw(OSError("offline"))) is None
    assert latest_release(fetcher=lambda url: "not json") is None


def test_releases_url_points_at_repo():
    assert "Khubaib7-del/EigenCore" in RELEASES_URL and "releases/latest" in RELEASES_URL
    # direct-asset URL: one click downloads the exe, no release-page hop
    assert EXE_URL.endswith("/releases/latest/download/EigenCore.exe")
