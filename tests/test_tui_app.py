"""Tests for the full-screen TUI (tui/app.py, tui/keys.py, tui/banner.py).

Everything is driven headlessly: keys go straight into ChatApp.handle_key /
InputLine.handle_key, frames are rendered into a recording Console, and the
registry uses a tmp cache dir with fake downloaded files.
"""

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from eigencore.engine.inference import GenerationConfig
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelRegistry, ModelSpec
from eigencore.tui import keys as K
from eigencore.tui.app import ChatApp, InputLine, ModelPicker, pick_glyphs
from eigencore.tui.banner import BANNER_LINES, TAGLINE, banner_text, render_banner
from eigencore.tui.chat import ChatUiConfig


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


def _spec() -> ModelSpec:
    return ModelSpec(
        name="test-model",
        repo_id="test/test",
        filename="test.gguf",
        params_b=7.0,
        quant="Q4_K_M",
        size_gb=3.9,
        task="general",
        context_length=8192,
        description="Test model",
    )


class FakeEngine:
    """Yields a fixed reply; can also yield forever until closed (for Ctrl-C)."""

    def __init__(self, reply=("Hello", " there", "!"), infinite=False):
        self._reply = reply
        self._infinite = infinite
        self.seen = []

    def stream_messages(self, messages, config):
        assert isinstance(config, GenerationConfig) and config.stream is True
        self.seen.append(list(messages))
        if self._infinite:
            while True:
                yield "tok"
        yield from self._reply


def _registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(cache_dir=tmp_path / "models")


def _mark_downloaded(registry: ModelRegistry, spec: ModelSpec) -> None:
    path = spec.local_path(registry.cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-gguf")


def _make_app(tmp_path, engine=None, on_model_change=None, spec=None):
    registry = _registry(tmp_path)
    spec = spec or _spec()
    _mark_downloaded(registry, spec)
    engine = engine or FakeEngine()
    app = ChatApp(
        engine_getter=lambda: engine,
        spec=spec,
        profile=_profile(),
        registry=registry,
        cfg=ChatUiConfig(),
        on_model_change=on_model_change,
        console=Console(width=100, height=30, file=io.StringIO()),
    )
    return app, registry, engine


def _drain(app: ChatApp, timeout: float = 5.0) -> None:
    """tick() until generation/download finishes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.tick()
        if not app._gen_active and app._download is None:
            return
        time.sleep(0.01)
    raise AssertionError("worker did not finish in time")


def _frame_text(app: ChatApp, width=100, height=30) -> str:
    con = Console(width=width, height=height, record=True, file=io.StringIO())
    app.console = con
    con.print(app._frame())
    return con.export_text()


# ----- keys -------------------------------------------------------------------


def test_parse_keys_specials():
    assert K.parse_keys("a\x1b[A\r\x7f\t\x03\x04") == [
        "a",
        K.UP,
        K.ENTER,
        K.BACKSPACE,
        K.TAB,
        K.CTRL_C,
        K.CTRL_D,
    ]
    assert K.parse_keys("\x1b") == [K.ESC]
    assert K.parse_keys("\x1b[3~") == [K.DELETE]
    assert K.parse_keys("\x1b[D\x1b[H\x1b[F") == [K.LEFT, K.HOME, K.END]


# ----- banner -----------------------------------------------------------------


def test_banner_text_contains_art_and_version():
    text = banner_text()
    assert len(BANNER_LINES) >= 5
    for line in BANNER_LINES:
        assert line.rstrip() in text
    assert "local-first LLM runtime" in TAGLINE
    con = Console(record=True, file=io.StringIO(), width=100)
    render_banner(con, animate=False)
    out = con.export_text()
    assert "_____" in out and "local-first LLM runtime" in out


# ----- input line -------------------------------------------------------------


def test_input_line_editing_and_submit():
    line = InputLine()
    for ch in "hello":
        assert line.handle_key(ch) == (None, None)
    assert line.buf == "hello" and line.cursor == 5
    line.handle_key(K.LEFT)
    line.handle_key(K.LEFT)
    line.handle_key(K.BACKSPACE)
    assert line.buf == "helo"  # cursor was at 3, removes index 2
    line.handle_key(K.HOME)
    line.handle_key("X")
    assert line.buf == "Xhelo" and line.cursor == 1
    line.handle_key(K.END)
    action, text = line.handle_key(K.ENTER)
    assert action == "submit" and text == "Xhelo"
    assert line.buf == "" and line.history == ["Xhelo"]


def test_input_line_history_and_completion():
    line = InputLine()
    for ch in "/he":
        line.handle_key(ch)
    line.handle_key(K.TAB)
    assert line.buf.startswith("/help"), line.buf
    line.handle_key(K.CTRL_C)
    assert line.buf == ""
    for ch in "first":
        line.handle_key(ch)
    line.handle_key(K.ENTER)
    line.handle_key(K.UP)
    assert line.buf == "first"
    line.handle_key(K.DOWN)
    assert line.buf == ""


# ----- app: generation ----------------------------------------------------------


def test_app_streams_reply_and_records_measured_tps(tmp_path):
    app, _, engine = _make_app(tmp_path)
    app.submit_line("hi there")
    assert app._gen_active
    _drain(app)
    assert app.messages[0] == {"role": "user", "content": "hi there"}
    assert app.messages[1]["role"] == "assistant"
    assert app.messages[1]["content"] == "Hello there!"
    meta = app.blocks[-1]["meta"]
    assert "tok/s" in meta and "measured this turn" in meta
    assert engine.seen[0][-1]["content"] == "hi there"
    assert app._last_tps and app._last_tps > 0


def test_app_ctrl_c_interrupts_infinite_stream(tmp_path):
    app, _, _ = _make_app(tmp_path, engine=FakeEngine(infinite=True))
    app.submit_line("go")
    app.tick()
    app.handle_key(K.CTRL_C)  # must stop the infinite generator
    _drain(app)
    assert not app._gen_active
    assert "interrupted" in app.blocks[-1]["meta"]
    # history kept: user message + partial assistant reply
    assert app.messages[0]["role"] == "user"
    assert app.messages[-1]["role"] == "assistant"


def test_app_model_text_is_sanitized(tmp_path):
    engine = FakeEngine(reply=("bad [bold]x[/] \x1b]0;pwn\x07 \x1b[2J",))
    app, _, _ = _make_app(tmp_path, engine=engine)
    app.submit_line("hi")
    _drain(app)
    text = app.blocks[-1]["text"]
    assert "\x1b" not in text and "[bold]" in text  # markup literal, escapes stripped


# ----- app: commands -------------------------------------------------------------


def test_app_commands_help_context_clear_unknown_exit(tmp_path):
    app, _, _ = _make_app(tmp_path)
    app.handle_command("/help")
    assert app.overlay is not None
    app.handle_key(K.ESC)
    assert app.overlay is None

    app.handle_command("/context")
    assert app.overlay is not None
    app.handle_key(K.ESC)

    app.submit_line("hello")
    _drain(app)
    assert len(app.messages) == 2
    app.handle_command("/clear")
    assert app.messages == []

    app.handle_command("/tokns")  # typo → suggestion
    assert "did you mean /tokens" in app.status[1]

    app.handle_command("/tokens 128")
    assert app.cfg.max_tokens == 128
    app.handle_command("/temp 0.2")
    assert app.cfg.temperature == 0.2
    app.handle_command("/tokens nope")
    assert app.status[0] == "err"

    app.handle_command("/exit")
    assert app.running is False


# ----- app: model picker / download / delete --------------------------------------


def test_picker_switch_download_delete(tmp_path, monkeypatch):
    app, registry, _ = _make_app(tmp_path)
    other = next(m for m in registry.list_models() if m.name != app.spec.name)
    _mark_downloaded(registry, other)

    switched = []
    app.on_model_change = lambda s: switched.append(s.name)

    app.handle_command("/models")
    picker = app.overlay
    assert isinstance(picker, ModelPicker)
    target = next(i for i, m in enumerate(picker.items) if m.name == other.name)
    while picker.idx < target:
        picker.handle_key(K.DOWN, app)
    while picker.idx > target:
        picker.handle_key(K.UP, app)
    picker.handle_key(K.ENTER, app)
    assert app.overlay is None
    assert switched == [other.name]
    assert app.spec.name == other.name

    # download a non-downloaded model (fake the network call)
    target_spec = next(m for m in registry.list_models() if not registry.is_downloaded(m))
    monkeypatch.setattr(
        registry,
        "download",
        lambda spec, progress=True: _mark_downloaded(registry, spec)
        or spec.local_path(registry.cache_dir),
    )
    app.request_download(target_spec)
    _drain(app)
    assert registry.is_downloaded(target_spec)
    assert app.status[0] == "ok"

    # delete it via confirm dialog (y)
    app.confirm_delete(target_spec)
    assert app.overlay is not None
    app.handle_key("y")
    assert app.overlay is None
    assert not registry.is_downloaded(target_spec)
    assert "deleted" in app.status[1]

    # deleting the active model is refused
    app.confirm_delete(app.spec)
    assert app.overlay is None
    assert "active model" in app.status[1]


def test_picker_delete_key_asks_confirmation(tmp_path):
    app, registry, _ = _make_app(tmp_path)
    other = next(m for m in registry.list_models() if m.name != app.spec.name)
    _mark_downloaded(registry, other)
    app.handle_command("/models")
    picker = app.overlay
    target = next(i for i, m in enumerate(picker.items) if m.name == other.name)
    picker.idx = target
    picker.handle_key("x", app)
    assert app.overlay is not None  # confirm dialog
    assert "Delete" in app.overlay.title
    app.handle_key("n")  # default deny path
    assert registry.is_downloaded(other)


# ----- app: rendering ---------------------------------------------------------------


def test_frame_renders_header_transcript_input(tmp_path):
    app, _, _ = _make_app(tmp_path)
    app.submit_line("hello world")
    _drain(app)
    text = _frame_text(app)
    g = app.glyphs
    assert "eigencore" in text
    assert app.spec.name in text
    assert "ctx" in text
    assert "hello world" in text
    assert "Hello there!" in text
    assert "tok/s" in text
    assert g["prompt"] in text  # input prompt glyph visible


def test_frame_renders_picker_overlay(tmp_path):
    app, _, _ = _make_app(tmp_path)
    app.handle_command("/models")
    text = _frame_text(app)
    assert "Models" in text
    assert app.spec.name in text
    assert "downloaded" in text


def test_glyphs_fallback():
    g = pick_glyphs()
    assert g["prompt"] and g["brand"]
