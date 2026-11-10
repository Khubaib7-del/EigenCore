"""Tests for the Phase 5 terminal chat UI (fake engine, scripted input)."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.engine.inference import GenerationConfig
from eigencore.fit import ModelGeometry
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec
from eigencore.tui.chat import ChatUI, ChatUiConfig


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
    def __init__(self, reply=("Hello", " there", "!")):
        self.profile = _profile()
        self.model_spec = _spec()
        self._reply = reply
        self.seen_messages = []

    def stream_messages(self, messages, config):
        assert isinstance(config, GenerationConfig) and config.stream is True
        self.seen_messages.append(list(messages))
        yield from self._reply


def _make_ui(inputs, reply=("Hello", " there", "!")):
    out = io.StringIO()
    it = iter(inputs)

    def fake_input(prompt):
        return next(it)

    ui = ChatUI(
        FakeEngine(reply),
        _spec(),
        _profile(),
        geometry=GEO,
        policy_label="read-only",
        cfg=ChatUiConfig(plain=True),
        out=out,
        inp=fake_input,
    )
    return ui, out


def test_plain_chat_turn_and_exit():
    ui, out = _make_ui(["hi", "/exit"])
    ui.run()
    text = out.getvalue()
    assert "eigencore chat | test-model" in text  # header
    assert "fits with margin" in text  # fit verdict from P2 estimator
    assert "Hello there!" in text  # streamed reply
    assert "tok/s (measured this turn)" in text
    assert "session ended" in text
    # multi-turn: second user message should include first exchange in history
    assert len(ui.engine.seen_messages) == 1


def test_slash_commands():
    ui, out = _make_ui(["/help", "/context", "/clear", "/exot", "/exit"])
    ui.run()
    text = out.getvalue()
    assert "/model" in text and "/exit" in text  # help lists commands
    assert "estimated tokens in history" in text  # /context
    assert "history cleared" in text
    assert "unknown command: /exot" in text and "Did you mean /exit?" in text


def test_multi_turn_history_grows():
    ui, _out = _make_ui(["one", "two", "/exit"])
    ui.run()
    # second turn sees user+assistant of first turn + new user message
    second_call = ui.engine.seen_messages[1]
    roles = [m["role"] for m in second_call]
    assert roles == ["user", "assistant", "user"]


def test_header_shows_policy_label():
    ui, _ = _make_ui(
        ["/exit"],
    )
    assert "read-only" in ui.header()


def test_eof_ends_session():
    def raise_eof(prompt):
        raise EOFError

    out = io.StringIO()
    ui = ChatUI(
        FakeEngine(),
        _spec(),
        _profile(),
        geometry=GEO,
        cfg=ChatUiConfig(plain=True),
        out=out,
        inp=raise_eof,
    )
    ui.run()
    assert "session ended" in out.getvalue()


GEO = ModelGeometry(num_layers=28, hidden_size=3584, num_attention_heads=28, num_kv_heads=4)
