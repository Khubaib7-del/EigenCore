"""Stage D slice 2: approval manager + executor approval gate + TUI card."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.agent.executor import ToolExecutor, ToolStatus
from eigencore.agent.policy import ApprovalManager, ToolCapability
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec
from eigencore.tui.chat import ChatUI, ChatUiConfig


def _add(a: int, b: int) -> int:
    return a + b


def _registry():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="add",
            description="add",
            handler=_add,
            capability=ToolCapability.READ,
            parameters=[
                ToolParam(name="a", type="integer", description="a"),
                ToolParam(name="b", type="integer", description="b"),
            ],
        )
    )
    return reg


def test_approval_denied_without_callback():
    ex = ToolExecutor(_registry(), approver=ApprovalManager())
    r = ex.execute("add", {"a": 1, "b": 2})
    assert r.status == ToolStatus.POLICY_DENIED
    assert "no approval callback" in r.error


def test_approval_denied_by_user():
    ex = ToolExecutor(_registry(), approver=ApprovalManager(), ask_fn=lambda n, a: False)
    r = ex.execute("add", {"a": 1, "b": 2})
    assert r.status == ToolStatus.POLICY_DENIED
    assert "user approval" in r.error


def test_approval_granted_runs():
    ex = ToolExecutor(_registry(), approver=ApprovalManager(), ask_fn=lambda n, a: True)
    r = ex.execute("add", {"a": 2, "b": 3})
    assert r.ok and r.output == 5


def test_session_approval_skips_second_ask():
    calls = []

    def ask(n, a):
        calls.append(n)
        return True

    mgr = ApprovalManager()
    ex = ToolExecutor(_registry(), approver=mgr, ask_fn=ask)
    ex.execute("add", {"a": 1, "b": 1})
    mgr.record("add", True, session=True)
    ex.execute("add", {"a": 2, "b": 2})
    assert calls == ["add"]  # second call did not ask


def _profile():
    return HardwareProfile(
        cpu_name="t",
        arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=InstructionSet.AVX2,
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        estimated_bandwidth_gbps=25.0,
        max_model_params_b=7.0,
        optimal_quantization="Q4_K_M",
        estimated_tokens_per_sec=15.0,
        recommended_threads=4,
        recommended_context_length=4096,
    )


def _spec():
    return ModelSpec(
        name="t",
        repo_id="t/t",
        filename="t.gguf",
        params_b=1.1,
        quant="Q4_K_M",
        size_gb=0.67,
        task="general",
        context_length=2048,
        description="t",
    )


class _NoEngine:
    profile = _profile()
    model_spec = _spec()


def test_tui_approval_card_renders_and_collects():
    out = io.StringIO()
    inputs = iter(["y"])
    ui = ChatUI(
        _NoEngine(),
        _spec(),
        _profile(),
        cfg=ChatUiConfig(plain=True),
        out=out,
        inp=lambda p: next(inputs),
    )
    ok = ui.render_approval_card("write_file", {"path": "x.txt", "api_key": "secret123"})
    assert ok is True
    text = out.getvalue()
    assert "Tool request" in text and "write_file" in text
    assert "secret123" not in text  # redacted
    assert "***REDACTED***" in text


def test_tui_approval_card_denies_on_n_and_eof():
    out = io.StringIO()
    inputs = iter(["n"])
    ui = ChatUI(
        _NoEngine(),
        _spec(),
        _profile(),
        cfg=ChatUiConfig(plain=True),
        out=out,
        inp=lambda p: next(inputs),
    )
    assert ui.render_approval_card("shell", {"cmd": "ls"}) is False

    def eof(p):
        raise EOFError

    ui2 = ChatUI(
        _NoEngine(), _spec(), _profile(), cfg=ChatUiConfig(plain=True), out=io.StringIO(), inp=eof
    )
    assert ui2.render_approval_card("shell", {"cmd": "ls"}) is False
