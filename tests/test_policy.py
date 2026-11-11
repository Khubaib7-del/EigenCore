"""Tests for the P4 tool policy engine, path confinement, and audit log."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.agent.executor import ToolExecutor, ToolStatus
from eigencore.agent.policy import (
    AuditLog,
    PolicyEngine,
    ToolCapability,
    default_policy,
    redact_args,
    redact_text,
    resolve_in_workspace,
)
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry


def _read_file(path: str) -> str:
    return Path(path).read_text()


def _write_file(path: str, content: str) -> str:
    Path(path).write_text(content)
    return "ok"


def _run(cmd: str) -> str:
    return cmd


def _sleepy(path: str) -> str:
    import time as _t

    _t.sleep(30)
    return "done"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="read_file",
            description="read a file",
            handler=_read_file,
            parameters=[
                ToolParam(name="path", type="string", description="file path", is_path=True)
            ],
            capability=ToolCapability.READ,
        )
    )
    reg.register(
        Tool(
            name="write_file",
            description="write a file",
            handler=_write_file,
            parameters=[
                ToolParam(name="path", type="string", description="file path", is_path=True),
                ToolParam(name="content", type="string", description="content"),
            ],
            capability=ToolCapability.WRITE,
        )
    )
    reg.register(
        Tool(
            name="shell",
            description="run a command",
            handler=_run,
            parameters=[ToolParam(name="cmd", type="string", description="command")],
            capability=ToolCapability.EXECUTE,
        )
    )
    return reg


def test_deny_by_default():
    policy = PolicyEngine(allowed_capabilities=set())
    tool = _registry().get("read_file")
    decision = policy.check(tool, {"path": "x"})
    assert not decision.allowed and "not allow-listed" in decision.reason


def test_read_allowed_write_denied():
    with tempfile.TemporaryDirectory() as d:
        policy = default_policy(workspace_root=Path(d))
        reg = _registry()
        assert policy.check(reg.get("read_file"), {"path": "a.txt"}).allowed
        assert not policy.check(reg.get("write_file"), {"path": "a.txt", "content": "x"}).allowed
        assert not policy.check(reg.get("shell"), {"cmd": "ls"}).allowed


def test_executor_enforces_policy_before_running():
    with tempfile.TemporaryDirectory() as d:
        policy = default_policy(workspace_root=Path(d))
        executor = ToolExecutor(_registry(), policy=policy)
        result = executor.execute("shell", {"cmd": "rm -rf /"})
        assert result.status == ToolStatus.POLICY_DENIED
        assert "Denied by policy" in result.error
        # handler must not have run: nothing to check directly, but status proves it
        ok = executor.execute("read_file", {"path": "x"})  # allowed but file missing
        assert ok.status == ToolStatus.ERROR  # reached the handler, failed on missing file


def test_path_traversal_rejected():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        policy = PolicyEngine(allowed_capabilities={ToolCapability.READ}, workspace_root=root)
        tool = _registry().get("read_file")
        assert not policy.check(tool, {"path": "../../etc/passwd"}).allowed
        assert not policy.check(tool, {"path": "/etc/passwd"}).allowed
        assert policy.check(tool, {"path": "sub/ok.txt"}).allowed
        assert policy.check(tool, {"path": str(root / "abs-ok.txt")}).allowed


def test_symlink_escape_rejected():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside_dir:
        root = Path(d)
        outside = Path(outside_dir) / "secret.txt"
        outside.write_text("secret")
        (root / "link.txt").symlink_to(outside)
        policy = PolicyEngine(allowed_capabilities={ToolCapability.READ}, workspace_root=root)
        tool = _registry().get("read_file")
        assert not policy.check(tool, {"path": "link.txt"}).allowed


def test_resolve_in_workspace():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        assert resolve_in_workspace("a/b.txt", root) == root / "a" / "b.txt"
        assert resolve_in_workspace(str(root / "x"), root) == root / "x"
        for bad in ("../escape", "/etc/hostname", root.parent / "escape"):
            try:
                resolve_in_workspace(str(bad), root)
                raise AssertionError(f"should have rejected {bad}")
            except ValueError:
                pass


def test_audit_log_records_decisions_and_redacts():
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "audit.jsonl"
        audit = AuditLog(log_path)
        policy = PolicyEngine(
            allowed_capabilities={ToolCapability.READ}, workspace_root=Path(d), audit_log=audit
        )
        reg = _registry()
        policy.check(reg.get("read_file"), {"path": "a.txt"})
        policy.check(reg.get("shell"), {"cmd": "ls"})
        assert audit.denied_count() == 1
        lines = [json.loads(x) for x in log_path.read_text().splitlines()]
        assert len(lines) == 2
        assert lines[0]["decision"] == "allowed"
        assert lines[1]["decision"] == "denied"
        assert lines[1]["tool_name"] == "shell"


def test_redaction():
    assert redact_args({"api_key": "abc123", "path": "x"}) == {
        "api_key": "***REDACTED***",
        "path": "x",
    }
    out = redact_text("failed with token=abc123 and password: hunter2")
    assert "abc123" not in out and "hunter2" not in out
    assert "***REDACTED***" in out


def test_executor_records_outcome_in_audit():
    with tempfile.TemporaryDirectory() as d:
        audit = AuditLog(Path(d) / "audit.jsonl")
        policy = PolicyEngine(
            allowed_capabilities={ToolCapability.READ}, workspace_root=Path(d), audit_log=audit
        )
        executor = ToolExecutor(_registry(), policy=policy)
        executor.execute("read_file", {"path": "missing.txt"})
        entry = audit.entries[-1]
        assert entry.status == ToolStatus.ERROR.name
        assert entry.decision == "allowed"


def test_executor_denies_etc_passwd_read():
    import tempfile as _tf

    with _tf.TemporaryDirectory() as d:
        audit = AuditLog(Path(d) / "audit.jsonl")
        policy = PolicyEngine(
            allowed_capabilities={ToolCapability.READ},
            workspace_root=Path(d),
            audit_log=audit,
        )
        executor = ToolExecutor(_registry(), policy=policy)
        result = executor.execute("read_file", {"path": "/etc/passwd"})
        assert result.status == ToolStatus.POLICY_DENIED
        assert "outside workspace" in result.error


def test_timeout_kills_child_process():
    import tempfile as _tf

    with _tf.TemporaryDirectory() as d:
        policy = PolicyEngine(allowed_capabilities={ToolCapability.READ}, workspace_root=Path(d))
        reg = ToolRegistry()
        reg.register(
            Tool(
                name="sleepy",
                description="sleeps",
                handler=_sleepy,
                parameters=[ToolParam(name="path", type="string", description="p", is_path=True)],
                capability=ToolCapability.READ,
            )
        )
        executor = ToolExecutor(reg, timeout=0.5, policy=policy)
        result = executor.execute("sleepy", {"path": "x.txt"})
        assert result.status == ToolStatus.TIMEOUT
        assert "killed" in result.error
        # no child left running
        import psutil

        children = psutil.Process().children(recursive=True)
        assert all("sleepy" not in " ".join(c.cmdline()) for c in children)


def test_limits_fn_builds_on_posix():
    import os

    executor = ToolExecutor(_registry(), max_cpu_seconds=7, max_memory_mb=256)
    if os.name == "posix":
        fn = executor._limits_fn()
        assert callable(fn)
    else:
        assert executor._limits_fn is not None  # method exists; Popen skips preexec on non-posix


def test_executor_accepts_resource_limit_kwargs():
    executor = ToolExecutor(
        _registry(), max_cpu_seconds=10, max_memory_mb=512, max_file_size_mb=32, max_processes=64
    )
    assert executor.max_cpu_seconds == 10
    assert executor.max_memory_mb == 512
    assert executor.max_file_size_mb == 32
    assert executor.max_processes == 64
    # default: NPROC not applied (0), because it breaks child library imports
    assert ToolExecutor(_registry()).max_processes == 0
