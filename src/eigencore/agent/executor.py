"""Safe tool execution with validation and timeouts."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from eigencore.agent.policy import (
    ApprovalManager,
    PolicyEngine,
    default_policy,
    redact_text,
    resolve_in_workspace,
)
from eigencore.agent.tool import Tool, ToolRegistry


class ToolStatus(Enum):
    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    VALIDATION_ERROR = auto()
    POLICY_DENIED = auto()


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_name: str
    status: ToolStatus
    output: Any = None
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def format_for_prompt(self) -> str:
        if self.ok:
            return f"[{self.tool_name}] Result: {self.output}"
        return f"[{self.tool_name}] Error ({self.status.name}): {self.error}"


@dataclass
class ExecutionLog:
    """Tracks all tool executions in an agent run."""

    entries: list[ToolResult] = field(default_factory=list)

    def add(self, result: ToolResult) -> None:
        self.entries.append(result)

    @property
    def total_calls(self) -> int:
        return len(self.entries)

    @property
    def success_count(self) -> int:
        return sum(1 for e in self.entries if e.ok)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.entries if not e.ok)

    @property
    def total_time(self) -> float:
        return sum(e.elapsed_seconds for e in self.entries)


class ToolExecutor:
    """Executes tools with policy enforcement, validation, killable timeouts, logging.

    Deny-by-default: when no policy is passed, a `default_policy()` (READ-only,
    workspace-confined) is used — an executor never runs a tool just because a
    caller forgot to attach a policy.
    """

    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = DEFAULT_TIMEOUT,
        max_output_chars: int = 4096,
        policy: PolicyEngine | None = None,
        max_cpu_seconds: int = 60,
        max_memory_mb: int = 4096,
        max_file_size_mb: int = 64,
        max_processes: int = 0,  # 0 = do not set RLIMIT_NPROC (breaks library imports); >0 enables it
        approver: ApprovalManager | None = None,
        ask_fn=None,  # callable(tool_name, args) -> bool; required when approver needs a human
    ):
        self.registry = registry
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.policy = policy if policy is not None else default_policy()
        # POSIX resource limits applied to the child process (best-effort;
        # silently skipped on platforms without resource/preexec_fn).
        self.max_cpu_seconds = max_cpu_seconds
        self.max_memory_mb = max_memory_mb
        self.max_file_size_mb = max_file_size_mb
        self.max_processes = max_processes
        self.approver = approver
        self.ask_fn = ask_fn
        self.log = ExecutionLog()

    def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.registry.get(tool_name)
        if tool is None:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                error=f"Unknown tool: '{tool_name}'. Available: {self.registry.tool_names}",
            )
            self.log.add(result)
            return result

        # Policy enforcement happens BEFORE validation/running, in code —
        # never in prompt text and never in the UI layer.
        decision = self.policy.check(tool, args)
        if not decision.allowed:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.POLICY_DENIED,
                error=f"Denied by policy: {decision.reason}",
            )
            self.log.add(result)
            if self.policy.audit_log is not None:
                entry = self.policy.audit_log.entries[-1]
                entry.status = ToolStatus.POLICY_DENIED.name
            return result

        # Approval gate: after policy, before validation/running. Enforcement
        # is here in code; the UI only renders and collects the answer.
        if self.approver is not None and self.approver.needs_approval(tool_name):
            if self.ask_fn is None:
                result = ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.POLICY_DENIED,
                    error="Denied: approval required but no approval callback is attached",
                )
                self.log.add(result)
                return result
            approved = bool(self.ask_fn(tool_name, args))
            self.approver.record(tool_name, approved)
            if not approved:
                result = ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.POLICY_DENIED,
                    error="Denied by user approval",
                )
                self.log.add(result)
                return result

        try:
            validated_args = tool.validate_args(args)
        except (ValueError, TypeError) as e:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.VALIDATION_ERROR,
                error=str(e),
            )
            self.log.add(result)
            return result

        # Resolve declared path params to workspace-confined absolute paths
        # BEFORE the handler sees them (no raw outside-workspace paths).
        if self.policy.workspace_root is not None:
            for param in tool.parameters:
                if (
                    param.is_path
                    and param.name in validated_args
                    and isinstance(validated_args[param.name], str)
                ):
                    validated_args[param.name] = str(
                        resolve_in_workspace(validated_args[param.name], self.policy.workspace_root)
                    )

        result = self._run_with_timeout(tool, validated_args)
        self.log.add(result)
        if self.policy.audit_log is not None:
            entry = self.policy.audit_log.entries[-1]
            entry.status = result.status.name
            entry.elapsed_seconds = result.elapsed_seconds
        return result

    def _run_with_timeout(self, tool: Tool, args: dict[str, Any]) -> ToolResult:
        """Run the handler in a child `python` process; on timeout, kill it.

        A thread-pool timeout cannot stop a running handler; a child process
        can. Uses subprocess+pickle (not multiprocessing spawn) because spawn
        is unreliable under pytest on Windows. Handlers must be picklable
        (top-level functions) — lambdas/closures are rejected up front.
        """
        start = time.perf_counter()
        try:
            payload = pickle.dumps((tool.handler, args))
        except Exception as e:
            return ToolResult(
                tool_name=tool.name,
                status=ToolStatus.ERROR,
                error=f"handler/args not picklable for process isolation: {e}",
                elapsed_seconds=time.perf_counter() - start,
            )
        proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            preexec_fn=self._limits_fn() if os.name == "posix" else None,
        )
        try:
            out, _ = proc.communicate(input=payload, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return ToolResult(
                tool_name=tool.name,
                status=ToolStatus.TIMEOUT,
                error=f"Tool execution timed out after {self.timeout}s (process killed)",
                elapsed_seconds=self.timeout,
            )
        elapsed = time.perf_counter() - start
        if proc.returncode != 0 or not out:
            return ToolResult(
                tool_name=tool.name,
                status=ToolStatus.ERROR,
                error=f"handler process exited with code {proc.returncode}",
                elapsed_seconds=elapsed,
            )
        try:
            kind, value = pickle.loads(out)
        except Exception:
            return ToolResult(
                tool_name=tool.name,
                status=ToolStatus.ERROR,
                error="handler process returned an unreadable result",
                elapsed_seconds=elapsed,
            )
        if kind == "ok":
            output = value
            if isinstance(value, str):
                output = redact_text(value)
                if len(output) > self.max_output_chars:
                    output = output[: self.max_output_chars] + "... [truncated]"
            return ToolResult(
                tool_name=tool.name,
                status=ToolStatus.SUCCESS,
                output=output,
                elapsed_seconds=elapsed,
            )
        return ToolResult(
            tool_name=tool.name,
            status=ToolStatus.ERROR,
            error=redact_text(value),
            elapsed_seconds=elapsed,
        )

    def _limits_fn(self):
        """Build a preexec_fn applying rlimits to the child (POSIX only)."""
        import resource

        max_cpu = self.max_cpu_seconds
        max_mem = self.max_memory_mb * 1024 * 1024
        max_fsize = self.max_file_size_mb * 1024 * 1024
        max_nproc = self.max_processes

        def apply() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, max_cpu + 5))
            resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_fsize, max_fsize))
            if max_nproc > 0:
                resource.setrlimit(resource.RLIMIT_NPROC, (max_nproc, max_nproc))

        return apply


_WORKER_CODE = """
import pickle, sys, traceback
try:
    handler, args = pickle.loads(sys.stdin.buffer.read())
    try:
        result = handler(**args)
        sys.stdout.buffer.write(pickle.dumps(("ok", result)))
    except Exception as e:
        sys.stdout.buffer.write(pickle.dumps(("err", f"{type(e).__name__}: {e}\\n{traceback.format_exc()}")))
except Exception as e:
    sys.stdout.buffer.write(pickle.dumps(("err", f"unpickle failed: {e}")))
"""
