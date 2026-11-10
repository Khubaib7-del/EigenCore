"""Safe tool execution with validation and timeouts."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from eigencore.agent.policy import PolicyEngine, default_policy, redact_text, resolve_in_workspace
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
    ):
        self.registry = registry
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.policy = policy if policy is not None else default_policy()
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
        """Run the handler in a spawned process; on timeout, kill it — a
        thread-pool timeout cannot stop a running handler."""
        start = time.perf_counter()
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_run_handler, args=(tool.handler, args, queue))
        proc.start()
        proc.join(timeout=self.timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return ToolResult(
                tool_name=tool.name,
                status=ToolStatus.TIMEOUT,
                error=f"Tool execution timed out after {self.timeout}s (process killed)",
                elapsed_seconds=self.timeout,
            )
        elapsed = time.perf_counter() - start
        try:
            kind, payload = queue.get_nowait()
        except Exception:
            return ToolResult(
                tool_name=tool.name,
                status=ToolStatus.ERROR,
                error="handler process exited without returning a result",
                elapsed_seconds=elapsed,
            )
        if kind == "ok":
            output = payload
            if isinstance(payload, str):
                output = redact_text(payload)
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
            error=redact_text(payload),
            elapsed_seconds=elapsed,
        )


def _run_handler(handler, args, queue) -> None:
    try:
        queue.put(("ok", handler(**args)))
    except Exception as e:
        queue.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
