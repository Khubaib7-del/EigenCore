"""Safe tool execution with validation and timeouts."""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from eigencore.agent.tool import Tool, ToolRegistry


class ToolStatus(Enum):
    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    VALIDATION_ERROR = auto()


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_name: str
    status: ToolStatus
    output: Any = None
    error: Optional[str] = None
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
    """Executes tools with validation, timeouts, and logging."""

    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = DEFAULT_TIMEOUT,
        max_output_chars: int = 4096,
    ):
        self.registry = registry
        self.timeout = timeout
        self.max_output_chars = max_output_chars
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

        result = self._run_with_timeout(tool, validated_args)
        self.log.add(result)
        return result

    def _run_with_timeout(self, tool: Tool, args: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(tool.handler, **args)
            try:
                output = future.result(timeout=self.timeout)
                elapsed = time.perf_counter() - start

                output_str = str(output)
                if len(output_str) > self.max_output_chars:
                    output_str = output_str[: self.max_output_chars] + "... [truncated]"
                    output = output_str

                return ToolResult(
                    tool_name=tool.name,
                    status=ToolStatus.SUCCESS,
                    output=output,
                    elapsed_seconds=elapsed,
                )
            except FuturesTimeout:
                future.cancel()
                return ToolResult(
                    tool_name=tool.name,
                    status=ToolStatus.TIMEOUT,
                    error=f"Tool execution timed out after {self.timeout}s",
                    elapsed_seconds=self.timeout,
                )
            except Exception as e:
                elapsed = time.perf_counter() - start
                return ToolResult(
                    tool_name=tool.name,
                    status=ToolStatus.ERROR,
                    error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                    elapsed_seconds=elapsed,
                )
