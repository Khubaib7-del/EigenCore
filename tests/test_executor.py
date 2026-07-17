"""Tests for tool execution with validation and timeouts."""

import time

from eigencore.agent.executor import ExecutionLog, ToolExecutor, ToolStatus
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry


def _add(a: int, b: int) -> int:
    return a + b


def _fail(**kwargs):
    raise RuntimeError("intentional failure")


def _slow(**kwargs):
    time.sleep(10)
    return "done"


def _long_output(**kwargs):
    return "x" * 10000


def _make_registry():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="add",
            description="Add two numbers",
            handler=_add,
            parameters=[
                ToolParam(name="a", type="integer", description="first"),
                ToolParam(name="b", type="integer", description="second"),
            ],
        )
    )
    reg.register(
        Tool(
            name="fail",
            description="Always fails",
            handler=_fail,
        )
    )
    reg.register(
        Tool(
            name="slow",
            description="Takes too long",
            handler=_slow,
        )
    )
    reg.register(
        Tool(
            name="verbose",
            description="Big output",
            handler=_long_output,
        )
    )
    return reg


class TestToolExecutor:
    def test_successful_execution(self):
        executor = ToolExecutor(_make_registry())
        result = executor.execute("add", {"a": 3, "b": 4})
        assert result.ok
        assert result.output == 7
        assert result.status == ToolStatus.SUCCESS
        assert result.elapsed_seconds > 0

    def test_unknown_tool(self):
        executor = ToolExecutor(_make_registry())
        result = executor.execute("nonexistent", {})
        assert not result.ok
        assert result.status == ToolStatus.ERROR
        assert "Unknown tool" in result.error

    def test_validation_error(self):
        executor = ToolExecutor(_make_registry())
        result = executor.execute("add", {"a": "not_int", "b": 1})
        assert not result.ok
        assert result.status == ToolStatus.VALIDATION_ERROR

    def test_missing_required_param(self):
        executor = ToolExecutor(_make_registry())
        result = executor.execute("add", {"a": 1})
        assert not result.ok
        assert result.status == ToolStatus.VALIDATION_ERROR
        assert "Missing required" in result.error

    def test_handler_exception(self):
        executor = ToolExecutor(_make_registry())
        result = executor.execute("fail", {})
        assert not result.ok
        assert result.status == ToolStatus.ERROR
        assert "intentional failure" in result.error

    def test_timeout(self):
        executor = ToolExecutor(_make_registry(), timeout=0.1)
        result = executor.execute("slow", {})
        assert not result.ok
        assert result.status == ToolStatus.TIMEOUT
        assert "timed out" in result.error

    def test_output_truncation(self):
        executor = ToolExecutor(_make_registry(), max_output_chars=100)
        result = executor.execute("verbose", {})
        assert result.ok
        assert len(str(result.output)) <= 120
        assert "truncated" in str(result.output)

    def test_format_for_prompt_success(self):
        executor = ToolExecutor(_make_registry())
        result = executor.execute("add", {"a": 1, "b": 2})
        text = result.format_for_prompt()
        assert "[add] Result:" in text
        assert "3" in text

    def test_format_for_prompt_error(self):
        executor = ToolExecutor(_make_registry())
        result = executor.execute("fail", {})
        text = result.format_for_prompt()
        assert "[fail] Error" in text


class TestExecutionLog:
    def test_log_tracking(self):
        executor = ToolExecutor(_make_registry())
        executor.execute("add", {"a": 1, "b": 2})
        executor.execute("add", {"a": 3, "b": 4})
        executor.execute("nonexistent", {})

        log = executor.log
        assert log.total_calls == 3
        assert log.success_count == 2
        assert log.error_count == 1
        assert log.total_time > 0

    def test_empty_log(self):
        log = ExecutionLog()
        assert log.total_calls == 0
        assert log.success_count == 0
        assert log.error_count == 0
        assert log.total_time == 0.0
