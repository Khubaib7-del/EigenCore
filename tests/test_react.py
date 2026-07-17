"""Tests for the ReAct agent loop."""

from eigencore.agent.react import AgentResult, AgentStep, ReActAgent, StepKind
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry


def _add(a: int, b: int) -> int:
    return a + b


def _search(query: str) -> str:
    return f"Results for '{query}': item1, item2, item3"


def _make_registry():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="add",
            description="Add two integers",
            handler=_add,
            parameters=[
                ToolParam(name="a", type="integer", description="first"),
                ToolParam(name="b", type="integer", description="second"),
            ],
        )
    )
    reg.register(
        Tool(
            name="search",
            description="Search for information",
            handler=_search,
            parameters=[
                ToolParam(name="query", type="string", description="search query"),
            ],
        )
    )
    return reg


class TestReActAgent:
    def test_system_prompt_includes_tools(self):
        reg = _make_registry()
        agent = ReActAgent(registry=reg)
        prompt = agent.system_prompt
        assert "add" in prompt
        assert "search" in prompt
        assert "Add two integers" in prompt

    def test_parse_thought(self):
        agent = ReActAgent(registry=_make_registry())
        step = agent.parse_step('{"type": "thought", "content": "I need to add"}')
        assert step.kind == StepKind.THOUGHT
        assert step.content == "I need to add"

    def test_parse_action(self):
        agent = ReActAgent(registry=_make_registry())
        step = agent.parse_step('{"type": "action", "tool": "add", "args": {"a": 3, "b": 4}}')
        assert step.kind == StepKind.ACTION
        assert step.tool_name == "add"
        assert step.tool_args == {"a": 3, "b": 4}

    def test_parse_final(self):
        agent = ReActAgent(registry=_make_registry())
        step = agent.parse_step('{"type": "final", "content": "The answer is 7"}')
        assert step.kind == StepKind.FINAL
        assert step.content == "The answer is 7"

    def test_parse_invalid_json(self):
        agent = ReActAgent(registry=_make_registry())
        step = agent.parse_step("this is not json")
        assert step.kind == StepKind.THOUGHT
        assert step.content == "this is not json"

    def test_parse_unknown_type(self):
        agent = ReActAgent(registry=_make_registry())
        step = agent.parse_step('{"type": "unknown", "content": "hmm"}')
        assert step.kind == StepKind.THOUGHT

    def test_execute_step_action(self):
        reg = _make_registry()
        agent = ReActAgent(registry=reg)
        step = AgentStep(
            kind=StepKind.ACTION,
            content="",
            tool_name="add",
            tool_args={"a": 5, "b": 3},
        )
        obs = agent.execute_step(step)
        assert obs is not None
        assert obs.kind == StepKind.OBSERVATION
        assert "8" in obs.content
        assert step.tool_result is not None
        assert step.tool_result.ok

    def test_execute_step_thought_returns_none(self):
        agent = ReActAgent(registry=_make_registry())
        step = AgentStep(kind=StepKind.THOUGHT, content="thinking...")
        assert agent.execute_step(step) is None

    def test_build_messages_initial(self):
        agent = ReActAgent(registry=_make_registry())
        messages = agent.build_messages("What is 2+2?", [])
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"

    def test_build_messages_with_history(self):
        agent = ReActAgent(registry=_make_registry())
        history = [
            AgentStep(kind=StepKind.THOUGHT, content="I should add"),
            AgentStep(
                kind=StepKind.ACTION,
                content="",
                tool_name="add",
                tool_args={"a": 2, "b": 2},
            ),
            AgentStep(kind=StepKind.OBSERVATION, content="[add] Result: 4"),
        ]
        messages = agent.build_messages("What is 2+2?", history)
        assert len(messages) == 5
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
        assert "Observation:" in messages[4]["content"]

    def test_run_with_immediate_final(self):
        agent = ReActAgent(registry=_make_registry())

        def mock_generate(messages, max_tokens):
            return '{"type": "final", "content": "42"}'

        result = agent.run("What is the meaning of life?", generate_fn=mock_generate)
        assert result.answer == "42"
        assert result.num_thoughts == 0
        assert result.num_actions == 0
        assert len(result.steps) == 1

    def test_run_think_then_act_then_final(self):
        agent = ReActAgent(registry=_make_registry())
        call_count = 0

        def mock_generate(messages, max_tokens):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '{"type": "thought", "content": "I need to add 3 and 4"}'
            elif call_count == 2:
                return '{"type": "action", "tool": "add", "args": {"a": 3, "b": 4}}'
            else:
                return '{"type": "final", "content": "3 + 4 = 7"}'

        result = agent.run("What is 3+4?", generate_fn=mock_generate)
        assert result.answer == "3 + 4 = 7"
        assert result.num_thoughts == 1
        assert result.num_actions == 1
        assert len(result.tool_calls) == 1

    def test_run_respects_max_steps(self):
        agent = ReActAgent(registry=_make_registry(), max_steps=3)
        call_count = 0

        def mock_generate(messages, max_tokens):
            nonlocal call_count
            call_count += 1
            return '{"type": "thought", "content": "still thinking..."}'

        result = agent.run("infinite thinker", generate_fn=mock_generate)
        assert call_count == 3
        assert result.answer == "still thinking..."

    def test_run_without_generate_fn(self):
        agent = ReActAgent(registry=_make_registry())
        result = agent.run("test")
        assert result.answer == ""
        assert result.steps == []

    def test_run_search_tool(self):
        agent = ReActAgent(registry=_make_registry())
        call_count = 0

        def mock_generate(messages, max_tokens):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '{"type": "action", "tool": "search", "args": {"query": "python"}}'
            else:
                return '{"type": "final", "content": "Found results about python"}'

        result = agent.run("Search for python", generate_fn=mock_generate)
        assert result.answer == "Found results about python"
        assert result.num_actions == 1
        obs_steps = [s for s in result.steps if s.kind == StepKind.OBSERVATION]
        assert len(obs_steps) == 1
        assert "python" in obs_steps[0].content


class TestAgentResult:
    def test_format_trace(self):
        result = AgentResult(
            answer="done",
            steps=[
                AgentStep(kind=StepKind.THOUGHT, content="Let me think"),
                AgentStep(
                    kind=StepKind.ACTION,
                    content="",
                    tool_name="add",
                    tool_args={"a": 1, "b": 2},
                ),
                AgentStep(kind=StepKind.OBSERVATION, content="[add] Result: 3"),
                AgentStep(kind=StepKind.FINAL, content="The answer is 3"),
            ],
        )
        trace = result.format_trace()
        assert "THOUGHT" in trace
        assert "ACTION" in trace
        assert "OBSERVATION" in trace
        assert "FINAL" in trace
        assert "add" in trace

    def test_empty_result(self):
        result = AgentResult(answer="")
        assert result.num_thoughts == 0
        assert result.num_actions == 0
        assert result.tool_calls == []
        assert result.total_tokens == 0
