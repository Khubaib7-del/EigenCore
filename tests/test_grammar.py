"""Tests for GBNF grammar generation."""

import pytest

from eigencore.agent.grammar import GBNFBuilder
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry


def _noop(**kwargs):
    return kwargs


def _make_registry(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


class TestGBNFBuilder:
    def test_empty_registry_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="empty tool registry"):
            GBNFBuilder.tool_call_grammar(reg)

    def test_single_tool_no_params(self):
        reg = _make_registry(Tool(name="ping", description="ping", handler=_noop))
        grammar = GBNFBuilder.tool_call_grammar(reg)
        assert "root" in grammar
        assert "call_ping" in grammar
        assert "tool_call" in grammar
        assert "ws" in grammar

    def test_single_tool_with_params(self):
        tool = Tool(
            name="search",
            description="search",
            handler=_noop,
            parameters=[
                ToolParam(name="query", type="string", description="q"),
                ToolParam(name="limit", type="integer", description="n"),
            ],
        )
        reg = _make_registry(tool)
        grammar = GBNFBuilder.tool_call_grammar(reg)
        assert "call_search" in grammar
        assert "query" in grammar
        assert "limit" in grammar
        assert "string" in grammar
        assert "integer" in grammar

    def test_multiple_tools(self):
        reg = _make_registry(
            Tool(name="read", description="read", handler=_noop),
            Tool(name="write", description="write", handler=_noop),
        )
        grammar = GBNFBuilder.tool_call_grammar(reg)
        assert "call_read" in grammar
        assert "call_write" in grammar
        assert "call_read | call_write" in grammar or "call_write | call_read" in grammar

    def test_enum_param(self):
        tool = Tool(
            name="format",
            description="format",
            handler=_noop,
            parameters=[
                ToolParam(
                    name="style",
                    type="string",
                    description="style",
                    enum=("json", "yaml", "toml"),
                ),
            ],
        )
        reg = _make_registry(tool)
        grammar = GBNFBuilder.tool_call_grammar(reg)
        assert "json" in grammar
        assert "yaml" in grammar
        assert "toml" in grammar

    def test_json_object_grammar(self):
        grammar = GBNFBuilder.json_object_grammar({"name": "string", "age": "integer"})
        assert "root" in grammar
        assert "name" in grammar
        assert "age" in grammar

    def test_json_object_grammar_empty(self):
        grammar = GBNFBuilder.json_object_grammar({})
        assert "root" in grammar
        assert '"{}"' in grammar or '"}"' in grammar

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            GBNFBuilder.json_object_grammar({"x": "complex"})

    def test_thought_or_action_grammar(self):
        reg = _make_registry(Tool(name="calc", description="calc", handler=_noop))
        grammar = GBNFBuilder.thought_or_action_grammar(reg)
        assert "thought_block" in grammar
        assert "action_block" in grammar
        assert "thought" in grammar
        assert "action" in grammar

    def test_grammar_has_primitives(self):
        reg = _make_registry(
            Tool(
                name="test",
                description="test",
                handler=_noop,
                parameters=[
                    ToolParam(name="s", type="string", description="s"),
                    ToolParam(name="i", type="integer", description="i"),
                    ToolParam(name="n", type="number", description="n"),
                    ToolParam(name="b", type="boolean", description="b"),
                ],
            )
        )
        grammar = GBNFBuilder.tool_call_grammar(reg)
        assert "string ::=" in grammar
        assert "integer ::=" in grammar
        assert "number ::=" in grammar
        assert "boolean ::=" in grammar
        assert "ws ::=" in grammar

    def test_boolean_values_in_grammar(self):
        reg = _make_registry(
            Tool(
                name="toggle",
                description="toggle",
                handler=_noop,
                parameters=[
                    ToolParam(name="enabled", type="boolean", description="on/off"),
                ],
            )
        )
        grammar = GBNFBuilder.tool_call_grammar(reg)
        assert '"true"' in grammar
        assert '"false"' in grammar
