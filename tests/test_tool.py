"""Tests for the tool definition and registry system."""

import pytest

from eigencore.agent.tool import Tool, ToolParam, ToolRegistry


def _echo(**kwargs):
    return kwargs


class TestToolParam:
    def test_validate_string(self):
        p = ToolParam(name="x", type="string", description="a string")
        assert p.validate("hello") == "hello"

    def test_validate_integer(self):
        p = ToolParam(name="x", type="integer", description="an int")
        assert p.validate(42) == 42

    def test_validate_integer_from_float(self):
        p = ToolParam(name="x", type="integer", description="an int")
        assert p.validate(5.0) == 5

    def test_validate_number(self):
        p = ToolParam(name="x", type="number", description="a number")
        assert p.validate(3.14) == 3.14

    def test_validate_number_from_int(self):
        p = ToolParam(name="x", type="number", description="a number")
        assert p.validate(3) == 3.0

    def test_validate_boolean(self):
        p = ToolParam(name="x", type="boolean", description="a bool")
        assert p.validate(True) is True

    def test_validate_wrong_type_raises(self):
        p = ToolParam(name="x", type="integer", description="an int")
        with pytest.raises(TypeError, match="expects integer"):
            p.validate("not an int")

    def test_validate_enum_pass(self):
        p = ToolParam(name="x", type="string", description="color", enum=("red", "blue"))
        assert p.validate("red") == "red"

    def test_validate_enum_fail(self):
        p = ToolParam(name="x", type="string", description="color", enum=("red", "blue"))
        with pytest.raises(ValueError, match="must be one of"):
            p.validate("green")

    def test_unknown_type_raises(self):
        p = ToolParam(name="x", type="complex", description="bad type")
        with pytest.raises(ValueError, match="Unknown param type"):
            p.validate("anything")


class TestTool:
    def test_create_tool(self):
        t = Tool(name="echo", description="echoes", handler=_echo)
        assert t.name == "echo"
        assert t.parameters == []

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="valid identifier"):
            Tool(name="not-valid", description="bad", handler=_echo)

    def test_validate_args_required(self):
        t = Tool(
            name="greet",
            description="greet",
            handler=_echo,
            parameters=[ToolParam(name="name", type="string", description="who")],
        )
        result = t.validate_args({"name": "Alice"})
        assert result == {"name": "Alice"}

    def test_validate_args_missing_required(self):
        t = Tool(
            name="greet",
            description="greet",
            handler=_echo,
            parameters=[ToolParam(name="name", type="string", description="who")],
        )
        with pytest.raises(ValueError, match="Missing required"):
            t.validate_args({})

    def test_validate_args_unknown(self):
        t = Tool(name="greet", description="greet", handler=_echo, parameters=[])
        with pytest.raises(ValueError, match="Unknown parameters"):
            t.validate_args({"bogus": 1})

    def test_validate_args_optional_default(self):
        t = Tool(
            name="greet",
            description="greet",
            handler=_echo,
            parameters=[
                ToolParam(name="name", type="string", description="who"),
                ToolParam(
                    name="loud",
                    type="boolean",
                    description="shout",
                    required=False,
                    default=False,
                ),
            ],
        )
        result = t.validate_args({"name": "Bob"})
        assert result == {"name": "Bob", "loud": False}

    def test_required_and_optional_params(self):
        t = Tool(
            name="search",
            description="search",
            handler=_echo,
            parameters=[
                ToolParam(name="query", type="string", description="q"),
                ToolParam(name="limit", type="integer", description="n", required=False),
            ],
        )
        assert len(t.required_params) == 1
        assert len(t.optional_params) == 1

    def test_format_for_prompt(self):
        t = Tool(
            name="calc",
            description="Calculate math",
            handler=_echo,
            parameters=[
                ToolParam(name="expr", type="string", description="expression"),
            ],
        )
        text = t.format_for_prompt()
        assert "calc" in text
        assert "Calculate math" in text
        assert "expr" in text


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        t = Tool(name="echo", description="echo", handler=_echo)
        reg.register(t)
        assert reg.get("echo") is t
        assert len(reg) == 1

    def test_register_duplicate_raises(self):
        reg = ToolRegistry()
        t = Tool(name="echo", description="echo", handler=_echo)
        reg.register(t)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(t)

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nope") is None

    def test_remove(self):
        reg = ToolRegistry()
        t = Tool(name="echo", description="echo", handler=_echo)
        reg.register(t)
        assert reg.remove("echo") is True
        assert reg.remove("echo") is False
        assert len(reg) == 0

    def test_contains(self):
        reg = ToolRegistry()
        t = Tool(name="echo", description="echo", handler=_echo)
        reg.register(t)
        assert "echo" in reg
        assert "nope" not in reg

    def test_tool_names(self):
        reg = ToolRegistry()
        reg.register(Tool(name="a", description="a", handler=_echo))
        reg.register(Tool(name="b", description="b", handler=_echo))
        assert set(reg.tool_names) == {"a", "b"}

    def test_format_for_prompt_empty(self):
        reg = ToolRegistry()
        assert "No tools" in reg.format_for_prompt()
