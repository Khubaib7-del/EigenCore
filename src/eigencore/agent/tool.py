"""Tool definitions and registry for agent function calling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ToolParam:
    """A single parameter in a tool's signature."""

    name: str
    type: str  # "string", "integer", "number", "boolean"
    description: str
    required: bool = True
    enum: Optional[tuple[str, ...]] = None
    default: Any = None

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }

    def validate(self, value: Any) -> Any:
        expected = self._TYPE_MAP.get(self.type)
        if expected is None:
            raise ValueError(f"Unknown param type: {self.type}")

        if not isinstance(value, expected):
            if self.type == "number" and isinstance(value, int):
                value = float(value)
            elif self.type == "integer" and isinstance(value, float) and value == int(value):
                value = int(value)
            else:
                raise TypeError(
                    f"Parameter '{self.name}' expects {self.type}, got {type(value).__name__}"
                )

        if self.enum and value not in self.enum:
            raise ValueError(f"Parameter '{self.name}' must be one of {self.enum}, got {value!r}")

        return value


@dataclass
class Tool:
    """A callable tool that an agent can invoke."""

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: list[ToolParam] = field(default_factory=list)

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(f"Tool name must be a valid identifier, got {self.name!r}")

    @property
    def required_params(self) -> list[ToolParam]:
        return [p for p in self.parameters if p.required]

    @property
    def optional_params(self) -> list[ToolParam]:
        return [p for p in self.parameters if not p.required]

    def param_names(self) -> set[str]:
        return {p.name for p in self.parameters}

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        validated: dict[str, Any] = {}

        for param in self.required_params:
            if param.name not in args:
                raise ValueError(f"Missing required parameter: '{param.name}'")

        for param in self.parameters:
            if param.name in args:
                validated[param.name] = param.validate(args[param.name])
            elif not param.required and param.default is not None:
                validated[param.name] = param.default

        unknown = set(args.keys()) - self.param_names()
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")

        return validated

    def format_for_prompt(self) -> str:
        lines = [f"### {self.name}", f"{self.description}", "Parameters:"]
        for p in self.parameters:
            req = " (required)" if p.required else f" (optional, default={p.default!r})"
            enum_hint = f", one of: {list(p.enum)}" if p.enum else ""
            lines.append(f"  - {p.name} ({p.type}{enum_hint}): {p.description}{req}")
        return "\n".join(lines)


class ToolRegistry:
    """Registry of tools available to an agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def format_for_prompt(self) -> str:
        if not self._tools:
            return "No tools available."
        sections = [tool.format_for_prompt() for tool in self._tools.values()]
        return "\n\n".join(sections)
