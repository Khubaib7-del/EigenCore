"""Agent framework — tool calling, GBNF grammars, and ReAct reasoning."""

from eigencore.agent.executor import ToolExecutor, ToolResult
from eigencore.agent.grammar import GBNFBuilder
from eigencore.agent.react import AgentResult, AgentStep, ReActAgent, StepKind
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry

__all__ = [
    "Tool",
    "ToolParam",
    "ToolRegistry",
    "GBNFBuilder",
    "ToolExecutor",
    "ToolResult",
    "ReActAgent",
    "AgentResult",
    "AgentStep",
    "StepKind",
]
