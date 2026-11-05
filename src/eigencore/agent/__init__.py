"""Agent framework — tool calling, GBNF grammars, and ReAct reasoning."""

from eigencore.agent.executor import ToolExecutor, ToolResult
from eigencore.agent.grammar import GBNFBuilder
from eigencore.agent.policy import (
    AuditEntry,
    AuditLog,
    PolicyDecision,
    PolicyEngine,
    ToolCapability,
    default_policy,
    redact_args,
    redact_text,
    resolve_in_workspace,
)
from eigencore.agent.react import AgentResult, AgentStep, ReActAgent, StepKind
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry

__all__ = [
    "AuditEntry",
    "AuditLog",
    "PolicyDecision",
    "PolicyEngine",
    "ToolCapability",
    "Tool",
    "ToolParam",
    "ToolRegistry",
    "default_policy",
    "redact_args",
    "redact_text",
    "resolve_in_workspace",
    "GBNFBuilder",
    "ToolExecutor",
    "ToolResult",
    "ReActAgent",
    "AgentResult",
    "AgentStep",
    "StepKind",
]
