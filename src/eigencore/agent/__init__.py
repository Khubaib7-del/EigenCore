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
    "AgentResult",
    "AgentStep",
    "AuditEntry",
    "AuditLog",
    "GBNFBuilder",
    "PolicyDecision",
    "PolicyEngine",
    "ReActAgent",
    "StepKind",
    "Tool",
    "ToolCapability",
    "ToolExecutor",
    "ToolParam",
    "ToolRegistry",
    "ToolResult",
    "default_policy",
    "redact_args",
    "redact_text",
    "resolve_in_workspace",
]
