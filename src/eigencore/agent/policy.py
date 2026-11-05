"""
Tool policy engine — Phase 4 (guardrails).

Enforcement lives here, in code, below any UI: a tool call is denied unless
its capability is explicitly allow-listed, and path arguments are confined to
an authorized workspace root (symlinks resolved, traversal rejected). Every
decision is written to an append-only audit log with sensitive values
redacted.

This is a policy layer, NOT a sandbox: handlers still run in-process. OS-level
process isolation is a separate, later hardening step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eigencore.agent.tool import Tool, ToolCapability

# Parameter names that look like paths and must stay inside the workspace.
_PATH_PARAM_NAMES = {
    "path",
    "file",
    "file_path",
    "filename",
    "target",
    "dest",
    "destination",
    "dir",
    "directory",
    "folder",
    "output_path",
    "out",
}

# Parameter names whose values must never appear in audit logs.
_SENSITIVE_PARAM_NAMES = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "auth",
    "credential",
    "credentials",
}

_SENSITIVE_RE = re.compile(
    r"(?i)(password|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|authorization)\s*[:=]\s*\S+"
)


def _is_path_param(name: str) -> bool:
    return name.lower() in _PATH_PARAM_NAMES


def _is_sensitive_param(name: str) -> bool:
    return name.lower() in _SENSITIVE_PARAM_NAMES


def redact_text(text: str) -> str:
    """Scrub obvious credential assignments from free text (tool output, errors)."""
    return _SENSITIVE_RE.sub(r"\1=***REDACTED***", text)


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    return {k: ("***REDACTED***" if _is_sensitive_param(k) else v) for k, v in args.items()}


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    capability: ToolCapability | None = None


@dataclass
class AuditEntry:
    timestamp: str
    tool_name: str
    capability: str
    args: dict[str, Any]
    decision: str  # "allowed" | "denied"
    reason: str
    status: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "capability": self.capability,
            "args": self.args,
            "decision": self.decision,
            "reason": self.reason,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
        }


class AuditLog:
    """Append-only JSONL audit trail for tool decisions and outcomes."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def denied_count(self) -> int:
        return sum(1 for e in self.entries if e.decision == "denied")


def resolve_in_workspace(path_value: str, workspace_root: Path) -> Path:
    """Resolve a path and require it to stay inside workspace_root.

    Rejects absolute paths outside the root, `..` traversal, and symlinks
    that resolve outside the root.
    """
    root = workspace_root.resolve()
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path {path_value!r} resolves to {resolved}, outside workspace {root}")
    return resolved


@dataclass
class PolicyEngine:
    """Decides whether a tool call may run.

    - Deny by default: a capability must be present in `allowed_capabilities`.
    - If `workspace_root` is set, path-like string args must resolve inside it.
    - Every decision is recorded in `audit_log`.
    """

    allowed_capabilities: set[ToolCapability] = field(default_factory=set)
    workspace_root: Path | None = None
    audit_log: AuditLog | None = None

    def check(self, tool: Tool, args: dict[str, Any]) -> PolicyDecision:
        cap = getattr(tool, "capability", ToolCapability.EXECUTE)
        if cap not in self.allowed_capabilities:
            decision = PolicyDecision(
                allowed=False,
                capability=cap,
                reason=(
                    f"capability {cap.name} is not allow-listed "
                    f"(allowed: {sorted(c.name for c in self.allowed_capabilities) or 'none'})"
                ),
            )
        else:
            decision = PolicyDecision(
                allowed=True, capability=cap, reason="capability allow-listed"
            )

        if decision.allowed and self.workspace_root is not None:
            for param in tool.parameters:
                if (
                    param.name in args
                    and _is_path_param(param.name)
                    and isinstance(args[param.name], str)
                ):
                    try:
                        resolve_in_workspace(args[param.name], self.workspace_root)
                    except ValueError as e:
                        decision = PolicyDecision(
                            allowed=False,
                            capability=cap,
                            reason=str(e),
                        )
                        break

        if self.audit_log is not None:
            self.audit_log.record(
                AuditEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    tool_name=tool.name,
                    capability=cap.name,
                    args=redact_args(args),
                    decision="allowed" if decision.allowed else "denied",
                    reason=decision.reason,
                )
            )
        return decision


def default_policy(
    workspace_root: Path | None = None, audit_log: AuditLog | None = None
) -> PolicyEngine:
    """Conservative default: read-only tools allowed, everything else denied."""
    return PolicyEngine(
        allowed_capabilities={ToolCapability.READ},
        workspace_root=workspace_root,
        audit_log=audit_log,
    )
