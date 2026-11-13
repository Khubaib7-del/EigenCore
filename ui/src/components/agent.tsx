import { useEffect, useRef, useState } from "react";
import type { ApprovalDecision, ApprovalRequest } from "../api";
import type { ToolCall } from "../state/store";
import { Icon, type IconName } from "./Icon";
import { Kbd, Shimmer } from "./primitives";

/* ── Thinking fold (after Beautiful UI's Thinking, MIT) ──────────────────
 * Live: shimmering "Thinking" with the reasoning streaming below, open.
 * Settled: collapses to "Reasoning · N words", expandable.
 */
export function ThinkingFold({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);
  const wasLive = useRef(live);
  useEffect(() => {
    if (wasLive.current && !live) setOpen(false);
    wasLive.current = live;
  }, [live]);
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  return (
    <div className={`thinking ${open ? "open" : ""}`}>
      <button className="thinking-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <Icon name="sparkle" size={13} />
        {live ? <Shimmer>Thinking</Shimmer> : <span>Reasoning</span>}
        {!live && <span className="muted">· {words} words</span>}
        <Icon name="chevronRight" size={13} className="thinking-chev" />
      </button>
      <div className="thinking-body" hidden={!open}>
        <p>{text}</p>
      </div>
    </div>
  );
}

/* ── Tool chips (after Beautiful UI's Tool Chips, MIT) ─────────────────── */
const TOOL_ICON: Record<string, IconName> = { read_file: "file", list_dir: "folder", run_shell: "terminal", web_get: "globe" };

export function ToolChips({ calls }: { calls: ToolCall[] }) {
  const [open, setOpen] = useState(false);
  if (!calls.length) return null;
  return (
    <div className="tools">
      <button className="tools-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span>
          {calls.length} tool call{calls.length > 1 ? "s" : ""}
        </span>
        <Icon name={open ? "chevronDown" : "chevronRight"} size={13} />
      </button>
      <div className="tool-row-list">
        {calls.map((c, i) => (
          <div key={i} className="tool-row">
            <Icon name={TOOL_ICON[c.tool_name] ?? "terminal"} size={13} />
            <span className="mono tool-name">{c.tool_name}</span>
            <span className="tool-chip mono">{c.summary}</span>
            <span className={`tool-status ${c.ok ? "ok" : "err"}`}>{c.ok ? "Done" : "Denied"}</span>
            {open && <div className="tool-detail mono">{c.ok ? `→ ${c.summary}` : "→ not run"}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Approval card ─────────────────────────────────────────────────────
 * One pending tool call. Args arrive already redacted by the policy engine.
 * Allow once = ASK approve; Always = SESSION (this tool, this session); Deny.
 */
const CAP_COPY: Record<ApprovalRequest["capability"], string> = {
  READ: "Reads files",
  WRITE: "Changes files",
  EXECUTE: "Runs a program",
  NETWORK: "Uses the network",
};

export function ApprovalCard({
  request,
  decision,
  onDecide,
}: {
  request: ApprovalRequest;
  decision?: ApprovalDecision;
  onDecide: (d: ApprovalDecision) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (decision) return;
    const k = (e: KeyboardEvent) => {
      if (!ref.current || document.activeElement?.tagName === "TEXTAREA") return;
      if (e.key === "Enter") onDecide("once");
      if (e.key.toLowerCase() === "a" && e.altKey) onDecide("session");
      if (e.key === "Escape") onDecide("deny");
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [decision, onDecide]);

  if (decision) {
    const txt = decision === "deny" ? "Denied" : decision === "session" ? "Allowed for this session" : "Allowed once";
    return (
      <div className="approval settled">
        <Icon name={decision === "deny" ? "x" : "check"} size={13} />
        <span className="mono">{request.tool_name}</span>
        <span className="muted">{txt}</span>
      </div>
    );
  }
  return (
    <div className="approval" ref={ref} role="group" aria-label={`Approve ${request.tool_name}`}>
      <div className="approval-head">
        <span className="approval-icon">
          <Icon name="shield" size={15} />
        </span>
        <div>
          <div className="approval-title">
            Allow <span className="mono">{request.tool_name}</span>?
          </div>
          <div className="muted approval-reason">{request.reason}</div>
        </div>
        <span className="chip mono cap">{CAP_COPY[request.capability]}</span>
      </div>
      <dl className="approval-args">
        {Object.entries(request.args).map(([k, v]) => (
          <div key={k}>
            <dt className="mono">{k}</dt>
            <dd className={`mono ${String(v).includes("REDACTED") ? "redacted" : ""}`}>{String(v).includes("REDACTED") ? "•••• redacted" : String(v)}</dd>
          </div>
        ))}
      </dl>
      <div className="approval-actions">
        <span className="muted approval-note">Checked by the tool policy. Paths stay inside your workspace.</span>
        <button className="btn btn-ghost btn-sm" onClick={() => onDecide("deny")}>
          Deny <Kbd>Esc</Kbd>
        </button>
        <button className="btn btn-secondary btn-sm" onClick={() => onDecide("session")} title="Allow this tool for the rest of the session">
          Always <Kbd>Alt A</Kbd>
        </button>
        <button className="btn btn-primary btn-sm" onClick={() => onDecide("once")}>
          Allow once <Kbd>↵</Kbd>
        </button>
      </div>
    </div>
  );
}
