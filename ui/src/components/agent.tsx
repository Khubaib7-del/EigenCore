import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ApprovalDecision, ApprovalRequest } from "../api";
import type { ToolCall } from "../state/store";
import { Icon, type IconName } from "./Icon";
import { Kbd, Shimmer } from "./primitives";

/* ── Thinking orb ────────────────────────────────────────────────────────
 * A flat monochrome orb on the left of the agent line. No gradients: a solid
 * core plus one ring whose motion says what the engine is doing. Colour only
 * for meaning (done = ok, stopped = warn, error = err), always next to a word.
 */
export type OrbState = "idle" | "thinking" | "writing" | "waiting" | "done" | "stopped" | "error";

export function ThinkingOrb({ state, size = 18 }: { state: OrbState; size?: number }) {
  return (
    <span className={`orb orb-${state}`} style={{ width: size, height: size }} aria-hidden>
      <span className="orb-ring" />
      <span className="orb-core" />
    </span>
  );
}

/** A status word that rolls up when it changes (Thinking → Analyzing → Writing). */
export function RollingWord({ words, interval = 1800 }: { words: string[]; interval?: number }) {
  const [i, setI] = useState(0);
  const key = words.join("|");
  useEffect(() => {
    setI(0);
    if (words.length < 2) return;
    const t = setInterval(() => setI((n) => (n + 1) % words.length), interval);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, interval]);
  const word = words[i % words.length] ?? "";
  return (
    <span className="roll" aria-live="polite">
      <span className="roll-word" key={word}>
        {word}
      </span>
    </span>
  );
}

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
        <Icon name="sparkles" size={13} />
        {live ? <Shimmer>Reasoning</Shimmer> : <span>Reasoning</span>}
        {!live && <span className="muted">· {words} words</span>}
        <Icon name="chevronRight" size={13} className="thinking-chev" />
      </button>
      <div className="thinking-body" hidden={!open}>
        <p>{text}</p>
      </div>
    </div>
  );
}

/* ── Git-style tree ──────────────────────────────────────────────────────
 * Rows with ├─ / └─ / │ guides, like `tree` or a git log graph. Used for
 * tool steps and for file lists in replies.
 */
export interface TreeRow {
  depth: number;
  icon: IconName;
  label: ReactNode;
  meta?: ReactNode;
  status?: ReactNode;
  muted?: boolean;
}

function guides(rows: TreeRow[]): string[] {
  // isLast[i]: no later sibling at the same depth before the parent closes
  const isLast = rows.map((r, i) => {
    for (let j = i + 1; j < rows.length; j++) {
      if (rows[j].depth < r.depth) return true;
      if (rows[j].depth === r.depth) return false;
    }
    return true;
  });
  const open: boolean[] = []; // open[d]: the ancestor at depth d still has siblings below
  return rows.map((r, i) => {
    let g = "";
    for (let d = 0; d < r.depth; d++) g += open[d] ? "│  " : "   ";
    g += isLast[i] ? "└─ " : "├─ ";
    open[r.depth] = !isLast[i];
    open.length = r.depth + 1;
    return g;
  });
}

export function TreeList({ rows, label }: { rows: TreeRow[]; label?: string }) {
  const g = guides(rows);
  return (
    <ul className="tree" role="tree" aria-label={label}>
      {rows.map((r, i) => (
        <li key={i} role="treeitem" className={`tree-row ${r.muted ? "muted" : ""}`}>
          <span className="tree-guide mono" aria-hidden>
            {g[i]}
          </span>
          <Icon name={r.icon} size={14} className="tree-icon" />
          <span className="tree-label">{r.label}</span>
          {r.meta && <span className="tree-meta mono">{r.meta}</span>}
          {r.status && <span className="tree-status">{r.status}</span>}
        </li>
      ))}
    </ul>
  );
}

/* ── Tool steps (was tool chips) ─────────────────────────────────────── */
const TOOL_ICON: Record<string, IconName> = {
  read_file: "fileCode",
  list_dir: "folderOpen",
  search_files: "search",
  write_file: "pencil",
  run_shell: "terminal",
  web_get: "globe",
};

export function ToolSteps({ calls, pending }: { calls: ToolCall[]; pending?: string | null }) {
  const [open, setOpen] = useState(true);
  const n = calls.length + (pending ? 1 : 0);
  if (!n) return null;
  const rows: TreeRow[] = calls.map((c) => ({
    depth: 0,
    icon: TOOL_ICON[c.tool_name] ?? "wrench",
    label: <span className="mono">{c.tool_name}</span>,
    meta: c.summary,
    status: (
      <span className={`step-status ${c.ok ? "ok" : "err"}`}>
        <Icon name={c.ok ? "check" : "ban"} size={12} />
        {c.ok ? "Done" : "Denied"}
      </span>
    ),
  }));
  if (pending)
    rows.push({
      depth: 0,
      icon: TOOL_ICON[pending] ?? "wrench",
      label: <span className="mono">{pending}</span>,
      meta: "waiting for you",
      status: (
        <span className="step-status warn">
          <Icon name="shield" size={12} />
          Needs approval
        </span>
      ),
    });
  return (
    <div className="steps">
      <button className="steps-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <Icon name="gitBranch" size={14} />
        <span>Steps</span>
        <span className="muted mono">{n}</span>
        <Icon name="chevronRight" size={13} className={`steps-chev ${open ? "open" : ""}`} />
      </button>
      {open && <TreeList rows={rows} label="Tool steps" />}
    </div>
  );
}

/* ── Approval card ─────────────────────────────────────────────────────
 * One pending tool call. Args arrive already redacted by the policy engine.
 * Allow once = ASK approve; Allow for session = SESSION; Deny. Focus starts
 * on Deny, the safe choice, so a stray Enter never runs a tool.
 */
const CAP: Record<ApprovalRequest["capability"], { copy: string; icon: IconName }> = {
  READ: { copy: "Reads files", icon: "fileCode" },
  WRITE: { copy: "Changes files", icon: "pencil" },
  EXECUTE: { copy: "Runs a program", icon: "terminal" },
  NETWORK: { copy: "Uses the network", icon: "globe" },
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
  const denyRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (decision) return;
    denyRef.current?.focus({ preventScroll: true });
    const k = (e: KeyboardEvent) => {
      if (!ref.current || document.activeElement?.tagName === "TEXTAREA") return;
      if (e.key === "Escape") onDecide("deny");
      else if (e.altKey && e.key.toLowerCase() === "a") onDecide("session");
      else if (e.altKey && e.key.toLowerCase() === "o") onDecide("once");
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [decision, onDecide]);

  if (decision) {
    const txt = decision === "deny" ? "Denied" : decision === "session" ? "Allowed for this session" : "Allowed once";
    return (
      <div className={`approval settled ${decision === "deny" ? "denied" : ""}`}>
        <Icon name={decision === "deny" ? "ban" : "shieldCheck"} size={14} />
        <span className="mono">{request.tool_name}</span>
        <span className="muted">{txt}</span>
      </div>
    );
  }
  const cap = CAP[request.capability];
  return (
    <div className="approval" ref={ref} role="group" aria-label={`Approve ${request.tool_name}`}>
      <div className="approval-head">
        <span className="approval-icon">
          <Icon name="shield" size={16} />
        </span>
        <div className="approval-text">
          <div className="approval-kicker">Approval needed</div>
          <div className="approval-title">
            Allow <span className="mono">{request.tool_name}</span>?
          </div>
          <div className="muted approval-reason">{request.reason}</div>
        </div>
        <span className="chip cap">
          <Icon name={cap.icon} size={12} />
          {cap.copy}
        </span>
      </div>
      <dl className="approval-args">
        {Object.entries(request.args).map(([k, v]) => (
          <div key={k}>
            <dt className="mono">{k}</dt>
            <dd className={`mono ${String(v).includes("REDACTED") ? "redacted" : ""}`}>
              {String(v).includes("REDACTED") ? (
                <>
                  <Icon name="lock" size={11} /> redacted
                </>
              ) : (
                String(v)
              )}
            </dd>
          </div>
        ))}
      </dl>
      <div className="approval-actions">
        <span className="muted approval-note">
          <Icon name="shieldCheck" size={13} /> Checked by the tool policy
        </span>
        <button ref={denyRef} className="btn btn-secondary btn-sm deny" onClick={() => onDecide("deny")}>
          Deny <Kbd>Esc</Kbd>
        </button>
        <button className="btn btn-secondary btn-sm" onClick={() => onDecide("session")} title="Allow this tool for the rest of the session">
          Allow for session <Kbd>Alt A</Kbd>
        </button>
        <button className="btn btn-primary btn-sm" onClick={() => onDecide("once")}>
          Allow once <Kbd>Alt O</Kbd>
        </button>
      </div>
    </div>
  );
}
