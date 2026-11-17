import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from "react";
import { api } from "../api";
import { chatBusy, useStore, type Chat, type Screen } from "../state/store";
import { GridLoader } from "./GridLoader";
import { Icon, type IconName } from "./Icon";
import { Spinner } from "./Spinner";
import { Button, Dialog, GlideMenu, IconButton, Kbd, SoonPill } from "./primitives";

/* ── Title bar for a frameless window ───────────────────────────────────
 * The whole bar is a drag region (pywebview: class "pywebview-drag-region";
 * -webkit-app-region for WebView2/Chromium). Buttons opt out of dragging.
 */
export function TitleBar() {
  const { state } = useStore();
  const win = (op: "minimize" | "toggle_maximize" | "close") => {
    if (op !== "minimize" || matchMedia("(prefers-reduced-motion: reduce)").matches) return api.windowControl(op);
    // fade the page out first, so the window does not vanish mid-frame
    const root = document.documentElement;
    root.classList.add("win-hiding");
    setTimeout(() => {
      api.windowControl("minimize");
      setTimeout(() => root.classList.remove("win-hiding"), 60);
    }, 110);
  };
  useEffect(() => {
    // the window came back (taskbar click, Alt+Tab): fade the page in
    const root = document.documentElement;
    let t: ReturnType<typeof setTimeout> | undefined;
    const show = () => {
      root.classList.remove("win-hiding");
      root.classList.add("win-showing");
      clearTimeout(t);
      t = setTimeout(() => root.classList.remove("win-showing"), 240);
    };
    const onVis = () => document.visibilityState === "visible" && show();
    document.addEventListener("eigencore:window", show);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      document.removeEventListener("eigencore:window", show);
      document.removeEventListener("visibilitychange", onVis);
      clearTimeout(t);
    };
  }, []);
  const onDouble = (e: ReactMouseEvent) => {
    if (!(e.target as HTMLElement).closest("button")) win("toggle_maximize");
  };
  const engine = state.engine;
  const status =
    engine.phase === "loading"
      ? "Loading model"
      : engine.phase === "reading"
        ? "Reading prompt"
        : engine.phase === "writing"
          ? "Writing"
          : engine.phase === "error"
            ? "Engine error"
            : state.activeModel
              ? "Ready"
              : "No model";
  return (
    <header className="titlebar pywebview-drag-region" onDoubleClick={onDouble}>
      <div className="titlebar-brand">
        <span className="wordmark">EigenCore</span>
        <span className="titlebar-sep" />
        {engine.phase === "loading" || engine.phase === "reading" || engine.phase === "writing" ? (
          <GridLoader kind={engine.phase === "loading" ? "loading" : engine.phase === "reading" ? "thinking" : "writing"} size={11} />
        ) : (
          <span className={`engine-dot ${engine.phase === "idle" && !state.activeModel ? "none" : engine.phase}`} aria-hidden />
        )}
        <span className="titlebar-status">{status}</span>
        {state.activeModel && <span className="mono titlebar-model">{state.activeModel}</span>}
      </div>
      {api.kind === "mock" && (
        <div className="titlebar-center">
          <span className="chip sample" title="Preview build: every value comes from the mock API">
            Preview · mock data
          </span>
        </div>
      )}
      <div className="titlebar-controls no-drag">
        <button aria-label="Minimize" onClick={() => win("minimize")}>
          <Icon name="minus" size={14} />
        </button>
        <button aria-label="Maximize" onClick={() => win("toggle_maximize")}>
          <Icon name="maximize" size={12} />
        </button>
        <button aria-label="Close" className="close" onClick={() => win("close")}>
          <Icon name="x" size={14} />
        </button>
      </div>
    </header>
  );
}

type NavItem = { id: Screen; label: string; icon: IconName; tint?: string } | { id: null; label: string; icon: IconName; soon: string; tint?: string };

/** Main nav. Items with `soon` are shown, muted, and do nothing yet. */
const NAV_TOP: NavItem[] = [
  { id: "chat", label: "Chat", icon: "chat", tint: "tint-blue" },
  { id: null, label: "Agents", icon: "bot", tint: "tint-violet", soon: "Agents: coming soon. Tool use with approvals is being wired up." },
  { id: null, label: "Files", icon: "folder", tint: "tint-amber", soon: "Workspace files: coming soon. Attach a folder for the agent to work in." },
  { id: "models", label: "Models", icon: "models", tint: "tint-purple" },
  { id: "monitor", label: "Monitor", icon: "activity", tint: "tint-green" },
];
const NAV_FOOT: NavItem[] = [
  { id: "profile", label: "Profile", icon: "cpu", tint: "tint-orange" },
  { id: "settings", label: "Settings", icon: "settings", tint: "tint-mono gear-nav" },
];
const NAV = [...NAV_TOP, ...NAV_FOOT].filter((n): n is { id: Screen; label: string; icon: IconName; tint?: string } => n.id !== null);

function NavButton({ n, collapsed, extra }: { n: NavItem; collapsed: boolean; extra?: ReactNode }) {
  const { state, dispatch } = useStore();
  if (n.id === null)
    return (
      <div className="nav-item soon-item" aria-disabled="true" title={n.soon} role="link">
        <Icon name={n.icon} size={16} className={n.tint} />
        {!collapsed && <span>{n.label}</span>}
        {!collapsed && <SoonPill title={n.soon} />}
      </div>
    );
  const id = n.id;
  return (
    <button
      data-glide
      className={`nav-item ${state.screen === id ? "active" : ""}`}
      aria-current={state.screen === id ? "page" : undefined}
      onClick={() => dispatch({ type: "nav", screen: id })}
      title={collapsed ? n.label : undefined}
    >
      <Icon name={n.icon} size={16} className={n.tint} />
      {!collapsed && <span>{n.label}</span>}
      {extra}
    </button>
  );
}

function HistoryItem({ c, active }: { c: Chat; active: boolean }) {
  const { dispatch } = useStore();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(c.title);
  const [confirm, setConfirm] = useState(false);
  const busy = chatBusy(c);
  const commit = () => {
    setEditing(false);
    if (draft.trim() && draft.trim() !== c.title) dispatch({ type: "renameChat", id: c.id, title: draft });
  };
  return (
    <div className={`history-item ${active ? "active" : ""} ${busy ? "busy" : ""}`} data-glide>
      {editing ? (
        <input
          className="history-rename"
          value={draft}
          autoFocus
          aria-label="Chat title"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") {
              setDraft(c.title);
              setEditing(false);
            }
          }}
        />
      ) : (
        <button className="history-open" onClick={() => dispatch({ type: "openChat", id: c.id })} onDoubleClick={() => setEditing(true)} title={c.title}>
          {busy === "approval" ? (
            <span className="history-wait" title="Waiting for your approval">
              <Icon name="shield" size={12} />
            </span>
          ) : busy ? (
            busy === "reading" ? (
              <Spinner kind="pulse" tone="violet" size={14} label="Reading the prompt" />
            ) : (
              <Spinner kind="linear-dots" tone="blue" size={14} label="Writing" />
            )
          ) : null}
          <span className="ellipsis">{c.title}</span>
        </button>
      )}
      {!editing && (
        <span className="history-actions">
          <IconButton
            icon="pencil"
            label="Rename"
            size={13}
            onClick={() => {
              setDraft(c.title);
              setEditing(true);
            }}
          />
          <IconButton icon="trash" label={busy ? "Stop the reply before deleting" : "Delete"} size={13} disabled={!!busy} onClick={() => setConfirm(true)} />
        </span>
      )}
      <Dialog
        open={confirm}
        title="Delete this chat?"
        onClose={() => setConfirm(false)}
        actions={
          <>
            <Button variant="ghost" onClick={() => setConfirm(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              icon="trash"
              onClick={() => {
                setConfirm(false);
                dispatch({ type: "deleteChat", id: c.id });
              }}
            >
              Delete
            </Button>
          </>
        }
      >
        <p>
          “{c.title}” and its {c.messages.length} message{c.messages.length === 1 ? "" : "s"} are removed from this device.
        </p>
      </Dialog>
    </div>
  );
}

/** The EigenCore mark (die + core), drawn small for the update button. */
function Mark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="40 40 420 420" aria-hidden className="eg-mark">
      <path
        className="eg-die"
        d="M157.5,48.5 L202.5,48.5 A37.5 37.5 0 0 1 240.0,86.0 L240.0,183.5 A52.5 52.5 0 0 0 292.5,236.0 L390.0,236.0 A37.5 37.5 0 0 1 427.5,273.5 L427.5,318.5 A105.0 105.0 0 0 1 322.5,423.5 L157.5,423.5 A105.0 105.0 0 0 1 52.5,318.5 L52.5,153.5 A105.0 105.0 0 0 1 157.5,48.5 Z"
      />
      <g className="eg-core">
        <rect x="300" y="112" width="122.5" height="122.5" rx="35" fill="#2E3A9E" />
        <rect x="286.2" y="96.6" width="122.5" height="122.5" rx="35" fill="#5B6CF0" />
        <rect x="318" y="128.4" width="58.9" height="58.9" rx="17" fill="#C9CFFF" />
      </g>
    </svg>
  );
}

/* ── Update button ─────────────────────────────────────────────────────
 * An icon, not a pill: the EigenCore mark with a small coloured badge for
 * the update state. Hovering (or focusing) it grows it into a pill that says
 * what is going on. A download draws a ring round the mark. Once installed,
 * or when already current, the badge is a green check until a new version
 * shows up.
 */
export function UpdateButton({ compact }: { compact?: boolean }) {
  const { state, actions } = useStore();
  const u = state.update;
  let label = "Check for updates";
  let sub = `v${state.version}`;
  let onClick: (() => void) | undefined = actions.checkUpdate;
  let pct: number | null = null;
  let badge: ReactNode = <Icon name="refresh" size={9} />;
  let tone = "idle";
  if (u.kind === "checking") {
    label = "Checking…";
    onClick = undefined;
    badge = <Spinner kind="comet" size={11} />;
    tone = "busy";
  } else if (u.kind === "available") {
    label = `${u.tag} is available`;
    sub = "Click to update";
    onClick = actions.installUpdate;
    badge = <Icon name="arrowDown" size={10} strokeWidth={2.4} />;
    tone = "available";
  } else if (u.kind === "downloading") {
    label = `Downloading ${u.tag}`;
    sub = `${Math.round(u.pct)}%`;
    pct = u.pct;
    onClick = undefined;
    badge = <Spinner kind="trace" size={11} />;
    tone = "busy";
  } else if (u.kind === "ready_to_install") {
    label = `Installing ${u.tag}`;
    sub = "EigenCore restarts on its own";
    onClick = undefined;
    badge = <Spinner kind="hourglass" size={11} />;
    tone = "busy";
  } else if (u.kind === "manual") {
    label = `Get ${u.tag}`;
    sub = "Opens the release page";
    onClick = () => api.openUrl(u.url);
    badge = <Icon name="arrowDown" size={10} strokeWidth={2.4} />;
    tone = "available";
  } else if (u.kind === "up_to_date") {
    label = "Up to date";
    sub = `v${state.version} · check again`;
    badge = <Icon name="check" size={10} strokeWidth={2.6} />;
    tone = "current";
  } else if (u.kind === "error") {
    label = "Update check failed";
    sub = "Click to retry";
    badge = <Icon name="alert" size={9} strokeWidth={2.2} />;
    tone = "error";
  }
  const R = 15;
  const C = 2 * Math.PI * R;
  return (
    <button
      className={`update-mark ${tone} ${compact ? "compact" : ""}`}
      onClick={onClick}
      disabled={!onClick}
      aria-label={`${label}. ${sub}`}
      title={compact ? `${label} · ${sub}` : undefined}
    >
      <span className="um-icon">
        <Mark size={20} />
        {pct !== null && (
          <svg className="um-ring" viewBox="0 0 34 34" aria-hidden>
            <circle cx="17" cy="17" r={R} />
            <circle cx="17" cy="17" r={R} className="um-ring-fill" strokeDasharray={C} strokeDashoffset={C * (1 - pct / 100)} />
          </svg>
        )}
        <span className="um-badge">{badge}</span>
      </span>
      {!compact && (
        <span className="um-text">
          <span className="um-label">{label}</span>
          <span className="um-sub mono">{sub}</span>
        </span>
      )}
    </button>
  );
}

export function Sidebar() {
  const { state, dispatch } = useStore();
  const collapsed = state.sidebarCollapsed;
  const groups = useMemo(() => {
    const g: Record<string, typeof state.chats> = {};
    for (const c of state.chats) (g[c.group] ??= []).push(c);
    return Object.entries(g);
  }, [state.chats]);
  const anyBusy = state.chats.some((c) => chatBusy(c));
  const engineLive = state.engine.phase !== "idle" && state.engine.phase !== "error";
  return (
    <nav className={`sidebar ${collapsed ? "collapsed" : ""}`} aria-label="Main">
      <div className="sidebar-top">
        <button className="new-chat" onClick={() => dispatch({ type: "newChat" })} title="New chat (Ctrl N)">
          <Icon name="plus" size={15} />
          {!collapsed && <span>New chat</span>}
          {!collapsed && <Kbd>Ctrl N</Kbd>}
        </button>
        <IconButton icon="sidebar" label={collapsed ? "Expand sidebar" : "Collapse sidebar"} onClick={() => dispatch({ type: "toggleSidebar" })} />
      </div>
      {collapsed ? (
        <IconButton icon="search" label="Search (Ctrl K)" className="collapsed-search" onClick={() => dispatch({ type: "palette", open: true })} />
      ) : (
        <button className="search-trigger" onClick={() => dispatch({ type: "palette", open: true })}>
          <Icon name="search" size={14} />
          <span>Search</span>
          <Kbd>Ctrl K</Kbd>
        </button>
      )}
      <GlideMenu className="nav-list">
        {NAV_TOP.map((n) => (
          <NavButton
            key={n.label}
            n={n}
            collapsed={collapsed}
            extra={
              n.id === "monitor" && engineLive ? (
                <span className={collapsed ? "nav-dot" : "nav-live"}>{collapsed ? null : <Spinner kind="bars" tone="pink" size={13} label="Live" />}</span>
              ) : n.id === "chat" && collapsed && anyBusy ? (
                <span className="nav-dot" />
              ) : undefined
            }
          />
        ))}
      </GlideMenu>
      {!collapsed && (
        <div className="history">
          <div className="history-head">
            <Icon name="history" size={13} />
            <span>Chats</span>
            <span className="mono">{state.chats.length}</span>
          </div>
          {state.chats.length === 0 && <div className="history-empty muted">Your chats show up here.</div>}
          <GlideMenu>
            {groups.map(([g, chats]) => (
              <div key={g} className="history-group">
                <div className="history-label">{g}</div>
                {chats.map((c) => (
                  <HistoryItem key={c.id} c={c} active={c.id === state.activeChatId && state.screen === "chat"} />
                ))}
              </div>
            ))}
          </GlideMenu>
        </div>
      )}
      {collapsed && <span className="grow" />}
      <GlideMenu className="nav-list nav-foot">
        {NAV_FOOT.map((n) => (
          <NavButton key={n.label} n={n} collapsed={collapsed} />
        ))}
      </GlideMenu>
      <div className="sidebar-foot">
        {!collapsed && <span className="mono muted version">v{state.version}</span>}
        <UpdateButton compact={collapsed} />
      </div>
    </nav>
  );
}

/* ── Command search (after Beautiful UI's Search, MIT) ─────────────────── */
export function CommandPalette() {
  const { state, dispatch } = useStore();
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        dispatch({ type: "palette", open: !state.paletteOpen });
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "n") {
        e.preventDefault();
        dispatch({ type: "newChat" });
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [state.paletteOpen, dispatch]);
  useEffect(() => {
    if (state.paletteOpen) {
      setQ("");
      setSel(0);
      setTimeout(() => input.current?.focus(), 0);
    }
  }, [state.paletteOpen]);
  if (!state.paletteOpen) return null;
  const items: { group: string; label: string; hint?: string; icon: IconName; run: () => void }[] = [
    ...NAV.map((n) => ({ group: "Go to", label: n.label, icon: n.icon, run: () => dispatch({ type: "nav", screen: n.id }) })),
    { group: "Actions", label: "New chat", hint: "Ctrl N", icon: "plus" as IconName, run: () => dispatch({ type: "newChat" }) },
    {
      group: "Actions",
      label: "Toggle theme",
      icon: "sun" as IconName,
      run: () => dispatch({ type: "settings", patch: { theme: state.settings.theme === "light" ? "dark" : "light" } }),
    },
    ...state.chats.map((c) => ({ group: "Chats", label: c.title, icon: "chat" as IconName, run: () => dispatch({ type: "openChat", id: c.id }) })),
  ];
  const filtered = items.filter((i) => i.label.toLowerCase().includes(q.toLowerCase()));
  const close = () => dispatch({ type: "palette", open: false });
  let lastGroup = "";
  return (
    <div className="scrim" onMouseDown={close}>
      <div className="palette" role="dialog" aria-label="Search" onMouseDown={(e) => e.stopPropagation()}>
        <div className="palette-input">
          <Icon name="search" size={16} />
          <input
            ref={input}
            value={q}
            placeholder="Search chats, screens and actions"
            onChange={(e) => {
              setQ(e.target.value);
              setSel(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") close();
              if (e.key === "ArrowDown") setSel((s) => Math.min(filtered.length - 1, s + 1));
              if (e.key === "ArrowUp") setSel((s) => Math.max(0, s - 1));
              if (e.key === "Enter" && filtered[sel]) filtered[sel].run();
            }}
          />
          <Kbd>Esc</Kbd>
        </div>
        <div className="palette-list">
          {filtered.length === 0 && <div className="palette-empty muted">No matches for “{q}”</div>}
          {filtered.map((it, i) => {
            const head = it.group !== lastGroup ? <div className="palette-group">{it.group}</div> : null;
            lastGroup = it.group;
            return (
              <div key={`${it.group}-${it.label}-${i}`}>
                {head}
                <button className={`palette-item ${i === sel ? "sel" : ""}`} onMouseEnter={() => setSel(i)} onClick={it.run}>
                  <Icon name={it.icon} size={15} />
                  <span className="ellipsis">{it.label}</span>
                  {it.hint && <Kbd>{it.hint}</Kbd>}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/** Version this UI was built for (vite.config.ts). */
export const UI_VERSION = __APP_VERSION__;

/**
 * Shown when the desktop app (Python) and this UI build disagree, e.g. an old
 * EigenCore process still serving its files after an update. Never in the
 * browser preview unless asked for with ?mismatch=<app version>.
 */
export function VersionBanner() {
  const { state } = useStore();
  const preview = api.kind !== "pywebview" ? new URLSearchParams(location.search).get("mismatch") : null;
  const appVersion = preview || state.version;
  if (!preview && (api.kind !== "pywebview" || !appVersion || appVersion === UI_VERSION)) return null;
  return (
    <div className="version-banner" role="alert">
      <Icon name="alert" size={14} />
      <span>
        UI build <span className="mono">{UI_VERSION}</span> does not match app <span className="mono">{appVersion}</span>. Close all EigenCore windows and
        restart.
      </span>
    </div>
  );
}

export function Toast() {
  const { state } = useStore();
  if (!state.toast) return null;
  return (
    <div className={`toast ${state.toast.tone ?? ""}`} role="status" key={state.toast.id}>
      <Icon name={state.toast.tone === "err" ? "alert" : "check"} size={14} />
      <span>{state.toast.text}</span>
    </div>
  );
}

/* ── Resize grips for the frameless desktop window ─────────────────────── */
export function ResizeGrips() {
  if (api.kind !== "pywebview") return null;
  const start = (edge: "e" | "s" | "se") => (e: ReactMouseEvent) => {
    e.preventDefault();
    const x0 = e.screenX;
    const y0 = e.screenY;
    const w0 = window.outerWidth;
    const h0 = window.outerHeight;
    let frame = 0;
    let last: [number, number] = [w0, h0];
    const move = (ev: MouseEvent) => {
      last = [edge === "s" ? w0 : w0 + ev.screenX - x0, edge === "e" ? h0 : h0 + ev.screenY - y0];
      if (!frame)
        frame = requestAnimationFrame(() => {
          frame = 0;
          api.windowControl("resize", Math.round(last[0]), Math.round(last[1]));
        });
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };
  return (
    <>
      <div className="resize-grip e" onMouseDown={start("e")} aria-hidden />
      <div className="resize-grip s" onMouseDown={start("s")} aria-hidden />
      <div className="resize-grip se" onMouseDown={start("se")} aria-hidden />
    </>
  );
}
