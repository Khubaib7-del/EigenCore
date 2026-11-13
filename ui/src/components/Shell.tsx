import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import { api } from "../api";
import { useStore, type Screen } from "../state/store";
import { Icon, type IconName } from "./Icon";
import { GlideMenu, IconButton, Kbd, PixelLoader } from "./primitives";

/* ── Title bar for a frameless window ───────────────────────────────────
 * The whole bar is a drag region (pywebview: class "pywebview-drag-region";
 * -webkit-app-region for WebView2/Chromium). Buttons opt out of dragging.
 */
export function TitleBar() {
  const { state } = useStore();
  const win = (op: "minimize" | "toggle_maximize" | "close") => api.windowControl(op);
  const onDouble = (e: ReactMouseEvent) => {
    if (!(e.target as HTMLElement).closest("button")) win("toggle_maximize");
  };
  const engine = state.engine;
  const status =
    engine.phase === "loading" ? "Loading model" : engine.phase === "reading" ? "Reading prompt" : engine.phase === "writing" ? "Writing" : engine.phase === "error" ? "Engine error" : state.activeModel ? "Ready" : "No model";
  return (
    <header className="titlebar pywebview-drag-region" onDoubleClick={onDouble}>
      <div className="titlebar-brand">
        <span className="wordmark">EigenCore</span>
        <span className="titlebar-sep" />
        <span className={`engine-dot ${engine.phase}`} aria-hidden />
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

const NAV: { id: Screen; label: string; icon: IconName }[] = [
  { id: "chat", label: "Chat", icon: "chat" },
  { id: "models", label: "Models", icon: "models" },
  { id: "monitor", label: "Monitor", icon: "monitor" },
  { id: "profile", label: "Profile", icon: "cpu" },
  { id: "settings", label: "Settings", icon: "settings" },
];

export function UpdateButton({ compact }: { compact?: boolean }) {
  const { state, actions } = useStore();
  const u = state.update;
  let label = "Check for updates";
  let onClick: (() => void) | undefined = actions.checkUpdate;
  let progress: number | null = null;
  if (u.kind === "checking") {
    label = "Checking…";
    onClick = undefined;
  } else if (u.kind === "available") {
    label = `Update to ${u.tag}`;
    onClick = actions.installUpdate;
  } else if (u.kind === "downloading") {
    label = `Downloading ${Math.round(u.pct)}%`;
    progress = u.pct;
    onClick = undefined;
  } else if (u.kind === "ready_to_install") {
    label = "Installing…";
    onClick = undefined;
  } else if (u.kind === "manual") {
    label = `Get ${u.tag}`;
    onClick = () => api.openUrl(u.url);
  } else if (u.kind === "up_to_date") label = "Up to date";
  else if (u.kind === "error") label = "Retry update check";
  if (compact)
    return (
      <IconButton
        icon={u.kind === "available" ? "download" : "refresh"}
        label={label}
        onClick={onClick}
        className={u.kind === "available" ? "has-update" : ""}
      />
    );
  return (
    <button className={`update-btn ${u.kind}`} onClick={onClick} disabled={!onClick}>
      {u.kind === "checking" ? <PixelLoader variant="orbit" /> : <Icon name={u.kind === "available" ? "download" : "refresh"} size={13} />}
      <span>{label}</span>
      {progress !== null && <span className="update-progress" style={{ width: `${progress}%` }} />}
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
  return (
    <nav className={`sidebar ${collapsed ? "collapsed" : ""}`} aria-label="Main">
      <div className="sidebar-top">
        <button className="new-chat" onClick={() => dispatch({ type: "newChat" })} title="New chat">
          <Icon name="plus" size={15} />
          {!collapsed && <span>New chat</span>}
          {!collapsed && <Kbd>Ctrl N</Kbd>}
        </button>
        <IconButton icon="sidebar" label={collapsed ? "Expand sidebar" : "Collapse sidebar"} onClick={() => dispatch({ type: "toggleSidebar" })} />
      </div>
      {!collapsed && (
        <button className="search-trigger" onClick={() => dispatch({ type: "palette", open: true })}>
          <Icon name="search" size={14} />
          <span>Search</span>
          <Kbd>Ctrl K</Kbd>
        </button>
      )}
      <GlideMenu className="nav-list">
        {NAV.map((n) => (
          <button
            key={n.id}
            data-glide
            className={`nav-item ${state.screen === n.id ? "active" : ""}`}
            aria-current={state.screen === n.id ? "page" : undefined}
            onClick={() => dispatch({ type: "nav", screen: n.id })}
            title={collapsed ? n.label : undefined}
          >
            <Icon name={n.icon} size={16} />
            {!collapsed && <span>{n.label}</span>}
            {!collapsed && n.id === "monitor" && state.engine.phase !== "idle" && state.engine.phase !== "error" && (
              <span className="nav-live">
                <PixelLoader />
              </span>
            )}
          </button>
        ))}
      </GlideMenu>
      {!collapsed && (
        <div className="history">
          <GlideMenu>
            {groups.map(([g, chats]) => (
              <div key={g} className="history-group">
                <div className="history-label">{g}</div>
                {chats.map((c) => (
                  <button
                    key={c.id}
                    data-glide
                    className={`history-item ${c.id === state.activeChatId && state.screen === "chat" ? "active" : ""}`}
                    onClick={() => dispatch({ type: "openChat", id: c.id })}
                  >
                    <span className="ellipsis">{c.title}</span>
                  </button>
                ))}
              </div>
            ))}
          </GlideMenu>
        </div>
      )}
      {collapsed && <span className="grow" />}
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
