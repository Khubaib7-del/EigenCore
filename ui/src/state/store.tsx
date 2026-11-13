import { createContext, useContext, useEffect, useMemo, useReducer, useRef, type Dispatch, type ReactNode } from "react";
import { api } from "../api";
import type {
  ApprovalDecision,
  ApprovalRequest,
  BackendEvent,
  ChatTurn,
  GenerationStats,
  HardwareProfile,
  ModelRow,
  Settings,
  UpdateState,
} from "../api";
import { CURRENT_VERSION, DEFAULT_SETTINGS, MOCK_MODELS, MOCK_PROFILE } from "../api/mockData";
import { splitThinking } from "../components/Markdown";
import { SEED_CHATS, SEED_TPS, SEED_TURNS } from "./seed";

export type Screen = "chat" | "models" | "monitor" | "profile" | "settings";

export interface ToolCall {
  tool_name: string;
  summary: string;
  ok: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  /** Raw text, including any <think>…</think> block. */
  content: string;
  model?: string;
  status?: "reading" | "streaming" | "done" | "stopped" | "error" | "awaiting_approval";
  stats?: GenerationStats | null;
  /** placeholder = nothing measured; mock = timings of the mock emitter. */
  statsKind?: "placeholder" | "mock" | "measured";
  approval?: ApprovalRequest & { decision?: ApprovalDecision };
  tools?: ToolCall[];
  startedAt?: number;
  error?: string;
}

export interface Chat {
  id: string;
  title: string;
  group: string;
  model: string;
  messages: Message[];
  /** ms since epoch; the sidebar group (Today, Yesterday...) is derived from it. */
  createdAt?: number;
}

export interface Turn {
  id: string;
  at: string;
  model: string;
  prompt_tokens: number | null;
  tokens: number;
  tok_per_sec: number;
  /** Output rate after the first token (desktop backend); null when not measured. */
  decode_tok_per_sec?: number | null;
  first_token_s: number | null;
  seconds: number;
  status: "done" | "stopped" | "error";
}

export type EnginePhase = "idle" | "loading" | "reading" | "writing" | "error";

export interface State {
  screen: Screen;
  sidebarCollapsed: boolean;
  settings: Settings;
  systemDark: boolean;
  models: ModelRow[];
  downloads: Record<string, number>;
  profile: HardwareProfile;
  version: string;
  engine: { phase: EnginePhase; message: string };
  activeModel: string | null;
  chats: Chat[];
  activeChatId: string | null;
  turns: Turn[];
  tps: number[];
  update: UpdateState;
  toast: { id: number; text: string; tone?: "ok" | "err" } | null;
  paletteOpen: boolean;
  /** Monitor shows labelled sample telemetry (preview only). */
  sampleTelemetry: boolean;
}

type Action =
  | { type: "event"; e: BackendEvent }
  | { type: "nav"; screen: Screen }
  | { type: "toggleSidebar" }
  | { type: "settings"; patch: Partial<Settings> }
  | { type: "systemDark"; dark: boolean }
  | { type: "newChat" }
  | { type: "openChat"; id: string }
  | { type: "userSend"; text: string }
  | { type: "approval"; msgId: string; decision: ApprovalDecision }
  | { type: "toast"; text: string; tone?: "ok" | "err" }
  | { type: "clearToast" }
  | { type: "palette"; open: boolean }
  | { type: "seed"; patch: Partial<State> };

const uid = () => Math.random().toString(36).slice(2, 10);
const nowHHMM = () => new Date().toTimeString().slice(0, 5);

function patchActiveChat(s: State, fn: (c: Chat) => Chat): State {
  return { ...s, chats: s.chats.map((c) => (c.id === s.activeChatId ? fn(c) : c)) };
}
function patchLastAssistant(s: State, fn: (m: Message) => Message): State {
  return patchActiveChat(s, (c) => {
    const idx = [...c.messages].reverse().findIndex((m) => m.role === "assistant");
    if (idx < 0) return c;
    const i = c.messages.length - 1 - idx;
    const messages = c.messages.slice();
    messages[i] = fn(messages[i]);
    return { ...c, messages };
  });
}

function onEvent(s: State, e: BackendEvent): State {
  switch (e.type) {
    case "dl_progress":
      return { ...s, downloads: { ...s.downloads, [e.name]: e.pct } };
    case "dl_done": {
      const { [e.name]: _done, ...rest } = s.downloads;
      void _done;
      return { ...s, downloads: rest, toast: { id: Date.now(), text: e.message, tone: e.ok ? "ok" : "err" } };
    }
    case "deleted":
      return {
        ...s,
        toast: { id: Date.now(), text: e.ok ? `Deleted ${e.name}` : `Could not delete ${e.name}`, tone: e.ok ? "ok" : "err" },
      };
    case "engine": {
      if (e.state === "loading") return { ...s, engine: { phase: "loading", message: e.message } };
      if (e.state === "ready") return { ...s, engine: { phase: "idle", message: e.message } };
      const err = { ...s, engine: { phase: "error" as const, message: e.message }, toast: { id: Date.now(), text: e.message, tone: "err" as const } };
      // A send that never started (no model loaded) must not spin forever.
      return patchLastAssistant(err, (m) => (m.status === "reading" && !m.content ? { ...m, status: "error", error: e.message } : m));
    }
    case "switched":
      return {
        ...s,
        activeModel: e.name,
        models: s.models.map((m) => ({ ...m, is_active: m.name === e.name })),
      };
    case "prompt_eval":
      if (e.started) {
        return patchLastAssistant({ ...s, engine: { phase: "reading", message: "Reading prompt" } }, (m) => ({ ...m, status: "reading" }));
      }
      return patchLastAssistant({ ...s, engine: { phase: "writing", message: "Writing" } }, (m) => ({ ...m, status: "streaming" }));
    case "tok":
      return patchLastAssistant(s, (m) => ({ ...m, content: m.content + e.text, status: "streaming" }));
    case "done": {
      const st = e.stats;
      const next = patchLastAssistant({ ...s, engine: { phase: "idle", message: "" } }, (m) => ({
        ...m,
        status: st.error ? "error" : st.interrupted ? "stopped" : "done",
        stats: st,
        statsKind: api.kind === "mock" ? "mock" : "measured",
      }));
      const turn: Turn = {
        id: uid(),
        at: nowHHMM(),
        model: s.activeModel ?? "—",
        prompt_tokens: st.prompt_tokens ?? null,
        tokens: st.tokens,
        tok_per_sec: st.tok_per_sec,
        decode_tok_per_sec: st.decode_tok_per_sec ?? null,
        first_token_s: st.first_token_s ?? null,
        seconds: st.seconds,
        status: st.error ? "error" : st.interrupted ? "stopped" : "done",
      };
      const rate = st.decode_tok_per_sec ?? st.tok_per_sec;
      return { ...next, turns: [turn, ...s.turns].slice(0, 20), tps: [...s.tps, rate].slice(-48) };
    }
    case "approval_request":
      return patchLastAssistant(s, (m) => ({ ...m, status: "awaiting_approval", approval: e.request }));
    case "tool_call":
      return patchLastAssistant(s, (m) => ({ ...m, tools: [...(m.tools ?? []), { tool_name: e.tool_name, summary: e.summary, ok: e.ok }] }));
    case "update":
      return { ...s, update: e.state };
  }
}

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case "event":
      return onEvent(s, a.e);
    case "nav":
      return { ...s, screen: a.screen, paletteOpen: false };
    case "toggleSidebar":
      return { ...s, sidebarCollapsed: !s.sidebarCollapsed };
    case "settings":
      return { ...s, settings: { ...s.settings, ...a.patch } };
    case "systemDark":
      return { ...s, systemDark: a.dark };
    case "newChat": {
      const c: Chat = { id: uid(), title: "New chat", group: "Today", model: s.activeModel ?? "", messages: [], createdAt: Date.now() };
      return { ...s, chats: [c, ...s.chats], activeChatId: c.id, screen: "chat", paletteOpen: false };
    }
    case "openChat":
      return { ...s, activeChatId: a.id, screen: "chat", paletteOpen: false };
    case "userSend": {
      let st = s;
      if (!st.activeChatId) st = reducer(st, { type: "newChat" });
      return patchActiveChat(st, (c) => ({
        ...c,
        title: c.messages.length === 0 ? a.text.slice(0, 48) : c.title,
        messages: [
          ...c.messages,
          { id: uid(), role: "user", content: a.text },
          { id: uid(), role: "assistant", content: "", model: st.activeModel ?? undefined, status: "reading", startedAt: Date.now() },
        ],
      }));
    }
    case "approval":
      return patchActiveChat(s, (c) => ({
        ...c,
        messages: c.messages.map((m) =>
          m.id === a.msgId && m.approval ? { ...m, approval: { ...m.approval, decision: a.decision }, status: "reading" } : m,
        ),
      }));
    case "toast":
      return { ...s, toast: { id: Date.now(), text: a.text, tone: a.tone } };
    case "clearToast":
      return { ...s, toast: null };
    case "palette":
      return { ...s, paletteOpen: a.open };
    case "seed":
      return { ...s, ...a.patch };
  }
}

/** Real data the desktop app loads before the first render (see main.tsx). */
export interface Boot {
  models: ModelRow[];
  profile: HardwareProfile;
  version: string;
  settings: Settings;
  chats: Chat[];
}

const THEME_KEY = "eigencore.theme";
function storedTheme(): Settings["theme"] | null {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === "dark" || t === "light" || t === "system" ? t : null;
  } catch {
    return null;
  }
}

function groupFor(createdAt: number | undefined): string {
  if (!createdAt) return "Earlier";
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const diff = Math.round((day(new Date()) - day(new Date(createdAt))) / 86400000);
  return diff <= 0 ? "Today" : diff === 1 ? "Yesterday" : diff < 7 ? "This week" : "Earlier";
}

/** Saved chats come back settled: nothing is streaming after a restart. */
export function restoreChats(raw: unknown[]): Chat[] {
  const out: Chat[] = [];
  for (const c of raw) {
    if (!c || typeof c !== "object") continue;
    const chat = c as Chat;
    if (typeof chat.id !== "string" || !Array.isArray(chat.messages)) continue;
    out.push({
      ...chat,
      group: groupFor(chat.createdAt),
      messages: chat.messages.map((m) =>
        m.status === "reading" || m.status === "streaming" || m.status === "awaiting_approval" ? { ...m, status: "stopped" } : m,
      ),
    });
  }
  return out;
}

/** Earlier turns of a chat as the engine should see them (reasoning stripped). */
export function historyOf(chat: Chat | undefined): ChatTurn[] {
  if (!chat) return [];
  const turns: ChatTurn[] = [];
  for (const m of chat.messages) {
    if (m.role === "user") turns.push({ role: "user", content: m.content });
    else if (m.status !== "error") {
      const answer = splitThinking(m.content).answer.trim();
      if (answer) turns.push({ role: "assistant", content: answer });
    }
  }
  return turns;
}

function desktopState(boot: Boot): State {
  return {
    screen: "chat",
    sidebarCollapsed: false,
    settings: boot.settings,
    systemDark: window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true,
    models: boot.models,
    downloads: {},
    profile: boot.profile,
    version: boot.version,
    engine: { phase: "idle", message: "" },
    activeModel: boot.models.find((m) => m.is_active)?.name ?? null,
    chats: boot.chats,
    activeChatId: null,
    turns: [],
    tps: [],
    sampleTelemetry: false,
    update: { kind: "idle" },
    toast: null,
    paletteOpen: false,
  };
}

/** Preview switches read from the URL, e.g. ?screen=models&theme=light&chat=empty */
function initialState(boot?: Boot): State {
  if (boot) return desktopState(boot);
  const q = new URLSearchParams(location.search);
  const screen = (q.get("screen") as Screen) || "chat";
  const theme = (q.get("theme") as Settings["theme"]) || storedTheme() || DEFAULT_SETTINGS.theme;
  const chats = SEED_CHATS.map((c) => ({ ...c, messages: c.messages.slice() }));
  const empty = q.get("chat") === "empty";
  const s: State = {
    screen,
    sidebarCollapsed: q.get("collapsed") === "1",
    settings: { ...DEFAULT_SETTINGS, theme },
    systemDark: window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true,
    models: MOCK_MODELS.map((m) => ({ ...m })),
    downloads: q.get("downloads") === "0" ? {} : { "phi-3-mini": 42 },
    profile: { ...MOCK_PROFILE },
    version: CURRENT_VERSION,
    engine: { phase: "idle", message: "qwen2.5-1.5b ready" },
    activeModel: "qwen2.5-1.5b",
    chats: empty ? [{ id: "new", title: "New chat", group: "Today", model: "qwen2.5-1.5b", messages: [] }, ...chats] : chats,
    activeChatId: empty ? "new" : "c1",
    // Sample telemetry only when asked for (?sample=1); otherwise Monitor starts empty.
    turns: q.get("sample") === "1" ? SEED_TURNS : [],
    tps: q.get("sample") === "1" ? SEED_TPS : [],
    sampleTelemetry: q.get("sample") === "1",
    update: { kind: "idle" },
    toast: null,
    paletteOpen: q.get("palette") === "1",
  };
  const up = q.get("update");
  if (up === "available") s.update = { kind: "available", tag: "v0.4.6", url: "" };
  if (up === "downloading") s.update = { kind: "downloading", tag: "v0.4.6", pct: 64 };
  if (up === "checking") s.update = { kind: "checking" };
  if (q.get("engine") === "writing") s.engine = { phase: "writing", message: "Writing" };
  return s;
}

interface Ctx {
  state: State;
  dispatch: Dispatch<Action>;
  actions: {
    send(text: string): void;
    stop(): void;
    download(name: string): void;
    remove(name: string): void;
    use(name: string): void;
    approve(msgId: string, reqId: string, d: ApprovalDecision): void;
    checkUpdate(): void;
    installUpdate(): void;
    rescanProfile(): void;
  };
}

const StoreCtx = createContext<Ctx | null>(null);

export function StoreProvider({ children, boot }: { children: ReactNode; boot?: Boot }) {
  const [state, dispatch] = useReducer(reducer, boot, initialState);
  const stateRef = useRef(state);
  stateRef.current = state;
  const desktop = api.kind === "pywebview";

  // Desktop: look for an update once at start (like the previous app did).
  useEffect(() => {
    if (desktop) api.checkForUpdate();
  }, [desktop]);

  // Theme is remembered: localStorage, and the desktop host's settings file.
  const theme = state.settings.theme;
  const firstTheme = useRef(true);
  useEffect(() => {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* storage disabled */
    }
    if (firstTheme.current) {
      firstTheme.current = false;
      return;
    }
    if (desktop) api.saveSettings({ theme }).catch(() => {});
  }, [theme, desktop]);

  // Desktop: chats are saved locally, a moment after they stop changing.
  const busy = state.engine.phase === "reading" || state.engine.phase === "writing";
  useEffect(() => {
    if (!desktop || busy) return;
    const t = setTimeout(() => api.saveChats(stateRef.current.chats.filter((c) => c.messages.length > 0)), 600);
    return () => clearTimeout(t);
  }, [state.chats, busy, desktop]);

  useEffect(
    () =>
      api.subscribe((e) => {
        dispatch({ type: "event", e });
        if (e.type === "dl_done" || e.type === "deleted" || e.type === "switched") {
          api.listModels().then((models) => dispatch({ type: "seed", patch: { models } }));
        }
      }),
    [],
  );
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const fn = () => dispatch({ type: "systemDark", dark: mq.matches });
    mq.addEventListener("change", fn);
    return () => mq.removeEventListener("change", fn);
  }, []);
  useEffect(() => {
    if (!state.toast) return;
    const t = setTimeout(() => dispatch({ type: "clearToast" }), 3200);
    return () => clearTimeout(t);
  }, [state.toast]);

  const actions = useMemo<Ctx["actions"]>(
    () => ({
      send(text) {
        const s = stateRef.current;
        const history = historyOf(s.chats.find((c) => c.id === s.activeChatId));
        dispatch({ type: "userSend", text });
        api.send(text, { history });
      },
      stop: () => api.stop(),
      download: (n) => {
        dispatch({ type: "event", e: { type: "dl_progress", name: n, pct: 0 } });
        api.download(n);
      },
      remove: (n) => api.delete(n),
      use: (n) => api.switchModel(n),
      approve(msgId, reqId, d) {
        dispatch({ type: "approval", msgId, decision: d });
        api.respondApproval(reqId, d);
      },
      checkUpdate: () => api.checkForUpdate(),
      installUpdate: () => api.installUpdate(),
      rescanProfile() {
        api
          .hardwareProfile(true)
          .then((profile) => {
            dispatch({ type: "seed", patch: { profile } });
            dispatch({ type: "toast", text: "Hardware profile updated", tone: "ok" });
          })
          .catch(() => dispatch({ type: "toast", text: "Could not re-scan this machine", tone: "err" }));
      },
    }),
    [],
  );

  return <StoreCtx.Provider value={{ state, dispatch, actions }}>{children}</StoreCtx.Provider>;
}

export function useStore() {
  const c = useContext(StoreCtx);
  if (!c) throw new Error("useStore outside StoreProvider");
  return c;
}

export function resolvedTheme(s: State): "dark" | "light" {
  if (s.settings.theme === "system") return s.systemDark ? "dark" : "light";
  return s.settings.theme;
}
