import { CURRENT_VERSION, DEFAULT_SETTINGS, MOCK_FIT_CHECKS, MOCK_MODELS, MOCK_NEXT_TAG, MOCK_PROFILE, MOCK_REPLY, MOCK_SEARCH } from "./mockData";
import type {
  ApprovalDecision,
  BackendEvent,
  EigenCoreApi,
  FitCheck,
  HardwareProfile,
  ModelRow,
  ModelSearch,
  SendOptions,
  Settings,
  Unsubscribe,
} from "./types";

/** Split text into token-ish pieces (word + leading space), keeping <think> tags whole. */
export function tokenize(text: string): string[] {
  return text.match(/<\/?think>|\s*[^\s<]+|\s+|</g) ?? [];
}

/**
 * In-memory stand-in for DesktopBackend. Emits the same event sequence the
 * Python side does, on timers. Every number it reports describes this mock
 * emitter, never a model or the user's hardware.
 */
export class MockApi implements EigenCoreApi {
  readonly kind = "mock" as const;
  private listeners = new Set<(e: BackendEvent) => void>();
  private models: ModelRow[] = MOCK_MODELS.map((m) => ({ ...m }));
  private settings: Settings = { ...DEFAULT_SETTINGS };
  private timers = new Set<number>();
  private genTimer: number | null = null;
  private cancelled = false;
  private pendingApproval: { id: string; text: string } | null = null;
  private approvedForSession = new Set<string>();
  private updateTimer: number | null = null;
  private finish: (() => void) | null = null;

  constructor(private tokenMs = 70) {}

  subscribe(listener: (e: BackendEvent) => void): Unsubscribe {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(e: BackendEvent) {
    for (const l of this.listeners) l(e);
  }

  private later(ms: number, fn: () => void) {
    const id = window.setTimeout(() => {
      this.timers.delete(id);
      fn();
    }, ms);
    this.timers.add(id);
    return id;
  }

  async version() {
    return CURRENT_VERSION;
  }
  async listModels() {
    return this.models.map((m) => ({ ...m }));
  }
  async hardwareProfile(_rescan?: boolean): Promise<HardwareProfile> {
    return { ...MOCK_PROFILE };
  }
  async loadChats(): Promise<unknown[]> {
    return [];
  }
  saveChats(_chats: unknown[]) {}
  async loadUiPrefs(): Promise<Record<string, unknown>> {
    return {};
  }
  saveUiPrefs(_patch: Record<string, unknown>) {}
  windowControl() {}
  openUrl(url: string) {
    window.open(url, "_blank", "noopener");
  }
  async getSettings() {
    return { ...this.settings };
  }
  async saveSettings(patch: Partial<Settings>) {
    this.settings = { ...this.settings, ...patch };
    return { ...this.settings };
  }

  /** SAMPLE results (MOCK_SEARCH), filtered by the words of the query; all of them when nothing matches.
   * Trusted uploaders only unless allUploaders (the "Community uploads" toggle), like the desktop. */
  async searchModels(query: string, allUploaders = false): Promise<ModelSearch> {
    const q = query.trim().toLowerCase();
    if (!q) return { query: q, results: [], error: null, all_uploaders: allUploaders };
    const words = q.split(/\s+/);
    const pool = MOCK_SEARCH.filter((m) => allUploaders || m.trusted);
    const hit = pool.filter((m) => words.every((w) => m.name.toLowerCase().includes(w)));
    const rows = (hit.length ? hit : pool).map((m) => {
      const known = this.models.find((x) => x.name === m.name);
      return { ...(known ?? m) };
    });
    await new Promise((r) => this.later(450, () => r(null)));
    return { query: q, results: rows, error: null, all_uploaders: allUploaders };
  }

  async checkFit(name: string): Promise<FitCheck> {
    await new Promise((r) => this.later(700, () => r(null)));
    const curated = this.models.find((m) => m.name === name);
    if (curated) return { name, fit: curated.fit, fit_detail: curated.fit_detail ?? null, error: null };
    return MOCK_FIT_CHECKS[name] ?? { name, fit: "unknown", fit_detail: null, error: "no sample header for this model" };
  }

  download(name: string) {
    if (!this.models.some((x) => x.name === name)) {
      // a search result: it joins the list, like the desktop's custom-models.json
      const found = MOCK_SEARCH.find((x) => x.name === name);
      if (found) {
        const check = MOCK_FIT_CHECKS[name];
        this.models.push({ ...found, source: "custom", fit: check?.fit ?? found.fit, fit_detail: check?.fit_detail ?? found.fit_detail });
      }
    }
    const m = this.models.find((x) => x.name === name);
    if (!m) return this.emit({ type: "dl_done", name, ok: false, message: `unknown model: ${name}` });
    if (m.downloaded) return this.emit({ type: "dl_done", name, ok: true, message: `${name} already downloaded` });
    let pct = 0;
    const step = Math.min(6, Math.max(0.6, 8 / m.size_gb));
    const tick = () => {
      pct = Math.min(100, pct + step * (0.6 + Math.random() * 0.8));
      this.emit({ type: "dl_progress", name, pct });
      if (pct < 100) this.later(220, tick);
      else {
        m.downloaded = true;
        this.emit({ type: "dl_done", name, ok: true, message: `downloaded ${name}` });
      }
    };
    this.later(150, tick);
  }

  delete(name: string) {
    const m = this.models.find((x) => x.name === name);
    if (!m || !m.downloaded || m.is_active) return this.emit({ type: "deleted", name, ok: false });
    m.downloaded = false;
    if (m.source === "custom") this.models = this.models.filter((x) => x !== m);
    this.emit({ type: "deleted", name, ok: true });
  }

  switchModel(name: string) {
    const m = this.models.find((x) => x.name === name);
    if (!m || !m.downloaded) return this.emit({ type: "engine", state: "error", message: `${name} is not downloaded` });
    this.emit({ type: "engine", state: "loading", message: `loading ${name}…` });
    this.later(1100, () => {
      this.models.forEach((x) => (x.is_active = x.name === name));
      this.emit({ type: "engine", state: "ready", message: `${name} ready` });
      this.emit({ type: "switched", name });
    });
  }

  send(text: string, _opts?: SendOptions) {
    text = text.trim();
    if (!text) return;
    if (!this.models.some((m) => m.is_active)) {
      return this.emit({ type: "engine", state: "error", message: "no model loaded — download one in Models and select it" });
    }
    this.cancelled = false;
    // Mock agent path: mentioning a file asks for a tool approval first.
    if (/\b(file|read|notes\.txt)\b/i.test(text) && !this.approvedForSession.has("read_file")) {
      const id = `apr-${Date.now()}`;
      this.pendingApproval = { id, text };
      this.later(500, () =>
        this.emit({
          type: "approval_request",
          request: {
            id,
            tool_name: "read_file",
            capability: "READ",
            args: { path: "~/Documents/notes.txt", api_key: "***REDACTED***" },
            reason: "Read the file you mentioned to answer the question.",
          },
        }),
      );
      return;
    }
    this.stream(MOCK_REPLY);
  }

  respondApproval(id: string, decision: ApprovalDecision) {
    if (this.pendingApproval && this.pendingApproval.id !== id) return;
    this.pendingApproval = null;
    if (decision === "deny") {
      this.emit({ type: "tool_call", tool_name: "read_file", summary: "Denied by you", ok: false });
      this.stream("I didn't read the file. Paste the text here and I can work with it directly.");
      return;
    }
    if (decision === "session") this.approvedForSession.add("read_file");
    this.emit({ type: "tool_call", tool_name: "read_file", summary: "~/Documents/notes.txt · 2.1 KB", ok: true });
    this.stream(MOCK_REPLY);
  }

  private stream(reply: string) {
    const toks = tokenize(reply);
    const start = performance.now();
    let firstAt: number | null = null;
    let markT = 0;
    let markN = 0;
    let i = 0;
    this.emit({ type: "prompt_eval", started: true });
    const step = () => {
      if (this.cancelled || i >= toks.length) {
        this.finish = null;
        const seconds = (performance.now() - start) / 1000;
        this.genTimer = null;
        this.emit({
          type: "done",
          stats: {
            tokens: i,
            seconds,
            tok_per_sec: seconds > 0 ? i / seconds : 0,
            interrupted: this.cancelled,
            error: null,
            first_token_s: firstAt === null ? null : (firstAt - start) / 1000,
            decode_tok_per_sec: firstAt !== null && i > 1 ? ((i - 1) * 1000) / (performance.now() - firstAt) : null,
          },
        });
        return;
      }
      const now = performance.now();
      if (firstAt === null) {
        firstAt = now;
        markT = now;
        markN = 1;
        this.emit({ type: "prompt_eval", started: false });
      }
      this.emit({ type: "tok", text: toks[i++] });
      // Live speed samples, like the backend's ("tps", …) events (mock emitter timing).
      if (now - markT >= 500 && i > markN) {
        this.emit({ type: "tps", sample: { tok_per_sec: ((i - markN) * 1000) / (now - markT), tokens: i, t: (now - firstAt) / 1000 } });
        markT = now;
        markN = i;
      }
      this.genTimer = this.later(this.tokenMs * (0.6 + Math.random() * 0.8), step);
    };
    this.finish = step;
    this.genTimer = this.later(700, step);
  }

  stop() {
    this.cancelled = true;
    if (this.genTimer !== null) {
      // Finish now instead of waiting for the next token tick.
      window.clearTimeout(this.genTimer);
      this.timers.delete(this.genTimer);
      this.genTimer = this.later(0, this.finish ?? (() => {}));
    }
    if (this.pendingApproval) {
      this.pendingApproval = null;
    }
  }

  checkForUpdate() {
    this.emit({ type: "update", state: { kind: "checking" } });
    this.later(900, () =>
      this.emit({
        type: "update",
        state: { kind: "available", tag: MOCK_NEXT_TAG, url: "https://github.com/Khubaib7-del/EigenCore/releases/latest" },
      }),
    );
  }

  installUpdate() {
    if (this.updateTimer) return;
    let pct = 0;
    const tick = () => {
      pct = Math.min(100, pct + 7);
      if (pct < 100) {
        this.emit({ type: "update", state: { kind: "downloading", tag: MOCK_NEXT_TAG, pct } });
        this.updateTimer = this.later(160, tick);
      } else {
        this.updateTimer = null;
        this.emit({ type: "update", state: { kind: "ready_to_install", tag: MOCK_NEXT_TAG } });
      }
    };
    tick();
  }
}
