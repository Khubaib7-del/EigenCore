/**
 * Types shared by the UI and the (future) Python bridge.
 *
 * The event union mirrors the tuples DesktopBackend puts on its `events`
 * queue in src/eigencore/desktop/backend.py:
 *
 *   ("dl_progress", name, pct)       download progress 0..100
 *   ("dl_done", name, ok, message)   download finished/failed
 *   ("engine", state, message)       "loading" / "ready" / "error"
 *   ("tok", text)                    streamed token
 *   ("done", stats)                  tokens, seconds, tok_per_sec, interrupted, error
 *   ("deleted", name, ok)
 *   ("switched", name)
 *
 *   ("prompt_eval", started)         True while the prompt is read, False at the first token
 *   ("update", state)                updater state (desktop/web.py), see UpdateState
 *
 * Anything marked PROPOSED does not exist in the backend yet; the UI is
 * ready for it and the mock emits it so the screens can be designed. The
 * desktop bridge (bridge.ts) never sends PROPOSED events.
 */

/** FitVerdict values from src/eigencore/fit/estimator.py */
export type FitVerdict = "fits_with_margin" | "tight" | "does_not_fit" | "unknown";

export type Provider = "tinyllama" | "microsoft" | "qwen" | "meta" | "mistral" | "gemma" | "deepseek" | "ai2" | "huggingface";

/** Mirrors desktop.backend.ModelRow (+ repo_id from ModelSpec, + UI-derived provider). */
export interface ModelRow {
  name: string;
  params_b: number;
  /** MoE only: active parameters per token, from the registry description. */
  active_params_b?: number;
  quant: string;
  size_gb: number;
  task: "general" | "code";
  context_length: number;
  description: string;
  downloaded: boolean;
  fit: FitVerdict;
  is_active: boolean;
  repo_id: string;
  provider: Provider;
}

export type EngineBackendState = "loading" | "ready" | "error";

/**
 * Payload of ("done", stats). `tokens` counts streamed pieces (one per token
 * for llama.cpp in practice). tok_per_sec is end to end (prompt + output);
 * decode_tok_per_sec is the output rate after the first token.
 * prompt_tokens is PROPOSED (not sent by the backend yet).
 */
export interface GenerationStats {
  tokens: number;
  seconds: number;
  tok_per_sec: number;
  interrupted: boolean;
  error: string | null;
  first_token_s?: number | null;
  decode_tok_per_sec?: number | null;
  prompt_tokens?: number | null;
}

/** Mirrors eigencore.agent.policy.ApprovalRequest (args already redacted). */
export interface ApprovalRequest {
  id: string;
  tool_name: string;
  capability: "READ" | "WRITE" | "EXECUTE" | "NETWORK";
  args: Record<string, string | number | boolean>;
  reason: string;
}

/** ApprovalMode: ASK (allow once) / SESSION (always, this session) / DENY. */
export type ApprovalDecision = "once" | "session" | "deny";

export type UpdateState =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "up_to_date"; current: string }
  | { kind: "available"; tag: string; url: string }
  | { kind: "downloading"; tag: string; pct: number }
  | { kind: "ready_to_install"; tag: string }
  /** Portable or pip copy: the release page was opened instead of updating in place. */
  | { kind: "manual"; tag: string; url: string; message: string }
  | { kind: "error"; message: string };

export type BackendEvent =
  | { type: "dl_progress"; name: string; pct: number }
  | { type: "dl_done"; name: string; ok: boolean; message: string }
  | { type: "engine"; state: EngineBackendState; message: string }
  | { type: "tok"; text: string }
  | { type: "done"; stats: GenerationStats }
  | { type: "deleted"; name: string; ok: boolean }
  | { type: "switched"; name: string }
  | { type: "prompt_eval"; started: boolean }
  | { type: "update"; state: UpdateState }
  /* PROPOSED events (mock only: the desktop chat has no tool calls yet) */
  | { type: "approval_request"; request: ApprovalRequest }
  | { type: "tool_call"; tool_name: string; summary: string; ok: boolean };

/** Mirrors desktop HardwareProfile (eigencore.hal.profiler). Estimates may be null in the UI until known. */
export interface HardwareProfile {
  cpu_name: string;
  arch: string;
  physical_cores: number;
  logical_cores: number;
  instruction_sets: string[];
  total_ram_gb: number;
  available_ram_gb: number;
  /** Profiler estimate, not a measurement. null = not shown in the preview. */
  estimated_bandwidth_gbps: number | null;
  max_model_params_b: number;
  optimal_quantization: string;
  recommended_threads: number;
  recommended_context_length: number;
  gpu_name: string | null;
  benchmarked: boolean;
}

export interface Settings {
  theme: "dark" | "light" | "system";
  models_dir: string;
  threads: number | "auto";
  context_length: number;
  kv_cache_type: "f16" | "q8_0";
  approvals: "ask" | "session";
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface SendOptions {
  max_tokens?: number;
  temperature?: number;
  /** Earlier turns of this chat, so the engine sees the whole conversation. */
  history?: ChatTurn[];
}

export type WindowOp = "minimize" | "toggle_maximize" | "close" | "resize";

export type Unsubscribe = () => void;

/**
 * The surface the UI talks to. MockApi (browser preview) and BridgeApi
 * (desktop app: `window.pywebview.api.*`, events turned into BackendEvent
 * with `fromTuple`) implement it.
 */
export interface EigenCoreApi {
  readonly kind: "mock" | "pywebview";
  version(): Promise<string>;
  listModels(): Promise<ModelRow[]>;
  /** rescan=true runs the hardware profiler again. */
  hardwareProfile(rescan?: boolean): Promise<HardwareProfile>;
  /** Saved chats (desktop: ~/.eigencore/desktop-ui.json; preview: none). */
  loadChats(): Promise<unknown[]>;
  saveChats(chats: unknown[]): void;
  windowControl(op: WindowOp, ...args: number[]): void;
  openUrl(url: string): void;
  getSettings(): Promise<Settings>;
  saveSettings(patch: Partial<Settings>): Promise<Settings>;
  download(name: string): void;
  delete(name: string): void;
  switchModel(name: string): void;
  send(text: string, opts?: SendOptions): void;
  stop(): void;
  respondApproval(id: string, decision: ApprovalDecision): void;
  checkForUpdate(): void;
  installUpdate(): void;
  subscribe(listener: (e: BackendEvent) => void): Unsubscribe;
}

/** Convert a Python event tuple (as JSON array) into a typed event. */
export function fromTuple(t: unknown[]): BackendEvent | null {
  const [kind, a, b, c] = t;
  switch (kind) {
    case "dl_progress":
      return { type: "dl_progress", name: String(a), pct: Number(b) };
    case "dl_done":
      return { type: "dl_done", name: String(a), ok: Boolean(b), message: String(c ?? "") };
    case "engine":
      return { type: "engine", state: a as EngineBackendState, message: String(b ?? "") };
    case "tok":
      return { type: "tok", text: String(a) };
    case "done":
      return { type: "done", stats: a as GenerationStats };
    case "deleted":
      return { type: "deleted", name: String(a), ok: Boolean(b) };
    case "switched":
      return { type: "switched", name: String(a) };
    case "prompt_eval":
      return { type: "prompt_eval", started: Boolean(a) };
    case "update":
      return { type: "update", state: a as UpdateState };
    default:
      return null;
  }
}
