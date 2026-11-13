import { fromTuple } from "./types";
import type {
  ApprovalDecision,
  BackendEvent,
  EigenCoreApi,
  HardwareProfile,
  ModelRow,
  SendOptions,
  Settings,
  Unsubscribe,
  WindowOp,
} from "./types";

/** Methods of desktop/web.py WebBridge, as pywebview exposes them (all return Promises). */
export interface PyApi {
  ready(): Promise<{ version: string; platform: string }>;
  version(): Promise<string>;
  list_models(): Promise<ModelRow[]>;
  hardware_profile(rescan?: boolean): Promise<HardwareProfile>;
  get_settings(): Promise<Settings>;
  save_settings(patch: Partial<Settings>): Promise<Settings>;
  load_chats(): Promise<unknown[]>;
  save_chats(chats: unknown[]): Promise<boolean>;
  download(name: string): Promise<void>;
  delete(name: string): Promise<boolean>;
  switch_model(name: string): Promise<void>;
  send(text: string, opts: SendOptions): Promise<void>;
  stop(): Promise<void>;
  check_for_update(): Promise<void>;
  install_update(): Promise<void>;
  window_control(op: WindowOp, ...args: number[]): Promise<boolean>;
  open_url(url: string): Promise<void>;
}

declare global {
  interface Window {
    pywebview?: { api: PyApi };
  }
}

/** Name of the CustomEvent the Python host dispatches; detail = list of event tuples. */
export const EVENT_NAME = "eigencore";

/** True when this page runs inside the EigenCore desktop window. */
export function inDesktopHost(): boolean {
  return typeof window.pywebview !== "undefined" || navigator.userAgent.includes("EigenCoreDesktop");
}

/** Resolves once `window.pywebview.api` is usable (pywebview fires `pywebviewready`). */
export function waitForPywebview(timeoutMs = 20000): Promise<PyApi> {
  return new Promise((resolve, reject) => {
    const ok = () => window.pywebview?.api && typeof window.pywebview.api.ready === "function";
    if (ok()) return resolve(window.pywebview!.api);
    const timer = window.setTimeout(() => reject(new Error("pywebview bridge did not become ready")), timeoutMs);
    window.addEventListener(
      "pywebviewready",
      () => {
        window.clearTimeout(timer);
        resolve(window.pywebview!.api);
      },
      { once: true },
    );
  });
}

function report(err: unknown) {
  console.error("[eigencore bridge]", err);
}

/** EigenCoreApi over the pywebview js_api (desktop/web.py WebBridge). */
export class BridgeApi implements EigenCoreApi {
  readonly kind = "pywebview" as const;
  private listeners = new Set<(e: BackendEvent) => void>();

  constructor(private py: PyApi) {
    window.addEventListener(EVENT_NAME, (ev) => {
      const detail = (ev as CustomEvent<unknown[][]>).detail;
      if (!Array.isArray(detail)) return;
      for (const t of detail) {
        const e = Array.isArray(t) ? fromTuple(t) : null;
        if (e) for (const l of this.listeners) l(e);
      }
    });
  }

  /** Tell Python the UI is listening; events start flowing after this. */
  async start() {
    await this.py.ready();
  }

  subscribe(listener: (e: BackendEvent) => void): Unsubscribe {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  version() {
    return this.py.version();
  }
  listModels() {
    return this.py.list_models();
  }
  hardwareProfile(rescan = false) {
    return this.py.hardware_profile(rescan);
  }
  getSettings() {
    return this.py.get_settings();
  }
  saveSettings(patch: Partial<Settings>) {
    return this.py.save_settings(patch);
  }
  loadChats() {
    return this.py.load_chats();
  }
  saveChats(chats: unknown[]) {
    this.py.save_chats(chats).catch(report);
  }
  download(name: string) {
    this.py.download(name).catch(report);
  }
  delete(name: string) {
    this.py.delete(name).catch(report);
  }
  switchModel(name: string) {
    this.py.switch_model(name).catch(report);
  }
  send(text: string, opts: SendOptions = {}) {
    this.py.send(text, opts).catch(report);
  }
  stop() {
    this.py.stop().catch(report);
  }
  respondApproval(_id: string, _decision: ApprovalDecision) {
    // The desktop chat has no tool calls yet, so no approvals are ever asked.
  }
  checkForUpdate() {
    this.py.check_for_update().catch(report);
  }
  installUpdate() {
    this.py.install_update().catch(report);
  }
  windowControl(op: WindowOp, ...args: number[]) {
    this.py.window_control(op, ...args).catch(report);
  }
  openUrl(url: string) {
    this.py.open_url(url).catch(report);
  }
}
