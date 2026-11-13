import { MockApi } from "./mock";
import type { EigenCoreApi } from "./types";

export * from "./types";

/**
 * The single API instance. Today it is always the mock. When the pywebview
 * bridge lands, this picks `PyWebviewApi` if `window.pywebview` exists.
 */
export const api: EigenCoreApi = new MockApi();
