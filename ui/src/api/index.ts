import { BridgeApi, inDesktopHost, waitForPywebview } from "./bridge";
import { MockApi } from "./mock";
import type { EigenCoreApi } from "./types";

export * from "./types";

/**
 * The single API instance: BridgeApi inside the desktop app (pywebview),
 * MockApi in a plain browser (`npm run dev`, design preview). Set once by
 * initApi() before the app renders.
 */
export let api: EigenCoreApi = new MockApi();

export async function initApi(): Promise<EigenCoreApi> {
  if (!inDesktopHost()) return api;
  try {
    const bridge = new BridgeApi(await waitForPywebview());
    await bridge.start();
    api = bridge;
  } catch (err) {
    console.error("[eigencore] desktop bridge unavailable, using the mock API", err);
  }
  return api;
}
