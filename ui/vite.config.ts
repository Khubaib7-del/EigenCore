import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** The app version this UI is built for, from src/eigencore/__init__.py. */
function appVersion(): string {
  const src = readFileSync(new URL("../src/eigencore/__init__.py", import.meta.url), "utf8");
  const m = /__version__\s*=\s*"([^"]+)"/.exec(src);
  if (!m) throw new Error("__version__ not found in src/eigencore/__init__.py");
  return m[1];
}

// Relative base so the built files load from file:// inside pywebview later.
// __APP_VERSION__ lets the desktop app spot a UI that does not match it.
export default defineConfig({
  base: "./",
  plugins: [react()],
  define: { __APP_VERSION__: JSON.stringify(appVersion()) },
  build: { outDir: "dist", sourcemap: false, target: "es2022" },
});
