import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { api, initApi } from "./api";
import { restoreChats, StoreProvider, type Boot } from "./state/store";

/** Desktop: load the real models, profile, settings and saved chats first. */
async function loadBoot(): Promise<Boot | undefined> {
  if (api.kind !== "pywebview") return undefined;
  const [version, models, profile, settings, chats] = await Promise.all([
    api.version(),
    api.listModels(),
    api.hardwareProfile(),
    api.getSettings(),
    api.loadChats(),
  ]);
  return { version, models, profile, settings, chats: restoreChats(chats) };
}

function showBootError(err: unknown) {
  const root = document.getElementById("root")!;
  root.innerHTML = "";
  const box = document.createElement("div");
  box.className = "boot-error";
  const h = document.createElement("h1");
  h.textContent = "EigenCore could not start";
  const p = document.createElement("pre");
  p.textContent = String(err instanceof Error ? err.message : err);
  const hint = document.createElement("p");
  hint.textContent = "Run eigencore-gui --flet to open the previous app. Details are in ~/.eigencore/logs/desktop.log.";
  box.append(h, p, hint);
  root.append(box);
}

async function main() {
  await initApi();
  let boot: Boot | undefined;
  try {
    boot = await loadBoot();
  } catch (err) {
    showBootError(err);
    return;
  }
  if (boot) {
    const theme = boot.settings.theme === "system" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : boot.settings.theme;
    document.documentElement.dataset.theme = theme;
  }
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <StoreProvider boot={boot}>
        <App />
      </StoreProvider>
    </StrictMode>,
  );
}

void main();
