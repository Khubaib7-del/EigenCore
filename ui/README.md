# EigenCore desktop UI

The EigenCore desktop interface, built as a web app: Vite + React +
TypeScript, styled with CSS variables. Since v0.4.6 this is what the Windows
app shows: `src/eigencore/desktop/web.py` opens `ui/dist` in a frameless
pywebview window (WebView2) and connects it to the real backend. In a plain
browser (`npm run dev`) it runs on a mock API instead, for design work.

## Run it

Needs Node 20 or newer.

```bash
cd ui
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check (tsc) + production build into ui/dist/
npm run typecheck  # tsc only
```

### Against the real backend

```bash
npm run build && eigencore-gui            # desktop window, loads ui/dist
# or with hot reload: keep `npm run dev` running, then
EIGENCORE_UI_URL=http://localhost:5173 eigencore-gui
```

`eigencore-gui --debug` opens the WebView dev tools.

### Preview switches (mock only)

Add these to the URL to open a screen in a given state (handy for reviews
and screenshots):

| Parameter | Effect |
| --- | --- |
| `screen=chat\|models\|monitor\|profile\|settings` | open that screen |
| `theme=dark\|light\|system` | theme on load |
| `chat=empty` | empty chat with the greeting |
| `confirm=<model>` | Models: open the delete dialog for that model |
| `sample=1` | Monitor: show labelled sample telemetry |
| `update=checking\|available\|downloading` | updater state |
| `palette=1` | open search (Ctrl K) |
| `collapsed=1` | collapsed sidebar |
| `downloads=0` | no download in progress |

## Screens

- **Chat**: greeting and suggestions when empty; model picker (downloaded
  models with provider mark and fit); streaming markdown with code blocks;
  a fold for reasoning (`<think>…</think>`, e.g. Qwen3); tool-call chips; an
  approval card for agent tools; per-message stats; Stop; Enter sends,
  Shift+Enter adds a new line; context meter.
- **Models**: the registry list with provider marks, size, quant, context,
  Hugging Face repo, fit chip (Fits / Tight / Does not fit / Unknown, always
  shown as a word), Download with progress, Use, Delete with a confirm
  dialog, active state.
- **Monitor**: engine state (Idle / Loading / Reading prompt / Writing),
  speed, tokens, first token, engine RAM (estimate), tok/s sparkline,
  context gauge, CPU threads, recent turns.
- **Profile**: CPU, RAM, bandwidth (estimate), GPU, and a recommendation
  marked "estimate until benchmarked".
- **Settings**: theme (dark / light / system), models folder, threads,
  context length, KV cache type (fp16 / q8_0), tool approvals, updates
  (Check for updates → Update to vX → Downloading → Restart to install),
  provider-mark credits.
- **Shell**: frameless title bar (drag area, minimise, maximise, close),
  collapsible sidebar with nav, chat history, version and update button,
  Ctrl K search, Ctrl N new chat.

## Desktop (real) vs browser preview (mock)

`src/api/index.ts` picks the API once, before the first render: inside the
desktop window (`window.pywebview` / the `EigenCoreDesktop` user agent) it
waits for `pywebviewready` and uses `BridgeApi` (`src/api/bridge.ts`);
anywhere else it uses `MockApi`.

In the desktop app everything on screen is real: the registry, what is on
disk, fit verdicts, the hardware profile (bandwidth and the size
recommendation are the profiler's estimates and say so), streamed replies,
measured speed and first-token time, and the updater. Chats start empty and
are saved to `~/.eigencore/desktop-ui.json`, with the theme. There is no
mock pill, no seeded chat and no sample telemetry. Settings the desktop
engine does not take yet (threads, context length, KV cache type, tool
approvals, changing the models folder) are shown as **Coming soon** and
cannot be changed; Run benchmark is the same. The approval card and tool
chips never appear, because the desktop chat has no tool calls yet.

In the browser preview, every value comes from the mock and is marked:
**Preview · mock data** in the title bar, a **mock stream** chip on
streamed replies, and Monitor numbers only with `?sample=1` under a
**Sample telemetry · not measured** badge.

## API layer

`src/api/types.ts` defines `EigenCoreApi` and the `BackendEvent` union. The
Python host sends batches of the tuples `DesktopBackend` puts on its queue,
as `CustomEvent("eigencore", { detail: [[...], ...] })`:

```
("dl_progress", name, pct)      → { type: "dl_progress", name, pct }
("dl_done", name, ok, message)  → { type: "dl_done", name, ok, message }
("engine", state, message)      → { type: "engine", state, message }   // loading | ready | error
("prompt_eval", started)        → { type: "prompt_eval", started }
("tok", text)                   → { type: "tok", text }
("done", stats)                 → { type: "done", stats }              // tokens, seconds, tok_per_sec, first_token_s, decode_tok_per_sec, interrupted, error
("deleted", name, ok)           → { type: "deleted", name, ok }
("switched", name)              → { type: "switched", name }
("update", state)               → { type: "update", state }           // updater (desktop/web.py)
```

`fromTuple()` converts one tuple. Still **PROPOSED** (mock only):
`approval_request`, `tool_call`, and `prompt_tokens` in `done`.

## Design notes

- Monochrome chrome: black and grey in dark, white and grey in light. Colour
  appears only for meaning (ok, warning, error), and muted. No brand colour.
- Tokens live in `src/styles/tokens.css` (`[data-theme="dark"]` and
  `[data-theme="light"]`).
- Motion is 120 to 240 ms and turns off under `prefers-reduced-motion`.
- Fonts: Inter and JetBrains Mono, bundled at build time (no CDN).

## Credits

See [CREDITS.md](./CREDITS.md). Component patterns follow Beautiful UI (MIT,
Shane Levine). Provider marks come from lobe-icons (MIT, LobeHub). Fonts are
OFL-1.1.
