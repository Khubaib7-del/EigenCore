# EigenCore desktop UI (preview)

A new desktop interface for EigenCore, built as a web app: Vite + React +
TypeScript, styled with CSS variables. **This is UI only.** Every value comes
from a mock API (`src/api/mock.ts`). The current Flet app in
`src/eigencore/desktop/` is unchanged and is still what ships.

## Run it

Needs Node 20 or newer.

```bash
cd ui
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check (tsc) + production build into ui/dist/
npm run typecheck  # tsc only
```

### Preview switches

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

## What is mock and what is real

| Real (from the repo) | Mock |
| --- | --- |
| Model names, sizes, quant, context, descriptions, repos (`models/registry.py`) | Which models are downloaded, the active model, fit verdicts |
| Event names and payloads (`desktop/backend.py`) | The reply text and its timing |
| Fit verdict values (`fit/estimator.py`) | Hardware numbers (an example i7-10610U laptop) |
| Approval modes and redaction (`agent/policy.py`) | The tool calls |
| Version 0.4.5 and the updater flow (`desktop/updater.py`) | The "newer" release tag |

Nothing on screen is a measurement. Seeded replies show `— tok/s`. Replies
streamed by the mock show the mock's own timing with a **mock stream** chip.
Monitor numbers only appear with `?sample=1`, under a **Sample telemetry ·
not measured** badge. The title bar always shows **Preview · mock data**.

## API layer

`src/api/types.ts` defines `EigenCoreApi` and the `BackendEvent` union. The
events mirror the tuples `DesktopBackend` puts on its queue:

```
("dl_progress", name, pct)      → { type: "dl_progress", name, pct }
("dl_done", name, ok, message)  → { type: "dl_done", name, ok, message }
("engine", state, message)      → { type: "engine", state, message }   // loading | ready | error
("tok", text)                   → { type: "tok", text }
("done", stats)                 → { type: "done", stats }              // tokens, seconds, tok_per_sec, interrupted, error
("deleted", name, ok)           → { type: "deleted", name, ok }
("switched", name)              → { type: "switched", name }
```

`fromTuple()` converts a tuple sent as a JSON array. A few events are marked
**PROPOSED**: the backend does not send them yet (`prompt_eval`,
`approval_request`, `tool_call`, `update`, and `first_token_s` /
`prompt_tokens` in `done`).

## Planned wiring (later)

1. Load `ui/dist/index.html` in a frameless **pywebview** window (Edge
   WebView2 on Windows). `vite.config.ts` uses `base: "./"`, so the build
   works from `file://`.
2. Expose `DesktopBackend` through `js_api`; a `PyWebviewApi` class
   implements `EigenCoreApi` by calling `window.pywebview.api.*`.
3. A Python thread drains `backend.events` and pushes each tuple to the page
   with `window.evaluate_js(...)`; the UI turns it into a `BackendEvent`
   with `fromTuple`.
4. The title bar uses the `pywebview-drag-region` class; its buttons call
   window minimise / maximise / close.
5. Add `first_token_s`, `prompt_tokens` and the approval / tool-call /
   updater events on the Python side.

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
