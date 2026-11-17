# Changelog

All notable changes to EigenCore. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the git tags. Dates are commit dates.

## [Unreleased]

### Added
- Two-tier model catalog. The curated list grows from 9 to 19 models, with picks for chat, coding and reasoning on 8, 16, 32 and 64 GB machines, favouring mixture-of-experts models (they read only a few GB per token, so they decode faster on a CPU). New: `granite-4.2-3b`, `granite-4.0-h-tiny`, `qwen3.5-4b`, `qwen3.5-9b`, `deepseek-r1-0528-qwen3-8b`, `gpt-oss-20b`, `qwen3-coder-30b-a3b`, `qwen3.6-35b-a3b`, `qwen3-next-80b-a3b`, `qwen3-coder-next-80b-a3b`. Every file, size and licence was checked against the Hugging Face API, and every context length and model shape against the file's own GGUF header. Each entry now also stores that shape (`geometry`), a `ram_class`, a `priority` and the date it was checked (`verified_at`).
- Hugging Face search, for anything not on the list: `eigencore search <words>` and a search box on the desktop Models screen. It finds GGUF repos (no account needed), picks the Q4_K_M file (or the nearest 4-bit one), reads the licence and flags non-commercial ones. Results are cached for a day in `~/.eigencore/catalog-cache.json`; offline it shows the last results or says it could not search. Repos named "abliterated" / "uncensored" are hidden. Search shows trusted uploaders only by default (known quantisers such as ggml-org, unsloth, bartowski and lmstudio-community, and the model makers' own orgs); community uploads appear with `eigencore search --all-uploaders` or the "Community uploads" switch next to the search box, marked with a "Community" chip. A model you download from search stays in your list (`~/.eigencore/models/custom-models.json`) until you delete it.
- Fit before download: every model now shows "fits", "tight" or "too big" for this machine before you download it, judged against the machine's RAM budget: total RAM minus a reserve for the OS and other apps (20 % of RAM, at least 1.5 GB; on top of that the estimate counts 2 GB for the runtime and wants 1 GB of headroom for "fits"). Free RAM at that moment does not change the verdict, since it swings with whatever else is open; when a model fits the budget but not what is free right now, the details say so. Each row also shows the largest context that fits next to it, and a decode-speed ceiling (memory bandwidth ÷ bytes read per token, labelled as an estimate; real speed is lower). For search results, "Check fit" reads the first few MB of the file (its GGUF header) over HTTP instead of downloading the model.
- Runtime check: a model whose architecture is newer than the installed llama-cpp-python shows "Runtime update" instead of failing after a download. The table of which llama-cpp-python release first loads each architecture and tokenizer is `src/eigencore/models/runtime_support.json`, built from llama.cpp's sources at each release.
- `eigencore download` and `-m` accept any Hugging Face repo as `org/repo` or `org/repo:file.gguf`.
- `scripts/refresh_models.py`: re-checks every curated entry against Hugging Face (sizes, licences, context, shape, minimum runtime, RAM class) and rewrites `models.json`. Runs locally, no CI or token needed; `--check` exits 1 when the list is out of date, `--runtime-table` rebuilds the runtime table.
- Desktop app: a set of 34 small spinners (`ui/src/components/Spinner.tsx`, styles in `ui/src/styles/spinners.css`). Use `<Spinner kind="ring" />`; `SPINNERS` lists every kind with its name, loop length, colour and intended use. They are plain SVG and CSS, draw in the current text colour (or a colour token via `tone`, or any colour via `color`), are meant for 16–24 px, and need no new packages. With reduced motion they hold still and only fade gently. 32 follow a reference gallery the team picked (Arc, Atom, Blocks, Clock, Comet, Compass, Dual, Eclipse, Fill, Flip, Gather, Leap, Linear dots, Loading, Loop, Morph, Orbit, Phases, Pulse, Radar, Ring, Ripple, Scan, Slide, Snake, Split, Steps, Swirl, Trace, plus Circular dots, Classic and Classic v2, which the reference only named, so we drew those three ourselves). Bars and Hourglass are new.
- Desktop app: outside the chat, busy states use the new spinners instead of the grid loader. Buttons can show a spinner while busy (the Models "Downloading…" button does). A model download row shows Steps next to its percentage. The update mark shows Comet while checking, Trace while downloading and Hourglass while installing, and Settings uses the same. A chat in the sidebar shows Pulse while the prompt is read and Linear dots while the reply is written. The Monitor's live chip and nav item show Bars. An approved tool call now shows a "Running" row in the tool steps (Radar for a file search, Orbit for the rest) until its result arrives. The chat itself, the title bar and the Monitor engine card keep the grid loader.

### Changed
- `recommend()` now ranks by the fit verdict at 8K context (a model that fits beats one that is tight), then the curated priority, then the task, instead of "largest model under free RAM − 2 GB", and judges fit against total RAM minus the system reserve rather than the RAM free at that moment, so the pick no longer changes with what else is open. A task can also match a tag, so `reasoning` works. Automatic picks (chat / code): 8 GB RAM `qwen3.5-4b` / `granite-4.2-3b`, 16 GB `qwen3.5-9b` / `granite-4.2-8b`, 32 GB `qwen3.6-35b-a3b` / `qwen3-coder-30b-a3b`, 64 GB `qwen3-next-80b-a3b` / `qwen3-coder-next-80b-a3b`.
- `eigencore models` shows the fit before download for curated models, and `eigencore profile` shows the recommended model's fit against the RAM budget. Picking a model by name (`-m`) is refused only when the file alone is bigger than the budget (it used to compare against free RAM).

### Fixed
- Fit estimator, hybrid models: the KV cache counted every layer. Granite 4.0-H, Qwen3-Next, Qwen3.5 and similar models keep a KV cache only in their attention layers (for example 4 of 40 in Granite 4.0 H-Tiny); the rest are Mamba or linear-attention layers with a small fixed state. The estimate now counts only attention layers (from the per-layer `head_count_kv`, `attention.recurrent_layers` or `full_attention_interval` in the GGUF; llama.cpp's default of every 4th layer when Qwen3-Next files leave it out), and sums per-layer head counts and widths where layers differ (Gemma 4). The GGUF reader now keeps these per-layer values instead of dropping them.
- Downloads of files kept in a sub-folder of the repo (for example `Q4_K_M/model-00001-of-00002.gguf`) were looked for in the wrong place after download. Files now keep their in-repo folders under `~/.eigencore/models/<org>--<repo>/`.

## [0.4.11] - 2026-11-17

A new look for chat and the composer, colour across the app, a smoother minimize on Windows, and a verified model list with two new 8B models.

### Added
- Desktop app: one grid loader for the whole app (`ui/src/components/GridLoader.tsx`). It is a 3×3 grid with 19 patterns (drive, orbit, pulse, snake, rain, scan, checker, spiral, cross, diagonal, sparkle, heartbeat, leap, fill, wave, breathe, and static check, fail and halt) and one colour per job: thinking, reasoning, writing, writing code, tools, approval, loading a model, downloading, checking for updates, searching, live, idle, done, error and stopped. It replaces the orb, the equaliser and the two pixel loaders. Every loader shares one clock, and reduced motion holds a still frame.
- Chat: a "jump to latest" button. Scrolling up while a reply streams frees the view instead of pinning it to the bottom. The arrow brings it back, and sending a message sticks to the bottom again.
- Chat: a copy button under your own message, shown on hover.
- Model registry v2: the curated list now lives in `src/eigencore/models/models.json` (`schema_version` 2, shipped in the wheel and in the Windows build) instead of Python literals. Each entry adds `license` (plus `license_name` / `license_url` when the card gives them), `noncommercial`, `tags`, `arch` (GGUF `general.architecture`), `min_runtime` (first llama-cpp-python release that loads the arch) and `size_bytes`. `load_registry()` reads it; `REGISTRY`, `resolve()`, `recommend()` and `download()` work as before.
- Optional remote list: with `EIGENCORE_MODELS_REMOTE=1` (or `load_registry(remote=True)`) the `models.json` attached to the latest GitHub Release is tried first with a 3 s timeout; any failure falls back to the bundled copy. Off by default.
- New models: `granite-4.2-8b` (`ibm-granite/granite-4.2-8b-GGUF`, Q4_K_M, 5.35 GB, 131,072 ctx, Apache-2.0) and `twil-8b` (`webAI-Official/TwIL-8B`, balanced Q4_K_M, 5.35 GB, 131,072 ctx, webAI Non-Commercial License). Both need llama-cpp-python 0.3.17 or newer (`granite-docling` tokenizer).
- Desktop app: model rows carry `license` and `noncommercial`; the Models screen shows a licence chip, or an amber "NC" chip for non-commercial models.

### Changed
- Chat: your message is a rounded block with more contrast. A reply shows its loader, status word and timer only while it is in progress. When it finishes, the footer always shows the total time, and the model name, the speed, token and first-token meters, copy and regenerate appear on hover.
- Chat: code is syntax-coloured (keywords, strings, numbers, comments, calls, types, decorators, operators), with a separate palette for dark and light. Icons in the chat, the sidebar, tool steps and Monitor are coloured. Text and chrome stay black and white.
- Composer: it starts as one compact row (attach, the box, send) and opens when you click or type in it, showing mode, model, reply options, the context meter and the "/" hint. The reply-options button is now a gear that turns on hover and makes a half turn while open. The attach "+" turns into a ×. "Tools (soon)" moved into the attach menu.
- Update button: the EigenCore mark with a coloured badge instead of a pill. Hovering it grows it into a pill with the version and what a click does. Updates show a blue download badge that pulses, downloads show a progress ring, and an up-to-date or freshly installed app shows a green check until the next version is out.
- Monitor: the tokens-per-second chart draws a teal-to-violet line over a blue fade with an amber average line, the context ring is a gradient that turns amber and then red as it fills, and the stat icons and CPU thread cells are coloured.
- Registry sizes and context lengths now match the files on Hugging Face (file bytes from the HF API, context from each file's GGUF header): qwen2.5-1.5b 1.12 GB / 32,768 ctx, qwen2.5-3b 2.10 GB / 32,768, qwen2.5-coder-1.5b 1.12 GB / 32,768, qwen2.5-7b 4.68 GB (both shards) / 131,072, qwen3-30b-a3b 18.56 GB / 40,960, phi-3-mini 2.39 GB. The engine still opens a smaller context sized for the machine.
- `qwen2.5-3b` is flagged non-commercial (Qwen Research licence).
- `recommend()` (automatic model choice, including the task router) skips non-commercial models and models whose `min_runtime` is newer than the installed llama-cpp-python. They can still be picked by name. On a 16 GB machine the automatic coding model is now `granite-4.2-8b`.

### Fixed
- Windows: minimize and restore no longer cut out abruptly. The frameless window gets the minimize, maximize and system-menu styles, so Windows animates it and a second click on the taskbar button minimizes it. The page fades out before it minimizes and back in when it returns. Needs a check on a real Windows PC.
- Windows: bringing a maximized window back from the taskbar no longer forgets that it is maximized (the next Maximize click did nothing before).
- The window starts in the saved theme's colour, so a light-theme user no longer sees a dark flash on start.

## [0.4.10] - 2026-11-17

Models load again in the Windows app, the window opens straight to full screen, and the handwritten "Eigen Core" reacts to the pointer.

### Added
- Desktop app: the "Eigen" and "Core" words react to the pointer. On hover a word lifts, leans toward the pointer and a phosphor light sweeps across the ink; a click writes it again with the pen.
- `packaging/windows/check_cpu_baseline.py`: the release and PR builds fail if the bundled llama.cpp contains AVX-512 instructions.

### Changed
- Version 0.4.10. The browser preview's mock updater now demos `v0.4.11`.
- Chat box placeholder: "Choose a model to start" when a model is downloaded but none is loaded (it said "Download a model" before), and "Loading the model…" while one loads.
- Title bar: the status dot is grey for "No model" (it was green, like "Ready").
- Settings: the context length hint no longer calls the value "the profiler's pick" (0.4.9 sizes it per model).

### Fixed
- Windows app: loading any model failed with `[WinError -1073741795] Windows Error 0xc000001d` (illegal instruction) in 0.4.9. The release build installed `llama-cpp-python>=0.3.0`; the newest version (0.3.37) had no prebuilt wheel, so pip compiled llama.cpp on the GitHub build machine with `GGML_NATIVE=ON`, tuned to that machine's CPU (most likely AVX-512, which most laptops do not have). 0.4.8 was built the same way and worked, most likely because its build machine had no AVX-512. The build now installs only the prebuilt CPU wheel, pinned to `llama-cpp-python==0.3.36` (AVX2 + FMA + F16C baseline, no AVX-512, checked with the new script), and never compiles it.
- Engine: a CPU without AVX2 now gets a clear message ("this CPU lacks AVX2 ...") before loading, and an illegal-instruction crash during load is reported in plain words instead of the raw Windows error.
- Desktop app (Windows): the window first appeared offset at its 1180 × 760 default and then jumped to maximized. It is now created at the full size of the primary screen and maximized. Restoring it from maximized goes back to 1180 × 760, centred.
- Agent (ReAct): the GBNF grammar was built but never passed to the model, so tool calls were not constrained. `Forge.run_agent` now passes `ReActAgent.grammar` (llama.cpp `LlamaGrammar`, compiled once per grammar) to every step. The grammar now matches what the agent parses (`thought` / `action` / `final` with `content`, `tool`, `args`; the old one had no `final` and used other keys). It also fixes problems that stopped it working in llama.cpp: rule names with `_` (rejected by the parser), unlimited whitespace and digits that let a model spin until `max_tokens`, and raw control characters inside strings. Optional tool parameters are now optional. Checked with a 15M test model: 10 of 10 steps were valid JSON.
- `--pld`: the prompt-lookup path pasted prompt text into the reply without the model producing it, so it could change the answer. The flag is now hidden and ignored with a warning; replies are exactly what the model samples. A real draft-and-verify version is planned.
- One load path: the CLI (`run`, `chat`, the TUI model switch, `analyze`, `benchmark`) and the desktop app now build engines through `eigencore.forge.open_engine`, the same function Forge uses, instead of each creating `InferenceEngine` itself. Without `-m`, `run` asks Forge's task router for the model (a coding prompt gets the coding model) and prints the routing reason; `chat` keeps `--task`. The router does not swap models in the middle of a chat (loading takes too long on CPU).

## [0.4.9] - 2026-11-17

The desktop app opens full screen with a new animated monitor, provider logos are back in colour, and the engine now sizes context and the KV cache from the model's real shape.

### Added
- Desktop app: the retro monitor on the empty chat is bigger and deeper (a chunky CRT). On the first open of each session it plays a short power-up: the cable plugs in, the screen warms up, a boot screen runs, then "Eigen" and "Core" are written by hand in a smooth script, like Apple's "hello". The screen has tabs: HOME, SYS, FIT and SPD (a live tokens-per-second line). The handwriting font is Sacramento (SIL OFL 1.1, credited in `ui/CREDITS.md`).
- Desktop app: model provider logos are shown in their official colours.

### Changed
- Desktop app: the window opens maximized instead of 1180 × 760.
- Version 0.4.9. The browser preview's mock updater now demos `v0.4.10`.

### Fixed
- Context default: the engine opened every model with the profiler's geometry-free `0.002 × params` rule (falling back to 2048), clamped to the registry context. It now computes `n_ctx` with the fit estimator from the model's real GGUF geometry and the chosen KV type: the largest power-of-two context whose KV cache fits the RAM budget, capped at 16K on CPU by default (llama.cpp reserves the whole KV cache up front) and clamped to the GGUF maximum context, with a floor of 2048 (never above the model's maximum). The profiler figure is used only when the GGUF cannot be read. Benchmark snapshots record the `n_ctx` actually used. The CLI `run` line, the TUI header, context bar and `/context`, and the desktop settings now show this `n_ctx` the model actually opened with, instead of the profiler figure.
- Fit estimator, MoE: active weights per token were `params × expert_used_count / expert_count`, which drops attention, embeddings and shared experts. They are now summed from the GGUF tensor sizes: every never-routed tensor in full plus `used/count` × the `*_exps` tensors. Without tensor sizes (model not downloaded) no active figure is guessed and the decode ceiling uses the full weights.
- Fit estimator: the KV-cache size used `head_dim = hidden / heads`, which is wrong for Qwen3 and Gemma. It now reads `{arch}.attention.key_length` / `value_length` from the GGUF (K and V dims counted separately) and falls back to `hidden / heads` only when they are missing.
- GGUF metadata: the gguf-package reader opened files with an invalid mode (`"no"`) and returned strings and scalars as raw arrays, so it never worked and every geometry read fell back to loading the model through llama.cpp. It now opens read-only and decodes strings and scalars.
- `--kv`: the engine passed `type_kv`, which llama-cpp-python does not have (it was silently ignored), and mapped `fp16`/`q8_0` to 0/1, which are ggml F32/F16. It now passes `type_k`/`type_v` with ggml F16=1 and Q8_0=8. A q8_0 V cache needs flash attention, so `--kv q8_0 --no-flash` is now a clear usage error instead of quietly keeping V at f16.

## [0.4.8] - 2026-11-15

A fix for the desktop app showing the old screens after an update.

### Fixed
- Desktop app (Windows): after updating to 0.4.7 the sidebar said v0.4.7 but the 0.4.6 screens appeared. The version comes from the app itself; the screens come from a small local web server inside WebView2. The exact cause on the affected PC is not confirmed (no old EigenCore process was running), so 0.4.8 removes every way old screens could be shown:
  - **Its own port.** The screens were served on `127.0.0.1:42001`, pywebview's default, which every pywebview app shares, and that server lets a second program bind the same port on Windows. EigenCore now uses its own port, 47613, only after checking that nothing else holds it (an exclusive bind on Windows); otherwise 47614 to 47622, otherwise any free port. The port used is written to `~/.eigencore/logs/desktop.log`.
  - **No cached screens.** The page is opened as `index.html?v=<version>`, and when the app version changes EigenCore clears WebView2's caches in `~/.eigencore/webview` (Cache, Code Cache, GPUCache, Service Worker) before the window opens. Local Storage and IndexedDB are kept.
  - **Your data stays.** Chats and the theme were already saved by the app in `~/.eigencore/desktop-ui.json`, so the new port does not lose them. Reply options (temperature, longest reply) were only kept by the page, per port; they are now saved by the app too. Because the port changed, they go back to their defaults once after this update.
  - **One copy at a time.** Opening EigenCore while it is already running brings the open window to the front (or says it is already running) instead of starting a second copy. Windows uses a named mutex; macOS and Linux use a lock file in `~/.eigencore`.
  - **The process ends when the window closes.** Closing the window (the X in the title bar, Alt+F4, or an update) stops any running reply, and if the process is still alive 3 seconds later it is ended, so no windowless EigenCore is left behind.
  - **Smoother installer.** Updating no longer asks "the folder already exists" (`DirExistsWarning=no`), a running EigenCore is closed before files are copied, even if it does not answer (`CloseApplications=force`), and the old `_internal` folder is still removed first. The PR build now compiles the installer and checks the silent in-app update end to end: install the last release, run the new installer the way the app does, and confirm the new screens are installed and served after it reopens.
  - **A warning if they ever differ.** The UI now carries the version it was built for and shows a bar if the app reports another one: "UI build X does not match app Y. Close all EigenCore windows and restart." This only works from 0.4.8 on.

### Changed
- Version 0.4.8 (pyproject, `eigencore.__version__`, CLI, installer script). The browser preview's mock updater now demos `v0.4.9`; `?mismatch=<version>` previews the warning bar.

## [0.4.7] - 2026-11-15

New desktop chat and a new terminal chat. In short:

- **Desktop app:** an agent-style chat (thinking orb and status word on each reply, code blocks that stay intact while streaming, per-reply stats, file trees), reply options and `/` commands in the composer, rename / delete chats, and a small 3D retro computer on the empty chat.
- **Monitor:** the tokens-per-second chart now fills in live while a reply is written and keeps one point per finished reply.
- **Terminal:** `eigencore chat` redone to the v2 design in `docs/design/TUI.md` (header with model, context and fit; Markdown with framed code; state line; measured stats counted in tokens).
- **Fixed:** the TUI tests now pass on Windows CI, plus the fixes listed below.
- **Not yet:** tools in chat, file attach, Agent / Plan modes and changing the context window are still not in the desktop app (shown as **Soon**); tool approvals appear only in the browser preview's mock data. See *Not yet in the desktop UI* below.
- Screenshots in the README (`docs/screenshots/`), from the browser preview on mock data and a demo terminal engine.

### Added
- Desktop chat: the wordmark computer is now a real 3D model (CSS 3D, no WebGL): a cream CRT with a deep rear housing and recessed glass on a low desktop case with floppy slot, vents and keyboard, shaded from one light with a contact shadow; teal phosphor screen with a pixel "EC" mark; tilts a few degrees toward the pointer on hover, dips when pressed, real 3D bezel buttons and power LED, cooler grey plastic in the light theme. The composer row now wraps instead of overflowing in narrow windows.
- Desktop chat: the empty chat's retro computer is now a small computer inside an "EIGEN [computer] CORE" wordmark, with a live, clickable screen (boot line, echo of the message box, pages for version / CPU / threads / model / context and the model's fit verdict, real loading and no-model states; CSS-only animation, paused when hidden, off under reduced motion).
- Desktop chat: an interactive retro computer on the empty chat (real version, CPU, threads and model on its screen; its keys and suggestions fill the message box), a thinking orb and rolling status word on each reply (Thinking / Analyzing / Reading prompt before the first token, Writing / Writing code while streaming), per-reply stats as small bars, code blocks that render correctly while the fence is still open (language, file icon, line numbers, copy), and path lists shown as a git-style file tree.
- Composer: reply options (temperature and longest reply, sent to the engine), `/` commands (`/new`, `/models`, `/monitor`, `/profile`, `/settings`, `/theme`), mode menu and attach menu.
- Sidebar: rename and delete chats (saved), a bar loader on a chat while it is generating, Profile and Settings at the foot.
- More inline icons (folder-open, file-code, git-branch, tree, sparkles, shield-check, sliders and more), all on one 1.5 px stroke.
- `tui/sanitize.py`: one sanitiser for all model text. Strips CSI, OSC, DCS / SOS / PM / APC in 7-bit and 8-bit (C1) forms and every other control character except newline and tab (the old one let `\x9b`, BEL and carriage returns through).
- `tui/render.py` (Markdown with framed code blocks, git-style tree renderer, turn-stats formatter) and `tui/components.py` (tools label, SSD-paging warning, tool approval card per §3.5 with default-deny focus and a plain `[y/N]` prompt). The approval card is a tested component only; nothing in chat shows it.
- ASCII glyph fallback (`[#]`, `>`, ASCII boxes and spinner) when the terminal encoding is not UTF-8, on the legacy Windows console without Windows Terminal, or with `EIGENCORE_ASCII=1`.
- `tests/test_tui_v2.py`: header snapshots at 80 / 100 / 120 columns and plain, sanitiser acceptance strings in both modes, streaming code fences, state line and stats, Ctrl-C, no regenerate on error, scrolling, SSD warning, approval card, tree, capability probe.

### Changed
- Version 0.4.7 (pyproject, `eigencore.__version__`, CLI, installer script). The browser preview's mock updater now demos `v0.4.8`. Installed 0.4.6 apps update to 0.4.7 in-app as usual (same asset names).
- `eigencore chat` full-screen TUI redone to the v2 design (`docs/design/TUI.md` §3.1). Header: `◩ EigenCore <version> · chat` with `local · ready / thinking / writing` on the right, and a second row with model + quant, context (`≈used / set tok`), fit verdict with headroom (`est.`) and the tools state. Conversation uses `you` / `eigencore` labels with indented bodies; replies render as Markdown with framed, syntax-highlighted code blocks that stay correct while a code fence is still streaming. Footer: `›` prompt with placeholder, command hints, live `/command` suggestions, `ctrl-c stops`, and a context bar that turns amber at 85 %. Narrow terminals (< 100 columns) get the collapsed header and footer; PgUp / PgDn scroll the conversation.
- While a reply streams, an animated state line shows what is happening (`Thinking…` before the first token, `Writing…` / `Writing code…` after) and the elapsed seconds; `EIGENCORE_REDUCED_MOTION=1` makes it static. When the reply ends it becomes the measured stats line: `decode <n> tok/s · prefill <t> s · <n> tokens · measured this turn`.
- Speed is now counted in tokens, not stream chunks: `InferenceEngine.stream_messages` records `last_stream_stats` (token count from the model's tokenizer, time to first token, decode time) and there is a new `InferenceEngine.count_tokens()`. Decode tok/s excludes the first token's wait. Without a token count the UI says `chunks/s (stream chunks, not tokens)`.
- `--plain` / non-TTY output follows §3.2: `EigenCore <ver> | <model> <quant> | ctx <n> | fit: <verdict> (<±h> GB est.) | tools: off`, `you>` / `eigencore>` prompts, `[decode … tok/s measured | prefill … s | … tokens]`, `[stopped]` on Ctrl-C.
- The tools field shows what is attached (`tools off` in chat, which has no tools) instead of the hard-coded "read-only (policy engine attached in P4)". The approval prompt is no longer wired into plain chat.
- Before loading a model whose file is larger than free RAM, `eigencore chat` asks first (default: don't load); in the full-screen app, switching to such a model from `/models` shows the SSD-paging warning with `[c] continue  [q] pick another model  [x] cancel`.

### Fixed
- Desktop Monitor: the tokens-per-second graph stayed empty. It only got one value per finished reply and drew nothing until there were two (so after the first reply it still said "No turns yet", and it never moved while a reply was being written). The backend now emits a live `("tps", {tok_per_sec, tokens, t})` sample about every 0.5 s while writing, measured over that interval; the chart plots those live (**This reply**) and one point per finished reply (**Per reply**), and draws from the first point. Replies that produced no tokens no longer add a 0.
- Plain mode: an engine error mid-stream could re-run the prompt through the single-prompt fallback (double reply). The fallback is now chosen before streaming starts.
- Arrow keys split across reads (slow ptys, SSH) were taken as Esc plus typed text; the key reader now waits up to 100 ms for the rest of an escape sequence.
- TUI tests failed on Windows CI because Rich picked the legacy Windows console renderer; the tests now pin Rich to the VT renderer so Windows and Linux render the same.

### Not yet in the desktop UI
- Shown as **Soon** and not clickable: Agents and Files in the sidebar, attach files / folder, Agent and Plan modes, Tools, and changing the context window. Tool steps and the approval card (Deny / Allow for session / Allow once, focus on Deny) appear only in the browser preview's mock data.

## [0.4.6] - 2026-11-13

### Added
- New desktop UI. The Windows app (installer, portable exe and in-app update) now opens the web interface from `ui/` (Vite + React + TypeScript) in a frameless pywebview window using Microsoft Edge WebView2, with its own title bar (drag, double-click to maximise, minimise / maximise / close) and edge grips for resizing. Screens: Chat, Models, Monitor, Profile, Settings, plus Ctrl K search and Ctrl N new chat.
- Everything in it is real data: the model registry, files on disk and fit verdicts, the hardware profile (bandwidth and the size recommendation are labelled as the profiler's estimates), streamed replies, measured first-token time and output speed per reply (Chat and Monitor), and the updater. Chats start empty and are saved, with the theme, to `~/.eigencore/desktop-ui.json`.
- `desktop/web.py`: the pywebview host. `WebBridge` is the `js_api` object over the existing `DesktopBackend` and `updater`; backend events reach the page in batches as `CustomEvent("eigencore")`. `ui/src/api/bridge.ts` implements the UI's `EigenCoreApi` over it; in a plain browser the UI keeps using the mock API for design work.
- `eigencore-gui --flet` opens the previous Flet app. It also opens automatically when the web UI can't start (pywebview missing, no WebView2 runtime, UI files missing, or the page doesn't load within 25 s); the reason goes to `~/.eigencore/logs/desktop.log`.
- `eigencore-gui --self-test REPORT.json` (used by CI on the packaged exe) reports what a build can load without opening a window. New `Desktop build` workflow builds both Windows packages on pull requests and runs it; nothing is uploaded.
- `docs/PROJECT_JOURNEY.md`: a plain-language, detailed account of the project — what EigenCore is, the ordered journey (honesty review, packaging, Windows fixes, releases 0.4.0–0.4.5, terminal TUI, desktop app + installer + in-app updater), analysis conclusions, current state, full architecture and release pipeline, and a glossary for every technical term. Linked from `docs/INDEX.md`.

### Changed
- Windows packaging uses PyInstaller directly (`packaging/windows/EigenCore.spec`) instead of `flet pack`. The release job builds the UI with Node 20 (`npm ci && npm run build`) and bundles `ui/dist`, pywebview (WebView2) with pythonnet, llama.cpp, and Flet with its desktop client for the fallback. Asset names are unchanged (`EigenCore-Setup.exe`, `EigenCore.exe`), so 0.4.5 updates to 0.4.6 in-app as usual.
- `pywebview` added to the `desktop` extra. CI also type-checks and builds `ui/`.
- Desktop backend: emits `prompt_eval` (prompt being read / first token), adds `first_token_s` and `decode_tok_per_sec` to the per-reply stats, keeps each reply in the conversation (before, the model never saw its own earlier answers in the desktop chat), and accepts the chat history from the UI.

### Not yet in the desktop UI
- Settings for threads, context length, KV cache type, tool approvals and the models folder are shown as "Coming soon" (the desktop engine doesn't take them yet). Run benchmark in Profile is the same; use `eigencore benchmark`. The desktop chat has no tool calls, so no approval prompts appear.

## [0.4.5] - 2026-11-13

### Added
- Windows installer `EigenCore-Setup.exe` (Inno Setup, `packaging/windows/EigenCore.iss`). Per-user install into `%LOCALAPPDATA%\Programs\EigenCore` with no admin prompt, Start menu entry (desktop shortcut optional), and an entry in Settings > Apps > Installed apps with a normal uninstaller. Models in `~/.eigencore` are never touched by install, update or uninstall.
- In-app updates: the sidebar button checks GitHub (**Check for updates**, then **Update to vX**), downloads the new installer with progress and runs it silently. It closes the app, updates in place (fixed AppId) and reopens EigenCore. Portable and pip installs open the release page instead. Logic lives in `desktop/updater.py` and is unit-tested.

### Fixed
- Title-bar close (X) did nothing: Flet 1.x `page.window.close()` is a coroutine and was called from a plain lambda, so it never ran. It is now awaited (falling back to `destroy()`). Regression test added.
- Sidebar **Download** and the update banner did nothing: `page.launch_url` does not exist in Flet 1.x. Replaced by the updater flow above.
- Model **Delete** confirmation never opened: `page.open()` and `page.close_dialog()` were removed in Flet 1.x. Now uses `page.show_dialog()` and `page.pop_dialog()`.
- Double-clicking the title bar now maximizes or restores the window.

### Changed
- Release workflow: one Windows job builds the installer and the portable `EigenCore.exe` and publishes the GitHub Release directly. It no longer uses `actions/upload-artifact`/`download-artifact`, so it consumes no Actions artifact storage. The Windows release no longer waits on the PyPI job.
- README rewritten for first-time users: one download button for the installer, five numbered steps, plain-language sections on updating, uninstalling and what to expect. Terminal install follows. Developer material (build, speed arithmetic, status table, architecture) is grouped under "For developers", with the long tables folded.

## [0.4.4] fixes (re-tagged 2026-11-13)

### Fixed
- Packaged exe could not load any model (`cannot find ... llama_cpp\\lib`): the release workflow now passes `--pyinstaller-build-args "--collect-all" "llama_cpp"` to `flet pack`, so PyInstaller bundles the native DLL.
- Desktop app crash on startup (`'ProfileView' object has no attribute 'attach'` in the packaged exe): all views now share a `View` base class implementing `attach`/`build`/`handle_event`; regression test runs the same polymorphic loop as `app.main`.
- Desktop app crash on startup (`Container(...) Control must be added to the page first`): `main()` called `content.update()` before `page.add(...)`; layout is now added to the page before the first `show()`, and all control updates guard the unmounted `RuntimeError` (Flet 1.x raises it from the `page` property). Regression test drives `main()` with a mock page and reproduces the exact error on the old code.

### Changed
- Desktop UI polish toward the reference design: frameless window with a quiet custom title bar (drag area + ghost min/max/close buttons, OS chrome hidden), chat empty-state hero with the brand banner, avatar message bubbles, engine status pill in the chat header, rounded composer, and live per-model progress bars during downloads.
- Windows download is now one click on the `.exe` itself: the release workflow builds a single-file `EigenCore.exe` (`flet pack --onefile`) and attaches it directly; the README **Download** button and the app's sidebar **Download** button link to `releases/latest/download/EigenCore.exe` instead of the release page / zip.

## [0.4.4] - 2026-11-13

### Added
- Release pipeline now builds the Windows desktop app on a `windows-latest` runner (`flet pack`) and attaches `EigenCore-win-x64.zip` to every GitHub release; README hero has a **Download for Windows** button linking to the latest release, and the quick start is exe-first for beginners (pip/CLI moved to a collapsible section).
- Desktop app sidebar: brand logo on top, version + **Download** button at the bottom (opens the latest release, so updates are one click); the app checks GitHub once at startup and shows an update banner when a newer version exists (`latest_release`/`is_newer` in the desktop backend, covered by tests).

### Added
- Windows-first desktop app (`eigencore-gui` / `python -m eigencore.desktop`), new optional extra `eigencore[desktop]` (Flet 1.x, no new hard dependency). Dark theme from the design handbook; three screens — Chat (model dropdown, Markdown streaming, stop button, measured tok/s per turn), Models (fit chips, live download progress, delete-from-disk with confirm, active model protected), Profile (hardware + recommendation card, speeds labeled estimate). Logic lives in a UI-agnostic, headlessly-tested backend (`src/eigencore/desktop/backend.py`); 12 tests in `tests/test_desktop.py`. Packaging: `flet pack` on Windows — see `docs/desktop.md`.

## [0.4.3] - 2026-11-11

### Added
- Full-screen chat TUI for `eigencore chat` on a TTY (zero new dependencies — Rich Live + stdlib key reader): ASCII banner with reveal animation, first-run onboarding, then a full-screen app with header (model, context bar, fit, last measured tok/s), streaming transcript, editable input with history and slash-command Tab completion. Slash commands: `/help /models /model /download <name> /delete <name> /context /clear /tokens /temp /exit`. The `/models` picker switches the active model (engine reload), downloads (`d`), and deletes (`x`, with confirmation) models from disk. Ctrl-C interrupts generation without losing history; `--plain`/non-TTY keeps the previous line-mode UI.
- `src/eigencore/tui/app.py` (ChatApp, InputLine, overlays), `src/eigencore/tui/keys.py` (cross-platform key reader), `src/eigencore/tui/banner.py` (banner art + reveal).

## [0.4.2] - 2026-11-11

### Fixed
- Model registry: `qwen2.5-7b` filename corrected to the sharded HF pair (`qwen2.5-7b-instruct-q4_k_m-0000{1,2}-of-00002.gguf` — 404 before); `phi-3-mini` moved to `bartowski/Phi-3-mini-4k-instruct-GGUF` (TheBloke repo removed). Downloads are shard-aware (`snapshot_download` with allow_patterns), drop the deprecated `local_dir_use_symlinks`, and raise a clean one-line error instead of an HF traceback.
- CLI UX: bare `eigencore` on a TTY opens an interactive home menu (profile / models / download / delete / chat / run); new `eigencore delete <name>` frees disk (shard-aware, idempotent); `run`/`chat`/`analyze`/`benchmark` no longer auto-download a missing default — they fall back to an already-downloaded model or exit with a clear hint. `models` shows `unknown*` fits with a note that exact fit needs the file downloaded.

### Fixed
- Model registry: `qwen2.5-7b` filename corrected to the sharded HF pair (`qwen2.5-7b-instruct-q4_k_m-0000{1,2}-of-00002.gguf` — 404 before); `phi-3-mini` moved to `bartowski/Phi-3-mini-4k-instruct-GGUF` (TheBloke repo removed). Downloads are shard-aware (`snapshot_download` with allow_patterns), drop the deprecated `local_dir_use_symlinks`, and raise a clean one-line error instead of an HF traceback.
- CLI UX: bare `eigencore` on a TTY opens an interactive home menu (profile / models / download / delete / chat / run); new `eigencore delete <name>` frees disk (shard-aware, idempotent); `run`/`chat`/`analyze`/`benchmark` no longer auto-download a missing default — they fall back to an already-downloaded model or exit with a clear hint. `models` shows `unknown*` fits with a note that exact fit needs the file downloaded.

## [0.4.1] - 2026-11-11

### Changed
- Packaging: `llama-cpp-python` moved from a hard dependency to the optional `[inference]` extra, because PyPI has no Windows wheel for it and a source build needs a C++ toolchain (broke `pip install eigencore` on plain Windows machines). Core CLI now installs with a bare `pip install eigencore`; inference commands raise a clear hint to install `eigencore[inference]` with the prebuilt wheel index. Added `python -m eigencore` entry point. README/CONTRIBUTING dev setup uses `.[dev,inference]`; CI installs `.[dev,inference]` with the CPU wheel index.

### Changed
- Packaging: `llama-cpp-python` moved from a hard dependency to the optional `[inference]` extra, because PyPI has no Windows wheel for it and a source build needs a C++ toolchain (broke `pip install eigencore` on plain Windows machines). Core CLI now installs with a bare `pip install eigencore`; inference commands raise a clear hint to install `eigencore[inference]` with the prebuilt wheel index. Added `python -m eigencore` entry point. README quick start updated to the two-step install.

## [0.4.0] - 2026-11-11

### Added
- PyPI release pipeline: `release.yml` now publishes sdist+wheel to PyPI on `v*` tags via trusted publishing (OIDC, `pypa/gh-action-pypi-publish`, no stored tokens). After tagging `v0.4.0` and configuring the GitHub OIDC publisher at pypi.org, `pip install eigencore` works. README quick start updated accordingly (prebuilt llama.cpp wheel index documented for Windows/CPU).

### Added
- Stage E slice 3: activation-sparsity experiment scaffolding (REPORTED-ONLY, not wired into inference). `engine/sparsity_experiment.py`: per-layer activation sparsity measurement, sparse-mask plan that only applies above a minimum sparsity, and the kill criterion (measured quality delta > allowed → killed). Rationale recorded: on bandwidth-bound CPU decode the compute saving is small, so the kill criterion matters. No claim without owner runs.
- Stage E slice 2: layer-pruning experiment scaffolding (REPORTED-ONLY, not wired into inference). `engine/pruning.py` computes ShortGPT Block Influence per layer from boundary hidden states, builds a prune plan that protects head/tail layers, and enforces the kill criterion (measured quality delta > allowed → killed, no pruning). No quality/performance claim without the owner's calibration + perplexity runs.
- Stage E slice 1: Q4_K reference kernel (`kernels/q4k.py`) — full GGUF super-block dequant (fp16 d/dmin, packed 6-bit sc/min, 4-bit quants, `d*sc*q - dmin*m`) plus grouped quantized dot, on `BackendProtocol`/`ScalarBackend`. **Bit-exact vs `gguf.quants.dequantize` for Q4_K** on random blocks. Reference only; no performance claim, not wired into inference.
- Stage D slice 3: adaptive terminal chrome + first-run onboarding. New `tui/theme.py` capability probe (`--plain`, non-TTY, `NO_COLOR`, `TERM=dumb` → plain; width tiers wide ≥100 / narrow; header shortens on narrow). New `tui/onboarding.py`: first-run model picker grouped by fit verdict (MoE shows resident vs read-per-token GiB), SSD-paging warning before load when the file exceeds free RAM, config persisted to `~/.eigencore/config.json` (`onboarded`) so it runs once; skipped entirely when non-interactive. Speed in the picker is labeled ESTIMATE.
- Stage D slice 2: policy-backed approvals. New `agent.policy.ApprovalManager` (ASK / SESSION / DENY, session-scoped per tool name, audit-logged). `ToolExecutor(approver=..., ask_fn=...)` consults it after policy passes but before running; no callback attached means deny. Chat UI renders an approval card (exact redacted args, y/s/n) via `ChatUI.render_approval_card`, plus `/approvals`. Enforcement stays in the executor, never in prompt text or rendering.
- Stage D slice 1: tool-handler process isolation hardened with POSIX rlimits on the child worker (`RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_FSIZE`; `RLIMIT_NPROC` opt-in only, default off because it breaks library imports in the child). Configurable via `ToolExecutor(max_cpu_seconds=, max_memory_mb=, max_file_size_mb=, max_processes=)`; skipped safely on non-POSIX. Kill-on-timeout unchanged.
- Stage C slice 4: prompt-lookup decoding behind `--pld` on `run`/`chat` (default OFF until the owner's §9 item-9 measurements decide). `engine.prompt_lookup.longest_prompt_continuation` finds the longest prompt-verbatim continuation; the engine emits it directly and swallows the covered backend tokens to keep the stream aligned; copied chars reported via `engine.last_pld_copied_chars`. No speedup claimed.
- Stage C slice 3: GPU/VRAM fit and layer offload. `profile_hardware()` detects NVIDIA VRAM via `nvidia-smi` (never raises; zeros when absent) and `HardwareProfile` gains `gpu_name`/`total_vram_gb`/`available_vram_gb`, shown in `profile`. `fit.gpu_layers_for_vram()` computes how many layers fit in VRAM after a reserve and the KV cache; `estimate_fit(..., available_vram_gb=...)` reports `gpu_layers`/`gpu_weights_gb` with a note. `InferenceEngine(..., n_gpu_layers=...)` replaces the hardcoded 0; `eigencore run`/`chat` gained `--gpu-layers` and auto-recommend offload when a GPU is detected and the flag is omitted (0 still means CPU-only).
- Stage C slice 2: KV-cache type and flash-attention options. `InferenceEngine(..., kv_type="fp16"|"q8_0", flash_attn=...)` maps to llama.cpp `type_kv` (0/1); `eigencore run` and `eigencore chat` gained `--kv [fp16|q8_0]` and `--no-flash`. `q8_0` halves KV-cache memory; the fit estimator already accounts for it via `kv_quant`.
- MoE-aware fit (Stage C slice 1): `ModelGeometry.expert_count`/`expert_used_count` parsed from GGUF (`{arch}.expert_count*`); `ModelSpec.experts_total`/`experts_active` with `is_moe`; registry gains `qwen3-30b-a3b` (Qwen3-30B-A3B, 128 experts / 8 active, Q4_K_M ≈ 18.6 GiB). `estimate_fit` now reports `active_weights_gb`, `resident`, and a `decode_ceiling_tok_s` = bandwidth ÷ active bytes-per-token (ceiling, not a measurement); non-resident models get an explicit mmap/SSD-paging warning. CLI shows MoE active params, resident/paging status, and the ceiling. No dense-model SSD streaming built, per the review.

### Fixed
- Chat TUI: model output is sanitized (ANSI/OSC/CSI stripped, newline/tab kept) and Rich-escaped; Ctrl-C mid-stream keeps partial history without regenerating; stream errors keep the partial response and never double-print.
- Benchmark harness: completion tokens counted via the backend tokenizer (chars/4 fallback only when no tokenizer), and the summary reports cold (first run per case) vs warm runs separately instead of one mixed median.
- Agent safety: `ToolExecutor` is now deny-by-default (defaults to `default_policy()` — READ-only, workspace-confined — when no policy is passed; `Forge.agent()`/`ReActAgent` inherit this). Path confinement uses explicitly declared `ToolParam(is_path=True)` parameters (no more name heuristics) and handlers receive resolved workspace-confined absolute paths. Timeouts now run handlers in a spawned process that is terminated/killed on timeout. Successful string output is credential-redacted. Tests cover `/etc/passwd` denial and child-process kill.
- Q4_0 reference dequantizer: corrected to the ggml strided nibble layout (byte j low nibble → element j, high nibble → element j+16) and now also accepts raw 18-byte blocks; Q8_0 accepts 34-byte blocks. Verified bit-exact against `gguf.quants.dequantize` for Q4_0 and Q8_0.
- Fit estimator: all sizes in GiB end to end; weights use actual GGUF file size when available; KV cache computed only from architecture-aware GGUF geometry (`{arch}.block_count` / `embedding_length` / `attention.head_count*`), otherwise verdict is `UNKNOWN` instead of a 15–30×-low heuristic; `max_context_for_budget` returns 0 without geometry. CLI `profile`/`models`/`chat` pass real GGUF metadata when the model is downloaded. Profiler GiB/GB mix fixed and 25% of usable RAM reserved for KV, so recommended context no longer pins at 2048 below ~38 GiB. Added `gguf` dependency and `eigencore.fit.read_gguf_model_info`.

### Documentation
- Added a full review: truth audit, 30–40B memory and bandwidth arithmetic, ranked issues, prioritized plan and owner benchmark list (`docs/review/`).
- Brand v2: flat logo, small mark, favicon, lockups, wordmarks and banners as SVG in `assets/brand/` (PNG/ICO exports to follow); v2 design handbook.
- Brand v2 fix: eigen-axis realigned to a clean 45° dotted line inside the die, evenly spaced dots, no trace crossings; middle trace now ends on the axis.
- Added MIT `LICENSE` (© 2026 Khubaib Nazeer).
- Added terminal UI status, technology decision and v2 screen mock-ups (`docs/design/TUI.md`, `docs/design/tui/`).
- Added presentation materials: slides v2 text with change list, and a study guide (`docs/presentation/`).
- README rewritten to install from source (not on PyPI), with a SHIPPED / REPORTED-ONLY / PLANNED status table and known bugs.
- Review notes added to research and positioning docs whose claims overstate the code.
- Added CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue and PR templates, and this changelog.

### Known issues (found by the review, not yet fixed)
- Fit estimator KV fallback underestimates by ~15–30×, and the CLI always uses it.
- Profiler GiB/GB mix pins recommended context at 2048 below ~38 GB RAM.
- Q4_0 reference dequantizer and its test use the wrong nibble order.
- Tool policy is opt-in in the agent API; executor timeout does not stop handlers; path checks depend on parameter names.
- Chat crashes on Rich markup in model output and passes terminal escape sequences.
- Speculative verifier is not a correct rejection sampler.

## [Post-0.3.0 on main] - 2026-07-18 to 2026-11-08
Not tagged; package version is still 0.3.0.

### Added
- Phase 1 benchmark harness with quality cases and JSON reports.
- Conservative model-fit estimator with KV-cache and context budgeting.
- Deny-unless-allow-listed tool policy engine with path confinement and audit log.
- Streaming chat UI with plain mode, slash commands and fit-aware header.
- Backend protocol and scalar reference kernels with tests.
- Brand kit and animated banner.

### Changed
- Optimization statistics labelled as reported-only; none is wired into the backend.
- CI pins ruff below 0.15.

### Fixed
- `generate`/`stream` routed through chat completion.

## [0.3.0] - 2026-07-18
### Added
- Agent framework: tool system, GBNF grammar builder, ReAct loop, wired into the Forge API, with tests.
- `benchmark` command; Phase 2 optimizers attached to the inference engine (reporting only).
### Changed
- README rewritten.

## [0.2.0] - 2026-07-17
### Added
- Hardware profiler with CPUID detection and model registry.
- Inference engine over llama.cpp with hardware-aware configuration.
- Task router, attention-aware context manager, QLoRA training components, sparsity analysis tools.
- CLI and Forge API; architecture, research and roadmap docs.
- Layer-skipping scheduler, speculative decoding module and sparsity predictor (not wired into inference).
- GitHub Actions CI and release workflow.

## [0.1.0] - 2026-07-17
### Added
- Project skeleton with `pyproject.toml` and README.
