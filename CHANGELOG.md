# Changelog

All notable changes to EigenCore. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the git tags. Dates are commit dates.

## [Unreleased]

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
