# Changelog

All notable changes to EigenCore. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the git tags. Dates are commit dates.

## [Unreleased]

### Added
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
