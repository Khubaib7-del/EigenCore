# EigenCore roadmap and implementation status

**Status snapshot:** 2026-11-05. Status below is based on repository source, not a claim that each feature is production-ready or benchmark-verified.

## What exists in this checkout

| Area | Source-backed implementation | Not established by code presence |
| --- | --- | --- |
| Hardware and model fit | Python HAL profiling; `ModelRegistry` list/recommend/download; inference options derived from profiles | Reliable tok/s predictions across hardware, safe fit for every context/model, or near-optimal configuration |
| Inference | `InferenceEngine` delegates GGUF inference to `llama-cpp-python`; generation, streaming, chat, and optimization components/statistics | EigenCore-owned low-level inference kernels or proven quality/speed gains from layer skipping, sparsity, or speculative decoding |
| Routing and training | `TaskRouter`; CPU trainer and convergence utilities; optional training dependencies | Router accuracy or a memory/performance guarantee for fine-tuning |
| Agents and context | ReAct loop, `ToolRegistry`, `ToolExecutor`, GBNF builder, context manager with SQLite persistence and budget handling | OS-level tool sandbox, approval policy, safe autonomous file/shell access, or validated context-retention quality |
| User experience | Click CLI (`profile`, `models`, `download`, `run`, `chat`, `analyze`, `benchmark`, `train`) and `Forge` Python API | Full-screen adaptive TUI, bare-command onboarding, or direct safe tool workflow in chat |

See [system architecture](../architecture/system-overview.md) for module flow and [local inference roadmap](../architecture/local-inference-roadmap.md) for the 30–40B target.

## Recommended next work

### 1. Establish trustworthy quality and performance baselines — **in progress**

Implemented: `eigencore.benchmark` harness (`src/eigencore/benchmark/harness.py`) with deterministic quality cases, prefill/decode timing separation (TTFT vs decode tok/s), SHA-256 model identity, hardware snapshot, and JSON evidence reports via `eigencore benchmark --report <path.json>`. Covered by `tests/test_benchmark.py`.

Use the current backend before replacing it. Select representative consumer CPU/GPU configurations and model/quant combinations. Capture warm/cold load, prompt prefill, decode throughput, memory use, context size, backend flags, and raw run results. Pair throughput with task-specific quality comparisons; include accuracy retention for every optimization. Avoid universal performance claims.

**Exit evidence:** reproducible scripts/configurations, named hardware/model matrix, raw benchmark artifacts, and quality results with an explicit acceptance threshold.

### 2. Make hardware-aware selection conservative and inspectable

Separate weight, runtime, and KV-cache estimates; account for unknown metadata and keep a safety margin. Treat Core i7/GPU names as insufficient without memory capacity, generation, backend, and context. Present user overrides and explain the risk when an estimate is tight.

**Exit evidence:** fixture-tested fit decisions, correct unavailable/unknown handling, and safe behavior near memory limits on the target matrix.

### 3. Gate inference optimizations on correctness

Audit which optimization components actually change backend execution. First integrate or disable components whose reported stats are not connected to measured work. Differential-test quantized/SIMD kernels against a scalar reference and the trusted backend. Evaluate layer skipping, sparse execution, context compression, and speculative decoding on fixed prompts and quality checks.

**Exit evidence:** correctness tests, measured performance improvement on named workloads, quality within the chosen threshold, and a tested fallback path.

### 4. Harden agent tools before exposing them in chat

The current Python executor is not a process sandbox. Define capabilities, deny-by-default policy, path and network restrictions, observable approvals, process isolation, resource limits, and auditable outcomes before adding shell or write tools to an interactive assistant.

**Exit evidence:** threat model, policy tests, isolation tests, adversarial tool-output handling, cancellation/timeout behavior, and approval audit evidence.

### 5. Prototype the adaptive terminal chat UI

Use [terminal chat design](../design/terminal-chat.md) as a proposal. Integrate the existing chat/stream API, provide plain-mode fallback, explain model fit, and keep rendering separate from inference and policy. Do not change the bare `eigencore` behavior or command names without a compatibility decision.

**Exit evidence:** interaction checks at small and large terminal sizes, PowerShell/Windows Terminal and Unix shell checks, keyboard-only operation, plain mode, and no accidental model downloads/tool execution.

### 6. Explore EigenCore-owned low-level kernels

Only start after baseline profiling identifies a bottleneck. Keep `llama-cpp-python` as reference/backend fallback while prototypes cover well-scoped numerical primitives with documented layouts, scalar implementation, platform dispatch, and differential tests.

**Exit evidence:** supported-platform correctness and reproducible speed/quality comparisons before enabling a custom kernel by default.

## Former roadmap claims

The earlier draft proposed goals such as near-optimal automatic speed, fine-tuning at a fixed RAM ceiling, and practical agent operation on a fixed memory tier. Those were targets, not measured acceptance results; they are intentionally not carried forward as current guarantees. The project is MIT licensed; see repository metadata for current package version and dependencies.
