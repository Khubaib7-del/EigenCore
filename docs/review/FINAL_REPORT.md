# EigenCore review: final report

**For:** Khubaib (owner). **Reviewer:** external review, read-only, written for the owner and for hoplite.
**Checkout reviewed:** `main` at `c92e3af` (2026-11-08). **Date:** 2026-11-09.
**Method:** every doc and every source file was read; the six component reports in [`research/`](research/) go deeper and cite their sources. Probes that exercised the code ran out of tree on Linux; nothing in `src/` was changed. No benchmark was run: this sandbox is not your laptop, and every speed figure below is arithmetic or a third-party report, labelled as such.

Labels used everywhere in this review:

- **SHIPPED**: code exists on `main`, runs, and is exercised by a test or a CLI path.
- **REPORTED-ONLY**: code exists and prints numbers or plans, but nothing reaches the llama.cpp call.
- **PLANNED**: words in docs only.
- Evidence tags: **[verified]** read in code or in an opened source, **[inferred]** reasoned from verified facts, **[unverified]** not checked.

---

## 1. The short version

1. **The 30-40B dense target does not fit 8-16 GB of RAM at 4-bit, and no amount of kernel work changes that.** A 35B model at Q4_K_M is about 21 GB of weights before KV cache and runtime. On a 16 GB laptop it can only run by paging from SSD, which puts decode well under 1 tok/s (estimate). Even with enough RAM, an i7-10610U's 45.8 GB/s peak caps a 21 GB dense model at about 2 tok/s. [inferred from verified specs]
2. **Decode on this hardware is bandwidth-bound, not compute-bound**, by roughly 2x. Faster SIMD or assembly does not raise the decode ceiling; it can help prefill. llama.cpp already ships hand-tuned AVX2 quantized kernels. [inferred; see `research/kernels.md`]
3. **The realistic route to "30B-class on a laptop" is a Mixture-of-Experts model** (for example Qwen3-30B-A3B: ~3.3B active of 30.5B), which llama.cpp already runs. Each token reads ~1.8 GB instead of ~21 GB. It still has to be held in RAM or paged from SSD, so the owner must measure it. [verified model card; tok/s inferred]
4. **What EigenCore owns today is the judgement layer** (profile, fit, benchmark harness, tool policy, chat UI), and several parts of it give wrong answers today. The most damaging: the KV-cache fallback underestimates by about 15-30x, the CLI always uses that fallback, the Q4_0 reference kernel and its test are both wrong, the tool timeout does not bound time, and the agent path runs without the policy.
5. **Layer skipping, activation sparsity and speculative decoding are REPORTED-ONLY.** None reaches the backend. The speculative verifier is also not a correct algorithm, and two "speedup" printers turn unrelated numbers into speedups.

---

## 2. Runs today through llama.cpp vs custom code here vs planned only

| Concept | Runs today through llama.cpp (llama-cpp-python) | Custom code that exists in this repo | Planned only |
| --- | --- | --- | --- |
| GGUF loading, tokenizing, matmul, attention, sampling | **Yes**, all of it: `engine/inference.py:85-94` calls `Llama(...)` and `create_chat_completion` | None; EigenCore owns no transformer math | PLANNED: own backend behind `BackendProtocol` |
| SIMD kernels (AVX2/AVX-512/NEON) | **Yes**, ggml's own quantized dot products and repacking | `kernels/reference.py`: NumPy scalar RMSNorm, Q8_0 and Q4_0 dequant, Q8_0 dot. Q4_0 is **wrong** (see 3.2). Not called by inference. | SIMD/C/assembly kernels, runtime dispatch |
| Hardware profile | n/a | SHIPPED: `hal/profiler.py` ISA detection, RAM, memcpy bandwidth probe (underestimates), thread and context recommendation (context always 2048 on 8-16 GB, unit bug) | Multithreaded bandwidth test |
| Model fit | n/a (llama.cpp has its own `--fit` / `llama-fit-params`, not used) | SHIPPED: `fit/estimator.py` weights + reserve + KV + margin, verdicts, max-context solver. CLI never passes geometry, so every CLI estimate uses the broken heuristic. | GGUF geometry, VRAM, compute buffer |
| Benchmark harness | Calls the backend and times it | SHIPPED: `benchmark/harness.py`, JSON report with SHA-256, TTFT, decode tok/s, quality cases. Counts stream chunks, not tokens; includes the cold first run. | Published runs on named hardware |
| KV cache | **Yes**, fp16 by default; llama-cpp-python also exposes `type_k`/`type_v`, `flash_attn` (not used) | Estimate only | KV quantization choices in UI |
| mmap / SSD paging | **Yes**, `use_mmap=True` is the llama-cpp-python default, so a too-big model pages from disk | Nothing; no warning when the model exceeds RAM | PLANNED: "SSD streaming" (no design in docs) |
| GPU offload | Supported by llama.cpp | `n_gpu_layers=0` hard-coded | GPU/VRAM fit, `n_gpu_layers` wiring |
| Speculative decoding | llama.cpp supports draft models; llama-cpp-python ships `LlamaPromptLookupDecoding` (not used) | REPORTED-ONLY: `engine/speculative.py` initialised, never called; verifier is not a correct algorithm | Wiring |
| Layer skipping | No per-token skip API | REPORTED-ONLY: `engine/layer_skip.py` plans + invented speedup/loss constants | Dynamic skip; offline pruning not planned |
| Activation sparsity | No | REPORTED-ONLY: `engine/sparse_inference.py` never fed activations; `analysis/sparsity.py` measures output **logits**, not neurons | Neuron-level sparse execution |
| MoE ("activate only part of the model") | **Yes**, llama.cpp runs MoE GGUFs; registry lists none | Nothing MoE-aware in fit or registry | Not in docs as the main path (should be) |
| Context compression | n/a | `context/manager.py` (attention-sink-aware trimming, SQLite) exists; quality not evaluated | Evaluation |
| Agent tools | n/a | SHIPPED: ReAct loop, typed tools, GBNF builder, policy engine, audit log. Policy is **opt-in**, timeout does not bound time, handlers run in-process. | Process isolation, approval cards |
| Terminal chat | Streams from backend | SHIPPED core: `tui/chat.py`, plain mode, slash commands, fit header. Crashes on `[/]` in model text, passes terminal escapes, traceback on Ctrl-C. | Full-screen layout, onboarding, tools in chat |
| Training | n/a | `training/trainer.py` QLoRA wrapper (torch/peft/bitsandbytes) | CPU QLoRA is [unverified]; not evaluated |

---

## 3. What is overclaimed or broken, ranked by impact

Impact = how likely it misleads a user or audience, times how bad the result is.

### 3.1 High

1. **"30-40B on 8-16 GB" as a target, without the arithmetic next to it** (README reality check is good; `docs/architecture/competitive-landscape.md` and `docs/research/emergent-gaps.md` are not). See section 4. Risk: the presentation and README promise something physics rules out for dense models.
2. **KV-cache fallback is ~15-30x too low and is the path the CLI always takes.** `fit/estimator.py:132-133` and `hal/profiler.py:233-236` use 0.0005 GB per 1K tokens per 1B params. Llama-3.1-8B at 8K context: geometry gives 1.00 GiB, heuristic 0.033 GB (~30x low); Qwen2.5-32B ~15x low. The docstring calls it "conservative". The CLI never builds a `ModelGeometry`, so `eigencore profile/models` verdicts rest on it. Also `head_dim = hidden/heads` is wrong for Gemma-family models (use `<arch>.attention.key_length`), and q8_0 is 34 bytes per 32 values, not 1 byte each. [verified, `research/engine-fit.md`]
3. **Profiler unit mix-up**: GiB/GB mixing makes the recommended context 2048 on every machine under ~38 GB, so "512-8,192" in docs and slides never happens on the target. [verified]
4. **Q4_0 reference dequantizer is wrong, and its test encodes the same mistake.** `kernels/reference.py:84-89` interleaves nibbles; ggml puts the low nibble of byte j at element j and the high nibble at j+16. Max error vs gguf-py: 5.50. A "reference" that is wrong would certify a wrong SIMD kernel. Q8_0 matches exactly. [verified by round-trip against `gguf.quants`]
5. **Agent safety claims exceed the code.** [verified by probe, `research/tui-sandbox.md`]
   - "Deny-by-default": `ToolExecutor(policy=None)` is the default; `Forge.agent()` (`forge.py:177`) and `ReActAgent` build executors without a policy, so an EXECUTE tool runs. `default_policy` is exported and never used.
   - Path confinement only checks parameters whose names are on a fixed list; a READ tool with a `src` argument read `/etc/passwd` under `default_policy(workspace_root=...)`. The handler receives the raw string, not the resolved path (check-then-use).
   - The timeout returns `TIMEOUT` but blocks until the handler finishes (`with ThreadPoolExecutor` waits on exit): a 3 s handler with `timeout=0.2` took 3.0 s.
6. **Speculative decoder is not speculative decoding.** `engine/speculative.py:172-179` accepts when p/q >= a fixed threshold (no randomness, no residual resampling), so it does not preserve the target distribution; the correction token is always id 0; the docstring claims the draft "shares the same GGUF file (different quant)" and that CPU verification "is still sequential", both wrong. The "3-4x" claim needs acceptance >= 0.8, near-free drafts and free verification at once. [verified, `research/specdec.md`]

### 3.2 Medium

7. **`eigencore analyze` measures output-logit sparsity and prints a "theoretical CPU speedup"** from it (`analysis/sparsity.py:100-135`, `cli/main.py:296,367`). Logit sparsity says nothing about FFN neurons. The slides v1 repeated "measure activation sparsity". [verified]
8. **"70-90% of neurons are near-zero in dense models"** (`emergent-gaps.md`, `sparse-activation-and-routing.md`, `sparse_inference.py` docstring). True for ReLU models (OPT); modern SwiGLU models (Llama, Qwen, Mistral) have little natural sparsity (ProSparse, TurboSparse). The "branch prediction makes skipping free on CPU" argument is wrong: branchy code defeats SIMD; the CPU win would come from not reading weight rows, which needs a predictor and a neuron-bundled weight layout. PowerInfer is cited as "Du et al., 2024"; it is Song et al. (arXiv 2023, SOSP 2024). [verified, `research/sparse.md`]
9. **Invented constants presented as estimates**: layer skip "<2% perplexity at 20-30% skip" (no source), `expected_speedup = 1/(1-skip_rate)`, `estimated_loss = (1-importance)*0.01`; competitive matrix "5-10x compounding". [verified]
10. **Competitive landscape is factually wrong in places**: llama.cpp has speculative decoding (CPU included) and fine-tuning work; Ollama does pick GPU layers from a memory estimate. [verified for llama.cpp speculative; Ollama per `research/engine-fit.md`]
11. **README quick start says `pip install eigencore`**; the name is not on PyPI (404 on 2026-11-09). **No LICENSE file** although README and `pyproject.toml` say MIT [verified]; **fixed in this PR** (MIT, © 2026 Khubaib Nazeer).
12. **Chat TUI robustness**: model text is parsed as Rich markup (a `[/]` crashes the session), raw escape sequences reach the terminal, Ctrl-C exits with a traceback, a mid-stream `AttributeError` regenerates and double-prints. Header says `tools: read-only (policy engine attached in P4)` although chat exposes no tools. [verified by probe]
13. **Metadata hard-coded to `llama.*` keys** (`inference.py:101-102`): Qwen, Phi-3, Gemma GGUFs get 0 layers, so even the reported-only optimizers are never built for 5 of the 6 registry models. [verified]

### 3.3 Low

14. Bandwidth probe: single-threaded Python copy with fresh pages, doubled; returned 8.1 GB/s in this sandbox against a 45.8 GB/s spec for the i7-10610U. Treat its tok/s figure as noise. [verified]
15. Weights at 0.55 bytes/param for Q4_K_M; real files run ~4.8-4.9 bits/weight (~0.61 B/param). Slides v1 inherited 0.55. [verified via gguf-parser data, `research/engine-fit.md`]
16. Harness counts streamed chunks as tokens and includes the cold first run in the median. [verified]
17. Windows ISA fallback can claim AVX-512 on 12th-14th gen i7 (fused off). [verified in source, `research/kernels.md`]

### 3.4 Claims in my own earlier output (slides v1) that this review corrects

- "≈ 19 GB" for 35B Q4_K_M: use ≈ 21 GB (0.61 B/param).
- "38 GB/s example bandwidth": replace with the i7-10610U's published 45.8 GB/s peak, labelled peak.
- "Context 512-8,192 sized to RAM": on 8-16 GB it is always 2048 (bug).
- "Deny-by-default" on the architecture slide: true only when a policy is passed.
- "analyze: measure activation sparsity": it measures logits.
- "Differential correctness tests" for kernels: the tests are self-consistent, not differential, and Q4_0 is wrong.
- "Accepted tokens can keep the target's output distribution": true of the published method, not of this repo's verifier.

All are fixed in [`../presentation/SLIDES_V2.md`](../presentation/SLIDES_V2.md).

### 3.5 hoplite's reported status, checked

| hoplite said | Finding |
| --- | --- |
| "Every concept is labeled SHIPPED, REPORTED-ONLY or PLANNED" | **Not true on `main`.** No file uses these labels; README uses Done / Audit done / Core shipped / In progress. This PR introduces the labels. |
| Remaining: GGUF geometry parsing | Agree, and it is more urgent than stated: it is a correctness bug in what the CLI prints today, not a refinement. |
| GPU/VRAM fit and `n_gpu_layers` | Agree it is missing; disagree on priority for a CPU-first, 8-16 GB laptop target. Do it after MoE fit and mmap awareness. |
| OS process isolation for tool handlers | Agree. But first make the policy the default and fix the timeout; isolation on top of an opt-in policy protects nothing. |
| TUI approval cards | Agree, after isolation (design doc says the same). |
| Speculative decoding wiring | Disagree with wiring the current module. Delete or rewrite `verify_draft`; wire llama-cpp-python's prompt-lookup decoding first. |
| Full-screen TUI + onboarding | Agree, lower priority than the escape/markup/Ctrl-C fixes. |
| SIMD/assembly only if benchmarks show a bottleneck | Agree, with one addition: fix the Q4_0 reference first, or the differential tests are worthless. |
| Not doing: rename to "eigen", sandbox-only llama_cpp test failures, ruff >= 0.15 | Fine. The name stays EigenCore in this PR too. |
| Missing from hoplite's list | KV heuristic, profiler unit bug, Q4_0 bug, policy not default, timeout, path bypass, TUI crash/escapes, `analyze` meaning, `llama.*` metadata keys, PyPI line, LICENSE, chunk-vs-token counting. |

---

## 4. The 30-40B claim, with the arithmetic

All figures are arithmetic or published specs. None is a measurement.

**Weights.** bytes = params x bits-per-weight / 8. Q4_K_M files average ~4.85 bpw (~0.61 B/param).

| Model | Q4_K_M weights | Q8_0 (~8.5 bpw) | F16 |
| --- | --- | --- | --- |
| 30B dense | ≈ 18.3 GB | ≈ 31.9 GB | 60 GB |
| 35B dense | ≈ 21.3 GB | ≈ 37.2 GB | 70 GB |
| 40B dense | ≈ 24.4 GB | ≈ 42.5 GB | 80 GB |

Add a runtime reserve (EigenCore uses 2 GB) and KV cache. KV per token = layers x kv_heads x head_dim x 2 x bytes. For a Qwen2.5-32B-class geometry (64 layers, 8 KV heads, 128 dim, fp16) that is 256 KiB per token: 1 GiB at 4K context, 8 GiB at 32K. [inferred from published geometry]

**Fit verdict.** On 16 GB (≈ 13-14 GB actually free), 30-40B at Q4_K_M **does not fit**. On 8 GB it is far off. To fit 35B in ~12 GB of weights needs ≤ ~2.7 bpw (IQ2/Q2-class), which costs measurable quality. [inferred]

**Decode ceiling if it did fit.** Each generated token of a dense model reads every weight once.

tok/s ≤ sustained bandwidth / weight bytes = 45.8 GB/s (i7-10610U published peak) / 21.3 GB ≈ **2.1 tok/s**, and real decode lands below a peak-bandwidth ceiling. [inferred]

**When it does not fit (mmap paging from SSD).** Estimates from `research/layerskip-ssd.md`, assuming 50 GB/s RAM, 1-7 GB/s effective SSD reads:

| RAM | SSD 1 GB/s | 3 GB/s | 7 GB/s |
| --- | --- | --- | --- |
| 16 GB, best case | 0.10-0.13 tok/s | 0.30-0.36 | 0.64-0.76 |
| 8 GB, best case | 0.06-0.07 | 0.18-0.20 | 0.41-0.46 |

**Realistic range for dense 30-40B on this laptop: roughly 0.05-0.8 tok/s**, depending on the SSD and page-cache behaviour. That is a "it loads" result, not an interactive one. [inferred]

**MoE alternative.** Qwen3-30B-A3B activates ~3.3B parameters per token: ≈ 2 GB read per token at Q4. In-RAM ceiling ≈ 45.8 / 2 ≈ 20+ tok/s; the whole model (~18-19 GB at Q4_K_M) still has to be resident or paged, so on 16 GB it pages, and the estimate drops to roughly 2-12 tok/s depending on cache hit rate. One third-party llama.cpp patch (Swap-MoE) self-reports 7.0 tok/s for a 35B-A3B Q4_K_M on a 13th-gen i7, 16 GB, NVMe, no GPU; that is a different CPU and a patched loader. [verified page; numbers self-reported]

**Which techniques could change the picture, and which only look good on paper**

| Technique | Changes bytes read per token? | Verdict for this target |
| --- | --- | --- |
| MoE model choice | Yes, ~10x fewer | **Real.** The only route to "30B-class" that is plausibly interactive on 16 GB. Needs MoE-aware fit and measurement. |
| Lower-bit quant (Q3/IQ2) | Yes | **Real**, with quality cost; must be measured with `llama-perplexity` and the task suite. |
| KV-cache quantization (`type_k/type_v` q8_0) | Reduces KV memory, not weight reads | Real for context length; small effect on tok/s. |
| Speculative decoding | Fewer full passes per accepted token | **Possibly real** on CPU (third-party reports range from slower to ~1.7x); prompt-lookup is the cheapest test. Measure. |
| SSD streaming of a dense model | No, adds slower reads | Makes it load, not run usefully. Real only for MoE. |
| Dynamic per-token layer skipping | Would, but no backend API and quality loss | **Paper only** here. Offline pruning (ShortGPT-style) to a smaller GGUF is the testable version. |
| Neuron-level activation sparsity on dense SwiGLU models | Would, with a custom kernel and layout | **Paper only** today. Needs ReLU-fied models (PowerInfer/TurboSparse) or a TEAL-style kernel whose gains are shown on GPU, not CPU. |
| Assembly/SIMD kernels | No | Does not move the decode ceiling. Helps prefill a little; see section 5. |

---

## 5. Bandwidth-bound or compute-bound, and what assembly adds

- **Decode (batch 1, matrix-vector):** about 2 FLOPs per weight over ~0.56 bytes per weight ≈ 3.6 FLOP/byte. The i7-10610U's ridge point is ≈ 384 GFLOP/s / ~40 GB/s ≈ 9.6 FLOP/byte, and its realistic quantized-dot throughput is ~150 G weights/s against ~71 G weights/s that memory can supply. **Decode is bandwidth-bound with about a 2x margin.** At 15 W the CPU may throttle and the margin shrinks, so thread count matters, but a faster inner loop does not raise tok/s. [inferred; arithmetic in `research/kernels.md`]
- **Prefill (matrix-matrix over N prompt tokens):** compute-bound from N ≈ 3. This is where kernel quality shows (TTFT).
- **What llama.cpp already has:** hand-written AVX2/AVX-VNNI/NEON quantized dot products, runtime weight repacking, and the llamafile matrix kernels. A published change that sped up its CPU matmul gave +18% prompt processing and +2.7% generation. [verified, PR #8713]
- **So does an assembly layer add anything?** For decode speed, very little over ggml. It adds value in three places: (1) as a learning and teaching artefact with differential tests against ggml, (2) for an operation ggml lacks (a TEAL-style activation-sparse matvec, a new quant format, a fused op), (3) as an upstream contribution to ggml, which is how a speedup would actually reach users. Writing a parallel engine in assembly is high effort for no decode gain. **Disagreement stated plainly:** the owner's "assembly layer for the heavy math" is the wrong first lever for this goal; the lever is bytes per token (model choice, quantization, MoE). The risk of the assembly-first approach is months spent re-deriving ggml's kernels while the fit and safety bugs above ship to users.

---

## 6. What is solid

- The README's "Reality check" paragraph and the honest P3 audit (`tests/test_optimization_audit.py` proves flags do not change backend kwargs). [verified]
- The fit estimator's shape: separate weights, reserve, KV, margin, three verdicts, max-context solver. The formula with real geometry is correct. [verified]
- The benchmark report's shape: schema version, SHA-256 model identity, hardware snapshot, TTFT separated from decode, deterministic quality cases. [verified]
- The policy engine's shape: capabilities, deny-unless-allow-listed when attached, audit log with redaction, symlink and `..` rejection for named path params. [verified]
- Q8_0 reference matches gguf-py bit for bit. [verified]
- CI on Ubuntu and Windows, Python 3.10 and 3.12; 173 tests. [verified count]
- The docs already separate implemented and proposed in most places (system overview, roadmap, terminal-chat). [verified]

## 7. What is missing

- MoE-aware fit (resident bytes vs bytes read per token) and MoE models in the registry.
- Any warning when a model exceeds RAM and will page from SSD; `use_mmap`/`use_mlock` controls.
- Exposed `type_k/type_v`, `flash_attn`, `n_ubatch` choices.
- Real geometry from GGUF metadata; weight size from the file.
- A multithreaded bandwidth test.
- A published benchmark run on named hardware (none exists).
- Correct install instructions, LICENSE and community files: all added in this PR.

---

## 8. Updated plan, ordered by impact

Each item names its acceptance test. "Owner" means it needs your machine.

### Stage A: make what ships tell the truth (correctness, days)

1. **Fit from real metadata.** Read `<arch>.block_count`, `.attention.head_count_kv`, `.attention.key_length/value_length`, `.context_length` from `Llama.metadata` (or `gguf` without loading weights); weights from file/tensor bytes; exact KV type sizes; fix GiB/GB; CLI passes geometry; heuristic either removed or labelled "unknown" with a deliberately high value. *Accept:* fixture GGUFs (Llama-3.1-8B, Qwen2.5-7B, a Gemma) give KV within 1% of the geometry formula; estimator total within 10% of `llama-fit-params` / peak RSS (owner).
2. **Fix Q4_0 reference and its test.** Take raw 18-byte blocks; compare with `gguf.quants.dequantize`. *Accept:* bit-exact against gguf-py on random and model-derived blocks for Q4_0 and Q8_0.
3. **Make the agent path safe by default.** `ToolExecutor` requires a policy (default `default_policy()`); `Forge.agent`/`ReActAgent` pass one; tools declare path params explicitly; handlers receive the resolved path; timeout kills a child process. *Accept:* EXECUTE tool denied through `Forge.agent` with no arguments; `src="/etc/passwd"` denied; a 3 s handler with 0.2 s timeout returns in < 0.5 s and the child is gone.
4. **Chat robustness.** Escape Rich markup in model text, strip C0/C1 control and OSC sequences, catch Ctrl-C to stop generation, no regenerate on mid-stream errors, honest tools label. *Accept:* tests with `[/]`, `\x1b]0;x\x07`, `\x1b[2J`, KeyboardInterrupt mid-stream, in both Rich and plain modes.
5. **Quarantine the speculative verifier and the speedup printers.** Delete or rewrite `verify_draft` as Leviathan rejection sampling over full distributions; remove `expected_speedup`, `potential_speedup` and "theoretical CPU speedup" from user output; rename `analyze` output to "logit sparsity". *Accept:* chi-square test that accepted-token frequencies match the target distribution; no CLI output contains "speedup" without a measured baseline.
6. **Metadata by architecture.** Replace `llama.*` keys with `general.architecture` prefix lookup. *Accept:* Qwen and Phi GGUF fixtures produce non-zero layers.
7. **Docs and packaging truth.** This PR covers docs, LICENSE and the v2 brand SVGs. Remaining: commit the v2 PNG/ICO exports; README installs from git until a PyPI release exists.

### Stage B: measure (owner, one evening)

8. Run the benchmark list in section 9. Commit the JSON under `benchmarks/<machine>/` with the machine named. This is the first evidence the project has.

### Stage C: the 30B-class goal, done the realistic way

9. **MoE first-class:** add one MoE GGUF (Qwen3-30B-A3B class) to the registry; fit estimator reports resident bytes and active bytes; profile warns "will page from SSD" when resident > free RAM; expose `use_mmap`/`use_mlock`. *Accept:* owner measures decode tok/s at 4K context on 16 GB, warm, 5 runs, median and spread.
10. **Prompt-lookup speculative decoding** via `llama_cpp.llama_speculative.LlamaPromptLookupDecoding`. *Accept:* greedy output identical with it on and off; enabled by default only if median speedup >= 1.2x on the owner's laptop for the chat and code suites.
11. **KV and attention options:** expose `type_k/type_v` (f16, q8_0), `flash_attn`, `n_ubatch`; fit estimator uses the chosen KV type. *Accept:* owner's `llama-bench -ctk/-ctv/-fa` sweep plus quality suite.
12. **GPU/VRAM fit and `n_gpu_layers`** for owners who have a GPU. *Accept:* fit verdict matches whether load succeeds on a fixture set.

### Stage D: product polish

13. Process isolation (child process per tool call; POSIX rlimits and process groups; Windows Job Object; optional Landlock on Linux). *Accept:* isolation tests in `research/tui-sandbox.md`.
14. Approval cards wired to the policy engine, then tools in chat. *Accept:* no tool runs without an approval event in the audit log.
15. Adaptive TUI and onboarding per [`../design/TUI.md`](../design/TUI.md) (status, technology choice, six screens, work order) and the design handbook. *Accept:* 80x24, 120x40, plain mode, SSH, Windows Terminal, `NO_COLOR`.

### Stage E: research, behind flags, each with a kill criterion

16. **Kernels:** Q4_K reference; C intrinsics library (optionally NASM objects) behind `BackendProtocol` with runtime dispatch, called via cffi/nanobind; differential tests vs gguf-py. *Kill criterion:* if a Q4_K matvec does not reach >= 80% of measured STREAM bandwidth, keep it as a teaching artefact and do not route inference through it.
17. **Offline layer pruning** (ShortGPT-style block-influence ranking) producing a smaller GGUF. *Kill criterion:* perplexity and task-suite loss above the chosen threshold.
18. **Activation sparsity:** either evaluate PowerInfer with a TurboSparse/ProSparse model, or prototype a TEAL-style sparse matvec in the kernel library. *Kill criterion:* no CPU speedup at equal quality.
19. **Dense-model SSD streaming**: do not build. Keep mmap paging as a documented, warned fallback.

---

## 9. Benchmark runs the owner must perform

On the presentation laptop, plugged in, performance power plan, nothing else running. Record CPU model, RAM size and speed, SSD model, OS, llama.cpp / llama-cpp-python version and build flags for every run. Report the median of 5 and the spread; keep the raw JSON.

1. **Bandwidth:** Intel MLC `--max_bandwidth` or STREAM Triad at 1, 2, 4, 8 threads; compare with `eigencore profile`.
2. **Baseline decode and prefill:** `llama-bench -m <model> -p 512 -n 128 -t 2,4,8 -r 5 -o json` for a 7-8B Q4_K_M (fits) and compute achieved GB/s = model bytes x tg tok/s.
3. **KV and attention:** `llama-bench -ctk f16,q8_0 -ctv f16,q8_0 -fa 0,1 -d 0,4096`.
4. **Over-RAM dense model:** a 32B Q4_K_M with mmap on 16 GB; record tg tok/s, load time, page faults, SSD read rate. Expect it to be slow; the number is the point.
5. **MoE model:** Qwen3-30B-A3B class Q4_K_M (and a Q3 variant) at 4K context; same metrics.
6. **SSD:** `fio` sequential and 4K random read, plus mmap-style read, so the paging numbers can be explained.
7. **Fit check:** `llama-fit-params --model <gguf>` and peak RSS (`/usr/bin/time -v` or Task Manager peak working set) vs `estimate_fit()` at 2K, 8K, 32K.
8. **EigenCore harness vs llama-bench** on the same model and threads, warm and cold separately, after the token-count fix.
9. **Speculative (after item 10):** prompt-lookup on/off, greedy, chat and code suites.
10. **Quality:** `llama-perplexity` on a fixed text for each quant used, plus the harness quality suite.

Until these exist, every tok/s figure in docs and slides stays labelled "ceiling estimate".

---

## 10. Disagreements, stated plainly

| Owner's approach | Why I disagree | Alternative | Risk if unchanged |
| --- | --- | --- | --- |
| Dense 30-40B on 8-16 GB as the headline | Weights alone exceed RAM; decode then runs from SSD at < 1 tok/s (estimate) | Headline "30B-class MoE on 16 GB, measured"; dense 7-14B as the interactive default | Audience or users test it and conclude the project overclaims |
| Assembly/SIMD layer for the heavy math as a core differentiator | Decode is bandwidth-bound; ggml already has tuned AVX2 kernels | Kernel work as a measured, differential-tested research track; contribute wins upstream | Months of effort with no tok/s change |
| Per-token layer skipping | No backend hook; published results need calibration or fine-tuning; quality risk | Offline pruning evaluated with perplexity | Silent quality loss if ever enabled |
| Activating specific neurons for specific tasks | Dense SwiGLU models are not naturally sparse; needs special models or kernels | Use MoE models (the trained version of this idea) | Research without a path to users |
| SSD streaming | Valuable for MoE, not for dense | Treat mmap paging as a warned fallback; invest in MoE | Building a slower mmap |
| Speculative decoding as already implemented | Algorithm is incorrect | Prompt lookup via llama-cpp-python first | Output quality changes silently once wired |

## 11. Rebrand and licence (done in this PR)

- **Brand v2** in `assets/brand/` (SVG sources): flat mark (one shadow instead of ~30 stacked copies, no gradients or glow, three traces, new dotted *eigen-axis* into the core), small mark/favicon without traces, lockups for dark and light grounds, outlined Space Grotesk Bold wordmark, static and animated banner (plays once, honours `prefers-reduced-motion`). Spec and rules: `docs/design/DESIGN_HANDBOOK.md`. Name unchanged.
- **Raster exports** (PNG, ICO) are not in this PR, by the no-binaries rule; they are in the v2 kit zip the owner holds, and the regeneration command is in `assets/brand/README.md`. Until committed, the PNG/ICO files in the repo are v1.
- **LICENSE**: MIT, © 2026 Khubaib Nazeer, matching `pyproject.toml`.
- **TUI**: current status, bugs, technology decision and v2 screens in `docs/design/TUI.md` with SVG mock-ups in `docs/design/tui/`.

## 12. Component reports

- [Inference engine and fit estimator](research/engine-fit.md)
- [Sparse activation](research/sparse.md)
- [Layer skipping and SSD streaming](research/layerskip-ssd.md)
- [Speculative decoding](research/specdec.md)
- [Kernels](research/kernels.md)
- [TUI and agent sandboxing](research/tui-sandbox.md)

Each marks its conclusions [verified], [inferred] or [unverified] and lists its sources.
