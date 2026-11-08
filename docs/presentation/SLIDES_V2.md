# EigenCore talk: slides v2 (text)

**Talk:** "Why Running Large LLMs Locally Is a Memory-Bandwidth Problem", EE2013 Computer Architecture, FAST-NU Lahore.
**What this is:** the slide-by-slide text for version 2 of the deck, plus a change list against version 1. The deck itself is a binary `.pptx` and is **not** committed; version 1 stays untouched outside the repo, and version 2 is built from this text as a separate file.
**Kept from v1:** 13 slides, same order, same per-slide timings (≈ 5.5 min total), same motion (fades and morphs, builds auto-play after each transition, clicks only advance).
**Changed:** the v2 identity from [`../design/DESIGN_HANDBOOK.md`](../design/DESIGN_HANDBOOK.md), corrected numbers and claims from [`../review/FINAL_REPORT.md`](../review/FINAL_REPORT.md), and a status tag on every claim.

Tags: **[SHIPPED]** code on `main` that runs · **[REPORTED-ONLY]** code that reports plans or numbers but never reaches llama.cpp · **[PLANNED]** docs only · **[ESTIMATE]** arithmetic, not a measurement. No benchmark number appears in this deck: none has been run on named hardware yet.

## Visual changes for the whole deck

- Mark: replace the v1 gradient logo with the v2 flat mark (`docs/design/eigencore-mark-v2.svg`) on slides 1, 7 and 13. The opening video keeps its timing; its last frame cross-fades to the v2 mark. Re-render the intro once the owner approves the mark.
- Grounds: unchanged from v1 (stone `#ECE9E2` content, graphite `#1F2226` anchors), already matching the handbook.
- Accent: teal `#1F6B66` stays the single content accent; core indigo appears only inside the mark. Clay `#B04E2C` only for "does not fit", REPORTED-ONLY outlines and "not an OS sandbox". No amber or gold anywhere.
- Status chips (new): small rounded chips in the top-right of each claim block. SHIPPED = solid teal, REPORTED-ONLY = clay outline, PLANNED = grey hatched, ESTIMATE = muted italic text. Chips fade in with their block; they add no time.

---

## Slide 1 · Opening (≈ 20 s)

- Title: Why Running Large LLMs Locally Is a Memory-Bandwidth Problem
- Subtitle: Weights, KV Cache & CPU Inference · EigenCore
- Presenter line: (blank until the owner supplies names; do not invent)

**Notes:** Good morning. This is EigenCore, a CPU-first toolkit for running language models on your own machine. The question behind it: why does a laptop that happily runs a browser and an IDE choke on a 35-billion-parameter model? The short answer is in the title. It's less about raw compute and more about memory: how much the machine can hold, and how fast it can move it.

## Slide 2 · The problem (≈ 25 s)

- Title: A downloaded model is not a running model
- Lead: Whether a GGUF file is usable depends on three limits of your machine, not on the file alone.
- Cards: 01 Capacity: do the weights fit in RAM at all? · 02 Context: how much KV cache can you afford? · 03 Bandwidth: how fast can memory feed the cores?

**Notes:** unchanged from v1.

## Slide 3 · Weights (≈ 30 s)

- Title: The weights alone outgrow 16 GB
- Equation: 35B params × ≈ 0.61 B/param ≈ 21 GB  [ESTIMATE]
- Captions: parameter count · bytes per param, Q4_K_M (≈ 4.85 bits, typical GGUF file) · weights only
- Block chart (1 block = 1 GB, hatched past 16 GB): F16 70 GB · Q8_0 ≈ 37 GB · Q5_K_M ≈ 25 GB · Q4_K_M ≈ 21 GB
- Footer: quantization shrinks the problem; it does not remove it

**Notes:** Start with capacity. Weight memory is roughly parameter count times bytes per parameter. Q4_K_M files average about 4.85 bits per weight, so about 0.61 bytes. 35 billion times 0.61 is about 21 GB. That's arithmetic, not a measurement. In F16 the same model is 70 GB; in Q8_0 about 37. Even the 4-bit version runs past the line of a 16 GB machine before we've counted anything else.

## Slide 4 · Memory budget (≈ 35 s)

- Title: Memory is more than weights
- Formula: headroom = available RAM − (weights + runtime reserve + KV cache)  [SHIPPED: `fit/estimator.py`]
- Container label: 16 GB installed · usable RAM is less
- Overflow label: ≈ 5 GB over on weights alone, before reserve and KV cache
- Bars: Weights ≈ 21 GB (35B Q4_K_M) · Runtime 2 GB · KV cache grows · Margin 1 GB
- Verdicts: fits_with_margin (headroom ≥ 1 GB) · tight (0 to 1 GB) · does_not_fit (< 0) ← this machine

**Notes:** The weights are only the first line of the budget. EigenCore's fit estimator adds a 2 GB runtime reserve, the KV cache, and wants a 1 GB margin. Against 16 GB, we're about 5 GB over on weights alone, so the verdict is does_not_fit. With llama.cpp's default memory-mapping, a model like this will still "load" and then read weights from the SSD on every token, which is far slower. The estimator also solves backwards for the largest context that fits.

## Slide 5 · KV cache (≈ 30 s)

- Title: Context length has a memory price
- Formula: KV bytes = layers × kv_heads × head_dim × 2 (K,V) × bytes × ctx  [SHIPPED in the estimator]
- Example box: EXAMPLE GEOMETRY · LLAMA 3.1 8B (model config) · 32 layers · 8 KV heads · 128 dim · fp16 = 128 KiB per token  [ESTIMATE]
- Lead: Cached keys and values spare recomputing attention over earlier tokens. The memory is the price.
- Stepped chart: 4K 0.5 · 8K 1 · 16K 2 · 32K 4 · 64K 8 · 128K 16 GiB; dashed line: 16 GB, a whole laptop, cache alone
- Small honest line (new): Today's CLI falls back to a rough per-parameter guess that reads ~30× low; reading real geometry from the GGUF is the next fix.  [PLANNED]

**Notes:** During generation the model keeps keys and values for every previous token, so the cache grows linearly with context. With Llama 3.1 8B's published geometry at fp16 that's 128 KiB per token: half a gigabyte at 4K, 16 GiB at 128K. One honest note: EigenCore's formula is right, but its command line currently falls back to a per-parameter guess that underestimates this by about thirty times. Reading the real geometry from the model file is the next fix.

## Slide 6 · Bandwidth (≈ 35 s)

- Title: Every new token reads the whole model
- Formula: decode tok/s ≤ memory bandwidth ÷ weight bytes
- Big numbers: 45.8 GB/s ÷ 21 GB ≈ 2 tok/s
- Captions: i7-10610U peak (Intel spec) · 35B Q4_K_M weights · CEILING ESTIMATE · NOT MEASURED
- Ruler: 0 · 0.5 s · 1 s · token 1: stream all 21 GB · token 2: stream all 21 GB
- Footer: Bandwidth-bound, not compute-bound: about 3.6 FLOPs per byte read, below what the cores could do. Faster arithmetic does not raise this ceiling.

**Notes:** The key slide. To produce one token, a dense model reads essentially all of its weights, so bandwidth divided by weight bytes is an upper bound. This laptop CPU's published peak is 45.8 GB/s; divided by 21 GB, that's about 2 tokens per second as a ceiling, and real decode lands below it. Each token is one full pass over 21 GB, about half a second. The arithmetic intensity is about 3.6 FLOPs per byte, under the point where these cores would run out of compute, so the cores mostly wait on memory. That's why faster SIMD alone doesn't fix decode.

## Slide 7 · Meet EigenCore (≈ 25 s)

- Title: Meet EigenCore
- Tagline: Know what your machine can run, before you run it.
- Steps: profile · read the hardware [SHIPPED] → fit · budget the memory [SHIPPED] → run · infer via llama.cpp [SHIPPED, via llama-cpp-python] → measure · benchmark evidence [SHIPPED harness, no published runs yet]
- Footer: Python 3.10+ · MIT · GGUF models from Hugging Face · the transformer math is llama.cpp's

**Notes:** EigenCore is not a new inference engine; it's the judgement around one. It profiles the hardware, budgets memory before you load, runs inference through llama.cpp, and measures what happened. The math is llama.cpp's. EigenCore's own parts are profiling, fit estimation, the benchmark harness, and tool guardrails.

## Slide 8 · Hardware profile (≈ 25 s)

- Title: It starts by reading the machine
- Command: `$ eigencore profile`  [SHIPPED]
- ISA staircase: 128-bit SSE2 · SSE4.1/4.2 · 256-bit AVX · AVX2 · FMA · F16C · 512-bit AVX-512 F · BW · VNNI · matrix tiles AMX · Arm: NEON, SVE
- Rows: RAM: available / total GB · Bandwidth: rough single-thread probe (multithreaded test PLANNED) · Threads: physical cores · Context: 2,048 by default on 8-16 GB today (sizing fix PLANNED)

**Notes:** `eigencore profile` detects the instruction-set ladder, which tells llama.cpp which SIMD paths it can use. It reads available versus total RAM and recommends threads equal to physical cores. Two honest limits: the bandwidth probe is single-threaded and reads low, and a unit mix-up means the default context is 2,048 on machines like this. Both are on the fix list.

## Slide 9 · Architecture (≈ 40 s)

- Title: A thin layer of judgement around llama.cpp
- Inference path: CLI / TUI → Orchestration (Forge API, router, context manager) → InferenceEngine → llama-cpp-python → llama.cpp (SIMD kernels)  [SHIPPED]
- Side boxes: HAL + fit estimator [SHIPPED] · Benchmark → run.json [SHIPPED]
- Tool path: ReAct agent → Policy engine (deny unless allow-listed; READ, WRITE, EXECUTE, NETWORK) → ToolExecutor → JSONL audit log  [SHIPPED when a policy is attached]
- Clay caveat: same Python process · not an OS sandbox · agent API does not attach a policy by default yet
- Legend: teal = EigenCore's own layer

**Notes:** Data flows top to bottom; teal boxes are EigenCore's own. The tool path: an agent proposes a call, the policy engine checks the capability and workspace paths, and only then does the executor run the handler, with every decision in an audit log. Two caveats in red: handlers run in the same Python process, so this is a policy layer, not a sandbox; and the agent API doesn't attach a policy by default yet. Making it the default is the first fix on that path.

## Slide 10 · Evidence (≈ 35 s)

- Title: Measure it before you claim it
- Command: `$ eigencore benchmark --report run.json`  [SHIPPED]
- Bars: prefill · prompt → TTFT · decode tok/s
- Definitions: Time to first token: dominated by prefill, processing the whole prompt. Decode tokens/second: steady generation, the number the bandwidth ceiling bounds.
- run.json field names (values omitted): schema_version, hardware, model.sha256, engine, results[time_to_first_token_s, decode_tokens_per_second, quality_pass], summary
- Footer: no results on a slide until they're run on a named machine

**Notes:** unchanged from v1, plus one sentence at the end: "No runs have been published yet; the first ones are on this laptop, and they'll be labelled with the machine."

## Slide 11 · Interface (≈ 25 s)

- Title: Fit-aware chat in the terminal
- Mock-up header (v2 layout from the handbook): `◩ EigenCore · chat` / `model ‹name› ‹quant›   ctx ‹used› / ‹set›   fit ‹verdict› +‹h› GB   tools off`  [SHIPPED: core chat; full-screen layout PLANNED]
- Caption: mock-up, not a screenshot · ‹ › = runtime values · `--plain` for scripts
- CLI list: profile · models · download · run · chat · analyze (logit sparsity, not neurons) · benchmark · train (optional LoRA/QLoRA)

**Notes:** `eigencore chat` streams a multi-turn chat with a plain fallback. The header keeps the memory story on screen: model, quant, context, fit verdict and headroom, and whether tools are on. One correction from an earlier version of this talk: `analyze` measures sparsity of the output logits, not of neurons inside the model.

## Slide 12 · Status (≈ 35 s)

- Title: What is done and what is not
- Tiles:
  - P1 Benchmark harness · Done  [SHIPPED]
  - P2 Model-fit estimator · Done, geometry fix next  [SHIPPED]
  - P3 Optimization audit · Audit done, nothing wired  [REPORTED-ONLY optimizations]
  - P4 Tool policy · Policy done; default-on, isolation, approvals pending  [SHIPPED / PLANNED]
  - P5 Chat TUI · Core shipped; full-screen pending  [SHIPPED / PLANNED]
  - P6 Kernels · Protocol + scalar references; Q4_0 reference being fixed; no SIMD  [SHIPPED / PLANNED]
- Box 1 (clay outline), REPORTED-ONLY: Layer skipping · activation sparsity · speculative decoding. None reaches llama.cpp.
- Box 2, NEXT [PLANNED]: 30B-class via a Mixture-of-Experts model (~3B active of 30B), measured on this laptop.
- Box 3, HONEST LIMITS: Tool handlers run in-process: not an OS sandbox. No speed claim until it's measured.

**Notes:** Solid tiles are done, hatched are partial. The harness and fit estimator work; the estimator's geometry fix is next. The optimization audit proved none of the speed-ups reaches llama.cpp, so they're labelled reported-only. The policy layer exists but isn't on by default in the agent API yet. P6 has a backend interface and NumPy reference kernels; one of them, Q4_0, has a layout bug being fixed before any SIMD work. For 30B-class models on a laptop, the realistic next step is a Mixture-of-Experts model, which only reads about 3 billion parameters per token, and then measuring it.

## Slide 13 · Takeaway (≈ 20 s)

- Kicker: THE TAKEAWAY
- Statement: Local inference is not only about how much compute you have. It is about how efficiently your hardware can move and hold the model's data.
- New sub-line: The lever is bytes per token: quantization and sparse-by-design models, not faster arithmetic.
- Questions?
- REPO github.com/Khubaib7-del/EigenCore

**Notes:** We started with a 21 GB model and a 16 GB laptop. Local inference is about whether your hardware can hold the weights and KV cache, and how fast it can move them; that sets a hard ceiling on tokens per second. So the lever is bytes per token. EigenCore makes those limits visible before you load a model, and its next job is to measure them honestly. Thank you, happy to take questions.

---

## Change list (v1 → v2)

| Slide | Old text (v1) | New text (v2) | Reason |
| --- | --- | --- | --- |
| all | v1 gradient logo; no status markers | v2 flat mark; SHIPPED / REPORTED-ONLY / PLANNED / ESTIMATE chips | Rebrand; brief requires every claim tagged |
| 3 | "× 0.55 B" · "≈ 19 GB" · Q4_K_M 19.25 GB · Q8_0 37 GB · Q5_K_M 24 GB | "× ≈ 0.61 B" · "≈ 21 GB" · Q4_K_M ≈ 21 GB · Q8_0 ≈ 37 GB · Q5_K_M ≈ 25 GB | Real Q4_K_M files average ~4.85 bpw; 0.55 was the estimator's assumption |
| 4 | "5+ GB short, before any KV cache"; weights ≈ 19 GB | "≈ 5 GB over on weights alone, before reserve and KV cache"; weights ≈ 21 GB; notes mention mmap paging | Corrected weight size; explains why it "loads" anyway |
| 5 | (no caveat) | Adds: CLI falls back to a heuristic ~30× low; GGUF geometry fix PLANNED | The CLI never passes geometry; the fallback underestimates |
| 6 | "38 GB/s" example bandwidth ÷ 19 GB ≈ 2 tok/s; "the cores mostly wait on RAM" | "45.8 GB/s" i7-10610U peak (Intel spec) ÷ 21 GB ≈ 2 tok/s, ceiling estimate; adds 3.6 FLOP/byte | Replaces an unnamed example with a published spec; adds the bound arithmetic |
| 7 | "run: infer via llama.cpp" | adds SHIPPED tags; "measure" notes no published runs | Tagging; no evidence exists yet |
| 8 | "Bandwidth ~ GB/s · timed memcpy benchmark"; "Context 512-8,192 tokens, sized to RAM left" | "rough single-thread probe (multithreaded test PLANNED)"; "2,048 by default on 8-16 GB today (fix PLANNED)" | Probe reads low; unit bug pins context at 2048 below ~38 GB RAM |
| 9 | "Policy engine: deny-by-default" | "deny unless allow-listed", "SHIPPED when a policy is attached"; caveat: agent API attaches none by default | `Forge.agent()` builds an executor without a policy |
| 10 | (unchanged slide) | notes add "no runs published yet" | Honesty |
| 11 | "analyze: measure activation sparsity" | "analyze (logit sparsity, not neurons)"; v2 header layout | `analysis/sparsity.py` measures output logits |
| 12 | P6 "Protocol + scalar kernels + tests · no optimized kernels yet"; "RESEARCHED, NOT SHIPPED … accepted tokens can keep the target's output distribution" | P6 "Q4_0 reference being fixed; no SIMD"; box relabelled REPORTED-ONLY; distribution sentence removed; new NEXT box: MoE path | Q4_0 reference and its test are wrong; repo's verifier does not preserve the distribution; MoE is the realistic 30B route |
| 13 | "We started with a 19 GB model" | "21 GB model"; adds "The lever is bytes per token" | Corrected number; the review's main conclusion |

Timing: no slide gains more than one short sentence of speech, and slide 12 trades the distribution sentence for the MoE sentence, so the total stays ≈ 5.5 min. Rehearse slides 5, 8 and 12 once, since their notes changed most.
