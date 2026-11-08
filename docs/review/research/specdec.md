# EigenCore review — Speculative decoding (Gap 5)

Scope: `src/eigencore/engine/speculative.py` (235 lines), `tests/test_speculative.py`, `docs/research/emergent-gaps.md` §Gap 5. Target: Core i7 laptop, 8–16 GB RAM, 30–40B GGUF via llama-cpp-python. Tags: **[verified]** = checked in code or an opened source; **[inferred]** = my reasoning/arithmetic; **[unverified]** = plausible, not checked.

## 1. What exists (file:line)
- `SpeculativeDecoder` (speculative.py:86–235): stores `draft_fn`/`verify_fn` (:106–107) but **never calls them**. It never drafts or verifies anything. **[verified]**
- `verify_draft` (:130–203) takes only *scalar* logprobs of the drafted tokens, not full distributions. It accepts while `exp(lp_target − lp_draft) >= acceptance_threshold` (default 0.3) (:172–174). **[verified]**
- Also: draft-length adaptation (:205–214), a cache heuristic (:113–128) and an estimator (:223–235). **[verified]**
- Not wired in: `inference.py:67` builds it, and `:237–243, 281–286` only report stats with `"applied": False`. The README admits this. **[verified]**
- Tests (11) check the heuristic, counters and monotonicity only. Nothing checks distribution preservation, the correction or bonus token, or greedy equivalence. Two adaptation tests (test_speculative.py:66, 74) pass even if adaptation does nothing. **[verified]**

## 2. Bugs and overclaims
1. **Wrong acceptance rule.** Leviathan/Chen accept draft token x with probability min(1, p(x)/q(x)). On rejection they resample from norm(max(0, p−q)). If all γ tokens pass, they sample a bonus token from p. This keeps the target distribution exactly ([Leviathan, Thm/App. A](https://arxiv.org/html/2211.17192v2); [Chen](https://arxiv.org/abs/2302.01318)). EigenCore makes a *deterministic* accept whenever p/q ≥ 0.3. That keeps tokens the target finds 3× less likely than the draft does, which skews output toward the draft. It can also break greedy equivalence, because it accepts a non-argmax token whenever the ratio clears 0.3. **[verified code; consequence inferred]**
2. **Correction token is always 0.** `int(np.argmax(verifier_logprobs[i:][:1]))` (:179) takes argmax over a slice of length 1, so the result is always 0. Token id 0 then gets returned as the "correction". And because the function only receives scalars, it has no target distribution to correct *from*. **[verified by reading; numpy semantics]**
3. **No bonus token.** When all γ tokens are accepted, the algorithm should emit γ+1 tokens. This one emits γ. **[verified]**
4. **`estimate_speedup` overstates gains** (:230–235). It uses (1+αn)/(1+cn), which treats accepts as independent rather than prefix-geometric and assumes verification is free. Example: α=0.6, γ=5, c=0.05 gives **3.2×** from EigenCore vs **1.91×** from Leviathan. **[verified arithmetic]**
5. `SpeculativeResult.speedup` (:49–54) is always (n+1)/n. `draft_time_ms` is always 0, and `verify_time_ms` times a Python loop (:185–195). **[verified]**
6. **Wrong docstring claims:**
   - "Draft shares the same memory-mapped GGUF file … (different quant)" (:14). One GGUF holds one tensor set at one quant, so this can't happen. Self-drafting from one file is a separate technique (layer-skip/early exit, e.g. LayerSkip). **[inferred; GGUF format]**
   - "Verify step is still sequential on CPU" (:10). Wrong. Verification evaluates γ+1 positions in one batched forward pass, like prefill ([llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)). That is the whole win on a bandwidth-bound CPU. **[verified]**
   - "Acceptance threshold adapts" (:15). Only the draft length adapts. **[verified]**
   - "CPU branch prediction" (:8–9) has nothing to do with the speedup. **[inferred]**
7. **Docs (emergent-gaps.md:109–125):**
   - "3–4× faster on CPU" has no measurement behind it.
   - "60M model always in L2 cache": 60M params is roughly 35 MB at Q4 and 120 MB at F16. Laptop L2 is about 1–2 MB per core and L3 is at most about 24 MB. **[inferred arithmetic; cache sizes unverified]**
   - No 60M draft shares a tokenizer with the 30–40B targets. llama.cpp refuses a draft whose vocab differs ([`common_speculative_are_compatible`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/speculative.cpp): vocab size within 128, same BOS/EOS, same token text). **[verified]**
   - The 3B verifier in the diagram contradicts the target. `competitive-landscape.md:15` advertises "tiered drafting", which does not exist. **[verified]**

## 3. Comparable work (cited, opened)
- **Leviathan et al. 2023**, https://arxiv.org/abs/2211.17192. Exact sampling, 2–3× on T5-XXL/TPU. Expected walltime factor **S = (1−α^{γ+1}) / ((1−α)(γc+1))**. It assumes γ+1 parallel target evaluations cost the same as one. **[verified]**
- **Chen et al. 2023 (DeepMind)**, https://arxiv.org/abs/2302.01318. Modified rejection sampling, 2–2.5× on Chinchilla-70B (distributed). **[verified]**
- GPU results, all opened **[verified]**: [Medusa](https://arxiv.org/abs/2401.10774) (heads + tree attention, 2.2–3.6×, needs fine-tuning); [EAGLE-3](https://arxiv.org/abs/2503.01840) (up to 6.5×); [Lookahead](https://arxiv.org/abs/2402.02057) (no draft, up to 1.8×, spends extra FLOPs, a bad fit for CPU **[inferred]**); [Prompt lookup](https://github.com/apoorvumang/prompt-lookup-decoding) (n-gram copy from the prompt, ~2.4× on summarization/QA); [LayerSkip](https://arxiv.org/abs/2404.16710) (early-exit self-speculation, ≤2.16×, needs special training).
- **llama.cpp** ([docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)): `llama-server -md` with `--spec-draft-n-max` (default 3). `--spec-type` covers draft-simple, draft-eagle3, draft-mtp and ngram-*. It prints acceptance stats. An EAGLE-3 draft is listed for **Qwen3-32B**, a direct match for this target. **[verified]**
- **llama-cpp-python** ([README](https://raw.githubusercontent.com/abetlen/llama-cpp-python/main/README.md), [llama_speculative.py](https://raw.githubusercontent.com/abetlen/llama-cpp-python/main/llama_cpp/llama_speculative.py)): `Llama(draft_model=LlamaPromptLookupDecoding(num_pred_tokens=…))`, with 2 recommended for CPU. Prompt lookup is the only built-in draft. `Llama.generate` batch-evaluates the draft and accepts up to the first mismatch with the target's own sample. That is exact for greedy and lower-acceptance at temp>0. **[verified code; exactness inferred]**

**CPU evidence:**
- ggerganov: verifying N tokens costs T1 < T_N < N·T1, so the "free verification" assumption fails ([#3649](https://github.com/ggml-org/llama.cpp/issues/3649)). The same thread has a user reporting 1.3× on **pure CPU** with a 70B Q3_K_S target and a TinyLlama draft (1.5→2 t/s). **[verified; anecdote]**
- Qwen2.5-1.5B target / 0.5B draft on 2 CPU threads: **1.5–2.5× slower** even at 70% acceptance, because the draft was only 1.9× faster than the target ([AI Frontier Post](https://aifrontierpost.com/articles/speculative-decoding-llamacpp-tutorial/)). **[verified]**
- Qwen2.5-Coder-3B on CPU only: 1.72× overall, 2.03× on math, 1.01× on chat ([InventiveHQ](https://www.inventivehq.com/blog/llama-cpp-speculative-decoding-consumer-gpu)). **[verified]**
- Net: the result depends on the cost ratio, acceptance and task, and can fall below 1×. **[inferred]**

## 4. Illustrative speedups (ILLUSTRATIVE ONLY, not measurements)
Generalised formula: S = E/(γc + v), where E = (1−α^{γ+1})/(1−α) and v = T_verify(γ+1)/T_1. v=1 is Leviathan's assumption; on CPU, v>1.

| α | γ | c | v | E tokens/round | S |
|---|---|---|---|---|---|
| 0.8 | 5 | 0.05 | 1.0 | 3.69 | 2.95× |
| 0.8 | 5 | 0.05 | 1.5 | 3.69 | 2.11× |
| 0.6 | 5 | 0.05 | 1.5 | 2.38 | 1.36× |
| 0.6 | 5 | 0.15 | 1.5 | 2.38 | 1.06× |
| 0.7 | 3 | 0.5 | 1.2 | 2.53 | **0.94× (slower)** |
| 0.7 | 4 | ≈0 (prompt lookup) | 1.3 | 2.77 | 2.13× |
| 0.4 | 4 | ≈0 (prompt lookup) | 1.3 | 1.65 | 1.27× |

"3–4×" needs roughly α ≥ 0.8, c ≤ 0.05 and v ≈ 1 all at once. That is unproven on a laptop. **[inferred]**

**RAM caveat.** A 32B model at Q4_K_M (~4.8 bpw) is about 19–20 GB, more than 16 GB of RAM, so it either pages through mmap or needs a 2–3 bpw quant. A 0.5B draft takes ~0.5 GB of page cache on top. Under paging, batched verification may help *more*, but only measurement can show it. **[inferred/unverified]**

## 5. Alternatives (ranked for this target)
1. **Prompt lookup via llama-cpp-python** (`num_pred_tokens=2–4`). No extra RAM, output-preserving, strong for RAG, code-edit and agent tool echo. Fastest to wire in. **[inferred]**
2. **llama-server `--spec-type ngram-mod` / `ngram-map-k`**: same idea, with the C++ implementation.
3. **Draft model through llama-server** (`-md` with Qwen2.5-0.5B or Qwen3-0.6B for a same-family 32B). This uses llama.cpp's own rejection sampling. Only worth it if measured c ≤ ~0.1.
4. **EAGLE-3 for Qwen3-32B** (`--spec-type draft-eagle3`). Highest acceptance, but CPU performance is **[unverified]**.

## 6. Recommended next steps with acceptance tests
- **S1:** Mark experimental and remove the §2.6–2.7 claims. *Test:* a grep finds none.
- **S2:** Delete `verify_draft`, or rewrite it on full per-position distributions (γ+1 × V) with min(1,p/q) acceptance, residual resampling, a bonus token and a greedy mode (correction = argmax p). *Tests:* (a) a chi-square check over 200k trials on an 8-token toy vocab matches p (p>0.01), and the current code must **fail** it; (b) a rejection never yields a constant 0; (c) all-accept returns γ+1 tokens; (d) `estimate_speedup` equals Leviathan's formula to 1e-9.
- **S3:** Wire option 1 behind a flag. *Tests:* temp-0 token ids are identical to the non-speculative run (20 prompts × 256 tokens); `applied_optimizations` contains `"speculative"` only when the flag is on; tok/s and acceptance are measured values.
- **S4:** Add draft-model or EAGLE-3 support through llama-server only if the benchmarks in §7 show ≥1.2× median on the owner's laptop.

## 7. Benchmarks the owner must run (on the i7 laptop, plugged in, 3 runs, report median)
1. `llama-bench -m <target> -p 1,2,4,6,8,12 -n 64`: get v(γ) from pp-k vs tg speed. Repeat for the draft to get c.
2. Baseline `llama-server` on the target: `predicted_per_second`, peak RSS, major page faults.
3. Same with `-md <same-family 0.5–0.6B> --spec-type draft-simple --spec-draft-n-max {2,3,5,8}`, at temp 0 and 0.7.
4. `--spec-type ngram-mod` and llama-cpp-python `LlamaPromptLookupDecoding(num_pred_tokens ∈ {2,4,10})`.
5. Prompt categories: code, chat, summarization/RAG, agent tool calls (≥5 prompts each, 256+ tokens).
6. Report tok/s, speedup, acceptance and RSS. Check the measured speedup against S using the measured α, c and v.
