# EigenCore review — Sparse activation

Scope: `src/eigencore/engine/sparse_inference.py`, `src/eigencore/analysis/sparsity.py`, `cli/main.py analyze`, `engine/inference.py` wiring, `docs/research/sparse-activation-and-routing.md`, `docs/research/emergent-gaps.md` Gap 3, `router/classifier.py`. Read-only review; no code changed. Tags: [verified] = checked in code or an opened source; [inferred] = my reasoning; [unverified] = not checked.

## What exists (file:line)

- `engine/sparse_inference.py:1-14`: the module docstring claims "70-90% activation sparsity in dense transformer models" and says "skipping a zero multiply is a correctly-predicted branch". `SparsityPredictor` (58-173) counts how often each neuron passes `|a|>=threshold` and builds a mask from frequency (`predict_mask`, 106-132). `create_execution_plan` (134-158) reports `speedup = 1/(1-sparsity)`, capped at 5×. `SparsityCache` (176-218) is an LRU cache keyed by an MD5 of the last 32 tokens. [verified]
- `analysis/sparsity.py:85-135`: `analyze_logits` thresholds the **output logit vector** (`layer_name="output_logits"`). `analyze_token_probabilities` (137-184) gives the share of the vocabulary outside the top 95% of probability mass. Both compute `potential_speedup = 1/(1-x)`. The class docstring (87-95) admits these are "proxy measurements". [verified]
- `cli/main.py:10,296`: `analyze` is described as "measure activation sparsity". Lines 333-367 read llama.cpp logits and print "Theoretical CPU speedup from sparsity". [verified]
- `engine/inference.py:101-116`: the predictor is built from `llama.block_count` and `llama.embedding_length`. In `_collect_stats` (265-273) every plan carries `applied: False`, and `_applied_optimizations()` returns `[]` (238-244). [verified]
- `router/classifier.py:1-8`: a keyword classifier that routes between whole small models ("application-level MoE"). It never touches neurons. [verified]

## Bugs and overclaims

1. **The `analyze` command measures the wrong quantity.** Logits are vocabulary scores, not FFN activations. How many of them sit near zero depends on the threshold and the logit scale, and says nothing about which weight rows could be skipped. Applying `1/(1-x)` to that number is not a speedup estimate. "Probability concentration" is a property of the output distribution, not of how sparse the model is internally. [verified code; inferred interpretation]
2. **The predictor is dead code.** No caller of `.observe()` exists anywhere in `src/` (grep shows none), so `is_warmed_up` can never become true and `sparse_inference` stats are never emitted. `SparsityCache.get/put` are also never called. llama-cpp-python exposes no per-layer activations, which the analyzer's own docstring concedes (`analysis/sparsity.py:89-91`). [verified]
3. **Wrong neuron count.** `neurons_per_layer = embedding_length` (hidden size). The sparsifiable FFN neurons are the intermediate width (`feed_forward_length`), which is several times larger. [verified code; inferred consequence]
4. **Llama-only metadata keys.** Qwen, Mistral-family and MoE GGUFs use their own architecture prefixes, so on those models the predictor is silently never built. [verified keys; inferred for other architectures]
5. **Rate dilution.** `observe()` increments the global `_total_samples` once for every layer it is called on (line 101). If it were wired up, `activation_rates` (121) would be divided by roughly layers × tokens, and warm-up would finish after 16 layer calls. [verified]
6. **The mask is static, not contextual.** It is built from average firing frequency and ignores the current input. Deja Vu's point is that contextual sparsity is input-dependent and needs a per-layer predictor that runs on that layer's input. [verified paper; inferred gap]
7. **The documentation's claims fail against the literature.**
   - (a) The 70-90% figure comes from ReLU models. ProSparse reports that GELU/Swish LLMs lack intrinsic activation sparsity. TurboSparse says SwiGLU/GeGLU "exhibit limited sparsity" and that a plain ReLU swap "fails to achieve sufficient sparsity". [verified]
   - (b) "Nobody built on it" is false: Deja Vu, PowerInfer, LLM in a Flash, TEAL and others did. [verified]
   - (c) PowerInfer is cited as "Du et al., 2024". The authors are Song, Mi, Xie and Chen (arXiv 2023; SOSP 2024). [verified]
   - (d) "Skipping 70% of multiplies is a direct 3× speedup" ignores the decode bottleneck. TEAL notes that small-batch decode is memory-bound, so the gain comes from not *transferring* weight columns, and weights must be stored column- or row-major for contiguous loads. A per-element branch defeats SIMD and saves no bandwidth. [verified TEAL text; inferred CPU consequence]
   - (e) PowerInfer's README states that running llama.cpp-format weights gives "no performance gain"; its speedups need ReLU-sparse models plus predictors. [verified]

## Comparable work (opened)

- **Deja Vu** (Liu et al., ICML 2023), https://arxiv.org/abs/2310.17157: per-layer contextual-sparsity predictors with an async, hardware-aware implementation; >2× over FasterTransformer on OPT-175B (GPU). [verified]
- **PowerInfer** (Song et al.), https://arxiv.org/abs/2312.12456: hot/cold neurons follow a power-law distribution; hot ones go on the GPU and cold ones run on the CPU; adaptive predictors plus neuron-aware sparse operators; up to 11.69× over llama.cpp on an RTX 4090. Repo: https://github.com/SJTU-IPADS/PowerInfer (x86 AVX2 CPU-only supported). [verified]
- **PowerInfer-2**, https://arxiv.org/abs/2406.06282: neuron-cluster scheduling with I/O pipelining; 47B on a phone at 11.68 tok/s. [verified]
- **LLM in a Flash** (Alizadeh et al.), https://arxiv.org/abs/2312.11514: windowing (reuse recently activated neurons) and row-column bundling for contiguous reads; runs models up to 2× DRAM size, 4-5× faster on CPU than naive loading. [verified]
- **ReLU Strikes Back**, https://arxiv.org/abs/2310.04564: switching back to ReLU costs little accuracy and cuts inference compute up to 3×. [verified]
- **ProSparse**, https://arxiv.org/abs/2402.13516: ReLU plus progressive regularisation; 89.32% sparsity on LLaMA2-7B; up to 4.52× inference speedup. [verified]
- **TurboSparse**, https://arxiv.org/abs/2406.05955: dReLU; Mistral activates 2.5B and Mixtral 4.3B params per iteration; 2-5× decode speedup. [verified]
- **CATS**, https://arxiv.org/abs/2404.08763: within 1-2% of base at 50% sparsity without fine-tuning; ~15% latency gain from a custom GPU kernel. [verified]
- **TEAL**, https://arxiv.org/abs/2408.14690: training-free magnitude sparsity on all hidden states; 40-50% model-wide on Llama-2/3 and Mistral (7B-70B); up to 1.53× / 1.8× decode at 40% / 50%; kernels are GPU-oriented. [verified]
- **Q-Sparse**, https://arxiv.org/abs/2407.10969: top-K activation sparsity learned in training or continued training. [verified]

## Alternatives for an i7 laptop (8-16 GB, 30-40B class)

- **A. Use a MoE model. This is the realistic path now.** Qwen3-30B-A3B has 30.5B total / 3.3B activated params (128 experts, 8 active; https://huggingface.co/Qwen/Qwen3-30B-A3B). DeepSeek-V2-Lite has 16B / 2.4B (https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite). Mixtral uses 12.9B of 46.7B per token (https://mistral.ai/news/mixtral-of-experts/). llama.cpp already runs MoE GGUFs, so this needs no new kernel. [verified]
  - Bytes read per token scale with *active* params × bits/weight, so decode traffic for an A3B model is about 1/9 that of a dense 30B at the same quantisation. [inferred]
  - Caveat: *total* weights must still be resident or mmap'd. A 30B at ~4-bit will not fit in 8 GB and is tight in 16 GB, so expert paging from SSD (LLM-in-a-Flash style) or lower-bit quants become the real engineering problem. [inferred; owner must measure]
- **B. Use a pre-sparsified model with PowerInfer.** ProSparse-Llama-2 or TurboSparse-Mistral/Mixtral are available on the CPU-only x86 path. That means a different backend (a llama.cpp fork), not llama-cpp-python. [verified availability; CPU speedup unverified]
- **C. TEAL-style magnitude sparsity in a custom ggml kernel.**
  - The kernel would threshold the input of the down and up/gate projections, then do column-gathered GEMV over a column-major (input-sparse) layout, fused with dequantisation.
  - That requires forking llama.cpp and adding a per-tensor calibrated threshold plus a new layout. Expected gain is bounded by TEAL's GPU numbers (≤1.8× at 50%). The CPU gain is unproven, and the gather breaks the block-quant layout. [inferred]
- **D. Drop the claim.** Keep application-level routing (`router/`) as the "only part of the model runs" story, and relabel the sparsity work as research. [inferred]

## Recommended next steps, with acceptance tests

1. **Fix the claims.** Rename `analyze` to "logit/output-distribution stats", remove `potential_speedup`, and correct Gap 3 and the PowerInfer citation. *Acceptance:* no text says "activation sparsity" or "speedup" for logit data; a test asserts the report has no speedup field.
2. **Delete or quarantine the predictor and cache** (or mark them experimental) until a real activation source exists. Fix the per-layer sample counter, read `{arch}.feed_forward_length`, and make metadata parsing architecture-agnostic. *Acceptance:* a unit test with 2 layers × N tokens gives correct per-layer rates; the predictor builds on Qwen and Mistral GGUFs.
3. **Measure real activation sparsity offline** with a PyTorch/transformers hook on the FFN intermediate for one dense SwiGLU model and one ReLU-fied model. *Acceptance:* a per-layer histogram, plus the share of |a| under thresholds that keep perplexity within 1% on a fixed eval set.
4. **Add MoE as a supported target.** Add a registry entry for Qwen3-30B-A3B (and DeepSeek-V2-Lite for 8 GB) with mmap enabled. *Acceptance:* it loads and generates on a 16 GB i7 without OOM; tok/s is reported against a dense model of similar quality.
5. **Option C only after step 3** shows ≥40% usable sparsity on the target model.

## Benchmarks the owner must run

- Decode and prefill tok/s and peak RSS: Qwen3-30B-A3B (Q4 and Q3) vs dense Qwen-32B-class vs a 7-8B dense model, on the actual i7, at 8 and 16 GB.
- Page-fault / SSD read rate when the MoE model exceeds RAM (mmap): this is where the speed is decided.
- PowerInfer CPU-only: ProSparse-Llama-2-7B vs llama.cpp on the original Llama-2-7B, same quantisation.
- For any custom kernel: GEMV µs at 0/25/40/50% column sparsity vs dense ggml Q4 GEMV, plus perplexity deltas.
- Report the same prompts, thread counts, and context lengths for every run; no estimated speedups in the CLI.
