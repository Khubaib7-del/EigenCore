# EigenCore review: (A) Layer skipping · (B) SSD streaming

Scope: a read-only checkout of `main` at c92e3af. Target: Core i7 laptop, 8–16 GB RAM, NVMe (assumed), 30–40B models. Tags: [verified] = code read or source opened · [inferred] = my reasoning · [unverified] = claim not checked.

---

## A. Layer skipping (`src/eigencore/engine/layer_skip.py`, `docs/research/emergent-gaps.md` Gap 1)

### What exists
- `LayerSkipScheduler` (L58–268) with STATIC / ADAPTIVE / CALIBRATED plans that return a list of layer indices plus `expected_speedup` and `expected_quality_loss`. [verified]
- It is built in `engine/inference.py:105` and only read at `inference.py:254–262`, where stats are emitted with `"applied": False`. `Llama(...)` (`inference.py:86–94`) gets no skip information. `tests/test_optimization_audit.py:95–96` asserts `backend_wired == []` and `applied is False`, and `cli/main.py:409` admits the speed-up is ~1.0x. [verified]

### Bugs and overclaims
1. **Docstring L8: "<2% perplexity increase at 20–30% skip rate" has no source** and no benchmark in the repo backs it. In the published work, training-free removal at that scale degrades quality measurably. Gromov et al. needed a "healing" fine-tune. [verified that the claim is unsourced; magnitude inferred from the cited papers]
2. **`expected_speedup = 1/(1-skip_rate)` (L185, 221, 259) is arithmetic, not a measurement.** It ignores attention, KV, embeddings/LM head, and threading overhead. **`quality_loss = skip_rate*0.06` (L186), `*0.04` (L222) and `(1-importance)*0.01` (L247) are invented constants.** [verified]
3. **The skipper never starts on non-Llama GGUFs.** `inference.py:101` reads only `llama.block_count`, but GGUF keys are prefixed by `general.architecture` (e.g. `qwen2.block_count`). On Qwen, Mistral-arch and similar models the scheduler is `None`. [verified for the code; key naming inferred from the GGUF convention]
4. **At complexity=1.0 STATIC still skips one layer.** `max(1, int(... * 0))` at L176 always returns at least 1. [verified]
5. **The heuristics are wrong.** It protects the first and last 15% and skips middle layers at fixed strides (L167–179). ShortGPT and Gromov find the **deeper** layers (except the final ones) most redundant, and they prune **contiguous** blocks chosen by similarity, not strided ones. ADAPTIVE marks a layer skip-safe when its "sparsity" is above 0.7 (L152), which is not a layer-redundancy criterion. `calibrate()` uses the median of an L2-norm importance, but ShortGPT's Block Influence is 1 − cos(input, output). [verified code; inferred comparison]
6. **The Gap 1 doc (L12, L25) claims "no existing CPU runtime does this" and a per-token "skip to layer 28".** Per-token early exit without training breaks the KV cache: later tokens attend to hidden states that skipped tokens never produced. CALM lists this ("attending back to missing hidden representations") as a core problem, and LayerSkip needs training-time layer dropout plus an early-exit loss. [verified from abstracts]
7. **llama-cpp-python has no API for this.** The high-level `Llama` exposes `use_mmap`, `use_mlock` and `kv_overrides` but no per-token layer mask or tensor override. Real skipping needs a llama.cpp fork or a pruned GGUF. [verified by grep of `llama_cpp/llama.py`; absence in the C API is inferred]

### Comparable work (opened)
- ShortGPT (Block Influence, static layer removal): https://arxiv.org/abs/2403.03853
- Gromov et al., "Unreasonable Ineffectiveness of the Deeper Layers": up to ~half the layers removed with small loss **after QLoRA healing**: https://arxiv.org/abs/2403.17887
- SLEB (block elimination for real end-to-end speed-up): https://arxiv.org/abs/2402.09025
- LaCo (layer collapse, >80% task performance at 25–30% pruning, so not "<2%"): https://arxiv.org/abs/2402.11187
- LayerSkip (Meta): early exit plus self-speculative decoding, 1.8–2.2x, **requires their training recipe**: https://arxiv.org/abs/2404.16710
- CALM (confidence-based dynamic early exit, up to 3x): https://arxiv.org/abs/2207.07061

**Static vs dynamic** [inferred]: static pruning (ShortGPT/SLEB/LaCo) gives a smaller GGUF that runs on unmodified llama.cpp, so it saves RAM, disk bytes and compute. Dynamic skipping (CALM/LayerSkip) needs graph surgery, KV handling and usually training, and saves no memory because every layer must still be resident.

### Alternatives
1. An offline **static pruner** that writes a new GGUF (BI-ranked contiguous block drop), optionally followed by a LoRA heal.
2. **Self-speculative decoding** with a LayerSkip checkpoint, or a small draft model (llama.cpp already supports draft models).
3. Delete the dynamic scheduler, or relabel it "planned".

### Next steps and acceptance tests
- Fix the block_count lookup (use `general.architecture`). **Test:** the scheduler is non-None for one Llama GGUF and one Qwen GGUF.
- Fix complexity=1.0. **Test:** `plan(1.0).layers_to_skip == []`.
- Remove the "<2%" claim and the made-up constants, or replace them with measured numbers. **Test:** grep shows no unsourced numeric claims.
- Prototype the static pruner. **Acceptance:** at 25% removal, report WikiText-2 perplexity Δ and the MMLU/HellaSwag Δ against the base model, plus the measured tok/s gain and file-size drop on the target laptop.

### Benchmarks owner must run
Use `llama-perplexity` and `llama-bench` on base vs pruned GGUFs (0/12.5/25%) for one 7–8B and one ~32B model: report perplexity, tg128 tok/s, pp512 tok/s and RSS.

---

## B. SSD streaming / offloading

### What exists
**Nothing in EigenCore.** It has no mmap, ssd or offload code [verified by grep]. `Llama(...)` (`inference.py:86`) passes no `use_mmap` or `use_mlock`, so llama-cpp-python's default **`use_mmap=True`** applies [verified, `llama.py:70`]. A model larger than RAM therefore already "runs" by OS demand paging. EigenCore adds nothing to that and controls none of it.

### Physics: dense vs MoE (estimates, [inferred])
Assumptions:
- Model is 35B dense Q4_K_M ≈ 19.5 GB.
- Laptop RAM bandwidth is ≈ 50 GB/s.
- Weight RAM left after OS, KV and apps is ≈ 10–12 GB on a 16 GB machine and 3–5 GB on an 8 GB machine.
- Effective SSD read is 1, 3 or 7 GB/s. mmap page-fault reads often fall well below the rated sequential speed; one fio mmap simulation measured ~1.5 GB/s.
- A dense model reads **every weight on every token**.
- Best case: the resident part stays cached, so time per token = R/50 + (19.5−R)/SSD.
- Worst case: LRU thrashes on the cyclic sweep, so all 19.5 GB come from disk each token.

| RAM | SSD 1 GB/s | 3 GB/s | 7 GB/s |
|---|---|---|---|
| 16 GB (best) | 0.10–0.13 tok/s | 0.30–0.36 | 0.64–0.76 |
| 8 GB (best) | 0.06–0.07 | 0.18–0.20 | 0.41–0.46 |
| Worst (LRU thrash) | 0.05 | 0.15 | 0.36 |

The all-in-RAM ceiling is only ≈ 2.6 tok/s (50/19.5). **Dense 30–40B on this laptop is a sub-1 tok/s proposition from SSD.** Skipping 25% of layers cuts disk bytes 25%, which is still below 1 tok/s.

**MoE** (e.g. Qwen3-30B-A3B, ~3B active, ≈ 1.8 GB touched per token at Q4) needs only 1.8 GB per token. Routing locality also keeps hot experts in page cache. Estimated tok/s:

| Page-cache hit rate | SSD 1 GB/s | 3 GB/s | 7 GB/s |
|---|---|---|---|
| 0% | 0.6 | 1.7 | 3.9 |
| 80% | 2.6 | 6.7 | 12.5 |

These numbers are before the CPU-compute cap. That is the order-of-magnitude reason SSD streaming suits MoE and not dense models.

### Field evidence
- **Swap-MoE**, a patch to llama.cpp: on a 13th-gen i7 with 16 GB RAM, NVMe and CPU only, **Qwen3.5-35B-A3B Q4_K_M (19.7 GB) ran at 7.01 tok/s** with expert streaming. The authors report that stock loading "fails to launch" because the loader pre-touches tensors. https://github.com/ek15072809/Swap-MoE [verified page; numbers are self-reported]
- **DeepSeek-R1 671B** (212 GB quant) ran from NVMe via mmap on a 96 GB RAM machine at ~1.1–2.3 tok/s. SSD reads during inference were 2–5 GB/s against a 12.4 GB/s rated drive, and `expert_used_count=4` was used to go faster. https://huggingface.co/unsloth/DeepSeek-R1-GGUF/discussions/13 [verified]
- **llama.cpp Q&A**: with mmap on (the default), weights are read on demand; with `--no-mmap`, a model larger than RAM OOMs. One user reports `load_tensors()` MAP_POPULATE defeats lazy paging. Another runs Qwen3.6-35B-A3B Q5 (25 GB) on 16 GB RAM plus 6 GB VRAM at 2–10 tok/s. https://github.com/ggml-org/llama.cpp/discussions/19163 [verified, anecdotal]
- **llama.cpp server flags**: `-ot/--override-tensor`, `--cpu-moe`, `--n-cpu-moe`, and mmap/mlock load modes. https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md [verified]
- **Research systems**:
  - LLM in a Flash runs models 2x DRAM size, 4–5x faster than naive CPU loading: https://arxiv.org/abs/2312.11514
  - PowerInfer-2 serves a 47B model on a phone at 11.68 tok/s using neuron-cluster I/O pipelining: https://arxiv.org/abs/2406.06282
  - FlexGen does throughput-oriented GPU/CPU/disk offload (1 tok/s on OPT-175B at batch 144, so not interactive): https://arxiv.org/abs/2303.06865
  - AirLLM loads layer by layer, aimed at GPU VRAM limits and bottlenecked on disk: https://github.com/lyogavin/airllm [all verified, abstracts/README]

### Bugs and overclaims
Any EigenCore claim of "SSD streaming" for 30–40B dense models on 8–16 GB would be an overclaim [inferred]. EigenCore also gives no warning when a model exceeds RAM, and has no `use_mlock` or `use_mmap` controls.

### Alternatives (ranked)
1. **Target MoE models** (Qwen3-30B-A3B class, gpt-oss-20b) with default mmap. Expose `use_mmap`, `use_mlock` and `kv_overrides` in the config.
2. Shrink the model to fit RAM: lower-bit quantization (IQ3/IQ4) or 14B dense.
3. For a real expert cache, adopt or upstream a llama.cpp patch (Swap-MoE style: routed-expert lazy paging plus prefetch). Do not build it in Python.

### Next steps and acceptance tests
- Add a pre-load check: model size vs available RAM → choose mmap, warn, and report expected disk-bound tok/s from the formula above. **Test:** a 20 GB GGUF on a 16 GB machine logs the warning and its estimate.
- Add `use_mmap` and `use_mlock` passthrough. **Test:** both values reach `Llama(...)` as set.
- Write an MoE "fits-via-SSD" profile. **Acceptance:** Qwen3-30B-A3B Q4_K_M ≥ 3 tok/s tg on a 16 GB i7 with NVMe, with resident memory below physical RAM.

### Benchmarks owner must run
1. `fio` random 4K–1M read and sequential read on the target SSD, to get the real effective GB/s.
2. `llama-bench` tg128/pp512 for a dense ~32B Q4_K_M and Qwen3-30B-A3B Q4_K_M at 8 GB and 16 GB. Cap memory with a cgroup or `systemd-run -p MemoryMax=` so 8 GB can be simulated on a 16 GB box.
3. Log major page faults and disk read MB/s (`vmstat`, `iostat`) per token, and compare them with the estimates table.
4. Measure `--no-mmap` vs mmap time-to-first-token for a model that fits in RAM.
