# EigenCore review — Inference engine & Fit estimator

Scope: a read-only checkout of `main` at c92e3af. Labels: **[verified]** = code read or source opened, **[inferred]** = my reasoning, **[unverified]** = not checked. Arithmetic was run against the repo's own functions (`PYTHONPATH=src`), not on target hardware. None of the numbers below are benchmarks.

---

## 1. Inference engine (`engine/inference.py`, `benchmark/harness.py`)

### What exists in the repo
- `load()` builds `Llama(n_ctx, n_threads, n_threads_batch, n_batch, n_gpu_layers=0, verbose=False)` (inference.py:86-94). It leaves `use_mmap`, `use_mlock`, `n_ubatch`, `flash_attn`, `type_k` and `type_v` at their defaults [verified].
- `_init_optimizers` reads only `llama.block_count` and `llama.embedding_length` (L100-102) [verified].
- `_compute_batch_size` picks 1024, 512 or 256 by ISA (L118-125). The docstring says "based on available RAM", but RAM is never read [verified].
- Layer-skip, sparsity and speculative stats are reported but **not applied**. The code says so itself (`applied_optimizations=[]`, L46-49, L237-244). This is honest, but every "speedup" field is a plan, not a measurement [verified].
- The harness counts tokens as streamed chunk count (harness.py:236). It estimates prompt tokens as `len(prompt)//4` (L245). The first run is cold and is still recorded alongside the others (L300) [verified].

### Bugs and overclaims
1. **Metadata keys are hard-coded to `llama.*`.** Per the GGUF spec, keys are prefixed with `[llm]` (the architecture), e.g. `qwen2.block_count` (https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md, L457-477). For qwen2, gemma, phi3 and others, `num_layers` becomes 0 and the optimizers silently skip [verified]. Fix: read `general.architecture` first.
2. **`n_gpu_layers=0` is hard-coded.** That is right for a CPU-first tool, but it is not configurable [verified].
3. **No evidence for the `n_batch` choice.** n_batch mainly affects prefill. llama-bench sweeps `-b`/`-ub` for exactly this purpose (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/llama-bench/README.md) [verified]. The i7-10610U is AVX2-only (https://www.intel.com/content/www/us/en/products/sku/201896/…/specifications.html), so it always gets 512 [verified].
4. **The harness measures differently from llama-bench.** It counts chunks, not tokens, and it includes chat-template, tokenization and sampling overhead. llama-bench excludes tokenization and sampling and reports mean ± sd over `-r` reps after a warmup (same README) [verified]. The two sets of numbers cannot be compared [inferred].

### What llama-cpp-python already provides (API reference, https://llama-cpp-python.readthedocs.io/en/latest/api-reference/) [verified]
- `use_mmap=True` and `use_mlock=False` are the defaults, alongside `n_ubatch=512`, `flash_attn=False`, `type_k/type_v: Optional[int]`, `offload_kqv` and `kv_overrides`.
- `self.metadata = self._model.metadata()`, a full GGUF key-value dict.
- Low-level functions `llama_model_size`, `llama_model_n_params`, `llama_state_get_size`, `llama_supports_mmap` and `llama_supports_mlock` (exact post-load weight bytes and parameter count).
- Not exposed in the API reference: llama.cpp's `llama_params_fit`, which lives in `common/` rather than libllama [verified absent from the docs; whether it can be bound at all is inferred].
- KV block sizes from `ggml-common.h`: `block_q8_0` = fp16 scale + 32 int8 = 34 B per 32 elements (1.0625 B/elem); `block_q4_0` = 18 B per 32 (0.5625) (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h) [verified].

### Alternatives
Pass KV type, flash-attn, n_ubatch and mlock through from a profile. Use `llama-bench -o json` for decode and prefill numbers.

### Recommended next steps (ordered)
1. **Read keys as `f"{arch}.…"`.** *Test:* a qwen2 or gemma GGUF yields `num_layers` > 0, equal to `{arch}.block_count`.
2. **Expose `use_mmap`, `use_mlock`, `flash_attn`, `type_k/type_v` and `n_ubatch` in the config, and log what was used.** *Test:* the values appear in `engine_snapshot()` and round-trip through the harness JSON.
3. **Replace the ISA→n_batch table with a measured default, or keep llama.cpp's defaults until there is data.** *Test:* a llama-bench `-b 256,512,1024 -ub 128,256,512` sweep on the target is committed, and the default matches the best pp result.
4. **Harness: use `usage.completion_tokens`, discard warmup runs, and report median plus spread.** *Test:* the token count equals llama-cpp's usage field. On one model, the harness tg number is within a stated tolerance of llama-bench `tg128`.

---

## 2. Fit estimator (`fit/estimator.py`, `hal/profiler.py`)

### What exists in the repo
- `BYTES_PER_PARAM` (estimator.py:336-344). KV is computed as `layers×kv_heads×head_dim×2×bytes×ctx` (L437-444). `head_dim = hidden//n_heads` (L381-384). The heuristic is `0.0005·params_b·ctx/1000` GB (L446, L521). `KV_BYTES_PER_ELEMENT_Q8 = 1` (L350) [verified].
- **The CLI never passes `geometry`** (cli/main.py:55, 70, 104), so every shipped estimate uses the heuristic [verified].
- Profiler: a single-threaded `bytearray(src)` copy of 64 MB, ×2 (profiler.py:133-155). Tok/s = `bw/(params·0.55)`, capped at 100 (L221-227). Context is derived from the same heuristic and capped at 8192 (L233-241) [verified].

### Bugs and overclaims (arithmetic from the repo's own functions)
1. **The KV heuristic underestimates badly.** Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128, from HF config https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/raw/main/config.json) is 128 KiB/token at fp16. At 8K context: geometry gives 1.00 GiB, the heuristic gives 0.033 GB, about 30× low. Qwen2.5-32B (64 layers, 8 KV heads, 5120/40=128; https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/raw/main/config.json): 2.0 GiB vs 0.134 GB, about 15× low [verified by running the code]. The code calls this "conservative", which is wrong.
2. **The `max_context_for_budget` heuristic over-recommends.** For a 32.8B Q4_K_M model with 30 GB available, the heuristic returns 131072 tokens. Real geometry returns 32768 [verified by running].
3. **head_dim is wrong for models that declare `key_length`.** Gemma-7B: 3072/16=192 vs actual head_dim 256 (25% low). Gemma-2-9B: 224 vs 256 (https://huggingface.co/unsloth/gemma-2-9b-it/raw/main/config.json) [verified]. The GGUF spec has optional `attention.key_length`/`value_length`, which default to n_embd/n_head (gguf.md L476-477) [verified]. Sliding-window layers (Gemma-2/3) are not modelled [verified absent].
4. **q8_0 is counted as 1.0 B/elem, but it is 1.0625.** That is 6% low [verified].
5. **Weight bytes per parameter are low for K-quants.** gguf-parser's README shows Qwen2.5-7B Q4_K_M at 4.36 GiB / 7.62 B = 4.91 bpw (≈0.61 B/param), and Llama-3.1-70B Q4_K_M at 4.82 bpw. The table assumes 0.55 (https://github.com/gpustack/gguf-parser-go) [verified]. The GGUF file size is the ground truth.
6. **Unit mix-up in the profiler (new finding).** `max_params = usable·1024³/(bpp·1e9)` mixes GiB and GB, so `model_size_gb` is always about 7% larger than `usable_ram_gb`. As a result `remaining_ram_gb < 0` and context falls back to **2048** for every available RAM below roughly 38 GB. Running the code: avail 6, 10, 14, 30 and 40 GB all give ctx 2048, and 64 GB gives 8192 [verified]. So the profiler's "up to 8192" never happens on the 8–16 GB target. The 2048 comes from a bug, not from reasoning.
7. **"Max model" fills all RAM.** 14 GB available gives 23.4B params and no room for KV [verified].
8. **The bandwidth probe measures the wrong thing.** It is single-threaded, and it allocates fresh 64 MB pages every iteration, so page-fault cost is included [inferred]. It returned 8.1 GB/s in this sandbox, which is meaningless for a laptop. For reference: i7-10610U is listed at 45.8 GB/s max, with DDR4-2666 at 2 ch × 8 B × 2666 MT/s ≈ 42.7 GB/s theoretical (Intel ARK above) [verified spec plus arithmetic]. The ×2 "read+write" factor is the STREAM counting convention for Copy. Write-allocate hardware actually moves more (https://www.cs.virginia.edu/stream/ref.html) [verified].
9. **Tok/s = bw/size is a ceiling, not an estimate.** The repo's own roadmap says the same (docs/architecture/local-inference-roadmap.md:13-19) [verified].

### What comparable projects do
- **llama.cpp `--fit` (default on) and `llama-fit-params`:** they build dummy model and context objects and use the backend's own projected memory use. They reduce ctx, then n_gpu_layers, then tensor overrides to meet a margin, and print `llama_memory_breakdown_print` (model/context/compute) (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/fit-params/README.md; PR https://github.com/ggml-org/llama.cpp/pull/16653) [verified]. llama-bench also accepts `-fitt/-fitc` [verified].
- **gguf-parser-go:** reads metadata only (it works on remote files too). It estimates RAM/VRAM per ctx, mmap, flash-attn and offload layer counts, handles SWA and MLA, and claims a deviation of about 100 MiB. MAX TPS comes from user-supplied FLOPS and bandwidth ("speed of light") [verified README; accuracy claim unverified].
- **Ollama:** the new estimator is weights + KV using `embedding_length/head_count`, with no compute buffer. Open PR #17201 adds per-layer `key_length`/`value_length` and a compute term, and reports a −20.6% error on deepseek-v2 before the fix (https://api.github.com/repos/ollama/ollama/pulls/17201). The older `EstimateGPULayers` counted KV plus graph partial/full plus overhead (search excerpts only) [partly verified]. Ollama is also moving to tracking real allocations (PR #11090) [verified from excerpt; unverified in depth].

### Alternatives
- Before loading: `gguf.GGUFReader(path).fields` and `.tensors` (pip `gguf`; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/examples/reader.py) [verified]. Summing tensor bytes gives the exact weights.
- Shell out to `llama-fit-params` or gguf-parser and compare against them.
- Bandwidth: STREAM Triad with OpenMP and arrays at least 4× LLC (for 8 MB LLC that means ≥32 MB per array; STREAM ref) [verified rule], or Intel MLC `--max_bandwidth` (needs root/MSR on Linux; Windows driver via `-e0`) (https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html) [verified].

### Recommended next steps (ordered)
1. **Fill `ModelGeometry` from GGUF.** Use `block_count`, `head_count`, `head_count_kv` (scalar or per-layer), and `key_length`/`value_length` with the n_embd/n_head fallback. Pass it from the CLI. *Test:* for Llama-3.1-8B, `per_token` = 131072 B. For Gemma-2-9B, head_dim 256 is used.
2. **Weights = summed tensor bytes (or file size), not params×bpp.** *Test:* matches `llama_model_size()` after load to within 1%.
3. **Use exact KV type sizes** (f16 2, q8_0 34/32, q4_0 18/32) and separate K and V types. *Test:* unit tests against the ggml block sizes.
4. **Remove the heuristic, or return `UNKNOWN` when geometry is missing.** *Test:* no "conservative" label on a value lower than the geometry value.
5. **Fix the GiB/GB mix and reserve KV before maximising params.** *Test:* profiler ctx changes monotonically with RAM, and the reported "max model" leaves room for KV at ctx ≥ 4096.
6. **Add a compute-buffer term and validate against llama.cpp.** *Test:* the estimate is within ±10% of `llama_memory_breakdown_print` (or llama-fit-params) for 3 architectures × 2 contexts.
7. **Replace the bandwidth probe with a multi-threaded NumPy or C triad over ≥256 MB, and label tok/s as an upper bound.** *Test:* result within ±15% of STREAM Triad on the same machine.

---

## Benchmarks the owner must run on his own machine
1. `mlc --max_bandwidth` (or STREAM Triad with `OMP_NUM_THREADS` = 1, 2, 4, 8) vs EigenCore `_measure_bandwidth()`.
2. `llama-bench -m <model> -p 512 -n 128 -t 2,4,8 -b 256,512,1024 -ub 128,256,512 -r 5 -o json`, on AC power, noting thermals.
3. `llama-bench -ctk f16,q8_0,q4_0 -ctv f16,q8_0 -fa on,off -d 0,4096`: KV-type and depth impact on tg and pp.
4. `llama-bench -lm mmap,mlock,none` with a model larger than free RAM (e.g. a 32B Q4_K_M on 16 GB): records SSD-paging tok/s and load time.
5. `llama-fit-params --model <gguf>` plus peak RSS (`/usr/bin/time -v`) vs `estimate_fit()` at ctx 2048, 8192 and 32768.
6. EigenCore harness vs llama-bench tg128 on the same model and threads, warm vs cold.

Arithmetic for the target, not a measurement: a 32.8B Q4_K_M at about 0.61 B/param ≈ 20 GB of weights, which already exceeds 16 GB RAM before KV. With mmap it will page from SSD, and decode will be SSD-bound rather than DRAM-bound [inferred].
