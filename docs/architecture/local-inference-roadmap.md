# Local inference: accuracy and speed roadmap

**Status:** proposed engineering direction; no performance target in this document is verified. EigenCore currently delegates inference to `llama-cpp-python`.

## Current optimization integration caveat

`InferenceEngine` initializes layer-skip/sparsity components and exposes optimization statistics, but these are not measured end-to-end backend savings. Its speculative decoder is initialized but is not wired into the generation path. Treat all associated speedup fields as plans/estimates, not measured performance, until the execution path and benchmark evidence demonstrate otherwise.

## The 30–40B constraint

Parameter count alone is not a fit estimate. A 35B dense model stored at four bits has a raw lower bound of about 17.5 GB (about 16.3 GiB) for weights, and a typical Q4_K_M file (~4.85 bits per weight) is about 21 GB; quantization metadata, tensors kept at higher precision, allocator/backend overhead, the OS, and KV cache all add to that. KV-cache memory varies with layer count, KV-head count, head dimension, context length, and cache type. The model card and actual GGUF metadata must drive the estimate.

Decode for a dense model is often memory-bandwidth-bound: each generated token needs substantial weight traffic. Dividing available bandwidth by weight bytes gives only an optimistic ceiling; real rates are lower due to cache misses, dequantization, synchronization, sampling, thermal limits, and other overhead. GPU offload helps only to the extent the model layers and working set fit available VRAM; hybrid CPU/GPU execution also transfers or coordinates data.

Consequences:

- A Core i7 label is insufficient to predict performance. Generation, memory channels, RAM configuration, cooling, power mode, GPU and VRAM, backend build, model, quant, and context matter.
- 30–40B quantized models may be loadable on some consumer systems with sufficient RAM, but “runs” does not promise interactive speed. Smaller dense models or sparse/MoE models may be a better speed/quality fit, but must be evaluated per model.
- Mixture-of-Experts models read only their active experts per token (for example ~3.3B of 30.5B parameters), so they are the realistic 30B-class option for 16 GB machines; all experts still need RAM or SSD paging. PLANNED, unmeasured.
- Do not publish a tok/s guarantee from a bandwidth formula. Label all estimates as estimates and compare with warm/cold measured prefill and decode results.

## Work sequence

### 1. Make a trustworthy baseline

Use the existing llama.cpp-backed path first. Record model name/revision, GGUF hash, quant type, prompt tokens, output tokens, context, CPU/GPU and memory, OS, backend/version/build flags, thread count, GPU layers, power mode, and warm/cold state. Separate model load, prompt prefill, and decode throughput. Repeat runs and report median plus spread; retain raw artifacts.

Quality baseline should include task-specific examples and exact expected behavior where possible. For open-ended tasks record a fixed evaluation set and rubric/model version. Keep raw full-precision or higher-quality quant results as reference; compare each lower quant and each optimization on the same prompts and seeds where supported.

### 2. Improve fit and orchestration before custom kernels

- Estimate weights, runtime reserve, and KV cache independently; show available system RAM and VRAM, plus a safety margin.
- Make context length a first-class choice. Offer a conservative default and explain the memory/performance tradeoff instead of silently truncating.
- Select backend, threads, and GPU offload from detected capabilities, but let the user inspect and override every choice.
- Prefer efficient supported quantization and a well-built backend. Keep model download, integrity checking, and load failure messages explicit.
- Benchmark prompt prefill and token decode separately. Cache prefix/KV data only with correct session isolation and model/adapter identity.

### 3. Gate algorithmic optimizations on quality

| Candidate | Potential benefit | Required evidence / risk |
| --- | --- | --- |
| KV-cache quantization | Lower memory use, possibly longer usable context | Compare quality/perplexity and decode speed by cache type; validate attention behavior. |
| Speculative decoding | More accepted target tokens per target-model pass | Use a correct target/draft acceptance method; verify output distribution and benchmark draft overhead/acceptance rate. |
| Layer skipping | Less compute for selected layers | Requires model-specific calibration; compare task quality and output stability. Never silently enable an unvalidated plan. |
| Sparse activation/MoE-aware routing | Avoid work for inactive experts in compatible models | Confirm backend actually skips inactive work; a sparse architecture label alone does not establish execution savings. |
| Context compression | Fit useful history into a bounded window | Evaluate retention of constraints, code, names, and citations; preserve source references and disclose omissions. |
| SIMD/dequantization kernels | Improve CPU throughput | Differential-test against a trusted backend across supported ISA/quant combinations; scalar fallback and numerical tolerances required. |

### 4. Own low-level math incrementally

Start with a documented backend boundary and small independently tested kernels, not a replacement transformer engine. For each kernel define input layout, quant block format, rounding, accumulation precision, edge cases, and reference implementation. Candidate progression: tensor/quant readers; scalar dequantization; Q4/Q5/Q8 dot products; SIMD dispatch (AVX2/AVX-512/NEON only when OS and CPU support are verified); normalization/positional encoding; attention/KV-cache primitives. Keep a scalar reference and compare randomized and model-derived tensors to the backend. Add assembly/intrinsics only after a profile identifies a hot path.

No kernel becomes default until it passes correctness checks, supported-platform builds, representative model tests, and a benchmark showing a meaningful benefit without crossing the quality threshold. Retain llama.cpp as a compatibility/reference backend while the work is experimental.

## Release evidence checklist

- Hardware and model test matrix is named and reproducible.
- Fit estimate includes weights, runtime reserve, and KV cache; insufficient-memory behavior is safe.
- Prefill and decode metrics are reported separately with raw samples and measurement settings.
- Quality comparison passes the selected task suite; changes to quantization/skipping/compression document regressions and intended tradeoffs.
- Backend fallback and CPU/GPU feature dispatch are tested on supported configurations.
- User-facing performance estimates are labeled, and no result is generalized beyond measured configurations.
