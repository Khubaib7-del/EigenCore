# EigenCore kernels review: SIMD/assembly layer

Repo: Khubaib7-del/EigenCore @ c92e3af (read-only clone). Tags: **[verified]** = checked in code, by a run, or on a page I opened · **[inferred]** = my own reasoning or arithmetic · **[unverified]** = not checked.

## 1. What exists (file:line)
- `src/eigencore/kernels/reference.py:48-53` `rms_norm`; `:56-68` `dequantize_q8_0` (takes an `(n,33)` array: column 0 is the scale, then 32 quants); `:71-90` `dequantize_q4_0`; `:93-114` `quantized_dot_q8_0` (dequantizes, then runs a float dot product per group); `:30-45` `BackendProtocol`; `:117-134` `ScalarBackend`. [verified]
- Nothing calls the kernels. The only references are the re-exports at `src/eigencore/__init__.py:31,58`. Inference runs through llama-cpp-python, so these kernels never run on the hot path. [verified]
- `tests/test_kernels.py`: 6 tests. `:49-94` locks in Q4_0 nibble order as **interleaved** (`expected[0::2]=low`, `[1::2]=high`). [verified]
- ISA detection, `hal/profiler.py:91-108`: substring search on `/proc/cpuinfo`. `hal/_cpuid_win.py`:
  - `:28-41` `_check_os_enabled_avx` always returns True.
  - `:74-77` treats AVX as implying FMA and F16C.
  - `:80-81` treats AVX512F as implying BW.
  - `:103-104` fallback claims **i7-12/13/14 have AVX-512**. That is wrong for consumer Alder/Raptor Lake, which ship it disabled [inferred, common knowledge]. These chips have AVX-VNNI instead, and no detection maps it.
  [verified for code]
- The roadmap (`local-inference-roadmap.md:48-52`) correctly says to profile before writing assembly. `matrix-math-foundations.md:57-67` already states that decode is bandwidth-bound. [verified]

## 2. Format correctness against ggml
I cross-checked against `gguf` (gguf-py) `quantize`/`dequantize`. (Check script not committed; it round-trips random blocks through `gguf.quants`.)

| Format | ggml spec | EigenCore | Result |
|---|---|---|---|
| Q8_0 | 34 B per block: fp16 d + 32×int8; x = q·d | Math is right, but the container is `(n,33)` float, not the raw 34-byte block | **max abs diff 0.0** when the scale is passed as float [verified]. If a caller builds the block as an int8 array, the scale is silently truncated: diff 3.9 [verified] |
| Q4_0 | 18 B per block: fp16 d + 16 B; byte j: low nibble → element j, high nibble → element j+16; x = (q−8)·d | Interleaves the halves (low → 2j, high → 2j+1) | **WRONG: max abs diff 5.50** [verified] |

ggml's `dequantize_row_q4_0` writes `y[i*qk + j] = x0*d; y[i*qk + j + qk/2] = x1*d` (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c) [verified]. gguf-py does the same with `>> [0,4]` and a reshape [verified, installed package]. The unit test passes only because it encodes the same mistake.

Two further gaps:
- `quantized_dot_q8_0` does not match ggml numerics. ggml quantizes the activation to Q8_0 (Q8_K for k-quants) and accumulates in integers. As written, `group_size` has no effect. [verified code; ggml path inferred from `vec_dot_type` discussion in PR #10446]
- There is no Q4_K, which is the format most 30B GGUFs actually use. ggml's `block_q4_K` is 2·fp16 + 12 B scales + 128 B quants = 144 B per 256 weights = **0.5625 B/weight** (`ggml-common.h` static_assert, QK_K=256, K_SCALE_SIZE=12) [verified].

## 3. Bound analysis: compute-bound or bandwidth-bound?
i7-10610U: 4C/8T, 15 W TDP (25 W cTDP-up), max memory bandwidth 45.8 GB/s (LPDDR4-2933) or 42.7 GB/s (DDR4-2666 dual-channel), AVX2 and no AVX-512 (https://www.intel.com/content/www/us/en/products/sku/201896/…/specifications.html) [verified]. All figures below are [inferred].

- **Decode (matvec, batch 1).** Each weight is read once for 2 FLOPs, so intensity = 2 / 0.5625 = **3.6 FLOP/B**.
- **Peak compute.** A Skylake-class core does 2 FMA × 8 fp32 × 2 = 32 FLOP/cycle. Four cores at a sustained ~3 GHz gives ≈ **384 GFLOP/s**, so the ridge point is 384 / 40 ≈ **9.6 FLOP/B**.
- **The quantized path is slower than that peak.** Unpacking nibbles, `vpmaddubsw`, `vpmaddwd`, converting and scaling come to ~6–8 µops per 32 weights. That is roughly 12–15 weights/cycle/core, or ~150 G weights/s across 4 cores. Memory supplies 40e9 / 0.5625 ≈ **71 G weights/s**.
- **Verdict:** decode is **bandwidth-bound, with only about a 2× margin**. If the laptop throttles to ~2 GHz at 15 W, the margin shrinks. Thread count and frequency therefore matter, but faster SIMD cannot raise the tokens-per-second ceiling.
- **Ceiling for 32B Q4_K (if it fit in RAM).** ~18 GB is read per token, so 40 / 18 ≈ **2.2 tok/s** theoretical and ~1.5 practical.
- **Fit is the bigger problem.** 32B at Q4_K_M is ~19 GB, which does not fit in 8–16 GB. Pages then stream from the SSD at ~2–3 GB/s, which means **≤0.15 tok/s**. Realistic options:
  - ≤2.5 bpw (IQ2/Q2_K, ~10 GB, with a quality cost);
  - an MoE model such as Qwen3-30B-A3B, where only ~3B parameters are active (~1.7 GB read per token) but all weights must still be resident, so Q3-class at ~13 GB on 16 GB RAM.
  This fit problem dwarfs any kernel work.
- **Prefill (GEMM, N tokens).** Intensity is about 3.6·N FLOP/B, so prefill is **compute-bound from N ≈ 3**. 32B costs ~64 GFLOP per token; at 150–300 effective GFLOP/s that is **~2–5 tok/s prefill**. This is the phase where kernel quality shows.

## 4. What llama.cpp/ggml already does
- **Hand-tuned SIMD dot products** per architecture in `ggml/src/ggml-cpu/arch/x86/quants.c` (200 KB, with `__AVXVNNI__` / `__AVX512VNNI__` paths) and NEON under `arch/arm` [verified, directory listing via GitHub API and file grep].
- **Runtime repacking.** Q4_0_N_M file types were removed in favour of repacking Q4_0 at load time (PR #10446, merged 2024-12-07: https://github.com/ggml-org/llama.cpp/pull/10446) [verified]. `repack.cpp` now contains `gemv/gemm_q4_0_8x8_q8_0`, `q4_K_8x8_q8_K`, `q2_K_8x8_q8_K` and `iq4_nl_8x8`, plus a 665 KB `arch/x86/repack.cpp` [verified]. AVX2 Q4_0_8_8 (PR #8713: https://github.com/ggml-org/llama.cpp/pull/8713) gave **+18% pp512 but only +2.7% tg128** on a 7600X [verified]. That result confirms the analysis above: kernels speed up prefill, not decode.
- **llamafile tinyBLAS/sgemm** (Justine Tunney, https://justine.lol/matmul/): 30–500% faster prompt eval for F16/Q8_0/Q4_0, while eval tok/s stayed flat (for example, Mistral 7B q4_0 on Skylake: prompt 17→28 tok/s, eval 7→7). It was upstreamed as `ggml-cpu/llamafile/` [verified]. She writes C++ intrinsics and reads the disassembly rather than hand-writing assembly [verified].
- **AMX** (`ggml-cpu/amx/`) and **KleidiAI** (`ggml-cpu/kleidiai/`, Arm) backends exist [verified dirs]. Neither applies to Comet Lake [inferred].

## 5. Where custom kernels add value, and where they are redundant
**Redundant on this laptop:** Q4_0/Q8_0/Q4_K AVX2 dot products, repacking, sgemm prefill, AVX-512/AMX/KleidiAI paths. ggml already has all of them, and a Python-side kernel cannot replace ggml's kernels inside llama-cpp-python's graph anyway [inferred].

**Real value:**
1. **Activation-sparse matvec (TEAL).** TEAL reaches 40–50% sparsity training-free, with 1.53–1.8× decode speedups, and is compatible with weight quantization (https://arxiv.org/abs/2408.14690) [verified, GPU results]. On CPU this is the one kernel that cuts bytes read. It needs a column-blocked layout so skipped input channels skip weight memory [inferred]. I did not confirm whether ggml has it [unverified].
2. A new quant/layout format or fused ops that ggml lacks (for example sparse + quantized together).
3. Learning: an SIMD layer with differential tests is a credible portfolio piece.

Ship any real speedup as a ggml fork/PR built into llama-cpp-python (`CMAKE_ARGS`), not as a parallel engine.

## 6. Recommended next steps, with acceptance tests
1. **Fix Q4_0 ordering** (low nibble → j, high nibble → j+16) and rewrite `test_kernels.py:49-94`.
   - *Accept when:* 10k random blocks round-tripped through `gguf.quantize` and our dequant give **bit-exact** results (max abs diff == 0).
2. **Take raw bytes**: `(n,34)` / `(n,18)` uint8 blocks, scale read via `.view(np.float16)`.
   - *Accept when:* bit-exact against gguf-py, and passing a non-uint8 array raises.
3. **Add a Q4_K dequant.**
   - *Accept when:* bit-exact against gguf-py on random data and on 3 real tensors read with `gguf.GGUFReader`.
4. **Add a ggml-faithful `vec_dot_q4_K_q8_K` reference** that quantizes the activation and uses integer accumulation.
   - *Accept when:* within a documented bound (for example rel ≤1e-5) of ggml's `test-backend-ops` MUL_MAT output for the same inputs.
5. **Add a C-intrinsics kernel** (optionally a NASM `.asm` object linked in) in a shared library with cpuid+xgetbv runtime dispatch. Build with scikit-build-core; bind with cffi API mode or nanobind; zero-copy NumPy buffers; call per matrix, never per row (Python call overhead is µs-scale).
   - *Accept when:* bit-exact/bounded against the reference across 1000 randomized shapes, including tail blocks, and ≥ 80% of measured STREAM bandwidth on a 4096×11008 matvec.
6. **Fix ISA detection:**
   - match whole tokens, not substrings;
   - add `avx_vnni`, `avx512vl` and `avx512_bf16`;
   - delete the i7-12/13/14 AVX-512 claim;
   - implement a real XGETBV check.
   - *Accept when:* fixture tests pass on cpuinfo strings from Comet Lake, Alder Lake and Zen 4.

## 7. Benchmarks the owner must run (on the i7-10610U, plugged in, performance governor)
- **Bandwidth:** STREAM Triad or Intel MLC. This is the real roofline denominator; it is probably 30–38 GB/s, and single-channel RAM halves it.
- **llama.cpp speed:** `llama-bench -m <model> -p 512 -n 128 -t 1,2,4,8 -r 5` on 7B/8B Q4_0, Q4_K_M and Q8_0. Then compute achieved GB/s = model bytes × tg tok/s. If that is ≥ 80% of STREAM, decode kernels are not the bottleneck.
- **A/B builds:** with and without `GGML_LLAMAFILE`, and with repack on and off (check `--help` for the flag in your build).
- **Profile:** `perf record -g` (or VTune Hotspots + Memory Access) on `llama-bench` to get the share of time in `ggml_vec_dot_q4_K_q8_K` / gemv, plus `perf stat` LLC-miss and frequency counters to spot thermal throttling.
- **Target model:** the 30–40B model at the chosen quant, with `free -m` / page-fault counts to confirm whether it fits in RAM before attributing slowness to compute.
