# Matrix Mathematics & Hardware Execution

**Date:** 2026-07-17

## Why Row Echelon Form Breaks LLMs

Gaussian elimination / Row Echelon Form is designed to solve systems of linear equations by introducing zeros below pivot points. While this simplifies matrices algebraically, it is destructive for neural network weights:

**Destruction of information (rank loss):** Row reduction changes relationships between numbers. LLM weight matrices contain highly complex, non-linear relationships. Row-reducing a weight matrix permanently erases subtle linguistic patterns learned during training.

**Information compression vs. equation solving:** Gaussian elimination simplifies equations but doesn't compress information. Making an LLM cheaper to run requires dimensionality reduction (keeping core meaning with fewer numbers), not algebraic simplification.

## The Correct Approach: Low-Rank Decomposition (SVD / LoRA)

Instead of equation solving, use Singular Value Decomposition (SVD):

A massive `1000 × 1000` matrix (1,000,000 parameters) is factored into two smaller matrices:
- `1000 × k` and `k × 1000` where k << 1000

For k=4: only 8,000 parameters to store and compute, while preserving the subspace that carries meaningful weight relationships.

**LoRA (Low-Rank Adaptation) applies this to fine-tuning:**
- Freeze base weights W (no gradient computation needed)
- Train two small matrices A (d × r) and B (r × d) where r is the rank (typically 4-16)
- At inference: W' = W + A·B
- The adapter (A·B) is ~10MB vs the full model's ~2GB

## SIMD Hardware Execution

Modern CPUs contain hidden specialized AI engines accessible through assembly-level vector instructions:

**Intel/AMD:**
- AVX2: 256-bit vectors (8 floats simultaneously)
- AVX-512: 512-bit vectors (16 floats simultaneously)
- AMX (Advanced Matrix Extensions): dedicated matrix multiply tiles

**ARM:**
- NEON: 128-bit SIMD
- SVE (Scalable Vector Extension): variable-width vectors

**C++ SIMD example:**

```cpp
#include <immintrin.h>  // Intel/AMD AVX intrinsics

// Multiply 8 floats simultaneously in a single CPU clock cycle
void fast_matrix_multiply(float* matrixA, float* matrixB, float* result) {
    __m256 a = _mm256_loadu_ps(matrixA);   // load 8 floats into vector register
    __m256 b = _mm256_loadu_ps(matrixB);   // load 8 floats into vector register
    __m256 c = _mm256_mul_ps(a, b);        // multiply all 8 pairs at once
    _mm256_storeu_ps(result, c);           // store result back to memory
}
```

By scaling this to thousands of matrix rows, you get a custom runtime engine executing model inference at hardware speed.

## Memory Bandwidth: The Real CPU Bottleneck

On CPU, the bottleneck is NOT compute (FLOPs) — it's memory bandwidth.

**Formula:** `tokens_per_second ≈ memory_bandwidth / model_size_in_memory`

For i7-10610U (~30 GB/s bandwidth) with 3B model at Q4 (~2GB):
- Theoretical max: ~15 tok/s
- Practical (overhead): ~10-12 tok/s

This ceiling exists regardless of how clever the SIMD kernels are. It's why activation sparsity (skipping multiplications = skipping memory reads) and speculative decoding (fewer forward passes = fewer full-model memory sweeps) are the highest-leverage optimizations for CPU.

## Quantization

Reducing numerical precision of weights to save memory and compute:

- FP32: 4 bytes per parameter → 3B model = 12GB (doesn't fit 16GB RAM)
- FP16: 2 bytes per parameter → 3B model = 6GB
- INT8: 1 byte per parameter → 3B model = 3GB
- INT4 (Q4): 0.5 bytes per parameter → 3B model = 1.5GB

Standard quantization formats: GGUF (used by llama.cpp), AWQ, GPTQ.

Quality degradation at Q4 is minimal for models > 3B parameters — the redundancy in larger models absorbs the precision loss.
