"""
Scalar reference kernels — Phase 6 slice 1.

These are deliberately simple, correct-by-inspection implementations of the
core math a custom inference engine needs. They serve two purposes:

1. A trusted reference for differential-testing any future optimized
   (SIMD/assembly) kernel — an optimized kernel must match these within a
   documented tolerance.
2. A backend boundary (`BackendProtocol`) so a future native engine can be
   swapped in behind the same interface without touching callers.

NO performance is claimed for these. They exist to make correctness
measurable before optimization.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

# GGUF Q8_0 block: 32 values per block, one fp16 scale.
Q8_0_BLOCK_SIZE = 32
# GGUF Q4_0 block: 32 values per block (4-bit packed), one fp16 scale.
Q4_0_BLOCK_SIZE = 32


class BackendProtocol(Protocol):
    """Interface a custom/native inference backend must satisfy.

    Implementations (e.g. a future compiled sidecar) plug in here; the
    llama.cpp path remains the default until a kernel proves itself.
    """

    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray: ...

    def dequantize_q8_0(self, blocks: np.ndarray) -> np.ndarray: ...

    def dequantize_q4_0(self, packed: np.ndarray, scales: np.ndarray) -> np.ndarray: ...

    def quantized_dot_q8_0(
        self, packed: np.ndarray, vec: np.ndarray, group_size: int = Q8_0_BLOCK_SIZE
    ) -> float: ...


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Root-mean-square layer norm: x / rms(x) * weight."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    ms = np.mean(x * x)
    return x * (1.0 / math.sqrt(float(ms) + eps)) * weight


def dequantize_q8_0(blocks: np.ndarray) -> np.ndarray:
    """Dequantize GGUF Q8_0 blocks.

    `blocks` has shape (n_blocks, 1 + Q8_0_BLOCK_SIZE): column 0 is the fp16
    scale, the rest are int8 quantized values. Returns float32 vector of
    length n_blocks * Q8_0_BLOCK_SIZE.
    """
    blocks = np.asarray(blocks)
    if blocks.ndim != 2 or blocks.shape[1] != 1 + Q8_0_BLOCK_SIZE:
        raise ValueError(f"expected (n, {1 + Q8_0_BLOCK_SIZE}) blocks, got {blocks.shape}")
    scales = blocks[:, 0].astype(np.float32)
    quants = blocks[:, 1:].astype(np.float32)
    return (scales[:, None] * quants).reshape(-1)


def dequantize_q4_0(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize GGUF Q4_0 blocks.

    `packed` is uint8 of shape (n_blocks, Q4_0_BLOCK_SIZE // 2); each byte
    holds two 4-bit values (low nibble first). `scales` is fp16 (n_blocks,).
    Returns float32 of length n_blocks * Q4_0_BLOCK_SIZE.
    """
    packed = np.asarray(packed)
    scales = np.asarray(scales, dtype=np.float32)
    if packed.ndim != 2 or packed.shape[1] != Q4_0_BLOCK_SIZE // 2:
        raise ValueError(f"expected (n, {Q4_0_BLOCK_SIZE // 2}) packed, got {packed.shape}")
    if scales.shape[0] != packed.shape[0]:
        raise ValueError("scales/packed block count mismatch")
    low = (packed & 0x0F).astype(np.float32) - 8.0
    high = (packed >> 4).astype(np.float32) - 8.0
    # interleave: low nibble = even index, high nibble = odd index
    out = np.empty((packed.shape[0], Q4_0_BLOCK_SIZE), dtype=np.float32)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return (scales[:, None] * out).reshape(-1)


def quantized_dot_q8_0(
    packed: np.ndarray, vec: np.ndarray, group_size: int = Q8_0_BLOCK_SIZE
) -> float:
    """Dot product of a Q8_0-quantized weight with a float vector.

    Dequantizes block-by-block and accumulates in float32. This is the
    hot inner loop of CPU LLM decode, so it's the first kernel a native
    implementation would replace.
    """
    w = dequantize_q8_0(packed)
    v = np.asarray(vec, dtype=np.float32)
    if w.shape[0] != v.shape[0]:
        raise ValueError(f"shape mismatch: weights {w.shape} vs vec {v.shape}")
    n_groups = w.shape[0] // group_size
    acc = 0.0
    for g in range(n_groups):
        acc += float(
            np.dot(
                w[g * group_size : (g + 1) * group_size], v[g * group_size : (g + 1) * group_size]
            )
        )
    return acc


class ScalarBackend:
    """Reference implementation of BackendProtocol (NumPy, scalar loops)."""

    name = "scalar-numpy"

    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return rms_norm(x, weight, eps)

    def dequantize_q8_0(self, blocks: np.ndarray) -> np.ndarray:
        return dequantize_q8_0(blocks)

    def dequantize_q4_0(self, packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
        return dequantize_q4_0(packed, scales)

    def quantized_dot_q8_0(
        self, packed: np.ndarray, vec: np.ndarray, group_size: int = Q8_0_BLOCK_SIZE
    ) -> float:
        return quantized_dot_q8_0(packed, vec, group_size)
