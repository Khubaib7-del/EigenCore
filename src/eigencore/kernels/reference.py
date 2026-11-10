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
# Raw on-disk block sizes: 2-byte fp16 scale + payload.
Q8_0_BLOCK_BYTES = 2 + Q8_0_BLOCK_SIZE  # 34
Q4_0_BLOCK_BYTES = 2 + Q4_0_BLOCK_SIZE // 2  # 18


class BackendProtocol(Protocol):
    """Interface a custom/native inference backend must satisfy.

    Implementations (e.g. a future compiled sidecar) plug in here; the
    llama.cpp path remains the default until a kernel proves itself.
    """

    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray: ...

    def dequantize_q8_0(self, blocks: np.ndarray) -> np.ndarray: ...

    def dequantize_q4_0(
        self, packed: np.ndarray, scales: np.ndarray | None = None
    ) -> np.ndarray: ...

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


def dequantize_q4_0(packed: np.ndarray | bytes, scales: np.ndarray | None = None) -> np.ndarray:
    """Dequantize GGUF Q4_0 blocks (ggml layout).

    ggml packs 32 values into 16 bytes per block: byte j holds element j in
    its LOW nibble and element j+16 in its HIGH nibble (strided, not
    interleaved), each nibble biased by -8 and scaled by the block's fp16
    scale. Accepts either (a) a raw bytes buffer of n_blocks × 18-byte
    blocks, or (b) `packed` uint8 (n, 16) plus `scales` fp16 (n,).
    Returns float32 of length n_blocks * Q4_0_BLOCK_SIZE.
    """
    if isinstance(packed, (bytes, bytearray)):
        raw = np.frombuffer(bytes(packed), dtype=np.uint8)
        if raw.size % Q4_0_BLOCK_BYTES != 0:
            raise ValueError(f"raw buffer not a multiple of {Q4_0_BLOCK_BYTES} bytes")
        blocks = raw.reshape(-1, Q4_0_BLOCK_BYTES)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
        packed = blocks[:, 2:]
    else:
        packed = np.asarray(packed)
        if scales is None:
            raise ValueError("scales required when packed is an array")
        scales = np.asarray(scales, dtype=np.float32)
        if packed.ndim != 2 or packed.shape[1] != Q4_0_BLOCK_SIZE // 2:
            raise ValueError(f"expected (n, {Q4_0_BLOCK_SIZE // 2}) packed, got {packed.shape}")
        if scales.shape[0] != packed.shape[0]:
            raise ValueError("scales/packed block count mismatch")
    low = (packed & 0x0F).astype(np.float32) - 8.0
    high = (packed >> 4).astype(np.float32) - 8.0
    # ggml strided order: low nibbles fill elements 0..15, high nibbles 16..31
    out = np.empty((packed.shape[0], Q4_0_BLOCK_SIZE), dtype=np.float32)
    out[:, : Q4_0_BLOCK_SIZE // 2] = low
    out[:, Q4_0_BLOCK_SIZE // 2 :] = high
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

    def dequantize_q4_0(
        self, packed: np.ndarray | bytes, scales: np.ndarray | None = None
    ) -> np.ndarray:
        return dequantize_q4_0(packed, scales)

    def quantized_dot_q8_0(
        self, packed: np.ndarray, vec: np.ndarray, group_size: int = Q8_0_BLOCK_SIZE
    ) -> float:
        return quantized_dot_q8_0(packed, vec, group_size)
