"""
Q4_K reference kernel — Stage E slice 1.

GGUF Q4_K super-block layout (256 values = 8 sub-blocks of 32):
  bytes [0:2)   fp16  d      (super-block scale)
  bytes [2:4)   fp16  dmin   (super-block min)
  bytes [4:16)  12 bytes of 8x (6-bit sc + 6-bit min), packed per ggml
  bytes [16:144) 128 bytes of 4-bit quants, low nibble = even index

Dequant: y[j] = d*sc[b]*q[j] - dmin*m[b]   (ggml uses d*sc*q - dm, note the MINUS)

This is the scalar reference: bit-exact-tested against gguf-py. A future
SIMD/C-intrinsics kernel must match it within documented tolerance.
NO performance is claimed here.
"""

from __future__ import annotations

import numpy as np

QK_K = 256  # values per Q4_K super-block
Q4_K_BLOCK_BYTES = 2 + 2 + 12 + QK_K // 2  # 144
Q4_K_SCALE_SIZE = 12


def _get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 12 scale bytes into 8 (sc, min) 6-bit pairs (ggml layout)."""
    scales = np.asarray(scales, dtype=np.uint8).reshape(-1, 3, 4)
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape(-1, 8).astype(np.float32), mn.reshape(-1, 8).astype(np.float32)


def dequantize_q4_k(blocks: np.ndarray | bytes) -> np.ndarray:
    """Dequantize Q4_K super-blocks to float32 (n_blocks * 256,).

    Accepts a raw bytes buffer of n*144 bytes, or a uint8 array (n, 144).
    """
    if isinstance(blocks, (bytes, bytearray)):
        raw = np.frombuffer(bytes(blocks), dtype=np.uint8)
        if raw.size % Q4_K_BLOCK_BYTES != 0:
            raise ValueError(f"raw buffer not a multiple of {Q4_K_BLOCK_BYTES} bytes")
        blocks = raw.reshape(-1, Q4_K_BLOCK_BYTES)
    blocks = np.asarray(blocks, dtype=np.uint8)
    if blocks.ndim != 2 or blocks.shape[1] != Q4_K_BLOCK_BYTES:
        raise ValueError(f"expected (n, {Q4_K_BLOCK_BYTES}) blocks, got {blocks.shape}")
    n_blocks = blocks.shape[0]

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    sc, m = _get_scale_min(blocks[:, 4 : 4 + Q4_K_SCALE_SIZE])
    qs = blocks[:, 4 + Q4_K_SCALE_SIZE :]

    # 4-bit quants: byte j//2 within each 32-value sub-block, low nibble even
    qs = qs.reshape(n_blocks, -1, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    qs = (qs & np.uint8(0x0F)).reshape(n_blocks, -1, 32).astype(np.float32)

    d_sc = (d[:, None] * sc).reshape(n_blocks, -1, 1)
    dm = (dmin[:, None] * m).reshape(n_blocks, -1, 1)
    return (d_sc * qs - dm).reshape(n_blocks, QK_K).reshape(-1)


def quantized_dot_q4_k(packed: np.ndarray | bytes, vec: np.ndarray) -> float:
    """Dot product of a Q4_K-quantized weight row with a float vector.

    Dequantizes super-block by super-block and accumulates in float32.
    """
    w = dequantize_q4_k(packed)
    v = np.asarray(vec, dtype=np.float32)
    if w.shape[0] != v.shape[0]:
        raise ValueError(f"shape mismatch: weights {w.shape} vs vec {v.shape}")
    acc = 0.0
    for i in range(0, w.shape[0], QK_K):
        acc += float(np.dot(w[i : i + QK_K], v[i : i + QK_K]))
    return acc
