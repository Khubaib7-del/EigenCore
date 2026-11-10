"""Differential/correctness tests for the P6 scalar reference kernels.

These pin the reference math so any future optimized kernel can be
compared against it. Tolerances are documented per test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.kernels import (
    Q4_0_BLOCK_SIZE,
    Q8_0_BLOCK_SIZE,
    ScalarBackend,
    dequantize_q4_0,
    dequantize_q8_0,
    quantized_dot_q8_0,
    rms_norm,
)

rng = np.random.default_rng(42)


def test_rms_norm_matches_manual_formula():
    x = rng.standard_normal(128).astype(np.float32)
    w = rng.standard_normal(128).astype(np.float32)
    out = rms_norm(x, w, eps=1e-5)
    expected = x / np.sqrt(np.mean(x**2) + 1e-5) * w
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    # RMS of normalized vector ≈ 1 before weight
    normalized = rms_norm(x, np.ones_like(w))
    assert abs(float(np.sqrt(np.mean(normalized**2))) - 1.0) < 1e-4


def test_q8_0_dequant_roundtrip():
    n_blocks = 64
    scales = rng.uniform(0.001, 0.1, n_blocks).astype(np.float16)
    quants = rng.integers(-127, 128, (n_blocks, Q8_0_BLOCK_SIZE), dtype=np.int8)
    blocks = np.concatenate([scales.astype(np.float32)[:, None], quants], axis=1)
    out = dequantize_q8_0(blocks)
    assert out.shape == (n_blocks * Q8_0_BLOCK_SIZE,)
    expected = (scales.astype(np.float32)[:, None] * quants.astype(np.float32)).reshape(-1)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)


def test_q4_0_dequant_nibble_order():
    # ggml strided layout: byte j low nibble -> element j, high nibble -> element j+16
    packed = np.arange(16, dtype=np.uint8).reshape(1, 16) * 0x11  # 0x00,0x11,...,0xFF
    scales = np.array([1.0], dtype=np.float16)
    out = dequantize_q4_0(packed, scales)
    assert out.shape == (Q4_0_BLOCK_SIZE,)
    expected = np.concatenate(
        [np.arange(16, dtype=np.float32) - 8.0, np.arange(16, dtype=np.float32) - 8.0]
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_q4_0_raw_block_input():
    # raw 18-byte block: fp16 scale 1.0 + 16 packed bytes
    packed = (np.arange(16, dtype=np.uint8) * 0x11).reshape(1, 16)
    raw = np.concatenate([np.array([1.0], dtype=np.float16).view(np.uint8), packed[0]]).tobytes()
    out = dequantize_q4_0(raw)
    assert out.shape == (Q4_0_BLOCK_SIZE,)
    np.testing.assert_allclose(
        out, dequantize_q4_0(packed, np.array([1.0], dtype=np.float16)), atol=1e-6
    )
    try:
        dequantize_q4_0(b"\x00" * 7)  # not a multiple of 18
        raise AssertionError("should reject bad raw length")
    except ValueError:
        pass


def test_q4_0_q8_0_bitexact_vs_gguf_py():
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize as gguf_dequantize

    rng_local = np.random.default_rng(7)
    # Q4_0: random raw blocks
    n = 8
    raw_q4 = rng_local.integers(0, 256, n * 18, dtype=np.uint8).tobytes()
    ours = dequantize_q4_0(raw_q4)
    ref = gguf_dequantize(np.frombuffer(raw_q4, dtype=np.uint8), gguf.GGMLQuantizationType.Q4_0)
    np.testing.assert_array_equal(ours, ref.astype(np.float32))
    # Q8_0: random raw blocks
    raw_q8 = rng_local.integers(0, 256, n * 34, dtype=np.uint8).tobytes()
    blocks = np.frombuffer(raw_q8, dtype=np.uint8).reshape(n, 34)
    ours8 = dequantize_q8_0(
        np.concatenate(
            [
                blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)[:, None],
                blocks[:, 2:].copy().view(np.int8).astype(np.float32),
            ],
            axis=1,
        )
    )
    ref8 = gguf_dequantize(np.frombuffer(raw_q8, dtype=np.uint8), gguf.GGMLQuantizationType.Q8_0)
    np.testing.assert_allclose(ours8, ref8.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_q4_0_shape_validation():
    try:
        dequantize_q4_0(np.zeros((4, 4), dtype=np.uint8), np.ones(4, dtype=np.float16))
        raise AssertionError("should reject wrong packed width")
    except ValueError:
        pass


def test_quantized_dot_q8_0_matches_dequantized_dot():
    n_blocks = 32
    total = n_blocks * Q8_0_BLOCK_SIZE
    scales = rng.uniform(0.001, 0.1, n_blocks).astype(np.float16)
    quants = rng.integers(-127, 128, (n_blocks, Q8_0_BLOCK_SIZE), dtype=np.int8)
    blocks = np.concatenate([scales.astype(np.float32)[:, None], quants], axis=1)
    vec = rng.standard_normal(total).astype(np.float32)
    got = quantized_dot_q8_0(blocks, vec)
    expected = float(np.dot(dequantize_q8_0(blocks), vec))
    assert abs(got - expected) < 1e-2 * max(1.0, abs(expected))  # fp32 accumulation tolerance


def test_scalar_backend_implements_protocol():
    backend = ScalarBackend()
    x = rng.standard_normal(32).astype(np.float32)
    w = np.ones(32, dtype=np.float32)
    np.testing.assert_allclose(backend.rms_norm(x, w), rms_norm(x, w), rtol=1e-7)
    n_blocks = 4
    scales = np.full(n_blocks, 0.01, dtype=np.float16)
    quants = rng.integers(-127, 128, (n_blocks, Q8_0_BLOCK_SIZE), dtype=np.int8)
    blocks = np.concatenate([scales.astype(np.float32)[:, None], quants], axis=1)
    np.testing.assert_allclose(backend.dequantize_q8_0(blocks), dequantize_q8_0(blocks))
    assert backend.name == "scalar-numpy"
