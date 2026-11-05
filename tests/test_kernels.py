"""Differential/correctness tests for the P6 scalar reference kernels.

These pin the reference math so any future optimized kernel can be
compared against it. Tolerances are documented per test.
"""

import sys
from pathlib import Path

import numpy as np

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
    # one block, known values: scale=1.0, packed bytes chosen so low nibbles = 0..7, high = 8..15
    packed = np.array(
        [
            [
                0x10,
                0x32,
                0x54,
                0x76,
                0x98,
                0xBA,
                0xDC,
                0xFE,
                0x21,
                0x43,
                0x65,
                0x87,
                0xA9,
                0xCB,
                0xED,
                0x0F,
            ]
        ],
        dtype=np.uint8,
    )
    scales = np.array([1.0], dtype=np.float16)
    out = dequantize_q4_0(packed, scales)
    assert out.shape == (Q4_0_BLOCK_SIZE,)
    expected_low = (
        np.array(
            [0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF],
            dtype=np.float32,
        )
        - 8.0
    )
    expected_high = (
        np.array(
            [0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x0],
            dtype=np.float32,
        )
        - 8.0
    )
    expected = np.empty(Q4_0_BLOCK_SIZE, dtype=np.float32)
    expected[0::2] = expected_low
    expected[1::2] = expected_high
    np.testing.assert_allclose(out, expected, atol=1e-6)


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
