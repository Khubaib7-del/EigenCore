"""Phase 6: own low-level kernels (reference implementations + backend boundary)."""

from eigencore.kernels.q4k import Q4_K_BLOCK_BYTES, QK_K, dequantize_q4_k, quantized_dot_q4_k
from eigencore.kernels.reference import (
    Q4_0_BLOCK_SIZE,
    Q8_0_BLOCK_SIZE,
    BackendProtocol,
    ScalarBackend,
    dequantize_q4_0,
    dequantize_q8_0,
    quantized_dot_q8_0,
    rms_norm,
)

__all__ = [
    "Q4_K_BLOCK_BYTES",
    "QK_K",
    "Q4_0_BLOCK_SIZE",
    "Q8_0_BLOCK_SIZE",
    "BackendProtocol",
    "ScalarBackend",
    "dequantize_q4_k",
    "dequantize_q4_0",
    "dequantize_q8_0",
    "quantized_dot_q4_k",
    "quantized_dot_q8_0",
    "rms_norm",
]
