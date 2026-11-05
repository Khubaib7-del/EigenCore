"""Phase 6: own low-level kernels (reference implementations + backend boundary)."""

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
    "Q4_0_BLOCK_SIZE",
    "Q8_0_BLOCK_SIZE",
    "BackendProtocol",
    "ScalarBackend",
    "dequantize_q4_0",
    "dequantize_q8_0",
    "quantized_dot_q8_0",
    "rms_norm",
]
