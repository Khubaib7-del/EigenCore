"""Phase 2 model-fit estimation."""

from eigencore.fit.estimator import (
    BYTES_PER_PARAM,
    DEFAULT_RUNTIME_RESERVE_GB,
    DEFAULT_SAFETY_MARGIN_GB,
    FitEstimate,
    FitVerdict,
    ModelGeometry,
    estimate_fit,
    estimate_kv_cache_gib,
    estimate_weights_gib,
    max_context_for_budget,
)
from eigencore.fit.gguf_meta import GGUFModelInfo, read_gguf_model_info

__all__ = [
    "BYTES_PER_PARAM",
    "DEFAULT_RUNTIME_RESERVE_GB",
    "DEFAULT_SAFETY_MARGIN_GB",
    "GGUFModelInfo",
    "FitEstimate",
    "FitVerdict",
    "ModelGeometry",
    "estimate_fit",
    "estimate_kv_cache_gib",
    "estimate_weights_gib",
    "max_context_for_budget",
    "read_gguf_model_info",
]
