"""Phase 2 model-fit estimation."""

from eigencore.fit.estimator import (
    BYTES_PER_PARAM,
    DEFAULT_RUNTIME_RESERVE_GB,
    DEFAULT_SAFETY_MARGIN_GB,
    FitEstimate,
    FitVerdict,
    ModelGeometry,
    estimate_fit,
    estimate_kv_cache_gb,
    estimate_weights_gb,
    max_context_for_budget,
)

__all__ = [
    "BYTES_PER_PARAM",
    "DEFAULT_RUNTIME_RESERVE_GB",
    "DEFAULT_SAFETY_MARGIN_GB",
    "FitEstimate",
    "FitVerdict",
    "ModelGeometry",
    "estimate_fit",
    "estimate_kv_cache_gb",
    "estimate_weights_gb",
    "max_context_for_budget",
]
