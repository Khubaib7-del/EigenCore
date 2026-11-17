"""
Model-fit estimation — Phase 2 (honest hardware fit).

Estimates the memory needed to *load and run* a model, separating:
  - weights        (actual GGUF file size when known, else params × bytes-per-param)
  - runtime reserve (allocator/backend/OS overhead, configurable)
  - KV cache       (context-length dependent; needs per-model geometry)

All sizes are **GiB** (1 GiB = 1024**3 bytes) end to end. Without model
geometry the KV cache is unknowable, so the verdict is `UNKNOWN` rather than
a made-up heuristic — a Core i7 label alone cannot predict whether a 35B Q4
model is usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Approximate bytes per parameter for common GGUF quant types.
# These are planning estimates, not measurements.
BYTES_PER_PARAM: dict[str, float] = {
    "Q4_K_M": 0.55,
    "Q4_0": 0.50,
    "Q5_K_M": 0.68,
    "Q6_K": 0.84,
    "Q8_0": 1.05,
    "F16": 2.0,
    "F32": 4.0,
}

# KV cache per token, per layer, per KV head, in bytes at fp16.
# kv_bytes_per_token = layers × kv_heads × (key_dim + value_dim) × 2 bytes
# key_dim/value_dim come from GGUF {arch}.attention.key_length/value_length;
# only when those are absent is head_dim = hidden_size / n_heads (Qwen3 and
# Gemma use a head_dim that differs from hidden/heads).
# For grouped-query attention kv_heads < n_heads.
FP16_BYTES_PER_ELEMENT = 2
KV_BYTES_PER_ELEMENT_Q8 = 1  # q8_0 KV cache

DEFAULT_RUNTIME_RESERVE_GB = 2.0
DEFAULT_SAFETY_MARGIN_GB = 1.0


class FitVerdict(str, Enum):
    FITS_WITH_MARGIN = "fits_with_margin"
    TIGHT = "tight"
    DOES_NOT_FIT = "does_not_fit"
    UNKNOWN = "unknown"


@dataclass
class ModelGeometry:
    """Architecture metadata needed for a KV-cache estimate.

    All fields optional: when absent the estimator falls back to a
    conservative generic estimate and marks the result as UNKNOWN-precision.
    """

    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None  # GQA/MQA: defaults to num_attention_heads
    expert_count: int | None = None  # MoE: total experts per MoE layer
    expert_used_count: int | None = None  # MoE: experts activated per token
    key_length: int | None = None  # GGUF {arch}.attention.key_length (per-head K dim)
    value_length: int | None = None  # GGUF {arch}.attention.value_length (per-head V dim)

    @property
    def kv_heads(self) -> int | None:
        return self.num_kv_heads or self.num_attention_heads

    @property
    def head_dim(self) -> int | None:
        """Per-head K dimension: GGUF key_length, else hidden_size / n_heads."""
        if self.key_length:
            return self.key_length
        if self.hidden_size and self.num_attention_heads:
            return self.hidden_size // self.num_attention_heads
        return None

    @property
    def head_dim_v(self) -> int | None:
        """Per-head V dimension: GGUF value_length, else the K head dim."""
        return self.value_length or self.head_dim

    @property
    def is_complete(self) -> bool:
        return all(
            v is not None for v in (self.num_layers, self.hidden_size, self.num_attention_heads)
        )

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_count and self.expert_used_count)


def kv_bytes_per_token(geometry: ModelGeometry, kv_quant: str = "fp16") -> int:
    """KV cache bytes per token: layers × kv_heads × (K dim + V dim) × bytes/element."""
    bytes_per_elem = (
        FP16_BYTES_PER_ELEMENT if kv_quant.lower() in ("fp16", "f16") else KV_BYTES_PER_ELEMENT_Q8
    )
    return int(
        geometry.num_layers
        * geometry.kv_heads
        * (geometry.head_dim + geometry.head_dim_v)
        * bytes_per_elem
    )


@dataclass
class FitEstimate:
    weights_gb: float
    active_weights_gb: float | None  # MoE only: weights actually read per decode token
    runtime_reserve_gb: float
    kv_cache_gb: float
    total_gb: float
    available_ram_gb: float
    headroom_gb: float
    verdict: FitVerdict
    context_length: int
    quant: str
    resident: bool  # file fully in RAM (False → mmap/SSD paging)
    decode_ceiling_tok_s: float | None  # bandwidth / active bytes-per-token ceiling
    gpu_layers: int | None = None  # layers offloaded to GPU, if a VRAM budget was given
    gpu_weights_gb: float = 0.0  # share of weights resident in VRAM
    notes: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.verdict in (FitVerdict.FITS_WITH_MARGIN, FitVerdict.TIGHT)


def gpu_layers_for_vram(
    num_layers: int,
    layer_bytes: int,
    available_vram_gb: float,
    kv_cache_gb: float = 0.0,
    vram_reserve_gb: float = 1.0,
) -> int:
    """How many layers fit in VRAM: (vram - reserve - kv) / layer_bytes, capped at num_layers.

    layer_bytes is the per-layer weight size (uniform approximation; real
    per-layer sizes vary slightly). Returns 0 when nothing fits.
    """
    if num_layers <= 0 or layer_bytes <= 0 or available_vram_gb <= 0:
        return 0
    budget_bytes = max((available_vram_gb - vram_reserve_gb - kv_cache_gb) * 1024**3, 0.0)
    return min(num_layers, int(budget_bytes // layer_bytes))


def estimate_weights_gib(
    params_b: float,
    quant: str,
    file_size_bytes: int | None = None,
) -> tuple[float, str | None]:
    """Weight size in GiB.

    Uses the actual GGUF file size when available (ground truth, includes
    tokenizer/mmap overhead); otherwise params × bytes-per-param.
    """
    if file_size_bytes is not None:
        return file_size_bytes / 1024**3, None
    bpp = BYTES_PER_PARAM.get(quant.upper())
    if bpp is None:
        return params_b * 0.61, f"unknown quant {quant!r}; assumed ~0.61 bytes/param (Q4_K_M class)"
    return params_b * bpp, None


def estimate_kv_cache_gib(
    context_length: int,
    geometry: ModelGeometry | None,
    params_b: float,
    kv_quant: str = "fp16",
) -> tuple[float | None, str | None]:
    """KV cache size in GiB for a context length.

    layers × kv_heads × (K dim + V dim) × bytes-per-element × ctx.
    Returns (None, note) when geometry is incomplete — no heuristic.
    """
    if geometry is None or not geometry.is_complete:
        return None, "no GGUF geometry available; KV cache unknown"
    per_token = kv_bytes_per_token(geometry, kv_quant)
    return per_token * context_length / 1024**3, None


def is_expert_tensor(name: str) -> bool:
    """True for routed-expert tensors (`blk.N.ffn_{gate,up,down,gate_up}_exps.*`).

    Shared experts (`*_shexp`) and the router (`ffn_gate_inp`) run for every
    token, so they are not routed.
    """
    return any(part.endswith("_exps") for part in name.split("."))


def moe_active_weight_bytes(
    total_tensor_bytes: int,
    expert_tensor_bytes: int,
    expert_used_count: int,
    expert_count: int,
) -> int:
    """Weight bytes read per decode token for an MoE model.

    Every never-routed tensor (attention, embeddings, norms, router, shared
    experts, dense layers) counts in full; routed-expert tensors count by the
    fraction of experts used per token.
    """
    dense = total_tensor_bytes - expert_tensor_bytes
    return int(dense + expert_tensor_bytes * expert_used_count / expert_count)


def decode_ceiling_tok_s(bandwidth_gbps: float, active_weights_gib: float) -> float:
    """Optimistic decode ceiling: bandwidth / bytes streamed per token.

    For dense models active == resident weights. For MoE, only the activated
    experts' weights are read per token (plus shared/non-expert weights, which
    we approximate as included in the active figure). This is a CEILING —
    real decode is lower due to cache misses, dequant, sampling overhead.
    """
    if active_weights_gib <= 0:
        return 0.0
    return bandwidth_gbps / active_weights_gib


def estimate_fit(
    params_b: float,
    quant: str,
    context_length: int,
    available_ram_gb: float,
    geometry: ModelGeometry | None = None,
    runtime_reserve_gb: float = DEFAULT_RUNTIME_RESERVE_GB,
    safety_margin_gb: float = DEFAULT_SAFETY_MARGIN_GB,
    kv_quant: str = "fp16",
    file_size_bytes: int | None = None,
    bandwidth_gbps: float | None = None,
    active_params_b: float | None = None,
    available_vram_gb: float = 0.0,
    active_weight_bytes: int | None = None,
) -> FitEstimate:
    """Full memory fit estimate (GiB) for loading + running a model at a context length.

    MoE: pass `active_weight_bytes` (from GGUF tensor sizes, see
    `moe_active_weight_bytes`) or an explicit `active_params_b` — resident
    memory still uses the full file size, but the decode ceiling uses active
    weights. Without either, the ceiling uses the full weights.
    """
    notes: list[str] = []
    weights_gb, note = estimate_weights_gib(params_b, quant, file_size_bytes)
    if note:
        notes.append(note)
    active_weights_gb: float | None = None
    if active_weight_bytes is not None:
        active_weights_gb = active_weight_bytes / 1024**3
        if geometry is not None and geometry.is_moe:
            notes.append(
                f"MoE: {geometry.expert_count} experts, {geometry.expert_used_count} active/token "
                f"→ ≈ {active_weights_gb:.1f} GiB of weights read per token (GGUF tensor sizes)"
            )
    elif active_params_b is not None and active_params_b < params_b:
        active_weights_gb, _ = estimate_weights_gib(active_params_b, quant)
    elif geometry is not None and geometry.is_moe:
        notes.append(
            "MoE: active weights unknown without GGUF tensor sizes; "
            "decode ceiling uses the full weights"
        )
    kv_gb_raw, note = estimate_kv_cache_gib(context_length, geometry, params_b, kv_quant)
    if note:
        notes.append(note)
    kv_gb = kv_gb_raw or 0.0

    total = weights_gb + runtime_reserve_gb + kv_gb
    headroom = available_ram_gb - total
    resident = weights_gb <= available_ram_gb

    if kv_gb_raw is None:
        verdict = FitVerdict.UNKNOWN
        notes.append("verdict UNKNOWN: KV cache excluded (no geometry); real usage is higher")
    elif headroom >= safety_margin_gb:
        verdict = FitVerdict.FITS_WITH_MARGIN
    elif headroom >= 0:
        verdict = FitVerdict.TIGHT
        notes.append(f"only {headroom:.1f} GB headroom; expect swapping/slowdown under load")
    else:
        verdict = FitVerdict.DOES_NOT_FIT
        notes.append(f"short by {-headroom:.1f} GB at context {context_length}")

    if not resident:
        notes.append(
            f"model file ({weights_gb:.1f} GiB) exceeds available RAM ({available_ram_gb:.1f} GiB): "
            "would run via mmap/SSD paging — expect severe decode slowdown (SSD read rate, not RAM bandwidth)"
        )

    ceiling = None
    if bandwidth_gbps is not None:
        active_gib = active_weights_gb if active_weights_gb is not None else weights_gb
        ceiling = decode_ceiling_tok_s(bandwidth_gbps, active_gib)

    gpu_layers = None
    gpu_weights_gb = 0.0
    if available_vram_gb > 0 and geometry is not None and geometry.num_layers:
        layer_bytes = int(weights_gb * 1024**3 / geometry.num_layers) if weights_gb > 0 else 0
        gpu_layers = gpu_layers_for_vram(geometry.num_layers, layer_bytes, available_vram_gb, kv_gb)
        gpu_weights_gb = gpu_layers * layer_bytes / 1024**3
        if gpu_layers > 0:
            notes.append(
                f"GPU offload: ~{gpu_layers}/{geometry.num_layers} layers "
                f"(≈{gpu_weights_gb:.1f} GiB) fit in {available_vram_gb:.1f} GiB VRAM; "
                "rest stays in RAM"
            )
        else:
            notes.append("VRAM present but too small for even one layer after reserve/KV")

    return FitEstimate(
        weights_gb=weights_gb,
        active_weights_gb=active_weights_gb,
        runtime_reserve_gb=runtime_reserve_gb,
        kv_cache_gb=kv_gb,
        total_gb=total,
        available_ram_gb=available_ram_gb,
        headroom_gb=headroom,
        verdict=verdict,
        context_length=context_length,
        quant=quant,
        resident=resident,
        decode_ceiling_tok_s=ceiling,
        gpu_layers=gpu_layers,
        gpu_weights_gb=gpu_weights_gb,
        notes=notes,
    )


def max_context_for_budget(
    params_b: float,
    quant: str,
    available_ram_gb: float,
    geometry: ModelGeometry | None = None,
    runtime_reserve_gb: float = DEFAULT_RUNTIME_RESERVE_GB,
    safety_margin_gb: float = DEFAULT_SAFETY_MARGIN_GB,
    model_context_limit: int = 128 * 1024,
    kv_quant: str = "fp16",
    file_size_bytes: int | None = None,
) -> int:
    """Largest context length that fits the RAM budget (power-of-two capped).

    Returns 0 when geometry is unknown — without per-token KV bytes a context
    recommendation would be a guess.
    """
    if geometry is None or not geometry.is_complete:
        return 0
    weights_gb, _ = estimate_weights_gib(params_b, quant, file_size_bytes)
    budget_gb = available_ram_gb - runtime_reserve_gb - safety_margin_gb - weights_gb
    if budget_gb <= 0:
        return 0
    per_token = kv_bytes_per_token(geometry, kv_quant)
    if per_token <= 0:
        return 0
    max_ctx = int(budget_gb * 1024**3 / per_token)
    # snap down to a power of two, clamp to model limit and a sane floor
    ctx = 1
    while ctx * 2 <= max_ctx and ctx * 2 <= model_context_limit:
        ctx *= 2
    return max(ctx, 0)
