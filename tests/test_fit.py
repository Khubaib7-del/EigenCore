"""Tests for the Phase 2 model-fit estimator (GiB end to end, GGUF-aware)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.fit import (
    FitVerdict,
    ModelGeometry,
    decode_ceiling_tok_s,
    estimate_fit,
    estimate_kv_cache_gib,
    estimate_weights_gib,
    max_context_for_budget,
)

GEO = ModelGeometry(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=32)


def test_weights_known_quants():
    assert estimate_weights_gib(35.0, "Q4_K_M")[0] == 35.0 * 0.55
    assert estimate_weights_gib(7.0, "F16")[0] == 14.0


def test_weights_prefer_file_size():
    # actual GGUF file size is ground truth
    gib, note = estimate_weights_gib(35.0, "Q4_K_M", file_size_bytes=21 * 1024**3)
    assert note is None and abs(gib - 21.0) < 1e-9


def test_weights_unknown_quant_flagged():
    gib, note = estimate_weights_gib(10.0, "IQ2_XXS")
    assert gib > 0 and note is not None and "unknown quant" in note


def test_kv_cache_with_geometry():
    # 32 layers, 32 kv heads, head_dim 128, fp16 → per token: 32*32*128*2*2 = 524288 B
    gib, note = estimate_kv_cache_gib(8192, GEO, params_b=32.0)
    assert note is None
    expected = 524288 * 8192 / 1024**3
    assert abs(gib - expected) < 1e-9


def test_kv_cache_unknown_without_geometry():
    # no heuristic anymore: unknown geometry → None + note
    gib, note = estimate_kv_cache_gib(4096, None, params_b=35.0)
    assert gib is None and note is not None and "geometry" in note
    partial = ModelGeometry(num_layers=32)
    gib, note = estimate_kv_cache_gib(4096, partial, params_b=35.0)
    assert gib is None and note is not None


def test_fit_verdicts():
    # 7B Q4 (~3.85 GiB weights) on 16 GiB with ctx 4096 → fits with margin
    est = estimate_fit(7.0, "Q4_K_M", 4096, 16.0, geometry=GEO)
    assert est.verdict == FitVerdict.FITS_WITH_MARGIN and est.fits
    assert est.headroom_gb == 16.0 - est.total_gb

    # 35B Q4 (~19.25 GiB weights) on 16 GiB → does not fit
    est = estimate_fit(35.0, "Q4_K_M", 4096, 16.0, geometry=GEO)
    assert est.verdict == FitVerdict.DOES_NOT_FIT and not est.fits
    assert est.headroom_gb < 0

    # boundary: total ≈ 3.85 weights + 2 reserve + 2.0 KV = 7.85 GiB,
    # so 8.5 GiB available leaves 0.65 GiB headroom → TIGHT
    est = estimate_fit(7.0, "Q4_K_M", 4096, 8.5, geometry=GEO, safety_margin_gb=1.0)
    assert est.verdict == FitVerdict.TIGHT
    assert 0 < est.headroom_gb < 1.0


def test_fit_unknown_verdict_without_geometry():
    est = estimate_fit(7.0, "Q4_K_M", 4096, 16.0)  # no geometry
    assert est.verdict == FitVerdict.UNKNOWN
    assert not est.fits
    assert any("UNKNOWN" in n for n in est.notes)


def test_fit_context_is_first_class():
    short = estimate_fit(35.0, "Q4_K_M", 2048, 24.0, geometry=GEO)
    long = estimate_fit(35.0, "Q4_K_M", 32768, 24.0, geometry=GEO)
    assert long.kv_cache_gb > short.kv_cache_gb
    assert long.total_gb > short.total_gb


def test_max_context_for_budget_geometry():
    # weights 7B Q4 ≈ 3.85 GiB; 16 GiB available, 2 reserve + 1 margin → ~11.15 GiB for KV
    ctx = max_context_for_budget(7.0, "Q4_K_M", 16.0, geometry=GEO)
    assert ctx > 0 and ctx & (ctx - 1) == 0  # power of two
    est = estimate_fit(7.0, "Q4_K_M", ctx, 16.0, geometry=GEO)
    assert est.verdict in (FitVerdict.FITS_WITH_MARGIN, FitVerdict.TIGHT)


def test_max_context_zero_without_geometry():
    # no geometry → no context recommendation (was a heuristic before)
    assert max_context_for_budget(7.0, "Q4_K_M", 16.0) == 0


def test_max_context_zero_when_model_too_big():
    assert max_context_for_budget(70.0, "Q4_K_M", 8.0, geometry=GEO) == 0


def test_max_context_respects_model_limit():
    ctx = max_context_for_budget(1.0, "Q4_K_M", 64.0, geometry=GEO, model_context_limit=4096)
    assert ctx <= 4096


def test_max_context_uses_file_size():
    # 21 GiB file on 16 GiB RAM → nothing fits
    assert (
        max_context_for_budget(35.0, "Q4_K_M", 16.0, geometry=GEO, file_size_bytes=21 * 1024**3)
        == 0
    )


MOE_GEO = ModelGeometry(
    num_layers=48,
    hidden_size=2048,
    num_attention_heads=32,
    num_kv_heads=4,
    expert_count=128,
    expert_used_count=8,
)


def test_moe_geometry_flag():
    assert MOE_GEO.is_moe
    assert not GEO.is_moe


def test_moe_fit_active_weights_and_ceiling():
    # 30.5B total, 8/128 experts → 1.9125B active; file 18.6 GiB; 16 GiB RAM; 40 GB/s
    est = estimate_fit(
        30.5,
        "Q4_K_M",
        4096,
        16.0,
        geometry=MOE_GEO,
        file_size_bytes=int(18.6 * 1024**3),
        bandwidth_gbps=40.0,
    )
    assert est.active_weights_gb is not None
    expected_active = 30.5 * 8 / 128 * 0.55
    assert abs(est.active_weights_gb - expected_active) < 1e-6
    # file (18.6 GiB) > RAM (16 GiB) → not resident, paging warning present
    assert est.resident is False
    assert any("SSD paging" in n for n in est.notes)
    # ceiling uses ACTIVE weights, not the resident file
    assert est.decode_ceiling_tok_s == 40.0 / est.active_weights_gb
    assert est.decode_ceiling_tok_s > 10


def test_dense_ceiling_uses_full_weights():
    est = estimate_fit(7.0, "Q4_K_M", 4096, 16.0, geometry=GEO, bandwidth_gbps=30.0)
    assert est.active_weights_gb is None
    assert est.decode_ceiling_tok_s == 30.0 / est.weights_gb


def test_decode_ceiling_helper():
    assert decode_ceiling_tok_s(40.0, 2.0) == 20.0
    assert decode_ceiling_tok_s(40.0, 0.0) == 0.0
