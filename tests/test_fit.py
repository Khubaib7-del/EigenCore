"""Tests for the Phase 2 model-fit estimator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.fit import (
    FitVerdict,
    ModelGeometry,
    estimate_fit,
    estimate_kv_cache_gb,
    estimate_weights_gb,
    max_context_for_budget,
)


def test_weights_known_quants():
    assert estimate_weights_gb(35.0, "Q4_K_M")[0] == 35.0 * 0.55
    assert estimate_weights_gb(7.0, "F16")[0] == 14.0


def test_weights_unknown_quant_flagged():
    gb, note = estimate_weights_gb(10.0, "IQ2_XXS")
    assert gb > 0 and note is not None and "unknown quant" in note


def test_kv_cache_with_geometry():
    # 32 layers, 32 kv heads, head_dim 128, fp16 → per token: 32*32*128*2*2 = 524288 B
    geo = ModelGeometry(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=32)
    gb, note = estimate_kv_cache_gb(8192, geo, params_b=32.0)
    assert note is None
    expected = 524288 * 8192 / 1024**3
    assert abs(gb - expected) < 1e-9


def test_kv_cache_heuristic_without_geometry():
    gb, note = estimate_kv_cache_gb(4096, None, params_b=35.0)
    assert note is not None and "heuristic" in note
    assert gb > 0


def test_fit_verdicts():
    # 7B Q4 (~3.85 GB weights) on 16 GB with ctx 4096 → fits with margin
    est = estimate_fit(7.0, "Q4_K_M", 4096, 16.0)
    assert est.verdict == FitVerdict.FITS_WITH_MARGIN and est.fits
    assert est.headroom_gb == 16.0 - est.total_gb

    # 35B Q4 (~19.25 GB weights) on 16 GB → does not fit
    est = estimate_fit(35.0, "Q4_K_M", 4096, 16.0)
    assert est.verdict == FitVerdict.DOES_NOT_FIT and not est.fits
    assert est.headroom_gb < 0

    # boundary: total is ~5.86 GB, so 6.5 GB available leaves 0.64 GB headroom
    # (positive but below the 1 GB safety margin) → TIGHT
    est = estimate_fit(7.0, "Q4_K_M", 4096, 6.5, safety_margin_gb=1.0)
    assert est.verdict == FitVerdict.TIGHT
    assert 0 < est.headroom_gb < 1.0


def test_fit_context_is_first_class():
    # same model, longer context costs more KV and can flip the verdict
    short = estimate_fit(35.0, "Q4_K_M", 2048, 24.0)
    long = estimate_fit(35.0, "Q4_K_M", 32768, 24.0)
    assert long.kv_cache_gb > short.kv_cache_gb
    assert long.total_gb > short.total_gb


def test_max_context_for_budget_geometry():
    geo = ModelGeometry(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=32)
    # weights 7B Q4 ≈ 3.85 GB; 16 GB available, 2 reserve + 1 margin → ~11.15 GB for KV
    ctx = max_context_for_budget(7.0, "Q4_K_M", 16.0, geometry=geo)
    assert ctx > 0 and ctx & (ctx - 1) == 0  # power of two
    # fitting at ctx must not exceed budget
    est = estimate_fit(7.0, "Q4_K_M", ctx, 16.0, geometry=geo)
    assert est.verdict in (FitVerdict.FITS_WITH_MARGIN, FitVerdict.TIGHT)


def test_max_context_zero_when_model_too_big():
    assert max_context_for_budget(70.0, "Q4_K_M", 8.0) == 0


def test_max_context_respects_model_limit():
    geo = ModelGeometry(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=32)
    ctx = max_context_for_budget(1.0, "Q4_K_M", 64.0, geometry=geo, model_context_limit=4096)
    assert ctx <= 4096
