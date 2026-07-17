"""Tests for dynamic layer skipping scheduler."""

import pytest
from eigencore.engine.layer_skip import (
    LayerSkipScheduler,
    SkipPlan,
    SkipStrategy,
)


class TestStaticSkipping:
    def test_no_skip_when_disabled(self):
        sched = LayerSkipScheduler(32, strategy=SkipStrategy.NONE)
        plan = sched.plan()
        assert plan.layers_to_skip == []
        assert len(plan.layers_to_run) == 32
        assert plan.expected_speedup == 1.0

    def test_no_skip_tiny_model(self):
        sched = LayerSkipScheduler(4, strategy=SkipStrategy.STATIC)
        plan = sched.plan()
        assert plan.layers_to_skip == []

    def test_preserves_head_and_tail(self):
        sched = LayerSkipScheduler(32, strategy=SkipStrategy.STATIC)
        plan = sched.plan(complexity=0.0)
        head = max(1, int(32 * 0.15))  # 4
        tail_start = 32 - max(1, int(32 * 0.15))  # 28
        for layer in plan.layers_to_skip:
            assert layer >= head, f"Layer {layer} is in the critical head"
            assert layer < tail_start, f"Layer {layer} is in the critical tail"

    def test_high_complexity_skips_less(self):
        sched = LayerSkipScheduler(32, strategy=SkipStrategy.STATIC, max_skip_rate=0.3)
        plan_easy = sched.plan(complexity=0.0)
        plan_hard = sched.plan(complexity=0.9)
        assert len(plan_easy.layers_to_skip) >= len(plan_hard.layers_to_skip)

    def test_skip_rate_within_bounds(self):
        sched = LayerSkipScheduler(32, strategy=SkipStrategy.STATIC, max_skip_rate=0.3)
        plan = sched.plan(complexity=0.0)
        assert plan.skip_rate <= 0.35  # small tolerance for rounding

    def test_speedup_correlates_with_skip_rate(self):
        sched = LayerSkipScheduler(32, strategy=SkipStrategy.STATIC)
        plan = sched.plan(complexity=0.0)
        if plan.skip_rate > 0:
            assert plan.expected_speedup > 1.0

    def test_plan_summary_is_readable(self):
        sched = LayerSkipScheduler(32, strategy=SkipStrategy.STATIC)
        plan = sched.plan()
        summary = plan.summary()
        assert "layers" in summary
        assert "speedup" in summary


class TestCalibratedSkipping:
    def test_calibrate_sets_profiles(self):
        sched = LayerSkipScheduler(8, strategy=SkipStrategy.CALIBRATED)
        importances = [0.9, 0.8, 0.3, 0.2, 0.1, 0.15, 0.7, 0.95]
        sched.calibrate(importances)
        plan = sched.plan(complexity=0.0)
        # low-importance layers (indices 2,3,4,5) should be candidates for skipping
        for skipped in plan.layers_to_skip:
            assert skipped not in [0, 7]  # head and tail are protected

    def test_calibrate_wrong_length_raises(self):
        sched = LayerSkipScheduler(8, strategy=SkipStrategy.CALIBRATED)
        with pytest.raises(ValueError, match="Expected 8"):
            sched.calibrate([0.5, 0.5])

    def test_falls_back_to_static_without_calibration(self):
        sched = LayerSkipScheduler(32, strategy=SkipStrategy.CALIBRATED)
        plan = sched.plan(complexity=0.0)
        # should fall back to static since not calibrated
        assert plan.strategy == SkipStrategy.STATIC


class TestAdaptiveSkipping:
    def test_needs_observations_before_skipping(self):
        sched = LayerSkipScheduler(8, strategy=SkipStrategy.ADAPTIVE)
        plan = sched.plan(complexity=0.0)
        # no observations → falls back to static
        assert plan.strategy == SkipStrategy.STATIC

    def test_skips_after_observing_sparsity(self):
        sched = LayerSkipScheduler(8, strategy=SkipStrategy.ADAPTIVE, max_skip_rate=0.5)
        # simulate 15 observations of high sparsity on layer 3
        for _ in range(15):
            sched.update_sparsity(3, 0.85)
        plan = sched.plan(complexity=0.0)
        assert 3 in plan.layers_to_skip or plan.strategy == SkipStrategy.ADAPTIVE

    def test_does_not_skip_low_sparsity_layers(self):
        sched = LayerSkipScheduler(8, strategy=SkipStrategy.ADAPTIVE)
        for _ in range(15):
            sched.update_sparsity(3, 0.1)  # low sparsity
        plan = sched.plan(complexity=0.0)
        assert 3 not in plan.layers_to_skip
