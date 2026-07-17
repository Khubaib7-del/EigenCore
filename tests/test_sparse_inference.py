"""Tests for sparsity-aware inference engine."""

import numpy as np
import pytest

from eigencore.engine.sparse_inference import (
    SparseExecutionPlan,
    SparsityCache,
    SparsityPredictor,
)


class TestSparsityPredictor:
    def test_not_warmed_up_returns_full_mask(self):
        pred = SparsityPredictor(num_layers=4, neurons_per_layer=100, warmup_samples=16)
        mask = pred.predict_mask(0)
        assert mask.active_count == 100
        assert mask.sparsity == 0.0

    def test_warmup_threshold(self):
        pred = SparsityPredictor(num_layers=2, neurons_per_layer=50, warmup_samples=5)
        for _ in range(5):
            pred.observe(0, np.random.randn(50))
        assert pred.is_warmed_up

    def test_observe_returns_sparsity(self):
        pred = SparsityPredictor(num_layers=2, neurons_per_layer=100, threshold=0.5)
        # all zeros → 100% sparsity
        sparsity = pred.observe(0, np.zeros(100))
        assert sparsity == 1.0

    def test_observe_all_active(self):
        pred = SparsityPredictor(num_layers=2, neurons_per_layer=100, threshold=0.01)
        sparsity = pred.observe(0, np.ones(100) * 5.0)
        assert sparsity == 0.0

    def test_high_sparsity_creates_sparse_mask(self):
        pred = SparsityPredictor(num_layers=2, neurons_per_layer=100, warmup_samples=10)
        sparse_acts = np.zeros(100)
        sparse_acts[:10] = 1.0  # only 10% active
        for _ in range(20):
            pred.observe(0, sparse_acts)
        mask = pred.predict_mask(0, aggressiveness=0.5)
        assert mask.active_count < 100
        assert mask.sparsity > 0.0

    def test_invalid_layer_raises(self):
        pred = SparsityPredictor(num_layers=2, neurons_per_layer=50)
        with pytest.raises(ValueError, match="Layer 5"):
            pred.observe(5, np.zeros(50))

    def test_execution_plan_covers_all_layers(self):
        pred = SparsityPredictor(num_layers=4, neurons_per_layer=64, warmup_samples=5)
        for _ in range(10):
            for layer in range(4):
                acts = np.random.randn(64) * 0.001  # very sparse
                pred.observe(layer, acts)
        plan = pred.create_execution_plan()
        assert len(plan.masks) == 4
        assert plan.total_neurons == 4 * 64

    def test_layer_stats_format(self):
        pred = SparsityPredictor(num_layers=3, neurons_per_layer=32)
        pred.observe(0, np.zeros(32))
        pred.observe(1, np.ones(32))
        stats = pred.layer_stats()
        assert len(stats) == 3
        assert stats[0]["avg_sparsity"] == 1.0
        assert stats[1]["avg_sparsity"] == 0.0


class TestSparseExecutionPlan:
    def test_plan_summary(self):
        pred = SparsityPredictor(num_layers=2, neurons_per_layer=100, warmup_samples=5)
        for _ in range(10):
            pred.observe(0, np.zeros(100))
            pred.observe(1, np.ones(100))
        plan = pred.create_execution_plan()
        summary = plan.summary()
        assert "sparsity" in summary
        assert "speedup" in summary

    def test_speedup_capped(self):
        pred = SparsityPredictor(num_layers=2, neurons_per_layer=100, warmup_samples=5)
        for _ in range(10):
            pred.observe(0, np.zeros(100))
            pred.observe(1, np.zeros(100))
        plan = pred.create_execution_plan(aggressiveness=1.0)
        assert plan.estimated_speedup <= 5.0


class TestSparsityCache:
    def test_cache_miss_returns_none(self):
        cache = SparsityCache(max_entries=10)
        assert cache.get([1, 2, 3]) is None
        assert cache.misses == 1

    def test_cache_hit_after_put(self):
        cache = SparsityCache(max_entries=10)
        plan = SparseExecutionPlan(
            masks=[],
            total_neurons=100,
            total_active=50,
            overall_sparsity=0.5,
            estimated_flop_savings=0.5,
            estimated_speedup=2.0,
        )
        cache.put([1, 2, 3], plan)
        result = cache.get([1, 2, 3])
        assert result is not None
        assert result.overall_sparsity == 0.5
        assert cache.hits == 1

    def test_lru_eviction(self):
        cache = SparsityCache(max_entries=3)
        dummy = SparseExecutionPlan(
            masks=[],
            total_neurons=0,
            total_active=0,
            overall_sparsity=0.0,
            estimated_flop_savings=0.0,
            estimated_speedup=1.0,
        )
        cache.put([1], dummy)
        cache.put([2], dummy)
        cache.put([3], dummy)
        cache.put([4], dummy)  # evicts [1]
        assert cache.get([1]) is None
        assert cache.get([4]) is not None

    def test_hit_rate(self):
        cache = SparsityCache()
        dummy = SparseExecutionPlan(
            masks=[],
            total_neurons=0,
            total_active=0,
            overall_sparsity=0.0,
            estimated_flop_savings=0.0,
            estimated_speedup=1.0,
        )
        cache.put([1], dummy)
        cache.get([1])  # hit
        cache.get([2])  # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_clear_resets(self):
        cache = SparsityCache()
        dummy = SparseExecutionPlan(
            masks=[],
            total_neurons=0,
            total_active=0,
            overall_sparsity=0.0,
            estimated_flop_savings=0.0,
            estimated_speedup=1.0,
        )
        cache.put([1], dummy)
        cache.get([1])
        cache.clear()
        assert cache.get([1]) is None
        assert cache.hits == 0
        assert cache.misses == 1
