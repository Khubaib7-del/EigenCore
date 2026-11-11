"""Stage E slice 3: activation-sparsity experiment (mask plan + kill criterion)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.engine.sparsity_experiment import (
    activation_sparsity,
    build_sparse_mask,
    build_sparsity_plan,
    layer_sparsity_profile,
)


def test_activation_sparsity_basic():
    a = np.array([[0.0, 0.001, 1.0, -2.0], [0.0, 0.0, 0.5, 0.0]])
    # |x|<=0.01: 0, 0.001, 0, 0, 0 → five of eight → 0.625
    assert abs(activation_sparsity(a, threshold=0.01) - 0.625) < 1e-12
    assert activation_sparsity(np.zeros((0, 4))) == 0.0


def test_layer_profile_shape():
    rng = np.random.default_rng(0)
    hs = rng.standard_normal((6, 8, 16))
    hs[:, :, :8] = 0.0  # half the neurons always zero → sparsity ≈ 0.5
    profile = layer_sparsity_profile(hs, threshold=0.01)
    assert profile.shape == (6,)
    assert np.all((profile >= 0.5) & (profile < 0.52))  # 0.5 + tiny normal tail


def test_sparse_mask_keeps_above_threshold():
    a = np.array([[0.0, 0.5], [0.2, 0.0]])
    mask = build_sparse_mask(a, threshold=0.01)
    assert mask.keep.tolist() == [True, True]  # mean |a| per neuron: [0.1, 0.25] both > 0.01
    assert abs(mask.sparsity) < 1e-12


def test_plan_applies_only_above_min_sparsity():
    rng = np.random.default_rng(1)
    hs = rng.standard_normal((4, 8, 32))
    hs[0] = 0.0  # layer 0 fully sparse (1.0)
    plan = build_sparsity_plan(hs, threshold=0.01, min_sparsity_to_apply=0.5)
    assert plan.masks[0].sparsity == 1.0  # all skipped
    assert plan.masks[1].sparsity == 0.0  # dense layer → full keep
    assert not plan.killed
    assert plan.overall_sparsity > 0


def test_kill_criterion_aborts():
    rng = np.random.default_rng(2)
    hs = rng.standard_normal((3, 4, 8))
    hs[:] = 0.0
    plan = build_sparsity_plan(
        hs, threshold=0.01, max_quality_delta_allowed=0.02, measured_quality_delta=0.1
    )
    assert plan.killed and plan.masks == []
    assert "kill criterion" in plan.kill_reason
