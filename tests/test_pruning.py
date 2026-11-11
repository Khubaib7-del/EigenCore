"""Stage E slice 2: ShortGPT BI scoring, prune plan, kill criterion."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.engine.pruning import block_influence, build_prune_plan, layer_bi_scores


def test_block_influence_identical_vectors_is_zero():
    x = np.array([1.0, 2.0, 3.0])
    assert abs(block_influence(x, x)) < 1e-12


def test_block_influence_orthogonal_is_one():
    assert abs(block_influence([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-12


def test_layer_bi_scores_shape_and_range():
    rng = np.random.default_rng(0)
    layers, samples, hidden = 10, 5, 32
    hs = rng.standard_normal((layers + 1, samples, hidden))
    scores = layer_bi_scores(hs)
    assert scores.shape == (layers,)
    assert np.all(scores >= 0.0) and np.all(scores <= 2.0)


def test_layer_bi_scores_detects_redundant_layer():
    # layer 5 passes its input through unchanged → BI ≈ 0 (lowest)
    rng = np.random.default_rng(1)
    layers, samples, hidden = 8, 4, 16
    hs = rng.standard_normal((layers + 1, samples, hidden))
    hs[6] = hs[5]  # layer 5 output == its input
    scores = layer_bi_scores(hs)
    assert int(np.argmin(scores)) == 5


def test_prune_plan_respects_head_tail_protection():
    rng = np.random.default_rng(2)
    scores = rng.random(32)
    plan = build_prune_plan(scores, num_layers=32, max_prune_fraction=0.3)
    assert len(plan.layers_to_prune) <= 9
    assert all(4 <= i < 28 for i in plan.layers_to_prune)  # 15% of 32 = 4 head, 4 tail
    assert not plan.killed
    # lowest-BI layers chosen
    chosen = set(plan.layers_to_prune)
    assert chosen == set(sorted(range(4, 28), key=lambda i: scores[i])[: len(chosen)])


def test_kill_criterion_aborts_pruning():
    scores = np.random.default_rng(3).random(32)
    plan = build_prune_plan(
        scores, num_layers=32, max_quality_delta_allowed=0.02, measured_quality_delta=0.05
    )
    assert plan.killed and plan.layers_to_prune == []
    assert "kill criterion" in plan.kill_reason


def test_no_measurement_means_not_killed_but_reported_only():
    scores = np.random.default_rng(4).random(16)
    plan = build_prune_plan(scores, num_layers=16)
    assert not plan.killed
