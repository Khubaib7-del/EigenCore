"""
Layer-pruning experiment — Stage E slice 2 (ShortGPT-style), REPORTED-ONLY.

ShortGPT (Men et al., 2024) scores each transformer layer by Block Influence:
the 1 - cosine similarity between the layer's input and output hidden states,
averaged over calibration samples. Low-BI layers are candidates for removal.

This module computes BI scores, builds a prune plan, and applies the KILL
CRITERION from the experiment plan: if the measured quality delta (e.g.
perplexity increase) exceeds `max_quality_delta`, the experiment is aborted
and NO layers are pruned. Nothing here is wired into inference; the backend
always runs all layers today.

No performance or quality claim is made without the owner's calibration +
perplexity runs (FINAL_REPORT §9 item 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def block_influence(layer_in: np.ndarray, layer_out: np.ndarray, eps: float = 1e-8) -> float:
    """BI for one layer on one sample: 1 - cos(x_in, x_out).

    `layer_in`/`layer_out`: (hidden,) vectors. Returns 0 when either is ~zero.
    """
    x = np.asarray(layer_in, dtype=np.float64).reshape(-1)
    y = np.asarray(layer_out, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("layer_in/layer_out must be same-shape non-empty vectors")
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < eps or ny < eps:
        return 0.0
    cos = float(np.dot(x, y) / (nx * ny))
    return 1.0 - max(-1.0, min(1.0, cos))


def layer_bi_scores(
    hidden_states: np.ndarray,
) -> np.ndarray:
    """Mean BI per layer over samples.

    `hidden_states`: (num_layers + 1, num_samples, hidden) — activations at each
    layer boundary (0 = model input, L = model output). Layer i's BI uses
    boundary i (input) and i+1 (output). Returns (num_layers,) float64.
    """
    hs = np.asarray(hidden_states, dtype=np.float64)
    if hs.ndim != 3 or hs.shape[0] < 2:
        raise ValueError("expected (layers+1, samples, hidden)")
    num_layers = hs.shape[0] - 1
    scores = np.zeros(num_layers)
    for i in range(num_layers):
        x = hs[i]  # (samples, hidden)
        y = hs[i + 1]
        nx = np.linalg.norm(x, axis=1)
        ny = np.linalg.norm(y, axis=1)
        denom = nx * ny
        cos = np.where(denom > 1e-16, (x * y).sum(axis=1) / np.maximum(denom, 1e-16), 0.0)
        cos = np.clip(cos, -1.0, 1.0)
        scores[i] = float(np.mean(1.0 - cos))
    return scores


@dataclass
class PrunePlan:
    layers_to_prune: list[int]
    bi_scores: np.ndarray
    max_quality_delta_allowed: float
    killed: bool = False
    kill_reason: str = ""
    notes: list[str] = field(default_factory=list)


def build_prune_plan(
    bi_scores: np.ndarray,
    num_layers: int,
    max_prune_fraction: float = 0.3,
    protect_head: float = 0.15,
    protect_tail: float = 0.15,
    max_quality_delta_allowed: float = 0.02,
    measured_quality_delta: float | None = None,
) -> PrunePlan:
    """Rank layers by BI ascending and pick the lowest-BI middle layers.

    First/last `protect_*` fraction of layers are never pruned (embedding/head
    criticality). Kill criterion: if `measured_quality_delta` is provided and
    exceeds `max_quality_delta_allowed`, the plan is killed (no pruning).
    """
    scores = np.asarray(bi_scores, dtype=np.float64)
    if scores.shape[0] != num_layers:
        raise ValueError("bi_scores length must equal num_layers")
    if measured_quality_delta is not None and measured_quality_delta > max_quality_delta_allowed:
        return PrunePlan(
            layers_to_prune=[],
            bi_scores=scores,
            max_quality_delta_allowed=max_quality_delta_allowed,
            killed=True,
            kill_reason=(
                f"quality delta {measured_quality_delta:.4f} > allowed "
                f"{max_quality_delta_allowed:.4f} — kill criterion hit, no pruning"
            ),
        )
    n_protect_head = int(num_layers * protect_head)
    n_protect_tail = int(num_layers * protect_tail)
    candidates = list(range(n_protect_head, num_layers - n_protect_tail))
    max_prune = int(num_layers * max_prune_fraction)
    candidates.sort(key=lambda i: scores[i])
    return PrunePlan(
        layers_to_prune=sorted(candidates[:max_prune]),
        bi_scores=scores,
        max_quality_delta_allowed=max_quality_delta_allowed,
    )
