"""
Activation-sparsity experiment — Stage E slice 3 (PowerInfer/TurboSparse/TEAL-
style), REPORTED-ONLY.

Idea: if a fraction of FFN neurons are near-zero for a given token, skip their
contribution (sparse GEMM) to save compute. On bandwidth-bound CPU decode the
win is small and quant/dequant dominates — hence the kill criterion.

This module measures per-layer activation sparsity on real activations,
builds a sparse mask plan, and enforces the kill criterion: if the measured
quality delta exceeds the allowed threshold, the experiment is aborted and no
sparsity is applied. Nothing here is wired into inference; the backend always
runs dense today. No quality/performance claim without the owner's runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def activation_sparsity(activations: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of |x| <= threshold. `activations`: (tokens, hidden)."""
    a = np.abs(np.asarray(activations, dtype=np.float64))
    if a.size == 0:
        return 0.0
    return float((a <= threshold).mean())


def layer_sparsity_profile(
    hidden_states: np.ndarray,  # (layers, tokens, hidden) FFN inputs per layer
    threshold: float = 0.01,
) -> np.ndarray:
    """Per-layer sparsity, (layers,)."""
    hs = np.asarray(hidden_states, dtype=np.float64)
    if hs.ndim != 3:
        raise ValueError("expected (layers, tokens, hidden)")
    return np.array([activation_sparsity(hs[i], threshold) for i in range(hs.shape[0])])


@dataclass
class SparseMask:
    layer: int
    keep: np.ndarray  # bool mask over hidden dim; True = compute

    @property
    def sparsity(self) -> float:
        return 1.0 - float(self.keep.mean())


def build_sparse_mask(
    activations: np.ndarray,
    threshold: float = 0.01,
) -> SparseMask:
    """Keep neurons whose |activation| exceeds threshold, averaged over tokens."""
    a = np.abs(np.asarray(activations, dtype=np.float64))
    mean_abs = a.mean(axis=0) if a.ndim == 2 else np.abs(a)
    return SparseMask(layer=-1, keep=mean_abs > threshold)


@dataclass
class SparsityPlan:
    masks: list[SparseMask]
    overall_sparsity: float
    max_quality_delta_allowed: float
    killed: bool = False
    kill_reason: str = ""
    notes: list[str] = field(default_factory=list)


def build_sparsity_plan(
    hidden_states: np.ndarray,  # (layers, tokens, hidden)
    threshold: float = 0.01,
    min_sparsity_to_apply: float = 0.5,
    max_quality_delta_allowed: float = 0.02,
    measured_quality_delta: float | None = None,
) -> SparsityPlan:
    """Build per-layer sparse masks; kill if quality delta exceeds allowed.

    Layers whose measured sparsity is below `min_sparsity_to_apply` get a full
    keep mask (applying sparsity there saves nothing meaningful).
    """
    profile = layer_sparsity_profile(hidden_states, threshold)
    if measured_quality_delta is not None and measured_quality_delta > max_quality_delta_allowed:
        return SparsityPlan(
            masks=[],
            overall_sparsity=float(profile.mean()),
            max_quality_delta_allowed=max_quality_delta_allowed,
            killed=True,
            kill_reason=(
                f"quality delta {measured_quality_delta:.4f} > allowed "
                f"{max_quality_delta_allowed:.4f} — kill criterion hit, no sparsity applied"
            ),
        )
    masks: list[SparseMask] = []
    for i in range(hidden_states.shape[0]):
        if profile[i] >= min_sparsity_to_apply:
            masks.append(build_sparse_mask(hidden_states[i], threshold))
            masks[-1].layer = i
        else:
            masks.append(SparseMask(layer=i, keep=np.ones(hidden_states.shape[2], dtype=bool)))
    overall = float(sum(m.sparsity for m in masks) / len(masks)) if masks else 0.0
    return SparsityPlan(
        masks=masks,
        overall_sparsity=overall,
        max_quality_delta_allowed=max_quality_delta_allowed,
        notes=[f"mean measured sparsity {profile.mean():.1%} at threshold {threshold}"],
    )
