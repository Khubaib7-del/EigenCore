"""
Sparsity-aware inference — exploits the 70-90% activation sparsity in dense
transformer models to skip near-zero neuron computations on CPU.

GPU warp divergence makes sparse execution inefficient on GPUs (all 32 threads
in a warp must execute the same instruction). CPUs have fine-grained branch
prediction — skipping a zero multiply is a correctly-predicted branch that
costs nearly nothing. This is the core CPU advantage EigenCore exploits.

This module provides:
1. SparsityPredictor: learns per-layer sparsity patterns from observation
2. SparseExecutionPlan: decides which neurons to skip for a given input
3. SparsityCache: caches sparsity masks across similar inputs for reuse
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NeuronMask:
    layer_index: int
    total_neurons: int
    active_indices: np.ndarray
    sparsity: float

    @property
    def active_count(self) -> int:
        return len(self.active_indices)

    @property
    def skip_count(self) -> int:
        return self.total_neurons - self.active_count


@dataclass
class SparseExecutionPlan:
    masks: list[NeuronMask]
    total_neurons: int
    total_active: int
    overall_sparsity: float
    estimated_flop_savings: float
    estimated_speedup: float

    def summary(self) -> str:
        return (
            f"Sparse plan: {self.overall_sparsity:.0%} sparsity across "
            f"{len(self.masks)} layers | {self.estimated_speedup:.1f}x speedup | "
            f"{self.estimated_flop_savings:.0%} FLOPs saved"
        )


class SparsityPredictor:
    """
    Learns and predicts per-layer activation sparsity patterns.

    Tracks running statistics of which neurons activate (|value| > threshold)
    across multiple inputs. After enough observations, it can predict which
    neurons are likely inactive for a new input, enabling preemptive skipping.
    """

    def __init__(
        self,
        num_layers: int,
        neurons_per_layer: int,
        threshold: float = 0.01,
        warmup_samples: int = 16,
    ):
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.threshold = threshold
        self.warmup_samples = warmup_samples
        self._activation_counts = np.zeros((num_layers, neurons_per_layer), dtype=np.int32)
        self._total_samples = 0
        self._layer_sparsity_history: list[list[float]] = [[] for _ in range(num_layers)]

    @property
    def is_warmed_up(self) -> bool:
        return self._total_samples >= self.warmup_samples

    def observe(self, layer_index: int, activations: np.ndarray) -> float:
        """
        Record which neurons activated for this input.
        Returns the observed sparsity for this layer.
        """
        if layer_index >= self.num_layers:
            raise ValueError(f"Layer {layer_index} >= num_layers {self.num_layers}")

        abs_acts = np.abs(activations)
        active_mask = abs_acts >= self.threshold
        active_count = int(np.sum(active_mask))
        sparsity = 1.0 - (active_count / len(activations)) if len(activations) > 0 else 0.0

        n = min(len(activations), self.neurons_per_layer)
        self._activation_counts[layer_index, :n] += active_mask[:n].astype(np.int32)
        self._total_samples += 1
        self._layer_sparsity_history[layer_index].append(sparsity)

        return sparsity

    def predict_mask(self, layer_index: int, aggressiveness: float = 0.5) -> NeuronMask:
        """
        Predict which neurons will be active for the next input.

        aggressiveness: 0.0 (conservative, skip only always-zero neurons)
                       to 1.0 (aggressive, skip anything below median activation rate)
        """
        if not self.is_warmed_up:
            return NeuronMask(
                layer_index=layer_index,
                total_neurons=self.neurons_per_layer,
                active_indices=np.arange(self.neurons_per_layer),
                sparsity=0.0,
            )

        activation_rates = self._activation_counts[layer_index] / self._total_samples
        cutoff = aggressiveness * np.median(activation_rates)
        active_indices = np.where(activation_rates > cutoff)[0]

        sparsity = 1.0 - (len(active_indices) / self.neurons_per_layer)

        return NeuronMask(
            layer_index=layer_index,
            total_neurons=self.neurons_per_layer,
            active_indices=active_indices,
            sparsity=sparsity,
        )

    def create_execution_plan(self, aggressiveness: float = 0.5) -> SparseExecutionPlan:
        """Generate a full execution plan across all layers."""
        masks = []
        total_neurons = 0
        total_active = 0

        for layer_idx in range(self.num_layers):
            mask = self.predict_mask(layer_idx, aggressiveness)
            masks.append(mask)
            total_neurons += mask.total_neurons
            total_active += mask.active_count

        overall_sparsity = 1.0 - (total_active / total_neurons) if total_neurons > 0 else 0.0
        flop_savings = overall_sparsity
        speedup = 1.0 / (1.0 - overall_sparsity) if overall_sparsity < 1.0 else 1.0
        speedup = min(speedup, 5.0)

        return SparseExecutionPlan(
            masks=masks,
            total_neurons=total_neurons,
            total_active=total_active,
            overall_sparsity=overall_sparsity,
            estimated_flop_savings=flop_savings,
            estimated_speedup=speedup,
        )

    def layer_stats(self) -> list[dict]:
        """Per-layer statistics summary."""
        stats = []
        for i in range(self.num_layers):
            history = self._layer_sparsity_history[i]
            stats.append({
                "layer": i,
                "avg_sparsity": float(np.mean(history)) if history else 0.0,
                "std_sparsity": float(np.std(history)) if history else 0.0,
                "samples": len(history),
            })
        return stats


class SparsityCache:
    """
    Caches sparsity masks for similar inputs to avoid recomputation.

    Uses a hash of the input token sequence to look up previously computed
    masks. LRU eviction keeps memory bounded.
    """

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, SparseExecutionPlan] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_input(self, token_ids: list[int]) -> str:
        key = ",".join(str(t) for t in token_ids[-32:])
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, token_ids: list[int]) -> SparseExecutionPlan | None:
        key = self._hash_input(token_ids)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, token_ids: list[int], plan: SparseExecutionPlan) -> None:
        key = self._hash_input(token_ids)
        self._cache[key] = plan
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0
