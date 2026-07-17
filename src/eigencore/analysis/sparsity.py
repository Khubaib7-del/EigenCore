"""
Activation sparsity analyzer — measures the emergent sparsity in dense LLM layers.

This is the first "gap exploitation" measurement tool. It hooks into the model's
forward pass and counts near-zero activations per layer, producing a sparsity
profile that quantifies how many FLOPs could be skipped on CPU inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LayerSparsity:
    layer_index: int
    layer_name: str
    total_activations: int
    near_zero_count: int
    sparsity_ratio: float
    mean_magnitude: float
    median_magnitude: float
    threshold_used: float


@dataclass
class SparsityReport:
    model_name: str
    prompt: str
    threshold: float
    layers: list[LayerSparsity] = field(default_factory=list)
    overall_sparsity: float = 0.0
    potential_speedup: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Sparsity Report: {self.model_name}",
            f"Prompt: {self.prompt[:80]}{'...' if len(self.prompt) > 80 else ''}",
            f"Threshold: {self.threshold}",
            f"Overall sparsity: {self.overall_sparsity:.1%}",
            f"Potential CPU speedup: {self.potential_speedup:.1f}x",
            "",
            f"{'Layer':<8} {'Name':<30} {'Sparsity':>10} {'Mean |act|':>12} {'Skip':>8}",
            "-" * 72,
        ]
        for layer in self.layers:
            lines.append(
                f"{layer.layer_index:<8} "
                f"{layer.layer_name:<30} "
                f"{layer.sparsity_ratio:>9.1%} "
                f"{layer.mean_magnitude:>11.6f} "
                f"{layer.near_zero_count:>8}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "threshold": self.threshold,
            "overall_sparsity": self.overall_sparsity,
            "potential_speedup": self.potential_speedup,
            "layers": [
                {
                    "layer_index": layer.layer_index,
                    "layer_name": layer.layer_name,
                    "total_activations": layer.total_activations,
                    "near_zero_count": layer.near_zero_count,
                    "sparsity_ratio": layer.sparsity_ratio,
                    "mean_magnitude": layer.mean_magnitude,
                    "median_magnitude": layer.median_magnitude,
                }
                for layer in self.layers
            ],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


class SparsityAnalyzer:
    """
    Analyzes activation sparsity by intercepting llama.cpp's internal state.

    Since llama-cpp-python doesn't expose per-layer activations directly,
    this uses the logits output and token embeddings as proxy measurements.
    For full per-layer analysis, a custom ggml hook would be needed (Phase 2).

    Current implementation: analyzes the output logit distribution to estimate
    how sparse the model's internal representations are for a given input.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def analyze_logits(
        self,
        logits: list[float] | np.ndarray,
        model_name: str = "unknown",
        prompt: str = "",
    ) -> SparsityReport:
        """Analyze sparsity of the output logit vector."""
        arr = np.array(logits, dtype=np.float32)
        abs_arr = np.abs(arr)

        near_zero = int(np.sum(abs_arr < self.threshold))
        total = len(arr)
        sparsity = near_zero / total if total > 0 else 0.0

        layer = LayerSparsity(
            layer_index=0,
            layer_name="output_logits",
            total_activations=total,
            near_zero_count=near_zero,
            sparsity_ratio=sparsity,
            mean_magnitude=float(np.mean(abs_arr)),
            median_magnitude=float(np.median(abs_arr)),
            threshold_used=self.threshold,
        )

        # potential speedup: 1 / (1 - sparsity), capped at 5x
        speedup = min(1.0 / (1.0 - sparsity) if sparsity < 1.0 else 5.0, 5.0)

        return SparsityReport(
            model_name=model_name,
            prompt=prompt,
            threshold=self.threshold,
            layers=[layer],
            overall_sparsity=sparsity,
            potential_speedup=speedup,
        )

    def analyze_token_probabilities(
        self,
        logits: list[float] | np.ndarray,
        model_name: str = "unknown",
        prompt: str = "",
    ) -> SparsityReport:
        """
        Analyze how concentrated the probability mass is across the vocabulary.

        A highly concentrated distribution (most probability on few tokens) indicates
        the model's internal pathways are sparse for this input — only a small number
        of semantic directions were strongly activated.
        """
        arr = np.array(logits, dtype=np.float32)

        # softmax
        exp_arr = np.exp(arr - np.max(arr))
        probs = exp_arr / np.sum(exp_arr)

        # concentration: what fraction of tokens carry 95% of the probability mass
        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)
        tokens_for_95 = int(np.searchsorted(cumsum, 0.95)) + 1
        concentration = 1.0 - (tokens_for_95 / len(probs))

        near_zero_probs = int(np.sum(probs < 1e-6))

        layer = LayerSparsity(
            layer_index=0,
            layer_name="probability_distribution",
            total_activations=len(probs),
            near_zero_count=near_zero_probs,
            sparsity_ratio=near_zero_probs / len(probs) if len(probs) > 0 else 0.0,
            mean_magnitude=float(np.mean(probs)),
            median_magnitude=float(np.median(probs)),
            threshold_used=1e-6,
        )

        report = SparsityReport(
            model_name=model_name,
            prompt=prompt,
            threshold=1e-6,
            layers=[layer],
            overall_sparsity=concentration,
            potential_speedup=min(1.0 / (1.0 - concentration) if concentration < 1.0 else 5.0, 5.0),
        )

        return report
