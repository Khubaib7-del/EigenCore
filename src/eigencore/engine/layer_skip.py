"""
Dynamic layer skipping — decides which transformer layers to skip based on
per-layer sparsity profiles and input complexity.

Transformer models have high layer redundancy: many middle layers produce
near-identical residual stream updates for routine tokens. Skipping them
on CPU saves wall-clock time proportional to the skip rate with minimal
quality loss (typically < 2% perplexity increase at 20-30% skip rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class SkipStrategy(Enum):
    NONE = auto()
    STATIC = auto()
    ADAPTIVE = auto()
    CALIBRATED = auto()


@dataclass
class LayerProfile:
    index: int
    importance: float
    avg_sparsity: float
    skip_safe: bool
    samples_seen: int = 0


@dataclass
class SkipPlan:
    total_layers: int
    layers_to_run: list[int]
    layers_to_skip: list[int]
    strategy: SkipStrategy
    expected_speedup: float
    expected_quality_loss: float

    @property
    def skip_rate(self) -> float:
        if self.total_layers == 0:
            return 0.0
        return len(self.layers_to_skip) / self.total_layers

    def summary(self) -> str:
        return (
            f"Skip {len(self.layers_to_skip)}/{self.total_layers} layers "
            f"({self.skip_rate:.0%}) | ~{self.expected_speedup:.1f}x speedup | "
            f"~{self.expected_quality_loss:.1%} quality loss"
        )


class LayerSkipScheduler:
    """
    Determines which transformer layers to skip during inference.

    Three strategies:
    1. STATIC — skip fixed layers based on position heuristics (no calibration needed)
    2. ADAPTIVE — adjust skip decisions per-token based on running sparsity stats
    3. CALIBRATED — use pre-computed per-layer importance scores from a calibration run

    The key insight: in most transformer models, the first ~15% and last ~15% of
    layers are critical (embedding projection + output head). The middle layers
    show high redundancy, especially for simple/repetitive tokens.
    """

    CRITICAL_HEAD_RATIO = 0.15
    CRITICAL_TAIL_RATIO = 0.15

    def __init__(
        self,
        num_layers: int,
        strategy: SkipStrategy = SkipStrategy.STATIC,
        max_skip_rate: float = 0.3,
        quality_threshold: float = 0.02,
    ):
        self.num_layers = num_layers
        self.strategy = strategy
        self.max_skip_rate = max_skip_rate
        self.quality_threshold = quality_threshold
        self._layer_profiles: list[LayerProfile] = []
        self._calibrated = False

    def plan(self, complexity: float = 0.5) -> SkipPlan:
        """
        Generate a skip plan for the current inference pass.

        complexity: 0.0 (trivial repetition) to 1.0 (novel/complex content).
        Lower complexity → more aggressive skipping.
        """
        if self.num_layers <= 4:
            return self._no_skip_plan()

        if self.strategy == SkipStrategy.NONE:
            return self._no_skip_plan()
        elif self.strategy == SkipStrategy.STATIC:
            return self._static_plan(complexity)
        elif self.strategy == SkipStrategy.ADAPTIVE:
            return self._adaptive_plan(complexity)
        elif self.strategy == SkipStrategy.CALIBRATED:
            if not self._calibrated:
                return self._static_plan(complexity)
            return self._calibrated_plan(complexity)

        return self._no_skip_plan()

    def calibrate(self, layer_importances: list[float]) -> None:
        """
        Load per-layer importance scores from a calibration run.
        Importance is typically measured as the mean L2 norm of each layer's
        residual stream contribution over a calibration dataset.
        """
        if len(layer_importances) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} importance scores, got {len(layer_importances)}"
            )

        self._layer_profiles = [
            LayerProfile(
                index=i,
                importance=imp,
                avg_sparsity=0.0,
                skip_safe=(imp < np.median(layer_importances)),
            )
            for i, imp in enumerate(layer_importances)
        ]
        self._calibrated = True

    def update_sparsity(self, layer_index: int, sparsity: float) -> None:
        """Update running sparsity stats for adaptive strategy."""
        if layer_index >= len(self._layer_profiles):
            while len(self._layer_profiles) <= layer_index:
                self._layer_profiles.append(
                    LayerProfile(
                        index=len(self._layer_profiles),
                        importance=1.0,
                        avg_sparsity=0.0,
                        skip_safe=False,
                    )
                )

        profile = self._layer_profiles[layer_index]
        n = profile.samples_seen
        profile.avg_sparsity = (profile.avg_sparsity * n + sparsity) / (n + 1)
        profile.samples_seen = n + 1

        if profile.avg_sparsity > 0.7 and profile.samples_seen >= 10:
            profile.skip_safe = True

    def _no_skip_plan(self) -> SkipPlan:
        return SkipPlan(
            total_layers=self.num_layers,
            layers_to_run=list(range(self.num_layers)),
            layers_to_skip=[],
            strategy=SkipStrategy.NONE,
            expected_speedup=1.0,
            expected_quality_loss=0.0,
        )

    def _static_plan(self, complexity: float) -> SkipPlan:
        """Skip middle layers at fixed intervals based on complexity."""
        head = max(1, int(self.num_layers * self.CRITICAL_HEAD_RATIO))
        tail = max(1, int(self.num_layers * self.CRITICAL_TAIL_RATIO))
        tail_start = self.num_layers - tail

        middle_layers = list(range(head, tail_start))
        if not middle_layers:
            return self._no_skip_plan()

        adjusted_skip = self.max_skip_rate * (1.0 - complexity)
        max_skippable = max(1, int(len(middle_layers) * adjusted_skip))

        step = max(1, len(middle_layers) // max_skippable) if max_skippable > 0 else 1
        skip_set = set(middle_layers[::step][:max_skippable])

        layers_to_run = [i for i in range(self.num_layers) if i not in skip_set]
        layers_to_skip = sorted(skip_set)

        skip_rate = len(layers_to_skip) / self.num_layers
        speedup = 1.0 / (1.0 - skip_rate) if skip_rate < 1.0 else 1.0
        quality_loss = skip_rate * 0.06

        return SkipPlan(
            total_layers=self.num_layers,
            layers_to_run=layers_to_run,
            layers_to_skip=layers_to_skip,
            strategy=SkipStrategy.STATIC,
            expected_speedup=speedup,
            expected_quality_loss=quality_loss,
        )

    def _adaptive_plan(self, complexity: float) -> SkipPlan:
        """Skip layers with high observed sparsity."""
        if not self._layer_profiles:
            return self._static_plan(complexity)

        head = max(1, int(self.num_layers * self.CRITICAL_HEAD_RATIO))
        tail = max(1, int(self.num_layers * self.CRITICAL_TAIL_RATIO))
        tail_start = self.num_layers - tail

        skip_set = set()
        max_total_skip = int(self.num_layers * self.max_skip_rate * (1.0 - complexity))

        candidates = [
            p for p in self._layer_profiles if p.skip_safe and head <= p.index < tail_start
        ]
        candidates.sort(key=lambda p: p.avg_sparsity, reverse=True)

        for profile in candidates[:max_total_skip]:
            skip_set.add(profile.index)

        layers_to_run = [i for i in range(self.num_layers) if i not in skip_set]
        layers_to_skip = sorted(skip_set)

        skip_rate = len(layers_to_skip) / self.num_layers
        speedup = 1.0 / (1.0 - skip_rate) if skip_rate < 1.0 else 1.0
        quality_loss = skip_rate * 0.04

        return SkipPlan(
            total_layers=self.num_layers,
            layers_to_run=layers_to_run,
            layers_to_skip=layers_to_skip,
            strategy=SkipStrategy.ADAPTIVE,
            expected_speedup=speedup,
            expected_quality_loss=quality_loss,
        )

    def _calibrated_plan(self, complexity: float) -> SkipPlan:
        """Skip layers ranked by calibrated importance scores."""
        head = max(1, int(self.num_layers * self.CRITICAL_HEAD_RATIO))
        tail = max(1, int(self.num_layers * self.CRITICAL_TAIL_RATIO))
        tail_start = self.num_layers - tail

        middle_profiles = [p for p in self._layer_profiles if head <= p.index < tail_start]
        middle_profiles.sort(key=lambda p: p.importance)

        max_total_skip = int(self.num_layers * self.max_skip_rate * (1.0 - complexity))
        skip_set = set()
        cumulative_loss = 0.0

        for profile in middle_profiles:
            estimated_loss = (1.0 - profile.importance) * 0.01
            if cumulative_loss + estimated_loss > self.quality_threshold:
                break
            if len(skip_set) >= max_total_skip:
                break
            skip_set.add(profile.index)
            cumulative_loss += estimated_loss

        layers_to_run = [i for i in range(self.num_layers) if i not in skip_set]
        layers_to_skip = sorted(skip_set)

        skip_rate = len(layers_to_skip) / self.num_layers
        speedup = 1.0 / (1.0 - skip_rate) if skip_rate < 1.0 else 1.0

        return SkipPlan(
            total_layers=self.num_layers,
            layers_to_run=layers_to_run,
            layers_to_skip=layers_to_skip,
            strategy=SkipStrategy.CALIBRATED,
            expected_speedup=speedup,
            expected_quality_loss=cumulative_loss,
        )
