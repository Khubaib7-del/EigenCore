"""
Convergence analysis utilities — implements the adaptive epoch scaling logic
independently of the training loop so it can be tested and visualized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConvergenceCheck:
    window_size: int
    window_losses: list[float]
    mean_loss: float
    std_dev: float
    coefficient_of_variation: float
    threshold: float
    converged: bool

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        return (
            f"[{status}] CV={self.coefficient_of_variation:.4f} "
            f"(threshold={self.threshold:.4f}) | "
            f"mean={self.mean_loss:.6f} std={self.std_dev:.6f} "
            f"window={self.window_size}"
        )


class ConvergenceMonitor:
    """
    Monitors training loss and determines phase transitions
    using the 1/4 consistency rule.
    """

    def __init__(
        self,
        consistency_window: float = 0.25,
        consistency_threshold: float = 0.02,
        epoch_scale_factor: float = 2.5,
        max_phases: int = 3,
    ):
        self.consistency_window = consistency_window
        self.consistency_threshold = consistency_threshold
        self.epoch_scale_factor = epoch_scale_factor
        self.max_phases = max_phases

        self.losses: list[float] = []
        self.current_phase = 1
        self.phase_boundaries: list[int] = []

    def record(self, loss: float) -> ConvergenceCheck:
        """Record a loss value and check convergence."""
        self.losses.append(loss)
        return self.check()

    def check(self) -> ConvergenceCheck:
        """Check current convergence state without recording a new loss."""
        if len(self.losses) < 4:
            return ConvergenceCheck(
                window_size=len(self.losses),
                window_losses=list(self.losses),
                mean_loss=sum(self.losses) / len(self.losses) if self.losses else 0,
                std_dev=0.0,
                coefficient_of_variation=1.0,
                threshold=self.consistency_threshold,
                converged=False,
            )

        window_size = max(int(len(self.losses) * self.consistency_window), 2)
        recent = self.losses[-window_size:]

        mean_loss = sum(recent) / len(recent)
        variance = sum((x - mean_loss) ** 2 for x in recent) / len(recent)
        std_dev = variance ** 0.5
        cv = std_dev / abs(mean_loss) if mean_loss != 0 else 0.0

        return ConvergenceCheck(
            window_size=window_size,
            window_losses=recent,
            mean_loss=mean_loss,
            std_dev=std_dev,
            coefficient_of_variation=cv,
            threshold=self.consistency_threshold,
            converged=cv < self.consistency_threshold,
        )

    def should_advance_phase(self) -> bool:
        """Check if we should advance to the next training phase."""
        if self.current_phase >= self.max_phases:
            return False
        return self.check().converged

    def advance_phase(self) -> int:
        """Advance to next phase. Returns the new epoch count for the phase."""
        self.phase_boundaries.append(len(self.losses))
        self.current_phase += 1
        self.losses.clear()
        return int(100 * (self.epoch_scale_factor ** (self.current_phase - 1)))

    def get_phase_summary(self) -> str:
        """Get a summary of all phases completed."""
        check = self.check()
        lines = [
            f"Phase {self.current_phase}/{self.max_phases}",
            f"  Epochs recorded: {len(self.losses)}",
            f"  {check.summary()}",
        ]
        if self.phase_boundaries:
            lines.append(f"  Phase transitions at epochs: {self.phase_boundaries}")
        return "\n".join(lines)
