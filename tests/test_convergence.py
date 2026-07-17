"""Tests for the convergence monitor — validates the 1/4 consistency rule."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.training.convergence import ConvergenceMonitor


def test_not_converged_with_decreasing_loss():
    """A steadily decreasing loss should NOT trigger convergence."""
    monitor = ConvergenceMonitor()
    for i in range(20):
        loss = 5.0 - (i * 0.2)  # linearly decreasing
        check = monitor.record(loss)
    assert not check.converged, "Steadily decreasing loss should not converge"


def test_converged_with_flat_loss():
    """A flat loss curve should trigger convergence."""
    monitor = ConvergenceMonitor()
    # first 10 epochs: decreasing
    for i in range(10):
        monitor.record(5.0 - (i * 0.3))
    # next 10 epochs: flat at ~2.0
    for i in range(10):
        check = monitor.record(2.0 + 0.001 * (i % 3))  # tiny noise
    assert check.converged, "Flat loss should trigger convergence"


def test_not_converged_with_oscillating_loss():
    """Oscillating loss should NOT trigger convergence."""
    monitor = ConvergenceMonitor(consistency_threshold=0.02)
    for i in range(20):
        loss = 3.0 + (0.5 if i % 2 == 0 else -0.5)  # oscillating ±0.5
        check = monitor.record(loss)
    assert not check.converged, "Oscillating loss should not converge"


def test_phase_advancement():
    """When converged, should advance to next phase."""
    monitor = ConvergenceMonitor()
    # simulate convergence
    for _ in range(20):
        monitor.record(2.0)

    assert monitor.should_advance_phase()
    new_epochs = monitor.advance_phase()
    assert monitor.current_phase == 2
    assert new_epochs > 100  # scaled up
    assert len(monitor.losses) == 0  # reset for new phase


def test_max_phases_respected():
    """Should not advance beyond max_phases."""
    monitor = ConvergenceMonitor(max_phases=2)

    # converge phase 1
    for _ in range(20):
        monitor.record(2.0)
    monitor.advance_phase()

    # converge phase 2
    for _ in range(20):
        monitor.record(1.5)

    assert not monitor.should_advance_phase()
    assert monitor.current_phase == 2


def test_coefficient_of_variation():
    """CV should be correctly calculated."""
    monitor = ConvergenceMonitor()

    # all same value → CV should be 0
    for _ in range(10):
        check = monitor.record(3.0)
    assert check.coefficient_of_variation == 0.0
    assert check.converged


def test_too_few_epochs():
    """Should not converge with < 4 data points."""
    monitor = ConvergenceMonitor()
    check = monitor.record(5.0)
    assert not check.converged
    check = monitor.record(4.0)
    assert not check.converged


if __name__ == "__main__":
    test_not_converged_with_decreasing_loss()
    print("test_not_converged_with_decreasing_loss PASSED")

    test_converged_with_flat_loss()
    print("test_converged_with_flat_loss PASSED")

    test_not_converged_with_oscillating_loss()
    print("test_not_converged_with_oscillating_loss PASSED")

    test_phase_advancement()
    print("test_phase_advancement PASSED")

    test_max_phases_respected()
    print("test_max_phases_respected PASSED")

    test_coefficient_of_variation()
    print("test_coefficient_of_variation PASSED")

    test_too_few_epochs()
    print("test_too_few_epochs PASSED")

    print("\nAll convergence tests passed.")
