"""Tests for the task router."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.hal.profiler import profile_hardware
from eigencore.router.classifier import TaskRouter, TaskDomain


def test_code_detection():
    hw = profile_hardware()
    router = TaskRouter(hw)

    domain, conf, _ = router.classify("Write a Python function to sort a list")
    assert domain == TaskDomain.CODE

    domain, conf, _ = router.classify("Debug this JavaScript error in my React component")
    assert domain == TaskDomain.CODE

    domain, conf, _ = router.classify("Implement a binary tree in C++")
    assert domain == TaskDomain.CODE


def test_math_detection():
    hw = profile_hardware()
    router = TaskRouter(hw)

    domain, conf, _ = router.classify("Calculate the integral of x^2 from 0 to 5")
    assert domain == TaskDomain.MATH

    domain, conf, _ = router.classify("Solve this matrix equation: A * x = b")
    assert domain == TaskDomain.MATH


def test_creative_detection():
    hw = profile_hardware()
    router = TaskRouter(hw)

    domain, conf, _ = router.classify("Write a short story about a robot")
    assert domain == TaskDomain.CREATIVE


def test_general_fallback():
    hw = profile_hardware()
    router = TaskRouter(hw)

    domain, conf, _ = router.classify("Hello, how are you?")
    assert domain == TaskDomain.GENERAL


def test_routing_returns_model():
    hw = profile_hardware()
    router = TaskRouter(hw)

    decision = router.route("Write a Python function to parse JSON")
    assert decision.domain == TaskDomain.CODE
    assert decision.model is not None
    assert decision.confidence > 0


def test_swap_logic():
    hw = profile_hardware()
    router = TaskRouter(hw)

    from eigencore.models.registry import ModelRegistry

    registry = ModelRegistry()
    general_model = registry.recommend(hw, "general")

    # low confidence should not trigger swap
    decision = router.route("hello")
    assert not router.should_swap(general_model, decision)


if __name__ == "__main__":
    test_code_detection()
    print("test_code_detection PASSED")

    test_math_detection()
    print("test_math_detection PASSED")

    test_creative_detection()
    print("test_creative_detection PASSED")

    test_general_fallback()
    print("test_general_fallback PASSED")

    test_routing_returns_model()
    print("test_routing_returns_model PASSED")

    test_swap_logic()
    print("test_swap_logic PASSED")

    print("\nAll router tests passed.")
