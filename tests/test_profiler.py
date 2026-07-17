"""Tests for the hardware profiler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.hal.profiler import profile_hardware, InstructionSet


def test_profile_returns_valid_data():
    hw = profile_hardware()

    assert hw.cpu_name, "CPU name should not be empty"
    assert hw.arch in ("amd64", "x86_64", "x86", "aarch64", "arm64"), f"Unexpected arch: {hw.arch}"
    assert hw.physical_cores >= 1
    assert hw.logical_cores >= hw.physical_cores
    assert hw.total_ram_gb > 0
    assert hw.available_ram_gb > 0
    assert hw.available_ram_gb <= hw.total_ram_gb
    assert hw.estimated_bandwidth_gbps > 0
    assert hw.max_model_params_b > 0
    assert hw.optimal_quantization in ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")
    assert hw.estimated_tokens_per_sec >= 0
    assert hw.recommended_threads >= 1
    assert hw.recommended_context_length >= 512


def test_summary_format():
    hw = profile_hardware()
    summary = hw.summary()

    assert "CPU:" in summary
    assert "Cores:" in summary
    assert "RAM:" in summary
    assert "Bandwidth:" in summary
    assert "Max model:" in summary


def test_instruction_set_detection():
    hw = profile_hardware()

    # on any modern x86 CPU, SSE2 should be detected
    if hw.arch in ("amd64", "x86_64"):
        assert InstructionSet.SSE2 in hw.instruction_sets


if __name__ == "__main__":
    test_profile_returns_valid_data()
    print("test_profile_returns_valid_data PASSED")

    test_summary_format()
    print("test_summary_format PASSED")

    test_instruction_set_detection()
    print("test_instruction_set_detection PASSED")

    print("\nAll profiler tests passed.")
