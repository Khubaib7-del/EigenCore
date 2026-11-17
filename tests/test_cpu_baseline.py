"""0.4.10: a CPU without the build's instruction set gets a clear error, not a crash."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packaging" / "windows"))

from eigencore.engine.inference import (
    InferenceEngine,
    cpu_unsupported_message,
    is_illegal_instruction,
    missing_cpu_features,
)
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec


def _profile(isa=InstructionSet.AVX | InstructionSet.AVX2, arch="x86_64"):
    return HardwareProfile(
        cpu_name="Test CPU",
        arch=arch,
        physical_cores=4,
        logical_cores=8,
        instruction_sets=isa,
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        estimated_bandwidth_gbps=25.0,
        max_model_params_b=7.0,
        optimal_quantization="Q4_K_M",
        estimated_tokens_per_sec=15.0,
        recommended_threads=4,
        recommended_context_length=2048,
    )


def _spec():
    return ModelSpec(
        name="t",
        repo_id="t/t",
        filename="t.gguf",
        params_b=1.1,
        quant="Q4_K_M",
        size_gb=0.67,
        task="general",
        context_length=2048,
        description="t",
    )


def _fake_llama(side_effect=None):
    fake = types.ModuleType("llama_cpp")
    fake.Llama = MagicMock(side_effect=side_effect, return_value=MagicMock(metadata={}))
    sys.modules["llama_cpp"] = fake
    return fake


def test_missing_features():
    assert missing_cpu_features(_profile()) == []
    assert missing_cpu_features(_profile(InstructionSet.SSE4_2 | InstructionSet.AVX)) == ["AVX2"]
    # unknown detection and non-x86 never block
    assert missing_cpu_features(_profile(InstructionSet.NONE)) == []
    assert missing_cpu_features(_profile(InstructionSet.NEON, arch="arm64")) == []


def test_illegal_instruction_detected():
    assert is_illegal_instruction(OSError("[WinError -1073741795] Windows Error 0xc000001d"))
    assert not is_illegal_instruction(OSError("[WinError 126] module not found"))


def test_load_blocks_cpu_without_avx2():
    fake = _fake_llama()
    engine = InferenceEngine(Path("dummy.gguf"), _profile(InstructionSet.AVX), _spec())
    with pytest.raises(RuntimeError, match="lacks AVX2"):
        engine.load()
    fake.Llama.assert_not_called()


def test_load_maps_illegal_instruction(monkeypatch):
    _fake_llama(OSError("[WinError -1073741795] Windows Error 0xc000001d"))
    engine = InferenceEngine(Path("dummy.gguf"), _profile(), _spec())
    monkeypatch.setattr(engine, "_default_context_length", lambda type_v: 2048)
    with pytest.raises(RuntimeError) as e:
        engine.load()
    assert "0xC000001D" in str(e.value)
    assert str(e.value) == cpu_unsupported_message()


def test_other_os_errors_pass_through(monkeypatch):
    _fake_llama(OSError("[WinError 126] module not found"))
    engine = InferenceEngine(Path("dummy.gguf"), _profile(), _spec())
    monkeypatch.setattr(engine, "_default_context_length", lambda type_v: 2048)
    with pytest.raises(OSError, match="126"):
        engine.load()


def test_baseline_check_flags_avx512():
    pytest.importorskip("capstone")
    from check_cpu_baseline import avx512_instructions

    assert avx512_instructions(bytes.fromhex("62f1744858c2"))  # vaddps zmm0, zmm1, zmm2
    assert not avx512_instructions(bytes.fromhex("c5f458c2"))  # vaddps ymm0, ymm1, ymm2
    assert not avx512_instructions(bytes.fromhex("c4e2f5a8c2"))  # vfmadd213pd ymm (FMA)


def test_primary_screen_area(monkeypatch):
    from eigencore.desktop import web

    screen = types.SimpleNamespace
    fake = types.ModuleType("webview")
    fake.screens = [
        screen(x=-1920, y=0, width=1920, height=1080),
        screen(x=0, y=0, width=1536, height=864),
    ]
    monkeypatch.setitem(sys.modules, "webview", fake)
    assert web.primary_screen_area() == (0, 0, 1536, 864)
    fake.screens = []
    assert web.primary_screen_area() is None
