"""Stage C: KV cache type and flash attention are plumbed into the backend load."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.engine.inference import InferenceEngine
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec


def _install_fake_llama_cpp():
    fake = types.ModuleType("llama_cpp")
    fake.Llama = MagicMock()
    sys.modules["llama_cpp"] = fake
    return fake


def _profile():
    return HardwareProfile(
        cpu_name="Test CPU",
        arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=InstructionSet.AVX2 | InstructionSet.SSE2,
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        estimated_bandwidth_gbps=25.0,
        max_model_params_b=7.0,
        optimal_quantization="Q4_K_M",
        estimated_tokens_per_sec=15.0,
        recommended_threads=4,
        recommended_context_length=4096,
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


def _load_with(**kwargs):
    fake = _install_fake_llama_cpp()
    fake.Llama.return_value = MagicMock(metadata={})
    engine = InferenceEngine(Path("dummy.gguf"), _profile(), _spec(), **kwargs)
    engine.load()
    return fake.Llama.call_args.kwargs


def test_defaults_are_fp16_kv_and_flash_on():
    kwargs = _load_with()
    assert kwargs["type_kv"] == 0  # fp16
    assert kwargs["flash_attn"] is True


def test_q8_0_kv_maps_to_type_1():
    kwargs = _load_with(kv_type="q8_0")
    assert kwargs["type_kv"] == 1


def test_no_flash():
    kwargs = _load_with(flash_attn=False)
    assert kwargs["flash_attn"] is False
