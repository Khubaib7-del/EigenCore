"""P3 audit tests: prove which optimization flags actually reach the backend.

These inject a fake `llama_cpp` module so they run without the real backend.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec


def _install_fake_llama_cpp():
    fake = types.ModuleType("llama_cpp")
    fake.Llama = MagicMock()
    sys.modules["llama_cpp"] = fake
    return fake


def _profile() -> HardwareProfile:
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


def _spec() -> ModelSpec:
    return ModelSpec(
        name="test-model",
        repo_id="test/test",
        filename="test.gguf",
        params_b=1.1,
        quant="Q4_K_M",
        size_gb=0.67,
        task="general",
        context_length=2048,
        description="Test model",
    )


def _loaded_engine():
    fake = _install_fake_llama_cpp()
    mock_llm = MagicMock()
    mock_llm.metadata = {"llama.block_count": "22", "llama.embedding_length": "2048"}
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    fake.Llama.return_value = mock_llm
    engine = InferenceEngine(Path("dummy.gguf"), _profile(), _spec())
    engine.load()
    return engine, mock_llm


def test_flags_do_not_change_backend_kwargs():
    engine, mock_llm = _loaded_engine()
    baseline = GenerationConfig(stream=False, enable_layer_skip=False, enable_sparse_inference=False)
    all_on = GenerationConfig(stream=False, enable_layer_skip=True, enable_sparse_inference=True, complexity=0.1)
    engine.chat([{"role": "user", "content": "a"}], baseline)
    kwargs_baseline = mock_llm.create_chat_completion.call_args.kwargs
    mock_llm.create_chat_completion.reset_mock()
    engine.chat([{"role": "user", "content": "a"}], all_on)
    kwargs_all_on = mock_llm.create_chat_completion.call_args.kwargs
    assert kwargs_baseline == kwargs_all_on


def test_result_reports_no_applied_optimizations():
    engine, _ = _loaded_engine()
    result = engine.chat(
        [{"role": "user", "content": "a"}],
        GenerationConfig(stream=False, enable_layer_skip=True, enable_sparse_inference=True),
    )
    assert result.applied_optimizations == []
    stats = result.optimization_stats
    assert stats["backend_wired"] == []
    assert stats["layer_skip"]["applied"] is False
    assert stats["speculative"]["applied"] is False
    assert "planned/reported only" in stats["note"]
