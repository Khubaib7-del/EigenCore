"""Tests for Phase 2 optimizer integration into the inference engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.engine.layer_skip import SkipStrategy
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec


def _mock_profile() -> HardwareProfile:
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


def _mock_spec() -> ModelSpec:
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


class TestEngineOptimizers:
    def test_engine_has_phase2_modules(self):
        engine = InferenceEngine(Path("dummy.gguf"), _mock_profile(), _mock_spec())
        assert engine.speculative_decoder is not None
        assert engine.speculative_decoder.draft_length == 5

    def test_optimizers_init_on_load(self):
        engine = InferenceEngine(Path("dummy.gguf"), _mock_profile(), _mock_spec())
        mock_llm = MagicMock()
        mock_llm.metadata = {
            "llama.block_count": "22",
            "llama.embedding_length": "2048",
        }

        with patch("llama_cpp.Llama", return_value=mock_llm):
            engine.load()

        assert engine.layer_skipper is not None
        assert engine.layer_skipper.num_layers == 22
        assert engine.sparsity_predictor is not None
        assert engine.sparsity_predictor.num_layers == 22
        assert engine.sparsity_predictor.neurons_per_layer == 2048

    def test_optimization_stats_populated_after_load(self):
        engine = InferenceEngine(Path("dummy.gguf"), _mock_profile(), _mock_spec())
        mock_llm = MagicMock()
        mock_llm.metadata = {
            "llama.block_count": "22",
            "llama.embedding_length": "2048",
        }

        with patch("llama_cpp.Llama", return_value=mock_llm):
            engine.load()

        stats = engine.optimization_stats
        assert "layer_skip" in stats
        assert stats["layer_skip"]["total_layers"] == 22
        assert stats["layer_skip"]["strategy"] == SkipStrategy.STATIC.name
        assert "sparsity_cache" in stats
        assert "speculative" in stats

    def test_generation_config_has_optimization_flags(self):
        config = GenerationConfig()
        assert config.enable_layer_skip is True
        assert config.enable_sparse_inference is True
        assert config.complexity == 0.5

    def test_stats_empty_when_not_loaded(self):
        engine = InferenceEngine(Path("dummy.gguf"), _mock_profile())
        assert engine.optimization_stats == {}

    def test_no_optimizers_when_metadata_missing(self):
        engine = InferenceEngine(Path("dummy.gguf"), _mock_profile(), _mock_spec())
        mock_llm = MagicMock()
        mock_llm.metadata = {}

        with patch("llama_cpp.Llama", return_value=mock_llm):
            engine.load()

        assert engine.layer_skipper is None
        assert engine.sparsity_predictor is None
