"""E0: the engine's n_ctx comes from the fit estimator with real GGUF geometry."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.engine.inference import InferenceEngine
from eigencore.fit import ModelGeometry
from eigencore.fit.estimator import MIN_DEFAULT_CONTEXT, default_context_length
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec

# 32 layers × 8 kv heads × (128 + 128) × 2 bytes = 128 KiB per token at fp16
GEO = ModelGeometry(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_kv_heads=8,
    key_length=128,
    value_length=128,
)


def test_context_from_ram_budget():
    # 4 GiB - 2 reserve - 1 margin - 0 weights = 1 GiB → 8192 tokens at 128 KiB
    assert default_context_length(GEO, 4.0, 32768, file_size_bytes=0) == 8192


def test_q8_kv_doubles_context():
    assert default_context_length(GEO, 4.0, 32768, kv_quant="q8_0", file_size_bytes=0) == 16384


def test_capped_at_16k_on_big_machines():
    assert default_context_length(GEO, 64.0, 131072, file_size_bytes=0) == 16384


def test_cap_is_adjustable():
    assert default_context_length(GEO, 64.0, 131072, file_size_bytes=0, cap=65536) == 65536


def test_clamped_to_model_max():
    assert default_context_length(GEO, 64.0, 4096, file_size_bytes=0) == 4096


def test_floor_when_budget_is_tiny():
    assert default_context_length(GEO, 3.0, 32768, file_size_bytes=0) == MIN_DEFAULT_CONTEXT


def test_floor_never_exceeds_model_max():
    assert default_context_length(GEO, 3.0, 1024, file_size_bytes=0) == 1024


def test_none_without_geometry():
    assert default_context_length(None, 16.0, 4096) is None
    assert default_context_length(ModelGeometry(num_layers=2), 16.0, 4096) is None


# ---- engine wiring -----------------------------------------------------------


def _profile(available_ram_gb: float) -> HardwareProfile:
    return HardwareProfile(
        cpu_name="Test CPU",
        arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=InstructionSet.AVX2 | InstructionSet.SSE2,
        total_ram_gb=16.0,
        available_ram_gb=available_ram_gb,
        estimated_bandwidth_gbps=25.0,
        max_model_params_b=7.0,
        optimal_quantization="Q4_K_M",
        estimated_tokens_per_sec=15.0,
        recommended_threads=4,
        recommended_context_length=2048,  # the profiler's geometry-free figure
    )


def _spec(ctx: int = 2048) -> ModelSpec:
    return ModelSpec(
        name="t",
        repo_id="t/t",
        filename="t.gguf",
        params_b=0.1,
        quant="Q4_K_M",
        size_gb=0.01,
        task="general",
        context_length=ctx,
        description="t",
    )


def _write_gguf(path: Path, context_length: int) -> None:
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), "qwen3")
    w.add_block_count(32)
    w.add_embedding_length(4096)
    w.add_head_count(32)
    w.add_head_count_kv(8)
    w.add_key_length(128)
    w.add_value_length(128)
    w.add_context_length(context_length)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _load(mp, model_path: Path, ram: float, spec: ModelSpec, **kw):
    fake = types.ModuleType("llama_cpp")
    fake.Llama = MagicMock()
    fake.Llama.return_value = MagicMock(metadata={})
    mp.setitem(sys.modules, "llama_cpp", fake)
    engine = InferenceEngine(model_path, _profile(ram), spec, **kw)
    # keep the optimizer setup out of this test
    engine._init_optimizers = lambda: None
    engine.load()
    return fake.Llama.call_args.kwargs, engine


def test_engine_uses_gguf_geometry_not_profiler(tmp_path, monkeypatch):
    p = tmp_path / "m.gguf"
    _write_gguf(p, context_length=40960)
    # file is a few KB; 4.1 GiB RAM → just over 1 GiB KV budget → 8192 tokens
    kwargs, engine = _load(monkeypatch, p, 4.1, _spec(ctx=2048))
    assert kwargs["n_ctx"] == 8192
    assert engine.n_ctx == 8192


def test_engine_context_clamped_to_gguf_max(tmp_path, monkeypatch):
    p = tmp_path / "m.gguf"
    _write_gguf(p, context_length=4096)
    kwargs, _ = _load(monkeypatch, p, 64.0, _spec())
    assert kwargs["n_ctx"] == 4096


def test_engine_q8_kv_sizes_context(tmp_path, monkeypatch):
    p = tmp_path / "m.gguf"
    _write_gguf(p, context_length=40960)
    kwargs, _ = _load(monkeypatch, p, 4.1, _spec(), kv_type="q8_0")
    assert kwargs["n_ctx"] == 16384


def test_engine_falls_back_without_gguf(monkeypatch):
    kwargs, engine = _load(monkeypatch, Path("missing.gguf"), 4.0, _spec(ctx=1024))
    assert kwargs["n_ctx"] == 1024  # profiler 2048 clamped to the registry context
    assert engine.n_ctx == 1024
