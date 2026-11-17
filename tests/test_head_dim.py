"""E0: head_dim comes from GGUF attention.key_length/value_length when present."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.fit import ModelGeometry, estimate_kv_cache_gib, max_context_for_budget
from eigencore.fit.estimator import kv_bytes_per_token
from eigencore.fit.gguf_meta import read_gguf_model_info

# Qwen3-0.6B-shaped geometry: hidden 1024 / 16 heads = 64, but head_dim is 128.
QWEN3_LIKE = ModelGeometry(
    num_layers=28,
    hidden_size=1024,
    num_attention_heads=16,
    num_kv_heads=8,
    key_length=128,
    value_length=128,
)


def test_head_dim_prefers_key_length():
    assert QWEN3_LIKE.head_dim == 128
    assert QWEN3_LIKE.head_dim_v == 128


def test_head_dim_falls_back_to_hidden_over_heads():
    g = ModelGeometry(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8)
    assert g.head_dim == 128
    assert g.head_dim_v == 128


def test_value_length_defaults_to_key_length():
    g = ModelGeometry(num_layers=2, hidden_size=64, num_attention_heads=2, key_length=48)
    assert g.head_dim == 48
    assert g.head_dim_v == 48


def test_kv_bytes_use_real_head_dim():
    # 28 layers × 8 kv heads × (128 + 128) × 2 bytes
    assert kv_bytes_per_token(QWEN3_LIKE) == 28 * 8 * 256 * 2
    assert kv_bytes_per_token(QWEN3_LIKE, "q8_0") == 28 * 8 * 256
    gib, note = estimate_kv_cache_gib(4096, QWEN3_LIKE, 0.6)
    assert note is None
    assert gib == pytest.approx(28 * 8 * 256 * 2 * 4096 / 1024**3)


def test_kv_bytes_asymmetric_k_v():
    g = ModelGeometry(
        num_layers=10,
        hidden_size=1024,
        num_attention_heads=8,
        num_kv_heads=2,
        key_length=192,
        value_length=128,
    )
    assert kv_bytes_per_token(g) == 10 * 2 * (192 + 128) * 2


def test_max_context_uses_real_head_dim():
    # The old hidden/heads head_dim (64) under-counts KV by 2x here, so the
    # real-geometry context must be no larger than the old one.
    old = ModelGeometry(num_layers=28, hidden_size=1024, num_attention_heads=16, num_kv_heads=8)
    budget = dict(params_b=0.6, quant="Q4_K_M", available_ram_gb=4.0)
    assert max_context_for_budget(geometry=QWEN3_LIKE, **budget) < max_context_for_budget(
        geometry=old, **budget
    )


def _write_gguf(path: Path, with_key_length: bool) -> None:
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), "qwen3")
    w.add_block_count(2)
    w.add_embedding_length(1024)
    w.add_head_count(16)
    w.add_head_count_kv(8)
    w.add_context_length(40960)
    if with_key_length:
        w.add_key_length(128)
        w.add_value_length(128)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_gguf_reader_reads_key_value_length(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p, with_key_length=True)
    info = read_gguf_model_info(p)
    assert info is not None
    assert info.geometry.key_length == 128
    assert info.geometry.value_length == 128
    assert info.geometry.head_dim == 128


def test_gguf_reader_without_key_length_falls_back(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p, with_key_length=False)
    info = read_gguf_model_info(p)
    assert info is not None
    assert info.geometry.key_length is None
    assert info.geometry.head_dim == 1024 // 16
