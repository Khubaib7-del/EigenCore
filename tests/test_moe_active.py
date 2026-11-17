"""E0: MoE active weights = never-routed tensors + used/count × *_exps tensors."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.fit.estimator import is_expert_tensor, moe_active_weight_bytes
from eigencore.fit.gguf_meta import read_gguf_model_info


def test_expert_tensor_names():
    assert is_expert_tensor("blk.0.ffn_gate_exps.weight")
    assert is_expert_tensor("blk.3.ffn_up_exps.weight")
    assert is_expert_tensor("blk.3.ffn_down_exps.weight")
    assert is_expert_tensor("blk.3.ffn_gate_up_exps.weight")
    assert is_expert_tensor("blk.3.ffn_down_exps.bias")
    # never routed: shared experts, router, attention, dense FFN, embeddings
    assert not is_expert_tensor("blk.0.ffn_gate_shexp.weight")
    assert not is_expert_tensor("blk.0.ffn_gate_inp.weight")
    assert not is_expert_tensor("blk.0.attn_q.weight")
    assert not is_expert_tensor("blk.0.ffn_up.weight")
    assert not is_expert_tensor("token_embd.weight")


def test_moe_active_bytes_formula():
    # 100 dense bytes + 1280 expert bytes at 8/128 → 100 + 80
    assert moe_active_weight_bytes(1380, 1280, 8, 128) == 180


def test_active_is_not_params_ratio():
    # The old rule (total × used/count) drops dense weights; ours keeps them.
    total, experts = 10_000, 8_000
    assert moe_active_weight_bytes(total, experts, 2, 8) == 2_000 + 2_000
    assert moe_active_weight_bytes(total, experts, 2, 8) > total * 2 // 8


def _write(path: Path, arch: str, moe: bool) -> dict[str, int]:
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(1)
    w.add_embedding_length(16)
    w.add_head_count(2)
    w.add_head_count_kv(2)
    w.add_context_length(4096)
    tensors = {
        "token_embd.weight": np.zeros((32, 16), dtype=np.float32),
        "blk.0.attn_q.weight": np.zeros((16, 16), dtype=np.float32),
    }
    if moe:
        w.add_expert_count(4)
        w.add_expert_used_count(1)
        tensors.update(
            {
                "blk.0.ffn_gate_inp.weight": np.zeros((4, 16), dtype=np.float32),
                "blk.0.ffn_gate_shexp.weight": np.zeros((8, 16), dtype=np.float32),
                "blk.0.ffn_gate_exps.weight": np.zeros((4, 8, 16), dtype=np.float32),
                "blk.0.ffn_up_exps.weight": np.zeros((4, 8, 16), dtype=np.float32),
                "blk.0.ffn_down_exps.weight": np.zeros((4, 16, 8), dtype=np.float32),
            }
        )
    else:
        tensors["blk.0.ffn_up.weight"] = np.zeros((32, 16), dtype=np.float32)
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {name: arr.nbytes for name, arr in tensors.items()}


def test_gguf_moe_active_bytes_from_tensors(tmp_path):
    p = tmp_path / "moe.gguf"
    sizes = _write(p, "qwen3moe", moe=True)
    info = read_gguf_model_info(p)
    assert info is not None
    total = sum(sizes.values())
    experts = sum(n for k, n in sizes.items() if "_exps" in k)
    assert info.tensor_bytes == total
    assert info.expert_tensor_bytes == experts
    # dense (embeddings + attention + router + shared expert) in full, experts × 1/4
    assert info.active_weight_bytes == (total - experts) + experts // 4


def test_gguf_dense_active_bytes_is_all_tensors(tmp_path):
    p = tmp_path / "dense.gguf"
    sizes = _write(p, "llama", moe=False)
    info = read_gguf_model_info(p)
    assert info is not None
    assert info.expert_tensor_bytes == 0
    assert info.active_weight_bytes == sum(sizes.values())
