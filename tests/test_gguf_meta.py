"""GGUF metadata is read with the pure-Python gguf package (no model load)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.fit.gguf_meta import read_gguf_model_info


def test_reads_geometry_with_gguf_package(tmp_path, monkeypatch):
    gguf = pytest.importorskip("gguf")
    # Make sure the llama_cpp fallback cannot be what answers.
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    p = tmp_path / "m.gguf"
    w = gguf.GGUFWriter(str(p), "llama")
    w.add_block_count(2)
    w.add_embedding_length(64)
    w.add_head_count(4)
    w.add_head_count_kv(2)
    w.add_context_length(4096)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    info = read_gguf_model_info(p)
    assert info is not None
    assert info.architecture == "llama"
    assert info.geometry.num_layers == 2
    assert info.geometry.hidden_size == 64
    assert info.geometry.num_attention_heads == 4
    assert info.geometry.num_kv_heads == 2
    assert info.context_length == 4096
