"""CLI UX fixes: sharded downloads, delete, registry filenames, clean bail."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.models.registry import REGISTRY, ModelRegistry


def test_registry_known_files_exist_on_hf_contract():
    # filenames must match the HF naming contract we verified via the API
    by_name = {m.name: m for m in REGISTRY}
    assert by_name["qwen2.5-7b"].filename == "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    assert by_name["qwen2.5-1.5b"].filename == "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    assert by_name["qwen2.5-3b"].filename == "qwen2.5-3b-instruct-q4_k_m.gguf"
    assert by_name["qwen2.5-coder-1.5b"].filename == "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    assert by_name["phi-3-mini"].repo_id == "bartowski/Phi-3-mini-4k-instruct-GGUF"
    assert by_name["phi-3-mini"].filename == "Phi-3-mini-4k-instruct-Q4_K_M.gguf"
    assert by_name["qwen3-30b-a3b"].filename == "Qwen3-30B-A3B-Q4_K_M.gguf"


def test_shard_prefix_detection():
    reg = ModelRegistry(cache_dir=Path("/tmp/x"))
    m7 = next(m for m in REGISTRY if m.name == "qwen2.5-7b")
    assert reg.shard_prefix(m7) == "qwen2.5-7b-instruct-q4_k_m"
    assert reg._shard_count(m7) == 2
    m1 = next(m for m in REGISTRY if m.name == "tinyllama-1.1b")
    assert reg.shard_prefix(m1) is None


def test_delete_removes_files_and_is_idempotent(tmp_path):
    reg = ModelRegistry(cache_dir=tmp_path)
    m = next(m for m in REGISTRY if m.name == "tinyllama-1.1b")
    p = m.local_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    assert reg.is_downloaded(m)
    assert reg.delete(m) is True
    assert not reg.is_downloaded(m)
    assert reg.delete(m) is False  # idempotent


def test_delete_sharded_removes_all_shards(tmp_path):
    reg = ModelRegistry(cache_dir=tmp_path)
    m = next(m for m in REGISTRY if m.name == "qwen2.5-7b")
    d = m.local_path(tmp_path).parent
    d.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        (d / f"qwen2.5-7b-instruct-q4_k_m-{i:05d}-of-00002.gguf").write_bytes(b"x")
    assert reg.is_downloaded(m)
    assert reg.delete(m) is True
    assert not reg.is_downloaded(m)
    assert not list(d.glob("*.gguf"))
