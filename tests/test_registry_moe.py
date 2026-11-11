"""Registry MoE support: spec fields and the Qwen3-30B-A3B entry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.models.registry import REGISTRY


def test_qwen3_30b_a3b_registered_as_moe():
    m = next(x for x in REGISTRY if x.name == "qwen3-30b-a3b")
    assert m.is_moe
    assert m.experts_total == 128 and m.experts_active == 8
    assert m.repo_id == "Qwen/Qwen3-30B-A3B-GGUF"
    assert m.filename == "Qwen3-30B-A3B-Q4_K_M.gguf"
    assert m.params_b == 30.5
    active = m.params_b * m.experts_active / m.experts_total
    assert abs(active - 1.90625) < 1e-9


def test_dense_specs_are_not_moe():
    dense = [x for x in REGISTRY if x.name != "qwen3-30b-a3b"]
    assert all(not x.is_moe for x in dense)
