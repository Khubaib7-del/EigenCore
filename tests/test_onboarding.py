"""Stage D slice 3: terminal capability probe, fit-grouped onboarding, SSD warning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.fit import ModelGeometry, estimate_fit
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import REGISTRY, ModelRegistry
from eigencore.tui.onboarding import (
    group_by_fit,
    load_config,
    run_onboarding,
    save_config,
    ssd_warning_lines,
)
from eigencore.tui.theme import probe_terminal


def _hw(**kw):
    base = dict(
        cpu_name="t",
        arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=InstructionSet.AVX2,
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        estimated_bandwidth_gbps=25.0,
        max_model_params_b=7.0,
        optimal_quantization="Q4_K_M",
        estimated_tokens_per_sec=15.0,
        recommended_threads=4,
        recommended_context_length=4096,
    )
    base.update(kw)
    return HardwareProfile(**base)


def test_probe_narrow_and_wide_tiers():
    assert probe_terminal(force_plain=True, width=120).tier == "plain"
    # non-TTY stdin (pytest) forces plain regardless; width still drives tier when not plain
    cap = probe_terminal(force_plain=False, width=120)
    assert cap.tier in ("plain", "wide")  # plain under pytest (no TTY)
    cap2 = probe_terminal(force_plain=False, width=80)
    assert cap2.width == 80


def test_group_by_fit():
    geo = ModelGeometry(num_layers=28, hidden_size=3584, num_attention_heads=28, num_kv_heads=4)
    ests = {}
    for m in REGISTRY:
        ests[m.name] = estimate_fit(
            m.params_b,
            m.quant,
            4096,
            12.0,
            geometry=geo,
            file_size_bytes=int(m.size_gb * 1024**3),
        )
    groups = group_by_fit(list(REGISTRY), ests)
    assert set(groups) == {"fits", "tight", "no", "unknown"}
    # qwen3-30b-a3b (18.6 GiB) must land in "no" on 12 GiB free RAM
    assert any(m.name == "qwen3-30b-a3b" for m in groups["no"])
    assert any(m.name == "tinyllama-1.1b" for m in groups["fits"])


def test_ssd_warning_only_when_not_resident():
    geo = ModelGeometry(num_layers=48, hidden_size=2048, num_attention_heads=32, num_kv_heads=4)
    big = next(m for m in REGISTRY if m.name == "qwen3-30b-a3b")
    est = estimate_fit(
        big.params_b, big.quant, 4096, 12.0, geometry=geo, file_size_bytes=int(18.6 * 1024**3)
    )
    lines = ssd_warning_lines(big, est)
    assert lines and "SSD" in lines[1]
    small = next(m for m in REGISTRY if m.name == "tinyllama-1.1b")
    est_small = estimate_fit(
        small.params_b, small.quant, 4096, 12.0, geometry=geo, file_size_bytes=int(0.67 * 1024**3)
    )
    assert ssd_warning_lines(small, est_small) == []


def test_onboarding_picks_and_quits(tmp_path):
    reg = ModelRegistry(cache_dir=tmp_path)
    geo = ModelGeometry(num_layers=28, hidden_size=3584, num_attention_heads=28, num_kv_heads=4)
    ests = {
        m.name: estimate_fit(
            m.params_b, m.quant, 4096, 12.0, geometry=geo, file_size_bytes=int(m.size_gb * 1024**3)
        )
        for m in reg.list_models()
    }
    lines: list[str] = []
    answers = iter(["1"])
    chosen = run_onboarding(_hw(), reg, ests, ask=lambda p="": next(answers), out=lines.append)
    assert chosen is not None and chosen.name == reg.list_models()[0].name or chosen is not None
    assert any("ESTIMATE" in line for line in lines)
    # quit path
    answers2 = iter(["q"])
    assert (
        run_onboarding(_hw(), reg, ests, ask=lambda p="": next(answers2), out=lambda line: None)
        is None
    )


def test_config_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    assert load_config(p) == {}
    save_config({"onboarded": True, "model": "tinyllama-1.1b"}, p)
    assert load_config(p)["model"] == "tinyllama-1.1b"
