"""Stage C slice 3: GPU/VRAM-aware fit and layer offload."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.fit import ModelGeometry, estimate_fit, gpu_layers_for_vram
from eigencore.hal.profiler import HardwareProfile, InstructionSet

GEO = ModelGeometry(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=32)


def _hw(**kw):
    base = dict(
        cpu_name="Test CPU",
        arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=InstructionSet.AVX2 | InstructionSet.SSE2,
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


def test_gpu_detection_defaults_safe():
    hw = _hw()
    assert hw.gpu_name is None and hw.total_vram_gb == 0.0
    hw_gpu = _hw(gpu_name="RTX 3060", total_vram_gb=12.0, available_vram_gb=10.0)
    assert "RTX 3060" in hw_gpu.summary() and "VRAM" in hw_gpu.summary()
    assert "CPU-only" in hw.summary()


def test_gpu_layers_for_vram_math():
    # 32 layers, 100 MiB/layer, 8 GiB VRAM, 1 GiB reserve, no KV → (8-1)/0.097 ≈ 72 → capped at 32
    assert gpu_layers_for_vram(32, 100 * 1024**2, 8.0) == 32
    # 2 GiB VRAM, 100 MiB/layer, 1 GiB reserve → (2-1) GiB / 100 MiB = 10 layers
    assert gpu_layers_for_vram(32, 100 * 1024**2, 2.0) == 10
    assert gpu_layers_for_vram(32, 100 * 1024**2, 0.5) == 0
    assert gpu_layers_for_vram(0, 100, 8.0) == 0


def test_gpu_layers_for_vram_subtracts_kv():
    # 8 GiB VRAM, 1 GiB reserve, 6 GiB KV → 1 GiB left → 10 layers of 100 MiB
    assert gpu_layers_for_vram(32, 100 * 1024**2, 8.0, kv_cache_gb=6.0) == 10


def test_estimate_fit_with_vram_reports_layers():
    # 3.85 GiB file / 32 layers ≈ 0.12 GiB/layer; 6 GiB VRAM − 1 reserve − KV(4096)
    # KV ≈ 2 GiB → ~3 GiB left → 24 layers
    est = estimate_fit(
        7.0,
        "Q4_K_M",
        4096,
        12.0,
        geometry=GEO,
        file_size_bytes=int(3.85 * 1024**3),
        bandwidth_gbps=25.0,
        available_vram_gb=6.0,
    )
    assert 0 < est.gpu_layers < 32
    assert est.gpu_weights_gb > 0
    assert any("GPU offload" in n for n in est.notes)


def test_estimate_fit_no_vram_stays_cpu():
    est = estimate_fit(7.0, "Q4_K_M", 4096, 12.0, geometry=GEO, bandwidth_gbps=25.0)
    assert est.gpu_layers is None and est.gpu_weights_gb == 0.0
