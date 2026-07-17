"""
Hardware Abstraction Layer — auto-detects CPU capabilities, RAM, and memory bandwidth
to determine optimal model configuration without manual user input.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from enum import Flag, auto

import psutil


class InstructionSet(Flag):
    NONE = 0
    SSE2 = auto()
    SSE4_1 = auto()
    SSE4_2 = auto()
    AVX = auto()
    AVX2 = auto()
    AVX512F = auto()
    AVX512BW = auto()
    AVX512VNNI = auto()
    FMA = auto()
    F16C = auto()
    AMX_TILE = auto()
    AMX_INT8 = auto()
    AMX_BF16 = auto()
    NEON = auto()
    SVE = auto()


@dataclass(frozen=True)
class HardwareProfile:
    cpu_name: str
    arch: str
    physical_cores: int
    logical_cores: int
    instruction_sets: InstructionSet
    total_ram_gb: float
    available_ram_gb: float
    estimated_bandwidth_gbps: float
    max_model_params_b: float
    optimal_quantization: str
    estimated_tokens_per_sec: float
    recommended_threads: int
    recommended_context_length: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        isa_names = [
            f.name for f in InstructionSet if f in self.instruction_sets and f.name != "NONE"
        ]
        lines = [
            f"CPU: {self.cpu_name} ({self.arch})",
            f"Cores: {self.physical_cores}P / {self.logical_cores}L",
            f"ISA: {', '.join(isa_names) if isa_names else 'baseline'}",
            f"RAM: {self.available_ram_gb:.1f} / {self.total_ram_gb:.1f} GB available",
            f"Bandwidth: ~{self.estimated_bandwidth_gbps:.1f} GB/s",
            "",
            f"Max model: ~{self.max_model_params_b:.1f}B params at {self.optimal_quantization}",
            f"Est. speed: ~{self.estimated_tokens_per_sec:.0f} tok/s",
            f"Threads: {self.recommended_threads}",
            f"Context: {self.recommended_context_length} tokens",
        ]
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  * {note}")
        return "\n".join(lines)


def _detect_instruction_sets_x86() -> InstructionSet:
    """Detect x86 SIMD capabilities via CPUID."""
    isa = InstructionSet.NONE

    if platform.system() == "Windows":
        try:
            from eigencore.hal._cpuid_win import detect_isa_windows

            return detect_isa_windows()
        except ImportError:
            pass

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()

        flag_map = {
            "sse2": InstructionSet.SSE2,
            "sse4_1": InstructionSet.SSE4_1,
            "sse4_2": InstructionSet.SSE4_2,
            "avx": InstructionSet.AVX,
            "avx2": InstructionSet.AVX2,
            "avx512f": InstructionSet.AVX512F,
            "avx512bw": InstructionSet.AVX512BW,
            "avx512_vnni": InstructionSet.AVX512VNNI,
            "fma": InstructionSet.FMA,
            "f16c": InstructionSet.F16C,
            "amx_tile": InstructionSet.AMX_TILE,
            "amx_int8": InstructionSet.AMX_INT8,
            "amx_bf16": InstructionSet.AMX_BF16,
        }
        for flag, isa_val in flag_map.items():
            if flag in cpuinfo:
                isa |= isa_val
    except FileNotFoundError:
        pass

    return isa


def _detect_instruction_sets_arm() -> InstructionSet:
    """Detect ARM SIMD capabilities."""
    isa = InstructionSet.NONE

    if platform.machine().lower() in ("aarch64", "arm64"):
        isa |= InstructionSet.NEON

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        if "sve" in cpuinfo:
            isa |= InstructionSet.SVE
    except FileNotFoundError:
        pass

    return isa


def _measure_bandwidth() -> float:
    """Estimate memory bandwidth via timed memcpy benchmark (GB/s)."""
    try:
        block_size = 64 * 1024 * 1024  # 64 MB
        src = bytearray(block_size)
        iterations = 5

        # warmup
        _ = bytes(src)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = bytearray(src)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        median_time = sorted(times)[len(times) // 2]
        bandwidth_gbps = (block_size / (1024**3)) / median_time
        # memcpy measures roughly half of peak bandwidth (read + write)
        return bandwidth_gbps * 2
    except Exception:
        return 20.0  # conservative fallback


def _get_cpu_name() -> str:
    """Get human-readable CPU name."""
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return name.strip()
        except Exception:
            pass

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line.lower():
                    return line.split(":")[1].strip()
    except FileNotFoundError:
        pass

    return platform.processor() or "Unknown CPU"


def _compute_model_limits(
    available_ram_gb: float,
    bandwidth_gbps: float,
) -> tuple[float, str, float, int, int]:
    """
    Given available RAM and bandwidth, compute:
    - max model size in billions of parameters
    - optimal quantization level
    - estimated tokens per second
    - recommended thread count
    - recommended context length

    Model size formula: params_B × bytes_per_param = model_memory_GB
    We reserve ~2GB for OS/KV-cache overhead, then fit the largest model possible.
    """
    usable_ram_gb = max(available_ram_gb - 2.0, 1.0)

    quant_options = [
        ("Q4_K_M", 0.55),  # ~0.55 bytes per param (4-bit with some FP16 layers)
        ("Q5_K_M", 0.68),  # ~0.68 bytes per param
        ("Q8_0", 1.05),  # ~1.05 bytes per param
        ("F16", 2.0),  # 2 bytes per param
    ]

    best_params_b = 0.0
    best_quant = "Q4_K_M"

    for quant_name, bytes_per_param in quant_options:
        max_params = (usable_ram_gb * 1024**3) / (bytes_per_param * 1e9)
        if max_params > best_params_b:
            best_params_b = max_params
            best_quant = quant_name

    # cap at realistic model sizes available in GGUF
    best_params_b = min(best_params_b, 70.0)

    # estimate tokens/sec: bandwidth / model_size_in_memory
    model_size_gb = best_params_b * 0.55  # assuming best quant (Q4)
    if model_size_gb > 0:
        est_tps = bandwidth_gbps / model_size_gb
        est_tps = min(est_tps, 100.0)  # cap at reasonable maximum
    else:
        est_tps = 0.0

    # recommended threads: physical cores (hyperthreads don't help much for inference)
    recommended_threads = psutil.cpu_count(logical=False) or 4

    # context length: depends on available RAM after model
    remaining_ram_gb = usable_ram_gb - model_size_gb
    # KV cache: ~0.5 MB per 1K context tokens per billion params at Q4
    if best_params_b > 0 and remaining_ram_gb > 0:
        kv_per_1k = 0.0005 * best_params_b  # GB per 1K tokens
        max_ctx = int((remaining_ram_gb / kv_per_1k) * 1000) if kv_per_1k > 0 else 2048
        recommended_ctx = min(max_ctx, 8192)  # cap at 8K for CPU
        recommended_ctx = max(recommended_ctx, 512)  # minimum usable
    else:
        recommended_ctx = 2048

    return best_params_b, best_quant, est_tps, recommended_threads, recommended_ctx


def profile_hardware() -> HardwareProfile:
    """Run full hardware profiling and return a HardwareProfile with recommendations."""
    arch = platform.machine().lower()
    is_arm = arch in ("aarch64", "arm64")
    is_x86 = arch in ("x86_64", "amd64", "x86")

    cpu_name = _get_cpu_name()

    if is_x86:
        isa = _detect_instruction_sets_x86()
    elif is_arm:
        isa = _detect_instruction_sets_arm()
    else:
        isa = InstructionSet.NONE

    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024**3)
    available_ram_gb = mem.available / (1024**3)

    bandwidth = _measure_bandwidth()

    max_params, quant, est_tps, threads, ctx = _compute_model_limits(available_ram_gb, bandwidth)

    notes = []
    if not is_x86 and not is_arm:
        notes.append(f"Unrecognized architecture: {arch}. Using conservative defaults.")
    if InstructionSet.AVX2 not in isa and is_x86:
        notes.append("No AVX2 detected — inference will be significantly slower.")
    if available_ram_gb < 4.0:
        notes.append("Very low available RAM. Close other applications for better performance.")
    if InstructionSet.AVX512F in isa:
        notes.append("AVX-512 detected — llama.cpp will use optimized 512-bit kernels.")
    if InstructionSet.AMX_INT8 in isa:
        notes.append("AMX detected — hardware matrix acceleration available.")

    return HardwareProfile(
        cpu_name=cpu_name,
        arch=arch,
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
        instruction_sets=isa,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        estimated_bandwidth_gbps=bandwidth,
        max_model_params_b=max_params,
        optimal_quantization=quant,
        estimated_tokens_per_sec=est_tps,
        recommended_threads=threads,
        recommended_context_length=ctx,
        notes=notes,
    )
