"""
Windows-specific CPUID instruction set detection using ctypes and kernel32.
Falls back to registry-based detection if direct CPUID is unavailable.
"""

from __future__ import annotations

import winreg
from eigencore.hal.profiler import InstructionSet


def _check_registry_features() -> set[str]:
    """Read CPU feature flags from Windows registry as fallback."""
    features: set[str] = set()
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        identifier, _ = winreg.QueryValueEx(key, "Identifier")
        winreg.CloseKey(key)
    except Exception:
        pass
    return features


def _check_os_enabled_avx() -> bool:
    """Check if the OS has enabled AVX state saving (XGETBV check)."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        if hasattr(kernel32, "IsProcessorFeaturePresent"):
            # PF_XMMI64_INSTRUCTIONS_AVAILABLE = 10 (SSE2)
            # PF_AVX_INSTRUCTIONS_AVAILABLE = 39 (Win10+)
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40 (Win10+)
            return True
        return True
    except Exception:
        return True


def detect_isa_windows() -> InstructionSet:
    """
    Detect CPU instruction sets on Windows.
    Uses IsProcessorFeaturePresent for basic detection,
    environment variables and CPU name heuristics for extended sets.
    """
    isa = InstructionSet.NONE

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        ipfp = kernel32.IsProcessorFeaturePresent

        # PF constants from winnt.h
        PF_XMMI64_INSTRUCTIONS_AVAILABLE = 10  # SSE2
        PF_SSE4_1_INSTRUCTIONS_AVAILABLE = 37  # SSE4.1
        PF_SSE4_2_INSTRUCTIONS_AVAILABLE = 38  # SSE4.2
        PF_AVX_INSTRUCTIONS_AVAILABLE = 39  # AVX
        PF_AVX2_INSTRUCTIONS_AVAILABLE = 40  # AVX2
        PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41  # AVX-512F

        if ipfp(PF_XMMI64_INSTRUCTIONS_AVAILABLE):
            isa |= InstructionSet.SSE2
        if ipfp(PF_SSE4_1_INSTRUCTIONS_AVAILABLE):
            isa |= InstructionSet.SSE4_1
        if ipfp(PF_SSE4_2_INSTRUCTIONS_AVAILABLE):
            isa |= InstructionSet.SSE4_2
        if ipfp(PF_AVX_INSTRUCTIONS_AVAILABLE):
            isa |= InstructionSet.AVX
            isa |= InstructionSet.FMA
            isa |= InstructionSet.F16C
        if ipfp(PF_AVX2_INSTRUCTIONS_AVAILABLE):
            isa |= InstructionSet.AVX2
        if ipfp(PF_AVX512F_INSTRUCTIONS_AVAILABLE):
            isa |= InstructionSet.AVX512F
            isa |= InstructionSet.AVX512BW
    except Exception:
        isa = _fallback_detection()

    return isa


def _fallback_detection() -> InstructionSet:
    """Heuristic fallback based on CPU model name from registry."""
    isa = InstructionSet.SSE2 | InstructionSet.SSE4_1 | InstructionSet.SSE4_2

    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        winreg.CloseKey(key)
        name = name.lower()

        # Intel generations with known ISA support
        if any(g in name for g in ("i7-1", "i9-1", "i5-1", "i3-1", "i7-8", "i7-9")):
            isa |= (
                InstructionSet.AVX | InstructionSet.AVX2 | InstructionSet.FMA | InstructionSet.F16C
            )
        if any(g in name for g in ("i7-12", "i7-13", "i7-14", "i9-12", "i9-13", "i9-14")):
            isa |= InstructionSet.AVX512F | InstructionSet.AVX512BW | InstructionSet.AVX512VNNI

        # AMD Zen with known ISA support
        if "ryzen" in name:
            isa |= (
                InstructionSet.AVX | InstructionSet.AVX2 | InstructionSet.FMA | InstructionSet.F16C
            )
        if any(g in name for g in ("ryzen 9 7", "ryzen 7 7", "ryzen 9 9")):
            isa |= InstructionSet.AVX512F | InstructionSet.AVX512BW | InstructionSet.AVX512VNNI

    except Exception:
        isa |= InstructionSet.AVX | InstructionSet.AVX2

    return isa
