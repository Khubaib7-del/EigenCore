"""Fail the build if the bundled llama.cpp uses AVX-512.

The Windows app ships one llama.cpp build for every PC. pip compiles
llama-cpp-python from source on the GitHub runner, and with llama.cpp's
default GGML_NATIVE=ON the DLLs are tuned to the runner's CPU, which often has
AVX-512. Most laptops do not, so loading a model crashed with Windows error
0xC000001D (illegal instruction) in 0.4.9. The workflows now build with
GGML_NATIVE=OFF and an AVX2 baseline; this check makes sure of it.

Usage: python check_cpu_baseline.py <dir-or-file> [...]
Needs: pip install pefile capstone
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_EVEX_ONLY = re.compile(r"\bzmm\d+|\b[xy]mm(?:1[6-9]|2\d|3[01])\b|\{k[1-7]\}|\bk[0-7]\b")


def avx512_instructions(code: bytes, base: int = 0, limit: int = 5) -> list[str]:
    """Up to `limit` AVX-512 instructions in `code`.

    AVX-512 is what uses zmm registers, xmm/ymm 16-31, or the k0-k7 opmasks.
    """
    import capstone

    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.skipdata = True
    hits: list[str] = []
    for _addr, _size, mnemonic, ops in md.disasm_lite(code, base):
        if _EVEX_ONLY.search(ops):
            hits.append(f"{_addr:#x}: {mnemonic} {ops}")
            if len(hits) >= limit:
                break
    return hits


def text_sections(path: Path):
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    for sec in pe.sections:
        if sec.Characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
            yield sec.VirtualAddress + pe.OPTIONAL_HEADER.ImageBase, sec.get_data()


def main(argv: list[str]) -> int:
    files: list[Path] = []
    for arg in argv:
        p = Path(arg)
        files += sorted(p.rglob("*.dll")) if p.is_dir() else [p]
    files = [f for f in files if "ggml" in f.name.lower() or "llama" in f.name.lower()]
    if not files:
        print("no ggml/llama DLLs found", file=sys.stderr)
        return 2
    bad = 0
    for f in files:
        hits: list[str] = []
        for base, code in text_sections(f):
            hits += avx512_instructions(code, base)
        print(f"{'FAIL' if hits else 'ok  '} {f.name}")
        for h in hits:
            print(f"       {h}")
        bad += bool(hits)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
