"""
GGUF metadata reader - architecture-aware model geometry.

Reads general.architecture and the per-architecture keys
({arch}.block_count, {arch}.embedding_length, {arch}.attention.head_count,
{arch}.attention.head_count_kv, {arch}.context_length) from a GGUF file.
Prefers the pure-Python gguf package; falls back to llama_cpp.Llama.metadata
when loadable that way. Returns None when neither is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eigencore.fit.estimator import ModelGeometry


@dataclass
class GGUFModelInfo:
    """Ground-truth facts read from a GGUF file."""

    architecture: str | None
    geometry: ModelGeometry
    context_length: int | None
    file_size_bytes: int | None
    param_count: int | None


def _field_value(field) -> object | None:
    try:
        part = field.parts[field.data[0]]
        if part is None:
            return None
        return part.tolist()
    except Exception:
        return None


def _from_gguf_package(path: Path) -> GGUFModelInfo | None:
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    try:
        reader = GGUFReader(str(path), mode="no")
    except Exception:
        return None
    fields = {f.name: f for f in reader.fields.values()}

    def get(*names: str):
        for n in names:
            f = fields.get(n)
            if f is not None:
                v = _field_value(f)
                if v is not None:
                    return v
        return None

    arch = get("general.architecture")
    architecture = str(arch) if arch is not None else None
    prefix = f"{architecture}." if architecture else "llama."

    def arch_int(*names: str) -> int | None:
        for n in names:
            v = get(prefix + n, n)
            if v is not None:
                return int(v)
        return None

    geometry = ModelGeometry(
        num_layers=arch_int("block_count"),
        hidden_size=arch_int("embedding_length"),
        num_attention_heads=arch_int("attention.head_count"),
        num_kv_heads=arch_int("attention.head_count_kv"),
    )
    ctx = arch_int("context_length") or arch_int("attention.context_length")
    params = get("general.parameter_count")
    file_size = path.stat().st_size if path.exists() else None
    return GGUFModelInfo(
        architecture=architecture,
        geometry=geometry,
        context_length=ctx,
        file_size_bytes=file_size,
        param_count=int(params) if params is not None else None,
    )


def _from_llama_metadata(path: Path) -> GGUFModelInfo | None:
    try:
        from llama_cpp import Llama
    except ImportError:
        return None
    try:
        llm = Llama(model_path=str(path), n_ctx=256, verbose=False)
        meta = dict(llm.metadata or {})
    except Exception:
        return None
    arch = meta.get("general.architecture")
    prefix = f"{arch}." if arch else "llama."

    def m(name: str) -> int | None:
        v = meta.get(prefix + name, meta.get(name))
        return int(v) if v is not None else None

    geometry = ModelGeometry(
        num_layers=m("block_count"),
        hidden_size=m("embedding_length"),
        num_attention_heads=m("attention.head_count"),
        num_kv_heads=m("attention.head_count_kv"),
    )
    return GGUFModelInfo(
        architecture=arch,
        geometry=geometry,
        context_length=m("context_length"),
        file_size_bytes=path.stat().st_size if path.exists() else None,
        param_count=None,
    )


def read_gguf_model_info(path: str | Path) -> GGUFModelInfo | None:
    """Best-effort GGUF metadata read; None when the file cannot be parsed."""
    path = Path(path)
    if not path.exists():
        return None
    return _from_gguf_package(path) or _from_llama_metadata(path)
