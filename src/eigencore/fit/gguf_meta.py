"""
GGUF metadata reader - architecture-aware model geometry.

Reads general.architecture and the per-architecture keys
({arch}.block_count, {arch}.embedding_length, {arch}.attention.head_count,
{arch}.attention.head_count_kv, {arch}.attention.key_length/value_length,
{arch}.context_length) from a GGUF file.
Prefers the pure-Python gguf package; falls back to llama_cpp.Llama.metadata
when loadable that way. Returns None when neither is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from eigencore.fit.estimator import ModelGeometry, is_expert_tensor, moe_active_weight_bytes

#: llama.cpp's default full-attention interval for archs whose GGUF may omit
#: ``{arch}.full_attention_interval`` (src/models/qwen3next.cpp: 4).
DEFAULT_FULL_ATTENTION_INTERVAL = {"qwen3next": 4, "qwen35": 4, "qwen35moe": 4}


@dataclass
class GGUFModelInfo:
    """Ground-truth facts read from a GGUF file."""

    architecture: str | None
    geometry: ModelGeometry
    context_length: int | None
    file_size_bytes: int | None
    param_count: int | None
    tensor_bytes: int | None = None  # sum of all tensor data bytes
    expert_tensor_bytes: int | None = None  # sum of routed-expert (*_exps) tensor bytes

    @property
    def active_weight_bytes(self) -> int | None:
        """Weight bytes read per decode token: all tensors for dense models;
        never-routed tensors + used/count × expert tensors for MoE. None when
        tensor sizes are unknown."""
        if self.tensor_bytes is None:
            return None
        g = self.geometry
        if g.is_moe and self.expert_tensor_bytes:
            return moe_active_weight_bytes(
                self.tensor_bytes, self.expert_tensor_bytes, g.expert_used_count, g.expert_count
            )
        return self.tensor_bytes


def _as_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, (list, tuple)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def geometry_from_kv(kv: dict[str, Any]) -> ModelGeometry:
    """ModelGeometry from GGUF metadata, hybrid-aware.

    * ``{arch}.attention.head_count_kv`` given per layer (Granite 4.0-H,
      Nemotron-H, LFM2): layers with 0 KV heads are Mamba / conv layers and
      keep no KV cache, so ``attention_layers`` counts only the non-zero ones.
    * ``{arch}.attention.recurrent_layers`` (per-layer flags): the
      non-recurrent layers are the attention layers.
    * ``{arch}.full_attention_interval`` (Qwen3-Next, Qwen3.5): only every
      n-th layer is full attention; the rest are linear attention with a fixed
      size state. llama.cpp defaults it to 4 for these archs when absent.
    """
    arch = kv.get("general.architecture")
    p = f"{arch}." if arch else "llama."

    def get(name: str) -> Any:
        return kv.get(p + name, kv.get(name))

    def num(name: str) -> Optional[int]:
        v = get(name)
        if isinstance(v, list):
            vals = [int(x) for x in v if x]
            return max(vals) if vals else None
        return _as_int(v)

    layers = num("block_count")
    attention_layers: Optional[int] = None
    kv_per_layer = get("attention.head_count_kv")
    recurrent = get("attention.recurrent_layers")
    if isinstance(kv_per_layer, list) and kv_per_layer:
        attention_layers = sum(1 for x in kv_per_layer if x)
    elif isinstance(recurrent, list) and recurrent:
        attention_layers = sum(1 for x in recurrent if not x)
    interval = _as_int(get("full_attention_interval")) or DEFAULT_FULL_ATTENTION_INTERVAL.get(
        str(arch)
    )
    if attention_layers is None and interval and interval > 1 and layers:
        attention_layers = layers // interval
    kv_elements = _per_layer_kv_elements(get, kv_per_layer)
    return ModelGeometry(
        num_layers=layers,
        hidden_size=num("embedding_length"),
        num_attention_heads=num("attention.head_count"),
        num_kv_heads=num("attention.head_count_kv"),
        expert_count=num("expert_count"),
        expert_used_count=num("expert_used_count"),
        key_length=num("attention.key_length"),
        value_length=num("attention.value_length"),
        attention_layers=attention_layers,
        kv_elements_per_token=kv_elements,
    )


def _per_layer_kv_elements(get, kv_per_layer: Any) -> Optional[int]:
    """Σ over layers of kv_heads × (K dim + V dim) when KV heads are given per layer.

    Sliding-window layers use ``attention.key_length_swa`` / ``value_length_swa``
    when present (Gemma 4). The window itself is not credited: every layer is
    counted at the full context, so the estimate errs high for SWA models.
    """
    if not isinstance(kv_per_layer, list) or not kv_per_layer:
        return None
    k = _as_int(get("attention.key_length"))
    v = _as_int(get("attention.value_length")) or k
    if not k:
        hidden = _as_int(get("embedding_length"))
        heads = get("attention.head_count")
        heads = max(heads) if isinstance(heads, list) and heads else _as_int(heads)
        if not hidden or not heads:
            return None
        k = v = hidden // heads
    k_swa = _as_int(get("attention.key_length_swa")) or k
    v_swa = _as_int(get("attention.value_length_swa")) or k_swa
    pattern = get("attention.sliding_window_pattern")
    pattern = pattern if isinstance(pattern, list) and len(pattern) == len(kv_per_layer) else None
    total = 0
    for i, heads_l in enumerate(kv_per_layer):
        if not heads_l:
            continue
        swa = bool(pattern[i]) if pattern else False
        total += int(heads_l) * ((k_swa + v_swa) if swa else (k + v))
    return total or None


def _field_value(field) -> object | None:
    """Scalar value of a GGUFReader field: str for strings, int/float/bool otherwise.

    Small numeric arrays (per-layer values such as Granite 4.0-H's
    ``head_count_kv``) come back as lists; string arrays (the tokenizer
    vocab) and long arrays return None.
    """
    try:
        from gguf import GGUFValueType

        if not field.types:
            return None
        if field.types[0] == GGUFValueType.ARRAY:
            if len(field.types) < 2 or field.types[1] == GGUFValueType.STRING:
                return None
            if len(field.data) > 4096:
                return None
            return [field.parts[i].tolist()[0] for i in field.data]
        part = field.parts[field.data[0]]
        if part is None:
            return None
        if field.types[0] == GGUFValueType.STRING:
            return bytes(part).decode("utf-8")
        value = part.tolist()
        return value[0] if isinstance(value, list) and len(value) == 1 else value
    except Exception:
        return None


def _from_gguf_package(path: Path) -> GGUFModelInfo | None:
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    try:
        reader = GGUFReader(str(path), mode="r")
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
    kv = {name: _field_value(f) for name, f in fields.items() if name.startswith(prefix)}
    kv["general.architecture"] = architecture
    geometry = geometry_from_kv(kv)
    ctx = _as_int(kv.get(prefix + "context_length"))
    params = get("general.parameter_count")
    file_size = path.stat().st_size if path.exists() else None
    tensor_bytes: int | None = None
    expert_bytes: int | None = None
    try:
        tensor_bytes = expert_bytes = 0
        for t in reader.tensors:
            n = int(t.n_bytes)
            tensor_bytes += n
            if is_expert_tensor(t.name):
                expert_bytes += n
    except Exception:
        tensor_bytes = expert_bytes = None
    return GGUFModelInfo(
        architecture=architecture,
        geometry=geometry,
        context_length=ctx,
        file_size_bytes=file_size,
        param_count=int(params) if params is not None else None,
        tensor_bytes=tensor_bytes,
        expert_tensor_bytes=expert_bytes,
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
    geometry = geometry_from_kv(meta)

    def m(name: str) -> int | None:
        return _as_int(meta.get(prefix + name, meta.get(name)))

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
