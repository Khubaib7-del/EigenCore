"""
Read a GGUF header over HTTP Range requests, before downloading the model.

The GGUF header (magic, version, tensor count, key/value metadata, tensor
descriptors) sits at the start of the file. A few MB of ``Range`` reads give
everything the fit estimator needs: layer count, attention / KV heads, K/V
head dims, MoE expert counts, context length and, from the tensor
descriptors, how many bytes are routed-expert weights. So a fit verdict can
be computed for a 20 GB file without downloading it.

Pure standard library (``urllib`` + ``struct``); no ``gguf`` package needed.
Every network call goes through one ``fetch(url, start, end)`` callable so
tests can replace it.
"""

from __future__ import annotations

import struct
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from eigencore.fit.estimator import ModelGeometry, is_expert_tensor, moe_active_weight_bytes
from eigencore.fit.gguf_meta import (  # noqa: F401  (re-exported)
    DEFAULT_FULL_ATTENTION_INTERVAL,
    GGUFModelInfo,
    _as_int,
    geometry_from_kv,
)

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{file}"
GGUF_MAGIC = b"GGUF"
DEFAULT_CHUNK = 4 << 20  # one 4 MiB Range read covers most headers
MAX_HEADER_BYTES = 64 << 20  # give up rather than download a model by accident
MAX_ARRAY_KEEP = 4096  # keep small numeric arrays (per-layer values); skip vocabularies
DEFAULT_ALIGNMENT = 32
USER_AGENT = "eigencore (gguf header probe)"

Fetch = Callable[[str, int, int], bytes]


# GGUF value types
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
_STRING, _ARRAY = 8, 9


class GGUFHeaderError(ValueError):
    """The bytes are not a GGUF header we can read."""


def http_range(url: str, start: int, end: int, timeout: float = 20.0) -> bytes:
    """Bytes ``start..end`` (inclusive) of ``url`` via an HTTP Range request."""
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(end - start + 1)
        # a server that ignores Range answers 200 with the whole file; we read
        # only what we asked for and stop
        return data


def hf_file_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return HF_RESOLVE.format(repo=repo_id, rev=revision, file=filename)


class _Reader:
    """Sequential reader over a growing prefix of a remote file."""

    def __init__(self, url: str, fetch: Fetch, chunk: int, max_bytes: int) -> None:
        self.url, self.fetch, self.chunk, self.max_bytes = url, fetch, chunk, max_bytes
        self.data = bytearray()
        self.pos = 0
        self.requests = 0

    def _need(self, n: int) -> None:
        while self.pos + n > len(self.data):
            start = len(self.data)
            if start >= self.max_bytes:
                raise GGUFHeaderError(f"GGUF header larger than {self.max_bytes} bytes")
            got = self.fetch(self.url, start, start + self.chunk - 1)
            self.requests += 1
            if not got:
                raise GGUFHeaderError("file ended inside the GGUF header")
            self.data += got[: self.chunk]

    def read(self, n: int) -> bytes:
        self._need(n)
        b = bytes(self.data[self.pos : self.pos + n])
        self.pos += n
        return b

    def skip(self, n: int) -> None:
        self._need(n)
        self.pos += n

    def unpack(self, fmt: str) -> Any:
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.read(self.unpack("Q")).decode("utf-8", "replace")


def _value(r: _Reader, vtype: int) -> Any:
    if vtype in _SCALARS:
        return r.unpack(_SCALARS[vtype])
    if vtype == _STRING:
        return r.string()
    if vtype == _ARRAY:
        etype = r.unpack("I")
        n = r.unpack("Q")
        if etype in _SCALARS:
            size = struct.calcsize(_SCALARS[etype])
            if n <= MAX_ARRAY_KEEP:
                raw = r.read(size * n)
                return list(struct.unpack("<" + _SCALARS[etype] * n, raw))
            r.skip(size * n)
            return None
        for _ in range(n):  # string arrays (vocab, merges): walk, don't keep
            _value(r, etype)
        return None
    raise GGUFHeaderError(f"unknown GGUF value type {vtype}")


@dataclass
class RemoteHeader:
    """What the start of a GGUF file says."""

    version: int
    tensor_count: int
    kv: dict[str, Any]
    tensors: list[tuple[str, int]] = field(default_factory=list)  # (name, data offset)
    data_start: Optional[int] = None  # absolute file offset of the tensor data
    bytes_read: int = 0
    requests: int = 0

    @property
    def architecture(self) -> Optional[str]:
        a = self.kv.get("general.architecture")
        return str(a) if a is not None else None

    @property
    def split_count(self) -> int:
        return int(self.kv.get("split.count") or 1)


def read_header(
    url: str,
    *,
    fetch: Optional[Fetch] = None,
    timeout: float = 20.0,
    chunk: int = DEFAULT_CHUNK,
    max_bytes: int = MAX_HEADER_BYTES,
    with_tensors: bool = True,
) -> RemoteHeader:
    """Parse the GGUF header at ``url`` with Range reads.

    ``with_tensors`` also reads the tensor descriptors (names and offsets;
    a few KB more), needed for MoE active-weight bytes.
    Raises GGUFHeaderError / OSError on failure.
    """
    fetch = fetch or (lambda u, s, e: http_range(u, s, e, timeout))
    r = _Reader(url, fetch, chunk, max_bytes)
    if r.read(4) != GGUF_MAGIC:
        raise GGUFHeaderError("not a GGUF file")
    version = r.unpack("I")
    if version > 0xFFFF:
        raise GGUFHeaderError("big-endian GGUF: llama.cpp cannot load it on x86 or ARM")
    if version < 2:
        raise GGUFHeaderError(f"GGUF version {version} is not supported")
    n_tensors = r.unpack("Q")
    n_kv = r.unpack("Q")
    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = _value(r, r.unpack("I"))
    hdr = RemoteHeader(version=version, tensor_count=n_tensors, kv=kv)
    if with_tensors:
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.unpack("I")
            r.skip(8 * n_dims + 4)  # dims (u64 each) + ggml type (u32)
            hdr.tensors.append((name, r.unpack("Q")))
        align = int(kv.get("general.alignment") or DEFAULT_ALIGNMENT)
        hdr.data_start = (r.pos + align - 1) // align * align
    hdr.bytes_read, hdr.requests = len(r.data), r.requests
    return hdr


# ----- metadata -> geometry ------------------------------------------------------


def tensor_byte_totals(
    hdr: RemoteHeader, file_size_bytes: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    """(all tensor bytes, routed-expert tensor bytes) from descriptor offsets.

    Each tensor's size is the gap to the next offset (the last one runs to
    the end of the file). Needs the file size and a single-file model; returns
    (None, None) otherwise.
    """
    if not hdr.tensors or hdr.data_start is None or not file_size_bytes or hdr.split_count > 1:
        return None, None
    ordered = sorted(hdr.tensors, key=lambda t: t[1])
    end = file_size_bytes - hdr.data_start
    total = expert = 0
    for i, (name, off) in enumerate(ordered):
        nxt = ordered[i + 1][1] if i + 1 < len(ordered) else end
        size = nxt - off
        if size < 0:
            return None, None
        total += size
        if is_expert_tensor(name):
            expert += size
    return total, expert


def model_info_from_header(
    hdr: RemoteHeader, file_size_bytes: Optional[int] = None
) -> GGUFModelInfo:
    geometry = geometry_from_kv(hdr.kv)
    arch = hdr.architecture
    p = f"{arch}." if arch else "llama."
    ctx = _as_int(hdr.kv.get(p + "context_length"))
    params = _as_int(hdr.kv.get("general.parameter_count"))
    total, expert = tensor_byte_totals(hdr, file_size_bytes)
    return GGUFModelInfo(
        architecture=arch,
        geometry=geometry,
        context_length=ctx,
        file_size_bytes=file_size_bytes,
        param_count=params,
        tensor_bytes=total,
        expert_tensor_bytes=expert,
    )


def active_weight_bytes(info: GGUFModelInfo) -> Optional[int]:
    """Bytes read per decode token (MoE: dense + used/count × expert tensors)."""
    g = info.geometry
    if info.tensor_bytes is None:
        return None
    if g.is_moe and info.expert_tensor_bytes:
        return moe_active_weight_bytes(
            info.tensor_bytes, info.expert_tensor_bytes, g.expert_used_count, g.expert_count
        )
    return info.tensor_bytes


def read_remote_model_info(
    repo_id: str,
    filename: str,
    *,
    size_bytes: Optional[int] = None,
    revision: str = "main",
    fetch: Optional[Fetch] = None,
    timeout: float = 20.0,
) -> GGUFModelInfo:
    """GGUFModelInfo for a file on Hugging Face, read from its header only."""
    hdr = read_header(hf_file_url(repo_id, filename, revision), fetch=fetch, timeout=timeout)
    return model_info_from_header(hdr, size_bytes)


def geometry_dict(info: GGUFModelInfo) -> dict[str, int]:
    """Compact, JSON-friendly geometry (what models.json stores per entry)."""
    g = info.geometry
    out = {
        "block_count": g.num_layers,
        "attention_layers": g.attention_layers,
        "embedding_length": g.hidden_size,
        "head_count": g.num_attention_heads,
        "head_count_kv": g.num_kv_heads,
        "key_length": g.key_length,
        "value_length": g.value_length,
        "expert_count": g.expert_count,
        "expert_used_count": g.expert_used_count,
        "kv_elements_per_token": g.kv_elements_per_token,
        "active_bytes": active_weight_bytes(info),
    }
    return {k: int(v) for k, v in out.items() if v is not None}


def geometry_from_dict(d: Optional[dict[str, Any]]) -> Optional[ModelGeometry]:
    """Inverse of geometry_dict (None when there is nothing usable)."""
    if not d:
        return None
    g = ModelGeometry(
        num_layers=d.get("block_count"),
        hidden_size=d.get("embedding_length"),
        num_attention_heads=d.get("head_count"),
        num_kv_heads=d.get("head_count_kv"),
        expert_count=d.get("expert_count"),
        expert_used_count=d.get("expert_used_count"),
        key_length=d.get("key_length"),
        value_length=d.get("value_length"),
        attention_layers=d.get("attention_layers"),
        kv_elements_per_token=d.get("kv_elements_per_token"),
    )
    return g if g.is_complete else None
