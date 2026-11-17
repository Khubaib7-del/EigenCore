"""
Per-machine fit verdict for a model, before it is downloaded.

``fit_report(spec, profile)`` answers, for this machine:

* **verdict** – ``fits`` / ``tight`` / ``too_big`` (or ``unknown`` when the
  model's shape is not known yet), at ``CHECK_CONTEXT`` tokens of context;
* **max_context** – the largest power-of-two context whose KV cache still
  fits next to the weights (capped at the model's own maximum);
* **decode ceiling** – memory bandwidth / bytes read per token. An optimistic
  *estimate*, never a measurement: real decode is lower;
* **runtime** – whether the installed llama-cpp-python knows the model's
  architecture (``min_runtime``).

Geometry comes from, in order: the downloaded file, the ``geometry`` block of
a curated ``models.json`` entry, or a GGUF header read over HTTP Range
(``fit.gguf_remote``). Without any of them only a "too big" verdict from the
file size alone is possible; anything else stays ``unknown``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from eigencore.fit.estimator import (
    DEFAULT_RUNTIME_RESERVE_GB,
    DEFAULT_SAFETY_MARGIN_GB,
    FitVerdict,
    ModelGeometry,
    decode_ceiling_tok_s,
    estimate_fit,
    max_context_for_budget,
)
from eigencore.fit.gguf_remote import geometry_from_dict
from eigencore.models.registry import ModelSpec, _version_tuple, runtime_version

#: Context the verdict is computed at (clamped to the model's maximum):
#: an everyday chat / coding session.
CHECK_CONTEXT = 8192
#: Bundled arch / tokenizer -> first llama-cpp-python release that loads it.
RUNTIME_SUPPORT_JSON = Path(__file__).with_name("runtime_support.json")

FITS, TIGHT, TOO_BIG, UNKNOWN = "fits", "tight", "too_big", "unknown"
_VERDICT = {
    FitVerdict.FITS_WITH_MARGIN: FITS,
    FitVerdict.TIGHT: TIGHT,
    FitVerdict.DOES_NOT_FIT: TOO_BIG,
    FitVerdict.UNKNOWN: UNKNOWN,
}
_UNSET: Any = object()


@lru_cache(maxsize=1)
def runtime_support() -> dict[str, dict[str, str]]:
    try:
        doc = json.loads(RUNTIME_SUPPORT_JSON.read_text(encoding="utf-8"))
        return {"arch": dict(doc.get("arch") or {}), "pre": dict(doc.get("pre") or {})}
    except (OSError, ValueError):
        return {"arch": {}, "pre": {}}


def min_runtime_for(arch: Optional[str], tokenizer_pre: Optional[str] = None) -> Optional[str]:
    """First llama-cpp-python release that loads ``arch`` (and the tokenizer pre-type).

    Returns None when unknown (not in the table: newer than the last scan, or
    no arch given), in which case callers should not block the model.
    """
    table = runtime_support()
    versions = []
    for kind, name in (("arch", arch), ("pre", tokenizer_pre)):
        if name and name in table[kind]:
            versions.append(table[kind][name])
    return max(versions, key=_version_tuple) if versions else None


def arch_is_known(arch: Optional[str]) -> bool:
    return bool(arch) and arch in runtime_support()["arch"]


@dataclass
class FitReport:
    verdict: str  # fits | tight | too_big | unknown
    context_checked: int
    max_context: Optional[int]  # None when geometry is unknown
    model_max_context: Optional[int]
    weights_gb: float  # GiB
    kv_cache_gb: Optional[float]  # GiB at context_checked
    total_gb: Optional[float]  # GiB: weights + KV + runtime reserve
    available_ram_gb: float
    decode_ceiling_tok_s: Optional[float]  # ESTIMATE (bandwidth / active bytes)
    active_weights_gb: Optional[float]
    runtime_ok: bool
    min_runtime: Optional[str]
    installed_runtime: Optional[str]
    basis: str  # file | curated | header | size-only
    notes: list[str] = field(default_factory=list)

    @property
    def runtime_note(self) -> Optional[str]:
        if self.runtime_ok:
            return None
        if self.installed_runtime is None:
            return None
        return (
            f"needs llama-cpp-python {self.min_runtime} or newer "
            f"(installed {self.installed_runtime})"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["runtime_note"] = self.runtime_note
        d["estimate"] = True  # every number here is a planning estimate
        return d


def fit_report(
    spec: ModelSpec,
    profile: Any,
    *,
    geometry: Optional[ModelGeometry] = None,
    active_bytes: Optional[int] = None,
    basis: Optional[str] = None,
    context: Optional[int] = None,
    installed_runtime: Any = _UNSET,
    kv_quant: str = "fp16",
) -> FitReport:
    """Fit verdict for ``spec`` on the machine described by ``profile``.

    ``geometry`` / ``active_bytes`` override the curated ``spec.geometry``
    (pass what a downloaded file or a remote header said).
    """
    avail = float(getattr(profile, "available_ram_gb", 0.0) or 0.0)
    bandwidth = getattr(profile, "estimated_bandwidth_gbps", None)
    if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
        bandwidth = None
    curated = getattr(spec, "geometry", None) or {}
    if geometry is None:
        geometry = geometry_from_dict(curated)
        if geometry is not None:
            basis = basis or "curated"
            active_bytes = active_bytes if active_bytes is not None else curated.get("active_bytes")
    basis = basis or ("header" if geometry is not None else "size-only")

    model_max = spec.context_length or None
    ctx = context or CHECK_CONTEXT
    if model_max:
        ctx = min(ctx, model_max)
    size_bytes = spec.size_bytes or int(spec.size_gb * 1e9)

    installed = runtime_version() if installed_runtime is _UNSET else installed_runtime
    need = spec.min_runtime or min_runtime_for(spec.arch)
    runtime_ok = not (installed and need and _version_tuple(installed) < _version_tuple(need))

    if geometry is None or not geometry.is_complete:
        weights_gb = size_bytes / 1024**3
        total = weights_gb + DEFAULT_RUNTIME_RESERVE_GB
        verdict = TOO_BIG if total > avail else UNKNOWN
        notes = ["model shape not read yet; KV cache not counted"]
        if verdict == TOO_BIG:
            notes = [f"the file alone ({weights_gb:.1f} GiB) plus runtime does not fit"]
        return FitReport(
            verdict=verdict,
            context_checked=ctx,
            max_context=0 if verdict == TOO_BIG else None,
            model_max_context=model_max,
            weights_gb=round(weights_gb, 2),
            kv_cache_gb=None,
            total_gb=None,
            available_ram_gb=round(avail, 2),
            decode_ceiling_tok_s=None,
            active_weights_gb=None,
            runtime_ok=runtime_ok,
            min_runtime=need,
            installed_runtime=installed,
            basis="size-only",
            notes=notes,
        )

    est = estimate_fit(
        spec.params_b,
        spec.quant,
        ctx,
        avail,
        geometry=geometry,
        kv_quant=kv_quant,
        file_size_bytes=size_bytes,
        bandwidth_gbps=None,
        active_weight_bytes=active_bytes,
    )
    max_ctx = max_context_for_budget(
        spec.params_b,
        spec.quant,
        avail,
        geometry=geometry,
        model_context_limit=model_max or 128 * 1024,
        kv_quant=kv_quant,
        file_size_bytes=size_bytes,
        runtime_reserve_gb=DEFAULT_RUNTIME_RESERVE_GB,
        safety_margin_gb=DEFAULT_SAFETY_MARGIN_GB,
    )
    ceiling = None
    if bandwidth is not None:
        active_gib = est.active_weights_gb if est.active_weights_gb is not None else est.weights_gb
        ceiling = round(decode_ceiling_tok_s(bandwidth, active_gib), 1)
    return FitReport(
        verdict=_VERDICT[est.verdict],
        context_checked=ctx,
        max_context=max_ctx,
        model_max_context=model_max,
        weights_gb=round(est.weights_gb, 2),
        kv_cache_gb=round(est.kv_cache_gb, 2),
        total_gb=round(est.total_gb, 2),
        available_ram_gb=round(avail, 2),
        decode_ceiling_tok_s=ceiling,
        active_weights_gb=(
            round(est.active_weights_gb, 2) if est.active_weights_gb is not None else None
        ),
        runtime_ok=runtime_ok,
        min_runtime=need,
        installed_runtime=installed,
        basis=basis,
        notes=list(est.notes),
    )


_RANK = {FITS: 0, TIGHT: 1, UNKNOWN: 2, TOO_BIG: 3}


def verdict_rank(verdict: str) -> int:
    """Sort key: fits < tight < unknown < too_big."""
    return _RANK.get(verdict, 2)


# RAM classes for the curated table (GB of installed RAM).
RAM_CLASSES = (8, 16, 32, 64)
#: Share of installed RAM a model may use (the OS and other apps keep the rest).
RAM_CLASS_SHARE = 0.8


def ram_class(spec: ModelSpec, geometry: Optional[ModelGeometry] = None) -> Optional[int]:
    """Smallest RAM class (8/16/32/64 GB) where file + KV at CHECK_CONTEXT + runtime
    reserve stays within ``RAM_CLASS_SHARE`` of installed RAM. None when > 64 GB."""
    geometry = geometry or geometry_from_dict(getattr(spec, "geometry", None))
    size_bytes = spec.size_bytes or int(spec.size_gb * 1e9)
    need = size_bytes / 1024**3 + DEFAULT_RUNTIME_RESERVE_GB
    if geometry is not None and geometry.is_complete:
        from eigencore.fit.estimator import kv_bytes_per_token

        ctx = min(CHECK_CONTEXT, spec.context_length or CHECK_CONTEXT)
        need += kv_bytes_per_token(geometry) * ctx / 1024**3
    for c in RAM_CLASSES:
        if need <= c * RAM_CLASS_SHARE:
            return c
    return None
