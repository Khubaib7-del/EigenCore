"""
Model registry — maps task types to optimal GGUF models based on hardware profile.
Handles model discovery, download, and local caching.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import hf_hub_download

from eigencore.hal.profiler import HardwareProfile

#: Version of the models.json layout this code reads.
SCHEMA_VERSION = 2
#: The curated list shipped inside the package (source of truth).
BUNDLED_MODELS_JSON = Path(__file__).with_name("models.json")
#: Newer list attached to the latest GitHub Release (opt-in, see load_registry).
REMOTE_MODELS_URL = "https://github.com/Khubaib7-del/EigenCore/releases/latest/download/models.json"
#: Set to 1 to try REMOTE_MODELS_URL first; any failure falls back to the bundled copy.
REMOTE_ENV = "EIGENCORE_MODELS_REMOTE"

REQUIRED_FIELDS = (
    "name",
    "repo_id",
    "filename",
    "params_b",
    "quant",
    "size_gb",
    "task",
    "context_length",
    "description",
    "license",
    "tags",
    "arch",
    "min_runtime",
    "noncommercial",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    filename: str
    params_b: float
    quant: str
    size_gb: float
    task: str
    context_length: int
    description: str
    experts_total: int | None = None  # MoE: total experts per MoE layer
    experts_active: int | None = None  # MoE: experts activated per token
    # registry v2 (all optional so older callers and tests keep working)
    license: str | None = None  # HF licence id, e.g. "apache-2.0", "mit", "other"
    license_name: str | None = None  # set when license == "other"
    license_url: str | None = None
    noncommercial: bool = False  # licence forbids commercial use
    tags: tuple[str, ...] = ()  # e.g. ("code", "reasoning")
    arch: str | None = None  # GGUF general.architecture
    min_runtime: str | None = None  # first llama-cpp-python version that loads `arch`
    size_bytes: int | None = None  # exact file size (sum of shards)

    @property
    def is_moe(self) -> bool:
        return self.experts_total is not None and self.experts_active is not None

    def local_path(self, cache_dir: Path) -> Path:
        safe_name = self.repo_id.replace("/", "--")
        return cache_dir / safe_name / self.filename


_SPEC_FIELDS = {f.name for f in fields(ModelSpec)}


def spec_from_dict(entry: dict[str, Any]) -> ModelSpec:
    """One models.json entry -> ModelSpec. Unknown keys are ignored (forward compatible)."""
    missing = [k for k in REQUIRED_FIELDS if k not in entry]
    if missing:
        raise ValueError(f"model entry {entry.get('name', '?')!r} is missing {missing}")
    kw = {k: v for k, v in entry.items() if k in _SPEC_FIELDS}
    kw["tags"] = tuple(kw.get("tags") or ())
    kw["noncommercial"] = bool(kw.get("noncommercial"))
    return ModelSpec(**kw)


def parse_models(doc: dict[str, Any]) -> list[ModelSpec]:
    """Validate a models.json document and return its specs."""
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"models.json schema_version {version!r}, expected {SCHEMA_VERSION}")
    specs = [spec_from_dict(e) for e in doc.get("models", [])]
    if not specs:
        raise ValueError("models.json has no models")
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate model names in models.json: {dupes}")
    return specs


def _fetch_remote(url: str, timeout: float) -> list[ModelSpec]:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "eigencore"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_models(json.loads(r.read().decode("utf-8")))


def load_registry(
    path: Path | None = None,
    *,
    remote: bool | None = None,
    url: str = REMOTE_MODELS_URL,
    timeout: float = 3.0,
) -> list[ModelSpec]:
    """Load the curated model list.

    The bundled ``models.json`` is the source of truth. With ``remote=True``
    (or ``EIGENCORE_MODELS_REMOTE=1``) the copy attached to the latest GitHub
    Release is tried first, with a short timeout; any failure (offline, 404,
    bad JSON, other schema) silently falls back to the bundled file.
    """
    if remote is None:
        remote = os.environ.get(REMOTE_ENV, "").strip().lower() in ("1", "true", "yes")
    if remote and path is None:
        try:
            return _fetch_remote(url, timeout)
        except Exception:
            pass
    src = path or BUNDLED_MODELS_JSON
    return parse_models(json.loads(src.read_text(encoding="utf-8")))


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def runtime_version() -> str | None:
    """Installed llama-cpp-python version, or None when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("llama-cpp-python")
    except PackageNotFoundError:
        return None


def runtime_supports(spec: ModelSpec, installed: str | None = None) -> bool:
    """False only when the installed llama-cpp-python is known to be older than spec.min_runtime."""
    installed = installed if installed is not None else runtime_version()
    if not installed or not spec.min_runtime:
        return True
    return _version_tuple(installed) >= _version_tuple(spec.min_runtime)


# curated registry of GGUF models, loaded from models.json (smallest to largest)
REGISTRY: list[ModelSpec] = load_registry()


class ModelRegistry:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or self._default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_cache_dir() -> Path:
        home = Path.home()
        return home / ".eigencore" / "models"

    def list_models(self, task: Optional[str] = None) -> list[ModelSpec]:
        models = REGISTRY
        if task:
            task_models = [m for m in models if m.task == task]
            if task_models:
                models = task_models
        return models

    def recommend(self, profile: HardwareProfile, task: str = "general") -> ModelSpec:
        """Pick the largest model that fits the hardware profile for the given task."""
        # Auto-pick never lands on a non-commercial licence or on an arch the
        # installed runtime can't load; those stay available by name.
        candidates = [
            m for m in self.list_models(task) if not m.noncommercial and runtime_supports(m)
        ]
        candidates = sorted(candidates, key=lambda m: m.params_b, reverse=True)

        for model in candidates:
            if model.size_gb <= (profile.available_ram_gb - 2.0):
                return model

        # fallback to smallest model
        return min(REGISTRY, key=lambda m: m.size_gb)

    def is_downloaded(self, model: ModelSpec) -> bool:
        """True when the model file (or all its shards) is present locally."""
        if not self.shard_prefix(model):
            return model.local_path(self.cache_dir).exists()
        prefix = self.shard_prefix(model)
        count = self._shard_count(model)
        return all(
            (
                model.local_path(self.cache_dir).parent / f"{prefix}-{i:05d}-of-{count:05d}.gguf"
            ).exists()
            for i in range(1, count + 1)
        )

    @staticmethod
    def shard_prefix(model: ModelSpec) -> str | None:
        """'file-00001-of-00002.gguf' → 'file' (None when not sharded)."""
        import re

        m = re.match(r"(.+)-(\d{5})-of-(\d{5})\.gguf$", model.filename)
        if m and m.group(2) == "00001":
            return m.group(1)
        return None

    @classmethod
    def _shard_count(cls, model: ModelSpec) -> int:
        import re

        m = re.match(r".+-\d{5}-of-(\d{5})\.gguf$", model.filename)
        return int(m.group(1)) if m else 1

    def download(self, model: ModelSpec, progress: bool = True) -> Path:
        """Download a model from HuggingFace Hub (all shards when sharded).

        Returns local path. Raises RuntimeError with a clean message on
        failure (404, network) — never a raw HF traceback at the caller.
        """
        local_path = model.local_path(self.cache_dir)
        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            prefix = self.shard_prefix(model)
            if prefix:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=model.repo_id,
                    allow_patterns=[f"{prefix}-*.gguf"],
                    local_dir=local_path.parent,
                )
            else:
                hf_hub_download(
                    repo_id=model.repo_id,
                    filename=model.filename,
                    local_dir=local_path.parent,
                )
        except Exception as e:
            raise RuntimeError(
                f"Download of {model.name} from {model.repo_id} failed: {type(e).__name__}: {e}"
            ) from e
        return local_path

    def delete(self, model: ModelSpec) -> bool:
        """Delete a downloaded model (all shards). Returns True if anything was removed."""
        local_path = model.local_path(self.cache_dir)
        removed = False
        prefix = self.shard_prefix(model)
        if prefix:
            for shard in local_path.parent.glob(f"{prefix}-*.gguf"):
                shard.unlink()
                removed = True
        elif local_path.exists():
            local_path.unlink()
            removed = True
        if removed:
            try:
                local_path.parent.rmdir()  # remove empty model dir
            except OSError:
                pass
        return removed

    def resolve(
        self,
        name: Optional[str],
        profile: HardwareProfile,
        task: str = "general",
    ) -> ModelSpec:
        """Resolve a model by name, or auto-select based on hardware profile."""
        if name:
            for model in REGISTRY:
                if model.name == name:
                    if model.size_gb > profile.available_ram_gb - 1.0:
                        raise ValueError(
                            f"Model {name} needs ~{model.size_gb:.1f}GB but only "
                            f"{profile.available_ram_gb:.1f}GB RAM available. "
                            f"Try a smaller model."
                        )
                    return model
            raise ValueError(
                f"Unknown model: {name}. Available: {', '.join(m.name for m in REGISTRY)}"
            )
        return self.recommend(profile, task)
