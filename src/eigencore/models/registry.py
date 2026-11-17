"""
Model registry — maps task types to optimal GGUF models based on hardware profile.
Handles model discovery, download, and local caching.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
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
    # catalog (all optional)
    # GGUF header geometry: block_count, attention_layers, embedding_length,
    # head_count, head_count_kv, key_length, value_length, expert_count,
    # expert_used_count, active_bytes. Lets the fit verdict run before download.
    geometry: dict | None = field(default=None, compare=False, hash=False)
    ram_class: int | None = None  # smallest RAM class (8/16/32/64 GB) it is listed for
    priority: int = 0  # curated preference among models that fit (higher first)
    verified_at: str | None = None  # date sizes/licence/header were last checked
    source: str = "curated"  # curated | hf-search | custom

    @property
    def repo_dir_name(self) -> str:
        return self.repo_id.replace("/", "--")

    @property
    def is_moe(self) -> bool:
        return self.experts_total is not None and self.experts_active is not None

    def local_path(self, cache_dir: Path) -> Path:
        return cache_dir / self.repo_dir_name / self.filename

    def repo_dir(self, cache_dir: Path) -> Path:
        """Local folder for the repo (files keep their in-repo sub-folders)."""
        return cache_dir / self.repo_dir_name


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


CUSTOM_MODELS_FILE = "custom-models.json"


class ModelRegistry:
    def __init__(self, cache_dir: Optional[Path] = None, catalog: Any = None):
        self.cache_dir = cache_dir or self._default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog = catalog  # HFCatalog, created on first use

    @staticmethod
    def _default_cache_dir() -> Path:
        home = Path.home()
        return home / ".eigencore" / "models"

    # ----- models picked from Hugging Face search ------------------------------
    @property
    def _custom_path(self) -> Path:
        return self.cache_dir / CUSTOM_MODELS_FILE

    def custom_models(self) -> list[ModelSpec]:
        """Models added from Hugging Face search (kept next to the downloads)."""
        try:
            doc = json.loads(self._custom_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for e in doc.get("models", []) if isinstance(doc, dict) else []:
            try:
                kw = {k: v for k, v in e.items() if k in _SPEC_FIELDS}
                kw["tags"] = tuple(kw.get("tags") or ())
                kw["source"] = "custom"
                out.append(ModelSpec(**kw))
            except TypeError:
                continue
        return out

    def add_custom(self, spec: ModelSpec) -> ModelSpec:
        """Remember a searched model so it stays in the list after download."""
        from dataclasses import asdict, replace

        if any(m.name == spec.name for m in REGISTRY):
            return spec
        spec = replace(spec, source="custom")
        keep = [m for m in self.custom_models() if m.name != spec.name] + [spec]
        doc = {"schema_version": SCHEMA_VERSION, "models": []}
        for m in keep:
            d = asdict(m)
            d["tags"] = list(m.tags)
            doc["models"].append(d)
        tmp = self._custom_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.replace(self._custom_path)
        return spec

    def remove_custom(self, name: str) -> None:
        keep = [m for m in self.custom_models() if m.name != name]
        if len(keep) == len(self.custom_models()):
            return
        from dataclasses import asdict

        doc = {"schema_version": SCHEMA_VERSION, "models": []}
        for m in keep:
            d = asdict(m)
            d["tags"] = list(m.tags)
            doc["models"].append(d)
        self._custom_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")

    @property
    def catalog(self):
        if self._catalog is None:
            from eigencore.models.hf_catalog import HFCatalog

            self._catalog = HFCatalog(cache_path=self.cache_dir.parent / "catalog-cache.json")
        return self._catalog

    def list_models(self, task: Optional[str] = None) -> list[ModelSpec]:
        models = REGISTRY + self.custom_models()
        if task:
            task_models = [m for m in models if m.task == task or task in m.tags]
            if task_models:
                models = task_models
        return models

    def recommend(self, profile: HardwareProfile, task: str = "general") -> ModelSpec:
        """Best curated model for the task on this machine.

        Candidates have the task as their main task or as a tag. They are
        ranked by fit verdict at 8K context (fits, then tight) against the
        machine's RAM budget (total RAM minus the system reserve, see
        ``fit_check.ram_budget_gb``; free RAM at the moment is not used), then the
        curated priority, then whether the task is the model's main task,
        then size. Auto-pick never lands on a non-commercial licence or on an arch
        the installed runtime can't load; those stay available by name.
        """
        from eigencore.models.fit_check import FITS, TIGHT, fit_report

        installed = runtime_version()
        pool = [m for m in REGISTRY if m.task == task or task in m.tags] or list(REGISTRY)
        candidates = [
            m for m in pool if not m.noncommercial and runtime_supports(m, installed or "")
        ]
        scored = []
        for m in candidates:
            verdict = fit_report(m, profile, installed_runtime=installed).verdict
            if verdict not in (FITS, TIGHT):
                continue
            scored.append(((verdict != FITS), -m.priority, m.task != task, -m.params_b, m.name, m))
        if scored:
            return min(scored, key=lambda t: t[:5])[-1]
        # nothing fits: the smallest model we may auto-pick
        return min(candidates or REGISTRY, key=lambda m: m.size_gb)

    def _shard_paths(self, model: ModelSpec) -> list[Path]:
        prefix = self.shard_prefix(model)
        count = self._shard_count(model)
        root = model.repo_dir(self.cache_dir)
        return [root / f"{prefix}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]

    def is_downloaded(self, model: ModelSpec) -> bool:
        """True when the model file (or all its shards) is present locally."""
        if not self.shard_prefix(model):
            return model.local_path(self.cache_dir).exists()
        return all(p.exists() for p in self._shard_paths(model))

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
        Files keep their in-repo folders under ``<cache>/<org>--<repo>/``.
        """
        local_path = model.local_path(self.cache_dir)
        if self.is_downloaded(model):
            return local_path

        root = model.repo_dir(self.cache_dir)
        root.mkdir(parents=True, exist_ok=True)

        try:
            prefix = self.shard_prefix(model)
            if prefix:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=model.repo_id,
                    allow_patterns=[f"{prefix}-*.gguf"],
                    local_dir=root,
                )
            else:
                hf_hub_download(
                    repo_id=model.repo_id,
                    filename=model.filename,
                    local_dir=root,
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
            for shard in self._shard_paths(model):
                if shard.exists():
                    shard.unlink()
                    removed = True
        elif local_path.exists():
            local_path.unlink()
            removed = True
        if removed:
            # remove now-empty folders up to the repo folder
            root = model.repo_dir(self.cache_dir)
            d = local_path.parent
            while True:
                try:
                    d.rmdir()
                except OSError:
                    break
                if d == root:
                    break
                d = d.parent
        return removed

    def find(self, name: str) -> Optional[ModelSpec]:
        """A curated or custom model by name, or a curated one by ``repo_id[:file]``."""
        models = self.list_models()
        hit = next((m for m in models if m.name == name), None)
        if hit is not None:
            return hit
        repo, _, file = name.partition(":")
        for m in models:
            if m.repo_id.lower() == repo.lower() and (not file or m.filename == file):
                return m
        return None

    def resolve_ref(self, ref: str) -> ModelSpec:
        """``repo_id[:file]`` from Hugging Face -> ModelSpec (curated entry when it is one).

        Raises ValueError when the repo has no usable GGUF or HF is unreachable.
        """
        known = self.find(ref)
        if known is not None:
            return known
        from eigencore.models.fit_check import min_runtime_for

        entry = self.catalog.entry(ref)
        if entry is None:
            why = self.catalog.last_error or "no GGUF file found"
            raise ValueError(f"Could not use {ref} from Hugging Face: {why}")
        return entry.to_spec(min_runtime=min_runtime_for(entry.arch))

    def resolve(
        self,
        name: Optional[str],
        profile: HardwareProfile,
        task: str = "general",
    ) -> ModelSpec:
        """Resolve a model by name or ``repo_id[:file]``, or auto-select for the hardware."""
        if name:
            model = self.find(name)
            if model is None and "/" in name:
                model = self.resolve_ref(name)
            if model is None:
                raise ValueError(
                    f"Unknown model: {name}. Available: {', '.join(m.name for m in REGISTRY)} "
                    "(or any Hugging Face GGUF repo as org/repo[:file.gguf])"
                )
            from eigencore.models.fit_check import machine_ram

            # by name, only refuse a file that alone exceeds the budget (a tight
            # model is the user's call; the fit chip and `models` say so)
            total_ram, budget, _free = machine_ram(profile)
            need = model.size_gb * 1e9 / 1024**3
            if need > budget:
                raise ValueError(
                    f"Model {name} needs ~{need:.1f} GB for the file alone but this machine's "
                    f"RAM budget is {budget:.1f} GB ({total_ram:.1f} GB RAM minus "
                    f"{total_ram - budget:.1f} GB for the OS and other apps). "
                    f"Try a smaller model."
                )
            return model
        return self.recommend(profile, task)
