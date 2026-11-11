"""
Model registry — maps task types to optimal GGUF models based on hardware profile.
Handles model discovery, download, and local caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from eigencore.hal.profiler import HardwareProfile


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

    @property
    def is_moe(self) -> bool:
        return self.experts_total is not None and self.experts_active is not None

    def local_path(self, cache_dir: Path) -> Path:
        safe_name = self.repo_id.replace("/", "--")
        return cache_dir / safe_name / self.filename


# curated registry of GGUF models — smallest to largest
REGISTRY: list[ModelSpec] = [
    ModelSpec(
        name="tinyllama-1.1b",
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        params_b=1.1,
        quant="Q4_K_M",
        size_gb=0.67,
        task="general",
        context_length=2048,
        description="Tiny general-purpose model. Fast on any CPU.",
    ),
    ModelSpec(
        name="phi-3-mini",
        repo_id="bartowski/Phi-3-mini-4k-instruct-GGUF",
        filename="Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        params_b=3.8,
        quant="Q4_K_M",
        size_gb=2.4,
        task="general",
        context_length=4096,
        description="Strong 3.8B model. Good reasoning for its size.",
    ),
    ModelSpec(
        name="qwen2.5-1.5b",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        params_b=1.5,
        quant="Q4_K_M",
        size_gb=1.0,
        task="general",
        context_length=4096,
        description="Efficient 1.5B model with strong multilingual support.",
    ),
    ModelSpec(
        name="qwen2.5-3b",
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        params_b=3.0,
        quant="Q4_K_M",
        size_gb=1.8,
        task="general",
        context_length=4096,
        description="Balanced 3B model. Good general reasoning.",
    ),
    ModelSpec(
        name="qwen2.5-coder-1.5b",
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        params_b=1.5,
        quant="Q4_K_M",
        size_gb=1.0,
        task="code",
        context_length=4096,
        description="Code-specialized 1.5B model.",
    ),
    ModelSpec(
        name="qwen2.5-7b",
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        # Q4_K_M is sharded into 2 parts on HF; download() fetches all shards
        filename="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        params_b=7.0,
        quant="Q4_K_M",
        size_gb=4.7,
        task="general",
        context_length=4096,
        description="Large 7B model. Best quality but needs 8GB+ free RAM.",
    ),
    ModelSpec(
        name="qwen3-30b-a3b",
        repo_id="Qwen/Qwen3-30B-A3B-GGUF",
        filename="Qwen3-30B-A3B-Q4_K_M.gguf",
        params_b=30.5,
        quant="Q4_K_M",
        size_gb=18.6,
        task="general",
        context_length=32768,
        description="MoE 30B (3.3B active/token). Decode is bandwidth-bound on ACTIVE weights; full file must be resident or mmap-paged.",
        experts_total=128,
        experts_active=8,
    ),
]


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
        candidates = self.list_models(task)
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
