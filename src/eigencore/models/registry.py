"""
Model registry — maps task types to optimal GGUF models based on hardware profile.
Handles model discovery, download, and local caching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download

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
        repo_id="bartowski/Phi-3.1-mini-4k-instruct-GGUF",
        filename="Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
        params_b=3.8,
        quant="Q4_K_M",
        size_gb=2.2,
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
        filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        params_b=7.0,
        quant="Q4_K_M",
        size_gb=4.4,
        task="general",
        context_length=4096,
        description="Large 7B model. Best quality but needs 8GB+ free RAM.",
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
        return model.local_path(self.cache_dir).exists()

    def download(self, model: ModelSpec, progress: bool = True) -> Path:
        """Download a model from HuggingFace Hub. Returns local path."""
        local_path = model.local_path(self.cache_dir)
        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = hf_hub_download(
            repo_id=model.repo_id,
            filename=model.filename,
            local_dir=local_path.parent,
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)

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
                f"Unknown model: {name}. "
                f"Available: {', '.join(m.name for m in REGISTRY)}"
            )
        return self.recommend(profile, task)
