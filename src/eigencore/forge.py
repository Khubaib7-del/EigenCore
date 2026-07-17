"""
EigenCore — the unified API surface.

Ties together all 4 layers into a single interface:
  Layer 1: Hardware profiling (auto-detect and configure)
  Layer 2: Inference (model loading, generation, streaming)
  Layer 3: Training (QLoRA fine-tuning with adaptive epochs)
  Layer 4: Agent orchestration (routing, context, tools) — Phase 3

Usage:
    from eigencore import Forge

    forge = Forge()                           # auto-profiles hardware
    print(forge.profile.summary())            # see what you're working with

    result = forge.generate("Hello world")    # auto-downloads + runs best model
    print(result.text)

    for token in forge.stream("Tell me about Python"):
        print(token, end="")

    decision = forge.route("Write a sorting algorithm")
    print(decision.domain, decision.model.name)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from eigencore.context.manager import ContextManager
from eigencore.engine.inference import GenerationConfig, GenerationResult, InferenceEngine
from eigencore.hal.profiler import HardwareProfile, profile_hardware
from eigencore.models.registry import ModelRegistry, ModelSpec
from eigencore.router.classifier import RoutingDecision, TaskRouter


class Forge:
    """
    Main entry point for the EigenCore framework.

    Auto-detects hardware, manages models, and provides inference +
    routing through a single API.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        task: str = "general",
        cache_dir: Optional[Path] = None,
        context_db: Optional[Path] = None,
    ):
        self.profile: HardwareProfile = profile_hardware()
        self.registry = ModelRegistry(cache_dir)
        self.router = TaskRouter(self.profile, self.registry)
        self.context = ContextManager(
            max_context_tokens=self.profile.recommended_context_length,
            db_path=context_db,
        )

        self._engine: Optional[InferenceEngine] = None
        self._current_model: Optional[ModelSpec] = None
        self._default_task = task

        if model_name:
            self._load_model(model_name)

    def _load_model(self, name: Optional[str] = None, task: Optional[str] = None) -> None:
        """Load a model by name or auto-select based on hardware + task."""
        spec = self.registry.resolve(name, self.profile, task or self._default_task)

        if self._current_model and self._current_model.name == spec.name:
            return  # already loaded

        # unload previous model
        if self._engine and self._engine.is_loaded:
            self._engine.unload()

        # download if needed
        if not self.registry.is_downloaded(spec):
            self.registry.download(spec)

        model_path = spec.local_path(self.registry.cache_dir)
        self._engine = InferenceEngine(model_path, self.profile, spec)
        self._engine.load()
        self._current_model = spec

    def _ensure_loaded(self, prompt: Optional[str] = None) -> None:
        """Ensure a model is loaded, using routing if prompt is provided."""
        if self._engine and self._engine.is_loaded:
            if prompt:
                decision = self.router.route(prompt)
                if self._current_model and self.router.should_swap(self._current_model, decision):
                    self._load_model(decision.model.name)
            return
        self._load_model()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> GenerationResult:
        """Generate a response. Auto-downloads model on first use."""
        self._ensure_loaded(prompt)
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs,
        )
        return self._engine.generate(prompt, config)

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[str]:
        """Stream tokens as they're generated."""
        self._ensure_loaded(prompt)
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        yield from self._engine.stream(prompt, config)

    def chat(
        self,
        message: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """
        Chat with context management. Messages are tracked and compressed
        automatically using attention-aware windowing.
        """
        self._ensure_loaded(message)
        self.context.add("user", message)
        ctx = self.context.get_context()

        messages = [{"role": m.role, "content": m.content} for m in ctx.messages]

        config = GenerationConfig(max_tokens=max_tokens, temperature=temperature, stream=False)
        result = self._engine.chat(messages, config)

        self.context.add("assistant", result.text)
        return result

    def route(self, prompt: str) -> RoutingDecision:
        """Classify a prompt and get the routing decision without generating."""
        return self.router.route(prompt)

    def unload(self) -> None:
        """Release the current model from memory."""
        if self._engine:
            self._engine.unload()
        self._current_model = None

    @property
    def loaded_model(self) -> Optional[str]:
        return self._current_model.name if self._current_model else None

    @property
    def context_stats(self) -> dict:
        return {
            "messages": self.context.message_count,
            "tokens": self.context.token_count,
            "max_tokens": self.context.max_context_tokens,
            "utilization": self.context.token_count / self.context.max_context_tokens,
        }
