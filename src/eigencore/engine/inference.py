"""
Inference engine — wraps llama.cpp with hardware-aware configuration.
Handles model loading, generation, and resource management.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)
    stream: bool = True


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    prompt_tokens: int


class InferenceEngine:
    def __init__(
        self,
        model_path: Path,
        profile: HardwareProfile,
        model_spec: Optional[ModelSpec] = None,
    ):
        self.model_path = model_path
        self.profile = profile
        self.model_spec = model_spec
        self._llm = None

    def load(self) -> None:
        """Load the model with hardware-optimized settings."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for inference. "
                "Install with: pip install llama-cpp-python"
            )

        ctx_length = self.profile.recommended_context_length
        if self.model_spec and self.model_spec.context_length < ctx_length:
            ctx_length = self.model_spec.context_length

        n_threads = self.profile.recommended_threads
        n_batch = self._compute_batch_size()

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=ctx_length,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            n_batch=n_batch,
            n_gpu_layers=0,  # CPU-only
            verbose=False,
        )

    def _compute_batch_size(self) -> int:
        """Determine optimal batch size based on available RAM and ISA."""
        if InstructionSet.AVX512F in self.profile.instruction_sets:
            return 1024
        elif InstructionSet.AVX2 in self.profile.instruction_sets:
            return 512
        else:
            return 256

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a complete response (non-streaming)."""
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if config is None:
            config = GenerationConfig(stream=False)

        start = time.perf_counter()

        response = self._llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or None,
            echo=False,
        )

        elapsed = time.perf_counter() - start
        text = response["choices"][0]["text"]
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0.0

        return GenerationResult(
            text=text,
            tokens_generated=completion_tokens,
            time_seconds=elapsed,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
        )

    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream tokens as they're generated."""
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if config is None:
            config = GenerationConfig(stream=True)

        for chunk in self._llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or None,
            echo=False,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def chat(
        self,
        messages: list[dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Chat-style completion with message list."""
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if config is None:
            config = GenerationConfig(stream=False)

        start = time.perf_counter()

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or None,
        )

        elapsed = time.perf_counter() - start
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0.0

        return GenerationResult(
            text=text,
            tokens_generated=completion_tokens,
            time_seconds=elapsed,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
        )

    def unload(self) -> None:
        """Release model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None
