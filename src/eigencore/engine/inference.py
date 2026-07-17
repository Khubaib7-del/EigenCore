"""
Inference engine — wraps llama.cpp with hardware-aware configuration.
Handles model loading, generation, and resource management.

Phase 2 integration: layer skipping, sparsity analysis, and speculative
decoding hooks are initialized on model load and report stats via
the `optimization_stats` property.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from eigencore.engine.layer_skip import LayerSkipScheduler, SkipStrategy
from eigencore.engine.sparse_inference import SparsityCache, SparsityPredictor
from eigencore.engine.speculative import SpeculativeDecoder
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
    enable_layer_skip: bool = True
    enable_sparse_inference: bool = True
    complexity: float = 0.5


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    prompt_tokens: int
    optimization_stats: Optional[dict] = None


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

        self._layer_skipper: Optional[LayerSkipScheduler] = None
        self._sparsity_predictor: Optional[SparsityPredictor] = None
        self._sparsity_cache = SparsityCache()
        self._speculative_decoder = SpeculativeDecoder(adaptive=True)

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

        self._init_optimizers()

    def _init_optimizers(self) -> None:
        """Initialize Phase 2 optimization modules based on loaded model."""
        metadata = self._llm.metadata if self._llm else {}
        num_layers = int(metadata.get("llama.block_count", 0))
        hidden_size = int(metadata.get("llama.embedding_length", 0))

        if num_layers > 0:
            self._layer_skipper = LayerSkipScheduler(
                num_layers=num_layers,
                strategy=SkipStrategy.STATIC,
                max_skip_rate=0.3,
            )

        if hidden_size > 0 and num_layers > 0:
            self._sparsity_predictor = SparsityPredictor(
                num_layers=num_layers,
                neurons_per_layer=hidden_size,
                warmup_samples=16,
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
            optimization_stats=self._collect_stats(config),
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
            optimization_stats=self._collect_stats(config),
        )

    def _collect_stats(self, config: GenerationConfig) -> dict:
        """Collect optimization statistics from Phase 2 modules."""
        stats: dict = {}

        if self._layer_skipper and config.enable_layer_skip:
            plan = self._layer_skipper.plan(complexity=config.complexity)
            stats["layer_skip"] = {
                "strategy": plan.strategy.name,
                "layers_skipped": len(plan.layers_to_skip),
                "total_layers": plan.total_layers,
                "skip_rate": plan.skip_rate,
                "expected_speedup": plan.expected_speedup,
            }

        if self._sparsity_predictor and config.enable_sparse_inference:
            if self._sparsity_predictor.is_warmed_up:
                exec_plan = self._sparsity_predictor.create_execution_plan()
                stats["sparse_inference"] = {
                    "overall_sparsity": exec_plan.overall_sparsity,
                    "estimated_speedup": exec_plan.estimated_speedup,
                    "flop_savings": exec_plan.estimated_flop_savings,
                }

        stats["sparsity_cache"] = {
            "hit_rate": self._sparsity_cache.hit_rate,
            "hits": self._sparsity_cache.hits,
            "misses": self._sparsity_cache.misses,
        }

        stats["speculative"] = {
            "draft_length": self._speculative_decoder.draft_length,
            "overall_accept_rate": self._speculative_decoder.stats.overall_accept_rate,
            "total_rounds": self._speculative_decoder.stats.total_rounds,
        }

        return stats

    def unload(self) -> None:
        """Release model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def optimization_stats(self) -> dict:
        """Summary of all Phase 2 optimization module states."""
        config = GenerationConfig()
        return self._collect_stats(config) if self._llm else {}

    @property
    def layer_skipper(self) -> Optional[LayerSkipScheduler]:
        return self._layer_skipper

    @property
    def sparsity_predictor(self) -> Optional[SparsityPredictor]:
        return self._sparsity_predictor

    @property
    def speculative_decoder(self) -> SpeculativeDecoder:
        return self._speculative_decoder
