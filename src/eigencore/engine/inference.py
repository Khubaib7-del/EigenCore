"""
Inference engine — wraps llama.cpp with hardware-aware configuration.
Handles model loading, generation, and resource management.

Phase 2 integration: layer skipping, sparsity analysis, and speculative
decoding hooks are initialized on model load and report stats via
the `optimization_stats` property.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from eigencore.engine.layer_skip import LayerSkipScheduler, SkipStrategy
from eigencore.engine.sparse_inference import SparsityCache, SparsityPredictor
from eigencore.engine.speculative import SpeculativeDecoder
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec

# llama-cpp-python takes the KV cache types as ggml_type enum values via
# `type_k` / `type_v` (there is no `type_kv`; unknown kwargs are swallowed).
# ggml.h: GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q8_0 = 8.
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
KV_NEEDS_FLASH = (
    "--kv {kv_type} needs flash attention (llama.cpp only quantizes the V cache "
    "with flash attention on). Drop --no-flash, or use --kv fp16."
)

_KV_TYPE_MAP = {"fp16": GGML_TYPE_F16, "q8_0": GGML_TYPE_Q8_0}


# Windows NTSTATUS for "illegal instruction" (0xC000001D), as ctypes reports it.
_ILLEGAL_INSTRUCTION_CODES = ("0xc000001d", "-1073741795")


def missing_cpu_features(profile: HardwareProfile) -> list[str]:
    """["AVX2"] when this x86 CPU lacks it, else [].

    The bundled llama.cpp build targets an AVX2 + FMA + F16C baseline (every
    AVX2 CPU has FMA and F16C, and the Windows detector cannot see those two
    separately). Nothing is blocked when detection failed (NONE = unknown) or
    on other architectures.
    """
    arch = (profile.arch or "").lower()
    if arch not in ("x86_64", "amd64", "x86", "i386", "i686"):
        return []
    isa = profile.instruction_sets
    if isa == InstructionSet.NONE or InstructionSet.AVX2 in isa:
        return []
    return ["AVX2"]


def is_illegal_instruction(err: BaseException) -> bool:
    text = str(err).lower()
    return any(code in text for code in _ILLEGAL_INSTRUCTION_CODES)


def cpu_unsupported_message(missing: list[str] | None = None) -> str:
    """User-facing reason a model cannot run on this CPU."""
    if missing:
        return (
            f"this CPU lacks {', '.join(missing)}. The inference engine needs AVX2, "
            "FMA and F16C (most PCs from 2013 on)."
        )
    return (
        "the inference engine used a CPU instruction this PC does not have "
        "(Windows error 0xC000001D). Update to the latest EigenCore; "
        "if it still happens, please report your CPU model."
    )


def kv_cache_types(kv_type: str, flash_attn: bool) -> tuple[int, int]:
    """(type_k, type_v) ggml enums for a --kv choice.

    llama.cpp only supports a quantized V cache with flash attention. Asking
    for a quantized KV cache with flash attention off is a contradiction, so it
    raises instead of quietly keeping half the cache at f16.
    """
    k = v = _KV_TYPE_MAP[kv_type]
    if v != GGML_TYPE_F16 and not flash_attn:
        raise ValueError(KV_NEEDS_FLASH.format(kv_type=kv_type))
    return k, v


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
    # Prompt-lookup decoding. Accepted but ignored: the old string-copy path
    # emitted prompt text the model never verified, which can change output.
    # A token-level draft + verify version is planned (E5).
    prompt_lookup: bool = False
    # GBNF grammar text (llama.cpp format). When set, sampling is constrained
    # to strings the grammar accepts (used by the ReAct agent).
    grammar: str | None = None


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    prompt_tokens: int
    optimization_stats: dict | None = None
    # Optimizations actually applied to the backend call. Empty today:
    # layer-skip/sparsity plans and the speculative decoder are computed and
    # reported but NOT wired into create_chat_completion yet.
    applied_optimizations: list[str] = field(default_factory=list)


class InferenceEngine:
    def __init__(
        self,
        model_path: Path,
        profile: HardwareProfile,
        model_spec: ModelSpec | None = None,
        kv_type: str = "fp16",
        flash_attn: bool = True,
        n_gpu_layers: int = 0,
    ):
        self.model_path = model_path
        self.profile = profile
        self.model_spec = model_spec
        # Load-time backend options: KV cache type ("fp16" default; "q8_0"
        # halves KV memory at a small quality cost) and flash attention.
        self.kv_type = kv_type
        self.flash_attn = flash_attn
        # 0 = CPU-only (default, safe). Set from the fit estimator's
        # gpu_layers_for_vram when a VRAM budget exists.
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self.n_ctx: int | None = None  # context length chosen at load()

        self._layer_skipper: LayerSkipScheduler | None = None
        self._sparsity_predictor: SparsityPredictor | None = None
        self._sparsity_cache = SparsityCache()
        self._speculative_decoder = SpeculativeDecoder(adaptive=True)
        self.last_pld_copied_chars = 0  # always 0: string-copy PLD is disabled
        self._grammar_cache: dict[str, object] = {}
        self.last_stream_stats: dict | None = None

    def load(self) -> None:
        """Load the model with hardware-optimized settings."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for inference. Install the backend extra: "
                'pip install "eigencore[inference]" '
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            )

        missing = missing_cpu_features(self.profile)
        if missing:
            raise RuntimeError(cpu_unsupported_message(missing))

        type_k, type_v = kv_cache_types(self.kv_type, self.flash_attn)
        ctx_length = self._default_context_length(type_v)
        self.n_ctx = ctx_length

        n_threads = self.profile.recommended_threads
        n_batch = self._compute_batch_size()

        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=ctx_length,
                n_threads=n_threads,
                n_threads_batch=n_threads,
                n_batch=n_batch,
                n_gpu_layers=self.n_gpu_layers,
                type_k=type_k,
                type_v=type_v,
                flash_attn=self.flash_attn,
                verbose=False,
            )
        except OSError as e:
            if is_illegal_instruction(e):
                raise RuntimeError(cpu_unsupported_message()) from e
            raise

        self._init_optimizers()

    def _default_context_length(self, type_v: int) -> int:
        """n_ctx from the fit estimator with the model's real GGUF geometry.

        Falls back to the profiler's geometry-free recommendation (clamped to
        the registry context) only when the GGUF geometry cannot be read.
        """
        from eigencore.fit.estimator import default_context_length
        from eigencore.fit.gguf_meta import read_gguf_model_info

        spec = self.model_spec
        info = read_gguf_model_info(self.model_path)
        if info is not None:
            ctx = default_context_length(
                info.geometry,
                self.profile.available_ram_gb,
                info.context_length or (spec.context_length if spec else None),
                # a q8_0 request with V kept at f16 is sized as f16 (conservative)
                kv_quant="q8_0" if type_v == GGML_TYPE_Q8_0 else "fp16",
                file_size_bytes=info.file_size_bytes,
                params_b=spec.params_b if spec else 0.0,
                quant=spec.quant if spec else "Q4_K_M",
            )
            if ctx is not None:
                return ctx

        ctx_length = self.profile.recommended_context_length
        if spec and spec.context_length < ctx_length:
            ctx_length = spec.context_length
        return ctx_length

    def _init_optimizers(self) -> None:
        """Initialize Phase 2 optimization modules based on loaded model.

        Metadata keys are architecture-prefixed (`{arch}.block_count`, ...),
        not always `llama.*` — resolve via the shared GGUF helper, falling
        back to raw metadata with an architecture prefix.
        """
        from eigencore.fit.gguf_meta import read_gguf_model_info

        info = read_gguf_model_info(self.model_path)
        if info is not None:
            num_layers = info.geometry.num_layers or 0
            hidden_size = info.geometry.hidden_size or 0
        else:
            metadata = dict(self._llm.metadata) if self._llm else {}
            arch = metadata.get("general.architecture", "llama")
            num_layers = int(
                metadata.get(f"{arch}.block_count", metadata.get("llama.block_count", 0))
            )
            hidden_size = int(
                metadata.get(f"{arch}.embedding_length", metadata.get("llama.embedding_length", 0))
            )

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

    def _grammar_kwargs(self, config: GenerationConfig) -> dict:
        """`grammar=` for create_chat_completion, compiled once per grammar text."""
        if not config.grammar:
            return {}
        cache = self.__dict__.setdefault("_grammar_cache", {})
        compiled = cache.get(config.grammar)
        if compiled is None:
            from llama_cpp import LlamaGrammar

            compiled = LlamaGrammar.from_string(config.grammar, verbose=False)
            cache[config.grammar] = compiled
        return {"grammar": compiled}

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate a complete response (non-streaming)."""
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if config is None:
            config = GenerationConfig(stream=False)

        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, config)

    def stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Stream tokens as they're generated."""
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if config is None:
            config = GenerationConfig(stream=True)

        messages = [{"role": "user", "content": prompt}]

        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or None,
            stream=True,
            **self._grammar_kwargs(config),
        ):
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

    def stream_messages(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Stream a chat completion from a full message list (multi-turn).

        `config.prompt_lookup` is ignored here: the string-copy shortcut emitted
        prompt text the model never produced. `last_pld_copied_chars` stays 0.
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        if config is None:
            config = GenerationConfig(stream=True)
        generated = ""
        self.last_pld_copied_chars = 0
        self.last_stream_stats = None  # set only when the stream completes
        t_start = time.perf_counter()
        t_first: float | None = None
        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or None,
            stream=True,
            **self._grammar_kwargs(config),
        ):
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                if t_first is None:
                    t_first = time.perf_counter()
                generated += token
                yield token
        t_end = time.perf_counter()
        # Measured stats for the UI. Token count comes from the model's own
        # tokenizer over the generated text (stream chunks are not tokens:
        # llama.cpp may merge several tokens into one chunk).
        try:
            n_tokens: int | None = self.count_tokens(generated)
        except Exception:
            n_tokens = None
        self.last_stream_stats = {
            "completion_tokens": n_tokens,
            "prefill_s": (t_first - t_start) if t_first is not None else None,
            "decode_s": (t_end - t_first) if t_first is not None else None,
            "token_source": "tokenizer (re-tokenized output)",
        }

    def count_tokens(self, text: str) -> int:
        """Number of tokens `text` encodes to with the loaded model's tokenizer."""
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        if not text:
            return 0
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
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
            **self._grammar_kwargs(config),
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
            applied_optimizations=self._applied_optimizations(),
        )

    def _applied_optimizations(self) -> list[str]:
        """Optimizations actually wired into the backend call.

        Currently none: layer-skip/sparsity only produce reported plans and
        the speculative decoder is never invoked during generation. Kept
        explicit so callers can distinguish measured from planned work.
        """
        return []

    def _collect_stats(self, config: GenerationConfig) -> dict:
        """Collect optimization statistics from Phase 2 modules."""
        stats: dict = {}
        stats["backend_wired"] = self._applied_optimizations()
        stats["note"] = (
            "stats are planned/reported only; no optimization is applied to the backend call yet"
        )

        if self._layer_skipper and config.enable_layer_skip:
            plan = self._layer_skipper.plan(complexity=config.complexity)
            stats["layer_skip"] = {
                "applied": False,
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
                    "applied": False,
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
            "applied": False,
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
    def layer_skipper(self) -> LayerSkipScheduler | None:
        return self._layer_skipper

    @property
    def sparsity_predictor(self) -> SparsityPredictor | None:
        return self._sparsity_predictor

    @property
    def speculative_decoder(self) -> SpeculativeDecoder:
        return self._speculative_decoder
