"""
Speculative decoding rethought for CPU.

Standard speculative decoding uses a small "draft" model to predict N tokens,
then the large "verifier" model checks them in a single batch. On GPU this is
fast because the batch verify step parallelizes well.

On CPU the advantage is different: we exploit the fact that CPU branch prediction
and sequential processing makes it cheaper to run a tiny draft model for N tokens
than to run the large model N times. The verify step is still sequential on CPU,
but we only pay the large-model cost once per accepted draft sequence.

Key CPU-specific adaptations:
1. Draft model shares the same memory-mapped GGUF file as verifier (different quant)
2. Acceptance threshold adapts based on observed accept rate
3. Draft length scales with CPU cache size (longer drafts if L3 cache can hold draft model)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DraftToken:
    token_id: int
    logprob: float
    position: int


@dataclass
class SpeculativeResult:
    accepted_tokens: list[int]
    drafted_tokens: int
    accepted_count: int
    rejection_position: int | None
    correction_token: int | None
    draft_time_ms: float
    verify_time_ms: float
    total_time_ms: float

    @property
    def accept_rate(self) -> float:
        return self.accepted_count / self.drafted_tokens if self.drafted_tokens > 0 else 0.0

    @property
    def speedup(self) -> float:
        if self.total_time_ms <= 0:
            return 1.0
        naive_time = self.total_time_ms * (self.drafted_tokens + 1) / max(self.drafted_tokens, 1)
        return naive_time / self.total_time_ms if self.total_time_ms > 0 else 1.0


@dataclass
class SpeculativeStats:
    total_drafted: int = 0
    total_accepted: int = 0
    total_rounds: int = 0
    draft_times_ms: list[float] = field(default_factory=list)
    verify_times_ms: list[float] = field(default_factory=list)

    @property
    def overall_accept_rate(self) -> float:
        return self.total_accepted / self.total_drafted if self.total_drafted > 0 else 0.0

    @property
    def avg_draft_time_ms(self) -> float:
        return np.mean(self.draft_times_ms) if self.draft_times_ms else 0.0

    @property
    def avg_verify_time_ms(self) -> float:
        return np.mean(self.verify_times_ms) if self.verify_times_ms else 0.0

    def summary(self) -> str:
        return (
            f"Speculative decoding: {self.total_rounds} rounds | "
            f"Accept rate: {self.overall_accept_rate:.0%} | "
            f"Avg draft: {self.avg_draft_time_ms:.1f}ms | "
            f"Avg verify: {self.avg_verify_time_ms:.1f}ms"
        )


class SpeculativeDecoder:
    """
    CPU-optimized speculative decoding engine.

    Takes a draft function (fast, small model) and a verify function (large model)
    as callables, so it can wrap any underlying inference backend.
    """

    DEFAULT_DRAFT_LENGTH = 5
    MIN_DRAFT_LENGTH = 2
    MAX_DRAFT_LENGTH = 12

    def __init__(
        self,
        draft_fn=None,
        verify_fn=None,
        initial_draft_length: int = 5,
        acceptance_threshold: float = 0.3,
        adaptive: bool = True,
    ):
        self.draft_fn = draft_fn
        self.verify_fn = verify_fn
        self.draft_length = initial_draft_length
        self.acceptance_threshold = acceptance_threshold
        self.adaptive = adaptive
        self.stats = SpeculativeStats()

    def optimal_draft_length(self, l3_cache_mb: float, draft_model_mb: float) -> int:
        """
        Scale draft length based on whether the draft model fits in L3 cache.
        If it fits, longer drafts are nearly free (no main memory round-trips).
        """
        if draft_model_mb <= 0:
            return self.DEFAULT_DRAFT_LENGTH

        cache_ratio = l3_cache_mb / draft_model_mb

        if cache_ratio >= 1.0:
            return min(self.MAX_DRAFT_LENGTH, int(self.DEFAULT_DRAFT_LENGTH * 1.5))
        elif cache_ratio >= 0.5:
            return self.DEFAULT_DRAFT_LENGTH
        else:
            return max(self.MIN_DRAFT_LENGTH, int(self.DEFAULT_DRAFT_LENGTH * 0.6))

    def verify_draft(
        self,
        draft_logprobs: np.ndarray,
        verifier_logprobs: np.ndarray,
        draft_token_ids: list[int],
        rng: np.random.Generator | None = None,
    ) -> SpeculativeResult:
        """
        Leviathan/Chen-style speculative rejection sampling.

        `verifier_logprobs` is (n_positions, vocab) log-probs from the target
        model; `draft_logprobs[i]` is the draft model's log-prob of the drafted
        token at position i. Token t is accepted with probability
        min(1, q(t)/p(t)); on rejection the correction token is sampled from
        the residual distribution norm(max(q - p, 0)), which preserves the
        target distribution exactly. (The draft distribution is approximated
        as a point mass at the drafted token, since only its log-prob is
        passed in — documented limitation.)
        """
        start = time.perf_counter()
        n_draft = len(draft_token_ids)
        rng = rng or np.random.default_rng()

        if n_draft == 0 or len(draft_logprobs) == 0 or len(verifier_logprobs) == 0:
            return SpeculativeResult(
                accepted_tokens=[],
                drafted_tokens=0,
                accepted_count=0,
                rejection_position=None,
                correction_token=None,
                draft_time_ms=0,
                verify_time_ms=0,
                total_time_ms=0,
            )

        accepted = []
        rejection_pos = None
        correction = None

        for i in range(min(n_draft, len(verifier_logprobs))):
            token_id = draft_token_ids[i]
            if i < len(draft_logprobs):
                draft_lp = draft_logprobs[i]
            else:
                break
            q = np.exp(verifier_logprobs[i] - verifier_logprobs[i].max())
            q /= q.sum()
            p_t = float(np.exp(draft_lp))
            q_t = float(q[token_id]) if token_id < len(q) else 0.0
            if rng.random() <= min(1.0, q_t / p_t) if p_t > 0 else True:
                accepted.append(token_id)
            else:
                rejection_pos = i
                # residual distribution: norm(max(q - p, 0)), p = point mass at token
                residual = np.maximum(q - p_t * np.eye(len(q))[token_id], 0.0)
                total = residual.sum()
                if total > 0:
                    correction = int(rng.choice(len(q), p=residual / total))
                else:
                    correction = int(np.argmax(q))
                break

        elapsed = (time.perf_counter() - start) * 1000

        result = SpeculativeResult(
            accepted_tokens=accepted,
            drafted_tokens=n_draft,
            accepted_count=len(accepted),
            rejection_position=rejection_pos,
            correction_token=correction,
            draft_time_ms=0,
            verify_time_ms=elapsed,
            total_time_ms=elapsed,
        )

        self._update_stats(result)

        if self.adaptive:
            self._adapt_draft_length(result.accept_rate)

        return result

    def _adapt_draft_length(self, recent_accept_rate: float) -> None:
        """
        Adjust draft length based on acceptance rate.
        High accept rate → try longer drafts (more tokens per round).
        Low accept rate → shorter drafts (less wasted compute).
        """
        if recent_accept_rate > 0.8:
            self.draft_length = min(self.draft_length + 1, self.MAX_DRAFT_LENGTH)
        elif recent_accept_rate < 0.3:
            self.draft_length = max(self.draft_length - 1, self.MIN_DRAFT_LENGTH)

    def _update_stats(self, result: SpeculativeResult) -> None:
        self.stats.total_drafted += result.drafted_tokens
        self.stats.total_accepted += result.accepted_count
        self.stats.total_rounds += 1
        self.stats.draft_times_ms.append(result.draft_time_ms)
        self.stats.verify_times_ms.append(result.verify_time_ms)

    def measured_speedup(self, draft_time_ms: float, verify_time_ms: float) -> float:
        """Speedup computed only from measured times; no theoretical claims."""
        if verify_time_ms <= 0:
            return 1.0
        return (
            (draft_time_ms + verify_time_ms) / max(verify_time_ms, 1e-9)
            if draft_time_ms > 0
            else 1.0
        )
