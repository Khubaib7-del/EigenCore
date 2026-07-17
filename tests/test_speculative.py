"""Tests for CPU-optimized speculative decoding."""

import numpy as np
import pytest
from eigencore.engine.speculative import SpeculativeDecoder, SpeculativeResult


class TestSpeculativeDecoder:
    def test_empty_draft_returns_empty(self):
        dec = SpeculativeDecoder()
        result = dec.verify_draft(
            draft_logprobs=np.array([]),
            verifier_logprobs=np.array([]),
            draft_token_ids=[],
        )
        assert result.accepted_tokens == []
        assert result.drafted_tokens == 0

    def test_all_tokens_accepted_when_verifier_agrees(self):
        dec = SpeculativeDecoder(acceptance_threshold=0.3)
        draft_lps = np.array([-1.0, -1.5, -2.0])
        # verifier gives same or higher logprobs → ratio >= 1.0 → all accepted
        verifier_lps = np.array([-0.8, -1.2, -1.5])
        tokens = [100, 200, 300]
        result = dec.verify_draft(draft_lps, verifier_lps, tokens)
        assert result.accepted_count == 3
        assert result.accepted_tokens == [100, 200, 300]

    def test_rejection_at_divergence(self):
        dec = SpeculativeDecoder(acceptance_threshold=0.5)
        # draft model is confident, verifier disagrees strongly on token 2
        draft_lps = np.array([-0.5, -0.5, -0.5])
        verifier_lps = np.array([-0.3, -0.4, -5.0])
        tokens = [10, 20, 30]
        result = dec.verify_draft(draft_lps, verifier_lps, tokens)
        assert result.accepted_count == 2
        assert result.rejection_position == 2

    def test_accept_rate_calculation(self):
        dec = SpeculativeDecoder(acceptance_threshold=0.5)
        draft_lps = np.array([-1.0, -1.0, -1.0, -1.0])
        verifier_lps = np.array([-0.5, -0.5, -10.0, -10.0])
        result = dec.verify_draft(draft_lps, verifier_lps, [1, 2, 3, 4])
        assert result.accept_rate == 0.5

    def test_stats_accumulate_across_rounds(self):
        dec = SpeculativeDecoder(acceptance_threshold=0.3)
        for _ in range(3):
            dec.verify_draft(
                np.array([-1.0, -1.0]),
                np.array([-0.5, -0.5]),
                [1, 2],
            )
        assert dec.stats.total_rounds == 3
        assert dec.stats.total_drafted == 6
        assert dec.stats.total_accepted == 6


class TestDraftLengthAdaptation:
    def test_increases_on_high_accept(self):
        dec = SpeculativeDecoder(initial_draft_length=5, adaptive=True)
        # simulate high accept rate
        draft_lps = np.array([-1.0] * 5)
        verifier_lps = np.array([-0.5] * 5)
        dec.verify_draft(draft_lps, verifier_lps, list(range(5)))
        assert dec.draft_length >= 5  # should increase or stay

    def test_decreases_on_low_accept(self):
        dec = SpeculativeDecoder(initial_draft_length=5, adaptive=True, acceptance_threshold=0.9)
        # all tokens rejected (very strict threshold, verifier gives lower logprobs)
        draft_lps = np.array([-0.1] * 5)
        verifier_lps = np.array([-5.0] * 5)
        dec.verify_draft(draft_lps, verifier_lps, list(range(5)))
        assert dec.draft_length <= 5

    def test_respects_bounds(self):
        dec = SpeculativeDecoder(initial_draft_length=2, adaptive=True, acceptance_threshold=0.9)
        draft_lps = np.array([-0.1, -0.1])
        verifier_lps = np.array([-5.0, -5.0])
        for _ in range(20):
            dec.verify_draft(draft_lps, verifier_lps, [1, 2])
        assert dec.draft_length >= dec.MIN_DRAFT_LENGTH
        assert dec.draft_length <= dec.MAX_DRAFT_LENGTH


class TestCacheAwareDraftLength:
    def test_longer_drafts_when_cache_fits(self):
        dec = SpeculativeDecoder()
        length = dec.optimal_draft_length(l3_cache_mb=12.0, draft_model_mb=8.0)
        assert length >= dec.DEFAULT_DRAFT_LENGTH

    def test_shorter_drafts_when_cache_small(self):
        dec = SpeculativeDecoder()
        length = dec.optimal_draft_length(l3_cache_mb=2.0, draft_model_mb=8.0)
        assert length <= dec.DEFAULT_DRAFT_LENGTH

    def test_handles_zero_model_size(self):
        dec = SpeculativeDecoder()
        length = dec.optimal_draft_length(l3_cache_mb=12.0, draft_model_mb=0.0)
        assert length == dec.DEFAULT_DRAFT_LENGTH


class TestSpeedupEstimate:
    def test_perfect_accept_rate_gives_high_speedup(self):
        dec = SpeculativeDecoder(initial_draft_length=8)
        speedup = dec.estimate_speedup(accept_rate=1.0, draft_cost_ratio=0.1)
        assert speedup > 3.0

    def test_zero_accept_rate_gives_low_speedup(self):
        dec = SpeculativeDecoder(initial_draft_length=8)
        speedup = dec.estimate_speedup(accept_rate=0.0, draft_cost_ratio=0.1)
        assert speedup < 1.2

    def test_higher_draft_cost_reduces_speedup(self):
        dec = SpeculativeDecoder(initial_draft_length=5)
        fast = dec.estimate_speedup(accept_rate=0.8, draft_cost_ratio=0.05)
        slow = dec.estimate_speedup(accept_rate=0.8, draft_cost_ratio=0.5)
        assert fast > slow
