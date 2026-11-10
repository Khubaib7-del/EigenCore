"""Tests for speculative decoding (correct rejection sampling, measured-only speedup)."""

import numpy as np

from eigencore.engine.speculative import SpeculativeDecoder


def _verifier_lps(rows: list[list[float]]) -> np.ndarray:
    """Build (positions, vocab) log-prob matrix from per-row logits."""
    a = np.array(rows, dtype=np.float64)
    return (
        a
        - a.max(axis=1, keepdims=True)
        - np.log(np.exp(a - a.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
    )


class TestSpeculativeDecoder:
    def test_empty_draft_returns_empty(self):
        dec = SpeculativeDecoder()
        result = dec.verify_draft(
            draft_logprobs=np.array([]),
            verifier_logprobs=np.zeros((0, 5)),
            draft_token_ids=[],
        )
        assert result.accepted_tokens == []
        assert result.drafted_tokens == 0

    def test_all_tokens_accepted_when_verifier_dominates(self):
        dec = SpeculativeDecoder()
        rng = np.random.default_rng(0)
        draft_lps = np.array([-3.0, -3.0, -3.0])  # p_t small
        # verifier puts nearly all mass on the drafted token 4 → q_t/p_t >= 1 → always accept
        verifier = _verifier_lps([[-20.0, -20.0, -20.0, -20.0, 0.0]] * 3)
        tokens = [4, 4, 4]
        result = dec.verify_draft(draft_lps, verifier, tokens, rng=rng)
        assert result.accepted_tokens == [4, 4, 4]
        assert result.rejection_position is None

    def test_rejection_and_correction_when_verifier_disagrees(self):
        dec = SpeculativeDecoder()
        rng = np.random.default_rng(1)
        draft_lps = np.array([-0.1, -0.1])  # p_t ≈ 0.9, very confident
        # verifier puts ~0 mass on drafted token 0, mass on token 1
        verifier = _verifier_lps([[-20.0, 0.0], [-20.0, 0.0]])
        result = dec.verify_draft(draft_lps, verifier, [0, 0], rng=rng)
        assert result.accepted_tokens == []
        assert result.rejection_position == 0
        assert result.correction_token == 1  # residual puts all mass on token 1

    def test_accept_rate_calculation(self):
        dec = SpeculativeDecoder()
        rng = np.random.default_rng(2)
        # half the positions accept deterministically (q_t >= p_t), half reject
        draft_lps = np.array([-0.1, -0.1, -0.1, -0.1])
        verifier = _verifier_lps([[0.0, -20.0], [0.0, -20.0], [-20.0, 0.0], [-20.0, 0.0]])
        tokens = [0, 0, 0, 0]
        result = dec.verify_draft(draft_lps, verifier, tokens, rng=rng)
        assert result.accepted_count == 2
        assert result.rejection_position == 2
        assert result.accept_rate == 0.5

    def test_stats_accumulate_across_rounds(self):
        dec = SpeculativeDecoder()
        rng = np.random.default_rng(3)
        for _ in range(3):
            dec.verify_draft(
                np.array([-3.0, -3.0]),
                _verifier_lps([[0.0, -20.0], [0.0, -20.0]]),
                [0, 0],
                rng=rng,
            )
        assert dec.stats.total_rounds == 3
        assert dec.stats.total_drafted == 6
        assert dec.stats.total_accepted == 6

    def test_correction_preserves_target_distribution(self):
        """Empirical correction distribution over many rejections must match
        the residual norm(max(q - p, 0)) — i.e. the target distribution."""
        dec = SpeculativeDecoder()
        vocab = 6
        logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0, -30.0])  # token 5 ≈ impossible
        verifier = _verifier_lps([logits])
        q = np.exp(verifier[0])
        # draft is overconfident in the LEAST likely token (5) → always rejected
        p_t = 0.95
        draft_lps = np.array([np.log(p_t)])
        residual = np.maximum(q - p_t * np.eye(vocab)[5], 0.0)
        expected = residual / residual.sum()
        rng = np.random.default_rng(42)
        counts = np.zeros(vocab)
        trials = 4000
        for _ in range(trials):
            r = dec.verify_draft(draft_lps, verifier, [5], rng=rng)
            assert r.rejection_position == 0
            counts[r.correction_token] += 1
        np.testing.assert_allclose(counts / trials, expected, atol=0.03)


class TestDraftLengthAdaptation:
    def test_increases_on_high_accept(self):
        dec = SpeculativeDecoder(initial_draft_length=5, adaptive=True)
        rng = np.random.default_rng(4)
        draft_lps = np.array([-3.0] * 5)
        verifier = _verifier_lps([[0.0] + [-20.0] * 4] * 5)
        dec.verify_draft(draft_lps, verifier, [0] * 5, rng=rng)
        assert dec.draft_length >= 5

    def test_decreases_on_low_accept(self):
        dec = SpeculativeDecoder(initial_draft_length=5, adaptive=True)
        rng = np.random.default_rng(5)
        draft_lps = np.array([-0.1] * 5)
        verifier = _verifier_lps([[-20.0, 0.0] for _ in range(5)])
        dec.verify_draft(draft_lps, verifier, [0] * 5, rng=rng)
        assert dec.draft_length <= 5

    def test_respects_bounds(self):
        dec = SpeculativeDecoder(initial_draft_length=2, adaptive=True)
        rng = np.random.default_rng(6)
        draft_lps = np.array([-0.1, -0.1])
        verifier = _verifier_lps([[-20.0, 0.0], [-20.0, 0.0]])
        for _ in range(20):
            dec.verify_draft(draft_lps, verifier, [0, 0], rng=rng)
        assert dec.MIN_DRAFT_LENGTH <= dec.draft_length <= dec.MAX_DRAFT_LENGTH


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


class TestMeasuredSpeedup:
    def test_measured_speedup_from_times_only(self):
        dec = SpeculativeDecoder()
        # draft 10ms + verify 100ms vs verify-only 100ms → tokens per round higher
        assert dec.measured_speedup(draft_time_ms=10.0, verify_time_ms=100.0) >= 1.0
        assert dec.measured_speedup(draft_time_ms=0.0, verify_time_ms=100.0) == 1.0
        assert dec.measured_speedup(draft_time_ms=5.0, verify_time_ms=0.0) == 1.0
