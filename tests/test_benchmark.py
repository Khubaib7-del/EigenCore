"""Tests for the Phase 1 benchmark/evaluation harness (no real model needed)."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.benchmark import (
    DEFAULT_CASES,
    BenchmarkReport,
    QualityCase,
    build_report,
    check_quality,
    measure_case,
    run_cases,
    sha256_file,
    summarize,
)
from eigencore.engine.inference import GenerationConfig
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec


def _profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_name="Test CPU",
        arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=InstructionSet.AVX2 | InstructionSet.SSE2,
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        estimated_bandwidth_gbps=25.0,
        max_model_params_b=7.0,
        optimal_quantization="Q4_K_M",
        estimated_tokens_per_sec=15.0,
        recommended_threads=4,
        recommended_context_length=4096,
    )


def _spec() -> ModelSpec:
    return ModelSpec(
        name="test-model",
        repo_id="test/test",
        filename="test.gguf",
        params_b=1.1,
        quant="Q4_K_M",
        size_gb=0.67,
        task="general",
        context_length=2048,
        description="Test model",
    )


class FakeEngine:
    """Stands in for InferenceEngine; streams deterministic chunks."""

    def __init__(self, chunks=None):
        self.profile = _profile()
        self.model_spec = _spec()
        self._chunks = chunks or ["def ", "quicksort", "():", " return", " ok"]

    def stream(self, prompt, config):
        yield from self._chunks


def test_sha256_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"eigencore")
        path = f.name
    assert sha256_file(path) == sha256_of(b"eigencore")
    Path(path).unlink()


def sha256_of(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_check_quality_pass_and_fail():
    case = QualityCase(id="c", prompt="p", expected_contains=["TCP", "UDP"], min_tokens=2)
    ok, detail = check_quality(case, "TCP vs UDP explained", 5)
    assert ok and detail == "ok"
    ok, detail = check_quality(case, "TCP only", 5)
    assert not ok and "UDP" in detail
    ok, detail = check_quality(case, "TCP UDP", 1)
    assert not ok and "too few tokens" in detail


def test_measure_case_separates_ttft_and_decode():
    engine = FakeEngine()
    case = QualityCase(
        id="c", prompt="hello", expected_contains=["ok"], min_tokens=1, max_tokens=16
    )
    config = GenerationConfig(stream=False)
    result = measure_case(engine, case, config)
    # no backend tokenizer on FakeEngine → chars/4 fallback: "def quicksort(): return ok" = 26 chars → 6
    assert result.completion_tokens == max(len("def quicksort(): return ok") // 4, 1)
    assert result.total_time_s >= result.time_to_first_token_s >= 0
    assert result.decode_time_s >= 0
    assert result.quality_pass


def test_run_cases_repeats():
    engine = FakeEngine()
    results = run_cases(engine, DEFAULT_CASES[:2], runs=2)
    assert len(results) == 4


def test_summarize():
    engine = FakeEngine()
    results = run_cases(engine, DEFAULT_CASES[:2], runs=2)
    s = summarize(results)
    assert s["cases"] == 4
    assert 0.0 <= s["quality_pass_rate"] <= 1.0
    assert s["all_decode_tok_s_median"] >= 0
    assert "cold" in s and "warm" in s
    assert len(s["cold"]) == 5 and len(s["warm"]) == 5


def test_measure_case_counts_tokens_via_tokenizer():
    class TokEngine(FakeEngine):
        def stream(self, prompt, config):
            for c in ["aa", "bb"]:
                yield c

        @property
        def _llm(self):
            class L:
                def tokenize(self, b):
                    return [0] * (len(b) // 2)  # 2 chars per token

            return L()

    case = QualityCase(id="c", prompt="p", expected_contains=["aabb"], min_tokens=1, max_tokens=8)
    result = measure_case(TokEngine(), case, GenerationConfig(stream=False))
    assert result.completion_tokens == 2  # tokenizer: 4 chars / 2 = 2, not 2 chunks
    assert result.warm is False
    assert measure_case(TokEngine(), case, GenerationConfig(stream=False), run_index=1).warm is True


def test_report_roundtrip_with_model_file():
    engine = FakeEngine()
    spec = _spec()
    with tempfile.TemporaryDirectory() as d:
        model_path = Path(d) / "test.gguf"
        model_path.write_bytes(b"fake-gguf-bytes")
        config = GenerationConfig(stream=False, temperature=0.0)
        results = run_cases(engine, DEFAULT_CASES[:1], config, runs=1)
        report = build_report(engine, spec, model_path, config, results)
        assert isinstance(report, BenchmarkReport)
        out = report.save(Path(d) / "reports" / "run.json")
        loaded = json.loads(out.read_text())
        assert loaded["schema_version"] == 1
        assert loaded["model"]["sha256"] == sha256_of(b"fake-gguf-bytes")
        assert loaded["hardware"]["cpu_name"] == "Test CPU"
        assert "AVX2" in loaded["hardware"]["instruction_sets"]
        assert loaded["engine"]["backend"] == "llama-cpp-python"
        assert loaded["summary"]["cases"] == 1


def test_report_handles_missing_model_file():
    engine = FakeEngine()
    report = build_report(engine, _spec(), Path("/nonexistent.gguf"), GenerationConfig(), [])
    assert report.model["sha256"] is None
    assert "warning" in report.model
