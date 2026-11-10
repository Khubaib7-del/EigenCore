"""
Benchmark and evaluation harness — Phase 1 (trustworthy baseline).

Produces reproducible JSON evidence artifacts: hardware snapshot, model file
identity (SHA-256), engine configuration, per-case prefill/decode timing,
decode throughput, and simple quality checks. No result from this harness is
a performance claim until the artifact (with its settings) is shared alongside it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.hal.profiler import HardwareProfile
from eigencore.models.registry import ModelSpec

SCHEMA_VERSION = 1


@dataclass
class QualityCase:
    """One evaluation case: a prompt plus cheap, checkable expectations."""

    id: str
    prompt: str
    expected_contains: list[str] = field(default_factory=list)
    min_tokens: int = 1
    max_tokens: int = 256
    note: str = ""


DEFAULT_CASES: list[QualityCase] = [
    QualityCase(
        id="explain-cpu",
        prompt="Explain how a CPU executes instructions, step by step, in one short paragraph.",
        expected_contains=["fetch"],
        max_tokens=128,
        note="basic technical explanation",
    ),
    QualityCase(
        id="quicksort",
        prompt="Write a Python function that implements quicksort. Reply with code only.",
        expected_contains=["def ", "quicksort", "return"],
        max_tokens=192,
        note="code generation sanity check",
    ),
    QualityCase(
        id="tcp-udp",
        prompt="List two key differences between TCP and UDP in one sentence each.",
        expected_contains=["TCP", "UDP"],
        max_tokens=96,
        note="factual recall",
    ),
    QualityCase(
        id="json-format",
        prompt='Reply with exactly this JSON and nothing else: {"ok": true}',
        expected_contains=['{"ok": true}'],
        max_tokens=32,
        note="instruction following / format adherence",
    ),
]


@dataclass
class CaseResult:
    case_id: str
    prompt: str
    output: str
    prompt_tokens_estimated: int
    completion_tokens: int
    time_to_first_token_s: float
    decode_time_s: float
    total_time_s: float
    decode_tokens_per_second: float
    quality_pass: bool
    quality_detail: str
    run_index: int = 0
    warm: bool = False


@dataclass
class BenchmarkReport:
    schema_version: int
    created_at: str
    git_commit: str | None
    eigencore_version: str
    hardware: dict[str, Any]
    model: dict[str, Any]
    engine: dict[str, Any]
    results: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a model file — ties results to exact model content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_snapshot(profile: HardwareProfile) -> dict[str, Any]:
    return {
        "cpu_name": profile.cpu_name,
        "arch": profile.arch,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "physical_cores": profile.physical_cores,
        "logical_cores": profile.logical_cores,
        "instruction_sets": [
            f.name for f in type(profile.instruction_sets) if f in profile.instruction_sets
        ],
        "total_ram_gb": profile.total_ram_gb,
        "available_ram_gb": profile.available_ram_gb,
        "estimated_bandwidth_gbps": profile.estimated_bandwidth_gbps,
        "notes": list(profile.notes),
    }


def model_snapshot(spec: ModelSpec, model_path: Path | None) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "name": spec.name,
        "repo_id": spec.repo_id,
        "filename": spec.filename,
        "params_b": spec.params_b,
        "quant": spec.quant,
        "size_gb": spec.size_gb,
        "task": spec.task,
        "context_length": spec.context_length,
    }
    if model_path is not None and Path(model_path).exists():
        snap["local_path"] = str(model_path)
        snap["sha256"] = sha256_file(model_path)
        snap["file_size_bytes"] = Path(model_path).stat().st_size
    else:
        snap["local_path"] = None
        snap["sha256"] = None
        snap["warning"] = "model file not found; sha256 not recorded"
    return snap


def engine_snapshot(engine: InferenceEngine, config: GenerationConfig) -> dict[str, Any]:
    p = engine.profile
    return {
        "n_ctx": min(p.recommended_context_length, engine.model_spec.context_length)
        if engine.model_spec
        else p.recommended_context_length,
        "n_threads": p.recommended_threads,
        "n_gpu_layers": 0,
        "backend": "llama-cpp-python",
        "generation": {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repeat_penalty": config.repeat_penalty,
            "enable_layer_skip": config.enable_layer_skip,
            "enable_sparse_inference": config.enable_sparse_inference,
        },
    }


def check_quality(case: QualityCase, output: str, completion_tokens: int) -> tuple[bool, str]:
    """Cheap deterministic quality gate; not a substitute for human eval."""
    problems: list[str] = []
    lowered = output.lower()
    for needle in case.expected_contains:
        if needle.lower() not in lowered:
            problems.append(f"missing expected text: {needle!r}")
    if completion_tokens < case.min_tokens:
        problems.append(f"too few tokens: {completion_tokens} < {case.min_tokens}")
    return (not problems, "; ".join(problems) if problems else "ok")


def measure_case(
    engine: InferenceEngine,
    case: QualityCase,
    config: GenerationConfig,
    run_index: int = 0,
) -> CaseResult:
    """Run one case via streaming so prefill (TTFT) and decode are separated.

    Completion tokens are counted with the backend tokenizer (or response
    usage), never by counting stream chunks. `run_index` 0 is tagged cold.
    """
    if config.stream:
        raise ValueError("measure_case requires a non-streaming config; it streams internally")

    stream_config = GenerationConfig(
        max_tokens=case.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repeat_penalty=config.repeat_penalty,
        stream=True,
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    for chunk in engine.stream(case.prompt, stream_config):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        chunks.append(chunk)
    total = time.perf_counter() - start

    ttft = (first_token_at - start) if first_token_at is not None else total
    output = "".join(chunks)
    completion_tokens = _count_tokens(engine, output)
    decode_time = max(total - ttft, 0.0)
    decode_tps = completion_tokens / decode_time if decode_time > 0 else 0.0
    quality_pass, quality_detail = check_quality(case, output, completion_tokens)

    return CaseResult(
        case_id=case.id,
        prompt=case.prompt,
        output=output,
        prompt_tokens_estimated=max(len(case.prompt) // 4, 1),
        completion_tokens=completion_tokens,
        time_to_first_token_s=ttft,
        decode_time_s=decode_time,
        total_time_s=total,
        decode_tokens_per_second=decode_tps,
        quality_pass=quality_pass,
        quality_detail=quality_detail,
        run_index=run_index,
        warm=run_index > 0,
    )


def _count_tokens(engine: InferenceEngine, text: str) -> int:
    """Token count from the backend tokenizer; falls back to a chars/4 estimate."""
    llm = getattr(engine, "_llm", None)
    if llm is not None and hasattr(llm, "tokenize"):
        try:
            return len(llm.tokenize(text.encode("utf-8")))
        except Exception:
            pass
    return max(len(text) // 4, 1)


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    def _stats(rows: list[CaseResult]) -> dict[str, float]:
        tps = [r.decode_tokens_per_second for r in rows]
        ttft = [r.time_to_first_token_s for r in rows]
        return {
            "decode_tok_s_median": statistics.median(tps) if tps else 0.0,
            "decode_tok_s_mean": statistics.mean(tps) if tps else 0.0,
            "decode_tok_s_min": min(tps) if tps else 0.0,
            "decode_tok_s_max": max(tps) if tps else 0.0,
            "ttft_s_median": statistics.median(ttft) if ttft else 0.0,
        }

    cold = [r for r in results if not r.warm]
    warm = [r for r in results if r.warm]
    passed = sum(1 for r in results if r.quality_pass)
    return {
        "cases": len(results),
        "quality_pass_rate": passed / len(results) if results else 0.0,
        **{f"all_{k}": v for k, v in _stats(results).items()},
        "cold": _stats(cold),
        "warm": _stats(warm),
        "note": "tokens counted via backend tokenizer; cold = first run per case, warm = later runs",
    }


def build_report(
    engine: InferenceEngine,
    spec: ModelSpec,
    model_path: Path | None,
    config: GenerationConfig,
    results: list[CaseResult],
) -> BenchmarkReport:
    from eigencore import __version__

    return BenchmarkReport(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        eigencore_version=__version__,
        hardware=hardware_snapshot(engine.profile),
        model=model_snapshot(spec, model_path),
        engine=engine_snapshot(engine, config),
        results=[asdict(r) for r in results],
        summary=summarize(results),
    )


def run_cases(
    engine: InferenceEngine,
    cases: list[QualityCase],
    config: GenerationConfig | None = None,
    runs: int = 1,
) -> list[CaseResult]:
    """Run each case `runs` times (first run is cold/warm-up, all are recorded)."""
    if config is None:
        config = GenerationConfig(
            max_tokens=256,
            temperature=0.0,
            stream=False,
            enable_layer_skip=False,
            enable_sparse_inference=False,
        )
    results: list[CaseResult] = []
    for case in cases:
        for run_idx in range(max(runs, 1)):
            results.append(measure_case(engine, case, config, run_index=run_idx))
    return results
