"""Stage C slice 4: prompt-lookup decoding matcher + engine flag (default OFF)."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.engine.prompt_lookup import longest_prompt_continuation
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec


def test_longest_continuation_basic():
    prompt = "def quicksort(arr): return arr"
    assert longest_prompt_continuation(prompt, "def quicksort(ar") == "r): return arr"


def test_longest_continuation_no_match():
    assert longest_prompt_continuation("hello world", "xyz") == ""
    assert longest_prompt_continuation("", "x") == ""
    assert longest_prompt_continuation("abc", "") == ""


def test_longest_continuation_match_at_end_returns_empty():
    # suffix matches but nothing follows in prompt → no continuation
    assert longest_prompt_continuation("abc", "abc") == ""


def _fake_llama():
    fake = types.ModuleType("llama_cpp")
    fake.Llama = MagicMock()
    sys.modules["llama_cpp"] = fake
    return fake


def _engine():
    fake = _fake_llama()
    fake.Llama.return_value = MagicMock(metadata={})
    hw = HardwareProfile(
        cpu_name="t",
        arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        instruction_sets=InstructionSet.AVX2,
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        estimated_bandwidth_gbps=25.0,
        max_model_params_b=7.0,
        optimal_quantization="Q4_K_M",
        estimated_tokens_per_sec=15.0,
        recommended_threads=4,
        recommended_context_length=4096,
    )
    spec = ModelSpec(
        name="t",
        repo_id="t/t",
        filename="t.gguf",
        params_b=1.1,
        quant="Q4_K_M",
        size_gb=0.67,
        task="general",
        context_length=2048,
        description="t",
    )
    engine = InferenceEngine(Path("dummy.gguf"), hw, spec)
    engine.load()
    return engine, fake


def _chunks(tokens):
    return [{"choices": [{"delta": {"content": t}}]} for t in tokens]


def test_pld_off_by_default():
    assert GenerationConfig().prompt_lookup is False


def test_pld_copies_verbatim_span_and_stays_aligned():
    engine, fake = _engine()
    prompt = "Please repeat: ALPHA-BETA-GAMMA end."
    # backend would emit the span token-by-token; PLD copies it in one go and
    # swallows the covered backend tokens
    fake.Llama.return_value.create_chat_completion.return_value = iter(
        _chunks(["Please repeat: ", "ALPHA", "-BETA", "-GAMMA", " end."])
    )
    cfg = GenerationConfig(stream=True, prompt_lookup=True)
    out = list(engine.stream_messages([{"role": "user", "content": prompt}], cfg))
    text = "".join(out)
    assert "ALPHA-BETA-GAMMA" in text
    assert text.endswith("end.")
    assert engine.last_pld_copied_chars >= len("ALPHA-BETA-GAMMA")


def test_pld_disabled_passes_backend_tokens_through():
    engine, fake = _engine()
    prompt = "repeat: XYZ"
    fake.Llama.return_value.create_chat_completion.return_value = iter(_chunks(["repeat: ", "XYZ"]))
    cfg = GenerationConfig(stream=True, prompt_lookup=False)
    out = list(engine.stream_messages([{"role": "user", "content": prompt}], cfg))
    assert "".join(out) == "repeat: XYZ"
    assert engine.last_pld_copied_chars == 0
