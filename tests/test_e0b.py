"""E0b: GBNF reaches the ReAct calls, --pld is ignored, frontends share Forge's load path."""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from eigencore.agent.grammar import GBNFBuilder
from eigencore.agent.react import ReActAgent, StepKind
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry
from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.hal.profiler import HardwareProfile, InstructionSet
from eigencore.models.registry import ModelSpec


def _noop(**_):
    return "ok"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="read_file",
            description="read a file",
            handler=_noop,
            parameters=[
                ToolParam(name="path", type="string", description="p"),
                ToolParam(name="limit", type="integer", description="l", required=False),
            ],
        )
    )
    reg.register(Tool(name="now", description="time", handler=_noop))
    return reg


def _profile() -> HardwareProfile:
    return HardwareProfile(
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


def _spec(name="t", task="general") -> ModelSpec:
    return ModelSpec(
        name=name,
        repo_id="t/t",
        filename=f"{name}.gguf",
        params_b=1.1,
        quant="Q4_K_M",
        size_gb=0.67,
        task=task,
        context_length=2048,
        description="t",
    )


# ----- grammar ----------------------------------------------------------------------


def _rule_names(grammar: str) -> list[str]:
    return [line.split("::=")[0].strip() for line in grammar.splitlines() if "::=" in line]


def test_rule_names_are_valid_llama_cpp_identifiers():
    # llama.cpp rejects '_' in rule names ("expecting newline or end at _call")
    for g in (
        GBNFBuilder.react_grammar(_registry()),
        GBNFBuilder.tool_call_grammar(_registry()),
        GBNFBuilder.react_grammar(ToolRegistry()),
    ):
        for name in _rule_names(g):
            assert re.fullmatch(r"[A-Za-z0-9-]+", name), name


def test_every_referenced_rule_is_defined():
    g = GBNFBuilder.react_grammar(_registry())
    defined = set(_rule_names(g))
    bodies = " ".join(line.split("::=", 1)[1] for line in g.splitlines() if "::=" in line)
    # strip quoted literals and character classes, then collect bare identifiers
    bare = re.sub(r'"(?:[^"\\]|\\.)*"|\[(?:[^\]\\]|\\.)*\]|\{[0-9,]+\}', " ", bodies)
    used = set(re.findall(r"[A-Za-z][A-Za-z0-9-]*", bare))
    assert used <= defined, used - defined


def test_react_grammar_matches_parse_step_shapes():
    g = GBNFBuilder.react_grammar(_registry())
    # the three shapes parse_step understands, and nothing else at the root
    assert '"\\"thought\\""' in g and '"\\"final\\""' in g and '"\\"action\\""' in g
    assert '"\\"content\\""' in g and '"\\"tool\\""' in g and '"\\"args\\""' in g
    assert g.splitlines()[0] == 'root ::= "{" ws (thought | action | final) ws "}"'


def test_optional_params_are_optional_in_grammar():
    g = GBNFBuilder.react_grammar(_registry())
    line = next(x for x in g.splitlines() if x.startswith("call-read-file ::="))
    assert '("\\"limit\\""' not in line  # not required ...
    assert '(ws "," ws "\\"limit\\"" ws ":" ws integer)?' in line  # ... but allowed


def test_no_tools_grammar_has_no_action_branch():
    g = GBNFBuilder.react_grammar(ToolRegistry())
    assert "action" not in g.splitlines()[0]
    assert "tool-call" not in g


def test_whitespace_and_numbers_are_bounded():
    # unbounded ws/digits let a model spin until max_tokens and break the JSON
    g = GBNFBuilder.react_grammar(_registry())
    assert "ws ::= [ \\t\\n]*" not in g
    assert "[0-9]+" not in g


def test_agent_exposes_grammar():
    agent = ReActAgent(_registry())
    assert agent.grammar == GBNFBuilder.react_grammar(agent.registry)


@pytest.mark.parametrize(
    "raw,kind",
    [
        ('{"type": "thought", "content": "plan"}', StepKind.THOUGHT),
        ('{"type":"action","tool":"read_file","args":{"path":"a.txt"}}', StepKind.ACTION),
        ('{"type": "final", "content": "done"}', StepKind.FINAL),
    ],
)
def test_grammar_shapes_parse(raw, kind):
    json.loads(raw)
    assert ReActAgent(_registry()).parse_step(raw).kind is kind


# ----- engine passes grammar --------------------------------------------------------


def _engine_with_fake_llama(monkeypatch):
    fake = types.ModuleType("llama_cpp")
    fake.Llama = MagicMock()
    fake.LlamaGrammar = MagicMock()
    fake.LlamaGrammar.from_string.side_effect = lambda text, verbose=False: ("G", text)
    llm = MagicMock(metadata={})
    fake.Llama.return_value = llm
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    engine = InferenceEngine(Path("dummy.gguf"), _profile(), _spec())
    engine.load()
    return engine, llm, fake


def test_chat_passes_compiled_grammar(monkeypatch):
    engine, llm, fake = _engine_with_fake_llama(monkeypatch)
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"type":"final","content":"x"}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    g = GBNFBuilder.react_grammar(_registry())
    engine.chat([{"role": "user", "content": "q"}], GenerationConfig(stream=False, grammar=g))
    engine.chat([{"role": "user", "content": "q"}], GenerationConfig(stream=False, grammar=g))
    assert llm.create_chat_completion.call_args.kwargs["grammar"] == ("G", g)
    fake.LlamaGrammar.from_string.assert_called_once()  # compiled once, then cached


def test_stream_passes_grammar_and_no_grammar_by_default(monkeypatch):
    engine, llm, _ = _engine_with_fake_llama(monkeypatch)
    llm.create_chat_completion.return_value = iter([])
    list(engine.stream_messages([{"role": "user", "content": "q"}], GenerationConfig()))
    assert "grammar" not in llm.create_chat_completion.call_args.kwargs
    llm.create_chat_completion.return_value = iter([])
    list(
        engine.stream_messages(
            [{"role": "user", "content": "q"}], GenerationConfig(grammar='root ::= "a"')
        )
    )
    assert llm.create_chat_completion.call_args.kwargs["grammar"] == ("G", 'root ::= "a"')


def test_forge_run_agent_sends_react_grammar(monkeypatch):
    from eigencore.forge import Forge

    forge = Forge.__new__(Forge)
    seen = []

    class _Eng:
        is_loaded = True

        def chat(self, messages, config):
            seen.append(config.grammar)
            return SimpleNamespace(text='{"type": "final", "content": "42"}')

    forge._engine = _Eng()
    forge._ensure_loaded = lambda prompt=None: None
    reg = _registry()
    result = forge.run_agent("what is 6*7", reg, max_steps=2)
    assert result.answer == "42"
    assert seen == [GBNFBuilder.react_grammar(reg)]


# ----- one load path ------------------------------------------------------------------


def test_open_engine_builds_with_options(tmp_path):
    from eigencore.forge import open_engine

    reg = SimpleNamespace(cache_dir=tmp_path)
    eng = open_engine(_spec(), _profile(), reg, kv_type="q8_0", flash_attn=True, n_gpu_layers=3)
    assert isinstance(eng, InferenceEngine)
    assert eng.model_path == _spec().local_path(tmp_path)
    assert (eng.kv_type, eng.flash_attn, eng.n_gpu_layers) == ("q8_0", True, 3)


def test_desktop_backend_uses_forge_load_path(monkeypatch, tmp_path):
    import eigencore.desktop.backend as backend_mod

    made = []
    monkeypatch.setattr(
        backend_mod, "open_engine", lambda spec, profile, registry: made.append(spec) or "E"
    )
    b = backend_mod.DesktopBackend(
        registry=SimpleNamespace(cache_dir=tmp_path, list_models=lambda: []),
        profile=_profile(),
    )
    assert b._default_engine_factory(_spec()) == "E"
    assert made == [_spec()]


def test_pick_model_routes_prompt_only_for_general_task():
    from eigencore.forge import pick_model

    code, general = _spec("coder", "code"), _spec("gen", "general")
    reg = SimpleNamespace(recommend=lambda profile, task: code if task == "code" else general)
    spec, decision = pick_model(_profile(), reg, "write a python function to sort a list")
    assert spec is code and decision is not None
    spec, decision = pick_model(_profile(), reg, "write a python function", task="general")
    assert spec is code
    spec, decision = pick_model(_profile(), reg, None, task="general")
    assert spec is general and decision is None
    spec, decision = pick_model(_profile(), reg, "write python code", task="math")
    assert decision is None


# ----- CLI -----------------------------------------------------------------------------


def test_cli_pld_is_ignored_with_a_warning(monkeypatch):
    import eigencore.cli.main as m

    monkeypatch.setattr(m, "profile_hardware", _profile)

    class _Stop(Exception):
        pass

    def _boom(*a, **k):
        raise _Stop

    monkeypatch.setattr(m, "ModelRegistry", _boom)
    res = CliRunner().invoke(m.cli, ["run", "-p", "hi", "--pld"])
    assert isinstance(res.exception, _Stop)
    assert "--pld is ignored" in res.output


def test_cli_pld_hidden_from_help():
    import eigencore.cli.main as m

    res = CliRunner().invoke(m.cli, ["run", "--help"])
    assert "--pld" not in res.output
