"""Registry v2: models.json is the source of truth for the curated list."""

import json

import pytest

from eigencore.hal.profiler import HardwareProfile
from eigencore.models import registry as reg
from eigencore.models.registry import (
    BUNDLED_MODELS_JSON,
    REGISTRY,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    ModelRegistry,
    load_registry,
    parse_models,
    runtime_supports,
)

OLD_NAMES = [
    "tinyllama-1.1b",
    "phi-3-mini",
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-coder-1.5b",
    "qwen2.5-7b",
    "qwen3-30b-a3b",
]


def _doc() -> dict:
    return json.loads(BUNDLED_MODELS_JSON.read_text(encoding="utf-8"))


def _profile(ram_gb: float) -> HardwareProfile:
    from unittest.mock import MagicMock

    p = MagicMock(spec=HardwareProfile)
    p.available_ram_gb = ram_gb
    return p


def test_bundled_json_loads_with_schema_v2():
    doc = _doc()
    assert doc["schema_version"] == SCHEMA_VERSION == 2
    specs = load_registry(remote=False)
    assert len(specs) == len(doc["models"]) >= 9
    assert [s.name for s in specs] == [s.name for s in REGISTRY]


def test_every_entry_has_required_fields():
    for entry in _doc()["models"]:
        missing = [k for k in REQUIRED_FIELDS if k not in entry]
        assert not missing, (entry.get("name"), missing)
        assert isinstance(entry["tags"], list) and entry["tags"]
        assert isinstance(entry["noncommercial"], bool)
        assert entry["filename"].endswith(".gguf")
        assert entry["size_gb"] > 0 and entry["context_length"] > 0
        assert entry["size_bytes"] > 0
        assert abs(entry["size_bytes"] / 1e9 - entry["size_gb"]) < 0.01


def test_names_are_unique():
    names = [s.name for s in REGISTRY]
    assert len(names) == len(set(names))


def test_duplicate_names_rejected():
    doc = _doc()
    doc["models"].append(dict(doc["models"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        parse_models(doc)


def test_wrong_schema_rejected():
    doc = _doc()
    doc["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        parse_models(doc)


@pytest.mark.parametrize("name", OLD_NAMES)
def test_old_names_still_resolve(name, tmp_path):
    r = ModelRegistry(cache_dir=tmp_path)
    spec = r.resolve(name, _profile(256.0))
    assert spec.name == name


def test_noncommercial_flag_on_twil():
    twil = next(s for s in REGISTRY if s.name == "twil-8b")
    assert twil.noncommercial is True
    assert twil.repo_id == "webAI-Official/TwIL-8B"
    assert twil.license == "other" and "non-commercial" in (twil.license_name or "")
    granite = next(s for s in REGISTRY if s.name == "granite-4.2-8b")
    assert granite.noncommercial is False
    assert granite.license == "apache-2.0"


def test_recommend_skips_noncommercial(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "runtime_version", lambda: "0.3.37")
    r = ModelRegistry(cache_dir=tmp_path)
    pick = r.recommend(_profile(16.0), task="code")
    assert not pick.noncommercial
    assert pick.name == "granite-4.2-8b"


def test_recommend_skips_arch_the_runtime_cannot_load(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "runtime_version", lambda: "0.3.9")
    r = ModelRegistry(cache_dir=tmp_path)
    pick = r.recommend(_profile(16.0), task="code")
    assert pick.name != "granite-4.2-8b"  # needs 0.3.17
    assert runtime_supports(pick, "0.3.9")


def test_runtime_supports_versions():
    granite = next(s for s in REGISTRY if s.name == "granite-4.2-8b")
    assert runtime_supports(granite, "0.3.36")
    assert not runtime_supports(granite, "0.3.16")


def test_remote_is_opt_in_and_falls_back(monkeypatch):
    calls = []

    def boom(url, timeout):
        calls.append((url, timeout))
        raise OSError("offline")

    monkeypatch.setattr(reg, "_fetch_remote", boom)
    monkeypatch.delenv(reg.REMOTE_ENV, raising=False)
    assert load_registry() and not calls  # off by default
    specs = load_registry(remote=True, timeout=0.5)
    assert calls == [(reg.REMOTE_MODELS_URL, 0.5)]
    assert [s.name for s in specs] == [s.name for s in REGISTRY]


def test_model_row_dict_carries_licence():
    from eigencore.desktop.backend import ModelRow
    from eigencore.desktop.web import model_row_dict

    twil = next(s for s in REGISTRY if s.name == "twil-8b")
    row = ModelRow(
        name=twil.name,
        params_b=twil.params_b,
        quant=twil.quant,
        size_gb=twil.size_gb,
        task=twil.task,
        context_length=twil.context_length,
        description=twil.description,
        downloaded=False,
        fit="unknown",
    )
    d = model_row_dict(row, twil)
    assert d["noncommercial"] is True
    assert d["license"] == twil.license_name
