"""Two-tier model catalog: remote GGUF header, HF search, fit verdicts, ranking.

No test touches the network: HTTP is replaced by in-memory fakes.
"""

from __future__ import annotations

import importlib.util
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from eigencore.fit.estimator import ModelGeometry, kv_bytes_per_token
from eigencore.fit.gguf_remote import (
    GGUFHeaderError,
    geometry_dict,
    geometry_from_dict,
    geometry_from_kv,
    model_info_from_header,
    read_header,
    read_remote_model_info,
    tensor_byte_totals,
)
from eigencore.models import fit_check
from eigencore.models import registry as reg
from eigencore.models.fit_check import (
    FITS,
    TIGHT,
    TOO_BIG,
    UNKNOWN,
    fit_report,
    min_runtime_for,
    ram_class,
)
from eigencore.models.hf_catalog import (
    HFCatalog,
    entry_from_info,
    is_noncommercial,
    licence_kind,
    parse_ref,
    pick_file,
    quant_of,
)
from eigencore.models.registry import REGISTRY, ModelRegistry, ModelSpec

ROOT = Path(__file__).resolve().parents[1]

# ----- a tiny GGUF writer ------------------------------------------------------------


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    out = _s(key)
    if isinstance(value, bool):
        return out + struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return out + struct.pack("<I", 4) + struct.pack("<I", value)
    if isinstance(value, float):
        return out + struct.pack("<I", 6) + struct.pack("<f", value)
    if isinstance(value, str):
        return out + struct.pack("<I", 8) + _s(value)
    if isinstance(value, list) and value and isinstance(value[0], str):
        body = b"".join(_s(v) for v in value)
        return out + struct.pack("<IIQ", 9, 8, len(value)) + body
    if isinstance(value, list) and value and isinstance(value[0], bool):
        body = b"".join(struct.pack("<?", v) for v in value)
        return out + struct.pack("<IIQ", 9, 7, len(value)) + body
    if isinstance(value, list):
        body = b"".join(struct.pack("<i", v) for v in value)
        return out + struct.pack("<IIQ", 9, 5, len(value)) + body
    raise TypeError(value)


def make_gguf(kv: dict, tensors: list[tuple[str, int]], align: int = 32) -> bytes:
    """GGUF v3 bytes: header + tensor infos (with the given data sizes) + zero data."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    head += b"".join(_kv(k, v) for k, v in kv.items())
    off = 0
    infos = b""
    for name, size in tensors:
        infos += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", size) + struct.pack("<I", 0)
        infos += struct.pack("<Q", off)
        off += size
    blob = head + infos
    pad = (-len(blob)) % align
    return blob + b"\0" * pad + b"\0" * off


def fake_fetch(blob: bytes, log: list | None = None):
    def fetch(url, start, end):
        if log is not None:
            log.append((url, start, end))
        return blob[start : end + 1]

    return fetch


LLAMA_KV = {
    "general.architecture": "llama",
    "general.license": "apache-2.0",
    "llama.block_count": 4,
    "llama.context_length": 4096,
    "llama.embedding_length": 256,
    "llama.attention.head_count": 8,
    "llama.attention.head_count_kv": 2,
    "tokenizer.ggml.pre": "llama-bpe",
    "tokenizer.ggml.tokens": ["<s>", "</s>", "hello"] * 50,
}


# ----- gguf_remote ---------------------------------------------------------------------


def test_read_header_kv_tensors_and_alignment():
    blob = make_gguf(LLAMA_KV, [("token_embd.weight", 64), ("blk.0.attn_q.weight", 32)])
    log: list = []
    hdr = read_header("u", fetch=fake_fetch(blob, log), chunk=4096)
    assert hdr.version == 3 and hdr.tensor_count == 2
    assert hdr.architecture == "llama"
    assert hdr.kv["llama.attention.head_count_kv"] == 2
    assert hdr.kv["tokenizer.ggml.tokens"] is None  # vocab walked, not kept
    assert [t[0] for t in hdr.tensors] == ["token_embd.weight", "blk.0.attn_q.weight"]
    assert hdr.data_start % 32 == 0
    assert len(blob) - hdr.data_start == 96
    assert log[0][1] == 0  # starts at byte 0


def test_read_header_spans_several_range_requests():
    blob = make_gguf(LLAMA_KV, [("a", 32)])
    log: list = []
    hdr = read_header("u", fetch=fake_fetch(blob, log), chunk=64)
    assert hdr.requests == len(log) > 3
    assert all(e - s == 63 for _, s, e in log)
    assert hdr.kv["llama.block_count"] == 4


def test_read_header_rejects_non_gguf_big_endian_and_huge_headers():
    with pytest.raises(GGUFHeaderError, match="not a GGUF"):
        read_header("u", fetch=fake_fetch(b"NOPE" + b"\0" * 64))
    be = b"GGUF" + struct.pack(">IQQ", 3, 0, 0)
    with pytest.raises(GGUFHeaderError, match="big-endian"):
        read_header("u", fetch=fake_fetch(be))
    blob = make_gguf(LLAMA_KV, [("a", 32)])
    with pytest.raises(GGUFHeaderError, match="larger than"):
        read_header("u", fetch=fake_fetch(blob), chunk=64, max_bytes=128)


def test_tensor_bytes_and_moe_active_weights_from_offsets():
    kv = dict(LLAMA_KV, **{"llama.expert_count": 8, "llama.expert_used_count": 2})
    tensors = [
        ("token_embd.weight", 1000),
        ("blk.0.ffn_up_exps.weight", 8000),
        ("blk.0.ffn_down_shexp.weight", 500),
        ("blk.0.attn_q.weight", 500),
    ]
    blob = make_gguf(kv, tensors)
    hdr = read_header("u", fetch=fake_fetch(blob))
    total, expert = tensor_byte_totals(hdr, len(blob))
    assert (total, expert) == (10000, 8000)
    info = model_info_from_header(hdr, len(blob))
    assert info.active_weight_bytes == 2000 + 8000 * 2 // 8
    assert geometry_dict(info)["active_bytes"] == 4000
    # without the file size (or for shards) no tensor totals are guessed
    assert tensor_byte_totals(hdr, None) == (None, None)
    hdr.kv["split.count"] = 2
    assert tensor_byte_totals(hdr, len(blob)) == (None, None)


def test_hybrid_per_layer_kv_heads_count_only_attention_layers():
    kv = {
        "general.architecture": "granitehybrid",
        "granitehybrid.block_count": 10,
        "granitehybrid.embedding_length": 1536,
        "granitehybrid.attention.head_count": 12,
        "granitehybrid.attention.head_count_kv": [0, 0, 4, 0, 0, 0, 0, 4, 0, 0],
    }
    g = geometry_from_kv(kv)
    assert g.attention_layers == 2 and g.num_kv_heads == 4 and g.head_dim == 128
    assert g.kv_elements_per_token == 2 * 4 * 256
    assert kv_bytes_per_token(g) == 2 * 4 * 256 * 2


def test_full_attention_interval_and_qwen3next_default():
    base = {
        "qwen35.block_count": 32,
        "qwen35.embedding_length": 2560,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
    }
    g = geometry_from_kv(
        dict(base, **{"general.architecture": "qwen35", "qwen35.full_attention_interval": 4})
    )
    assert g.attention_layers == 8
    nxt = {k.replace("qwen35", "qwen3next"): v for k, v in base.items()}
    g2 = geometry_from_kv(dict(nxt, **{"general.architecture": "qwen3next"}))
    assert g2.attention_layers == 8  # llama.cpp's default interval of 4
    dense = geometry_from_kv({**LLAMA_KV})
    assert dense.attention_layers is None and dense.kv_layers == 4
    assert kv_bytes_per_token(g) == 8 * 4 * 512 * 2


def test_per_layer_swa_dims_gemma_style():
    kv = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 6,
        "gemma4.embedding_length": 3840,
        "gemma4.attention.head_count": 16,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 1],
        "gemma4.attention.key_length": 512,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.key_length_swa": 256,
        "gemma4.attention.value_length_swa": 256,
        "gemma4.attention.sliding_window_pattern": [True] * 5 + [False],
    }
    g = geometry_from_kv(kv)
    assert g.kv_elements_per_token == 5 * 8 * 512 + 1 * 1024


def test_geometry_dict_round_trip():
    g = ModelGeometry(
        num_layers=40, hidden_size=1536, num_attention_heads=12, num_kv_heads=4, attention_layers=4
    )
    info = SimpleNamespace(geometry=g, tensor_bytes=None, expert_tensor_bytes=None)
    d = geometry_dict(info)
    assert d["attention_layers"] == 4 and "active_bytes" not in d
    assert geometry_from_dict(d).kv_layers == 4
    assert geometry_from_dict({}) is None
    assert geometry_from_dict({"block_count": 2}) is None  # incomplete


def test_read_remote_model_info_uses_hf_resolve_url():
    blob = make_gguf(LLAMA_KV, [("a", 32)])
    log: list = []
    info = read_remote_model_info(
        "org/repo", "m.gguf", size_bytes=len(blob), fetch=fake_fetch(blob, log)
    )
    assert log[0][0] == "https://huggingface.co/org/repo/resolve/main/m.gguf"
    assert info.architecture == "llama" and info.context_length == 4096
    assert info.tensor_bytes == 32


def test_local_reader_returns_per_layer_arrays(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "h.gguf"
    w = gguf.GGUFWriter(str(path), "granitehybrid")
    w.add_block_count(4)
    w.add_embedding_length(64)
    w.add_head_count(4)
    w.add_head_count_kv([0, 2, 0, 2])
    w.add_context_length(1024)
    import numpy as np

    w.add_tensor("token_embd.weight", np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    from eigencore.fit.gguf_meta import read_gguf_model_info

    info = read_gguf_model_info(path)
    assert info.geometry.attention_layers == 2
    assert info.geometry.num_kv_heads == 2


# ----- hf_catalog ----------------------------------------------------------------------


def _sib(*files):
    return [{"rfilename": f, "size": s} for f, s in files]


def test_pick_file_preference_and_skips():
    sib = _sib(
        ("mmproj-model-f16.gguf", 900),
        ("model-UD-Q4_K_M.gguf", 510),
        ("model-Q4_K_M.gguf", 500),
        ("model-Q8_0.gguf", 1000),
    )
    p = pick_file(sib)
    assert (p.filename, p.quant, p.size_bytes) == ("model-Q4_K_M.gguf", "Q4_K_M", 500)
    # only an unsloth dynamic quant -> that one
    p = pick_file(_sib(("m-UD-Q4_K_M.gguf", 7), ("m-Q8_0.gguf", 9)))
    assert p.quant == "UD-Q4_K_M"
    # single file beats shards; shards sum when only shards exist
    p = pick_file(
        _sib(
            ("Q4_K_M/m-Q4_K_M-00001-of-00002.gguf", 10),
            ("Q4_K_M/m-Q4_K_M-00002-of-00002.gguf", 5),
        )
    )
    assert p.filename == "Q4_K_M/m-Q4_K_M-00001-of-00002.gguf" and p.size_bytes == 15
    assert p.shards == 2
    # MXFP4 when that is all there is (gpt-oss); big-endian builds skipped
    assert pick_file(_sib(("gpt-oss-20b-MXFP4.gguf", 1))).quant == "MXFP4"
    assert pick_file(_sib(("m-BE-Q4_K_M.gguf", 1))) is None
    assert pick_file(_sib(("m-Q8_0.gguf", 1))) is None
    # an exact file asked for by name
    assert pick_file(sib, prefer="model-Q8_0.gguf").filename == "model-Q8_0.gguf"
    assert pick_file(sib, prefer="missing.gguf") is None


def test_quant_of():
    assert quant_of("x-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert quant_of("qwen2.5-7b-instruct-q4_k_m.gguf") == "Q4_K_M"
    assert quant_of("x-f16.gguf") is None


@pytest.mark.parametrize(
    "lic,name,nc,kind",
    [
        ("apache-2.0", None, False, "permissive"),
        ("mit", None, False, "permissive"),
        ("llama3.1", None, False, "restricted"),
        ("other", "deepseek-license", False, "restricted"),
        ("other", "mnpl", True, "noncommercial"),
        ("cc-by-nc-4.0", None, True, "noncommercial"),
        ("other", "qwen-research", True, "noncommercial"),
        ("other", "webai-non-commercial-license-ver.-1.0", True, "noncommercial"),
        (None, None, False, "unknown"),
    ],
)
def test_licence_detection(lic, name, nc, kind):
    assert is_noncommercial(lic, name) is nc
    assert licence_kind(lic, name) == kind


def _info(repo, files, lic="apache-2.0", arch="llama", ctx=4096, total=8e9, **card):
    return {
        "id": repo,
        "downloads": 10,
        "likes": 1,
        "tags": ["gguf", f"license:{lic}"],
        "cardData": {"license": lic, **card},
        "gguf": {
            "architecture": arch,
            "context_length": ctx,
            "total": total,
            "chat_template": "x" * 10,
        },
        "siblings": _sib(*files),
    }


def test_entry_from_info_and_spec():
    info = _info(
        "bartowski/Codestral-22B-v0.1-GGUF",
        [("Codestral-22B-v0.1-Q4_K_M.gguf", 13_341_237_600)],
        lic="other",
        license_name="mnpl",
        license_link="LICENSE.md",
    )
    e = entry_from_info(info)
    assert e.name == "bartowski/Codestral-22B-v0.1-GGUF:Codestral-22B-v0.1-Q4_K_M.gguf"
    assert e.noncommercial and e.licence_kind == "noncommercial"
    assert e.license_url.endswith("/bartowski/Codestral-22B-v0.1-GGUF/blob/main/LICENSE.md")
    assert e.trusted and "code" in e.tags
    spec = e.to_spec(min_runtime="0.3.0")
    assert spec.source == "hf-search" and spec.size_bytes == 13_341_237_600
    assert spec.task == "code" and spec.params_b == 8.0


def test_parse_ref():
    assert parse_ref("org/repo") == ("org/repo", None)
    assert parse_ref("org/repo:dir/f.gguf") == ("org/repo", "dir/f.gguf")


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes
        self.calls: list[str] = []
        self.offline = False

    def __call__(self, url, timeout):
        self.calls.append(url)
        if self.offline:
            raise OSError("offline")
        for key, value in self.routes.items():
            if key in url:
                return json.loads(json.dumps(value))
        raise OSError(f"404 {url}")


def _search_routes():
    rows = [
        {"id": "someone/Model-abliterated-GGUF"},
        {"id": "randomuser/Granite-GGUF"},
        {"id": "ibm-granite/granite-4.2-3b-GGUF"},
    ]
    return {
        "/api/models?": rows,
        "models/randomuser/Granite-GGUF": _info("randomuser/Granite-GGUF", [("g-Q4_K_M.gguf", 2)]),
        "models/ibm-granite/granite-4.2-3b-GGUF": _info(
            "ibm-granite/granite-4.2-3b-GGUF", [("granite-4.2-3b-Q4_K_M.gguf", 2_244_011_552)]
        ),
    }


def test_search_filters_hidden_ranks_trusted_and_caches(tmp_path):
    http = FakeHttp(_search_routes())
    clock = [1000.0]
    cat = HFCatalog(tmp_path / "c.json", http=http, clock=lambda: clock[0])
    res = cat.search("granite", trusted_only=False)
    assert [e.repo_id for e in res] == [
        "ibm-granite/granite-4.2-3b-GGUF",
        "randomuser/Granite-GGUF",
    ]
    assert [e.trusted for e in res] == [True, False]
    assert "filter=gguf" in http.calls[0] and "search=granite" in http.calls[0]
    assert not any("abliterated" in c for c in http.calls)
    n = len(http.calls)
    cat2 = HFCatalog(tmp_path / "c.json", http=http, clock=lambda: clock[0])
    assert [e.repo_id for e in cat2.search("granite", trusted_only=False)] == [
        e.repo_id for e in res
    ]
    assert len(http.calls) == n  # served from the disk cache
    saved = json.loads((tmp_path / "c.json").read_text())
    assert all("chat_template" not in json.dumps(v) for v in saved.values())


def test_search_trusted_uploaders_only_by_default(tmp_path):
    http = FakeHttp(_search_routes())
    cat = HFCatalog(tmp_path / "c.json", http=http)
    res = cat.search("granite")
    assert [e.repo_id for e in res] == ["ibm-granite/granite-4.2-3b-GGUF"]
    assert all(e.trusted for e in res)
    # the community repo is not even fetched
    assert not any("randomuser" in c for c in http.calls)
    # same query with the toggle on: served from the cached search, community added after
    more = cat.search("granite", trusted_only=False)
    assert [e.repo_id for e in more] == [
        "ibm-granite/granite-4.2-3b-GGUF",
        "randomuser/Granite-GGUF",
    ]


def test_search_offline_uses_stale_cache_then_reports_error(tmp_path):
    http = FakeHttp(_search_routes())
    clock = [0.0]
    cat = HFCatalog(tmp_path / "c.json", http=http, clock=lambda: clock[0])
    assert cat.search("granite", trusted_only=False)
    clock[0] += 10 * 24 * 3600  # cache is stale
    http.offline = True
    assert len(cat.search("granite", trusted_only=False)) == 2  # stale copy beats nothing
    assert len(cat.search("granite")) == 1
    assert cat.search("something else") == []
    assert "offline" in cat.last_error
    assert cat.search("  ") == []


def test_entry_offline_returns_none_with_error(tmp_path):
    http = FakeHttp({})
    http.offline = True
    cat = HFCatalog(tmp_path / "c.json", http=http)
    assert cat.entry("org/repo") is None
    assert "offline" in cat.last_error


# ----- fit verdicts ----------------------------------------------------------------------


def _prof(ram, bw=50.0, free=None):
    """``ram`` is the machine's TOTAL RAM; ``free`` (default: half of it) is only a hint."""
    return SimpleNamespace(
        total_ram_gb=ram,
        available_ram_gb=ram / 2 if free is None else free,
        estimated_bandwidth_gbps=bw,
    )


def _granite8b():
    return next(s for s in REGISTRY if s.name == "granite-4.2-8b")


def test_fit_report_verdicts_scale_with_ram():
    s = _granite8b()
    assert fit_report(s, _prof(32), installed_runtime="0.3.36").verdict == FITS
    assert fit_report(s, _prof(11), installed_runtime="0.3.36").verdict == TIGHT
    big = fit_report(s, _prof(8), installed_runtime="0.3.36")
    assert big.verdict == TOO_BIG and big.max_context == 0


def test_system_reserve_is_20_percent_with_a_floor():
    assert fit_check.system_reserve_gb(4) == 1.5  # floor
    assert fit_check.system_reserve_gb(8) == pytest.approx(1.6)
    assert fit_check.system_reserve_gb(16) == pytest.approx(3.2)
    assert fit_check.ram_budget_gb(32) == pytest.approx(25.6)
    assert fit_check.ram_budget_gb(1) == 0.0


def test_verdict_uses_total_ram_not_free_ram():
    s = _granite8b()
    busy = fit_report(s, _prof(16, free=2.0), installed_runtime="0.3.36")
    idle = fit_report(s, _prof(16, free=14.0), installed_runtime="0.3.36")
    assert busy.verdict == idle.verdict == FITS
    assert busy.max_context == idle.max_context
    assert busy.total_ram_gb == 16 and busy.budget_gb == pytest.approx(12.8)
    assert busy.system_reserve_gb == pytest.approx(3.2)
    # free RAM stays as a hint only
    assert busy.available_ram_gb == 2.0 and any("free right now" in n for n in busy.notes)
    assert not any("free right now" in n for n in idle.notes)
    d = busy.to_dict()
    assert d["budget_gb"] == busy.budget_gb and d["available_ram_gb"] == 2.0


def test_profile_without_total_falls_back_to_free():
    s = _granite8b()
    rep = fit_report(s, SimpleNamespace(available_ram_gb=16.0), installed_runtime="0.3.36")
    assert rep.total_ram_gb == 16.0 and rep.budget_gb == pytest.approx(12.8)


def test_every_curated_model_fits_or_is_tight_on_its_ram_class():
    for s in REGISTRY:
        rep = fit_report(s, _prof(s.ram_class), installed_runtime="0.3.36")
        assert rep.verdict in (FITS, TIGHT), (s.name, rep.verdict)


def test_fit_report_details_are_estimates():
    s = _granite8b()
    rep = fit_report(s, _prof(32), installed_runtime="0.3.36")
    assert rep.basis == "curated" and rep.context_checked == 8192
    assert rep.max_context >= 8192 and rep.max_context <= s.context_length
    assert rep.decode_ceiling_tok_s and rep.kv_cache_gb > 0
    d = rep.to_dict()
    assert d["estimate"] is True and d["runtime_note"] is None
    # no bandwidth figure -> no ceiling (never invented)
    assert (
        fit_report(s, _prof(32, bw=None), installed_runtime="0.3.36").decode_ceiling_tok_s is None
    )


def test_fit_report_moe_ceiling_uses_active_bytes():
    moe = next(s for s in REGISTRY if s.name == "qwen3-coder-30b-a3b")
    rep = fit_report(moe, _prof(64), installed_runtime="0.3.36")
    assert rep.active_weights_gb < rep.weights_gb / 4
    assert rep.decode_ceiling_tok_s == pytest.approx(50.0 / rep.active_weights_gb, rel=0.01)


def test_fit_report_hybrid_kv_is_small():
    tiny = next(s for s in REGISTRY if s.name == "granite-4.0-h-tiny")
    rep = fit_report(tiny, _prof(16), installed_runtime="0.3.36")
    assert rep.kv_cache_gb < 0.1  # 4 attention layers out of 40
    assert rep.max_context >= 262144


def test_fit_report_size_only_without_geometry():
    s = replace(_granite8b(), geometry=None)
    rep = fit_report(s, _prof(32), installed_runtime=None)
    assert rep.verdict == UNKNOWN and rep.basis == "size-only" and rep.max_context is None
    assert fit_report(s, _prof(4), installed_runtime=None).verdict == TOO_BIG


def test_runtime_gate():
    s = _granite8b()
    old = fit_report(s, _prof(32), installed_runtime="0.3.16")
    assert not old.runtime_ok and "0.3.17" in old.runtime_note
    assert fit_report(s, _prof(32), installed_runtime=None).runtime_ok  # not installed: no gate


def test_min_runtime_table():
    assert min_runtime_for("llama") == "0.3.0"
    assert min_runtime_for("qwen3moe") == "0.3.9"
    assert min_runtime_for("granitehybrid") == "0.3.13"
    assert min_runtime_for("qwen35") == "0.3.17"
    assert min_runtime_for("granite", "granite-docling") == "0.3.17"  # tokenizer raises the floor
    assert min_runtime_for("not-an-arch") is None
    assert fit_check.arch_is_known("gpt-oss")


def test_ram_class_rule():
    by = {s.name: s for s in REGISTRY}
    assert ram_class(by["tinyllama-1.1b"]) == 8
    assert ram_class(by["granite-4.2-8b"]) == 16
    assert ram_class(by["qwen3-coder-30b-a3b"]) == 32
    assert ram_class(by["qwen3-next-80b-a3b"]) == 64


# ----- curated core ----------------------------------------------------------------------


def test_curated_core_shape():
    assert 15 <= len(REGISTRY) <= 20
    for s in REGISTRY:
        assert s.geometry and geometry_from_dict(s.geometry) is not None, s.name
        assert s.ram_class in (8, 16, 32, 64), s.name
        assert s.verified_at, s.name
        assert s.arch and s.min_runtime, s.name
        need = min_runtime_for(s.arch)
        assert need is None or reg._version_tuple(s.min_runtime) >= reg._version_tuple(need), s.name
        assert s.ram_class == ram_class(s), s.name
        if s.is_moe:
            assert s.geometry.get("expert_count") == s.experts_total


def test_curated_core_covers_each_task_and_ram_class():
    allowed = [s for s in REGISTRY if not s.noncommercial]
    for task in ("chat", "code", "reasoning"):
        classes = {s.ram_class for s in allowed if task in s.tags}
        assert {8, 16, 32} <= classes, (task, classes)
    assert {s.ram_class for s in allowed if "code" in s.tags} >= {8, 16, 32, 64}
    assert any(s.is_moe for s in allowed if s.ram_class == 8)


# ----- registry: ranking, resolve, custom models ----------------------------------------


@pytest.fixture
def runtime_36(monkeypatch):
    monkeypatch.setattr(reg, "runtime_version", lambda: "0.3.36")


@pytest.mark.parametrize(
    "ram,task,want",
    [
        # ram = total installed RAM (GB); the budget is 80 % of it
        (8, "code", "granite-4.2-3b"),
        (16, "code", "granite-4.2-8b"),
        (32, "code", "qwen3-coder-30b-a3b"),
        (64, "code", "qwen3-coder-next-80b-a3b"),
        (8, "general", "qwen3.5-4b"),
        (16, "general", "qwen3.5-9b"),
        (24, "general", "qwen3.5-9b"),  # 19.2 GB budget: a 30B-A3B does not fit
        (32, "general", "qwen3.6-35b-a3b"),
        (64, "general", "qwen3-next-80b-a3b"),
    ],
)
def test_recommend_by_fit_then_priority(tmp_path, runtime_36, ram, task, want):
    r = ModelRegistry(cache_dir=tmp_path)
    assert r.recommend(_prof(ram), task).name == want


def test_recommend_prefers_fits_over_tight(tmp_path, runtime_36):
    r = ModelRegistry(cache_dir=tmp_path)
    pick = r.recommend(_prof(11), "code")  # granite-4.2-8b is only tight here
    assert pick.name == "granite-4.2-3b"


def test_recommend_reasoning_uses_tags(tmp_path, runtime_36):
    r = ModelRegistry(cache_dir=tmp_path)
    assert "reasoning" in r.recommend(_prof(16), "reasoning").tags


def test_recommend_nothing_fits_returns_smallest(tmp_path, runtime_36):
    r = ModelRegistry(cache_dir=tmp_path)
    assert r.recommend(_prof(1.0), "general").name == "tinyllama-1.1b"


class FakeCatalog:
    def __init__(self, entry=None, err=None):
        self._entry, self.last_error = entry, err
        self.asked: list[str] = []

    def entry(self, ref):
        self.asked.append(ref)
        return self._entry

    def search(self, q, **kw):
        self.search_kw = kw
        return [self._entry] if self._entry else []


def _codestral_entry():
    return entry_from_info(
        _info(
            "bartowski/Codestral-22B-v0.1-GGUF",
            [("Codestral-22B-v0.1-Q4_K_M.gguf", 13_341_237_600)],
            lic="other",
            license_name="mnpl",
            total=22.2e9,
        )
    )


def test_resolve_accepts_repo_and_file(tmp_path):
    cat = FakeCatalog(_codestral_entry())
    r = ModelRegistry(cache_dir=tmp_path, catalog=cat)
    spec = r.resolve("bartowski/Codestral-22B-v0.1-GGUF:Codestral-22B-v0.1-Q4_K_M.gguf", _prof(64))
    assert spec.source == "hf-search" and spec.noncommercial
    assert spec.min_runtime == "0.3.0"  # llama arch
    assert cat.asked == ["bartowski/Codestral-22B-v0.1-GGUF:Codestral-22B-v0.1-Q4_K_M.gguf"]
    with pytest.raises(ValueError, match="needs"):
        r.resolve("bartowski/Codestral-22B-v0.1-GGUF", _prof(8))


def test_resolve_curated_by_repo_id_without_network(tmp_path):
    cat = FakeCatalog()
    r = ModelRegistry(cache_dir=tmp_path, catalog=cat)
    spec = r.resolve("ibm-granite/granite-4.2-8b-GGUF", _prof(64))
    assert spec.name == "granite-4.2-8b" and cat.asked == []


def test_resolve_ref_offline_is_a_clear_error(tmp_path):
    r = ModelRegistry(cache_dir=tmp_path, catalog=FakeCatalog(err="URLError: offline"))
    with pytest.raises(ValueError, match="offline"):
        r.resolve("org/repo", _prof(64))
    with pytest.raises(ValueError, match="Unknown model"):
        r.resolve("no-such-model", _prof(64))


def test_custom_models_persist_and_leave(tmp_path):
    r = ModelRegistry(cache_dir=tmp_path)
    spec = _codestral_entry().to_spec()
    r.add_custom(spec)
    again = ModelRegistry(cache_dir=tmp_path)
    names = [m.name for m in again.list_models()]
    assert spec.name in names and len(names) == len(REGISTRY) + 1
    assert again.find(spec.name).source == "custom"
    again.remove_custom(spec.name)
    assert spec.name not in [m.name for m in ModelRegistry(cache_dir=tmp_path).list_models()]


def test_sharded_files_in_a_subfolder(tmp_path):
    spec = ModelSpec(
        name="org/r:Q4_K_M/m-Q4_K_M-00001-of-00002.gguf",
        repo_id="org/r",
        filename="Q4_K_M/m-Q4_K_M-00001-of-00002.gguf",
        params_b=1,
        quant="Q4_K_M",
        size_gb=1,
        task="general",
        context_length=4096,
        description="",
    )
    r = ModelRegistry(cache_dir=tmp_path)
    root = tmp_path / "org--r" / "Q4_K_M"
    root.mkdir(parents=True)
    (root / "m-Q4_K_M-00001-of-00002.gguf").write_bytes(b"x")
    assert not r.is_downloaded(spec)
    (root / "m-Q4_K_M-00002-of-00002.gguf").write_bytes(b"x")
    assert r.is_downloaded(spec)
    assert r.delete(spec)
    assert not (tmp_path / "org--r").exists()


def test_download_keeps_repo_folders(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(reg, "hf_hub_download", lambda **kw: calls.append(kw))
    r = ModelRegistry(cache_dir=tmp_path)
    spec = replace(REGISTRY[0], filename="sub/x.gguf")
    r.download(spec)
    assert calls[0]["local_dir"] == tmp_path / spec.repo_dir_name


# ----- desktop bridge -----------------------------------------------------------------


def test_bridge_search_and_check_fit(tmp_path, monkeypatch):
    from eigencore.desktop import web
    from eigencore.desktop.backend import DesktopBackend

    entry = _codestral_entry()
    registry = ModelRegistry(cache_dir=tmp_path / "models", catalog=FakeCatalog(entry))
    be = DesktopBackend(registry=registry, profile=_prof(64), engine_factory=lambda s: None)
    br = web.WebBridge(backend=be, emit=lambda p: None, state_path=tmp_path / "ui.json")
    out = br.search_models("codestral")
    assert out["error"] is None and len(out["results"]) == 1
    row = out["results"][0]
    assert row["name"] == entry.name and row["source"] == "hf-search"
    assert row["licence_kind"] == "noncommercial" and row["noncommercial"] is True
    assert row["fit"] == "unknown"  # size only, before the header is read
    assert registry.catalog.search_kw == {"trusted_only": True}  # trusted uploaders by default
    assert out["all_uploaders"] is False
    out = br.search_models("codestral", True)  # the "Community uploads" toggle
    assert registry.catalog.search_kw == {"trusted_only": False} and out["all_uploaders"] is True
    assert br.search_models("  ")["results"] == []

    kv = {
        "general.architecture": "llama",
        "llama.block_count": 56,
        "llama.context_length": 32768,
        "llama.embedding_length": 6144,
        "llama.attention.head_count": 48,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
    }
    blob = make_gguf(kv, [("a", 32)])
    import eigencore.fit.gguf_remote as gr

    real = gr.read_header
    monkeypatch.setattr(gr, "read_header", lambda url, **k: real(url, fetch=fake_fetch(blob)))
    res = br.check_fit(entry.name)
    assert res["error"] is None and res["fit"] == "fits_with_margin"
    assert res["fit_detail"]["basis"] == "header" and res["fit_detail"]["max_context"] >= 8192
    # curated rows answer from models.json, no header read
    assert br.check_fit("granite-4.2-8b")["fit_detail"]["basis"] == "curated"


def test_backend_download_of_a_search_result_adds_it(tmp_path, monkeypatch):
    import queue

    from eigencore.desktop.backend import DesktopBackend

    entry = _codestral_entry()
    registry = ModelRegistry(cache_dir=tmp_path / "models", catalog=FakeCatalog(entry))
    events: queue.Queue = queue.Queue()
    be = DesktopBackend(
        registry=registry, profile=_prof(64), engine_factory=lambda s: None, events=events
    )
    monkeypatch.setattr(be, "_download_with_progress", lambda spec: True)
    be.download(entry.name)
    ev = events.get(timeout=5)
    assert ev[0] == "dl_done" and ev[2] is True
    assert any(m.name == entry.name for m in registry.list_models())


def test_model_rows_have_fit_before_download(tmp_path):
    from eigencore.desktop.backend import DesktopBackend

    be = DesktopBackend(
        registry=ModelRegistry(cache_dir=tmp_path), profile=_prof(16), engine_factory=lambda s: None
    )
    rows = {r.name: r for r in be.model_rows()}
    assert rows["granite-4.2-8b"].fit == "fits_with_margin"
    assert rows["qwen3-next-80b-a3b"].fit == "does_not_fit"
    assert rows["granite-4.2-8b"].fit_detail["max_context"] >= 8192


# ----- refresh script -------------------------------------------------------------------


def _load_refresh():
    path = ROOT / "scripts" / "refresh_models.py"
    spec = importlib.util.spec_from_file_location("refresh_models", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_refresh_entry_fills_facts_without_network(monkeypatch):
    mod = _load_refresh()
    kv = dict(LLAMA_KV, **{"llama.context_length": 8192})
    blob = make_gguf(kv, [("a", 32)])
    info = _info("org/Tiny-GGUF", [("tiny-Q4_K_M.gguf", len(blob))], lic="cc-by-nc-4.0")
    monkeypatch.setattr(mod, "_http_json", lambda url, timeout: info)
    real = mod.read_header
    monkeypatch.setattr(mod, "read_header", lambda url, **k: real(url, fetch=fake_fetch(blob)))
    entry = {
        "name": "tiny",
        "repo_id": "org/Tiny-GGUF",
        "filename": "tiny-Q4_K_M.gguf",
        "params_b": 0.1,
        "quant": "Q4_K_M",
        "task": "general",
        "tags": ["chat"],
        "description": "d",
        "priority": 1,
    }
    new, notes = mod.refresh_entry(entry, "2026-11-17")
    assert new["size_bytes"] == len(blob) and new["context_length"] == 8192
    assert new["arch"] == "llama" and new["min_runtime"] == "0.3.0"
    assert new["license"] == "cc-by-nc-4.0" and new["noncommercial"] is True
    assert new["geometry"]["block_count"] == 4 and new["ram_class"] == 8
    assert new["params_b"] == 0.1 and new["priority"] == 1  # hand-written fields kept
    assert list(new)[:3] == ["name", "repo_id", "filename"]
    # a hand-set NC flag is never cleared
    info["cardData"]["license"] = "apache-2.0"
    new, notes = mod.refresh_entry(dict(entry, noncommercial=True), "2026-11-17")
    assert new["noncommercial"] is True and notes


# ----- CLI ------------------------------------------------------------------------------


def test_cli_search_trusted_by_default_all_uploaders_flag(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from eigencore.cli import main as cli_main

    cat = FakeCatalog(_codestral_entry())
    monkeypatch.setattr(
        cli_main, "ModelRegistry", lambda: ModelRegistry(cache_dir=tmp_path, catalog=cat)
    )
    monkeypatch.setattr(cli_main, "profile_hardware", lambda: _prof(64))
    res = CliRunner().invoke(cli_main.cli, ["search", "codestral"])
    assert res.exit_code == 0, res.output
    assert cat.search_kw["trusted_only"] is True
    assert "--all-uploaders" in res.output
    res = CliRunner().invoke(cli_main.cli, ["search", "codestral", "--all-uploaders"])
    assert res.exit_code == 0, res.output
    assert cat.search_kw["trusted_only"] is False
    assert "Uploader" in res.output and "not reviewed" in res.output
