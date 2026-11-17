#!/usr/bin/env python3
"""Regenerate the facts in src/eigencore/models/models.json from Hugging Face.

Run locally (no CI, no token needed):

    python scripts/refresh_models.py            # show what changed, write nothing
    python scripts/refresh_models.py --write    # update models.json in place
    python scripts/refresh_models.py --check    # exit 1 when models.json is stale
    python scripts/refresh_models.py --only granite-4.2-8b
    python scripts/refresh_models.py --runtime-table   # also rebuild runtime_support.json

For each entry it reads

* ``https://huggingface.co/api/models/<repo_id>?blobs=true`` -> exact file
  size (sum of shards), licence id / name / link, non-commercial flag;
* the GGUF header of the file over HTTP Range (a few MB, never the whole
  model) -> architecture, context length, geometry (layers, attention
  layers, heads, K/V dims, experts, active bytes per token), tokenizer type;

and derives ``min_runtime`` (from runtime_support.json) and ``ram_class``.
Hand-written fields (name, task, tags, description, priority) are kept. A
licence flagged non-commercial by hand is never silently cleared.

To add a model, append an entry with name, repo_id, filename, task, tags,
description and priority, then run with --write.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eigencore.fit.gguf_remote import (  # noqa: E402
    geometry_dict,
    hf_file_url,
    model_info_from_header,
    read_header,
)
from eigencore.models.fit_check import (  # noqa: E402
    RUNTIME_SUPPORT_JSON,
    min_runtime_for,
    ram_class,
)
from eigencore.models.hf_catalog import (  # noqa: E402
    _http_json,
    is_noncommercial,
    licence_of,
)
from eigencore.models.registry import (  # noqa: E402
    BUNDLED_MODELS_JSON,
    _version_tuple,
    spec_from_dict,
)

# Field order in the written file (unknown keys keep their place after these).
ORDER = (
    "name",
    "repo_id",
    "filename",
    "params_b",
    "quant",
    "size_bytes",
    "size_gb",
    "task",
    "tags",
    "ram_class",
    "priority",
    "context_length",
    "arch",
    "license",
    "license_name",
    "license_url",
    "noncommercial",
    "min_runtime",
    "experts_total",
    "experts_active",
    "geometry",
    "verified_at",
    "description",
)
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def _sizes(info: dict, filename: str) -> int:
    files = {s["rfilename"]: int(s.get("size") or 0) for s in info.get("siblings", [])}
    if filename not in files:
        raise SystemExit(f"{info.get('id')}: file {filename} not found on Hugging Face")
    m = _SHARD.search(filename)
    if not m:
        return files[filename]
    prefix = filename[: m.start()]
    return sum(s for f, s in files.items() if f.startswith(prefix) and _SHARD.search(f))


def refresh_entry(entry: dict, today: str) -> tuple[dict, list[str]]:
    repo, fn = entry["repo_id"], entry["filename"]
    info = _http_json(f"https://huggingface.co/api/models/{repo}?blobs=true", 30)
    size = _sizes(info, fn)
    lic, lic_name, lic_url = licence_of(info)
    hdr = read_header(hf_file_url(repo, fn), timeout=60)
    ginfo = model_info_from_header(hdr, size)
    kv = hdr.kv
    arch = hdr.architecture
    ctx = int(kv.get(f"{arch}.context_length") or entry.get("context_length") or 0)
    pre = kv.get("tokenizer.ggml.pre")
    geo = geometry_dict(ginfo)

    new = dict(entry)
    new["size_bytes"] = size
    new["size_gb"] = round(size / 1e9, 2)
    new["context_length"] = ctx
    new["arch"] = arch
    # params_b keeps the hand-written (model card) figure; filled only when missing
    if not entry.get("params_b"):
        total = kv.get("general.parameter_count") or (info.get("gguf") or {}).get("total")
        if total:
            new["params_b"] = round(int(total) / 1e9, 1)
    new["license"] = lic or entry.get("license")
    for k, v in (("license_name", lic_name), ("license_url", lic_url)):
        if v:
            new[k] = v
    nc = is_noncommercial(new["license"], new.get("license_name"))
    new["noncommercial"] = bool(nc or entry.get("noncommercial"))
    need = min_runtime_for(arch, pre)
    if need is None:
        need = entry.get("min_runtime")
    elif entry.get("min_runtime") and _version_tuple(entry["min_runtime"]) > _version_tuple(need):
        need = entry["min_runtime"]  # a hand-set higher floor wins
    new["min_runtime"] = need
    if geo.get("expert_count"):
        new["experts_total"] = geo["expert_count"]
        new["experts_active"] = geo.get("expert_used_count")
    new["geometry"] = geo
    new["ram_class"] = ram_class(spec_from_dict(_complete(new)))
    new["verified_at"] = today

    notes = []
    if entry.get("noncommercial") and not nc:
        notes.append("kept hand-set noncommercial=true (licence text not detected as NC)")
    if arch and min_runtime_for(arch) is None:
        notes.append(f"arch {arch!r} not in runtime_support.json (newer than the last scan?)")
    if pre and min_runtime_for(None, pre) is None:
        notes.append(f"tokenizer pre {pre!r} not in runtime_support.json")
    return {k: new[k] for k in _ordered(new)}, notes


def _complete(e: dict) -> dict:
    d = {
        "tags": [],
        "description": "",
        "license": None,
        "min_runtime": None,
        "noncommercial": False,
    }
    d.update(e)
    d.setdefault("params_b", 0.0)
    d.setdefault("quant", "Q4_K_M")
    d.setdefault("task", "general")
    return d


def _ordered(d: dict) -> list[str]:
    return [k for k in ORDER if k in d] + [k for k in d if k not in ORDER]


def diff(old: dict, new: dict) -> list[str]:
    out = []
    for k in _ordered(new):
        if old.get(k) != new.get(k):
            out.append(f"  {k}: {old.get(k)!r} -> {new.get(k)!r}")
    return out


# ----- runtime table ----------------------------------------------------------------


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "eigencore-refresh"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def build_runtime_table() -> dict:
    """arch / tokenizer pre -> first llama-cpp-python release that knows it.

    Uses PyPI for the release list, the GitHub contents API for each release's
    vendor/llama.cpp commit (unauthenticated: ~40 calls, under the 60/h limit),
    and raw.githubusercontent.com for llama.cpp sources at that commit.
    """
    pypi = json.loads(_get("https://pypi.org/pypi/llama-cpp-python/json"))
    versions = sorted(
        (v for v in pypi["releases"] if re.fullmatch(r"0\.3\.\d+", v)), key=_version_tuple
    )
    seen: dict[str, tuple[set, set]] = {}
    per_version = []
    for v in versions:
        vendor = json.loads(
            _get(f"https://api.github.com/repos/abetlen/llama-cpp-python/contents/vendor?ref=v{v}")
        )
        sha = next(e["sha"] for e in vendor if e["name"] == "llama.cpp")
        if sha not in seen:
            txt = ""
            for f in ("src/llama-arch.cpp", "src/llama-vocab.cpp", "src/llama.cpp"):
                try:
                    txt += _get(
                        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{sha}/{f}"
                    ).decode("utf-8", "replace")
                except OSError:
                    pass
            seen[sha] = (
                set(re.findall(r'LLM_ARCH_\w+,\s*"([^"]+)"', txt)),
                set(re.findall(r'tokenizer_pre\s*==\s*"([^"]+)"', txt)),
            )
        per_version.append((v, *seen[sha]))
        print(f"  {v}: {len(seen[sha][0])} archs, {len(seen[sha][1])} tokenizers", file=sys.stderr)

    def first(idx: int) -> dict[str, str]:
        names = set().union(*(row[idx] for row in per_version))
        out = {}
        for n in names:
            for i, row in enumerate(per_version):
                later = [r for r in per_version[i:] if r[idx]]  # skip scans that found nothing
                if row[idx] and all(n in r[idx] for r in later):
                    out[n] = row[0]
                    break
        return dict(sorted(out.items()))

    return {
        "_comment": (
            "First llama-cpp-python release whose vendored llama.cpp knows each GGUF "
            "general.architecture (arch) and tokenizer.ggml.pre type (pre). Built by "
            "scripts/refresh_models.py --runtime-table from llama.cpp src/llama-arch.cpp "
            "and src/llama-vocab.cpp at each release's vendor/llama.cpp commit. A name "
            "missing here is newer than the last release scanned."
        ),
        "scanned_up_to": versions[-1],
        "updated": dt.date.today().isoformat(),
        "arch": first(1),
        "pre": first(2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="update models.json in place")
    ap.add_argument("--check", action="store_true", help="exit 1 when anything changed")
    ap.add_argument("--only", action="append", help="refresh just these model names")
    ap.add_argument("--path", type=Path, default=BUNDLED_MODELS_JSON)
    ap.add_argument("--runtime-table", action="store_true", help="rebuild runtime_support.json")
    args = ap.parse_args(argv)

    if args.runtime_table:
        table = build_runtime_table()
        if args.write:
            RUNTIME_SUPPORT_JSON.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {RUNTIME_SUPPORT_JSON}")
        else:
            print(json.dumps({k: len(v) for k, v in table.items() if isinstance(v, dict)}))
        from eigencore.models import fit_check

        fit_check.runtime_support.cache_clear()

    doc = json.loads(args.path.read_text(encoding="utf-8"))
    today = dt.date.today().isoformat()
    changed = 0
    for i, entry in enumerate(doc["models"]):
        if args.only and entry["name"] not in args.only:
            continue
        try:
            new, notes = refresh_entry(entry, today)
        except Exception as e:  # report and keep going
            print(f"! {entry['name']}: {type(e).__name__}: {e}", file=sys.stderr)
            changed += 1
            continue
        d = [line for line in diff(entry, new) if not line.startswith("  verified_at")]
        if d:
            changed += 1
            print(f"{entry['name']}:")
            print("\n".join(d))
        for n in notes:
            print(f"  note: {n}")
        doc["models"][i] = new
    if args.write:
        doc["updated"] = today
        args.path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.path} ({len(doc['models'])} models)")
    elif not changed:
        print("models.json is up to date")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
