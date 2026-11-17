"""
Live Hugging Face catalog: search GGUF repos, pick the 4-bit file, read the licence.

Second tier of the model list (the first is the curated ``models.json``).
No account or token is needed: it uses the public Hub API

* ``GET /api/models?filter=gguf&search=<q>&sort=downloads&expand[]=...``
  for candidates (architecture, context, parameter count and licence come
  back in the same call), then
* ``GET /api/models/<repo>?blobs=true`` for the file list with exact sizes.

Results are cached on disk (``~/.eigencore/catalog-cache.json``, 24 h) and
every network call has a short timeout. Offline, ``search`` returns the
cached results for the same query, or an empty list with ``last_error`` set;
it never raises.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from eigencore.models.registry import ModelSpec

HF_API = "https://huggingface.co/api"
HF_WEB = "https://huggingface.co"
USER_AGENT = "eigencore (model catalog)"
DEFAULT_TIMEOUT = 8.0
CACHE_TTL_S = 24 * 3600
MAX_CACHE_ENTRIES = 200

#: Quant preference, best first: the plain Q4_K_M, then the closest 4-bit files.
QUANT_PREFERENCE = (
    "Q4_K_M",
    "UD-Q4_K_M",
    "Q4_K_XL",
    "UD-Q4_K_XL",
    "MXFP4",
    "Q4_K_S",
    "IQ4_XS",
    "Q4_0",
)
#: Helper files that are not the language model itself.
_SKIP_FILE = re.compile(r"(^|/)(mmproj|mtp-|eagle3|dflash|imatrix)", re.IGNORECASE)
#: Big-endian builds (for s390x) do not load on x86 or ARM.
_BIG_ENDIAN = re.compile(r"(^|[-_.])BE([-_.]|$)")
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
#: Repos hidden from search by default (they crowd the trending lists).
_HIDDEN_REPO = re.compile(r"abliterat|uncensor|heretic|nsfw|erotic|lewd", re.IGNORECASE)
#: Quantisers whose GGUF uploads we trust by default (plus each model's own org).
TRUSTED_AUTHORS = frozenset(
    {
        "ggml-org",
        "unsloth",
        "bartowski",
        "lmstudio-community",
        "ibm-granite",
        "Qwen",
        "mistralai",
        "microsoft",
        "google",
        "meta-llama",
        "LiquidAI",
        "allenai",
        "deepseek-ai",
        "HuggingFaceTB",
        "nvidia",
        "openai",
        "CohereLabs",
        "TheBloke",
    }
)

#: Licences that allow commercial use without extra conditions.
PERMISSIVE_LICENSES = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc0-1.0",
        "unlicense",
        "openrail",
        "bigscience-openrail-m",
        "creativeml-openrail-m",
    }
)
_NC_LICENSE = re.compile(
    r"(^|[-_ ])nc([-_ ]|$)|non[-_ ]?commercial|noncommercial|research|mnpl|non[-_ ]?production",
    re.IGNORECASE,
)

Http = Callable[[str, float], Any]  # (url, timeout) -> parsed JSON


def _http_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ----- licence --------------------------------------------------------------------


def licence_of(info: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(license id, license_name, license_url) from an HF model-info dict."""
    card = info.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if not lic:
        lic = next(
            (t.split(":", 1)[1] for t in info.get("tags") or [] if t.startswith("license:")),
            None,
        )
    name = card.get("license_name")
    link = card.get("license_link")
    if link and not str(link).startswith("http"):
        link = f"{HF_WEB}/{info.get('id', '')}/blob/main/{str(link).lstrip('./')}"
    return (str(lic).lower() if lic else None), name, link


def is_noncommercial(license_id: Optional[str], license_name: Optional[str] = None) -> bool:
    """True when the licence id or name says non-commercial / research-only."""
    for s in (license_id, license_name):
        if s and _NC_LICENSE.search(str(s)):
            return True
    return False


def licence_kind(license_id: Optional[str], license_name: Optional[str] = None) -> str:
    """'permissive' | 'noncommercial' | 'restricted' (custom terms) | 'unknown'."""
    if is_noncommercial(license_id, license_name):
        return "noncommercial"
    if not license_id:
        return "unknown"
    if license_id in PERMISSIVE_LICENSES:
        return "permissive"
    return "restricted"


# ----- file pick ------------------------------------------------------------------


@dataclass(frozen=True)
class PickedFile:
    filename: str  # first shard when sharded
    size_bytes: int  # sum of all shards
    quant: str
    shards: int = 1


def quant_of(filename: str) -> Optional[str]:
    base = filename.rsplit("/", 1)[-1]
    for q in sorted(QUANT_PREFERENCE, key=len, reverse=True):
        # longest first, so UD-Q4_K_M is not read as Q4_K_M
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(q)}(?![A-Za-z0-9])", base, re.IGNORECASE):
            return q
    return None


def pick_file(siblings: list[dict[str, Any]], prefer: Optional[str] = None) -> Optional[PickedFile]:
    """Choose the GGUF to download from a repo's file list.

    Preference: ``QUANT_PREFERENCE`` order; a single file over shards of the
    same quant; helper files (mmproj, MTP / EAGLE drafts, imatrix) skipped.
    ``prefer`` picks an exact filename when the caller named one.
    """
    files = [
        (s["rfilename"], int(s.get("size") or 0))
        for s in siblings
        if str(s.get("rfilename", "")).lower().endswith(".gguf")
        and not _SKIP_FILE.search(s["rfilename"])
        and not _BIG_ENDIAN.search(s["rfilename"].rsplit("/", 1)[-1])
    ]
    if prefer:
        hit = [f for f in files if f[0] == prefer]
        if not hit:
            return None
        return _with_shards(hit[0][0], files)
    for q in QUANT_PREFERENCE:
        cands = [f for f in files if quant_of(f[0]) == q]
        if not cands:
            continue
        singles = [f for f in cands if not _SHARD.search(f[0])]
        if singles:
            name, size = min(singles, key=lambda f: len(f[0]))
            return PickedFile(name, size, q)
        firsts = [f for f in cands if (_SHARD.search(f[0]) or [None, None])[1] == "00001"]
        if firsts:
            return _with_shards(sorted(firsts)[0][0], files, q)
    return None


def _with_shards(
    name: str, files: list[tuple[str, int]], quant: Optional[str] = None
) -> PickedFile:
    q = quant or quant_of(name) or "unknown"
    m = _SHARD.search(name)
    if not m:
        size = next(s for f, s in files if f == name)
        return PickedFile(name, size, q)
    prefix = name[: m.start()]
    parts = [(f, s) for f, s in files if f.startswith(prefix) and _SHARD.search(f)]
    return PickedFile(name, sum(s for _, s in parts), q, shards=int(m.group(2)))


# ----- tags -----------------------------------------------------------------------

_TAG_RULES = (
    ("code", re.compile(r"coder|code|devstral|codestral|starcoder", re.IGNORECASE)),
    ("reasoning", re.compile(r"think|reason|-r1|qwq|\bo1\b|gpt-oss", re.IGNORECASE)),
    ("vision", re.compile(r"-vl-|vision|llava|gemma-4|gemma-3-\d+b", re.IGNORECASE)),
)


_MOE_NAME = re.compile(r"-A\d+(\.\d+)?B\b|moe|mixtral|gpt-oss|-h-(tiny|small)", re.IGNORECASE)
_MOE_ARCHS = frozenset({"qwen3next", "gpt-oss", "mixtral", "dbrx", "kimi-linear"})
_HYBRID_ARCHS = frozenset(
    {
        "granitehybrid",
        "qwen3next",
        "qwen35",
        "qwen35moe",
        "nemotron_h",
        "nemotron_h_moe",
        "lfm2",
        "lfm2moe",
        "kimi-linear",
        "jamba",
        "falcon-h1",
    }
)


def tags_for(repo_id: str, info: dict[str, Any], arch: Optional[str], moe: bool) -> tuple[str, ...]:
    tags: list[str] = []
    hay = repo_id
    for tag, rx in _TAG_RULES:
        if rx.search(hay):
            tags.append(tag)
    hf_tags = {str(t).lower() for t in info.get("tags") or []}
    if "image-text-to-text" in hf_tags and "vision" not in tags:
        tags.append("vision")
    if moe or (arch or "").endswith("moe") or _MOE_NAME.search(repo_id) or arch in _MOE_ARCHS:
        tags.append("moe")
    if arch in _HYBRID_ARCHS:
        tags.append("hybrid")
    if not any(t in tags for t in ("code", "reasoning")):
        tags.insert(0, "chat")
    return tuple(dict.fromkeys(tags))


# ----- catalog entries ------------------------------------------------------------


@dataclass
class CatalogEntry:
    """One downloadable GGUF from a Hugging Face search."""

    repo_id: str
    filename: str
    size_bytes: int
    quant: str
    shards: int = 1
    arch: Optional[str] = None
    context_length: Optional[int] = None
    params_b: Optional[float] = None
    license: Optional[str] = None
    license_name: Optional[str] = None
    license_url: Optional[str] = None
    noncommercial: bool = False
    licence_kind: str = "unknown"
    tags: tuple[str, ...] = ()
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    trusted: bool = False
    source: str = "hf-search"

    @property
    def name(self) -> str:
        """The id ``resolve()`` accepts: ``repo_id:filename``."""
        return f"{self.repo_id}:{self.filename}"

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / 1e9, 2)

    def to_spec(self, min_runtime: Optional[str] = None) -> ModelSpec:
        task = "code" if "code" in self.tags else "general"
        desc = f"Hugging Face search result: {self.filename.rsplit('/', 1)[-1]}"
        if self.shards > 1:
            desc += f" (+{self.shards - 1} more parts)"
        return ModelSpec(
            name=self.name,
            repo_id=self.repo_id,
            filename=self.filename,
            params_b=self.params_b or 0.0,
            quant=self.quant,
            size_gb=self.size_gb,
            task=task,
            context_length=self.context_length or 4096,
            description=desc,
            license=self.license,
            license_name=self.license_name,
            license_url=self.license_url,
            noncommercial=self.noncommercial,
            tags=self.tags,
            arch=self.arch,
            min_runtime=min_runtime,
            size_bytes=self.size_bytes,
            source="hf-search",
        )

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["tags"] = list(self.tags)
        d["name"] = self.name
        d["size_gb"] = self.size_gb
        return d


def entry_from_info(
    info: dict[str, Any], prefer_file: Optional[str] = None
) -> Optional[CatalogEntry]:
    """CatalogEntry from ``/api/models/<repo>?blobs=true``; None when no usable GGUF."""
    repo_id = info.get("id") or info.get("modelId") or ""
    picked = pick_file(info.get("siblings") or [], prefer=prefer_file)
    if picked is None:
        return None
    gguf = info.get("gguf") or {}
    lic, lic_name, lic_url = licence_of(info)
    total = gguf.get("total")
    arch = gguf.get("architecture")
    author = repo_id.split("/", 1)[0]
    return CatalogEntry(
        repo_id=repo_id,
        filename=picked.filename,
        size_bytes=picked.size_bytes,
        quant=picked.quant,
        shards=picked.shards,
        arch=arch,
        context_length=gguf.get("context_length"),
        params_b=round(total / 1e9, 1) if total else None,
        license=lic,
        license_name=lic_name,
        license_url=lic_url or f"{HF_WEB}/{repo_id}",
        noncommercial=is_noncommercial(lic, lic_name),
        licence_kind=licence_kind(lic, lic_name),
        tags=tags_for(repo_id, info, arch, moe=False),
        downloads=int(info.get("downloads") or 0),
        likes=int(info.get("likes") or 0),
        gated=bool(info.get("gated")),
        trusted=author in TRUSTED_AUTHORS,
    )


def parse_ref(ref: str) -> tuple[str, Optional[str]]:
    """``org/repo`` or ``org/repo:file.gguf`` -> (repo_id, filename or None)."""
    ref = ref.strip()
    if ":" in ref:
        repo, file = ref.split(":", 1)
        return repo.strip(), (file.strip() or None)
    return ref, None


def looks_like_repo(ref: str) -> bool:
    repo, _ = parse_ref(ref)
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", repo))


def _drop_templates(v: Any) -> Any:
    """The GGUF chat template can be tens of KB per repo; nothing here needs it."""
    for item in v if isinstance(v, list) else [v]:
        if isinstance(item, dict) and isinstance(item.get("gguf"), dict):
            item["gguf"].pop("chat_template", None)
    return v


# ----- client ---------------------------------------------------------------------


class HFCatalog:
    """Search Hugging Face for GGUF models (no auth, cached, offline-safe)."""

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        *,
        http: Optional[Http] = None,
        timeout: float = DEFAULT_TIMEOUT,
        ttl_s: float = CACHE_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cache_path = cache_path or Path.home() / ".eigencore" / "catalog-cache.json"
        self._http = http or _http_json
        self.timeout = timeout
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: Optional[dict[str, Any]] = None
        self.last_error: Optional[str] = None

    # cache ------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if self._cache is None:
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if not isinstance(self._cache, dict):
                    self._cache = {}
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    def _save(self) -> None:
        cache = self._load()
        if len(cache) > MAX_CACHE_ENTRIES:
            for k in sorted(cache, key=lambda k: cache[k].get("t", 0))[
                : len(cache) - MAX_CACHE_ENTRIES
            ]:
                cache.pop(k, None)
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            tmp.replace(self.cache_path)
        except OSError:
            pass

    def _cached(self, key: str, fresh_only: bool = True) -> Any:
        hit = self._load().get(key)
        if not hit:
            return None
        if fresh_only and self._clock() - hit.get("t", 0) > self.ttl_s:
            return None
        return hit.get("v")

    def _get(self, key: str, url: str, clean: Optional[Callable[[Any], Any]] = None) -> Any:
        """Fresh cache, else network, else stale cache. Raises only when all fail."""
        with self._lock:
            v = self._cached(key)
        if v is not None:
            return v
        try:
            v = self._http(url, self.timeout)
            if clean is not None:
                v = clean(v)
        except Exception:
            with self._lock:
                stale = self._cached(key, fresh_only=False)
            if stale is not None:
                return stale
            raise
        with self._lock:
            self._load()[key] = {"t": self._clock(), "v": v}
            self._save()
        return v

    # API ----------------------------------------------------------------------
    def model_info(self, repo_id: str) -> dict[str, Any]:
        url = f"{HF_API}/models/{urllib.parse.quote(repo_id, safe='/')}?blobs=true"
        return self._get(f"info:{repo_id}", url, clean=_drop_templates)

    def entry(self, ref: str) -> Optional[CatalogEntry]:
        """CatalogEntry for ``repo_id[:file]`` (None when offline or no GGUF)."""
        repo, file = parse_ref(ref)
        try:
            self.last_error = None
            return entry_from_info(self.model_info(repo), prefer_file=file)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return None

    def search(
        self,
        query: str,
        *,
        limit: int = 12,
        sort: str = "downloads",
        include_hidden: bool = False,
        trusted_only: bool = False,
    ) -> list[CatalogEntry]:
        """GGUF repos matching ``query``, with the 4-bit file picked and sized.

        Never raises: offline or on an HF error it returns the last cached
        result for this query, or ``[]`` with ``last_error`` set.
        """
        self.last_error = None
        q = query.strip()
        if not q:
            return []
        params = [
            ("filter", "gguf"),
            ("search", q),
            (
                "sort",
                sort
                if sort in ("downloads", "trendingScore", "likes", "lastModified")
                else "downloads",
            ),
            ("limit", "50"),
            ("expand[]", "gguf"),
            ("expand[]", "cardData"),
            ("expand[]", "downloads"),
            ("expand[]", "likes"),
            ("expand[]", "tags"),
            ("expand[]", "gated"),
        ]
        url = f"{HF_API}/models?{urllib.parse.urlencode(params)}"
        try:
            rows = self._get(f"search:{sort}:{q.lower()}", url, clean=_drop_templates)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return []
        if not isinstance(rows, list):
            return []
        repos = []
        for row in rows:
            rid = row.get("id") or row.get("modelId") or ""
            if not rid or (not include_hidden and _HIDDEN_REPO.search(rid)):
                continue
            if trusted_only and rid.split("/", 1)[0] not in TRUSTED_AUTHORS:
                continue
            repos.append(rid)
            if len(repos) >= limit:
                break

        def one(rid: str) -> Optional[CatalogEntry]:
            try:
                return entry_from_info(self.model_info(rid))
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                return None

        with ThreadPoolExecutor(max_workers=6) as ex:
            out = [e for e in ex.map(one, repos) if e is not None]
        if out:
            self.last_error = None
        # trusted quantisers first, then downloads (HF order) within each group
        order = {r: i for i, r in enumerate(repos)}
        out.sort(key=lambda e: (not e.trusted, order.get(e.repo_id, 0)))
        return out
