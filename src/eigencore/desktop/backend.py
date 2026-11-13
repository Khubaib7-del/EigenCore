"""Desktop backend bridge — UI-agnostic and fully testable headlessly.

Wraps ModelRegistry + InferenceEngine and talks to the GUI only through a
thread-safe `events` queue, so the Flet layer is a thin renderer and all
logic (model rows, download/delete, engine load, streaming, cancel) is
tested without a display. Event tuples:

    ("dl_progress", name, pct)       download progress 0..100
    ("dl_done", name, ok, message)   download finished/failed
    ("engine", state, message)       "loading" / "ready" / "error"
    ("prompt_eval", started)         True when generation starts (prompt is
                                     being read), False at the first token
    ("tok", text)                    streamed token
    ("done", stats)                  generation finished (tokens, seconds,
                                     tok_per_sec, first_token_s,
                                     decode_tok_per_sec, interrupted, error)
    ("deleted", name, ok)
    ("switched", name)
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.fit import FitVerdict, estimate_fit, read_gguf_model_info
from eigencore.hal.profiler import HardwareProfile, profile_hardware
from eigencore.models.registry import ModelRegistry, ModelSpec

REPO = "Khubaib7-del/EigenCore"
RELEASES_URL = f"https://github.com/{REPO}/releases/latest"
EXE_URL = f"https://github.com/{REPO}/releases/latest/download/EigenCore.exe"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"


def _http_get(url: str, timeout: float = 8.0) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "eigencore-desktop", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def latest_release(fetcher: Optional[Callable[[str], str]] = None) -> Optional[dict]:
    """Latest GitHub release: {"tag": "v0.4.4", "url": ...}; None on any error."""
    fetcher = fetcher or _http_get
    try:
        data = json.loads(fetcher(API_LATEST))
        return {"tag": data.get("tag_name", ""), "url": data.get("html_url", RELEASES_URL)}
    except Exception:
        return None


def is_newer(tag: str, current_version: str) -> bool:
    """True when release tag v0.4.5 is newer than current_version 0.4.4."""
    try:
        return tuple(int(p) for p in tag.lstrip("vV").split(".")) > tuple(
            int(p) for p in current_version.split(".")
        )
    except ValueError:
        return False


@dataclass
class ModelRow:
    name: str
    params_b: float
    quant: str
    size_gb: float
    task: str
    context_length: int
    description: str
    downloaded: bool
    fit: str
    is_active: bool = False


def fit_label(spec: ModelSpec, registry: ModelRegistry, profile: HardwareProfile) -> str:
    """Fit verdict string; UNKNOWN until the file is downloaded (no geometry)."""
    if not registry.is_downloaded(spec):
        return FitVerdict.UNKNOWN.value
    try:
        info = read_gguf_model_info(spec.local_path(registry.cache_dir))
        est = estimate_fit(
            spec.params_b,
            spec.quant,
            profile.recommended_context_length,
            profile.available_ram_gb,
            geometry=info.geometry if info else None,
            file_size_bytes=info.file_size_bytes if info else None,
        )
        return est.verdict.value
    except Exception:
        return FitVerdict.UNKNOWN.value


class DesktopBackend:
    """Owns registry, profile, engine and all worker threads."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        profile: Optional[HardwareProfile] = None,
        engine_factory: Optional[Callable[[ModelSpec], object]] = None,
        events: Optional[queue.Queue] = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.profile = profile or profile_hardware()
        self._engine_factory = engine_factory or self._default_engine_factory
        self.events: queue.Queue = events if events is not None else queue.Queue()
        self.engine = None
        self.active_spec: Optional[ModelSpec] = None
        self.messages: list[dict[str, str]] = []
        self._cancel = threading.Event()
        self._gen_lock = threading.Lock()
        self._gen_active = False
        self._dl_lock = threading.Lock()
        self._downloading: set[str] = set()

    # ----- models -----------------------------------------------------------
    def find(self, name: str) -> Optional[ModelSpec]:
        return next((m for m in self.registry.list_models() if m.name == name), None)

    def model_rows(self) -> list[ModelRow]:
        return [
            ModelRow(
                name=m.name,
                params_b=m.params_b,
                quant=m.quant,
                size_gb=m.size_gb,
                task=m.task,
                context_length=m.context_length,
                description=m.description,
                downloaded=self.registry.is_downloaded(m),
                fit=fit_label(m, self.registry, self.profile),
                is_active=self.active_spec is not None and m.name == self.active_spec.name,
            )
            for m in self.registry.list_models()
        ]

    def downloaded_names(self) -> list[str]:
        return [m.name for m in self.registry.list_models() if self.registry.is_downloaded(m)]

    # ----- download / delete --------------------------------------------------
    def download(self, name: str) -> None:
        spec = self.find(name)
        if spec is None:
            self.events.put(("dl_done", name, False, f"unknown model: {name}"))
            return
        with self._dl_lock:
            if name in self._downloading:
                return
            if self.registry.is_downloaded(spec):
                self.events.put(("dl_done", name, True, f"{name} already downloaded"))
                return
            self._downloading.add(name)

        def worker() -> None:
            try:
                ok = self._download_with_progress(spec)
                if ok:
                    self.events.put(("dl_done", name, True, f"downloaded {name}"))
                else:
                    self.events.put(
                        ("dl_done", name, False, f"{name} missing on disk after download")
                    )
            except Exception as e:
                self.events.put(
                    ("dl_done", name, False, f"download failed: {type(e).__name__}: {e}")
                )
            finally:
                with self._dl_lock:
                    self._downloading.discard(name)

        threading.Thread(target=worker, daemon=True).start()

    def _download_with_progress(self, spec: ModelSpec) -> bool:
        import io

        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                gb = self._downloaded_gb(spec)
                pct = min(100.0, 100.0 * gb / max(spec.size_gb, 1e-9))
                self.events.put(("dl_progress", spec.name, pct))
                stop.wait(0.5)

        t = threading.Thread(target=poll, daemon=True)
        t.start()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                self.registry.download(spec, progress=False)
        finally:
            stop.set()
            t.join(timeout=2)
            self.events.put(("dl_progress", spec.name, 100.0))
        return self.registry.is_downloaded(spec)

    def _downloaded_gb(self, spec: ModelSpec) -> float:
        parent = spec.local_path(self.registry.cache_dir).parent
        total = 0
        if parent.exists():
            prefix = self.registry.shard_prefix(spec)
            pattern = f"{prefix}-*.gguf" if prefix else spec.filename
            for f in parent.glob(pattern):
                total += f.stat().st_size
        return total / (1024**3)

    def delete(self, name: str) -> bool:
        """Delete from disk. Active model is protected (switch first)."""
        spec = self.find(name)
        if spec is None or not self.registry.is_downloaded(spec):
            self.events.put(("deleted", name, False))
            return False
        if self.active_spec is not None and name == self.active_spec.name:
            self.events.put(("deleted", name, False))
            return False
        removed = self.registry.delete(spec)
        self.events.put(("deleted", name, removed))
        return removed

    # ----- engine -------------------------------------------------------------
    def _default_engine_factory(self, spec: ModelSpec):
        path = spec.local_path(self.registry.cache_dir)
        return InferenceEngine(path, self.profile, spec)

    def switch_model(self, name: str) -> None:
        """Load a downloaded model as the active engine (worker thread)."""
        spec = self.find(name)
        if spec is None or not self.registry.is_downloaded(spec):
            self.events.put(("engine", "error", f"{name} is not downloaded"))
            return

        def worker() -> None:
            self.events.put(("engine", "loading", f"loading {name}…"))
            try:
                old = self.engine
                engine = self._engine_factory(spec)
                engine.load()
                self.engine = engine
                self.active_spec = spec
                self.messages.clear()
                if old is not None and old is not engine:
                    with contextlib.suppress(Exception):
                        old.unload()
                self.events.put(("engine", "ready", f"{name} ready"))
                self.events.put(("switched", name))
            except Exception as e:
                msg = str(e)
                if "llama_cpp" in msg or "inference" in msg.lower():
                    msg += (
                        ' — install the backend: pip install "eigencore[inference]" '
                        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
                    )
                self.events.put(("engine", "error", f"could not load {name}: {msg}"))

        threading.Thread(target=worker, daemon=True).start()

    def ensure_active(self) -> Optional[ModelSpec]:
        """Active spec if ready; else kick off a switch to the first downloaded model."""
        if self.active_spec is not None and self.registry.is_downloaded(self.active_spec):
            return self.active_spec
        dl = self.downloaded_names()
        if dl:
            self.switch_model(dl[0])
        return None

    # ----- chat ---------------------------------------------------------------
    @property
    def gen_active(self) -> bool:
        return self._gen_active

    def send(
        self,
        text: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        history: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """Stream a reply to ``text`` on a worker thread.

        ``history`` (earlier user/assistant turns) replaces the backend's own
        conversation, so a UI with several chats keeps each one's context.
        """
        text = text.strip()
        if not text:
            return
        if self.engine is None or self.active_spec is None:
            self.events.put(
                ("engine", "error", "no model loaded — download one in Models and select it")
            )
            return
        with self._gen_lock:
            if self._gen_active:
                return
            self._gen_active = True
        self._cancel.clear()
        if history is not None:
            self.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
        self.messages.append({"role": "user", "content": text})
        msgs = list(self.messages)
        engine = self.engine
        cancel = self._cancel
        events = self.events

        history_ref = self.messages

        def worker() -> None:
            start = time.perf_counter()
            first_at: Optional[float] = None
            count = 0
            err = None
            gen = None
            parts: list[str] = []
            events.put(("prompt_eval", True))
            try:
                gen = engine.stream_messages(
                    msgs,
                    GenerationConfig(max_tokens=max_tokens, temperature=temperature, stream=True),
                )
                for tok in gen:
                    if cancel.is_set():
                        break
                    if first_at is None:
                        first_at = time.perf_counter()
                        events.put(("prompt_eval", False))
                    events.put(("tok", tok))
                    parts.append(tok)
                    count += 1
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            finally:
                if gen is not None and hasattr(gen, "close"):
                    with contextlib.suppress(Exception):
                        gen.close()
            end = time.perf_counter()
            elapsed = end - start
            # decode rate: pieces after the first one, over the time after it
            decode_s = (end - first_at) if first_at is not None else 0.0
            if parts:
                # keep the reply so the next turn has the whole conversation
                history_ref.append({"role": "assistant", "content": "".join(parts)})
            events.put(
                (
                    "done",
                    {
                        "tokens": count,
                        "seconds": elapsed,
                        "tok_per_sec": (count / elapsed) if elapsed > 0 else 0.0,
                        "first_token_s": (first_at - start) if first_at is not None else None,
                        "decode_tok_per_sec": (
                            (count - 1) / decode_s if count > 1 and decode_s > 0 else None
                        ),
                        "interrupted": cancel.is_set(),
                        "error": err,
                    },
                )
            )
            with self._gen_lock:
                self._gen_active = False

        threading.Thread(target=worker, daemon=True).start()

    def stop(self) -> None:
        self._cancel.set()
