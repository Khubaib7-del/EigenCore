"""
First-run onboarding (docs/design/TUI.md §3.3–3.4).

- Model choice grouped by fit verdict (fits / tight / does not fit), MoE
  models show resident GB vs read-per-token GB separately; speed is labeled
  "estimate" — never a measurement.
- SSD-paging warning before load when resident size exceeds free RAM.
- Plain mode prints the same information, one line per item; non-TTY or
  --plain never blocks: onboarding is skipped entirely when not interactive.

Config is persisted to ~/.eigencore/config.json so onboarding runs once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from eigencore.fit import FitEstimate, FitVerdict
from eigencore.hal.profiler import HardwareProfile
from eigencore.models.registry import ModelRegistry, ModelSpec


def default_config_path() -> Path:
    return Path.home() / ".eigencore" / "config.json"


def load_config(path: Optional[Path] = None) -> dict:
    path = path or default_config_path()
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_config(cfg: dict, path: Optional[Path] = None) -> None:
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def group_by_fit(
    models: list[ModelSpec],
    estimates: dict[str, FitEstimate],
) -> dict[str, list[ModelSpec]]:
    """Group models into fits / tight / no / unknown by verdict."""
    groups: dict[str, list[ModelSpec]] = {"fits": [], "tight": [], "no": [], "unknown": []}
    for m in models:
        est = estimates.get(m.name)
        v = est.verdict if est else FitVerdict.UNKNOWN
        key = {
            FitVerdict.FITS_WITH_MARGIN: "fits",
            FitVerdict.TIGHT: "tight",
            FitVerdict.DOES_NOT_FIT: "no",
            FitVerdict.UNKNOWN: "unknown",
        }[v]
        groups[key].append(m)
    return groups


def ssd_warning_lines(spec: ModelSpec, est: FitEstimate) -> list[str]:
    """Warning lines shown before loading a model whose file exceeds free RAM."""
    if est.resident:
        return []
    return [
        f"WARNING: {spec.name} file is ~{est.weights_gb:.1f} GiB but only "
        f"{est.available_ram_gb:.1f} GiB RAM is free.",
        "Loading it will page from SSD (mmap) — expect severe slowdown.",
        "Pick a smaller model, or proceed knowing decode may crawl.",
    ]


def run_onboarding(
    hw: HardwareProfile,
    registry: ModelRegistry,
    estimates: dict[str, FitEstimate],
    ask: Callable[[str], str],
    out: Callable[[str], None],
) -> ModelSpec | None:
    """Interactive first-run model picker. `ask` reads a line, `out` prints.

    Returns the chosen spec, or None if the user aborts. Caller decides
    whether to persist config (save_config with onboarded=true).
    """
    models = registry.list_models()
    groups = group_by_fit(models, estimates)
    out("EigenCore first run — choose a local model for this machine:")
    out(
        f"  RAM free: {hw.available_ram_gb:.1f} GiB"
        + (
            f" | GPU: {hw.gpu_name} ({hw.available_vram_gb:.1f} GiB VRAM free)"
            if hw.gpu_name
            else " | GPU: none"
        )
    )
    out("  Speed shown is an ESTIMATE (bandwidth ceiling), not a measurement.")
    out("")
    ordered: list[tuple[str, str]] = [
        ("fits", "Fits with margin"),
        ("tight", "Tight (may swap under load)"),
        ("unknown", "Unknown fit (model not downloaded yet)"),
        ("no", "Does not fit free RAM"),
    ]
    flat: list[ModelSpec] = []
    for key, label in ordered:
        if not groups[key]:
            continue
        out(f"[{label}]")
        for m in sorted(groups[key], key=lambda x: x.params_b):
            est = estimates.get(m.name)
            extra = ""
            if m.is_moe and est is not None and est.active_weights_gb is not None:
                extra = f" | resident {est.weights_gb:.1f} GiB, read/token {est.active_weights_gb:.1f} GiB"
            elif est is not None:
                extra = f" | ~{est.weights_gb:.1f} GiB"
            out(f"  {len(flat) + 1}. {m.name} ({m.params_b:.1f}B {m.quant}){extra}")
            flat.append(m)
        out("")
    if not flat:
        out("No models in registry.")
        return None
    out("Enter a number to select, or 'q' to quit:")
    answer = ask("").strip()
    if answer.lower() in ("q", ""):
        return None
    try:
        idx = int(answer) - 1
        if 0 <= idx < len(flat):
            return flat[idx]
    except ValueError:
        pass
    out("Invalid selection.")
    return None
