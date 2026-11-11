"""
EigenCore CLI — the user-facing interface.

Usage:
    eigencore profile              # show hardware profile + recommendations
    eigencore models               # list available models
    eigencore run --prompt "..."   # run inference (auto-selects model if not specified)
    eigencore chat                 # full-screen TUI chat on a TTY (banner, onboarding, model picker)
    eigencore download <model>     # pre-download a model
    eigencore analyze --prompt "." # measure activation sparsity on a prompt
    eigencore benchmark            # benchmark with/without Phase 2 optimizations
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eigencore.engine.inference import GenerationConfig, InferenceEngine
from eigencore.fit import (
    FitVerdict,
    estimate_fit,
    gpu_layers_for_vram,
    max_context_for_budget,
    read_gguf_model_info,
)
from eigencore.hal.profiler import profile_hardware
from eigencore.models.registry import ModelRegistry

console = Console()


def _resolve_or_bail(registry, model, hw, task="general"):
    """Resolve a model spec; if the resolved one isn't downloaded, fall back
    to the first downloaded model (with a note). Exit cleanly when none."""
    try:
        spec = registry.resolve(model, hw, task)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    if not registry.is_downloaded(spec):
        downloaded = next((m for m in registry.list_models() if registry.is_downloaded(m)), None)
        if downloaded is not None and model is None:
            console.print(
                f"[yellow]{spec.name} not downloaded; using downloaded [cyan]{downloaded.name}[/][/]"
            )
            return downloaded
        console.print(
            f"[red]Model {spec.name} is not downloaded and cannot be fetched automatically.[/]\n"
            f"  Download it with: eigencore download {spec.name}\n"
            f"  Or run bare `eigencore` for the interactive menu."
        )
        sys.exit(1)
    return spec


def _fit_for_spec(spec, hw, registry, kv_quant: str = "fp16"):
    """Fit estimate for a registry spec.

    Uses real GGUF metadata (geometry + file size) when the model file is
    downloaded; otherwise params-only, which yields an UNKNOWN verdict
    because the KV cache cannot be known without geometry.
    """
    info = None
    if registry.is_downloaded(spec):
        info = read_gguf_model_info(spec.local_path(registry.cache_dir))
    geometry = info.geometry if info else None
    file_size = info.file_size_bytes if info else None
    ctx_limit = info.context_length if info and info.context_length else spec.context_length
    est = estimate_fit(
        spec.params_b,
        spec.quant,
        hw.recommended_context_length,
        hw.available_ram_gb,
        geometry=geometry,
        file_size_bytes=file_size,
        bandwidth_gbps=hw.estimated_bandwidth_gbps,
        kv_quant=kv_quant,
        available_vram_gb=hw.available_vram_gb,
        active_params_b=(
            spec.params_b * spec.experts_active / spec.experts_total if spec.is_moe else None
        ),
    )
    gpu_layers = 0
    if hw.available_vram_gb > 0 and geometry is not None and geometry.num_layers and file_size:
        layer_bytes = int(file_size / geometry.num_layers)
        gpu_layers = gpu_layers_for_vram(
            geometry.num_layers, layer_bytes, hw.available_vram_gb, est.kv_cache_gb
        )
    max_ctx = max_context_for_budget(
        spec.params_b,
        spec.quant,
        hw.available_ram_gb,
        geometry=geometry,
        file_size_bytes=file_size,
        model_context_limit=ctx_limit,
    )
    return est, max_ctx, gpu_layers


@click.group(invoke_without_command=True)
@click.version_option(version="0.4.3", prog_name="eigencore")
@click.pass_context
def cli(ctx):
    """EigenCore — CPU-first LLM intelligence runtime."""
    if ctx.invoked_subcommand is None and sys.stdin.isatty():
        _home_menu(ctx)


def _home_menu(ctx):
    """Interactive home menu for bare `eigencore` on a TTY."""
    console.print(Panel("[bold]EigenCore[/] — local-first LLM runtime", border_style="cyan"))
    while True:
        console.print(
            "\n  [bold]1[/] profile   [bold]2[/] models   [bold]3[/] download   "
            "[bold]4[/] delete   [bold]5[/] chat   [bold]6[/] run   [bold]q[/] quit"
        )
        try:
            choice = console.input("[bold green]Choose:[/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        mapping = {
            "1": "profile",
            "2": "models",
            "3": "download",
            "4": "delete",
            "5": "chat",
            "6": "run",
            "q": None,
            "quit": None,
        }
        cmd = mapping.get(choice)
        if cmd is None:
            break
        try:
            if cmd in ("download", "delete"):
                name = console.input(f"Model name for {cmd}: ").strip()
                ctx.invoke(globals()[cmd], name=name)
            else:
                ctx.invoke(globals()[cmd])
        except SystemExit:
            pass  # subcommand called sys.exit — keep the menu alive


@cli.command()
def profile():
    """Detect hardware and show optimal model recommendations."""
    with console.status("[bold cyan]Profiling hardware..."):
        hw = profile_hardware()

    console.print()
    console.print(Panel(hw.summary(), title="[bold]Hardware Profile", border_style="cyan"))

    registry = ModelRegistry()
    rec = registry.recommend(hw)

    console.print()
    console.print(f"[bold green]Recommended model:[/] {rec.name} ({rec.description})")
    console.print(
        f"  Size: {rec.size_gb:.1f} GB | Params: {rec.params_b:.1f}B | Quant: {rec.quant}"
    )
    est, max_ctx, gpu_layers = _fit_for_spec(rec, hw, registry)
    verdict_style = {
        FitVerdict.FITS_WITH_MARGIN: "green",
        FitVerdict.TIGHT: "yellow",
        FitVerdict.DOES_NOT_FIT: "red",
        FitVerdict.UNKNOWN: "dim",
    }[est.verdict]
    console.print(
        f"  Fit estimate at ctx {est.context_length}: weights {est.weights_gb:.1f} GB + "
        f"reserve {est.runtime_reserve_gb:.1f} GB + KV {est.kv_cache_gb:.2f} GB = "
        f"{est.total_gb:.1f} GB of {est.available_ram_gb:.1f} GB available "
        f"[{verdict_style}]{est.verdict.value}[/] (headroom {est.headroom_gb:.1f} GB)"
    )
    if est.active_weights_gb is not None:
        console.print(
            f"  MoE: active weights/token ≈ {est.active_weights_gb:.1f} GiB "
            f"(resident file {est.weights_gb:.1f} GiB, {'RAM' if est.resident else 'mmap/SSD paging'})"
        )
    if est.decode_ceiling_tok_s is not None:
        console.print(
            f"  Decode ceiling estimate: ~{est.decode_ceiling_tok_s:.1f} tok/s "
            f"(bandwidth / active bytes per token — ceiling, not a measurement)"
        )
    if gpu_layers > 0:
        console.print(
            f"  [dim]GPU offload recommendation: {gpu_layers} layers to "
            f"{hw.gpu_name} ({hw.available_vram_gb:.1f} GiB VRAM free)[/]"
        )
    for note in est.notes:
        console.print(f"    [dim]* {note}[/]")
    if max_ctx > 0:
        console.print(f"  [dim]Max context that fits this budget: {max_ctx} tokens[/]")
    else:
        console.print(
            "  [dim]Exact context fit needs the model file downloaded (GGUF geometry); "
            "download with: eigencore download <name>[/]"
        )

    if not registry.is_downloaded(rec):
        console.print(f"\n  [dim]Download with:[/] eigencore download {rec.name}")
    else:
        console.print('\n  [dim]Already downloaded. Run with:[/] eigencore run --prompt "hello"')


@cli.command()
@click.option("--task", default=None, help="Filter by task type (general, code, math)")
def models(task: str | None):
    """List all available models."""
    registry = ModelRegistry()
    available = registry.list_models(task)

    hw = profile_hardware()

    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Active", justify="right")
    table.add_column("Quant")
    table.add_column("Size", justify="right")
    table.add_column("Task")
    table.add_column("Fits?", justify="center")
    table.add_column("Downloaded?", justify="center")
    table.add_column("Description")

    for m in sorted(available, key=lambda x: x.params_b):
        est, _, _ = _fit_for_spec(m, hw, registry)
        downloaded = registry.is_downloaded(m)
        if est.verdict == FitVerdict.FITS_WITH_MARGIN:
            fits_label, fits_style = "fits", "green"
        elif est.verdict == FitVerdict.TIGHT:
            fits_label, fits_style = "tight", "yellow"
        elif est.verdict == FitVerdict.UNKNOWN:
            fits_label, fits_style = "unknown*", "dim"
        else:
            fits_label, fits_style = "no", "red"
        table.add_row(
            m.name,
            f"{m.params_b:.1f}B",
            f"{m.params_b * m.experts_active / m.experts_total:.1f}B" if m.is_moe else "—",
            m.quant,
            f"{m.size_gb:.1f}GB",
            m.task,
            f"[{fits_style}]{fits_label}[/]",
            "[green]yes[/]" if downloaded else "[dim]no[/]",
            m.description,
        )

    console.print(table)
    console.print("[dim]* 'unknown*' = exact fit needs the model downloaded (GGUF geometry)[/]")


@cli.command()
@click.argument("name")
def download(name: str):
    """Download a model by name."""
    registry = ModelRegistry()
    hw = profile_hardware()

    try:
        model = registry.resolve(name, hw)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    if registry.is_downloaded(model):
        console.print(f"[green]Already downloaded:[/] {model.name}")
        console.print(f"  Path: {model.local_path(registry.cache_dir)}")
        return

    console.print(f"Downloading [cyan]{model.name}[/] ({model.size_gb:.1f} GB)...")
    console.print(f"  From: {model.repo_id}")

    try:
        path = registry.download(model)
        console.print(f"\n[green]Downloaded:[/] {path}")
    except Exception as e:
        console.print(f"\n[red]Download failed:[/] {e}")
        sys.exit(1)


@cli.command()
@click.argument("name")
def delete(name: str):
    """Delete a downloaded model (frees disk)."""
    registry = ModelRegistry()
    for model in registry.list_models():
        if model.name == name:
            if registry.delete(model):
                console.print(f"[green]Deleted:[/] {name}")
            else:
                console.print(f"[yellow]Not downloaded:[/] {name}")
            return
    console.print(f"[red]Unknown model:[/] {name}. Run: eigencore models")
    sys.exit(1)


@cli.command()
@click.option("--model", "-m", default=None, help="Model name (auto-selects if omitted)")
@click.option("--prompt", "-p", required=True, help="The prompt to send")
@click.option("--max-tokens", default=512, help="Maximum tokens to generate")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--task", default="general", help="Task type for auto model selection")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
@click.option(
    "--kv",
    "kv_type",
    default="fp16",
    type=click.Choice(["fp16", "q8_0"]),
    help="KV cache type (q8_0 halves KV memory)",
)
@click.option("--no-flash", is_flag=True, help="Disable flash attention")
@click.option(
    "--gpu-layers",
    default=0,
    help="Layers to offload to GPU (0 = CPU-only; auto-recommended when a GPU is detected and flag omitted)",
)
@click.option(
    "--pld",
    is_flag=True,
    help="Enable prompt-lookup decoding (copies prompt-verbatim spans; default OFF until measured)",
)
def run(
    model: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    task: str,
    no_stream: bool,
    kv_type: str,
    no_flash: bool,
    gpu_layers: int,
    pld: bool,
):
    """Run inference with a prompt."""
    hw = profile_hardware()
    registry = ModelRegistry()

    spec = _resolve_or_bail(registry, model, hw, task)
    model_path = spec.local_path(registry.cache_dir)

    console.print(
        f"[dim]Model: {spec.name} | Threads: {hw.recommended_threads} | Ctx: {hw.recommended_context_length}[/]"
    )

    if gpu_layers == 0 and hw.available_vram_gb > 0:
        _, _, gpu_layers = _fit_for_spec(spec, hw, registry, kv_quant=kv_type)
    if gpu_layers > 0:
        console.print(f"[dim]Offloading {gpu_layers} layers to {hw.gpu_name}[/]")
    engine = InferenceEngine(
        model_path, hw, spec, kv_type=kv_type, flash_attn=not no_flash, n_gpu_layers=gpu_layers
    )

    with console.status("[bold cyan]Loading model..."):
        engine.load()

    config = GenerationConfig(
        prompt_lookup=pld,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=not no_stream,
    )

    if no_stream:
        result = engine.generate(prompt, config)
        console.print()
        console.print(result.text)
        console.print(
            f"\n[dim]{result.tokens_generated} tokens in {result.time_seconds:.1f}s ({result.tokens_per_second:.1f} tok/s)[/]"
        )
    else:
        console.print()
        token_count = 0
        start_time = __import__("time").perf_counter()
        for token in engine.stream(prompt, config):
            console.print(token, end="", highlight=False)
            token_count += 1
        elapsed = __import__("time").perf_counter() - start_time
        tps = token_count / elapsed if elapsed > 0 else 0
        console.print(f"\n\n[dim]{token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)[/]")

    engine.unload()


@cli.command()
@click.option("--model", "-m", default=None, help="Model name (auto-selects if omitted)")
@click.option("--max-tokens", default=512, help="Maximum tokens per response")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--task", default="general", help="Task type for auto model selection")
@click.option(
    "--plain", is_flag=True, help="Plain-text mode: no panels/colors (pipes, SSH, screen readers)"
)
@click.option(
    "--kv",
    "kv_type",
    default="fp16",
    type=click.Choice(["fp16", "q8_0"]),
    help="KV cache type (q8_0 halves KV memory)",
)
@click.option("--no-flash", is_flag=True, help="Disable flash attention")
@click.option(
    "--gpu-layers",
    default=0,
    help="Layers to offload to GPU (0 = CPU-only; auto-recommended when a GPU is detected and flag omitted)",
)
@click.option(
    "--pld",
    is_flag=True,
    help="Enable prompt-lookup decoding (copies prompt-verbatim spans; default OFF until measured)",
)
def chat(
    model: str | None,
    max_tokens: int,
    temperature: float,
    task: str,
    plain: bool,
    kv_type: str,
    no_flash: bool,
    gpu_layers: int,
    pld: bool,
):
    """Start an interactive chat session (full-screen TUI on a TTY, streaming, slash
    commands; --plain or piped stdin gives the line-mode UI)."""
    hw = profile_hardware()
    registry = ModelRegistry()
    from eigencore.tui.theme import probe_terminal

    caps = probe_terminal(force_plain=plain)
    if sys.stdin.isatty() and not caps.plain:
        from eigencore.tui.banner import render_banner

        render_banner(console)

    spec = _resolve_or_bail(registry, model, hw, task)
    model_path = spec.local_path(registry.cache_dir)
    if gpu_layers == 0 and hw.available_vram_gb > 0:
        _, _, gpu_layers = _fit_for_spec(spec, hw, registry, kv_quant=kv_type)

    # SSD-paging warning before load when the file exceeds free RAM (TUI.md 3.4)
    est0, _, _ = _fit_for_spec(spec, hw, registry, kv_quant=kv_type)
    if not est0.resident:
        console.print(
            f"[yellow]WARNING:[/] {spec.name} file ~{est0.weights_gb:.1f} GiB > free RAM "
            f"{est0.available_ram_gb:.1f} GiB — loading will page from SSD (severe slowdown).[/]"
        )

    # First-run onboarding: only when interactive and not yet onboarded
    from eigencore.tui.onboarding import load_config, run_onboarding, save_config

    cfg_store = load_config()
    interactive = sys.stdin.isatty() and not plain
    if interactive and not cfg_store.get("onboarded") and model is None:
        estimates = {
            m.name: _fit_for_spec(m, hw, registry, kv_quant=kv_type)[0]
            for m in registry.list_models()
        }
        chosen = run_onboarding(
            hw,
            registry,
            estimates,
            ask=lambda prompt="": console.input(prompt),
            out=lambda line="": console.print(line),
        )
        if chosen is not None and chosen.name != spec.name:
            spec = chosen
            if not registry.is_downloaded(spec):
                console.print(f"Downloading [cyan]{spec.name}[/] ({spec.size_gb:.1f} GB)...")
                registry.download(spec)
            model_path = spec.local_path(registry.cache_dir)
        cfg_store["onboarded"] = True
        cfg_store["model"] = spec.name
        save_config(cfg_store)

    engine = InferenceEngine(
        model_path, hw, spec, kv_type=kv_type, flash_attn=not no_flash, n_gpu_layers=gpu_layers
    )

    with console.status("[bold cyan]Loading model..."):
        engine.load()

    from eigencore.tui.chat import ChatUiConfig

    info = read_gguf_model_info(model_path)
    geometry = info.geometry if info else None
    cfg = ChatUiConfig(
        plain=caps.plain,
        max_tokens=max_tokens,
        temperature=temperature,
        prompt_lookup=pld,
    )
    if caps.plain:
        from eigencore.tui.chat import ChatUI

        ui = ChatUI(
            engine,
            spec,
            hw,
            geometry=geometry,
            policy_label="read-only (policy engine attached in P4)",
            caps=caps,
            cfg=cfg,
        )
        ui.approval_card = ui.render_approval_card
        ui.run()
    else:
        from eigencore.tui.app import ChatApp

        state = {"engine": engine}

        def _switch_model(new_spec):
            """Reload the engine with another (downloaded) model from the TUI."""
            state["engine"].unload()
            state["engine"] = InferenceEngine(
                new_spec.local_path(registry.cache_dir),
                hw,
                new_spec,
                kv_type=kv_type,
                flash_attn=not no_flash,
                n_gpu_layers=gpu_layers,
            )
            with console.status("[bold cyan]Loading model..."):
                state["engine"].load()

        app = ChatApp(
            engine_getter=lambda: state["engine"],
            spec=spec,
            profile=hw,
            registry=registry,
            geometry=geometry,
            cfg=cfg,
            caps=caps,
            on_model_change=_switch_model,
        )
        app.run()
        engine = state["engine"]

    engine.unload()


@cli.command()
@click.option("--model", "-m", default=None, help="Model name (auto-selects if omitted)")
@click.option("--prompt", "-p", required=True, help="Prompt to analyze sparsity on")
@click.option("--threshold", default=0.01, help="Near-zero threshold")
@click.option("--output", "-o", default=None, help="Save report to JSON file")
def analyze(
    model: str | None,
    prompt: str,
    threshold: float,
    output: str | None,
):
    """Measure OUTPUT-LOGIT sparsity for a prompt (not neuron activation sparsity)."""
    from eigencore.analysis.sparsity import SparsityAnalyzer

    hw = profile_hardware()
    registry = ModelRegistry()

    spec = _resolve_or_bail(registry, model, hw)
    model_path = spec.local_path(registry.cache_dir)
    engine = InferenceEngine(model_path, hw, spec)

    with console.status("[bold cyan]Loading model..."):
        engine.load()

    console.print(f"[dim]Analyzing sparsity for: {prompt[:80]}[/]")

    try:
        llm = engine._llm

        llm.reset()
        tokens = llm.tokenize(prompt.encode("utf-8"))
        llm.eval(tokens)

        n_vocab = llm.n_vocab()
        logits_ptr = llm._ctx.get_logits()
        logits = [logits_ptr[i] for i in range(n_vocab)]

        analyzer = SparsityAnalyzer(threshold=threshold)

        report_logits = analyzer.analyze_logits(logits, spec.name, prompt)
        report_probs = analyzer.analyze_token_probabilities(logits, spec.name, prompt)

        console.print()
        console.print(
            Panel(
                report_logits.summary(),
                title="[bold]Logit Sparsity Analysis",
                border_style="yellow",
            )
        )
        console.print()
        console.print(
            Panel(
                report_probs.summary(),
                title="[bold]Probability Concentration Analysis",
                border_style="green",
            )
        )

        console.print("\n[bold]Key findings:[/]")
        console.print(
            f"  Logit sparsity: [cyan]{report_logits.overall_sparsity:.1%}[/] of logits near zero"
        )
        console.print(
            f"  Probability concentration: [cyan]{report_probs.overall_sparsity:.1%}[/] of mass in top tokens"
        )
        console.print("  (No speedup claimed: logit sparsity is not a measured inference speedup.)")

        if output:
            combined = {
                "logit_analysis": report_logits.to_dict(),
                "probability_analysis": report_probs.to_dict(),
            }
            Path(output).write_text(__import__("json").dumps(combined, indent=2))
            console.print(f"\n[dim]Report saved to {output}[/]")

    except Exception as e:
        console.print(f"[red]Analysis error:[/] {e}")
        import traceback

        traceback.print_exc()
    finally:
        engine.unload()


@cli.command()
@click.option("--model", "-m", default=None, help="Model name (auto-selects if omitted)")
@click.option("--prompt", "-p", default=None, help="Custom prompt to benchmark with")
@click.option("--max-tokens", default=128, help="Tokens to generate per run")
@click.option("--runs", default=3, help="Number of runs per configuration")
@click.option("--output", "-o", default=None, help="Save results to JSON file")
@click.option(
    "--report",
    "report_path",
    default=None,
    help="Save a full Phase 1 evidence report (hardware, model sha256, timing, quality) to this JSON path",
)
def benchmark(
    model: str | None,
    prompt: str | None,
    max_tokens: int,
    runs: int,
    output: str | None,
    report_path: str | None,
):
    """Benchmark inference across configurations.

    NOTE (P3 audit): the layer_skip / sparse / all_optimizations configs only
    change reported stats today — no optimization is wired into the backend
    call, so speedup vs baseline is expected to be ~1.0x (noise) until P3
    wires them in and the quality gate passes.
    """
    import json
    import time

    import numpy as np

    hw = profile_hardware()
    registry = ModelRegistry()

    spec = _resolve_or_bail(registry, model, hw)

    model_path = spec.local_path(registry.cache_dir)

    test_prompts = [
        prompt or "Explain how a CPU executes instructions step by step.",
        "Write a Python function that implements quicksort with comments.",
        "Summarize the key differences between TCP and UDP protocols.",
    ]

    if prompt:
        test_prompts = [prompt]

    console.print(
        Panel(
            f"Model: [cyan]{spec.name}[/] ({spec.params_b:.1f}B {spec.quant})\n"
            f"Tokens per run: {max_tokens}\n"
            f"Runs per config: {runs}\n"
            f"Test prompts: {len(test_prompts)}",
            title="[bold]EigenCore Benchmark",
            border_style="cyan",
        )
    )

    engine = InferenceEngine(model_path, hw, spec)
    with console.status("[bold cyan]Loading model..."):
        engine.load()

    if report_path:
        from eigencore.benchmark import DEFAULT_CASES, build_report, run_cases

        baseline_config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
            enable_layer_skip=False,
            enable_sparse_inference=False,
        )
        case_results = run_cases(engine, DEFAULT_CASES, baseline_config, runs=1)
        report = build_report(engine, spec, model_path, baseline_config, case_results)
        saved = report.save(report_path)
        console.print(
            f"[dim]Phase 1 evidence report saved to {saved} "
            f"(quality pass rate {report.summary['quality_pass_rate']:.0%}, "
            f"median decode {report.summary['decode_tok_s_median']:.1f} tok/s)[/]"
        )

    configs = [
        (
            "baseline",
            GenerationConfig(
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
                enable_layer_skip=False,
                enable_sparse_inference=False,
                complexity=0.5,
            ),
        ),
        (
            "layer_skip",
            GenerationConfig(
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
                enable_layer_skip=True,
                enable_sparse_inference=False,
                complexity=0.5,
            ),
        ),
        (
            "sparse",
            GenerationConfig(
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
                enable_layer_skip=False,
                enable_sparse_inference=True,
                complexity=0.5,
            ),
        ),
        (
            "all_optimizations",
            GenerationConfig(
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
                enable_layer_skip=True,
                enable_sparse_inference=True,
                complexity=0.3,
            ),
        ),
    ]

    results_data: dict = {"model": spec.name, "max_tokens": max_tokens, "runs": runs, "configs": {}}

    table = Table(title="Benchmark Results")
    table.add_column("Config", style="cyan")
    table.add_column("Avg tok/s", justify="right")
    table.add_column("Min tok/s", justify="right")
    table.add_column("Max tok/s", justify="right")
    table.add_column("Avg latency", justify="right")
    table.add_column("Speedup", justify="right", style="green")

    baseline_tps = 0.0

    for config_name, config in configs:
        all_tps: list[float] = []
        all_latency: list[float] = []

        console.print(f"\n  Running [cyan]{config_name}[/]...")

        for prompt_text in test_prompts:
            for run_idx in range(runs):
                start = time.perf_counter()
                result = engine.generate(prompt_text, config)
                elapsed = time.perf_counter() - start

                tps = result.tokens_generated / elapsed if elapsed > 0 else 0.0
                all_tps.append(tps)
                all_latency.append(elapsed)

                console.print(
                    f"    Run {run_idx + 1}/{runs}: "
                    f"{result.tokens_generated} tok in {elapsed:.1f}s "
                    f"({tps:.1f} tok/s)"
                )

        avg_tps = float(np.mean(all_tps))
        min_tps = float(np.min(all_tps))
        max_tps = float(np.max(all_tps))
        avg_latency = float(np.mean(all_latency))

        if config_name == "baseline":
            baseline_tps = avg_tps

        speedup = avg_tps / baseline_tps if baseline_tps > 0 else 1.0

        table.add_row(
            config_name,
            f"{avg_tps:.1f}",
            f"{min_tps:.1f}",
            f"{max_tps:.1f}",
            f"{avg_latency:.1f}s",
            f"{speedup:.2f}x",
        )

        results_data["configs"][config_name] = {
            "avg_tps": avg_tps,
            "min_tps": min_tps,
            "max_tps": max_tps,
            "avg_latency_s": avg_latency,
            "speedup": speedup,
            "samples": len(all_tps),
        }

    console.print()
    console.print(table)
    console.print(
        "[yellow]P3 audit:[/] optimization configs change reported stats only; "
        "no optimization is applied to the backend call yet, so 'Speedup' is "
        "noise until they are wired in.[/]"
    )

    opt_stats = engine.optimization_stats
    if opt_stats:
        console.print("\n[bold]Optimization Module Status:[/]")
        if "layer_skip" in opt_stats:
            ls = opt_stats["layer_skip"]
            console.print(
                f"  Layer skip: {ls['layers_skipped']}/{ls['total_layers']} layers "
                f"({ls['skip_rate']:.0%}) | ~{ls['expected_speedup']:.1f}x potential"
            )
        if "sparse_inference" in opt_stats:
            si = opt_stats["sparse_inference"]
            console.print(
                f"  Sparse inference: {si['overall_sparsity']:.0%} sparsity | "
                f"~{si['estimated_speedup']:.1f}x potential"
            )
        console.print(f"  Sparsity cache: {opt_stats['sparsity_cache']['hit_rate']:.0%} hit rate")
        console.print(
            f"  Speculative decoder: draft_length={opt_stats['speculative']['draft_length']}"
        )

    if output:
        results_data["optimization_stats"] = opt_stats
        Path(output).write_text(json.dumps(results_data, indent=2))
        console.print(f"\n[dim]Results saved to {output}[/]")

    engine.unload()


@cli.command()
@click.option(
    "--base-model", "-b", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model ID"
)
@click.option("--data", "-d", required=True, help="Path to training data (jsonl/json/txt/csv)")
@click.option("--output", "-o", default="./eigencore-adapters", help="Output directory for adapter")
@click.option("--lora-rank", default=8, help="LoRA rank (4-64)")
@click.option("--epochs", default=100, help="Initial epochs per phase")
@click.option("--lr", default=2e-4, help="Learning rate")
def train(
    base_model: str,
    data: str,
    output: str,
    lora_rank: int,
    epochs: int,
    lr: float,
):
    """Fine-tune a model with QLoRA on CPU using adaptive epoch scaling."""
    from eigencore.training.trainer import CPUTrainer, TrainingConfig

    hw = profile_hardware()

    config = TrainingConfig(
        base_model=base_model,
        dataset_path=data,
        output_dir=output,
        lora_rank=lora_rank,
        initial_epochs=epochs,
        learning_rate=lr,
    )

    def on_epoch(metrics):
        console.print(
            f"  Phase {metrics.phase} | Epoch {metrics.epoch} | "
            f"Loss: {metrics.train_loss:.4f} | "
            f"RAM: {metrics.memory_used_gb:.1f}GB | "
            f"Time: {metrics.elapsed_seconds:.0f}s"
        )

    console.print(
        Panel(
            f"Base model: [cyan]{base_model}[/]\n"
            f"Data: {data}\n"
            f"LoRA rank: {lora_rank}\n"
            f"Initial epochs: {epochs}\n"
            f"Adaptive scaling: 1/4 consistency rule",
            title="[bold]EigenCore Training",
            border_style="cyan",
        )
    )

    try:
        trainer = CPUTrainer(config, hw, on_epoch_end=on_epoch)
        result = trainer.train()
        console.print()
        console.print(
            Panel(result.summary(), title="[bold green]Training Complete", border_style="green")
        )
    except ImportError as e:
        console.print(f"\n[red]Missing dependencies:[/] {e}")
        console.print("[dim]Install with: pip install eigencore[train][/]")
        sys.exit(1)
    except MemoryError as e:
        console.print(f"\n[red]Memory error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
