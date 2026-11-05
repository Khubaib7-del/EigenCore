"""
EigenCore CLI — the user-facing interface.

Usage:
    eigencore profile              # show hardware profile + recommendations
    eigencore models               # list available models
    eigencore run --prompt "..."   # run inference (auto-selects model if not specified)
    eigencore chat                 # interactive chat session
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
from eigencore.fit import FitVerdict, estimate_fit, max_context_for_budget
from eigencore.hal.profiler import profile_hardware
from eigencore.models.registry import ModelRegistry

console = Console()


@click.group()
@click.version_option(version="0.3.0", prog_name="eigencore")
def cli():
    """EigenCore — CPU-first LLM intelligence runtime."""


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
    est = estimate_fit(
        rec.params_b, rec.quant, hw.recommended_context_length, hw.available_ram_gb
    )
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
    for note in est.notes:
        console.print(f"    [dim]* {note}[/]")
    max_ctx = max_context_for_budget(
        rec.params_b, rec.quant, hw.available_ram_gb, model_context_limit=rec.context_length
    )
    if max_ctx > 0:
        console.print(f"  [dim]Max context that fits this budget: {max_ctx} tokens[/]")
    else:
        console.print("  [red]Model does not fit even at minimum context[/]")

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
    table.add_column("Quant")
    table.add_column("Size", justify="right")
    table.add_column("Task")
    table.add_column("Fits?", justify="center")
    table.add_column("Downloaded?", justify="center")
    table.add_column("Description")

    for m in sorted(available, key=lambda x: x.params_b):
        est = estimate_fit(m.params_b, m.quant, hw.recommended_context_length, hw.available_ram_gb)
        downloaded = registry.is_downloaded(m)
        if est.verdict == FitVerdict.FITS_WITH_MARGIN:
            fits_label, fits_style = "fits", "green"
        elif est.verdict == FitVerdict.TIGHT:
            fits_label, fits_style = "tight", "yellow"
        else:
            fits_label, fits_style = "no", "red"
        table.add_row(
            m.name,
            f"{m.params_b:.1f}B",
            m.quant,
            f"{m.size_gb:.1f}GB",
            m.task,
            f"[{fits_style}]{fits_label}[/]",
            "[green]yes[/]" if downloaded else "[dim]no[/]",
            m.description,
        )

    console.print(table)


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
@click.option("--model", "-m", default=None, help="Model name (auto-selects if omitted)")
@click.option("--prompt", "-p", required=True, help="The prompt to send")
@click.option("--max-tokens", default=512, help="Maximum tokens to generate")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--task", default="general", help="Task type for auto model selection")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
def run(
    model: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    task: str,
    no_stream: bool,
):
    """Run inference with a prompt."""
    hw = profile_hardware()
    registry = ModelRegistry()

    try:
        spec = registry.resolve(model, hw, task)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    if not registry.is_downloaded(spec):
        console.print(
            f"Model [cyan]{spec.name}[/] not downloaded. Downloading ({spec.size_gb:.1f} GB)..."
        )
        try:
            registry.download(spec)
        except Exception as e:
            console.print(f"[red]Download failed:[/] {e}")
            sys.exit(1)

    model_path = spec.local_path(registry.cache_dir)

    console.print(
        f"[dim]Model: {spec.name} | Threads: {hw.recommended_threads} | Ctx: {hw.recommended_context_length}[/]"
    )

    engine = InferenceEngine(model_path, hw, spec)

    with console.status("[bold cyan]Loading model..."):
        engine.load()

    config = GenerationConfig(
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
@click.option("--plain", is_flag=True, help="Plain-text mode: no panels/colors (pipes, SSH, screen readers)")
def chat(
    model: str | None,
    max_tokens: int,
    temperature: float,
    task: str,
    plain: bool,
):
    """Start an interactive chat session (streaming, slash commands, plain mode available)."""
    hw = profile_hardware()
    registry = ModelRegistry()

    try:
        spec = registry.resolve(model, hw, task)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    if not registry.is_downloaded(spec):
        console.print(
            f"Model [cyan]{spec.name}[/] not downloaded. Downloading ({spec.size_gb:.1f} GB)..."
        )
        try:
            registry.download(spec)
        except Exception as e:
            console.print(f"[red]Download failed:[/] {e}")
            sys.exit(1)

    model_path = spec.local_path(registry.cache_dir)
    engine = InferenceEngine(model_path, hw, spec)

    with console.status("[bold cyan]Loading model..."):
        engine.load()

    from eigencore.tui.chat import ChatUI, ChatUiConfig

    ui = ChatUI(
        engine,
        spec,
        hw,
        policy_label="read-only (policy engine attached in P4)",
        cfg=ChatUiConfig(plain=plain or not sys.stdin.isatty(), max_tokens=max_tokens, temperature=temperature),
    )
    ui.run()

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
    """Measure activation sparsity for a prompt — the first gap exploitation metric."""
    from eigencore.analysis.sparsity import SparsityAnalyzer

    hw = profile_hardware()
    registry = ModelRegistry()

    try:
        spec = registry.resolve(model, hw)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    if not registry.is_downloaded(spec):
        console.print(
            f"Model [cyan]{spec.name}[/] not downloaded. Downloading ({spec.size_gb:.1f} GB)..."
        )
        try:
            registry.download(spec)
        except Exception as e:
            console.print(f"[red]Download failed:[/] {e}")
            sys.exit(1)

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
        console.print(
            f"  Theoretical CPU speedup from sparsity: [green]{report_logits.potential_speedup:.1f}x[/]"
        )

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

    try:
        spec = registry.resolve(model, hw)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    if not registry.is_downloaded(spec):
        console.print(f"[red]Model not downloaded.[/] Run: eigencore download {spec.name}")
        sys.exit(1)

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
