"""
EigenCore CLI — the user-facing interface.

Usage:
    eigencore profile              # show hardware profile + recommendations
    eigencore models               # list available models
    eigencore run --prompt "..."   # run inference (auto-selects model if not specified)
    eigencore chat                 # interactive chat session
    eigencore download <model>     # pre-download a model
    eigencore analyze --prompt "." # measure activation sparsity on a prompt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eigencore.hal.profiler import profile_hardware
from eigencore.models.registry import ModelRegistry
from eigencore.engine.inference import InferenceEngine, GenerationConfig

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="eigencore")
def cli():
    """EigenCore — CPU-first LLM intelligence runtime."""
    pass


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

    if not registry.is_downloaded(rec):
        console.print(f"\n  [dim]Download with:[/] eigencore download {rec.name}")
    else:
        console.print('\n  [dim]Already downloaded. Run with:[/] eigencore run --prompt "hello"')


@cli.command()
@click.option("--task", default=None, help="Filter by task type (general, code, math)")
def models(task: Optional[str]):
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
        fits = m.size_gb <= (hw.available_ram_gb - 2.0)
        downloaded = registry.is_downloaded(m)
        table.add_row(
            m.name,
            f"{m.params_b:.1f}B",
            m.quant,
            f"{m.size_gb:.1f}GB",
            m.task,
            "[green]yes[/]" if fits else "[red]no[/]",
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
    model: Optional[str],
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
def chat(
    model: Optional[str],
    max_tokens: int,
    temperature: float,
    task: str,
):
    """Start an interactive chat session."""
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

    console.print(
        Panel(
            f"Model: [cyan]{spec.name}[/] ({spec.params_b:.1f}B)\n"
            f"Context: {hw.recommended_context_length} tokens\n"
            f"Type [bold]exit[/] or [bold]quit[/] to end.",
            title="[bold]EigenCore Chat",
            border_style="cyan",
        )
    )

    messages: list[dict[str, str]] = []
    config = GenerationConfig(max_tokens=max_tokens, temperature=temperature, stream=False)

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            break

        messages.append({"role": "user", "content": user_input})

        with console.status("[dim]Thinking..."):
            result = engine.chat(messages, config)

        messages.append({"role": "assistant", "content": result.text})

        console.print(f"\n[bold cyan]EigenCore:[/] {result.text}")
        console.print(
            f"[dim]{result.tokens_generated} tokens | {result.tokens_per_second:.1f} tok/s[/]"
        )

    engine.unload()
    console.print("\n[dim]Session ended.[/]")


@cli.command()
@click.option("--model", "-m", default=None, help="Model name (auto-selects if omitted)")
@click.option("--prompt", "-p", required=True, help="Prompt to analyze sparsity on")
@click.option("--threshold", default=0.01, help="Near-zero threshold")
@click.option("--output", "-o", default=None, help="Save report to JSON file")
def analyze(
    model: Optional[str],
    prompt: str,
    threshold: float,
    output: Optional[str],
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
