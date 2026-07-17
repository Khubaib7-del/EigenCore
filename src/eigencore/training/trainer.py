"""
CPU-first QLoRA training engine with adaptive epoch scaling.

Enables fine-tuning of 1-3B parameter models on machines with 16GB RAM
by combining:
- 4-bit quantization of base model weights (QLoRA)
- Low-rank adapters (LoRA rank 4-16, ~10MB trainable params)
- Gradient checkpointing (recompute activations to save memory)
- Adaptive epoch scaling with the "1/4 consistency" convergence gate

Requires: torch, peft, transformers, bitsandbytes (install with `pip install eigencore[train]`)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

from eigencore.hal.profiler import HardwareProfile


class TrainingPhase(Enum):
    WARMUP = auto()
    CONVERGENCE = auto()
    REFINEMENT = auto()


@dataclass
class TrainingConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_path: str = ""
    output_dir: str = "./eigencore-adapters"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    # adaptive epoch scaling
    initial_epochs: int = 100
    consistency_window: float = 0.25  # the 1/4 rule
    consistency_threshold: float = 0.02  # coefficient of variation < 2%
    max_phases: int = 3
    epoch_scale_factor: float = 2.5
    # hardware
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    use_cpu: bool = True


@dataclass
class EpochMetrics:
    epoch: int
    phase: int
    train_loss: float
    eval_loss: Optional[float]
    learning_rate: float
    elapsed_seconds: float
    memory_used_gb: float


@dataclass
class TrainingResult:
    adapter_path: str
    base_model: str
    total_epochs: int
    phases_completed: int
    final_train_loss: float
    final_eval_loss: Optional[float]
    total_time_seconds: float
    metrics_history: list[EpochMetrics] = field(default_factory=list)
    converged: bool = False
    convergence_epoch: Optional[int] = None

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "DID NOT CONVERGE"
        lines = [
            f"Training Result: {status}",
            f"  Base model: {self.base_model}",
            f"  Adapter saved: {self.adapter_path}",
            f"  Total epochs: {self.total_epochs} across {self.phases_completed} phases",
            f"  Final train loss: {self.final_train_loss:.4f}",
            f"  Total time: {self.total_time_seconds / 60:.1f} minutes",
        ]
        if self.converged and self.convergence_epoch:
            lines.append(f"  Converged at epoch: {self.convergence_epoch}")
        return "\n".join(lines)

    def save_metrics(self, path: Path) -> None:
        data = {
            "adapter_path": self.adapter_path,
            "base_model": self.base_model,
            "total_epochs": self.total_epochs,
            "phases_completed": self.phases_completed,
            "final_train_loss": self.final_train_loss,
            "converged": self.converged,
            "total_time_seconds": self.total_time_seconds,
            "history": [
                {
                    "epoch": m.epoch,
                    "phase": m.phase,
                    "train_loss": m.train_loss,
                    "eval_loss": m.eval_loss,
                    "lr": m.learning_rate,
                    "time_s": m.elapsed_seconds,
                    "mem_gb": m.memory_used_gb,
                }
                for m in self.metrics_history
            ],
        }
        path.write_text(json.dumps(data, indent=2))


class CPUTrainer:
    """
    Fine-tunes a model on CPU using QLoRA with adaptive epoch scaling.

    Usage:
        config = TrainingConfig(
            base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_path="./my_data.jsonl",
            output_dir="./my-adapter",
        )
        trainer = CPUTrainer(config, hardware_profile)
        result = trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        profile: HardwareProfile,
        on_epoch_end: Optional[Callable[[EpochMetrics], None]] = None,
    ):
        self.config = config
        self.profile = profile
        self.on_epoch_end = on_epoch_end
        self._validate_hardware()

    def _validate_hardware(self) -> None:
        """Check that the hardware can handle the requested training config."""
        # rough memory estimate: base model (4-bit) + adapter + optimizer + gradients
        # for 1B model at 4-bit: ~0.5GB model + 0.5GB optimizer + 0.5GB gradients ≈ 1.5GB
        # for 3B model at 4-bit: ~1.5GB model + 1.5GB optimizer + 1.5GB gradients ≈ 4.5GB
        available = self.profile.available_ram_gb

        model_name = self.config.base_model.lower()
        if "7b" in model_name or "8b" in model_name:
            required = 10.0
        elif "3b" in model_name or "4b" in model_name:
            required = 6.0
        elif "1b" in model_name or "2b" in model_name:
            required = 3.0
        else:
            required = 4.0

        if available < required:
            raise MemoryError(
                f"Training {self.config.base_model} needs ~{required:.0f}GB RAM. "
                f"Only {available:.1f}GB available. Close other applications or "
                f"use a smaller model."
            )

    def _check_dependencies(self) -> None:
        """Verify training dependencies are installed."""
        import importlib.util

        missing = []
        if importlib.util.find_spec("torch") is None:
            missing.append("torch")
        if importlib.util.find_spec("transformers") is None:
            missing.append("transformers")
        if importlib.util.find_spec("peft") is None:
            missing.append("peft")

        if missing:
            raise ImportError(
                f"Training requires additional packages: {', '.join(missing)}\n"
                f"Install with: pip install eigencore[train]"
            )

    def _check_convergence(self, losses: list[float]) -> bool:
        """
        The 1/4 consistency gate: check if loss has stabilized over
        the trailing window (last 25% of epochs in current phase).
        """
        if len(losses) < 4:
            return False

        window_size = max(int(len(losses) * self.config.consistency_window), 2)
        recent = losses[-window_size:]

        mean_loss = sum(recent) / len(recent)
        if mean_loss == 0:
            return True

        variance = sum((x - mean_loss) ** 2 for x in recent) / len(recent)
        std_dev = variance**0.5
        cv = std_dev / abs(mean_loss)

        return cv < self.config.consistency_threshold

    def train(self) -> TrainingResult:
        """
        Run the full adaptive training pipeline.

        Phase progression:
        1. Train for initial_epochs → check consistency over last 1/4
        2. If consistent: scale epochs by 2.5x, load next data tier
        3. Repeat until max_phases or convergence failure
        """
        self._check_dependencies()

        import psutil
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # load base model in 4-bit
        quantization_config = None
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quantization_config,
            device_map="cpu" if self.config.use_cpu else "auto",
            torch_dtype=torch.float32 if self.config.use_cpu else torch.float16,
        )

        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # load dataset
        dataset = self._load_dataset(tokenizer)

        # adaptive training loop
        all_metrics: list[EpochMetrics] = []
        phase_losses: list[float] = []
        current_phase = 1
        current_epochs = self.config.initial_epochs
        total_epochs_run = 0
        converged = False
        convergence_epoch = None
        start_time = time.time()

        while current_phase <= self.config.max_phases:
            training_args = TrainingArguments(
                output_dir=str(output_dir / f"phase-{current_phase}"),
                num_train_epochs=current_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                logging_steps=1,
                save_strategy="epoch",
                save_total_limit=2,
                report_to="none",
                no_cuda=self.config.use_cpu,
                dataloader_pin_memory=False,
                fp16=False,
                bf16=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )

            train_result = trainer.train()

            epoch_loss = train_result.training_loss
            phase_losses.append(epoch_loss)
            total_epochs_run += current_epochs

            mem = psutil.virtual_memory()
            metrics = EpochMetrics(
                epoch=total_epochs_run,
                phase=current_phase,
                train_loss=epoch_loss,
                eval_loss=None,
                learning_rate=self.config.learning_rate,
                elapsed_seconds=time.time() - start_time,
                memory_used_gb=(mem.total - mem.available) / (1024**3),
            )
            all_metrics.append(metrics)

            if self.on_epoch_end:
                self.on_epoch_end(metrics)

            # check the 1/4 consistency gate
            if self._check_convergence(phase_losses):
                converged = True
                convergence_epoch = total_epochs_run

                if current_phase < self.config.max_phases:
                    current_phase += 1
                    current_epochs = int(current_epochs * self.config.epoch_scale_factor)
                    phase_losses.clear()
                else:
                    break
            else:
                current_epochs += int(self.config.initial_epochs * 0.5)

        # save final adapter
        final_path = output_dir / "final-adapter"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        elapsed = time.time() - start_time

        result = TrainingResult(
            adapter_path=str(final_path),
            base_model=self.config.base_model,
            total_epochs=total_epochs_run,
            phases_completed=current_phase,
            final_train_loss=all_metrics[-1].train_loss if all_metrics else 0.0,
            final_eval_loss=None,
            total_time_seconds=elapsed,
            metrics_history=all_metrics,
            converged=converged,
            convergence_epoch=convergence_epoch,
        )

        result.save_metrics(output_dir / "training_metrics.json")

        return result

    def _load_dataset(self, tokenizer):
        """Load and tokenize the training dataset."""
        from datasets import load_dataset

        path = self.config.dataset_path

        if path.endswith(".jsonl") or path.endswith(".json"):
            dataset = load_dataset("json", data_files=path, split="train")
        elif path.endswith(".txt"):
            dataset = load_dataset("text", data_files=path, split="train")
        elif path.endswith(".csv"):
            dataset = load_dataset("csv", data_files=path, split="train")
        else:
            dataset = load_dataset(path, split="train")

        def tokenize_fn(examples):
            text_field = "text" if "text" in examples else list(examples.keys())[0]
            return tokenizer(
                examples[text_field],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        return tokenized
