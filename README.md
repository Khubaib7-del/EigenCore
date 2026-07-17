# EigenCore

**CPU-first LLM intelligence runtime** — hardware-aware inference, training, and agent orchestration for machines without GPUs.

EigenCore auto-detects your hardware, selects the optimal model, and runs inference using every CPU optimization available. No manual configuration, no GPU required.

## What Makes This Different

| | llama.cpp | Ollama | LangChain | **EigenCore** |
|---|---|---|---|---|
| Hardware awareness | Manual flags | None | None | **Auto-profiles** |
| Multi-model routing | No | Manual | Manual | **Automatic** |
| Context compression | Fixed window | Fixed window | Token count | **Attention-aware** |
| Fine-tuning | No | No | No | **Built-in QLoRA** |
| Activation analysis | No | No | No | **Sparsity measurement** |

## Quick Start

```bash
pip install eigencore

# See what your hardware can run
eigencore profile

# Download the recommended model
eigencore download tinyllama-1.1b

# Run inference (auto-configures threads, context, batch size)
eigencore run --prompt "Explain binary search in one paragraph"

# Interactive chat with persistent context
eigencore chat

# Measure activation sparsity (first gap exploitation metric)
eigencore analyze --prompt "What is machine learning?"
```

## Python API

```python
from eigencore import Forge

# Auto-detect hardware, select optimal model
forge = Forge()

# Generate
result = forge.generate("Explain recursion simply")
print(result.text)
print(f"{result.tokens_per_second:.1f} tok/s")

# Stream
for token in forge.stream("Write a haiku about CPUs"):
    print(token, end="")

# Chat with attention-aware context management
response = forge.chat("What is a transformer?")
response = forge.chat("How does its attention mechanism work?")  # remembers context

# Route prompts to specialist models automatically
decision = forge.route("Debug this Python function")
print(decision.domain)  # TaskDomain.CODE
print(decision.model.name)  # qwen2.5-coder-1.5b
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Layer 4: Agent Framework (Phase 3)              │
│  Tool calling, multi-step reasoning, memory      │
├──────────────────────────────────────────────────┤
│  Layer 3: Training Engine                        │
│  QLoRA fine-tuning, adaptive epoch scaling       │
├──────────────────────────────────────────────────┤
│  Layer 2: Inference Runtime                      │
│  Auto-config, multi-model routing, streaming     │
├──────────────────────────────────────────────────┤
│  Layer 1: Hardware Abstraction Layer             │
│  CPUID detection, RAM profiling, bandwidth est.  │
└──────────────────────────────────────────────────┘
```

### Layer 1: Hardware Profiler

Detects CPU instruction sets (AVX2, AVX-512, AMX, NEON), measures RAM and memory bandwidth, and computes optimal model parameters:

```python
from eigencore import profile_hardware

hw = profile_hardware()
print(hw.summary())
# CPU: Intel Core i7-10610U (amd64)
# ISA: SSE2, SSE4_1, SSE4_2, AVX, AVX2, FMA, F16C
# RAM: 10.2 / 15.8 GB available
# Max model: ~3.4B params at Q4_K_M
# Est. speed: ~12 tok/s
```

### Layer 2: Multi-Model Router

Classifies prompts into task domains and selects specialist models automatically:

```python
from eigencore import TaskRouter, profile_hardware

router = TaskRouter(profile_hardware())

# Routes to code model
decision = router.route("Write a binary search in Python")
print(decision.domain)   # CODE
print(decision.model.name)  # qwen2.5-coder-1.5b

# Routes to general model
decision = router.route("Tell me about history")
print(decision.domain)   # GENERAL
```

### Layer 3: Training with Adaptive Epochs

Fine-tune models using QLoRA with the "1/4 consistency rule" — training automatically scales through phases based on convergence:

```bash
eigencore train \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data ./my_dataset.jsonl \
  --output ./my-adapter \
  --lora-rank 8
```

### Context Management

Attention-aware compression preserves the first tokens (attention sinks) and recent context while compressing middle messages:

```python
from eigencore import ContextManager

ctx = ContextManager(max_context_tokens=2048)
ctx.add("system", "You are a helpful assistant.")
ctx.add("user", "Long conversation...")
# ... many messages ...

window = ctx.get_context()
# First messages preserved (attention sinks)
# Middle messages compressed
# Recent messages intact
# Total tokens within budget
```

## Fine-Tuning on CPU

Requires additional dependencies:

```bash
pip install eigencore[train]
```

```python
from eigencore.training import CPUTrainer, TrainingConfig
from eigencore import profile_hardware

config = TrainingConfig(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset_path="./data.jsonl",
    output_dir="./adapter",
    lora_rank=8,
    initial_epochs=100,
)

trainer = CPUTrainer(config, profile_hardware())
result = trainer.train()
print(result.summary())
```

## Research: Emergent Gap Exploitation

EigenCore is built on the thesis that transformer architectures contain unexploited emergent properties. See `docs/research/` for detailed analysis:

1. **Layer Redundancy** — Middle layers produce near-identical outputs; dynamic skipping saves 40% compute
2. **Attention Sinks** — First tokens are load-bearing for attention; preserving them during compression maintains quality
3. **Activation Sparsity** — 70-90% of neurons output near-zero; skipping them is a 3x CPU speedup (CPUs handle irregular sparsity better than GPUs)
4. **Latent Steering** — Modify model behavior via vector arithmetic in hidden states, zero token overhead
5. **Speculative Decoding** — Tiny draft model in L3 cache drafts tokens; large model verifies in batch

## Development

```bash
git clone https://github.com/Khubaib7-del/eigencore.git
cd eigencore
pip install -e ".[dev]"

# Run tests
python tests/test_profiler.py
python tests/test_router.py
python tests/test_context.py
python tests/test_convergence.py
```

## License

MIT
