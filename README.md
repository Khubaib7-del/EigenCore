# EigenCore

[![CI](https://github.com/Khubaib7-del/EigenCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Khubaib7-del/EigenCore/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](https://github.com/Khubaib7-del/EigenCore/releases)

**CPU-first LLM intelligence runtime** — hardware-aware inference with dynamic layer skipping, activation sparsity exploitation, and speculative decoding. No GPU required.

EigenCore auto-detects your CPU capabilities, selects the optimal model, and runs inference using every optimization available — from SIMD batch sizing to predictive neuron masking. Built on the thesis that transformer architectures contain unexploited emergent properties that CPUs can leverage better than GPUs.

## Why EigenCore

| | llama.cpp | Ollama | LangChain | vLLM | **EigenCore** |
|---|---|---|---|---|---|
| Hardware auto-config | Manual flags | None | None | GPU only | **CPUID + bandwidth** |
| Multi-model routing | No | Manual | Manual | No | **Automatic** |
| Layer skipping | No | No | No | No | **Dynamic** |
| Sparsity exploitation | No | No | No | No | **Predictive masking** |
| Speculative decoding | No | No | No | GPU only | **CPU-optimized** |
| Context compression | Fixed window | Fixed window | Token count | PagedAttention | **Attention-aware** |
| Fine-tuning | No | No | No | No | **QLoRA on CPU** |

## Quick Start

```bash
pip install eigencore

# See what your hardware can run
eigencore profile

# Download the recommended model
eigencore download tinyllama-1.1b

# Run inference
eigencore run --prompt "Explain binary search in one paragraph"

# Interactive chat with persistent context
eigencore chat

# Measure activation sparsity
eigencore analyze --prompt "What is machine learning?"

# Benchmark Phase 2 optimizations
eigencore benchmark --runs 3 --output results.json
```

## Python API

```python
from eigencore import Forge

forge = Forge()  # auto-detect hardware, select optimal model

# Generate
result = forge.generate("Explain recursion simply")
print(result.text)
print(f"{result.tokens_per_second:.1f} tok/s")

# Stream
for token in forge.stream("Write a haiku about CPUs"):
    print(token, end="")

# Chat with attention-aware context management
forge.chat("What is a transformer?")
forge.chat("How does its attention mechanism work?")  # remembers context

# Route prompts to specialist models
decision = forge.route("Debug this Python function")
print(decision.domain)      # TaskDomain.CODE
print(decision.model.name)  # qwen2.5-coder-1.5b
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Layer 4: Agent Framework                     (Phase 3)  │
│  Tool calling, GBNF grammars, multi-step reasoning       │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Training Engine                                │
│  QLoRA fine-tuning, adaptive epoch scaling (1/4 rule)    │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Inference Runtime + Phase 2 Optimizations      │
│  ┌────────────┐ ┌──────────────┐ ┌───────────────────┐  │
│  │ Layer Skip │ │  Sparse Inf  │ │ Speculative Decode │  │
│  │ Scheduler  │ │  Predictor   │ │ (CPU-optimized)    │  │
│  └────────────┘ └──────────────┘ └───────────────────┘  │
│  Multi-model routing, streaming, context compression     │
├──────────────────────────────────────────────────────────┤
│  Layer 1: Hardware Abstraction Layer                     │
│  CPUID detection (AVX2/512/AMX/NEON), bandwidth est.     │
└──────────────────────────────────────────────────────────┘
```

## Phase 2: CPU Optimization Algorithms

### Dynamic Layer Skipping

Transformer models have high layer redundancy — many middle layers produce near-identical residual stream updates. EigenCore skips them on CPU with minimal quality loss.

```python
from eigencore import LayerSkipScheduler, SkipStrategy

scheduler = LayerSkipScheduler(num_layers=32, strategy=SkipStrategy.STATIC)
plan = scheduler.plan(complexity=0.3)  # low complexity → more skipping

print(plan.summary())
# Skip 7/32 layers (22%) | ~1.3x speedup | ~1.3% quality loss
```

Three strategies: **STATIC** (position heuristic), **ADAPTIVE** (learns from observed sparsity), **CALIBRATED** (pre-computed importance scores). Head and tail layers are always protected.

### Sparsity-Aware Inference

70-90% of neurons in dense transformers output near-zero values. CPUs handle this better than GPUs — branch prediction makes skipping a zero multiply nearly free, while GPU warp divergence forces all 32 threads to execute the same instruction.

```python
from eigencore import SparsityPredictor

predictor = SparsityPredictor(num_layers=22, neurons_per_layer=2048)

# Observe activations during warmup
for activations in warmup_data:
    predictor.observe(layer_idx, activations)

# Generate execution plan
plan = predictor.create_execution_plan(aggressiveness=0.5)
print(plan.summary())
# Sparse plan: 78% sparsity across 22 layers | 3.2x speedup | 78% FLOPs saved
```

### CPU-Optimized Speculative Decoding

Standard speculative decoding uses batch verification on GPU. EigenCore adapts it for CPU — draft length scales with L3 cache size, and acceptance threshold auto-tunes from observed accept rates.

```python
from eigencore import SpeculativeDecoder

decoder = SpeculativeDecoder(adaptive=True)

# Scale draft length based on cache
draft_len = decoder.optimal_draft_length(l3_cache_mb=12, draft_model_mb=8)

# Estimate theoretical speedup
speedup = decoder.estimate_speedup(accept_rate=0.8, draft_cost_ratio=0.1)
print(f"{speedup:.1f}x")  # 3.0x
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `eigencore profile` | Detect hardware, show recommendations |
| `eigencore models` | List models with hardware compatibility |
| `eigencore download <name>` | Download a model from HuggingFace |
| `eigencore run -p "..."` | Run inference with streaming |
| `eigencore chat` | Interactive chat with context memory |
| `eigencore analyze -p "..."` | Measure activation sparsity |
| `eigencore benchmark` | Compare optimization configurations |
| `eigencore train --data ./corpus` | QLoRA fine-tuning on CPU |

## Hardware Profiling

```python
from eigencore import profile_hardware

hw = profile_hardware()
print(hw.summary())
# CPU: Intel Core i7-10610U (amd64)
# Cores: 4P / 8L
# ISA: SSE2, SSE4_1, SSE4_2, AVX, AVX2, FMA, F16C
# RAM: 10.2 / 15.8 GB available
# Bandwidth: ~28.5 GB/s
#
# Max model: ~3.4B params at Q4_K_M
# Est. speed: ~12 tok/s
# Threads: 4
# Context: 4096 tokens
```

## Fine-Tuning on CPU

QLoRA training with adaptive epoch scaling — the "1/4 consistency rule" automatically detects convergence and scales to the next training phase.

```bash
pip install eigencore[train]

eigencore train \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data ./my_dataset.jsonl \
  --output ./my-adapter \
  --lora-rank 8
```

## Research

EigenCore exploits 5 emergent properties of transformer architectures. See [`docs/research/`](docs/research/) for detailed analysis with references:

1. **Layer Redundancy** — middle layers produce near-identical outputs; dynamic skipping saves ~20-30% compute
2. **Attention Sinks** — first tokens are structurally load-bearing; preserving them during compression maintains quality
3. **Activation Sparsity** — 70-90% of neurons output near-zero; CPU branch prediction handles this 3x better than GPU warp divergence
4. **Speculative Decoding on CPU** — draft model in L3 cache makes drafting nearly free; CPU-specific acceptance tuning
5. **Latent Space Steering** — modify model behavior via vector arithmetic in hidden states, zero token overhead

## Development

```bash
git clone https://github.com/Khubaib7-del/EigenCore.git
cd EigenCore
pip install -e ".[dev]"

# Run tests (71 tests)
pytest tests/ -v

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## License

MIT
