# System Architecture Overview

**Date:** 2026-07-17
**Status:** Conceptual — pre-implementation

## Core Thesis

Every existing LLM framework assumes either API access (LangChain, CrewAI) or GPU hardware (vLLM, TensorRT). No framework handles the full lifecycle — model selection → quantization → fine-tuning → inference → agent orchestration — optimized for 16-32GB RAM machines with no GPU.

This framework fills that gap by building a **CPU-first intelligence runtime** that understands its own hardware, exploits emergent transformer properties, and orchestrates multiple models into agent workflows.

## 4-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: Agent Framework                           │
│  Multi-model routing, tool calling, memory,         │
│  task decomposition                                 │
├─────────────────────────────────────────────────────┤
│  Layer 3: Training Engine                           │
│  QLoRA fine-tuning on CPU, dataset streaming,       │
│  adaptive epoch scaling, checkpoint management      │
├─────────────────────────────────────────────────────┤
│  Layer 2: Inference Runtime                         │
│  Model loading, quantized inference, KV-cache       │
│  management, token generation                       │
├─────────────────────────────────────────────────────┤
│  Layer 1: Hardware Abstraction Layer (HAL)          │
│  CPU detection (AVX2/512/AMX/NEON), RAM profiling,  │
│  auto model-sizing, memory bandwidth estimation     │
└─────────────────────────────────────────────────────┘
```

## Layer Details

### Layer 1 — Hardware Abstraction Layer (HAL)

The entry point that makes everything "just work." On startup:

1. Detects CPU instruction sets (CPUID on x86, /proc/cpuinfo on Linux)
2. Measures available RAM (total minus OS/app overhead)
3. Estimates memory bandwidth (quick memcpy benchmark)
4. Outputs: `{ max_model_params, optimal_quantization, max_batch_size, estimated_tok_per_sec }`

**Example output for i7-10610U, 16GB RAM:**
- Max model: ~3B at Q4_K_M (~2GB), or ~7B at Q2_K (~2.5GB, quality tradeoff)
- Memory bandwidth: ~30 GB/s → ceiling of ~15-20 tok/s for a 3B model
- Batch size: 1 (CPU inference is latency-bound, not throughput-bound)

**Implementation language:** C++ with Python bindings. Direct CPUID access and memory benchmarking require low-level system interaction.

**Differentiator:** No existing tool tells a user "given your exact hardware, here's what you can run and how fast."

### Layer 2 — Inference Runtime

Wraps `llama.cpp`'s C API but adds intelligent management:

**Automatic model selection:** User specifies a task ("code generation"). The manager checks the hardware profile, queries a local model registry, and picks the largest model that fits.

**KV-cache pressure management:** On a 16GB machine running a 3B model, KV cache for long contexts eats RAM fast. The manager monitors RSS and when it approaches the ceiling:
- Truncates old context (sliding window)
- Compresses KV cache (attention-aware pruning)
- Warns the user and suggests a smaller context window

**Multi-model routing (application-level MoE):**

```
User prompt → Router (tiny classifier, ~100M params)
                ├── Code task    → CodeLlama 3B Q4
                ├── Chat task    → Phi-3 3.8B Q4
                ├── Math task    → Mathstral 7B Q2
                └── Summary task → Qwen2 1.5B Q4
```

Only one model loaded in memory at a time. Router stays resident (tiny). Model swaps take 2-3 seconds on NVMe for a ~2GB file. Effective intelligence of a much larger model without exceeding RAM ceiling.

### Layer 3 — Training Engine

QLoRA fine-tuning on CPU is feasible with aggressive settings:

- Base model loaded in 4-bit quantization (~2GB RAM for 3B model)
- LoRA rank 8, targeting attention layers only → adapter is ~10MB
- Gradient checkpointing (recompute activations instead of storing them)
- Total RAM: ~6-8GB for a 3B model — fits 16GB with room for OS

**Adaptive epoch scaling (formalized):**

```python
phase = 1
epochs = 100
patience = epochs // 4  # "1/4 consistency" convergence gate

while phase <= 3:
    train(epochs, dataset=phase_data[phase])
    recent_losses = get_losses(last_n=patience)
    variance = std(recent_losses) / mean(recent_losses)  # coefficient of variation

    if variance < 0.02:  # <2% relative fluctuation = converged
        phase += 1
        epochs *= 2.5    # 100 → 250 → 625
        load_next_dataset_tier()
    else:
        epochs += 50     # not converged, keep training
```

User-facing: `framework train --base phi-3-mini --data ./my_corpus/ --task code-review`

### Layer 4 — Agent Framework

The product surface developers interact with:

```python
from framework import Agent, Tool, Router

search = Tool(name="search", fn=web_search)
calculator = Tool(name="calc", fn=eval_math)

agent = Agent(
    model="phi-3-mini-Q4",     # auto-resolved to best fit for hardware
    tools=[search, calculator],
    memory="sqlite",            # persistent conversation memory
    max_ram_gb=8                # hard ceiling
)

response = agent.run("Analyze time complexity of quicksort with 3 examples")
```

**Capabilities:**
- Tool calling via constrained generation (GBNF grammars in llama.cpp)
- Multi-step reasoning with automatic context compression
- Model hot-swapping via Layer 2 router when task domain changes
- Persistent session memory with attention-aware compression

## Build vs. Use

**Don't rewrite (use existing):**
- Inference kernels — `llama.cpp` / `ggml` (2+ years, hundreds of contributors)
- Quantization formats — GGUF is the standard
- LoRA math — HuggingFace PEFT / `unsloth`

**Build (novel contribution):**
- Hardware profiler with auto-configuration
- Dynamic layer skipping per-input
- Activation sparsity exploitation at runtime
- Attention-aware context compression
- Multi-model application-level routing
- Adaptive epoch training pipeline
- Latent space steering API
- Unified CLI/API surface

## Progression

**Framework** = the open-source library. Developers import it and build applications.
**Harness** = an opinionated agent runtime built on the framework (like Claude Code is built on the Claude API). Pre-configured model selections, tool integration, session memory, "just works" experience.
