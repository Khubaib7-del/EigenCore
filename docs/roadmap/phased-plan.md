# Development Roadmap

**Date:** 2026-07-17

## Phase 1: Hardware Profiler + Inference Manager (4-6 weeks)

**Goal:** CLI tool that auto-detects hardware, downloads an optimal model, and runs inference.

**Deliverables:**
- Hardware profiler (C++ with Python bindings)
  - CPUID instruction set detection (AVX2, AVX-512, AMX, NEON)
  - RAM measurement and availability estimation
  - Memory bandwidth benchmark (quick memcpy test)
  - Output: JSON profile with max_model_params, optimal_quant, estimated_tok_per_sec
- Inference manager wrapping llama.cpp
  - Model registry (local JSON mapping task types to model recommendations)
  - Auto-download from HuggingFace based on hardware profile
  - Quantization selection based on available RAM
- CLI interface: `framework run --model phi3 --prompt "hello"`
- First emergent gap implementation: activation sparsity measurement (instrument ggml tensors, count near-zero activations, log sparsity ratios per layer)

**Success criteria:** User runs one command on a fresh machine, framework profiles hardware, downloads appropriate model, runs inference at near-optimal speed without any manual configuration.

**Languages:** C++ for HAL, Python for CLI and orchestration, llama-cpp-python for inference binding.

---

## Phase 2: Multi-Model Router + Fine-Tuning Pipeline (4-6 weeks)

**Goal:** Automatic task routing between specialist models and QLoRA fine-tuning from CLI.

**Deliverables:**
- Task router
  - Tiny classifier model (~100M params) trained to categorize prompts into task domains
  - Model swap manager (unload current, load next, handle KV cache reset)
  - Latency-aware: only swap if task domain actually changed
- QLoRA training pipeline
  - 4-bit base model loading with LoRA adapter initialization
  - Gradient checkpointing for RAM efficiency
  - Adaptive epoch scaling with consistency gating (the 1/4 rule)
  - CLI: `framework train --base phi-3-mini --data ./corpus/ --task code-review`
  - Checkpoint management and adapter export
- Second emergent gap implementation: dynamic layer skipping (cosine similarity between consecutive layer outputs, skip threshold tuning)

**Success criteria:** User can fine-tune a 3B model on their own data using ≤8GB RAM, and the router correctly identifies task domains and loads appropriate specialist models.

---

## Phase 3: Agent Framework (after Phase 2 stabilizes)

**Goal:** Full agent runtime with tool calling, memory, and multi-step execution.

**Deliverables:**
- Agent API
  - Tool registration and execution
  - Constrained generation via GBNF grammars for reliable tool-call JSON output
  - Multi-step reasoning loop with automatic re-planning
- Context management
  - Attention-aware compression (preserve sink tokens, compress middle, keep recent)
  - SQLite-backed persistent session memory
  - Cross-session context retrieval
- Latent steering API
  - Steering vector extraction (contrastive activation pairs)
  - Runtime injection at configurable layer depth
  - User-facing controls for behavior modification without prompt overhead
- Speculative decoding for CPU
  - Tiny draft model (60M) kept in L3 cache
  - Batch verification by main model
  - Acceptance rate monitoring and draft model adaptation

**Success criteria:** A developer can build an agent application using local models that handles multi-turn conversations, calls tools, maintains memory across sessions, and runs at practical speeds on a 16GB RAM machine.

---

## Phase 4: Harness (product layer)

**Goal:** Opinionated agent runtime built on the framework — the "Claude Code" to the framework's "Claude API."

**Deliverables:**
- CLI/TUI interface for direct interaction
- Pre-configured model selections per hardware tier
- Built-in tool integrations (file system, web search, code execution)
- Session management with automatic context optimization
- "Just works" experience — user installs, runs, and has a functioning local AI assistant

---

## FYP Alignment

Phases 1 + 2 constitute a strong Final Year Project at FAST Lahore:
- **Research contribution:** Hardware-aware auto-configuration with benchmarks showing near-optimal performance vs. manual tuning
- **Engineering contribution:** Full-lifecycle CPU-first LLM framework
- **Measurable results:** Speedup benchmarks, accuracy retention under quantization/sparsity, fine-tuning convergence analysis
- **Publishable:** Activation sparsity exploitation on CPU as a first-class optimization strategy
