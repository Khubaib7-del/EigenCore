# Sparse Activation & Multi-Model Routing

**Date:** 2026-07-17

## Concept Origin

Independently derived concept: instead of passing input through every neuron in a 6B parameter matrix, use a router/gating mechanism to activate only specific neuron chunks sequentially for specific tasks. This reduces effective workload from 6B to ~1.5B per token.

This maps directly to Mixture of Experts (MoE) architecture used in production models (Mixtral, GPT-4).

## Intra-Model Routing (MoE Pattern)

```
[ Input Tokens / Embeddings ]
             │
             ▼
   ┌───────────────────┐
   │   Custom Router    │  ← Small gating network
   └───────────────────┘
      /      |      \
     ▼       ▼       ▼
 [Expert A] [Expert B] [Expert C]  ← Only active experts fire
  (Active)   (Asleep)   (Asleep)
```

**How it works in MoE models:**
- Router is a learned small neural network (not a static hash map)
- Analyzes input tokens and assigns them to top-k experts
- Only selected experts compute; rest stay dormant
- Router is trained end-to-end with the model

**Why a static hash/tree lookup won't work:**
Real routing requires generalization to unseen inputs. A hash map can only match patterns it was explicitly programmed for. The router must learn which expert handles which semantic domain — this is a classification problem, not a lookup problem.

**Key papers:** Switch Transformer (Fedus et al., 2021), Mixtral (Jiang et al., 2024).

## Application-Level Routing (This Framework's Approach)

Instead of modifying internal model architecture, route between multiple complete small models:

```
User prompt → Router (tiny classifier, ~100M params, always resident)
                ├── Code task    → CodeLlama 3B Q4
                ├── Chat task    → Phi-3 3.8B Q4
                ├── Math task    → Mathstral 7B Q2
                └── Summary task → Qwen2 1.5B Q4
```

**Advantages over intra-model MoE:**
- Works with any off-the-shelf model (no architectural modifications needed)
- Each specialist model can be independently fine-tuned for its domain
- Only one model in memory at a time (critical for 16GB RAM constraint)
- Model swap time on NVMe: 2-3 seconds for ~2GB file

**Advantages over single large model:**
- Effective intelligence of a specialist for each task domain
- Total parameter count across all models can exceed RAM — only active model matters
- Each model can use different quantization levels based on task sensitivity

## Runtime Activation Sparsity (Dense Model Exploitation)

Even in non-MoE (dense) models, 70-90% of neurons output near-zero values for any given input. This emergent sparsity can be exploited at inference time:

```
Standard:  output = W × input    (multiply ALL weight rows)

Sparse:
1. Compute activations for current layer
2. Build bitmask: neuron_active[i] = (|activation[i]| > threshold)
3. For next layer: only multiply rows of W where bitmask is 1
4. Skip ~70% of multiplications
```

**Why this helps CPU but not GPU:**
- GPUs are optimized for regular, dense computation. Irregular sparsity (different neurons active per input) causes thread divergence and warp inefficiency.
- CPUs handle branch prediction and conditional execution naturally. Skipping a multiplication on CPU is a genuine compute saving.

**This makes CPU inference potentially FASTER per-FLOP than GPU for sparse workloads.** Not faster overall (GPUs have vastly more raw throughput), but the efficiency gap narrows significantly.

## The Chunking Concept

Original idea: chunk embeddings into a hybrid data structure, pass chunks sequentially to the LLM, engineer specific neurons to activate for specific tasks.

**Formalized version:**
- Save model weights not as one giant matrix but sliced into semantic clusters (blocks)
- When input arrives, wrapper analyzes initial tokens
- Wrapper tells CPU: "Only load memory blocks 12, 45, 89 into cache/registers. Leave rest in RAM."
- Reduces active memory footprint and cache pressure

This is conceptually similar to how PowerInfer (Du et al., 2024) handles GPU-CPU hybrid inference by predicting which neurons will activate and pre-loading only those.
