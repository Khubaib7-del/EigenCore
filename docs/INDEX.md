# CPU-First LLM Framework — Documentation Index

## Architecture
- [System Overview](architecture/system-overview.md) — 4-layer architecture, build-vs-use decisions, framework-to-harness progression
- [Competitive Landscape](architecture/competitive-landscape.md) — positioning against llama.cpp, Ollama, LangChain, vLLM; unique value propositions
- [ML Foundations](architecture/ml-foundations.md) — training loop, architectures (perceptron → transformer), fine-tuning techniques, key metrics

## Research
- [Emergent Gaps](research/emergent-gaps.md) — 5 exploitable gaps in transformer architectures: layer redundancy, attention sinks, activation sparsity, latent arithmetic, speculative decoding
- [Matrix Math Foundations](research/matrix-math-foundations.md) — why row echelon fails for LLMs, SVD/LoRA decomposition, SIMD execution, memory bandwidth analysis, quantization
- [Sparse Activation & Routing](research/sparse-activation-and-routing.md) — intra-model MoE, application-level multi-model routing, runtime sparsity exploitation, chunking concept
- [Adaptive Training](research/adaptive-training.md) — progressive epoch scaling, 1/4 consistency rule, curriculum learning, CPU-specific training constraints
- [Gemini Session Insights](research/gemini-session-insights.md) — original 5 ideas from initial research session, how each maps to formal ML concepts

## Roadmap
- [Phased Plan](roadmap/phased-plan.md) — Phase 1 (HAL + inference), Phase 2 (router + training), Phase 3 (agent framework), Phase 4 (harness); FYP alignment
