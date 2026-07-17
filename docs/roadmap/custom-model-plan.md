# EigenCore-Coder: Custom Code Assistant Model

## Goal

Fine-tune an open-source code model (7-10B) into a specialized assistant for:
- Code learning across languages (Python, JS/TS, C/C++, Rust, SQL, Java)
- Web development error diagnosis and fixes
- Lab exercises and guided problem-solving
- Build tooling, debugging flows, and DevOps patterns

## Approach

Not training from scratch (infeasible without GPU clusters). Instead:
**QLoRA fine-tuning** on a strong base model, producing a lightweight LoRA adapter (~10-50MB) that specializes the base model for our domain.

### Base Model Candidates

| Model | Params | Why |
|-------|--------|-----|
| Qwen2.5-Coder-7B | 7B | Strong code benchmark scores, multilingual, active community |
| CodeLlama-7B | 7B | Meta's code-specialized Llama, well-tested |
| DeepSeek-Coder-V2-Lite | 2.4B active (16B total MoE) | MoE architecture, very efficient inference |
| StarCoder2-7B | 7B | BigCode project, trained on The Stack v2, broad language coverage |

### Dataset Sources (all public, $0)

1. **StackOverflow data dump** — question + accepted answer pairs, filtered by score > 5
2. **GitHub Issues** — error reports with resolution comments from popular repos (React, Next.js, Django, FastAPI, Express)
3. **MDN Web Docs** — structured error explanations and examples
4. **Common error databases** — compiler/runtime error messages mapped to explanations and fixes
5. **LeetCode/HackerRank editorial patterns** — algorithmic problem-solving approaches
6. **Build tool docs** — webpack, vite, cargo, pip, npm error patterns
7. **Custom lab exercises** — structured prompts with step-by-step solutions

### Dataset Format

```jsonl
{"instruction": "Fix this React error: Cannot update a component while rendering a different component", "input": "<code snippet>", "output": "<explanation + fix>"}
{"instruction": "Write a Python function that implements binary search", "input": "", "output": "<implementation with explanation>"}
```

### Training Strategy

1. **Proof-of-concept (CPU, current hardware)**
   - Base: Qwen2.5-Coder-1.5B (Q4 quantized)
   - LoRA rank: 8, alpha: 16
   - ~1000 curated examples
   - EigenCore's adaptive epoch scaling for convergence detection
   - Expected: 2-4 hours training on i7-10610U

2. **Full training (GPU, when available)**
   - Base: Qwen2.5-Coder-7B (Q4 quantized)
   - LoRA rank: 16, alpha: 32
   - ~10K-50K curated examples
   - Free GPU options: Google Colab (T4), Kaggle (P100), university lab
   - Expected: 4-8 hours on T4 GPU

3. **Evaluation**
   - HumanEval pass@1 (before vs after fine-tuning)
   - Custom error-fix benchmark (50 real-world web dev errors)
   - Qualitative: does it explain clearly, not just solve?

## Packaging

- LoRA adapter published to HuggingFace as `eigencore-coder-7b-lora`
- Downloadable through EigenCore CLI: `eigencore download eigencore-coder`
- Base model + adapter loaded together transparently by the inference engine
- Model card with training details, benchmarks, and limitations

## Timeline

This is a Phase 3-4 deliverable. Prerequisites:
- [ ] Phase 2: inference pipeline battle-tested with existing models
- [ ] Dataset pipeline built and curated
- [ ] GPU access secured (Colab/Kaggle/university)
- [ ] EigenCore training pipeline validated end-to-end on smaller scale

## Why This Matters

Most students use models. Very few fine-tune them. Building the framework AND a specialized model on top of it demonstrates:
- End-to-end ML pipeline understanding
- Data curation and quality judgment
- Training methodology (QLoRA, adaptive scheduling)
- Practical systems engineering (packaging, distribution, inference optimization)
