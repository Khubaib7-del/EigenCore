# ML Foundations & Training Theory

**Date:** 2026-07-17
**Purpose:** Reference document mapping foundational ML concepts to their role in the framework.

## The Training Loop

```
Forward Pass → Activation Function → Loss Computation → Backpropagation → Weight Update → Iterate
```

### Forward Pass
Input data flows through network layers. Each layer computes: `z = W·x + b` (linear transform) followed by activation function.

### Activation Functions
- **ReLU:** `f(x) = max(0, x)` — introduces non-linearity, causes the natural activation sparsity this framework exploits (outputs are exactly 0 for negative inputs)
- **Softmax:** `f(x_i) = e^{x_i} / Σe^{x_j}` — converts logits to probability distribution over vocabulary for next-token prediction
- **GELU / SiLU:** Smooth alternatives to ReLU used in modern transformers (GPT, Llama, Phi)

### Loss Computation
Cross-entropy loss for language models: measures how far the model's predicted probability distribution is from the true next token.

### Backpropagation
Compute gradients ∂L/∂w for every weight using the chain rule, propagating error backwards through layers. This is the expensive step — for a 3B model, you need to compute and store 3 billion gradient values.

### Weight Update (Gradient Descent)
```
w_new = w_old - η · ∂L/∂w
```
Where η (eta) is the learning rate. Too high → overshooting, too low → never converges.

Modern optimizers (Adam, AdamW) maintain per-parameter momentum and variance estimates, which is why optimizer states consume 2-3x the model size in memory.

## Network Architectures

### Single Perceptron
One neuron: `output = activation(w·x + b)`. Binary classification.

### Multi-Layer Perceptron (ANN)
Multiple layers of perceptrons with non-linear activations between them. Universal function approximator.

### Convolutional Neural Network (CNN)
Sliding filter kernels over spatial data (images). Weight sharing reduces parameters. Pooling layers reduce dimensionality. Used for classification, object detection.

### Transformer (LLM Architecture)
Self-attention mechanism: every token attends to every other token. Enables parallel processing and long-range dependencies. This is what the framework runs inference on.

Key transformer components:
- **Token embeddings:** Map discrete tokens to continuous vectors
- **Positional encoding:** Inject position information (RoPE in modern models)
- **Multi-head self-attention:** Q·K^T/√d_k → softmax → ·V (the core computation)
- **Feed-forward network:** Two linear layers with activation between them (this is where activation sparsity lives)
- **Layer normalization:** Stabilize training by normalizing hidden states

## Fine-Tuning Techniques

### Full Fine-Tuning
Update all model weights. Requires storing full gradient history and optimizer states.
Memory: 4-6x model size. Not feasible on CPU for models > 1B.

### LoRA (Low-Rank Adaptation)
Freeze base weights W. Train small adapter matrices A (d×r) and B (r×d).
At inference: W' = W + A·B where ΔW = A·B is the learned adaptation.
Memory: base model (frozen, 4-bit) + adapter (~10MB). Feasible on CPU.

### QLoRA
LoRA but with the base model quantized to 4-bit. Further reduces memory.
A 3B model: ~2GB base (Q4) + ~10MB adapter + ~4GB optimizer states = ~6GB total.
Fits in 16GB RAM with room for OS.

## Key Metrics

### Perplexity
`PP = e^{cross_entropy_loss}` — how "surprised" the model is by the next token. Lower is better. A perplexity of 1 means perfect prediction.

### Loss Curve
Plot of loss over training steps/epochs. Healthy training shows monotonic decrease with diminishing returns. The "consistency gating" approach monitors the variance of this curve over trailing windows.

### Convergence
When the loss curve flattens (gradient norms approach zero). The model has found a local minimum in the loss landscape.

### Overfitting
When training loss decreases but validation loss increases — the model memorized training data instead of learning generalizable patterns. Detected by comparing train vs. validation loss curves.

## Classification & Clustering

### Classification
Supervised learning: given labeled examples, predict category of new inputs. Used for the framework's task router (classify prompts into task domains).

### Clustering
Unsupervised learning: group similar data points without labels. Relevant for dataset organization in the training pipeline.

### Regression
Predict continuous values instead of categories. Used for the hardware profiler's performance estimation (predict tok/s given hardware specs).
