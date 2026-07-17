# Adaptive Epoch Scaling & Training Methodology

**Date:** 2026-07-17

## Core Concept

A multi-phased training approach where the number of epochs scales dynamically based on measured convergence stability. Instead of setting a fixed epoch count, the system monitors loss variance and automatically progresses through training phases.

## The "1/4 Consistency" Rule

**Principle:** If the model's loss remains stable (low variance) for the final 1/4 to 1/3 of the current training phase, the model has converged on its current data tier and is ready for more complex data.

```
[ Phase 1: 100 Epochs ] ──► Check Loss Variance ──► If Stable for 25+ epochs
                                                            │
                                                            ▼
[ Phase 2: 250 Epochs ] ──► Check Loss Variance ──► [ Phase 3: 1000+ Epochs ]
```

## Formal Implementation

```python
phase = 1
epochs = 100
patience = epochs // 4  # the 1/4 consistency window

while phase <= 3:
    train(epochs, dataset=phase_data[phase])

    # Measure stability: coefficient of variation over trailing window
    recent_losses = get_losses(last_n=patience)
    cv = std(recent_losses) / mean(recent_losses)

    if cv < 0.02:  # <2% relative fluctuation = converged
        phase += 1
        epochs = int(epochs * 2.5)   # scale: 100 → 250 → 625
        load_next_dataset_tier()
    else:
        epochs += 50  # not converged — extend current phase
```

## Connection to Established Techniques

**Curriculum Learning:** Training on progressively harder/more complex data. Phase 1 uses clean, filtered data. Phase 2 introduces complexity. Phase 3 uses full unfiltered corpus. This prevents the model from being overwhelmed early.

**Early Stopping with Patience:** The standard technique monitors validation loss and stops when it hasn't improved for N epochs. The framework extends this concept by using stability as a gate for phase transitions rather than just a stop signal.

**Gradient Checkpointing:** Recompute activations during backward pass instead of storing them. Trades compute for memory — critical for CPU training where RAM is the constraint. A 3B model training with gradient checkpointing fits in ~6-8GB instead of ~20GB+.

## What Makes a "Well-Designed LLM"

From the original discussion: if a model maintains loss consistency across 1/4 to 1/3 of epochs at increasingly larger scales, it has:

1. **Generalized** the underlying concepts (not just memorized training data)
2. **Avoided catastrophic forgetting** (hasn't lost Phase 1 knowledge while learning Phase 2)
3. **Stabilized its weight space** (gradients are small and consistent, not oscillating)

This is measurable via:
- **Perplexity** on held-out validation set (lower = better language modeling)
- **Loss variance** across epoch windows (lower = more stable)
- **Cross-phase performance** — test Phase 1 tasks after Phase 2 training (should not degrade)

## CPU-Specific Training Considerations

**Feasible on 16GB RAM:**
- 1-3B models with QLoRA (4-bit base + rank-8 adapter)
- Total memory: ~6-8GB including optimizer states
- Training speed: slow but functional (~hours per epoch on small datasets)

**Not feasible on 16GB RAM:**
- Full fine-tuning of any model > 1B (optimizer states alone need 4-6x model size)
- Training from scratch (requires storing full gradient history)

**The wrapper's role in training:**
- Monitor gradient norms per epoch (detect instability early)
- Automatically trigger checkpointing when convergence is detected
- Manage dataset loading in chunks to avoid memory spikes
- Log all metrics for the consistency analysis
