# D2: Mean-Weights Inference

**Date:** 2026-03-28

**Status:** Approved

**Prerequisite:** D1 complete


---

## Goal

Verified production inference path: deterministic forward pass using posterior mean weights ($\mu$). Confirm quality matches MC-averaged inference.

**Production story:** For serving predictions, use $\mu$ weights — deterministic, no sampling overhead. Reserve MC sampling only for uncertainty estimation (D3).

---

## Existing Infrastructure (already implemented + tested)

| Component | Location | What it does |
|-----------|----------|--------------|
| `BayesianLinear.mean_forward(x)` | `layers.py:87` | `F.linear(x, weight_mu, bias_mu)` |
| `BLoBLoRALinear.mean_forward(x)` | `lora.py:81` | `base_linear(x) + lora_A_mu @ lora_B * scaling` |
| `_use_mean` flag | `layers.py:57`, `lora.py:67` | Checked in `forward()`, routes to `mean_forward()` |
| `use_mean_weights(model)` | `layers.py:212` | Context manager: sets `_use_mean=True` on all stochastic layers |
| `model.generate(use_mean=True)` | `model.py:175` | Deterministic generation via context manager |
| Existing tests | `test_bayesian.py`, `test_lora.py` | Determinism, output preservation verified |

**No new layer-level code needed.** D2 focuses on evaluation integration and verification.

---

## New Behaviors

### B1: MC-averaged perplexity

**Given** a Bayesian model (variational FFN or BLoB LoRA) and evaluation data,
**When** I compute MC-averaged perplexity with N weight samples,
**Then** each of N forward passes uses a fresh weight draw (reparameterization trick), the cross-entropy losses are averaged, and perplexity = $\exp(\text{mean loss})$.

**Details:**
- Function: `compute_perplexity_mc(model, data, block_size, batch_size, device, n_samples, n_batches)` in `evaluate.py`
- For each batch: run N forward passes, each with independently sampled weights, average the loss
- Final perplexity = $\exp\left(\frac{1}{B}\sum_{b=1}^{B}\frac{1}{N}\sum_{n=1}^{N}\mathcal{L}_{b,n}\right)$
- Works with any model that has stochastic layers (BayesianLinear, BLoBLoRALinear)
- Deterministic models: N samples all give same loss (degenerates correctly)

### B2: Mean-weights vs MC-averaged parity

**Given** a trained Bayesian model with small learned $\sigma$ (typical: $\sigma \approx 0.01$–$0.02$),
**When** I compare mean-weights perplexity vs MC-averaged perplexity (N=20),
**Then** they are within 5% relative difference: $\frac{|\text{ppl}_\mu - \text{ppl}_{MC}|}{\text{ppl}_{MC}} < 0.05$

**Why not exact match:** Jensen's inequality — $\mathbb{E}_\theta[\text{softmax}(f_\theta(x))] \neq \text{softmax}(f_{\mathbb{E}[\theta]}(x))$. But with small $\sigma$, the gap is small.

**Note:** This is a property verified on real checkpoints (C1, C3), not a unit-testable invariant. Unit tests verify mechanics; checkpoint tests verify the 5% gate.

### B3: MC variance decreases with N

**Given** a Bayesian model,
**When** I compute MC-averaged perplexity with increasing N (1, 5, 20),
**Then** variance across repeated evaluations decreases (MC converges).

**Unit-testable:** Run `compute_perplexity_mc` twice with same data and N=1 → results differ. With N=large → results converge (difference shrinks).

### B4: Verification script

**Given** C-series checkpoints (C1 variational FFN, C3 BLoB LoRA) and Pile test data,
**When** I run `scripts/verify_mean_weights.py`,
**Then** it prints a comparison table:

```
Method          | Mean-Weights PPL | MC-Avg PPL (N=20) | Relative Diff
----------------|------------------|--------------------|--------------
C1 Var. FFN     |      xx.x        |       xx.x         |    x.x%
C3 BLoB LoRA    |      xx.x        |       xx.x         |    x.x%
```

And exits 0 if all relative differences < 5%, exits 1 otherwise.

---

## Acceptance Criteria

1. `compute_perplexity_mc()` in `evaluate.py` — tested, handles both layer types
2. Mean-weights perplexity = `use_mean_weights` context + existing `compute_perplexity` (no new function needed)
3. Unit tests verify: (a) MC averaging mechanics, (b) deterministic model degeneracy, (c) variance reduction with N
4. `scripts/verify_mean_weights.py` produces comparison table on C1+C3 checkpoints
5. Gate: <5% relative difference on both checkpoints

---

## Out of Scope

- TFB/Laplace MC-averaged perplexity (external sampling via `apply_sampled_params` — different mechanism, D3 territory)
- Production latency benchmarking (D3)
- Quality-cost tradeoff curves (D3)
- New layer-level code (already complete)
