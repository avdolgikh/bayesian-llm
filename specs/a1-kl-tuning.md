# A1 KL Weight Tuning Spec

**Date:** 2026-02-25
**Goal:** Fix near-zero MI separation between ID and OOD by increasing KL pressure on the Bayesian output head.

## Problem: Posterior Collapse

The first A1 run (MLflow `686e3201ae68444c9a9dade3717f016c`, 5K steps, `kl_weight=0.01`) produced **no meaningful MI separation**:

| Metric | ID | OOD | Ratio | Target |
|--------|---:|----:|------:|--------|
| MI | 0.0236 | 0.0253 | **1.07x** | >>1x |
| Flip rate | 12.7% | 13.0% | 1.02x | >>1x |
| Pred entropy | 3.71 | 4.39 | 1.18x | — |
| Perplexity | 65.62 | 630.46 | 9.6x | — |

The model sees a strong perplexity gap (9.6x) — it *knows* OOD from ID at the representation level. But the Bayesian head produces nearly identical weight samples every time, so MC sampling can't surface that knowledge as uncertainty.

### Root cause: KL is negligible

| Quantity | Value |
|----------|-------|
| Raw KL (summed, 25.7M Bayesian params) | 52.4M nats |
| Per-param KL | ~2.0 nats |
| `kl_weight` | 0.01 |
| `num_train_tokens` | 2.7M |
| Effective KL per token: `kl_weight * KL / N` | **0.19 nats** |
| CE loss | **4.18 nats** |
| KL as fraction of ELBO | **4.5%** |

At 4.5% of the ELBO, the model ignores the KL term entirely. The posterior σ collapses from `softplus(-5)` ≈ 0.007 at init to even smaller values during training. With near-deterministic weights, 30 MC samples produce nearly identical outputs → MI ≈ 0.

### Comparison to A0 baseline

| Metric | A0 (det.) | A1 (kl=0.01) | Note |
|--------|----------:|-------------:|------|
| test_id_ppl | 49.11 | 65.62 | A1 33% worse (still converging at step 5K) |
| test_ood_ppl | 540.28 | 630.46 | Similar ballpark |
| OOD/ID ratio | 11.0x | 9.6x | Similar |
| MI ratio | N/A | 1.07x | No signal |

A1 has not yet converged (val loss still decreasing at step 5000), so the perplexity gap to A0 would narrow with more steps. But more steps alone won't fix MI — that requires stronger KL pressure.

## Fix: Increase kl_weight to 0.1

### Rationale

With `kl_weight=0.1`, the effective KL contribution becomes:

```
0.1 * 52.4M / 2.7M ≈ 1.94 nats  (32% of ELBO, up from 4.5%)
```

This is real pressure. The model must maintain meaningful posterior variance (wider σ) to satisfy the prior, which produces:
- **More diverse weight samples** across MC passes
- **Higher MI on OOD inputs** where the model is uncertain about what weights are appropriate
- **Lower MI on ID inputs** where training data has constrained the posterior

The KL would likely settle at a different (lower) equilibrium — the model widens σ toward prior_std=1.0, reducing per-param KL, but the higher weight keeps the total contribution significant.

### Expected tradeoffs

| Effect | Direction | Why |
|--------|-----------|-----|
| MI separation (OOD/ID) | **Improve** (goal) | Wider posteriors → more weight variance → more disagreement on OOD |
| test_id_perplexity | Slightly worse | Model can't collapse to point estimate; pays a small CE cost |
| test_ood_perplexity | Similar or better | OOD is already high; wider posteriors don't hurt much |
| Training stability | Same (with annealing) | KL annealing ramps pressure gradually |

### KL annealing remains important

Keep `kl_annealing_steps=2000`. The annealing schedule:
- Steps 1–2000: `effective_kl_weight` ramps linearly from 0 → 0.1
- Steps 2000+: `effective_kl_weight` = 0.1 (full)

This lets the model fit the data (learn good μ) before KL pressure forces it to maintain wide σ. Without annealing, kl_weight=0.1 from step 1 could destabilize early training — the model hasn't learned anything yet but is already penalized for being far from the prior.

## Config Change

```yaml
# configs/a1_agnews.yaml
model:
  bayes:
    kl_weight: 0.1    # was 0.01 — posterior collapsed with too-cold KL
```

All other settings unchanged. If kl_weight=0.1 overshoots (MI is high but perplexity degrades badly), try 0.05 as a middle ground.

## Success Criteria

1. **MI ratio (OOD/ID) > 1.5x** — clear separation, ideally 2x+
2. **test_id_ppl ≤ 80** — some degradation from A0's 49.11 is acceptable, but not catastrophic
3. **Flip rate separation** — OOD flip rate noticeably higher than ID
4. **Val loss converges** within 5K steps (not still dropping at end)

## Run Plan

1. Update `kl_weight: 0.1` in `configs/a1_agnews.yaml`
2. Run `python experiments/a1_bayes_output.py --config configs/a1_agnews.yaml`
3. Compare MLflow metrics to run `686e3201` (kl_weight=0.01)
4. If MI separation is still weak, consider:
   - `kl_weight=0.5` (aggressive)
   - `prior_std=0.1` (tighter prior — penalizes large μ more, forces weights closer to zero)
   - `init_rho=-3` (start with σ≈0.05 instead of 0.007 — higher initial variance)
