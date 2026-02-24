# A0 Baseline Checklist — Before Moving to A1

**Date:** 2026-02-24
**Status:** Proposed

---

## Goal

Establish a solid deterministic A0 baseline on AG News with documented reference metrics. This baseline is the comparison target for A1 (Bayesian output head). "Solid" means converged, not overfitting catastrophically, and showing a meaningful ID vs OOD perplexity gap.

## Constraints

- Fits in GPU (RTX 4070, ~12 GB VRAM)
- Trains in ~30 min to 1 hour
- Same architecture will be reused for A1 with Bayesian components added on top

---

## What to Check from the A0 Run

### 1. Convergence

Is loss still decreasing at the final step? If yes, the model is undertrained — increase steps until val loss flattens.

### 2. Overfitting

Is train loss much lower than val loss? A large gap means memorization. Tune dropout / weight decay / model size.

### 3. ID vs OOD perplexity gap

There must be a meaningful gap (OOD > ID). If they're nearly equal, the model isn't learning domain-specific patterns, and Bayesian uncertainty measurement in A1 won't have a signal to amplify.

### 4. Generation quality

Does the generated sample look like AG News text? Sanity check that the model learned something real, not noise.

---

## Reference Metrics to Record

| Metric | What to record | Why it matters for A1 |
|---|---|---|
| `best_val_loss` | Lowest validation loss achieved | A1 must converge similarly |
| `best_val_step` | Step at which best val occurred | Overfitting signal — if best is early, model overfit past that point |
| `test_id_perplexity` | Baseline ID quality | A1 must match or beat this |
| `test_ood_perplexity` | Baseline OOD quality | A1 comparison target |
| OOD - ID gap | The perplexity difference | Bayesian MI should separate ID/OOD better than raw perplexity does |
| Training time (wall clock) | Total training duration | A1 will be slower (ELBO + KL), need to know the overhead |
| `n_params` | Parameter count | A1 adds params (mean + variance for Bayesian layers) |

Once established, these numbers go into AGENTS.md as the A0 reference.

---

## What is NOT Needed Before A1

- **Perfect perplexity** — "good enough and converged" is sufficient
- **Full HPO sweep** — one stable config is enough; sweep later if needed
- **TinyShakespeare baseline** — AG News is the dataset that matters (it has OOD split)

---

## A1 Comparison Contract

When A1 (Bayesian output head) is trained:

- **Same architecture** — same n_layer, n_embd, n_head, block_size
- **Same training budget** — same number of steps, same optimizer
- **Different loss** — A1 trains with ELBO (cross-entropy + KL divergence), A0 trains with cross-entropy only
- **A1 must match or beat A0 on standard metrics** (test_id_perplexity) — Bayesian should not degrade quality
- **A1 must additionally show** higher MI on OOD vs ID data — the whole point of Bayesian uncertainty

---

## Decision Flow

```
Run A0 on AG News
       |
       v
Did val loss flatten? ──no──> Increase steps
       |
      yes
       |
       v
Is OOD ppl > ID ppl? ──no──> Model isn't learning domain patterns.
       |                      Check data pipeline, increase capacity.
      yes
       |
       v
Fits in ~30 min? ────no───> Reduce model size or steps
       |
      yes
       |
       v
Document reference metrics in AGENTS.md
       |
       v
Move to A1
```
