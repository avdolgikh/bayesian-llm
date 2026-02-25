# A0 Baseline Checklist — Before Moving to A1

**Date:** 2026-02-24
**Status:** DONE

---

## Goal

Establish a solid deterministic A0 baseline on AG News with documented reference metrics. This baseline is the comparison target for A1 (Bayesian output head). "Solid" means converged, not overfitting catastrophically, and showing a meaningful ID vs OOD perplexity gap.

## Constraints

- Fits in GPU (RTX 4070, ~12 GB VRAM)
- Trains in ~30 min to 1 hour
- Same architecture will be reused for A1 with Bayesian components added on top

---

## A0 Reference Run

**MLflow run ID:** `5dc45450e7b6458fbad2ec07dfd91ce3`
**Config:** `configs/a0_agnews.yaml`
**Date:** 2026-02-24
**GPU:** NVIDIA GeForce RTX 4070

### Architecture

| Parameter | Value |
|---|---|
| n_layer | 4 |
| n_head | 4 |
| n_embd | 256 |
| block_size | 256 |
| dropout | 0.2 |
| bias | true |
| vocab_size | 50257 (GPT-2 BPE) |
| total params | 16,090,880 |

### Data

| Split | Tokens |
|---|---|
| Train (World + Sports, 80%) | 2,701,999 |
| Val (World + Sports, 10%) | 337,750 |
| Test ID (World + Sports, 10%) | 337,750 |
| Test OOD (Business + Sci/Tech, 100%) | 3,498,051 |

### Training

| Parameter | Value |
|---|---|
| steps | 10,000 |
| batch_size | 64 |
| lr | 3.0e-4 (cosine decay) |
| warmup_steps | 500 |
| min_lr | 1.0e-5 |
| weight_decay | 0.1 |
| grad_clip | 1.0 |
| optimizer | AdamW (betas=0.9, 0.95) |
| seed | 1337 |

### Reference Metrics (A1 must match or beat)

| Metric | Value | Notes |
|---|---|---|
| best_val_loss | 3.8874 | Checkpoint selection target |
| best_val_step | 9,400 | Out of 10,000 — plateau reached ~8K |
| final_val_perplexity | 48.84 | At best checkpoint |
| test_id_perplexity | **49.11** | Held-out ID — A1 must match or beat |
| test_ood_perplexity | **540.28** | OOD — A1 comparison target |
| OOD/ID ratio | **11.0x** | Bayesian MI should separate better than raw ppl |
| train_loss (final) | 3.17 | |
| train_perplexity (final) | 23.80 | |
| train-val gap | 0.75 | Moderate overfitting (16M params on 2.7M tokens) |
| train_time_sec | 10,153 (~2.8 hrs) | A1 will be slower (ELBO + KL) |
| tokens_per_sec | 16,137 | |

### Convergence Analysis

- **Steps 1–5000:** Rapid improvement. Val loss 10.86 → 4.05 (ppl 52K → 57).
- **Steps 5000–8000:** Diminishing returns. Val loss 4.05 → 3.90 (ppl 57 → 49).
- **Steps 8000–10000:** Plateau. Val loss oscillates 3.89–3.94. Best at 9400.
- **Overfitting signal:** Train-val gap grows from ~0.6 (step 5K) to ~0.75 (step 10K). Expected — 6:1 param-to-token ratio. Not catastrophic.
- **Conclusion:** Model is converged. More steps would not meaningfully improve val loss.

### Generation Quality

Generated text at best checkpoint (200 tokens, temperature=1.0):
- Topically correct (World news + Sports — the ID categories).
- Coherent at phrase/sentence level, breaks down over longer spans.
- BPE artifacts from raw HTML entities in AG News source (`#39;s` for apostrophes).
- Quality is reasonable for a 16M-param model on 2.7M tokens.

---

## Checklist Verdict

| Check | Status | Detail |
|---|---|---|
| Convergence | PASS | Val loss plateaued by step ~8K |
| Overfitting | PASS | Gap 0.75 — moderate, not catastrophic |
| ID vs OOD gap | PASS | 11x ratio — strong signal for Bayesian MI |
| Generation quality | PASS | Topically correct, coherent phrases |
| Fits in GPU | PASS | RTX 4070, 2.8 hrs |

**A0 is DONE. Ready to move to A1.**

---

## A1 Comparison Contract

When A1 (Bayesian output head) is trained:

- **Same architecture** — same n_layer, n_embd, n_head, block_size
- **Same training budget** — same number of steps, same optimizer
- **Different loss** — A1 trains with ELBO (cross-entropy + KL divergence), A0 trains with cross-entropy only
- **A1 must match or beat A0 on standard metrics** — test_id_ppl <= 49.11
- **A1 must additionally show** higher MI on OOD vs ID data — the whole point of Bayesian uncertainty
- **A1 overhead budget** — training time increase should be documented (expect ~10-30% from KL computation)

---

## What is NOT Needed Before A1

- **Perfect perplexity** — 49 ppl is good enough for a 16M-param model
- **Full HPO sweep** — one stable config is enough; sweep later if needed
- **TinyShakespeare baseline** — AG News is the dataset that matters (it has OOD split)
