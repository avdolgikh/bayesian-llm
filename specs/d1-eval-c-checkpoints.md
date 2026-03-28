# D1: Evaluate C Checkpoints with Community-Standard Metrics

**Date:** 2026-03-27
**Status:** PROPOSED
**Prerequisite:** D0 (metrics framework in `minigpt/uncertainty.py`) — DONE
**Scope:** C milestones only (16L scale). 4L (A/B) deferred.

---

## Goal

Run the full D0 metrics suite (AUROC, FPR@95, AUPRC, ECE, Brier, NLL, AURC) on all 6 C checkpoints. Produce the publication-ready table that upgrades MI ratio to community-standard numbers.

---

## Checkpoints Inventory

All checkpoints verified present in `data/checkpoints/`.

| Milestone | Type | Checkpoint | Dependencies | MC Method |
|-----------|------|-----------|-------------|-----------|
| **C0** | Deterministic | `c0/ckpt_best.pt` | None | No MC. Use max-prob as uncertainty score. |
| **C1** | Variational FFN | `c1/ckpt_best.pt` | None | Full MC (all weights sampled) |
| **C2** | Laplace FFN | `c2/laplace_state.pt` | `c0/ckpt_best.pt` as base model | Laplace sampling via `compute_laplace_uncertainty` |
| **C3** | BLoB LoRA | `c3/ckpt_best.pt` | `c3/base/ckpt_best.pt` (pretrained base) | Full MC (LoRA weights sampled) |
| **C4-TFB** | TFB LoRA | `c4_tfb/tfb_state.pt` | `c3/ckpt_best.pt` (BLoB→DeterministicLoRA conversion) | TFB sampling via `compute_tfb_uncertainty` |
| **C4-LAP** | Laplace LoRA | `c4_lap/laplace_state.pt` | `c3/ckpt_best.pt` (BLoB→DeterministicLoRA conversion) | Laplace sampling via `compute_laplace_uncertainty` |

### Checkpoint Format

Each `.pt` file contains:
```python
{
    "step": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "config": dict,              # full config used during training
    "best_val_loss": float,
    "rng_state": Tensor,
    "cuda_rng_state": Tensor,   # if CUDA
}
```

Post-hoc state files (`laplace_state.pt`, `tfb_state.pt`) have their own format — loaded via `load_laplace_state()` / `load_tfb_state()`.

---

## Loading Each Checkpoint

### C0: Deterministic Baseline

```
1. Build model from C0 config (16L/8H/512d, no Bayesian)
2. load_checkpoint("data/checkpoints/c0/ckpt_best.pt", model)
3. No MC — single forward pass
4. Uncertainty score: negated max softmax probability (1 - max_k p_k)
```

### C1: Variational FFN

```
1. Build model from C1 config (16L/8H/512d, bayes_ffn enabled, init_rho=-2.0)
2. load_checkpoint("data/checkpoints/c1/ckpt_best.pt", model)
3. MC: N forward passes with weight sampling
4. Uncertainty score: MI (primary), predictive entropy, max-prob
```

### C2: Laplace on Full Weights

```
1. Build model from C0 config (deterministic)
2. load_checkpoint("data/checkpoints/c0/ckpt_best.pt", model)
3. Load Laplace state: load_laplace_state("data/checkpoints/c2/laplace_state.pt")
4. MC: compute_laplace_uncertainty(model, state, ...) — Laplace sampling
5. Uncertainty score: MI from Laplace MC
```

### C3: BLoB LoRA

```
1. Build model from C3 config (16L/8H/512d, deterministic base)
2. load_checkpoint("data/checkpoints/c3/base/ckpt_best.pt", model)  # base
3. inject_lora(model, lora_cfg, bayesian=True)  # BLoB LoRA
4. Load BLoB LoRA weights from "data/checkpoints/c3/ckpt_best.pt"
5. MC: N forward passes (LoRA weights sampled via BLoBLoRALinear)
6. Uncertainty score: MI (primary), predictive entropy, max-prob
```

Config for LoRA injection: `rank=16, alpha=32, target="ffn", prior_std=0.2, init_g=0.1`.

### C4-TFB: TFB on LoRA

```
1. Build model from C3 config (16L/8H/512d, deterministic base)
2. Load C3 BLoB checkpoint → convert to DeterministicLoRA
   (same _prepare_model logic from c_pipeline.py: lora_A_mu → lora_A)
3. Load TFB state: load_tfb_state("data/checkpoints/c4_tfb/tfb_state.pt")
4. MC: compute_tfb_uncertainty(model, state, ...) — TFB sampling
5. Uncertainty score: MI from TFB MC
```

### C4-LAP: Laplace on LoRA

```
1. Build model from C3 config (16L/8H/512d, deterministic base)
2. Load C3 BLoB checkpoint → convert to DeterministicLoRA (same as C4-TFB)
3. Load Laplace state: load_laplace_state("data/checkpoints/c4_lap/laplace_state.pt")
4. MC: compute_laplace_uncertainty(model, state, ...) — Laplace LoRA sampling
5. Uncertainty score: MI from Laplace MC
```

---

## Dataset

The Pile domain-split (same as C training):
- **ID:** Wikipedia + StackExchange
- **OOD:** ArXiv, FreeLaw, PubMed

For C3/C4 (LoRA milestones), ID during fine-tuning was HackerNews, but for OOD detection evaluation we use the same ID/OOD split as C0/C1 for comparability.

Cached at `data/pile/{domain}_{token_limit}.pt`.

---

## Evaluation Protocol

For each checkpoint:

### Step 1: Load model + data

Load test_id and test_ood splits. Use same block_size=256 as training.

### Step 2: Per-sequence MC sampling

For each test sequence (N=20 MC passes for Bayesian, 1 pass for C0):
- Collect per-token probabilities from each MC pass
- Compute per-token: MI, predictive entropy, max softmax probability
- Aggregate to sequence-level: `aggregate_sequence_scores(method="mean")`

### Step 3: OOD detection metrics

Using sequence-level uncertainty scores (ID label=0, OOD label=1):
- `auroc(scores, labels)`
- `fpr_at_tpr(scores, labels, target_tpr=0.95)`
- `auprc(scores, labels)`

Compute for all three uncertainty scores: MI, predictive entropy, max-prob.

### Step 4: Calibration metrics

On ID test data (next-token prediction):
- `ece(max_probs, correct, n_bins=15)`
- `nll(probs, targets)` — per-token, averaged
- `brier_score(probs, targets)` — per-token, averaged

### Step 5: Selective prediction

Using MI as uncertainty score on combined ID+OOD test data:
- `risk_coverage_curve(uncertainties, correct)`
- `aurc(uncertainties, correct)`

---

## Output: Publication-Ready Table

```
| Milestone | Method          | MI Ratio | AUROC | FPR@95 | AUPRC | ECE   | Brier | NLL  | AURC  |
|-----------|-----------------|----------|-------|--------|-------|-------|-------|------|-------|
| C0        | Deterministic   |    —     |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |  ...  |
| C1        | Variational FFN |  1.32x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |  ...  |
| C2        | Laplace FFN     |  1.00x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |  ...  |
| C3        | BLoB LoRA       |  1.53x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |  ...  |
| C4-TFB    | TFB LoRA        |  1.35x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |  ...  |
| C4-LAP    | Laplace LoRA    |  1.00x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |  ...  |
```

Secondary table — uncertainty score comparison (all using AUROC):

```
| Milestone | MI AUROC | Pred. Entropy AUROC | Max-Prob AUROC |
|-----------|----------|---------------------|----------------|
| C0        |    —     |        ...          |      ...       |
| C1        |   ...    |        ...          |      ...       |
| ...       |   ...    |        ...          |      ...       |
```

---

## Implementation

Script: `scripts/eval_c_checkpoints.py`

```
Usage:
  python scripts/eval_c_checkpoints.py                    # eval all 6
  python scripts/eval_c_checkpoints.py --milestone c3     # eval one
  python scripts/eval_c_checkpoints.py --n-samples 10     # fewer MC passes (faster)
  python scripts/eval_c_checkpoints.py --n-sequences 200  # fewer test sequences
```

**Not TDD'd** — this is a runnable evaluation tool, not library code. The metrics functions it calls (D0) are already tested (45 tests).

### Key design decisions:
- Reuse `experiments/experiment_setup.py` for config/model setup where possible
- Reuse `_prepare_model` logic from `c_pipeline.py` for C4 BLoB→DeterministicLoRA conversion
- Reuse `compute_laplace_uncertainty` / `compute_tfb_uncertainty` for post-hoc MC
- Print markdown table to stdout for easy copy-paste into `report.md`
- Log to MLflow if available (optional `--no-mlflow` flag)

### Estimated runtime:
- C0: ~2 min (single forward pass, no MC)
- C1: ~20 min (N=20 full MC, 76M params all sampled)
- C2: ~15 min (Laplace MC, cheaper than variational)
- C3: ~5 min (N=20 MC, only LoRA sampled)
- C4-TFB: ~5 min (TFB MC, same cost as C3)
- C4-LAP: ~5 min (Laplace LoRA MC)
- **Total: ~50 min for all 6**

---

## Success Criteria

- All 6 rows populated with real numbers
- AUROC ranking correlates with MI ratio ranking (C3 > C4-TFB > C1 > C2 ≈ C4-LAP)
- C0 AUROC for max-prob gives a non-Bayesian baseline
- Tables copied into `report.md`
