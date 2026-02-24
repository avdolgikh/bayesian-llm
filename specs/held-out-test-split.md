# Held-out Test Set (ID + OOD) for Data Pipeline

**Date:** 2026-02-24
**Status:** Implemented

## Problem

The data pipeline returns `{train, val, test_ood}`. Val is used both for checkpoint selection during training AND as the "in-distribution test set" for final evaluation. This is methodologically wrong -- val is biased because the best checkpoint was selected to minimize val loss. For Phase 2 (Bayesian uncertainty), we need a truly held-out ID test set to compare MI on ID vs OOD without bias.

## Goal

Every run produces `{train, val, test_id, test_ood}` where `test_id` is never seen during training or checkpoint selection.

## Changes

### 1. `minigpt/config.py` -- add `test_fraction`

- Add `"test_fraction": 0.1` to `DEFAULT_CONFIG["data"]`
- Add validation in `validate_config()`: `val_fraction + test_fraction < 1.0`

### 2. `minigpt/data.py` -- three-way ID split

**`prepare_data()` (TinyShakespeare):**
- Add `test_fraction` param
- Split token stream: `[train 80% | val 10% | test_id 10%]`
- Return `{"train", "val", "test_id", "test_ood": None}`

**`prepare_agnews_data()` (AG News):**
- Add `test_fraction` param
- ID tokens split into `train / val / test_id` (80/10/10)
- OOD tokens remain `test_ood` (unchanged)
- Return `{"train", "val", "test_id", "test_ood"}`

**`load_dataset()` dispatcher:**
- Read `test_fraction` from config, pass through to both prepare functions

### 3. `configs/a0_baseline.yaml` and `configs/a0_agnews.yaml`

- Add `test_fraction: 0.1` to the `data:` section

### 4. `experiments/a0_baseline.py` -- evaluate on test_id

- Unpack `test_id = data["test_id"]`
- Print `test_id` token count
- After training, evaluate `test_id_perplexity` (separate from val)
- ID vs OOD comparison uses `test_id` vs `test_ood` (not val vs test_ood)
- Log `test_id_perplexity` and `test_ood_perplexity` to MLflow

### 5. No changes needed

- `minigpt/train.py` -- takes train+val, unaffected
- `minigpt/evaluate.py` -- generic over any tensor
- `minigpt/model.py`, `layers.py` -- no data logic
- Checkpoint format -- unchanged, `--resume` works as before

## Split Ratios

| Dataset | train | val | test_id | test_ood |
|---|---|---|---|---|
| TinyShakespeare (304K tok) | 80% (~243K) | 10% (~30K) | 10% (~30K) | None |
| AG News ID (~3M tok) | 80% (~2.4M) | 10% (~300K) | 10% (~300K) | all OOD cats (~3.5M) |

## Verification

1. Run `python experiments/a0_baseline.py --config configs/a0_baseline.yaml --set train.steps=50` -- confirm 4 splits printed, `test_id_perplexity` logged
2. Run `python experiments/a0_baseline.py --config configs/a0_agnews.yaml --set train.steps=50` -- confirm test_id AND test_ood perplexity printed, ID vs OOD comparison
3. Check MLflow: `test_id_perplexity` and `test_ood_perplexity` metrics present
