# A1 Tuning Round 2 — Four Fixes Before the Next Run

**Date:** 2026-02-26
**Previous run:** MLflow `4e1662e6` (kl_weight=0.2, 40K steps, init_rho=-5)
**Goal:** Fix four issues discovered during A1 round 1 analysis, then launch a cleaner run.

## Summary of Round 1 Results

| Metric | ID | OOD | Ratio | Note |
|--------|---:|----:|------:|------|
| MI | 0.0330 | 0.0448 | **1.36x** | Up from 1.07x (kl=0.01), but still weak |
| Flip rate | 14.3% | 16.4% | 1.15x | Modest separation |
| Pred entropy | 3.92 | 4.62 | 1.18x | — |
| Perplexity (ID) | 64.1 | — | — | Comparable to kl=0.01 run |
| Sigma (mean) | — | — | 0.224 | Grew from 0.007 but took 40K steps |
| Qualitative MI ratio | — | — | **1.02x** | **Broken** — scored generated text, not prompts |

Key takeaway: the model *can* separate ID from OOD via MI (1.36x), but init_rho=-5 throttled sigma growth, best-checkpoint selected the wrong model, and qualitative MI was measuring the wrong thing.

---

## Change 1: Fix Qualitative MI — Score Prompt Tokens

### Problem

`experiments/a1_bayes_output.py`, function `_run_qualitative_eval` (lines 84–96).

Currently the code:
1. Generates a continuation from a prompt using `model.generate(..., use_mean=True)`
2. Scores the **full generated sequence** with `score_sequence(model, full_ids, ...)`
3. Averages MI over the **continuation part** only (`metrics["mi"][len(tokens):]`)

Since generation uses `use_mean=True` (deterministic mean weights), the continuation is always in-distribution text *from the model's perspective*. MI is flat for both ID and OOD prompts → ratio 1.02x.

### Fix

Score MI on the **prompt tokens only**. The prompt IS the real OOD/ID text — that's what we want to measure uncertainty on.

```python
# BEFORE (lines 84-96):
full_ids = generated_ids[0]
if len(full_ids) > 1:
    metrics = score_sequence(model, full_ids, device, n_samples=n_samples)
    cont_mi = metrics["mi"][len(tokens):].mean().item()
    cont_flip = metrics["flip_rate"][len(tokens):].mean().item()
    cont_pred_ent = metrics["predictive_entropy"][len(tokens):].mean().item()

# AFTER:
prompt_ids = idx[0]  # original prompt tensor, not the generated sequence
if len(prompt_ids) > 1:
    metrics = score_sequence(model, prompt_ids, device, n_samples=n_samples)
    prompt_mi = metrics["mi"].mean().item()
    prompt_flip = metrics["flip_rate"].mean().item()
    prompt_pred_ent = metrics["predictive_entropy"].mean().item()
```

Still generate and display continuations for human inspection — just don't use them for MI scoring.

### Impact

- Only affects text artifacts (`qualitative_eval.txt`, `qualitative_metrics.json`)
- The primary numeric metrics (`mi_id_mean`, `mi_ood_id_ratio`) already use the correct global `compute_uncertainty_metrics` on test data — those are fine
- Variable naming: rename `cont_mi` → `prompt_mi` (etc.) throughout the function for clarity

---

## Change 2: Best-ELBO Checkpoint Selection

### Problem

`minigpt/train.py`, line 280:
```python
is_best = val_metrics["loss"] < best_val_loss
```

This compares raw CE loss only. For Bayesian models, CE val loss plateaued at step 8K while ELBO continued improving through step 40K (6.70 → 5.44). The best-CE checkpoint (step 8K, CE=4.18) has smaller sigmas and worse MI than the final checkpoint.

The `elbo_loss` key is already computed in `estimate_loss()` (line 82):
```python
result["elbo_loss"] = avg_ce + kl_scale * kl
```

### Fix

When `is_bayesian=True`, compare `val_metrics["elbo_loss"]` instead of `val_metrics["loss"]`:

```python
# Determine the metric for best-checkpoint selection
if is_bayesian:
    val_criterion = val_metrics.get("elbo_loss", val_metrics["loss"])
else:
    val_criterion = val_metrics["loss"]

is_best = val_criterion < best_val_loss
if is_best:
    best_val_loss = val_criterion
    best_val_step = step
    save_checkpoint(...)
```

Additional changes:
- Print which criterion is used at training start (e.g., `"Best-checkpoint criterion: ELBO"`)
- Log `best_val_elbo` to MLflow alongside existing `best_val_loss`

### Checkpoint stored value

`best_val_loss` in the checkpoint dict will hold ELBO for Bayesian runs. This is fine — it's only used for comparison within the same run, and resume will compare ELBO-to-ELBO consistently.

---

## Change 3: init_rho = -1

### Problem

`init_rho=-5` creates a softplus saturation bottleneck:

| init_rho | σ = softplus(ρ) | sigmoid(ρ) | Gradient attenuation |
|----------|----------------|------------|---------------------|
| -5 | 0.0067 | 0.0067 | **143x** |
| -3 | 0.0486 | 0.047 | 21x |
| -2 | 0.127 | 0.119 | 8.4x |
| **-1** | **0.313** | **0.269** | **3.7x** |
| 0 | 0.693 | 0.500 | 2.0x |

With init_rho=-5, the softplus derivative `sigmoid(-5) ≈ 0.007` attenuates gradients flowing to `weight_rho` by 143x. It took 40K steps just to reach σ ≈ 0.22.

### Fix

Set `init_rho = -1`:
- σ = softplus(-1) = 0.313 — starts near where the previous run ended (σ ≈ 0.22)
- sigmoid(-1) = 0.269 — only 3.7x gradient attenuation (vs 143x)
- Initial KL per weight drops from 4.5 to 0.67 nats (total: 58M → 8.6M)

### Files

```yaml
# configs/a1_agnews.yaml line 24
init_rho: -1.0    # was -5.0
```

```python
# minigpt/config.py DEFAULT_CONFIG, line 37
"init_rho": -1.0,  # was -5.0
```

No code changes in `layers.py` — `init_rho` is already parameterized.

---

## Change 4: Sigma Distribution Logging

### Problem

After training, we have no sigma statistics logged to MLflow. We had to use `scripts/eval_checkpoint.py` manually to check sigma health. This should be automated.

### Fix

1. **Extract** the sigma stats logic from `scripts/eval_checkpoint.py::_sigma_stats()` (lines 30–70) into a reusable function in `minigpt/layers.py`
2. **Call** it from `experiments/a1_bayes_output.py` after training completes
3. **Log** results as MLflow params

### New function in `minigpt/layers.py`

```python
def sigma_summary(model: torch.nn.Module) -> dict[str, float]:
    """Compute aggregate sigma statistics across all BayesianLinear layers.

    Returns dict with keys: sigma_mean, sigma_std, sigma_min, sigma_max,
    sigma_median, sigma_p5, sigma_p25, sigma_p75, sigma_p95.
    Returns empty dict if no BayesianLinear layers found.
    """
    all_sigmas = []
    for module in model.modules():
        if isinstance(module, BayesianLinear):
            sigma = torch.nn.functional.softplus(module.weight_rho).detach()
            all_sigmas.append(sigma.flatten())
    if not all_sigmas:
        return {}
    combined = torch.cat(all_sigmas)
    pcts = torch.tensor([0.05, 0.25, 0.75, 0.95])
    quantiles = torch.quantile(combined.float(), pcts)
    return {
        "sigma_mean": combined.mean().item(),
        "sigma_std": combined.std().item(),
        "sigma_min": combined.min().item(),
        "sigma_max": combined.max().item(),
        "sigma_median": combined.median().item(),
        "sigma_p5": quantiles[0].item(),
        "sigma_p25": quantiles[1].item(),
        "sigma_p75": quantiles[2].item(),
        "sigma_p95": quantiles[3].item(),
    }
```

### Logging in `experiments/a1_bayes_output.py`

After training, before the final eval block:
```python
from minigpt.layers import sigma_summary
stats = sigma_summary(model)
if stats:
    mlflow.log_params(stats)
```

### Refactor `scripts/eval_checkpoint.py`

Update `_sigma_stats()` to call `sigma_summary()` internally and just add the per-layer printing on top. This avoids duplicate logic.

---

## Config for Round 2

Only one config value changes from the previous run:

```yaml
# configs/a1_agnews.yaml
model:
  bayes:
    kl_weight: 0.2       # unchanged — working well
    init_rho: -1.0        # was -5.0 — fix softplus saturation
    prior_std: 1.0        # unchanged
```

Everything else stays the same (40K steps, kl_annealing=5000, etc.).

## Expected Outcomes

| Metric | Round 1 (init_rho=-5) | Round 2 Expected (init_rho=-1) |
|--------|----------------------:|-------------------------------:|
| MI ratio (OOD/ID) | 1.36x | **>1.5x** (faster sigma growth) |
| Sigma at step 1K | ~0.007 | ~0.3+ |
| Best checkpoint | step 8K (CE) | step with best ELBO |
| Qualitative MI ratio | 1.02x (broken) | Meaningful (fixed) |
| Sigma stats in MLflow | No | Yes |

## Success Criteria

1. **MI ratio (OOD/ID) > 1.5x** — ideally 2x+
2. **test_id_ppl ≤ 80** — some degradation from A0 (49.11) is acceptable
3. **Best checkpoint = best-ELBO** — not just best-CE
4. **Qualitative MI ratio > 1.2x** — prompt-based scoring shows separation
5. **Sigma stats logged** — visible in MLflow without manual scripts

## Implementation Order

1. Change 1 (qualitative MI fix) — `experiments/a1_bayes_output.py`
2. Change 4 (sigma logging) — `minigpt/layers.py`, `experiments/a1_bayes_output.py`, `scripts/eval_checkpoint.py`
3. Change 2 (best-ELBO) — `minigpt/train.py`
4. Change 3 (init_rho) — `configs/a1_agnews.yaml`, `minigpt/config.py`
5. Run: `python experiments/a1_bayes_output.py --config configs/a1_agnews.yaml`
