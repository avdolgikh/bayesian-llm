# B3: Post-hoc Bayesianization of LoRA — Technical Specification

Status: Draft for implementation planning
Date: 2026-03-12
Process mode: BDD → TDD → Code
Current stage: Stage #1 (spec only; no tests or code changes in this stage)

## 1. Overview

B3 completes the post-hoc + LoRA cell of the 2×2 comparative matrix. It takes a **deterministically trained LoRA** and adds Bayesian uncertainty **after training** — no variational inference, no KL, no ELBO.

Two approaches, same checkpoint:

| Approach | Method | Variance source | Cost |
|----------|--------|----------------|------|
| **B3-TFB** | Training-Free Bayesianization | SVD of $\mathbf{B}$ → scalar search on anchor data | Minutes (forward passes only) |
| **B3-LAP** | Diagonal Laplace on LoRA | Per-sample Fisher curvature on LoRA $\mathbf{A}$ params | Seconds (one gradient pass) |

Both produce a Gaussian posterior over LoRA $\mathbf{A}$ parameters, evaluated with the same MC protocol as A2/B1/B2.

### Why B3 matters

- Completes the 2×2 matrix. Without B3, the paper has no post-hoc + LoRA data point.
- Direct comparison to B2 (variational LoRA, MI ratio 1.13x): does post-hoc variance on the same LoRA subspace match train-time variational inference?
- Direct comparison to B1 (post-hoc full weights, MI ratio 1.00x): does restricting Laplace to a low-rank subspace help?
- TFB (NeurIPS 2025) is state-of-the-art for post-hoc LoRA Bayesianization. Testing it on our controlled setup is scientifically valuable regardless of outcome.

## 2. Goals

1. Complete the post-hoc + LoRA cell of the 2×2 matrix.
2. Compare TFB vs Laplace-LoRA on the same deterministic LoRA checkpoint.
3. Compare against B2 (variational LoRA) and B1 (post-hoc full weights) on the same evaluation protocol.
4. Determine whether post-hoc methods can extract OOD-discriminative uncertainty from a LoRA subspace.

## 3. Background

### 3.1 Deterministic LoRA (Standard)

Standard LoRA decomposes the weight update as:

$$\mathbf{h} = \left(\mathbf{W}_0 + \frac{\alpha}{r}\,\mathbf{B}\,\mathbf{A}\right) \mathbf{x}$$

- $\mathbf{W}_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$: frozen pretrained weight.
- $\mathbf{B} \in \mathbb{R}^{d_\text{out} \times r}$: deterministic, init zeros.
- $\mathbf{A} \in \mathbb{R}^{r \times d_\text{in}}$: deterministic, init Kaiming uniform.
- Both $\mathbf{A}$ and $\mathbf{B}$ are trained with standard CE loss (no KL, no ELBO).

B3 starts from this trained deterministic LoRA and adds uncertainty post-hoc.

### 3.2 TFB: Training-Free Bayesianization (NeurIPS 2025)

**Core idea:** Find the maximum amount of noise you can inject into trained LoRA $\mathbf{A}$ weights without degrading performance beyond a tolerance $\varepsilon$.

**Step 1 — SVD of B.** For each LoRA layer, compute compact SVD:

$$\mathbf{B} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^\top$$

The singular values $d_1 \geq d_2 \geq \dots \geq d_r$ encode which directions in $\mathbf{A}$-space are amplified most by $\mathbf{B}$.

**Step 2 — Variance parameterization.** Define per-element standard deviations:

$$\Omega_{ij} = \frac{\sigma_q}{d_i}$$

where $\sigma_q > 0$ is a single scalar. Directions corresponding to large singular values get less noise (they matter more); directions with small singular values get more noise. This reduces the entire variance search to **one scalar** $\sigma_q$.

**Step 3 — Binary search.** Find the maximum $\sigma_q$ such that:

$$\left|\mathcal{L}\!\left(\mathcal{D}_\text{anchor} \mid \mathbf{A}_\text{MAP}, \mathbf{B}, \boldsymbol{\Omega}(\sigma_q)\right) - \mathcal{L}\!\left(\mathcal{D}_\text{anchor} \mid \mathbf{A}_\text{MAP}, \mathbf{B}\right)\right| \leq \varepsilon$$

where $\mathcal{L}(\cdot \mid \boldsymbol{\Omega})$ denotes the expected loss under $K$ MC samples from $\mathcal{N}(\mathbf{A}_\text{MAP}, \mathrm{diag}(\boldsymbol{\Omega}^2))$. The anchor dataset $\mathcal{D}_\text{anchor}$ can be the LoRA training data or a holdout set.

**Step 4 — Posterior sampling at inference:**

$$\hat{\mathbf{A}} = \mathbf{A}_\text{MAP} + \boldsymbol{\Omega} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\,\mathbf{I})$$

**Theoretical grounding (TFB Theorem 4.2):** Under local convexity, this variance search is equivalent to generalized KL-regularized variational inference:

$$\min_{\sigma_q} \left[\mathcal{L}(\sigma_q) + \lambda\,\mathrm{KL}\!\left[q(\mathbf{A} \mid \sigma_q)\;\|\;p(\mathbf{A})\right]\right]$$

This connects the pragmatic search to principled Bayesian inference.

### 3.3 Laplace-LoRA: Diagonal Laplace on LoRA Parameters

**Core idea:** Fit a diagonal Gaussian posterior on LoRA $\mathbf{A}$ parameters using empirical Fisher curvature, then sample for MC evaluation.

**Step 1 — Curvature fitting.** Compute diagonal Fisher on LoRA $\mathbf{A}$ params:

$$F_{ii} = \mathbb{E}\!\left[\left(\frac{\partial \mathcal{L}}{\partial A_i}\right)^2\right]$$

Using per-sample squared gradients (same method as B1 Approach B, but restricted to LoRA params).

**Step 2 — Posterior:**

$$q(\mathbf{A}) = \mathcal{N}\!\left(\mathbf{A}_\text{MAP},\;\mathrm{diag}\!\left(\frac{s^2}{F_{ii} + \lambda}\right)\right)$$

where $\lambda > 0$ is damping and $s$ is sample scale.

**Step 3 — Sampling:** Same as B1 — draw from the Gaussian posterior, apply to LoRA params, run forward pass.

**Key difference from B1:** B1 applied Laplace to 2M FFN params with curvature ~$10^{-5}$ (too flat). Laplace-LoRA targets ~82K LoRA params that were trained for a specific adaptation task. The curvature may be more structured because LoRA training concentrates learning signal into fewer parameters.

### 3.4 Why Bayesianize Only A (Not B)

Both TFB and BLoB follow the same convention: **B is deterministic, A is stochastic.**

- TFB's variance parameterization ($\Omega_{ij} = \sigma_q / d_i$) requires B to be fixed — the SVD of B defines the variance structure.
- BLoB Theorem 3.1: if both A and B are stochastic, $\mathbb{E}[\mathbf{B}\mathbf{A}] \neq \mathbb{E}[\mathbf{B}]\,\mathbb{E}[\mathbf{A}]$, causing mean-field pathologies.
- For controlled comparison with B2, using the same A-only convention isolates the treatment variable (variational vs post-hoc).

## 4. Architecture

### 4.1 New Layer: `DeterministicLoRALinear` (in `minigpt/lora.py`)

A standard LoRA wrapper with no Bayesian parameters. Used for Phase 1 (deterministic LoRA training).

**Parameters:**
- `base_linear` (nn.Linear): frozen pretrained weight.
- `lora_B` (nn.Parameter): $\mathbb{R}^{d_\text{out} \times r}$. Init: zeros.
- `lora_A` (nn.Parameter): $\mathbb{R}^{r \times d_\text{in}}$. Init: Kaiming uniform.
- `scaling` (float): $\alpha / r$.

**Forward:**

```python
base_out = self.base_linear(x)
lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
return base_out + lora_out
```

No sampling, no KL, no G parameter. The simplest possible LoRA.

### 4.2 Injection: `inject_lora` Extension

Extend `inject_lora()` to support deterministic mode:

```python
inject_lora(model, lora_config, bayesian=True)   # existing BLoB behavior
inject_lora(model, lora_config, bayesian=False)  # new: DeterministicLoRALinear
```

When `bayesian=False`:
- Injects `DeterministicLoRALinear` instead of `BLoBLoRALinear`.
- No `prior_std` or `init_g` needed (ignored if present).
- Same freeze logic: all base params frozen, only LoRA A and B trainable.

### 4.3 New Module: `minigpt/tfb.py`

#### `TFBState` Dataclass

```python
@dataclass
class TFBState:
    sigma_q: float                              # fitted scalar variance
    svd_cache: dict[str, tuple[Tensor, Tensor, Tensor]]  # per-layer U, S, V
    a_map: dict[str, Tensor]                    # per-layer A_MAP
    epsilon: float                              # tolerance used in search
    anchor_loss: float                          # MAP loss on anchor data
```

#### `fit_tfb(model, anchor_data, ...) -> TFBState`

1. Compute MAP loss on anchor data (forward-only, no gradients).
2. For each `DeterministicLoRALinear` layer, compute SVD of `lora_B` and cache.
3. Binary search for $\sigma_q$:
   - At each candidate $\sigma_q$, draw $K$ MC samples of $\mathbf{A}$, compute average loss.
   - If $|\text{avg\_loss} - \text{MAP\_loss}| \leq \varepsilon$, increase $\sigma_q$; else decrease.
   - Converge when range is below a precision threshold (e.g., $10^{-4}$).
4. Return `TFBState` with the fitted $\sigma_q$.

#### `sample_tfb_params(state, seed) -> dict[str, Tensor]`

For each layer:

$$\hat{\mathbf{A}} = \mathbf{A}_\text{MAP} + \boldsymbol{\Omega} \odot \boldsymbol{\epsilon}$$

where $\Omega_{ij} = \sigma_q / d_i$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.

Returns dict mapping param name → sampled A tensor.

#### `save_tfb_state` / `load_tfb_state`

Serialize/deserialize `TFBState` via `torch.save`/`torch.load`.

### 4.4 Laplace-LoRA: Reuse B1 Infrastructure

**No new module needed.** Extend existing `minigpt/laplace.py`:

- Add `select_params` mode `"lora"`: selects `lora_A` params from all `DeterministicLoRALinear` layers.
- Everything else (`fit_laplace`, `sample_laplace_params`, `apply_sampled_params`, save/load) works unchanged.

The key insight: `fit_laplace` already computes per-sample squared gradients and returns a `LaplaceState`. The only change is **what params are selected** — LoRA A instead of FFN weights.

### 4.5 MC Evaluation via Context Manager

Both TFB and Laplace-LoRA produce sampled param dicts. Reuse `apply_sampled_params` from `laplace.py` to temporarily patch the model, then run forward pass. This is the same pattern as B1.

For TFB, add `compute_tfb_uncertainty` and `score_sequence_tfb` functions that mirror `compute_laplace_uncertainty` and `score_sequence_laplace`, but use `sample_tfb_params` instead of `sample_laplace_params`.

## 5. Three-Phase Pipeline

### 5.1 Phase 1: Deterministic LoRA Training

**Goal:** Train a standard (non-Bayesian) LoRA on the same setup as B2 R2.

**Base model:** Reuse the existing B2 pretrain checkpoint (`data/checkpoints/b2_pretrain/ckpt_best.pt`, cat 1 World, val ppl=46.5). No base pretraining needed — it was already done in B2.

| Setting | Value | Matches B2 R2 |
|---------|-------|---------------|
| Base checkpoint | B2 pretrain (cat 1 World, val ppl=46.5) | Same (reused, not retrained) |
| LoRA data | Cat 2 (Sports) | Same |
| OOD eval | Cats 3+4 (Business + Sci/Tech) | Same |
| LoRA rank | 16 | Same |
| LoRA alpha | 32.0 | Same |
| LoRA target | ffn | Same |
| LR | 3.0e-4 | Same |
| Steps | 10,000 | Same |
| KL weight | **0.0** | **Different (no ELBO)** |
| Weight decay | **0.01** | **Different (standard regularization)** |

**Output:** Deterministic LoRA checkpoint at `data/checkpoints/b3_lora/ckpt_best.pt`.

### 5.2 Phase 2a: TFB Fitting

**Input:** Deterministic LoRA checkpoint from Phase 1.

**Procedure:**
1. Load model + LoRA checkpoint.
2. Compute MAP loss on anchor data (LoRA training data = cat 2 Sports).
3. SVD of each `lora_B` matrix.
4. Binary search for $\sigma_q$ with tolerance $\varepsilon$.
5. Save `TFBState`.

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `epsilon` | 0.1 | Cross-entropy tolerance (~10% loss increase allowed) |
| `n_search_samples` | 10 | MC samples per binary search step |
| `n_anchor_batches` | 20 | Batches of anchor data for loss estimation |
| `search_range` | [1e-4, 10.0] | Initial σ_q search range |
| `search_precision` | 1e-4 | Stop when range < this |

### 5.3 Phase 2b: Laplace-LoRA Fitting

**Input:** Same deterministic LoRA checkpoint.

**Procedure:**
1. Load model + LoRA checkpoint.
2. Select LoRA A params (`select_params(model, "lora")`).
3. Fit diagonal Fisher curvature via per-sample gradients (reuse `fit_laplace`).
4. Save `LaplaceState`.

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `damping` | 1.0 | Same as B1 starting point |
| `sample_scale` | 1.0 | Same as B1 starting point |
| `n_curvature_batches` | 20 | Same as B1 |
| `selection_mode` | `"lora"` | New mode: selects LoRA A params |

### 5.4 Phase 3: MC Evaluation

Same protocol as A2/B1/B2:
- $N = 20$ MC forward passes with sampled LoRA A params.
- Temperature = 0 (all stochasticity from param sampling).
- Metrics: MI, predictive entropy, expected entropy, flip rate.
- Evaluation on test_id (Sports) and test_ood (Business + Sci/Tech).
- Qualitative prompt-panel scoring.

## 6. Experiment Script: `experiments/b3_post_hoc_lora.py`

Single script with phase selection:

```
python experiments/b3_post_hoc_lora.py --phase train --config configs/b3_lora_agnews.yaml
python experiments/b3_post_hoc_lora.py --phase tfb --config configs/b3_lora_agnews.yaml
python experiments/b3_post_hoc_lora.py --phase laplace --config configs/b3_lora_agnews.yaml
python experiments/b3_post_hoc_lora.py --phase full --config configs/b3_lora_agnews.yaml  # all three
```

**`--phase train`**: Phase 1 only. Train deterministic LoRA, save checkpoint.

**`--phase tfb`**: Phase 2a + 3. Load checkpoint, fit TFB, evaluate MI. Requires trained LoRA.

**`--phase laplace`**: Phase 2b + 3. Load checkpoint, fit Laplace, evaluate MI. Requires trained LoRA.

**`--phase full`**: All phases sequentially: train → TFB eval → Laplace eval.

Additional args:
- `--base-checkpoint`: Override base checkpoint path (default: from config `lora.base_checkpoint`; reuses B2 pretrain).
- `--lora-checkpoint`: Override LoRA checkpoint path (for --phase tfb/laplace).

Reuses shared infrastructure: `experiment_setup.py`, `eval_utils.py`, `mlflow_utils.py`.

## 7. Configuration

### 7.1 `configs/b3_lora_agnews.yaml`

```yaml
experiment:
  name: b3-post-hoc-lora
  run_name: b3-lora-agnews
  mlflow_uri: "sqlite:///mlflow.db"

data:
  dataset: agnews
  id_categories: [2]
  ood_categories: [3, 4]
  val_fraction: 0.1
  test_fraction: 0.1

model:
  block_size: 256
  n_layer: 4
  n_head: 4
  n_embd: 256
  dropout: 0.2
  bias: true
  bayes_head:
    enabled: false
  bayes_ffn:
    enabled: false
  bayes_attn_v:
    enabled: false

lora:
  rank: 16
  alpha: 32.0
  target: ffn
  prior_std: 0.2       # unused for deterministic LoRA, kept for config compat
  init_g: 0.1          # unused for deterministic LoRA, kept for config compat
  base_checkpoint: "data/checkpoints/b2_pretrain/ckpt_best.pt"

train:
  steps: 10000
  batch_size: 32
  block_size: 256
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 500
  min_lr: 1.0e-5
  grad_clip: 1.0
  eval_interval: 500
  eval_iters: 20
  checkpoint_interval: 2000
  checkpoint_dir: "data/checkpoints/b3_lora"
  gradient_accumulation_steps: 1
  kl_weight: 0.0
  adam_beta1: 0.9
  adam_beta2: 0.95
  seed: 1337
  device: auto

tfb:
  epsilon: 0.1
  n_search_samples: 10
  n_anchor_batches: 20
  search_min: 1.0e-4
  search_max: 10.0
  search_precision: 1.0e-4

laplace:
  damping: 1.0
  sample_scale: 1.0
  n_curvature_batches: 20
  selection_mode: lora

eval:
  sample_tokens: 200
  temperature: 0.8
  num_samples: 20
  n_perplexity_batches: 20
  qualitative_prompts_per_category: 5
  qualitative_max_new_tokens: 100
  qualitative_seed: 42
```

## 8. Evaluation Protocol

Same as A2/B1/B2. No changes to evaluation methodology.

| Metric | Method | Notes |
|--------|--------|-------|
| Test ID perplexity | Deterministic forward (no sampling) | Baseline model quality |
| Test OOD perplexity | Deterministic forward (no sampling) | Higher than ID = good |
| MI (ID) | $N=20$ MC passes, sampled LoRA A | Stochasticity from posterior sampling |
| MI (OOD) | $N=20$ MC passes, sampled LoRA A | Should be higher than MI (ID) |
| MI ratio (batch) | $\mathrm{MI}_\text{OOD} / \mathrm{MI}_\text{ID}$ | Primary success metric |
| MI ratio (qualitative) | Per-prompt scoring, category averages | Curated prompts |
| Flip rate | Top-1 argmax disagreement | Sanity check |

**Key comparisons:**

| Method | Type | Bayesian params | MI ratio (batch) | MI ratio (qual) |
|--------|------|----------------|-----------------|-----------------|
| A2 | Variational, full | 4.2M | 1.43x | 1.70x |
| B1 | Post-hoc, full | 2.1M | 1.00x | 0.99x |
| B2 | Variational, LoRA | 163K | 1.13x | 1.02x |
| **B3-TFB** | Post-hoc, LoRA | ~82K | **?** | **?** |
| **B3-LAP** | Post-hoc, LoRA | ~82K | **?** | **?** |

## 9. Outputs

### New Files

| File | Description |
|------|-------------|
| `minigpt/tfb.py` | TFBState, fit_tfb, sample_tfb_params, compute_tfb_uncertainty, score_sequence_tfb, save/load |
| `configs/b3_lora_agnews.yaml` | B3 config (LoRA training + TFB + Laplace settings) |
| `experiments/b3_post_hoc_lora.py` | Three-phase experiment script |
| `tests/test_tfb.py` | Unit tests for TFB module |
| `tests/test_b3_deterministic_lora.py` | Unit tests for DeterministicLoRALinear |

### Modified Files

| File | Change |
|------|--------|
| `minigpt/lora.py` | Add `DeterministicLoRALinear`; extend `inject_lora` with `bayesian` param |
| `minigpt/laplace.py` | Add `"lora"` mode to `select_params` |
| `minigpt/config.py` | Add TFB config parsing; handle `bayesian` flag in LoRA config building |

### MLflow Logging

- **Phase 1 (train):** Standard training metrics + perplexity. Milestone tag: `b3-train`.
- **Phase 2a (TFB):** `tfb.sigma_q`, `tfb.epsilon`, `tfb.anchor_loss`, `tfb.fit_time_sec`, MI metrics. Milestone tag: `b3-tfb`.
- **Phase 2b (Laplace):** `laplace.damping`, `laplace.sample_scale`, curvature stats, MI metrics. Milestone tag: `b3-laplace`.

## 10. BDD — Behaviors

### B3-1: Deterministic LoRA training

- **Given** a pretrained base checkpoint and AG News category-split data.
- **When** training deterministic LoRA with CE loss (kl_weight=0).
- **Then** the model fine-tunes on ID data (Sports) and produces a checkpoint with trained `lora_A` and `lora_B` params. ID perplexity improves over the base model.

### B3-2: TFB variance search

- **Given** a deterministic LoRA checkpoint and anchor data.
- **When** running TFB binary search with tolerance $\varepsilon$.
- **Then** a scalar $\sigma_q > 0$ is found such that MC-sampled loss stays within $\varepsilon$ of MAP loss. The search converges (terminates in bounded iterations).

### B3-3: TFB posterior sampling

- **Given** a fitted TFBState.
- **When** sampling with fixed seed.
- **Then** samples are reproducible. Different seeds produce different A samples. The variance structure reflects B's singular values (larger singular values → less noise on corresponding A rows).

### B3-4: Laplace-LoRA curvature fitting

- **Given** a deterministic LoRA checkpoint.
- **When** fitting diagonal Fisher on LoRA A params.
- **Then** curvature tensors match LoRA A param shapes. Curvature values are finite and non-negative.

### B3-5: Laplace-LoRA posterior sampling

- **Given** a fitted LaplaceState on LoRA params.
- **When** sampling with fixed seed.
- **Then** samples are reproducible. Posterior variance = $s^2 / (F + \lambda)$.

### B3-6: MC uncertainty evaluation

- **Given** either TFB or Laplace posterior on LoRA A.
- **When** running MC evaluation (N=20 forward passes).
- **Then** MI, predictive entropy, expected entropy, and flip rate are computed for ID and OOD. Results are consistent with the standard evaluation protocol.

### B3-7: End-to-end pipeline

- **Given** configs and data.
- **When** running `experiments/b3_post_hoc_lora.py --phase full`.
- **Then** the full pipeline (train → TFB → Laplace → evaluate both) completes without errors.

## 11. TDD Plan (Stage #2)

### `tests/test_b3_deterministic_lora.py`

1. **`test_deterministic_lora_forward_matches_base_at_init`** — At initialization (B=0), DeterministicLoRALinear output equals base_linear output.
2. **`test_deterministic_lora_no_kl`** — DeterministicLoRALinear has no `kl_loss` method (it's not a BayesianModule).
3. **`test_deterministic_lora_all_params_trainable`** — `lora_A` and `lora_B` require grad; base_linear params don't.
4. **`test_deterministic_lora_forward_deterministic`** — Same input always produces same output (no sampling).
5. **`test_inject_lora_bayesian_false`** — `inject_lora(model, cfg, bayesian=False)` replaces FFN layers with `DeterministicLoRALinear`.
6. **`test_inject_lora_bayesian_false_freezes_base`** — After injection, only LoRA params have `requires_grad=True`.
7. **`test_deterministic_lora_param_count`** — Correct number of LoRA params for known dimensions.

### `tests/test_tfb.py`

8. **`test_tfb_svd_cache_shapes`** — SVD of B produces U, S, V with correct shapes per layer.
9. **`test_tfb_variance_structure`** — $\Omega_{ij} = \sigma_q / d_i$: larger singular values produce smaller variance.
10. **`test_tfb_sampling_reproducible`** — Same seed → identical A samples. Different seed → different samples.
11. **`test_tfb_zero_sigma_returns_map`** — $\sigma_q = 0$ returns exact $\mathbf{A}_\text{MAP}$.
12. **`test_tfb_search_converges`** — Binary search terminates within max iterations on a toy model.
13. **`test_tfb_search_respects_tolerance`** — Found $\sigma_q$ satisfies $|\text{loss}(\sigma_q) - \text{MAP loss}| \leq \varepsilon$.
14. **`test_tfb_state_save_load_roundtrip`** — Save TFBState → load → all fields match.
15. **`test_tfb_sampling_changes_logits`** — With $\sigma_q > 0$, different MC samples produce different logits.
16. **`test_tfb_mc_metrics_protocol`** — `compute_tfb_uncertainty` returns dict with `mi_mean`, `predictive_entropy_mean`, `expected_entropy_mean`, `flip_rate`.

### `tests/test_laplace.py` (additions to existing file)

17. **`test_select_params_lora_mode`** — `select_params(model, "lora")` returns only `lora_A` params from `DeterministicLoRALinear` layers.
18. **`test_laplace_on_lora_curvature_shapes`** — Curvature tensors match LoRA A shapes.
19. **`test_laplace_on_lora_sampling_changes_logits`** — Sampled LoRA A params produce different logits than MAP.

## 12. Acceptance Criteria

### Functional (must pass)

1. **Deterministic LoRA trains and checkpoints.** Phase 1 completes, checkpoint saved with trained `lora_A` and `lora_B`.
2. **TFB binary search converges.** `fit_tfb` returns a finite $\sigma_q > 0$ within bounded iterations.
3. **TFB posterior sampling is reproducible.** Fixed seed → identical samples.
4. **TFB variance reflects SVD structure.** Rows of $\boldsymbol{\Omega}$ corresponding to larger singular values of $\mathbf{B}$ have smaller variance.
5. **Laplace-LoRA curvature is non-trivial.** `select_params(model, "lora")` returns correct params; curvature is finite and non-negative.
6. **Laplace-LoRA sampling is reproducible.** Fixed seed → identical samples.
7. **MC evaluation produces MI metrics.** Both TFB and Laplace pipelines return MI, entropy, flip rate for ID and OOD.
8. **End-to-end script completes.** `--phase full` runs without errors on RTX 4070.
9. **All existing tests pass.** No regressions in the 84 existing tests.
10. **Checkpoint save/load roundtrip.** TFBState and LaplaceState serialize correctly.

### Scientific (expected, not hard-gated)

11. **Deterministic LoRA adapts.** Test ID ppl < base model ppl (fine-tuning works).
12. **TFB MI ratio > 1.0x.** Some OOD signal detected via TFB posterior.
13. **Laplace-LoRA curvature is more structured than B1.** LoRA params have higher and more varied curvature than B1's full FFN params (hypothesis: LoRA concentrates learning signal).
14. **TFB outperforms Laplace-LoRA.** TFB's SVD-informed variance should produce better OOD discrimination than uniform diagonal Laplace (hypothesis based on TFB paper results).

### Negative-Result Contingency

If both TFB and Laplace-LoRA give MI ratio ≈ 1.00x:
- This completes the 2×2 matrix with a symmetric result: post-hoc methods fail regardless of weight scope (full or LoRA).
- Combined with B1 (1.00x) and B2 (1.13x): **variational training is necessary for OOD-discriminative epistemic uncertainty in language models.**
- This is a strong, publishable conclusion for the comparative paper.

## 13. Hyperparameter Tuning Strategy

**TFB:** The primary tunable is $\varepsilon$ (tolerance). If $\sigma_q$ is too small (no MI signal), decrease $\varepsilon$. If too large (catastrophic noise), increase $\varepsilon$. Start with $\varepsilon = 0.1$.

**Laplace-LoRA:** If MI ratio = 1.00x (like B1), try:
- Lower damping (0.1, 0.01) to let curvature dominate.
- Sweep `sample_scale` (0.01–1.0).
- If curvature is still too flat, this confirms B1's finding extends to LoRA.

One variable at a time. Document each run in AGENTS.md.

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Deterministic LoRA checkpoint not saved correctly | B3 can't start post-hoc fitting | Test save/load roundtrip; verify checkpoint contains lora_A and lora_B |
| TFB binary search doesn't converge | No TFB result | Add max iteration limit; log search trajectory; fall back to grid search |
| B matrix near-singular (tiny singular values) | $\Omega$ blows up for small $d_i$ | Clamp singular values: $d_i \geq d_\text{min}$ (e.g., $10^{-6}$) |
| Laplace curvature too flat (like B1) | MI ratio 1.00x | Valid scientific finding; document and compare with B1 |
| LoRA training data too small for Fisher estimation | Noisy curvature | Use anchor data (larger) for TFB; use training data + val for Laplace |

## 15. References

- **TFB** (NeurIPS 2025) — Training-Free Bayesianization for LoRA. [arXiv:2412.05723](https://arxiv.org/abs/2412.05723)
- **Laplace-LoRA** (ICLR 2024) — Laplace approximation on LoRA params. [arXiv:2308.13111](https://arxiv.org/abs/2308.13111)
- **BLoB** (NeurIPS 2024) — Bayesian LoRA by Backpropagation. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- **ScalaBL** (UAI 2025) — Scalable Bayesian LoRA via subspace inference. [arXiv:2506.21408](https://arxiv.org/abs/2506.21408)
- **Laplace Redux** (NeurIPS 2021) — Effortless Bayesian Deep Learning. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)

## 16. Stage Boundary Checklist

Stage #1 (this doc):
- [x] What/why/how documented
- [x] Math + plain English for both approaches
- [x] BDD behaviors defined
- [x] TDD test plan specified
- [x] Acceptance criteria defined
- [x] Hyperparameter tuning strategy
- [x] Negative-result contingency

Stage #2 (next, separate):
- [ ] Write tests for non-existing B3 code
- [ ] Freeze tests with user sign-off

Stage #3 (after freeze):
- [ ] Implement B3 to satisfy frozen tests
