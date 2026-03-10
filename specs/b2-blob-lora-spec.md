# B2: BLoB-Style Bayesian LoRA — Technical Specification

## 1. Overview

B2 implements Bayesian Low-Rank Adaptation by Backpropagation (BLoB, NeurIPS 2024) on MiniGPT.

The core idea: freeze a pretrained deterministic model, inject low-rank adapters into FFN layers,
and train those adapters with variational inference. The Bayesian uncertainty in the adapter
parameters captures epistemic uncertainty about domain adaptation from general text to news.

**Two-phase pipeline:**
1. **Pretrain** miniGPT on TinyShakespeare (general domain, deterministic).
2. **LoRA fine-tune** on AG News ID topics with BLoB variational inference.

Evaluate MI on ID vs OOD, same protocol as A2 and B1.

## 2. Goals

1. Complete the variational+LoRA cell of the 2x2 comparative matrix.
2. Test whether Bayesian LoRA produces OOD-discriminative uncertainty (MI ratio > 1.0x)
   with orders of magnitude fewer Bayesian parameters than A2 (estimated ~16K vs 4.2M).
3. Compare against A2 (variational, full weights, MI ratio 1.43x) on the same evaluation protocol.
4. Provide the trained-LoRA checkpoint that B3 (TFB/Laplace-LoRA) will use for post-hoc
   Bayesianization.

## 3. Background: BLoB Algorithm

### 3.1 Standard LoRA

LoRA decomposes the weight update as:

$$\mathbf{h} = \left(\mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}\right) \mathbf{x}$$

- $\mathbf{W}_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$: frozen pretrained weight.
- $\mathbf{B} \in \mathbb{R}^{d_\text{out} \times r}$: low-rank down-projection. Initialized to zeros.
- $\mathbf{A} \in \mathbb{R}^{r \times d_\text{in}}$: low-rank up-projection. Initialized with Kaiming uniform.
- $\alpha / r$: scaling factor. $\alpha = 2r$ is standard (scale $= 2.0$).
- Only $\mathbf{B}$ and $\mathbf{A}$ are trained; $\mathbf{W}_0$ is frozen.
- At initialization, $\mathbf{B} = 0$ so LoRA contribution is zero — model starts as the pretrained model.

### 3.2 BLoB Asymmetric Design

BLoB Bayesianizes **only the A matrix**; B remains deterministic.

**Rationale (BLoB Theorem 3.1):** If both $\mathbf{A}$ and $\mathbf{B}$ are stochastic,
$\mathbb{E}[\mathbf{B}\mathbf{A}] \neq \mathbb{E}[\mathbf{B}]\,\mathbb{E}[\mathbf{A}]$,
which makes the mean of the weight update not equal to the product of means. With $\mathbf{B}$
initialized at zeros, the posterior mean would be stuck at zero early in training. Fixing
$\mathbf{B}$ as deterministic avoids this and halves the Bayesian parameter overhead.

**Posterior on A:**

$$q(\mathbf{A} \mid \boldsymbol{\theta}) = \prod_{i,j} \mathcal{N}\!\left(A_{ij} \mid M_{ij},\; \sigma_{ij}^2\right)$$

- $\mathbf{M} \in \mathbb{R}^{r \times d_\text{in}}$: posterior mean matrix. Trainable. Init: Kaiming uniform.
- $\mathbf{G} \in \mathbb{R}^{r \times d_\text{in}}$: variance parameter, $\sigma_{ij} = G_{ij}^2$. Trainable. Init: $\mathcal{U}(\varepsilon/\sqrt{2},\; \varepsilon)$.
- The $G^2$ parameterization (not softplus) is the BLoB convention, providing stronger
  initial gradients for faster variance learning.

**Sampling (reparameterization trick):**

$$\hat{\mathbf{A}} = \mathbf{M} + \mathbf{G}^2 \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

$$\mathbf{h} = \left(\mathbf{W}_0 + \frac{\alpha}{r}\,\mathbf{B}\,\hat{\mathbf{A}}\right) \mathbf{x}$$

No Flipout in B2. Standard reparameterization is sufficient for 4L miniGPT with batch_size=32.
Flipout is reserved for C milestone (16L, batch_size=8).

### 3.3 KL Computation (BLoB Theorem 3.2)

The low-rank posterior on $\mathbf{A}$ induces a full-weight posterior on $\mathbf{W}$. BLoB
proves that the KL in full weight space reduces to a cheap diagonal KL in A-space:

$$\mathrm{KL}\!\left[q(\mathbf{A}\mid\boldsymbol{\theta})\;\|\;p(\mathbf{A})\right] = \sum_{i,j}\left[\frac{M_{ij}^2 + \sigma_{ij}^2}{2\sigma_p^2} - \ln\frac{\sigma_{ij}}{\sigma_p} - \frac{1}{2}\right]$$

where $p(\mathbf{A}) = \mathcal{N}(\mathbf{0},\; \sigma_p^2\,\mathbf{I})$ is the prior on A elements.

This is the same form as our existing `BayesianLinear` KL, just with $G^2$ parameterization
instead of $\text{softplus}(\rho)$.

### 3.4 ELBO Objective

$$\mathcal{L} = \mathrm{CE}\!\left(\mathcal{D} \mid \hat{\mathbf{A}}, \mathbf{B}\right) + \frac{\lambda_\text{kl}}{N_\text{tokens}}\;\mathrm{KL}\!\left[q(\mathbf{A}\mid\boldsymbol{\theta})\;\|\;p(\mathbf{A})\right]$$

Same ELBO structure as A2. The existing training loop handles this via `kl_weight` ($\lambda_\text{kl}$)
and `model.kl_loss()`.

## 4. Architecture

### 4.1 New Module: `minigpt/lora.py`

#### `LoRAConfig` dataclass

```python
rank: int = 8               # LoRA rank r
alpha: float = 16.0         # scaling factor (effective scale = alpha / rank)
target: str = "ffn"         # which layers to inject: "ffn"
prior_std: float = 0.2      # prior std on A elements (sigma_p)
init_g: float = 0.05        # G init range: U(init_g/sqrt(2), init_g)
```

#### `BLoBLoRALinear(BayesianModule)`

The core new layer. Wraps a frozen `nn.Linear` and adds a Bayesian LoRA adapter.

**Parameters:**
- `base_linear` (nn.Linear): frozen pretrained weight. Not a parameter — stored but not trained.
- `lora_B` (nn.Parameter): $\mathbb{R}^{d_\text{out} \times r}$. Deterministic, trainable. Init: zeros.
- `lora_A_mu` (nn.Parameter): $\mathbb{R}^{r \times d_\text{in}}$. Posterior mean. Init: Kaiming uniform.
- `lora_A_g` (nn.Parameter): $\mathbb{R}^{r \times d_\text{in}}$. Variance param. Init: $\mathcal{U}(\texttt{init\_g}/\sqrt{2},\; \texttt{init\_g})$.
  $\sigma = \texttt{lora\_A\_g}^{\,2}$.
- `scaling` (float): $\alpha / r$.

**Forward pass:**

$$\hat{\mathbf{A}} = \mathbf{M} + \mathbf{G}^2 \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

$$\mathbf{y} = \mathbf{W}_0\,\mathbf{x} + \frac{\alpha}{r}\;\mathbf{B}\,\hat{\mathbf{A}}\,\mathbf{x}$$

```python
base_out = base_linear(x)                           # frozen, no grad
A_sample = lora_A_mu + lora_A_g**2 * epsilon        # reparameterization
lora_out = (x @ A_sample.T) @ lora_B.T * scaling    # B @ A_sample @ x
return base_out + lora_out
```

When `_use_mean` is True (deterministic eval): use `lora_A_mu` directly, no sampling.
When `_frozen_weight` is set (MC eval): use cached A sample for coherent multi-token scoring.

**`kl_loss()` method:**
Diagonal KL between $q(\mathbf{A}) = \mathcal{N}(\mathbf{M},\;\mathrm{diag}(\mathbf{G}^4))$
and $p(\mathbf{A}) = \mathcal{N}(\mathbf{0},\;\sigma_p^2\,\mathbf{I})$.
Same formula as Section 3.3.

**`freeze_sample()` / `unfreeze_sample()`:**
Sample A once and cache for coherent forward passes during MC evaluation.
Same protocol as BayesianLinear — required for sequence scoring.

**`mean_forward(x)`:**
Forward using posterior mean only (no sampling). For deterministic evaluation.

#### `inject_lora(model, lora_config) -> MiniGPT`

Post-construction injection function:
1. **Freeze** all base model parameters (`requires_grad = False`).
2. **Find** target linear layers in MLP blocks (MLP.fc and MLP.proj).
3. **Replace** each target `DeterministicLinear` with a `BLoBLoRALinear` wrapper
   that takes ownership of the original `nn.Linear`.
4. Return the modified model. Only LoRA parameters have `requires_grad = True`.

Target mapping for `target="ffn"`:
- `blocks[i].mlp.fc` -> BLoBLoRALinear wrapping fc.linear
- `blocks[i].mlp.proj` -> BLoBLoRALinear wrapping proj.linear

Total LoRA adapters: 2 per block x 4 blocks = 8 adapters.

#### Parameter count estimate ($r = 8$, $d_\text{embd} = 256$)

Per adapter (e.g., MLP.fc: $256 \to 1024$):
- `lora_B`: $1024 \times 8 = 8{,}192$ (deterministic)
- `lora_A_mu`: $8 \times 256 = 2{,}048$ (Bayesian mean)
- `lora_A_g`: $8 \times 256 = 2{,}048$ (Bayesian variance)
- Total per adapter: $12{,}288$

Per adapter (e.g., MLP.proj: $1024 \to 256$):
- `lora_B`: $256 \times 8 = 2{,}048$
- `lora_A_mu`: $8 \times 1024 = 8{,}192$
- `lora_A_g`: $8 \times 1024 = 8{,}192$
- Total per adapter: $18{,}432$

$4 \text{ blocks} \times (12{,}288 + 18{,}432) = \mathbf{122{,}880}$ **total LoRA params**.
Of those, Bayesian params ($\mathbf{M}$ + $\mathbf{G}$): $\mathbf{81{,}920}$ (~82K).
Compared to A2: 4.2M Bayesian params. **~51x parameter reduction.**

### 4.2 Integration with Existing Infrastructure

#### `sum_kl_loss` (layers.py)
Already walks all `BayesianModule` children. Since `BLoBLoRALinear` extends `BayesianModule`,
KL aggregation works without changes.

#### `_has_bayesian_body` (uncertainty.py)
Currently checks for `BayesianLinear` instances only. Must be extended to also detect
`BLoBLoRALinear` (or any stochastic `BayesianModule` subclass) in the model body.
This ensures the MC evaluation uses the full forward pass path (N full passes),
not the head-only shortcut.

#### `frozen_bayesian_sample` / `use_mean_weights` (layers.py)
Currently operate on `BayesianLinear` instances only. Must be extended to also cover
`BLoBLoRALinear` layers (freeze_sample, unfreeze_sample, _use_mean).

Alternative: refactor these context managers to operate on `BayesianModule` with a
`has_stochastic_forward` property, so new stochastic module types are handled automatically.

#### `sigma_summary` (layers.py)
Currently collects $\text{softplus}(\rho)$ from `BayesianLinear`. Must be extended to
also collect $G^2$ from `BLoBLoRALinear` layers for variance diagnostics.

#### `_configure_optimizer` (train.py)
Filters `requires_grad` — works correctly since base model is frozen. For Phase 2:
- `lora_B`: 2D, gets weight_decay.
- `lora_A_mu`: 2D, gets weight_decay.
- `lora_A_g`: variance param, should get **no weight_decay** (KL regularizes).
  Naming convention `_g` must be added to the no-decay filter (like `_rho`).

Alternative: Phase 2 config sets `weight_decay=0.0` globally (BLoB paper convention:
KL is the only regularizer). This avoids optimizer surgery.

#### `train()` (train.py)
Reused directly for both phases. Phase 1: deterministic (kl_weight=0).
Phase 2: Bayesian LoRA (kl_weight>0, only LoRA params trainable).

#### Eval utilities (eval_utils.py)
`eval_perplexity_suite`, `eval_mi_suite`, `run_qualitative_suite` — all reused as-is.
Perplexity eval uses `use_mean_weights` context (must cover LoRA layers).

## 5. Two-Phase Pipeline

### 5.1 Phase 1: Pretrain on TinyShakespeare

**Goal:** Train a deterministic miniGPT on general-domain text. This becomes the frozen base
model for LoRA fine-tuning.

**Config:** `configs/b2_pretrain_shakespeare.yaml`

Inherits B1 training hyperparams (validated, well-tuned):
- Architecture: 4L/4H/256d, deterministic (all bayes_* disabled)
- Dataset: TinyShakespeare
- Training: lr=3e-4, weight_decay=0.1, warmup=1000, batch_size=32, block_size=256
- Steps: **10,000–20,000** (TinyShakespeare is ~304K tokens; 100K steps would be ~2700 epochs)
- Checkpoint: saves to `data/checkpoints/` (reusable by Phase 2 and B3)

**Output:** A deterministic checkpoint (`ckpt_best.pt`) trained on general-domain English.
The model should show reasonable val loss (not overfitting severely). Exact perplexity targets
are not gated — the pretrained model just needs to be a competent language model.

**Evaluation:** Basic val/test perplexity logged. No MI evaluation (deterministic model).

### 5.2 Phase 2: BLoB LoRA Fine-Tune on AG News

**Goal:** Fine-tune the pretrained model on AG News ID topics with Bayesian LoRA.
The Bayesian uncertainty in LoRA params should discriminate ID from OOD inputs.

**Config:** `configs/b2_blob_agnews.yaml`

Key settings:
- Dataset: AG News (ID: World+Sports, OOD: Business+Sci/Tech)
- Base checkpoint: path to Phase 1 checkpoint
- LoRA: rank=8, alpha=16, target=ffn, prior_std=0.2, init_g=0.05
- Training: lr=1e-4 (lower than pretrain — fine-tuning), weight_decay=0.0 (KL regularizes),
  kl_weight>0 (e.g., 0.2, tunable), steps TBD (likely 5K–20K for fine-tuning)
- Eval: same protocol as A2 (MI, perplexity, qualitative)

**Pipeline:**
1. Load pretrained checkpoint from Phase 1.
2. Inject BLoB LoRA adapters into FFN layers via `inject_lora()`.
3. Verify: all base params frozen, only LoRA params trainable.
4. Train with ELBO ($\mathcal{L} = \mathrm{CE} + \lambda_\text{kl} \cdot \mathrm{KL} / N_\text{tokens}$).
5. Save best checkpoint (ELBO criterion).
6. Evaluate: perplexity (ID/OOD), MI (ID/OOD), qualitative prompt panel.
7. Log everything to MLflow.

### 5.3 Experiment Script: `experiments/b2_blob_lora.py`

Single script that orchestrates both phases:
- `--phase pretrain`: Run Phase 1 only.
- `--phase finetune`: Run Phase 2 only (requires `--base-checkpoint`).
- `--phase full`: Run Phase 1 then Phase 2 (default).
- `--skip-pretrain`: Alias for `--phase finetune` (with auto-detected checkpoint path from config).

Reuses shared infrastructure: `experiment_setup.py`, `eval_utils.py`, `mlflow_utils.py`.

Phase 2 evaluation flow mirrors `runner.py`:
1. Perplexity suite (mean weights).
2. Generated sample.
3. MI suite (MC sampling).
4. Qualitative prompt panel.
5. Sigma summary (LoRA $G^2$ stats).
6. Final KL.

## 6. Configuration

### 6.1 `configs/b2_pretrain_shakespeare.yaml`

```yaml
experiment:
  name: b2-pretrain
  run_name: b2-pretrain-shakespeare
  mlflow_uri: "sqlite:///mlflow.db"

data:
  dataset: tinyshakespeare
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

train:
  steps: 15000
  batch_size: 32
  block_size: 256
  lr: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 1000
  min_lr: 1.0e-5
  grad_clip: 1.0
  eval_interval: 1000
  eval_iters: 20
  checkpoint_interval: 5000
  checkpoint_dir: "data/checkpoints/b2_pretrain"
  gradient_accumulation_steps: 1
  kl_weight: 0.0
  adam_beta1: 0.9
  adam_beta2: 0.95
  seed: 1337
  device: auto

eval:
  sample_tokens: 200
  temperature: 0.8
```

### 6.2 `configs/b2_blob_agnews.yaml`

```yaml
experiment:
  name: b2-blob
  run_name: b2-blob-agnews
  mlflow_uri: "sqlite:///mlflow.db"

data:
  dataset: agnews
  id_categories: [1, 2]
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
  rank: 8
  alpha: 16.0
  target: ffn
  prior_std: 0.2
  init_g: 0.05
  base_checkpoint: "data/checkpoints/b2_pretrain/ckpt_best.pt"

train:
  steps: 10000
  batch_size: 32
  block_size: 256
  lr: 1.0e-4
  weight_decay: 0.0
  warmup_steps: 500
  min_lr: 1.0e-5
  grad_clip: 1.0
  eval_interval: 500
  eval_iters: 20
  checkpoint_interval: 2000
  checkpoint_dir: "data/checkpoints/b2_blob"
  gradient_accumulation_steps: 1
  kl_weight: 0.2
  adam_beta1: 0.9
  adam_beta2: 0.95
  seed: 1337
  device: auto

eval:
  sample_tokens: 200
  temperature: 0.8
  num_samples: 20
  n_perplexity_batches: 20
  qualitative_prompts_per_category: 5
  qualitative_max_new_tokens: 100
  qualitative_seed: 42
```

## 7. Evaluation Protocol

Same as A2, B1. No changes to evaluation methodology — this is critical for controlled comparison.

| Metric | Method | Notes |
|---|---|---|
| Test ID perplexity | Mean weights (no sampling) | Should be lower than A2 (pretrain+finetune advantage) |
| Test OOD perplexity | Mean weights (no sampling) | Higher than ID = good |
| MI (ID) | $N=20$ MC passes, full forward | Stochasticity from LoRA $\mathbf{A}$ sampling |
| MI (OOD) | $N=20$ MC passes, full forward | Should be higher than MI (ID) |
| MI ratio (batch) | $\mathrm{MI}_\text{OOD} / \mathrm{MI}_\text{ID}$ | Primary success metric |
| MI ratio (qualitative) | Per-prompt scoring, category averages | Curated prompts |
| Flip rate | Top-1 argmax disagreement across MC samples | Sanity check |
| Sigma summary | $G^2$ statistics across all LoRA $\mathbf{A}$ params | Posterior learning diagnostic |
| KL trajectory | Per-eval-step KL values | Should decrease or stabilize (healthy) |

**Key comparisons:**
- B2 MI ratio vs A2 MI ratio (1.43x batch / 1.70x qualitative) — within same order?
- B2 MI ratio vs B1 MI ratio (1.00x) — LoRA variational vs full-weight post-hoc.
- B2 Bayesian param count (~82K) vs A2 (4.2M) — parameter efficiency.
- B2 training cost (fine-tune only) vs A2 (full training from scratch).

## 8. Outputs

### New Files
| File | Description |
|---|---|
| `minigpt/lora.py` | LoRAConfig, BLoBLoRALinear, inject_lora(), LoRA utility functions |
| `configs/b2_pretrain_shakespeare.yaml` | Phase 1 config |
| `configs/b2_blob_agnews.yaml` | Phase 2 config |
| `experiments/b2_blob_lora.py` | Two-phase experiment script |
| `tests/test_lora.py` | Unit tests for LoRA module |

### Modified Files
| File | Change |
|---|---|
| `minigpt/layers.py` | Extend context managers and sigma_summary for BLoBLoRALinear |
| `minigpt/uncertainty.py` | Extend `_has_bayesian_body` to detect LoRA layers |
| `minigpt/config.py` | Add LoRA config parsing and validation |
| `minigpt/train.py` | Extend `_configure_optimizer` to handle LoRA variance params |

### MLflow Logging (Phase 2)
- All standard metrics (train/val loss, perplexity, LR, KL)
- LoRA-specific: rank, alpha, prior_std, init_g, base_checkpoint path
- LoRA param count (total, Bayesian)
- Sigma summary (G^2 statistics)
- MI metrics (ID, OOD, ratio)
- Qualitative report
- Milestone tag: `b2`

## 9. Acceptance Criteria

### Functional (must pass)

1. **End-to-end pipeline completes.** Phase 1 pretrain -> Phase 2 LoRA fine-tune -> MI evaluation
   runs without errors on RTX 4070.
2. **Base model is fully frozen.** After `inject_lora()`, zero base model parameters have
   `requires_grad=True`. Only LoRA parameters are trainable.
3. **LoRA initialization preserves model.** At initialization (B=0), the model output is
   identical to the pretrained base model (forward pass produces same logits).
4. **KL is correct.** `model.kl_loss()` returns the sum of diagonal KL terms from all
   `BLoBLoRALinear` layers. KL is zero when $\mathbf{M} = 0$ and $\sigma = \sigma_p$.
5. **MC sampling produces variation.** With LoRA Bayesian params, different MC samples produce
   different logits (stochasticity comes from $\mathbf{A}$ matrix sampling).
6. **Checkpoint save/load roundtrip.** Save LoRA model state -> load -> identical forward pass
   (both base weights and LoRA params restored correctly).
7. **Perplexity evaluation uses mean weights.** `use_mean_weights` context manager correctly
   disables sampling in BLoBLoRALinear layers.
8. **Sigma summary reports $G^2$ statistics.** `sigma_summary()` includes LoRA variance params.
9. **All existing tests pass.** No regressions in the 56 existing tests.

### Scientific (expected, not hard-gated)

10. **Phase 1:** Pretrained model achieves reasonable TinyShakespeare perplexity (val loss
    converges, no severe overfitting).
11. **Phase 2:** LoRA fine-tuning reduces AG News ID perplexity below the pretrained model's
    ID perplexity (adaptation is working).
12. **MI ratio > 1.0x.** OOD MI is measurably higher than ID MI (epistemic signal exists
    in LoRA params). This is the primary scientific hypothesis.
13. **Sigma statistics show learned posteriors.** $G^2$ values are not collapsed to near-zero
    (posterior collapse) or blown up uniformly (noise). Range should show differentiation
    (some params more uncertain than others).
14. **KL trajectory is healthy.** Decreasing or stable over training, not pathologically rising.

### Negative-Result Contingency

If MI ratio = 1.00x (like B1):
- This is still a valid scientific result for the comparative study.
- Document the finding: "Bayesian LoRA variational inference on ~82K adapter params does not
  produce OOD-discriminative uncertainty on 4L miniGPT / AG News."
- Investigate: is the LoRA rank too small? Is the adaptation too easy (val loss drops fast)?
  Are posteriors collapsed?
- Proceed to B3 regardless — the trained LoRA checkpoint is still needed.

## 10. Hyperparameter Tuning Strategy

For the first run, use BLoB paper defaults adapted to our scale:

| Parameter | Value | Rationale |
|---|---|---|
| rank | 8 | BLoB default. Reasonable for n_embd=256. |
| alpha | 16 | scale=2.0, standard LoRA convention |
| prior_std | 0.2 | BLoB default (bayes_beta) |
| init_g | 0.05 | BLoB default (bayes_eps). Gives initial $\sigma \approx 0.0025$ |
| kl_weight | 0.2 | Same as A2 (our validated value) |
| lr | 1e-4 | Lower than pretrain (fine-tuning convention) |
| weight_decay | 0.0 | BLoB convention (KL regularizes) |
| steps | 10,000 | Start here, adjust based on val loss curve |

If first run shows issues:
- **Posterior collapse ($\sigma \to 0$):** Increase init_g (0.1, 0.2). Decrease kl_weight.
- **Too much noise (MI high but equal for ID/OOD):** Decrease init_g. Increase kl_weight.
- **Slow adaptation:** Increase lr (3e-4). Increase rank (16).
- **Overfitting:** Decrease steps. Add dropout to LoRA (not standard, last resort).

One variable at a time. Document each run in AGENTS.md.

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LoRA rank too small for 256d model | No meaningful adaptation, trivial corrections | Try rank 4, 8, 16. With $d_\text{embd}=256$, even rank 4 captures ~1.5% of the weight space. |
| TinyShakespeare pretrain overfits | Base model memorizes Shakespeare, poor generalization to news | Monitor val loss curve. Stop early if val loss diverges from train loss. 15K steps is conservative. |
| Bayesian params too few for MI signal | Insufficient stochasticity for MI discrimination | LoRA A params are in the critical FFN pathway. Even ~82K params should create measurable variation. Compare flip rate to A2. |
| $G^2$ parameterization numerical issues | $G$ values too small $\to$ $\sigma$ vanishes; too large $\to$ $\sigma$ explodes | Init in $[0.035, 0.05]$ range (BLoB default). Monitor sigma summary. Clamp if needed. |
| Integration breaks existing tests | LoRA code changes affect A-series behavior | Run full test suite after every change. Existing BayesianLinear behavior must not change. |

## 12. References

- **BLoB** (NeurIPS 2024) — Bayesian Low-Rank Adaptation by Backpropagation.
  [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- **LoRA** (ICLR 2022) — Low-Rank Adaptation of Large Language Models.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Wang-ML-Lab bayesian-peft** — Reference implementation.
  [GitHub](https://github.com/Wang-ML-Lab/bayesian-peft)
