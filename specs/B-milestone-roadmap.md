# B Milestone Roadmap

This roadmap defines the next phase after A3 implementation. It is designed to keep momentum,
reduce risk, and stay aligned with practical Bayesian uncertainty methods for LLM adapters.

## 1. Why This Order

Use this sequence:

1. **B1: Post-hoc Laplace baseline (adapter-only)**
2. **B2: Flipout estimator in `BayesianLinear`**
3. **B3: Bayesian LoRA (diagonal posterior)**
4. **B4: Kronecker-factored curvature/covariance for adapters**

Rationale:
- Laplace first gives a strong Bayesian baseline without changing training dynamics.
- Flipout then improves VI stability for future Bayesian training runs.
- Bayesian LoRA starts with the simplest posterior (diagonal), then upgrades to structured
  approximations only if justified by results.

## 2. Core Concepts (Implementation Guidance)

## 2.1 Flipout (Variance Reduction for VI)

- Problem: reparameterized VI often uses one noisy weight sample shared across the mini-batch,
  causing high gradient variance.
- Mechanism: Flipout applies pseudo-independent sign perturbations per example to decorrelate
  gradients at near-single-sample cost.
- Expected effect: more stable optimization and less noisy learning curves for deep transformers.

## 2.2 Laplace Approximation (Post-hoc Bayesianization)

- Start from deterministic trained weights (MAP-like solution).
- Approximate local posterior as Gaussian around optimum using curvature:
  `q(theta) ~ N(theta_map, (H + lambda I)^-1)` where practical `H` is usually GGN/Fisher-style.
- Benefit: convert deterministic model/adapters to Bayesian uncertainty estimates with minimal
  training-loop risk.

## 2.3 Kronecker-Factored Approximations (K-FAC style)

- Full Hessian/Fisher is too large for LLM-scale params.
- Use layer/block factorizations (Kronecker structure) for adapter matrices:
  `C ~ A kron B` (or equivalent precision form).
- Benefit: tractable memory/compute for LoRA/adapters on consumer GPUs.

## 3. Repo-Level Milestones

## B1: Post-hoc Laplace Baseline (Adapter-Only)

Goal:
- Add a no-VI Bayesian baseline for uncertainty estimation on top of deterministic training.

Exact tasks:
1. Add new module: `minigpt/laplace.py`
   - curvature accumulation API (GGN/Fisher-style approximation for selected params).
   - damping/regularization handling.
   - posterior sampling from approximated Gaussian.
2. Add adapter-capable deterministic baseline path (minimal first step):
   - either lightweight LoRA module support in model path, or scoped selected linear params.
3. Add evaluation bridge:
   - reuse `minigpt/uncertainty.py` MI pipeline with sampled Laplace weights.
4. Add script:
   - `scripts/fit_laplace.py` (load deterministic checkpoint -> fit curvature stats -> save state).
5. Add experiment entry:
   - `experiments/b1_laplace_baseline.py`
6. Add config:
   - `configs/b1_laplace_agnews.yaml`
7. Add tests:
   - `tests/test_laplace.py` for curvature stats shape, damping behavior, sampling determinism under
     fixed seeds, and MI path compatibility.
8. Add MLflow logging fields:
   - `laplace.damping`, curvature stats summary, posterior sample scale, MI metrics.

Acceptance criteria:
1. End-to-end run completes from deterministic checkpoint -> Laplace fit -> MI eval.
2. OOD MI > ID MI on AG News split (same protocol as A2/A3).
3. Overhead acceptable on RTX 4070 (document fit time + eval time).
4. No training-loop changes required for the deterministic phase.

## B2: Flipout Estimator in BayesianLinear

Goal:
- Improve stability of VI-based Bayesian training (A3.2 and B1).

Exact tasks:
1. Extend `BayesianLinear` in `minigpt/layers.py`:
   - add estimator mode: `reparam` (current) | `flipout`.
   - implement sign-perturbation logic for batched inputs.
2. Extend Bayesian config surface:
   - add `estimator` option to model Bayesian component configs.
3. Wire estimator through model construction:
   - `minigpt/config.py`, `minigpt/model.py`.
4. Add targeted tests:
   - `tests/test_bayesian.py` or `tests/test_flipout.py`:
     - shape invariants
     - stochasticity invariants
     - finite gradients
     - deterministic behavior when seed is fixed.
5. Add one A3.2 config:
   - `configs/a3_2_agnews_flipout.yaml`
6. Add experiment entry (optional thin wrapper):
   - `experiments/a3_2_bayes_ffn_attn_v_flipout.py`

Acceptance criteria:
1. Flipout mode reproduces baseline metrics at minimum (no regression collapse).
2. Training variance across seeds is reduced vs reparameterization baseline (documented).
3. KL/ELBO trends are at least as stable as current A3 training.

## B3: Bayesian LoRA (Diagonal Posterior)

Goal:
- Move Bayesian uncertainty to open-weight LLM adapters with practical compute.

Exact tasks:
1. Add LoRA layers/module wrappers:
   - `minigpt/lora.py` (or new package submodule if cleaner).
2. Add Bayesian LoRA params:
   - posterior params for LoRA adapter weights (`mu`, `rho`).
3. Add training path:
   - experiment script `experiments/b1_bayes_lora.py`
   - config `configs/b1_bayes_lora.yaml`
4. Reuse uncertainty eval path:
   - MI/entropy/flip-rate for ID vs OOD.
5. Add tests:
   - `tests/test_bayes_lora.py`:
     - adapter injection correctness
     - KL aggregation
     - sampling invariants
     - checkpoint save/load integrity.

Acceptance criteria:
1. Trains on target hardware without OOM.
2. MI separation on ID/OOD is measurable and stable across at least 2 seeds.
3. Adapter-only Bayesian parameter count is within planned budget.

## B4: Kronecker-Factored Approximation Upgrade

Goal:
- Replace/augment diagonal posterior approximation for adapters with structured curvature.

Exact tasks:
1. Add structured curvature module:
   - `minigpt/curvature.py` (Kronecker stats per adapter block).
2. Add approximation mode configs:
   - `diag` | `kfac` for posterior approximation.
3. Add fitting/inference support:
   - extend `scripts/fit_laplace.py` and/or B1 training/eval scripts.
4. Add tests:
   - matrix shape/PSD checks
   - damping behavior
   - sampling consistency.

Acceptance criteria:
1. Memory footprint materially lower than naive dense covariance.
2. Runtime feasible on RTX 4070.
3. Equal or improved uncertainty quality vs diagonal baseline at similar cost.

## 4. Shared Evaluation and Tracking Requirements

Apply to B1 through B4:

1. Keep evaluation protocol consistent with A2/A3:
   - same ID/OOD split definition
   - same MI computation pipeline
   - same qualitative prompt-panel reporting.
2. Track these metrics in MLflow:
   - `mi_id_mean`, `mi_ood_mean`, `mi_ood_id_ratio`
   - `flip_rate_id`, `flip_rate_ood`
   - `test_id_perplexity`, `test_ood_perplexity`
   - calibration: ECE, NLL, Brier (sequence-level aggregation).
3. Add ranking metrics for uncertainty quality:
   - AUROC/AUPRC using sequence-level MI as OOD score.
4. Add cost metrics:
   - wall-clock train/eval time
   - tokens/sec
   - peak VRAM
   - Bayesian/adaptation parameter counts.

## 5. Decision Gates

Gate G1 (after B1):
- If Laplace baseline already provides strong and stable MI separation with low complexity,
  prioritize Laplace/K-FAC track for B1 productionization.

Gate G2 (after B2):
- If Flipout materially stabilizes VI runs, adopt Flipout as default for future VI experiments.

Gate G3 (after B3):
- If Bayesian LoRA diagonal is promising but under-calibrated or unstable, proceed to B4.
- If Bayesian LoRA diagonal underperforms Laplace baseline, pause and reassess VI complexity.

## 6. Risks and Mitigations

1. Curvature approximation instability:
   - use damping schedule and strict numerical checks.
2. Over-complexity before evidence:
   - enforce gate-based progression; no jump to K-FAC without B1/B3 signals.
3. Evaluation drift:
   - keep A2/A3 evaluation protocol fixed and versioned in configs/spec notes.

## 7. References to Align Implementation

- Yang et al., 2024, *Bayesian Low-Rank Adaptation for Large Language Models*:
  https://arxiv.org/abs/2308.13111
- Lin et al., 2026, *Bayesian-LoRA: Probabilistic Low-Rank Adaptation*:
  https://arxiv.org/abs/2601.21003

Use these for implementation details on adapter posterior structure and efficient approximations,
especially for K-FAC-like factorizations.
