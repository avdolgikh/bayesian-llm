# B1 Technical Spec - Post-hoc Laplace Baseline (Stage #1)

Status: Draft for implementation planning  
Date: 2026-03-05  
Process mode: BDD -> TDD -> Code  
Current stage: Stage #1 (spec only; no tests or code changes in this stage)

## 1. What This Milestone Is

### B1 in one sentence
B1 adds Bayesian uncertainty to a trained deterministic model *after training* by fitting a Laplace approximation around the learned weights, then running uncertainty evaluation via parameter sampling.

### Why B1 now
A0/A1/A2 are complete, and A3 is closed. The next priority is scalable uncertainty methods for adapter-style setups. Laplace is the safest first B-track step because:
- It does not change base training dynamics.
- It gives a strong Bayesian baseline quickly.
- It provides a direct comparison target for later Bayesian LoRA (B3).

This follows `specs/B-milestone-roadmap.md` order:
1. B1 Laplace baseline
2. B2 Flipout
3. B3 Bayesian LoRA (diagonal)
4. B4 Kronecker-factored upgrade

## 2. Explanation

Think of the trained model as a point on a landscape:
- Deterministic training gives one "best point" (best weights).
- Laplace asks: "How flat or sharp is the landscape around this point?"
- If the area is flat, nearby weights are also plausible.
- If sharp, only very nearby weights are plausible.

B1 turns that local shape into a Gaussian distribution over selected weights. Then we:
1. Sample several nearby weight versions.
2. Run prediction multiple times.
3. Measure disagreement (MI) between predictions.

High disagreement on OOD data means the model "knows it does not know."

## 3. Mathematical Idea

## 3.1 Setup
Let model parameters be split into:
- `psi`: frozen deterministic backbone parameters
- `phi`: selected parameters for Laplace posterior (adapter-only target in B1)

We start from a deterministic checkpoint with MAP-like estimate `phi_hat`.

## 3.2 Local quadratic approximation
Around `phi_hat`, approximate negative log posterior by a quadratic:

`-log p(phi | D) ~= const + 0.5 * (phi - phi_hat)^T * H * (phi - phi_hat)`

where `H` is a curvature approximation (GGN/Fisher-style, damped).

This yields Gaussian posterior:

`q(phi) = N(phi_hat, (H + lambda I)^-1)`

- `lambda > 0` is damping for numerical stability.
- B1 first version uses tractable structure (start diagonal over selected parameters).

## 3.3 Predictive uncertainty
For input `x`, Bayesian predictive:

`p(y | x, D) = integral p(y | x, phi, psi) q(phi) dphi`

Approximate with Monte Carlo samples `phi^(i) ~ q(phi)`:

`p_bar = (1/N) * sum_i softmax(logits_i)`

Then compute:
- Predictive entropy `H[p_bar]`
- Expected entropy `(1/N) * sum_i H[p_i]`
- Mutual information `MI = H[p_bar] - E[H[p_i]]`

This MI is the epistemic signal used in A-milestones too, so protocol stays comparable.

## 4. Why Laplace (vs VI first)

- Lower risk: no ELBO/KL instability in base training loop.
- Faster baseline: fit curvature from checkpoint, then sample.
- Better milestone control: isolates Bayesianization from training noise.
- Direct bridge to later structured curvature (B4).

## 5. Papers and References

Primary references for B1 track:
- Laplace-LoRA (ICLR 2024): https://arxiv.org/pdf/2308.13111
- Training-Free Bayesianization for LoRA (2024): https://arxiv.org/pdf/2412.05723
- BLoB (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/7d53575463291ea6b5a23cf6e571f59b-Paper-Conference.pdf
- ScalaBL (2025): https://arxiv.org/pdf/2506.21408
- Internal roadmap: `specs/B-milestone-roadmap.md`

## 6. Scope for B1

In scope:
- Post-hoc Laplace over selected parameters (`phi`) only.
- End-to-end pipeline: checkpoint -> curvature fit -> posterior sampling -> MI evaluation.
- MLflow tracking for Laplace-specific metrics.
- Reuse existing `minigpt/uncertainty.py` metrics protocol.

Out of scope:
- Full dense Hessian/covariance.
- Full Bayesian retraining of model.
- Flipout estimator changes (belongs to B2).
- Bayesian LoRA posterior learning (belongs to B3).
- Kronecker-factored posterior for adapters (belongs to B4).

## 7. BDD (Behavior-Driven Design) for B1

## 7.1 User stories

1. As a researcher, I can load a deterministic checkpoint and fit Laplace stats on selected params.
2. As a researcher, I can sample posterior parameter draws and run MI ID-vs-OOD evaluation.
3. As a researcher, I can compare B1 uncertainty quality and cost against A2/A3.

## 7.2 Core behaviors

Behavior B1-1: Curvature fitting
- Given checkpoint + config + selected params
- When fitting Laplace stats
- Then curvature summaries are saved and valid (finite, non-negative in expected forms)

Behavior B1-2: Posterior sampling
- Given fitted Laplace state
- When sampling with fixed seed
- Then samples are deterministic/reproducible
- And sample scale tracks configured damping/temperature

Behavior B1-3: Uncertainty evaluation
- Given Laplace posterior and AG News split
- When running MC uncertainty eval
- Then MI metrics are produced for ID and OOD
- And run logs include MI ratio and cost metrics

Behavior B1-4: End-to-end script
- Given deterministic checkpoint
- When running `experiments/b1_laplace_baseline.py`
- Then full pipeline completes without manual patching

## 8. TDD Plan (Stage #2, tests first; freeze before code)

Stage #2 will add tests for non-existing code and freeze them before implementation.

Planned test files:
- `tests/test_laplace.py`
- (optional) `tests/test_b1_pipeline.py` for wiring/integration smoke

Planned test cases:

1. `test_curvature_stats_shapes_match_selected_params`
- Curvature tensors/shapes align exactly with selected parameter tensors.

2. `test_damping_makes_posterior_finite_and_stable`
- With damping > 0, posterior variance is finite and positive.

3. `test_sampling_reproducible_with_fixed_seed`
- Same seed -> identical sampled tensors.
- Different seed -> different samples.

4. `test_zero_sample_scale_returns_map_params`
- Sampling scale = 0 returns exact MAP (`phi_hat`) for selected params.

5. `test_laplace_sampling_changes_logits`
- At positive sample scale, repeated forward passes produce non-identical logits.

6. `test_mi_pipeline_accepts_laplace_sampler`
- Existing uncertainty pipeline can run with Laplace-sampled params.

7. `test_checkpoint_roundtrip_laplace_state`
- Save/load of Laplace state preserves all required tensors/metadata.

8. `test_param_selection_scope_is_respected`
- Only selected params are Bayesianized; non-selected params remain unchanged.

Freeze rule:
- After Stage #2 tests are created and reviewed, no behavior-changing edits to tests during Stage #3 except explicit user-approved corrections.

## 9. Code Implementation Blueprint (Stage #3 target)

## 9.1 New module: `minigpt/laplace.py`

Proposed responsibilities:
- Parameter selection utilities for `phi`.
- Curvature accumulation (GGN/Fisher-style approximation).
- Damping and posterior scale handling.
- Sampling API for selected params.
- Serialization of Laplace state.

Proposed data structures:
- `LaplaceConfig`: damping, sample_scale, selection mode, data pass config.
- `LaplaceState`: param names, `phi_hat`, curvature stats, damping metadata.

Proposed APIs (exact signatures can adjust during Stage #3):
- `fit_laplace(model, data, selection, cfg) -> LaplaceState`
- `sample_laplace_params(state, seed=None) -> dict[name, tensor]`
- `apply_sampled_params(model, sampled_params)` context manager
- `save_laplace_state(state, path)` / `load_laplace_state(path)`

## 9.2 New script: `scripts/fit_laplace.py`

CLI intent:
- Input: checkpoint + config + selection + damping options.
- Output: saved Laplace state artifact.

Example flow:
1. Load checkpoint/model config.
2. Select target params.
3. Run curvature accumulation pass(es).
4. Save state file under `data/checkpoints/` or configured path.
5. Log summaries to MLflow if enabled.

## 9.3 New experiment: `experiments/b1_laplace_baseline.py`

Pipeline:
1. Load deterministic checkpoint.
2. Fit/load Laplace state.
3. Evaluate perplexity and MI (ID/OOD) with posterior sampling.
4. Run qualitative MI panel.
5. Log all metrics and artifacts to MLflow.

## 9.4 New config: `configs/b1_laplace_agnews.yaml`

Required fields:
- Deterministic checkpoint path
- Laplace param selection mode
- Damping
- Number of curvature batches/passes
- Sampling count `N` for uncertainty
- MLflow tags and run name

## 9.5 MLflow fields for B1

Add/track:
- `laplace.damping`
- `laplace.selection_mode`
- `laplace.num_selected_params`
- curvature summaries (min/mean/max or per-block summary)
- `laplace.sample_scale`
- Standard uncertainty metrics:
  - `mi_id_mean`, `mi_ood_mean`, `mi_ood_id_ratio`
  - `flip_rate_id`, `flip_rate_ood`
  - `test_id_perplexity`, `test_ood_perplexity`
- Cost metrics:
  - fit time, eval time, tokens/sec, peak VRAM

## 10. Post-train Workflow (Exact "what to do after training")

Given a deterministic checkpoint:

1. Fit Laplace state
- Run `scripts/fit_laplace.py` on selected params with damping.
- Save state artifact.

2. Evaluate uncertainty
- Run `experiments/b1_laplace_baseline.py` using checkpoint + Laplace state.
- Perform MC posterior sampling during eval.
- Compute MI ID vs OOD with same protocol as A2/A3.

3. Compare and decide
- Compare B1 metrics vs A2/A3 and later B3.
- Use Gate G1 from `specs/B-milestone-roadmap.md`.

## 11. Acceptance Criteria for B1

Mandatory:
1. End-to-end run completes from deterministic checkpoint -> Laplace fit -> MI eval.
2. OOD MI > ID MI on AG News split (same protocol as A2/A3).
3. No deterministic training loop modifications required for base training phase.
4. Tests in `tests/test_laplace.py` pass.
5. MLflow logs include Laplace-specific and standard uncertainty metrics.

Quality targets (soft):
- Runtime feasible on RTX 4070.
- Stable metrics across at least 2 seeds for sampling/eval phase.

## 12. Risks and Guardrails

Risk 1: Curvature instability
- Mitigation: mandatory damping, finite checks, fallback scale.

Risk 2: Over-complex first version
- Mitigation: diagonal/structured-light approximation first; no K-FAC in B1.

Risk 3: Evaluation drift from A-series
- Mitigation: keep uncertainty protocol fixed and comparable to A2/A3.

## 13. Stage Boundary Checklist

Stage #1 (this doc):
- [x] What/why/how documented
- [x] Math + plain English explanation included
- [x] BDD behaviors defined
- [x] TDD test plan specified
- [x] Acceptance criteria defined

Stage #2 (next, separate):
- [ ] Write tests for non-existing B1 code
- [ ] Freeze tests with user sign-off

Stage #3 (after freeze):
- [ ] Implement B1 to satisfy frozen tests

