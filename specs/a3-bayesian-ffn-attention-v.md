# A3 - Bayesian FFN + Attention Value Projection (V)

This document defines milestone A3: keep Bayesian FFN from A2, add Bayesian uncertainty to
attention value projections only, and keep the rest of attention deterministic.

## 1. Context

A2 is complete and validated:
- Bayesian FFN produced stable epistemic signal on AG News.
- Best run: MI ratio 1.43x (batch), 1.70x (qualitative).
- Confirmation run reproduced the behavior (1.36x batch, 1.55x qualitative).

The next question is whether adding attention uncertainty improves ID/OOD separation beyond A2,
without destabilizing training.

## 2. Scope and Non-Goals

In-scope:
- Bayesian FFN remains enabled (same as A2).
- Attention V projection becomes Bayesian.
- Q, K, and attention output projection remain deterministic.
- Same training/evaluation pipeline as A2 (ELBO + MI with MC weight sampling).

Out-of-scope:
- Bayesian Q/K in this milestone.
- Full-covariance Gaussian posterior.
- LoRA track (B1).

## 3. Posterior Choice: Keep Diagonal Gaussian

Decision: continue mean-field (diagonal) Gaussian posteriors for all Bayesian layers in A3.

Why:
1. Scaling: diagonal adds 2 parameters per weight (`mu`, `rho`), while full covariance scales
   quadratically in layer width and is not practical for this model size.
2. Stability: current ELBO + KL setup is already validated for diagonal posteriors in A1/A2.
3. Throughput: uncertainty eval already needs N full forward passes for Bayesian body; adding
   full-covariance sampling would significantly increase compute and memory pressure.
4. Comparability: A3 should isolate architectural impact (adding attention uncertainty), not mix in
   a different posterior family.

Interpretation:
- Full covariance is research-interesting, but not the next pragmatic step.
- If diagonal saturates later, the next complexity step should be low-rank+diagonal, not full.

## 4. Why Only Attention V in A3

Attention computes:

`softmax(QK^T / sqrt(d)) V`

Roles:
- `Q/K`: routing (where to attend).
- `V`: content transport (what information is passed).

Rationale for V-only first:
1. Lower instability risk: stochastic `Q/K` perturbs logits before softmax and can create routing
   noise; stochastic `V` keeps routing deterministic.
2. Better attribution: MI changes can be interpreted as content uncertainty, not attention-map
   volatility.
3. Cleaner ablation: A3 tests whether content-path uncertainty in attention adds signal on top of
   FFN uncertainty.
4. Incremental complexity: one new Bayesian attention component before any Q/K expansion.

## 5. Architecture Plan

## 5.1 Config Surface

Add a dedicated config block:

```yaml
model:
  bayes_head:   # existing
  bayes_ffn:    # existing
  bayes_attn_v: # new in A3
```

`bayes_attn_v` uses the same schema as other Bayesian blocks:
- `enabled`
- `prior_std`
- `init_rho`

KL scaling is global:
- `train.kl_weight`

## 5.2 `GPTConfig` and Config Builder

Files:
- `minigpt/model.py`
- `minigpt/config.py`

Changes:
- Add `bayes_attn_v: BayesConfig` to `GPTConfig`.
- Add defaults in `DEFAULT_CONFIG["model"]["bayes_attn_v"]` (disabled by default).
- Update `build_gpt_config()` to map YAML into `bayes_attn_v`.

## 5.3 Attention Module Refactor (Q/K/V Split)

Current code uses one fused linear: `attn.qkv`.
To Bayesianize only V, split into explicit projections:
- `q_proj`: deterministic
- `k_proj`: deterministic
- `v_proj`: configurable (Bayesian via `bayes_attn_v`)

`attn.proj` remains deterministic in A3.

Files:
- `minigpt/model.py` (`CausalSelfAttention`, `Block`, `MiniGPT`)

## 5.4 KL Weight Resolution

File:
- `experiments/runner.py`

Planned behavior:
- Read a single global `train.kl_weight` and use it for ELBO KL scaling.
- Layer-level Bayesian blocks do not define their own KL weights.

This keeps KL objective configuration in the training section, where it belongs.

## 6. New Experiment Artifacts

Create:
- `configs/a3_agnews.yaml`
- `experiments/a3_bayes_ffn_attn_v.py`

Use the same pattern as A1/A2:
- thin script wrapper calling `run_experiment(milestone="a3", ...)`
- all hyperparameters explicit in YAML

Proposed initial config:
- Same data split as A2 (ID: World+Sports, OOD: Business+Sci/Tech).
- `bayes_head.enabled: false`
- `bayes_ffn.enabled: true` with A2-validated settings.
- `bayes_attn_v.enabled: true` with conservative init.

Initial hyperparameter defaults:
- `model.bayes_ffn`: `prior_std=1.0`, `init_rho=-2.0`
- `model.bayes_attn_v`: `prior_std=1.0`, `init_rho=-3.0`
- `train.kl_weight=0.2`
- `train.steps=100000`, `eval.num_samples=20`

Notes:
- `init_rho=-3.0` for V is intentionally conservative to avoid introducing large routing-adjacent
  noise via attention pathways.

## 7. Parameter Impact (A3 vs A2)

Given `n_embd=256`, `n_layer=4`, `bias=true`:

Per block V projection params:
- deterministic weights+bias: `256*256 + 256 = 65,792`
- Bayesian params (`mu` + `rho`): `131,584`

Across 4 blocks:
- additional Bayesian params: `526,336` (~0.53M)

Total Bayesian params in A3 (expected):
- A2 FFN Bayesian params: `4,204,544`
- plus attention V: `526,336`
- A3 total: `4,730,880` (~4.73M)

This is a modest increase over A2, not a parameter explosion.

## 8. Evaluation Protocol

Same as A2:
1. Train with ELBO (KL annealing unchanged).
2. Evaluate perplexity with mean weights.
3. Evaluate MI on ID and OOD with `N=20` MC samples.
4. Run qualitative prompt panel.
5. Log sigma summary and KL trends.

Primary comparison baseline is A2 R2 and A2 confirmation run.

## 9. Success Criteria

A3 is considered successful if:
1. ID/OOD MI separation remains strong and ideally improves over A2.
2. No pathological KL behavior (rising/diverging KL unrelated to learning).
3. Sigma stats indicate learned posteriors (not collapse to near-zero everywhere).
4. ID perplexity regression remains moderate (no major degradation vs A2).

Recommended quantitative guardrails:
- Batch MI ratio >= 1.35x (minimum to clear A2-confirmation level).
- Qualitative MI ratio >= 1.55x (minimum to clear A2-confirmation level).
- Target for improvement: exceed A2 best (1.43x batch, 1.70x qualitative).

## 10. Required Tests

File:
- `tests/test_bayesian.py`

Add/update tests:
1. A3 architecture test: `attn.v_proj` Bayesian, `attn.q_proj/k_proj/proj` deterministic.
2. FFN remains Bayesian under A3 config.
3. Weight tying still active (head deterministic).
4. `_has_bayesian_body()` remains `True`.
5. Forward stochasticity remains present.

Adjust existing A2 attention tests that currently assert `attn.qkv` exists/deterministic; after
Q/K/V split they should assert deterministic `q_proj` and `k_proj` instead.

## 11. Implementation Checklist

1. Add `bayes_attn_v` to config datamodel and defaults.
2. Refactor attention from fused `qkv` to `q_proj/k_proj/v_proj`.
3. Wire `v_proj` Bayesian via `bayes_attn_v`.
4. Update runner KL-weight resolution for multi-component Bayesian models.
5. Add `configs/a3_agnews.yaml`.
6. Add `experiments/a3_bayes_ffn_attn_v.py`.
7. Update tests for attention split and A3 invariants.
8. Run `uv run pytest tests/ -q` and `uv run ruff check minigpt/ experiments/ tests/`.
9. Run first A3 training (`seed=1337`) and log to MLflow.
10. Document results in `AGENTS.md`.

Implementation status (2026-03-05):
- Done: steps 1-8.
- Pending: step 9 (first full A3 train/eval run and MLflow metrics).
- Partial: step 10 (implementation status documented; run results pending).

## 12. Follow-Up After A3

If A3 shows clear gain without large perplexity cost:
- Keep V-only as default attention Bayesianization path.
- Optional A3b later: add Bayesian K, then Q, as separate ablations.

If A3 gain is marginal:
- Prefer moving to B1 (Bayesian LoRA) rather than escalating attention complexity in miniGPT.
