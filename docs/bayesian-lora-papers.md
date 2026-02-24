# Bayesian LoRA for LLMs — Paper Notes

Four papers on Bayesian treatment of Low-Rank Adapters.
Relevant to milestone **B1** (Bayesian LoRA on open-weight LLM).

---

## 1. BLoB — Bayesian Low-Rank Adaptation by Backpropagation (NeurIPS 2024)

[PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d53575463291ea6b5a23cf6e571f59b-Paper-Conference.pdf)

**Core idea.** Train-time variational Bayes for LoRA. Keep B deterministic; place a factorized Gaussian on A: q(A) = prod N(M_ij, Omega_ij^2). This induces a full-weight Gaussian q(vec(W)) = N(mu_q, Sigma_q) where mu_q = vec(W_0 + BM) and Sigma_q = (I_n x B) diag(vec(Omega)^2) (I_n x B^T) (Theorem 3.1).

**KL trick.** Choose a low-rank prior in full weight space so that KL[q(vec(W)) || P(vec(W))] = KL[q(A) || P(A)] (Theorem 3.2) — reduces to closed-form diagonal KL (Eq. 11). No expensive full-space KL needed.

**Training.** Joint mean + variance optimization via reparameterization + Flipout sampling. ELBO objective.

**Cost.** Per-parameter variance: rd additional params (r = rank, d = hidden dim). ~1.4x LoRA params.

**Expressivity.** Per-element Omega gives fine-grained, per-parameter uncertainty — most expressive among the four methods.

**When to use.** Full fine-tuning access, best uncertainty + generalization needed. Joint mean/variance learning outperforms post-hoc methods.

---

## 2. TFB — Training-Free Bayesianization (2024)

[arXiv:2412.05723](https://arxiv.org/abs/2412.05723)

**Core idea.** Post-hoc: convert an already-trained LoRA into a Bayesian adapter without retraining. Define a projected isotropic Gaussian posterior: q(vec(W) | B) = N(mu_q, proj(sigma_q^2 I)).

**Why one scalar sigma_q.** Isotropic variational family — uncertainty controlled by a single scalar. Makes generalized VI reducible to variance-maximization search. Parameter overhead: O(r), shared across layers.

**Calibration procedure.** Given anchor dataset D, metric l, and tolerance epsilon, solve: max sigma_q s.t. |l(D|after) - l(D|before)| <= epsilon (Eq. 8). Solved via binary search.

**Why SVD of B.** LoRA factors aren't unique: (B, A) and (BS, S^{-1}A) give same Delta W. SVD B = U diag(d) V^T yields canonical factorization B' = U diag(d), A' = V^T A. Singular values then scale noise so "isotropic in projected space" is well-defined.

**Cost.** Negligible — no retraining, only binary search over sigma_q. ~1000 params total.

**When to use.** Already have LoRA checkpoints (from any method) and want uncertainty now with zero retraining.

---

## 3. Laplace-LoRA — Bayesian Low-rank Adaptation via Laplace Approximation (2023)

[arXiv:2308.13111](https://arxiv.org/abs/2308.13111)

**Core idea.** Post-hoc Laplace approximation over LoRA parameters. After MAP training, fit a Gaussian posterior centered at theta_MAP with covariance from the inverse Hessian (Fisher).

**Hessian.** Kronecker-Factored (KFAC) approximation of the Fisher, with low-rank factors (rank n_kfac ~ 8). Diagonal Laplace tested but inferior — KFAC captures weight correlations.

**Linearization.** Linearized Laplace exclusively: f(x) ~ f_MAP(x) + J^T (theta - theta_MAP). Linearized predictions outperform weight-space sampling.

**Scope.** Two variants: LA (all LoRA layers) vs LLLA (last-layer only). LA consistently better.

**Prediction.** MC sampling via Cholesky decomposition. Outperforms probit and Laplace bridge closed-form alternatives.

**Results (LLaMA-2-7B).** ECE drops ~87% vs MAP (e.g. 2.1 vs 31.2 on Winogrande-S). NLL drops ~49%. Accuracy maintained. Beats MC dropout, checkpoint ensembles, and competitive with temperature scaling — without needing validation set.

**Cost.** Memory: +1-5%. Runtime: ~10% for Hessian accumulation (one-time, post-hoc).

---

## 4. ScalaBL — Scalable Bayesian Low-Rank Adaptation via Stochastic Variational Subspace Inference (2025)

[arXiv:2506.21408](https://arxiv.org/abs/2506.21408)

**Core idea.** Bayesian inference in an r-dimensional subspace (r = LoRA rank). Repurpose LoRA projection matrices as the subspace basis: W = W_0 + B diag(s) A, where s in R^r are subspace coordinates with a diagonal Gaussian posterior: q(s) = N(s_mu, diag(s_sigma)).

**Subspace construction.** SVD of A fixes canonical basis. Right singular vectors become A; singular values become the learnable variational parameters s.

**Training.** Standard reparameterization trick (no Flipout): W_t = W_0 + B diag(s_mu + s_sigma * epsilon_t) A. ELBO with KL weight beta (max 0.1).

**Cost.** Only 2r additional params per layer (~1000 total). This is ~1500x fewer than BLoB for comparable calibration.

**Results (Qwen2.5-7B).** Matches BLoB on NLL (0.31 vs 0.30) and accuracy (90.16 vs 89.53 on ARC-C). ECE slightly higher (5.03 vs 4.03) but far better than MAP (10.11). Scales to Qwen2.5-32B — largest Bayesian LLM to date.

**Inference.** N=10 posterior samples at test time for Bayesian model averaging.

**When to use.** Scaling to large models where BLoB's parameter overhead is prohibitive. Best parameter-efficiency vs uncertainty trade-off.

---

## Comparison Matrix

| | BLoB | TFB | Laplace-LoRA | ScalaBL |
|---|---|---|---|---|
| **When** | Train-time | Post-hoc | Post-hoc | Train-time |
| **Posterior** | Per-param diagonal | Isotropic (1 scalar) | KFAC Gaussian | Diagonal in r-dim subspace |
| **Extra params** | rd | O(r) | 0 (Hessian stored) | 2r |
| **Expressivity** | High | Low | Medium | Medium |
| **Retraining** | Yes (ELBO) | No | No (needs Hessian pass) | Yes (ELBO) |
| **Scales to** | 7B | 7B+ | 7B | 32B |
| **Best for** | Max calibration | Existing checkpoints | Post-hoc, no extra params | Large models |
