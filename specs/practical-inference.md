# Practical Inference with Bayesian LLMs

The N-forward-pass uncertainty procedure is conceptually clean but has real cost implications. This doc thinks through what's actually needed at inference time, and how to do better.

---

## The Core Cost Problem

Sampling N=10 times for every token is 10x expensive. For autoregressive generation of L tokens, that's N*L forward passes. For miniGPT this is fine, but for real LLMs it's prohibitive.

So the question becomes: **can we separate generation from uncertainty estimation?**

---

## Do We Even Need to Sample Weights?

The learned posterior N(μ, σ²) has **very small σ** after training (initialized at softplus(-5) ≈ 0.007, typically stays small or shrinks). Samples are tightly clustered around μ — 99.7% fall within μ ± 3σ, a tiny range. So in practice we're never really drawing from (-∞, +∞).

This raises the question: **if σ is so small that samples ≈ μ, why sample at all for generation?**

Three options for single-pass generation:

1. **Just use μ (mean weights)** — deterministic, no sampling. For generation quality this is usually sufficient. The standard practical choice.

2. **Truncated normal** — clamp samples to [μ - kσ, μ + kσ] with k=2 or 3. Easy to implement (sample normally, then clamp). In practice nearly identical to untruncated because σ is already tiny.

3. **Posterior temperature** — instead of ε ~ N(0, 1), use ε ~ N(0, α²) with α < 1. This is a "temperature" on the weight posterior: α=0 gives mean weights, α=1 gives full Bayesian sampling. Lets you dial the exploration/stability tradeoff continuously.

The key insight: **for generation, option 1 (just use μ) is almost always right.** The value of weight sampling is in *measuring disagreement* (MI, variance), not in producing better text. So the real question is not "how to sample better for generation" but "how to cheaply estimate how much the weights *would* disagree if we did sample" — which is the closed-form uncertainty propagation path below.

---

## Posterior Predictive Mean != Mean-Weights Model

A tempting thought: "if N→∞, averaging probs converges to the posterior predictive mean — isn't that just using the mean weights?"

**No.** Because of nonlinearity:

```
E_θ[softmax(f_θ(h))]  ≠  softmax(f_{E[θ]}(h))
```

The softmax (and the deeper nonlinear stack) means that averaging predictions over weight samples is NOT the same as making a single prediction with averaged weights. Jensen's inequality breaks this. The posterior predictive mean captures richer information — it's a mixture of models, not a single model.

However, for **generation quality** (just getting good text), Bayesian averaging often isn't worth the cost. The mean-weights model is usually fine for generation. The value of Bayesian inference is in **uncertainty estimation**, not in better point predictions.

---

## Practical Inference Strategies

### Strategy 1: Sample per sequence, not per token

Draw **one** θ ~ q(θ) and keep it fixed for the entire decode. This is:
- **1x cost** (same as deterministic inference)
- Produces **coherent** output (same weights throughout)
- Equivalent to "one sample from the posterior"

To estimate uncertainty: generate K=2-4 full sequences with different weight draws, compare them. Much cheaper than N passes per token. The disagreement between sequences is a rough epistemic uncertainty signal.

### Strategy 2: Deterministic generation + separate uncertainty scoring

1. **Generate** using mean weights (μ parameters) — 1x cost, like a normal LLM
2. **Score** the generated sequence with N forward passes (weight sampling) to compute per-token MI

This decouples generation from uncertainty. You get the generated text fast, then score it for uncertainty as a post-processing step. The scoring is N forward passes over a fixed sequence (no autoregressive loop), so it parallelizes cleanly.

### Strategy 3: Closed-form uncertainty propagation (no MC at all)

For Bayesian layers with Gaussian posteriors, we can **propagate uncertainty analytically** through the linear parts and approximate through nonlinearities. This avoids MC sampling entirely.

For a Bayesian linear layer y = Wx + b with W ~ N(M, diag(Ω²)):

Given a fixed input x:
- **Mean output:** μ_y = Mx + b
- **Variance:** Var(y_k) = Σ_j Ω²_kj · x²_j

Cost: O(d_out × d_in) — same as the forward pass itself.

For **LoRA-style** adapters y = W₀x + BAx, where A ~ N(M_A, diag(Ω²)):
- Let z = Ax (dimension r)
- μ_z = M_A · x
- Var(z_k) = Σ_j Ω²_kj · x²_j
- Σ_y = B · diag(Var(z)) · Bᵀ

Since r is small (4-16), this is very cheap.

#### Turning layer-level variance into token-level uncertainty

From cheap to better:

- **Proxy 1 (very cheap):** Use trace(Σ) or per-dimension std at the final hidden layer as an epistemic uncertainty score. Just a scalar per token position.

- **Proxy 2 (still cheap):** Linearize the final logits around the mean hidden state:
  ```
  Σ_logits ≈ J · Σ_h · Jᵀ
  ```
  where J = ∂logits/∂h at the mean forward pass. Can get J via one backward pass on the top-k logits, or use a low-rank approximation. Gives per-vocabulary uncertainty.

- **Proxy 3 (best without MC):** Approximate the logit distribution as Gaussian, then compute entropy of the resulting softmax distribution analytically or via Gauss-Hermite quadrature. More involved but still single-pass.

---

## Averaging Logits vs Averaging Probs

The TFB paper averages softmax probabilities across weight samples. An alternative: average the **logits** and softmax once.

- **Average probs** (TFB): E[softmax(z)] — a proper Bayesian model average. More theoretically grounded.
- **Average logits**: softmax(E[z]) — simpler, avoids the "softmax is a nonlinear bottleneck" issue. Equivalent to using the mean logits, which is what the mean-weights model approximates for a single linear output layer.

For a Bayesian output head (just one linear layer is Bayesian), averaging logits is closer to the mean-weights model. For deeper Bayesian layers, the difference grows. Worth benchmarking both in A1.

---

## KV Cache Interaction

When attention layers (Q/K/V) are **deterministic** (Phases A1, A2): the KV cache is shared across weight samples in some cases.

- **A1 (only output head is Bayesian):** Full KV cache is shared across all N passes. The N passes only diverge at the final projection. Very cheap — essentially one forward pass + N matrix multiplies.
- **A2 (FFN layers are Bayesian):** KV values are still produced by deterministic Q/K/V projections, but they receive different inputs (the residual stream diverges after each Bayesian FFN). **Cannot share KV cache.**
- **If Q/K/V become Bayesian:** Each weight sample produces different K/V. No cache sharing.

---

## Summary: What to Do at Each Phase

| Phase | Generation | Uncertainty estimation | Cost |
|-------|-----------|----------------------|------|
| A1 (Bayesian output head) | Mean weights (1x) | Closed-form Σ_logits from output head variance, or cheap N-pass (shared KV cache) | ~1x |
| A2 (Bayesian FFN) | Mean weights (1x) | N-pass scoring (no cache sharing) or uncertainty propagation through FFN chain | ~Nx for MC, ~2x for propagation |
| B1 (Bayesian LoRA) | Frozen base model (1x) | Closed-form via low-rank structure, or few-sample MC on adapter only | ~1x + ε |

The phased Bayesian strategy naturally goes from cheap to expensive inference. Closed-form uncertainty propagation is most practical for A1 and B1 (where Bayesian params are concentrated in one or few linear layers). A2 requires propagation through multiple nonlinear layers, making MC more practical.

---

## Open Questions for Implementation

- Should we implement a `mean_forward()` method on BayesianLinear that uses μ weights and also returns per-output variance? Would unify the closed-form path.
- Is linearization (Proxy 2) accurate enough for the deep nonlinear stack in A2, or does it underestimate uncertainty?
- Benchmark: logit averaging vs prob averaging in A1 — does it matter for MI estimation?
