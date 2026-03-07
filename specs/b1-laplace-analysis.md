# Detailed Analysis of B1 Laplace Results

## The Result: MI Ratio ~ 1.00x -- Why?

The model clearly separates ID from OOD by perplexity (40.99 vs 476.42), so the MAP weights contain domain knowledge. But the Laplace MC sampling produces **identical** uncertainty for ID and OOD. The reason is a combination of two issues: a **curvature estimation problem** and a **hyperparameter scale problem**.

---

## Parameter Deep Dive

### 1. `damping` (lambda = 1.0) -- Prior Precision

This is the precision (inverse variance) of the Gaussian **prior** on the weights. In the Bayesian posterior:

```
posterior_precision_i = curvature_i + damping
posterior_variance_i  = 1 / (curvature_i + damping)
posterior_std_i       = 1 / sqrt(curvature_i + damping)
```

Meaning: "before seeing data, I believe each weight has variance <= 1/lambda = 1.0". The damping acts as a **floor** on the precision -- it prevents the variance from being infinite when curvature is zero.

**Our case**: curvature_mean = 0.000001, damping = 1.0:
- Typical param: sigma = 1/sqrt(0.000001 + 1.0) = **1.0**
- Max curvature param (h=0.05): sigma = 1/sqrt(0.05 + 1.0) = **0.976**
- Ratio between most and least curved: **1.025x** -- essentially no differentiation!

The damping completely **drowns out** the curvature. All 2M parameters get sigma ~ 1.0 regardless of their Fisher values.

### 2. `sample_scale` (s = 1.0) -- Posterior Temperature

Multiplies the posterior standard deviation when sampling:

```
theta_sampled = theta_MAP + sample_scale * sigma * epsilon,  where epsilon ~ N(0,1)
```

With s=1.0 and sigma ~ 1.0, you're adding **Gaussian noise with std ~ 1.0** to every FFN weight. Typical trained FFN weights have magnitude ~0.01-0.1. So you're adding noise **10-100x the weight magnitude**. This is catastrophic -- the model outputs are essentially random noise on every MC sample.

That's why:
- Predictive entropy ~ 9.1 (close to log(50257) = 10.8, nearly uniform over vocabulary)
- Flip rate ~ 0.89 (predictions change ~90% of the time between samples)
- MI is high (1.7) but **identical for ID and OOD** -- pure noise is equally noisy everywhere

### 3. `n_curvature_batches` (100) -- Fisher Estimation Budget

Number of mini-batches used to estimate the diagonal Fisher. More batches = more stable estimate. **This parameter is fine at 100** -- the problem isn't estimation noise, it's a systematic issue with the curvature values being near-zero.

---

## Why Is the Curvature So Small?

Yes, the diagonal Fisher IS the expectation of squared gradients: **h_i = E[g_i^2]**.

But there's a critical bug in how we compute it. Look at `fit_laplace()` in `laplace.py:94-107`:

```python
for _ in range(n_batches):
    x, y = get_batch(data, block_size, batch_size, device)  # batch_size=32
    model.zero_grad()
    logits, loss = model(x, y)  # loss = mean CE over 32*256 = 8192 tokens
    loss.backward()             # grad = mean gradient over 8192 tokens
    curvature_acc[name].add_(grad.detach() ** 2)  # square of the mean!
curvature_acc[name].div_(n_batches)  # average over batches
```

The model's loss (line 154 of model.py) is `F.cross_entropy(..., reduction='mean')` -- **averaged over all B*T = 8192 tokens**. So:

```
grad = (1/8192) * Sum_t grad_CE_t
grad^2 = (1/8192)^2 * (Sum_t grad_CE_t)^2    <-- WHAT WE COMPUTE
```

vs. the correct diagonal Fisher:

```
Fisher_i = (1/N) * Sum_t (grad_CE_t)^2     <-- WHAT WE WANT
```

These are very different! By the bias-variance decomposition:

```
E[g^2] = (E[g])^2 + Var(g)
correct Fisher ~ Var(per-token gradients)    (since E[g] ~ 0 at convergence)
our estimate   ~ (E[g])^2 ~ 0                (since mean gradient -> 0 at convergence)
```

**At convergence, per-token gradients point in different directions and cancel in the mean.** The mean gradient -> 0, so (mean gradient)^2 -> 0. But the individual gradients are non-zero -- they just cancel. We're throwing away all the Fisher information.

This is **not overfitting** -- it's the definition of convergence. The problem is that we're computing the **wrong quantity**.

---

## The Gist of the Squared Gradients Trick

You have a trained model -- a single point theta_MAP in weight space. The question Laplace asks is: **how much could we wiggle each weight before the loss gets noticeably worse?**

### Step 1: Pass data, compute gradients

We pass training data through the frozen MAP model, compute loss, and backpropagate. But we don't update weights. We just **collect the gradients**.

### Step 2: Square the gradients -- this is the key

The squared gradient g_i^2 for parameter theta_i approximates the **curvature** (second derivative) of the loss surface at that point. This is the diagonal of the Fisher Information Matrix.

**The intuition:**

- **Large g_i^2** -> the loss is **steep** in the theta_i direction -> if you wiggle this weight even slightly, predictions change a lot -> the model is **certain** about this weight's value -> **small posterior variance** (sigma^2 = 1/g^2)

- **Small g_i^2** -> the loss is **flat** in the theta_i direction -> you could move this weight and predictions barely change -> the model is **uncertain** about this weight's value -> **large posterior variance** (sigma^2 = 1/g^2)

### Step 3: Invert to get uncertainty

```
posterior_variance_i = 1 / (curvature_i + damping)
```

High curvature -> low variance -> confident.
Low curvature -> high variance -> uncertain.

### So it's NOT "gradients as errors"

It's more like: **gradients as sensitivity probes**. We're measuring how sensitive the loss is to each parameter. Parameters the data "strongly constrains" (high sensitivity) get tight posteriors. Parameters the data "doesn't care about" (low sensitivity) get wide posteriors.

Think of it as a **landscape metaphor**:
- You're standing at the bottom of a valley (MAP estimate)
- In some directions the valley walls are steep -> you know precisely where the bottom is -> low uncertainty
- In other directions it's a broad flat plain -> the bottom could be anywhere nearby -> high uncertainty
- g^2 measures the steepness in each parameter's direction

### Why this gives OOD detection

When you sample from this posterior (wiggle weights according to their uncertainty), the wiggles affect OOD predictions more than ID predictions because:
- The MAP weights were optimized for ID data
- ID predictions are robust to small weight perturbations (they sit in a well-constrained region)
- OOD predictions rely on weight configurations that happen to work "by accident" -- they're fragile to perturbation

So: sample weights -> predictions vary more on OOD -> higher MI on OOD -> detection.

### The Bug's Impact Restated

Our current code computes `(mean_of_gradients)^2` instead of `mean_of_(gradients^2)`. This gives us the **squared mean** instead of the **mean of squares**. At convergence, gradients across different samples point in different directions and cancel out -- the mean -> 0, so (mean)^2 -> 0. But individual squared gradients are positive and non-zero. We're computing the sensitivity of the **average** data point (~ 0 by definition at convergence), instead of the **average sensitivity** across data points (> 0, informative).

---

## Should We Use Non-Training Data for Curvature?

**No. Only training data.** The curvature answers a single question: "how precisely does the training data pin down each weight?" OOD doesn't enter the picture here.

The Laplace posterior is: `p(theta|D_train) ~ p(D_train|theta) * p(theta)`. The curvature (Hessian/Fisher) of the log-likelihood must be computed on training data. Using held-out data would give a different distribution that isn't the true posterior.

### Where OOD detection actually happens

It happens **downstream**, at inference time:

1. **Curvature (training data only)** -> "weight theta_42 is tightly constrained (high Fisher), weight theta_99 is loosely constrained (low Fisher)"
2. **MC sampling** -> wiggle weights according to step 1's uncertainty
3. **Evaluate on ID input** -> predictions stay stable, because the MAP weights were optimized for exactly this kind of input. The well-constrained weights (high Fisher) are the ones that matter for ID, and they barely move.
4. **Evaluate on OOD input** -> predictions fluctuate, because the model's ability to handle OOD relies on weight configurations that are **accidental**, not learned. These "accidental" configurations are fragile to perturbation.

The OOD signal emerges naturally from the mismatch between "what the model learned" (captured by curvature from training data) and "what OOD data needs" (not captured, hence fragile).

### Why using OOD for curvature would be wrong

If you used OOD data for curvature, you'd be measuring "how sensitive are the weights to OOD data" -- and then constraining weights in directions that matter for OOD. This would actually **reduce** OOD uncertainty, which is the opposite of what you want.

### ICLA: curvature may not even matter

The ICLA paper (WACV 2025, Zhdanov et al.) found something even more provocative: **for OOD detection, curvature often hurts**. They replace the Hessian with the identity matrix entirely and just optimize the prior precision. This consistently outperformed standard Laplace for OOD detection on CIFAR-10/100 and ImageNet-200. The intuition: what matters for OOD detection is that the MAP weights encode domain knowledge, and noise at the right scale perturbs OOD predictions more than ID predictions.

---

## What Should We Change?

There are three approaches, from quick to principled:

### Approach A: Quick Hyperparameter Fix (No Code Changes)

Since curvature ~ 0, our posterior is effectively:
```
theta ~ N(theta_MAP, (sample_scale^2 / damping) * I)
```
This is identity-curvature Laplace (ICLA) by accident! We just need the right **effective sigma**:
```
effective_sigma = sample_scale / sqrt(damping)
```

From A2 experience, the sweet spot for sigma is **0.1-0.3**. So:

| damping | sample_scale | effective_sigma | Expected behavior |
|---------|-------------|-----------------|-------------------|
| 1.0     | 0.1         | 0.10            | Conservative, some MI signal |
| 1.0     | 0.2         | 0.20            | A2 sweet spot |
| 1.0     | 0.3         | 0.30            | More noise, potentially better signal |
| 1.0     | 0.01        | 0.01            | Too small, collapse to MAP |

Since curvature is negligible, **damping doesn't matter much** -- only the ratio `sample_scale/sqrt(damping)` matters. Keep damping=1.0, tune sample_scale.

**This is the fastest experiment to run.** Just change `sample_scale` in the YAML and re-run with `--skip-train --laplace-state`.

### Approach B: Fix the Fisher Computation (Correct the Code)

Process sequences one at a time to get per-sample gradients:

```python
for _ in range(n_batches):
    x, y = get_batch(data, block_size, batch_size, device)
    for b in range(batch_size):
        model.zero_grad()
        logits, loss = model(x[b:b+1], y[b:b+1])
        loss.backward()
        for name in param_names:
            grad = selection[name].grad
            if grad is not None:
                curvature_acc[name].add_(grad.detach() ** 2)
        total_samples += 1
for name in param_names:
    curvature_acc[name].div_(total_samples)
```

This gives curvature values ~**1000x larger** (~ batch_size^2 factor). Expected curvature range: ~0.001-50 instead of 0.000001-0.05. Then damping=1.0 would actually differentiate: high-curvature params get smaller sigma, low-curvature params get larger sigma.

**Additionally**, the proper Bayesian scaling is `precision = damping + N_total * per_sample_Fisher`, where N_total is all training sequences (~10,500). Our current code only sees 3,200 samples and averages. We should either:
- Accumulate (sum, not average), scaling naturally with data seen, OR
- Multiply the averaged Fisher by N_total

### Approach C: Optimize Prior Precision (Most Principled -- Empirical Bayes)

The `laplace-torch` library and Laplace Redux paper (Daxberger et al., NeurIPS 2021) recommend optimizing the prior precision via **marginal likelihood maximization**. This automatically finds the damping that balances data fit vs. prior. We could implement a simple grid search over damping using a validation metric (e.g., maximize MI_OOD / MI_ID ratio, or minimize validation NLL).

---

## Recommendation

**Start with Approach A** -- it's zero code changes and will tell us immediately whether the Laplace pipeline *can* produce OOD signal at all:

1. Re-run with `--skip-train --laplace-state data/checkpoints/laplace_state.pt` and try `sample_scale` values: **0.05, 0.1, 0.2, 0.3**
2. If we see MI ratio > 1.0 for any setting, the pipeline works and we just needed the right scale
3. If none work -> the identity-curvature approach (constant sigma for all params) might fundamentally limit OOD detection compared to A-series (where different params learn different sigma)

**Then do Approach B** -- fix the Fisher computation. This is the right thing to do because:
- Per-parameter curvature differentiation is the whole point of Laplace vs. just adding uniform noise
- High-curvature (important) params should be perturbed less
- Low-curvature (uncertain) params should be perturbed more
- This information is lost with our current near-zero Fisher

The compute cost of per-sample processing is ~32x more backward passes for fitting, but the fit was only 8 seconds, so ~4 minutes. Acceptable.

---

## Summary Table

| Issue | Current | Should Be |
|-------|---------|-----------|
| Fisher computation | (mean grad)^2 ~ 0 | mean(grad^2) >> 0 |
| Curvature values | 0.000001 (meaningless) | 0.01-10 (informative) |
| damping vs curvature | damping >> curvature | damping ~ curvature |
| effective sigma | ~1.0 (catastrophic noise) | ~0.1-0.3 (meaningful perturbation) |
| MI ratio | 1.00x (random noise) | target > 1.2x |

---

## References

- [Identity Curvature Laplace Approximation for Improved OOD Detection (WACV 2025)](https://arxiv.org/abs/2312.10464)
- [Laplace Redux -- Effortless Bayesian Deep Learning (NeurIPS 2021)](https://arxiv.org/pdf/2106.14806)
- [A Scalable Laplace Approximation for Neural Networks (ICLR 2018)](https://openreview.net/pdf?id=Skdvd2xAZ)
- [laplace-torch library](https://github.com/aleximmer/Laplace)
- [Modern Arts of Laplace Approximations (Kristiadi blog)](https://agustinus.kristia.de/blog/laplace/)
