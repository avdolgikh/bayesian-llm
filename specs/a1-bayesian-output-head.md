# A1 — Bayesian Output Head: Full Design Document

This document answers every question from `specs/a1-initial-discussion.md` and serves as the implementation blueprint for milestone A1.

---

## 1. What Is a "Bayesian Output Head"?

In A0, the model ends with a deterministic linear projection:

```
hidden (n_embd=256) → lm_head (DeterministicLinear) → logits (vocab_size=50257)
```

In A1, we replace `lm_head` with a **BayesianLinear** layer. Instead of storing fixed weight matrix $W$, we store a *distribution* over weights: each weight $w_{ij}$ has its own Gaussian posterior $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$. On every forward pass, we *sample* concrete weights from this distribution, producing different logits each time. The rest of the model (embeddings, attention, FFN, LayerNorm) stays deterministic.

This is called **mean-field variational inference** — "mean-field" because each weight has an independent Gaussian (diagonal covariance), "variational" because we optimize the posterior parameters $(\mu, \sigma)$ via gradient descent.

---

## 2. Why the Output Head First (Not FFN)?

A natural question: FFN layers store factual knowledge (per mechanistic interpretability research), so wouldn't making them Bayesian give a stronger epistemic signal? Yes — and that's exactly what A2 does. A1 starts with the output head for five deliberate reasons:

**1. Pipeline validation at minimal cost.** A1 introduces a lot of new machinery simultaneously: ELBO training, KL computation, reparameterization trick, multi-pass inference, MI measurement, mean-forward generation. If any of these break, you want exactly *one* Bayesian layer to debug — not 8 (2 linears $\times$ 4 blocks).

**2. Inference efficiency.** With only the head Bayesian, the entire transformer body is deterministic — run it once, get hidden states, then do $N$ cheap matrix multiplies through `lm_head`. If FFN were Bayesian instead, every forward pass diverges inside the transformer at every block — $N$ full forward passes from the start. A1 gives us an almost-free multi-pass evaluation to validate the uncertainty math before paying the real cost in A2.

**3. Direct logit-level uncertainty.** The output head maps directly to token probabilities. Uncertainty in `lm_head` weights $\to$ uncertainty in logits $\to$ directly interpretable MI. With Bayesian FFN, the uncertainty propagates through residual connections, LayerNorm, attention — harder to reason about whether MI is working correctly. A1 gives a clean sanity check: "does the math produce sensible numbers?"

**4. Harder architectural test.** Ironically, FFN would be easier architecturally — no weight tying to break, no parameter explosion from `vocab_size`. But that's exactly why A1 is the harder test of the infrastructure. If ELBO training works with ~25M extra Bayesian params on the head (untied), it'll definitely work on the smaller FFN layers in A2.

**5. Weak signal is the point.** We expect A1 MI separation to be modest — the head only captures "uncertainty in final prediction," not "uncertainty about stored knowledge" (that's FFN's job). If we see *any* MI gap in A1, we know the framework works. A2 is where the strong epistemic signal comes, and by then the entire measurement pipeline is proven.

**In short:** A1 is a controlled experiment on the Bayesian infrastructure, not an attempt to maximize uncertainty signal.

---

## 3. Weight Tying — Why It Must Go

In A0, we tie `lm_head.linear.weight = token_emb.weight` (GPT-2 style). This saves ~12.9M parameters by sharing the embedding matrix with the output projection.

**In A1, weight tying is no longer possible.** The output head becomes Bayesian: it stores `weight_mu` and `weight_rho` (two parameter tensors), not a single `weight`. The embedding layer remains a standard `nn.Embedding`. These are fundamentally different objects — one is a distribution, the other is a point estimate. They cannot share storage.

**Impact:** Parameter count goes up. With `n_embd=256` and `vocab_size=50257`:
- A0 (tied): ~16M params
- A1 (untied, Bayesian head): ~16M deterministic + ~25.7M Bayesian ($256 \times 50257 \times 2$ for $\mu$ and $\rho$, weight + bias) ≈ ~42M total

This is expected and acceptable. The Bayesian parameters are the point — they encode uncertainty.

---

## 4. BayesianLinear — Every Detail Explained

### 4.1. Parameters: $\mu$ and $\rho$

Each BayesianLinear stores **four** parameter tensors (if `bias=True`):

| Parameter | Shape | Meaning |
|-----------|-------|---------|
| `weight_mu` | `(out, in)` | Mean of the weight posterior |
| `weight_rho` | `(out, in)` | Unconstrained parameter that encodes the standard deviation |
| `bias_mu` | `(out,)` | Mean of the bias posterior |
| `bias_rho` | `(out,)` | Unconstrained parameter that encodes the bias standard deviation |

**Why $\rho$ instead of $\sigma$ directly?** Standard deviation $\sigma$ must be positive ($\sigma > 0$). If we stored $\sigma$ as an `nn.Parameter`, gradient descent could push it negative, which is nonsensical. Instead, we store the unconstrained $\rho \in (-\infty, +\infty)$ and convert:

$$\sigma = \text{softplus}(\rho) = \log(1 + e^{\rho})$$

### 4.2. Softplus — The Positivity Transform

Softplus is a smooth, differentiable approximation to ReLU:

$$\text{softplus}(x) = \log(1 + e^x)$$

Properties:
- Always positive: $\text{softplus}(x) > 0$ for all $x$
- For large $x$: $\text{softplus}(x) \approx x$ (linear)
- For very negative $x$: $\text{softplus}(x) \approx e^x \approx 0^+$ (exponentially small but never zero)

**Initial value:** $\rho$ is initialized to $-5$, so $\sigma = \text{softplus}(-5) \approx 0.0067$. This means the posterior starts very tight around the mean — nearly deterministic. The model learns to widen $\sigma$ where uncertainty is warranted.

### 4.3. `_sample` — The Reparameterization Trick

```python
def _sample(self, mu, rho):
    sigma = F.softplus(rho)
    eps = torch.randn_like(mu)      # ε ~ N(0, 1)
    return mu + sigma * eps         # w = μ + σε
```

This is the **reparameterization trick** (Kingma & Welling, 2014; Blundell et al., 2015 "Bayes by Backprop"). The key insight:

**Problem:** We want to optimize $\mu$ and $\sigma$ by gradient descent, but we can't backpropagate through a random sampling operation $w \sim \mathcal{N}(\mu, \sigma^2)$ — the sampling itself is not differentiable.

**Solution:** Rewrite the sample as a deterministic function of the parameters plus external noise:

$$w = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

Now $w$ is a differentiable function of $\mu$ and $\sigma$ (with $\epsilon$ treated as a fixed constant for backprop purposes). Gradients flow through:

$$\frac{\partial w}{\partial \mu} = 1, \quad \frac{\partial w}{\partial \sigma} = \epsilon$$

### 4.4. `reset_parameters` — Weight Initialization

```python
def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight_mu, a=5**0.5)   # Standard init for μ
    nn.init.constant_(self.weight_rho, -5.0)              # σ ≈ 0.007 → start near-deterministic
    if self.bias_mu is not None:
        fan_in = self.in_features
        bound = 1 / fan_in**0.5
        nn.init.uniform_(self.bias_mu, -bound, bound)     # Standard bias init
        nn.init.constant_(self.bias_rho, -5.0)            # Same tight σ
```

**Is this Bayesian-specific?** Partially:
- **$\mu$ init** (Kaiming/uniform for weights, fan-in-scaled for biases): Same as standard `nn.Linear` — we want the mean to start at a sensible point.
- **$\rho = -5$**: This is Bayesian-specific. Starting with very small $\sigma$ means the model begins nearly deterministic (like a standard network) and gradually *learns* to be uncertain where needed. This is important for training stability.

### 4.5. `forward` — Sampling Weights and Computing Output

```python
def forward(self, x):
    weight = self._sample(self.weight_mu, self.weight_rho)   # Sample W ~ q(W)
    bias = self._sample(self.bias_mu, self.bias_rho)         # Sample b ~ q(b)  (if bias)
    return F.linear(x, weight, bias)                          # y = Wx + b
```

Every forward pass draws *fresh* weight samples. Two calls with the same input $x$ will produce *different* outputs (because $\epsilon$ is re-drawn). This is the source of epistemic uncertainty — the disagreement between different weight samples.

`F.linear(x, weight, bias)` is just PyTorch's functional linear layer: $y = xW^T + b$. Same math as `nn.Linear`, but with our sampled weights instead of fixed ones.

---

## 5. Prior, Posterior, and Bayesian Neural Networks

### 5.1. The Bayesian Idea

In a standard neural network, weights $W$ are point estimates (single values). In a **Bayesian** neural network, weights are *random variables* with distributions.

**Bayes' theorem for weights:**

$$p(W | \mathcal{D}) = \frac{p(\mathcal{D} | W) \cdot p(W)}{p(\mathcal{D})}$$

| Term | Name | Meaning |
|------|------|---------|
| $p(W)$ | **Prior** | Our belief about $W$ before seeing data |
| $p(\mathcal{D} \| W)$ | **Likelihood** | How well the data fits given specific weights |
| $p(W \| \mathcal{D})$ | **Posterior** | Updated belief about $W$ after seeing data |
| $p(\mathcal{D})$ | **Evidence** | Normalizing constant (intractable for NNs) |

### 5.2. The Problem: Intractable Posterior

For neural networks, $p(W | \mathcal{D})$ is **intractable** — we can't compute it analytically. The weight space is millions of dimensions, and the likelihood is a complex nonlinear function.

### 5.3. Variational Inference — The Solution

We **approximate** the true posterior $p(W | \mathcal{D})$ with a simpler distribution $q_\theta(W)$ that we *can* work with. In our case:

$$q_\theta(W) = \prod_{ij} \mathcal{N}(w_{ij} \mid \mu_{ij}, \sigma_{ij}^2)$$

This is **mean-field Gaussian** — each weight independently follows a Gaussian, parameterized by $(\mu_{ij}, \sigma_{ij})$. The parameters $\theta = \{\mu, \rho\}$ are what we optimize.

### 5.4. Our Prior

We use a simple **isotropic Gaussian prior**:

$$p(W) = \prod_{ij} \mathcal{N}(w_{ij} \mid 0, \sigma_{\text{prior}}^2)$$

With `prior_std = 1.0`, this means $p(w_{ij}) = \mathcal{N}(0, 1)$. This encodes a gentle preference for small weights — no strong opinions about what the weights should be before training.

### 5.5. Comparison to TensorFlow Probability

In TFP, you'd write something like:

```python
tfp.layers.DenseVariational(
    units=vocab_size,
    make_posterior_fn=...,  # N(mu, softplus(rho)²)
    make_prior_fn=...,      # N(0, 1)
    kl_weight=1/N,
)
```

TFP wraps all the mechanics (reparameterization, KL computation, prior specification) into the layer. Our `BayesianLinear` does the same thing manually. The math is identical — TFP just hides it behind an API. The advantage of manual implementation: we see and control every detail.

---

## 6. The ELBO — Bayesian Loss Function

### 6.1. Derivation

We want $q_\theta(W)$ to be close to the true posterior $p(W | \mathcal{D})$. We measure "closeness" by **KL divergence**:

$$D_{\text{KL}}[q_\theta(W) \| p(W | \mathcal{D})] = \mathbb{E}_{q}\left[\log \frac{q_\theta(W)}{p(W | \mathcal{D})}\right]$$

We can't compute this directly (we don't know $p(W|\mathcal{D})$), but we can derive a computable lower bound on the log-evidence. After rearranging:

$$\log p(\mathcal{D}) \geq \underbrace{\mathbb{E}_{q_\theta(W)}[\log p(\mathcal{D} | W)]}_{\text{expected log-likelihood}} - \underbrace{D_{\text{KL}}[q_\theta(W) \| p(W)]}_{\text{KL to prior}}$$

This is the **Evidence Lower BOund (ELBO)**. Maximizing the ELBO is equivalent to minimizing:

$$\mathcal{L}_{\text{ELBO}} = -\mathbb{E}_{q_\theta(W)}[\log p(\mathcal{D} | W)] + D_{\text{KL}}[q_\theta(W) \| p(W)]$$

### 6.2. The Two Terms

**Term 1 — Data fit (expected log-likelihood):**

$$-\mathbb{E}_{q(W)}[\log p(\mathcal{D} | W)] \approx \text{CrossEntropy}(y, \hat{y})$$

In practice: sample one set of weights $W \sim q(W)$, do one forward pass, compute cross-entropy loss. This is a single-sample Monte Carlo estimate. It's noisy but unbiased, and it works because we're doing this over many minibatches during training.

**Term 2 — KL regularizer (complexity cost):**

$$D_{\text{KL}}[q_\theta(W) \| p(W)]$$

Penalizes the posterior for deviating from the prior. Prevents the model from being overly confident (very small $\sigma$) or from putting all its mass on a single point. This is the Bayesian version of weight regularization — it plays a role analogous to L2 regularization (weight decay) but is principled.

### 6.3. Combined Loss in Code

```python
loss = cross_entropy_loss + kl_weight * kl_loss / num_train_tokens
```

The `kl_weight` and division by `num_train_tokens` are important:
- **`/ num_train_tokens`**: The ELBO derives from summing over the entire dataset. When we train on minibatches, the KL term (which doesn't depend on data) must be scaled down. The standard approach: divide KL by the number of training data points (or tokens).
- **`kl_weight`**: An additional tuning knob. `kl_weight=1.0` is theoretically correct, but in practice values like 0.1 or 0.01 can help early training stability (the model first learns to fit data, then gradually incorporates the KL penalty). This is called **KL annealing** or just KL scaling.

---

## 7. KL Divergence — The Manual Formula

### 7.1. KL Between Two Gaussians (Closed Form)

For a single weight with posterior $q = \mathcal{N}(\mu, \sigma^2)$ and prior $p = \mathcal{N}(0, \sigma_{\text{prior}}^2)$:

$$D_{\text{KL}}[\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, \sigma_{\text{prior}}^2)] = \log\frac{\sigma_{\text{prior}}}{\sigma} + \frac{\sigma^2 + \mu^2}{2\sigma_{\text{prior}}^2} - \frac{1}{2}$$

### 7.2. Mapped to Code

```python
def _kl_normal(self, mu, rho):
    sigma = F.softplus(rho)                     # σ = softplus(ρ)
    prior_var = self.prior_std ** 2              # σ²_prior
    kl = (
        torch.log(self.prior_std / sigma)        # log(σ_prior / σ)
        + (sigma**2 + mu**2) / (2 * prior_var)  # (σ² + μ²) / (2σ²_prior)
        - 0.5                                     # -1/2
    )
    return kl.sum()                              # Sum over all weights in this layer
```

Each term, mapped:
- `torch.log(self.prior_std / sigma)` → $\log\frac{\sigma_{\text{prior}}}{\sigma}$
- `(sigma**2 + mu**2) / (2 * prior_var)` → $\frac{\sigma^2 + \mu^2}{2\sigma_{\text{prior}}^2}$
- `- 0.5` → $-\frac{1}{2}$

The `.sum()` aggregates KL across all weights in one layer. `sum_kl_loss(model)` then sums across all BayesianLinear layers.

### 7.3. Why Not `torch.distributions`?

We *could* write:

```python
from torch.distributions import Normal, kl_divergence
q = Normal(mu, F.softplus(rho))
p = Normal(0, self.prior_std)
kl = kl_divergence(q, p).sum()
```

This would give the **exact same result**. Our manual formula is mathematically identical. We don't use `torch.distributions` because:
1. The formula is short and transparent — easy to verify and debug
2. No dependency on distribution API internals
3. Clearer pedagogically

Either approach is fine. We can switch to `torch.distributions` later if needed, with zero behavioral change.

---

## 8. Architecture — How the Bayesian Head Is Constructed

### 8.1. Design Decision: Config Flag, Not Subclass

We do **not** create a separate `BayesianMiniGPT` class. Instead, we use the existing `BayesConfig` mechanism already built into the architecture:

```yaml
# configs/a1_agnews.yaml
model:
  bayes:
    enabled: true       # ← This is the switch
    prior_std: 1.0
    kl_weight: 1.0
```

**The current problem:** `CausalSelfAttention` and `MLP` both receive `config.bayes` and pass it directly to `make_linear()`. When `bayes.enabled=true`, *every* linear layer in the model becomes Bayesian — attention Q/K/V, attention projection, FFN up/down, and `lm_head`. For A1, we only want `lm_head` to be Bayesian.

### 8.2. Selective Bayesian Layers (Option A)

**The fix (Option A):** `MiniGPT.__init__` creates a forced-deterministic `BayesConfig(enabled=False)` and passes it to `CausalSelfAttention` and `MLP`. Only `lm_head` receives the real `config.bayes`. This is explicit — no registry, no string matching, you read the constructor and immediately see what's Bayesian.

```python
# In MiniGPT.__init__:
no_bayes = BayesConfig(enabled=False)

# All transformer internals forced deterministic:
# CausalSelfAttention gets no_bayes → self.qkv, self.proj are DeterministicLinear
# MLP gets no_bayes → self.fc, self.proj are DeterministicLinear
self.blocks = nn.ModuleList([Block(config, no_bayes) for _ in range(config.n_layer)])

# Only lm_head gets the real config:
self.lm_head = make_linear(config.n_embd, config.vocab_size, config.bayes, bias=False)
```

For A2, we'll add a `bayes_ffn: BayesConfig` field to `GPTConfig` and pass it to `MLP`, keeping the same pattern — each layer group gets its own explicit Bayesian config.

### 8.3. Weight Tying Removal

When `bayes.enabled=True`, the output head is a `BayesianLinear` (no `.linear` attribute). We must skip weight tying:

```python
# In MiniGPT.__init__:
if not config.bayes.enabled:
    self.lm_head.linear.weight = self.token_emb.weight   # A0: weight tying
# else: no tying — BayesianLinear has its own (mu, rho) params
```

---

## 9. Training — ELBO, Sampling, Backpropagation

### 9.1. Modified Forward Pass

Each forward pass through `BayesianLinear` samples fresh weights. The `model(x, targets)` call:

1. Token + position embeddings (deterministic)
2. Transformer blocks (deterministic in A1)
3. LayerNorm (deterministic)
4. **lm_head** — samples $W \sim \mathcal{N}(\mu, \sigma^2)$ → produces logits
5. Cross-entropy loss on logits

### 9.2. ELBO Training Loop

```python
# In train():
logits, ce_loss = model(x, y)                    # CE with sampled weights
kl = model.kl_loss()                              # Sum KL across all Bayesian layers
loss = ce_loss + kl_weight * kl / num_train_tokens
loss.backward()                                    # Gradients flow through reparameterization trick
optimizer.step()
```

### 9.3. Backpropagation Through Sampled Weights

Thanks to the reparameterization trick:
- $w = \mu + \text{softplus}(\rho) \cdot \epsilon$
- $\epsilon$ is treated as a constant during backprop (it was sampled before the forward pass)
- Gradients $\frac{\partial \mathcal{L}}{\partial \mu}$ and $\frac{\partial \mathcal{L}}{\partial \rho}$ flow normally through the computation graph
- PyTorch autograd handles this automatically — no special code needed

The optimizer updates $\mu$ and $\rho$ like any other parameters. Over many steps, $\mu$ converges to a good point estimate (similar to a deterministic network's weights), while $\sigma = \text{softplus}(\rho)$ converges to reflect the model's uncertainty about each weight.

### 9.4. What Changes in `train.py`

**Key design: one `train()` function for both deterministic and Bayesian models.** No flags, no branching in the experiment script. The ELBO naturally degrades to pure CE when there are no Bayesian layers:

```python
# Training step — works for both A0 (deterministic) and A1 (Bayesian):
logits, ce_loss = model(x, y)
kl = model.kl_loss()                              # Returns 0.0 if no BayesianLinear layers
loss = ce_loss + kl_weight * kl / num_train_tokens # When kl=0 → loss = ce_loss (pure A0 path)
loss.backward()
```

**Auto-detection mechanism:** `model.kl_loss()` (via `sum_kl_loss`) walks all modules. If none are `BayesianLinear`, it returns `tensor(0.0)`. The `kl_weight * 0.0 / N` term vanishes — no cost, no code path difference. This means:
- `a0_baseline.py` can use the same `train()` and get pure CE (because `bayes.enabled=false` → all `DeterministicLinear` → KL=0)
- `a1_bayes_output.py` uses the same `train()` and gets ELBO (because `lm_head` is `BayesianLinear` → KL>0)

Additional logging when KL > 0:
- `kl_loss` per eval step
- `elbo_loss` (= CE + weighted KL) per eval step
- `ce_loss` separately, so we can track data fit vs regularization

---

## 10. Generation

### 10.1. Research Mode — Bayesian Sampling

For research and uncertainty analysis, we generate with **weight sampling** and **temperature=0** (greedy decoding from logits):

```python
for i in range(N):                               # N = 3-10 samples
    # Each call to model() draws fresh weights from q(W)
    generated = model.generate(prompt, max_new_tokens=100, temperature=0.0)
```

- **Temperature=0**: `argmax` over logits — no randomness from the decoding process
- **All stochasticity from Bayesian weights**: different weight samples → different logits → potentially different argmax tokens
- This isolates **epistemic uncertainty** (weight uncertainty) from **aleatoric uncertainty** (inherent randomness in decoding)

### 10.2. Frozen Sampling — One Weight Draw Per Sequence

**Problem with naive generation:** The current `BayesianLinear.forward()` calls `_sample()` on every invocation — drawing fresh weights each time. During autoregressive decoding, `generate()` calls `model(x)` once per token. This means each token is generated by a *different* weight sample — like switching brains mid-sentence. This is both theoretically wrong (not a coherent sample from the posterior) and practically harmful (incoherent text).

**Solution: freeze the sampled weights for the duration of a `generate()` call.** Sample $W \sim q(W)$ once before decoding starts, hold it fixed for all $L$ steps. Each generated sequence represents "one model drawn from the posterior."

```python
# Mechanism: BayesianLinear gets a freeze/unfreeze API
class BayesianLinear(BayesianModule):
    def freeze_sample(self):
        """Sample weights once and cache them for subsequent forward calls."""
        self._frozen_weight = self._sample(self.weight_mu, self.weight_rho)
        self._frozen_bias = self._sample(self.bias_mu, self.bias_rho) if self.bias_mu is not None else None

    def unfreeze_sample(self):
        """Clear cached weights — resume fresh sampling on every forward call."""
        self._frozen_weight = None
        self._frozen_bias = None

    def forward(self, x):
        if self._frozen_weight is not None:
            return F.linear(x, self._frozen_weight, self._frozen_bias)
        weight = self._sample(self.weight_mu, self.weight_rho)
        bias = self._sample(self.bias_mu, self.bias_rho) if self.bias_mu is not None else None
        return F.linear(x, weight, bias)
```

**Usage in generation (research mode):**

```python
# Generate K sequences, each with a different frozen weight sample
for k in range(K):
    model.lm_head.freeze_sample()          # Draw one W ~ q(W), hold fixed
    seq = model.generate(prompt, max_new_tokens=100, temperature=0.0)
    model.lm_head.unfreeze_sample()        # Back to fresh sampling
```

A convenience context manager on the model keeps it clean:

```python
with model.frozen_bayesian_sample():       # freeze all BayesianLinear layers
    seq = model.generate(prompt, max_new_tokens=100, temperature=0.0)
# auto-unfreezes on exit
```

**When to use which mode:**

| Mode | Weights | Use case |
|------|---------|----------|
| Fresh sampling (default) | Re-sampled every `forward()` | Training (ELBO), scoring fixed sequences (MI computation) |
| Frozen sampling | Sampled once, held fixed | Autoregressive generation (research mode) |
| Mean weights | $\mu$ only, no sampling | Production generation, deterministic eval |

**Note:** For MI computation on a fixed sequence (Section 11), fresh sampling is correct — each of the $N$ forward passes runs over the full sequence in one call, so weights are already consistent across all $T$ positions within each pass (see 11.1).

### 10.3. Production Mode — Mean Weights

For practical text generation, we use the **posterior mean** $\mu$ directly, without sampling:

```python
# mean_forward: uses mu directly instead of sampling
def mean_forward(self, x):
    weight = self.weight_mu                      # No sampling, just μ
    bias = self.bias_mu if self.bias_mu is not None else None
    return F.linear(x, weight, bias)
```

This gives deterministic output (like a standard neural network) and is 1x cost. The mean-weights model is almost always sufficient for generation quality — the value of weight sampling is in *measuring disagreement*, not in producing better text.

We add a `use_mean` flag or `mean_forward()` method to `BayesianLinear`. The model's `generate()` method accepts a `use_mean=True` parameter.

---

## 11. Uncertainty Estimation — Full Math

### 11.1. Setup

Given an input sequence of $T$ tokens, we run $N$ forward passes through the model, each time sampling fresh weights $W^{(i)} \sim q(W)$. For each token position $t$:

- Collect logit vectors: $z_t^{(1)}, z_t^{(2)}, \ldots, z_t^{(N)}$
- Convert to probability distributions: $p_t^{(i)} = \text{softmax}(z_t^{(i)})$

Here, $p_t^{(i)}$ is a probability vector over the entire vocabulary $V$, produced by the $i$-th weight sample at position $t$. Each $p_t^{(i)}(v)$ is the probability assigned to vocabulary token $v$.

**Important: weights are sampled once per forward pass, not per token.** Each call to `model(x)` triggers `lm_head._sample()` once, drawing a single weight matrix $W^{(i)}$ that is applied to **all $T$ token positions simultaneously** (it's a batched matrix multiply). This means:
- Each forward pass = one coherent "model" (same weights for every position)
- $N$ passes = $N$ different models, each giving a full `(T, vocab)` logit matrix
- No per-token re-sampling — that would be theoretically wrong (breaks the "one model per sample" interpretation) and slower

### 11.2. Mean Predictive Distribution

Average the probability vectors across all $N$ weight samples:

$$\bar{p}_t(v) = \frac{1}{N} \sum_{i=1}^{N} p_t^{(i)}(v)$$

This is the **posterior predictive distribution** (Monte Carlo approximation). It represents the model's best guess after accounting for weight uncertainty. Note: this is *not* the same as using mean weights (see Section 9.2) — averaging predictions is different from predicting with average weights due to nonlinearity.

### 11.3. Predictive Entropy — Total Uncertainty

$$H[\bar{p}_t] = -\sum_{v \in V} \bar{p}_t(v) \log \bar{p}_t(v)$$

This is the **Shannon entropy** of the mean predictive distribution. It measures **total uncertainty** — both from the model not knowing the right answer (epistemic) and from the task being inherently ambiguous (aleatoric).

**In code:**
```python
p_bar = probs.mean(dim=0)                          # (seq_len, vocab_size)
predictive_entropy = -(p_bar * torch.log(p_bar + 1e-10)).sum(dim=-1)  # (seq_len,)
```

### 11.4. Expected Entropy — Aleatoric Uncertainty

$$\bar{E}_t = \frac{1}{N} \sum_{i=1}^{N} H[p_t^{(i)}] = \frac{1}{N} \sum_{i=1}^{N} \left(-\sum_{v \in V} p_t^{(i)}(v) \log p_t^{(i)}(v)\right)$$

This is the **average entropy of individual predictions**. Each $p_t^{(i)}$ is the softmax output from one weight sample. Its entropy measures how spread out that particular prediction is.

**Why is this aleatoric (irreducible) uncertainty?** Even if we knew the *exact* right weights (zero epistemic uncertainty), each individual forward pass would still produce a softmax distribution with some entropy — because language is inherently ambiguous. The next token often has multiple plausible continuations. This "spread" in each single prediction reflects the intrinsic uncertainty of the task, not our ignorance about weights.

**In code:**
```python
per_sample_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (N, seq_len)
expected_entropy = per_sample_entropy.mean(dim=0)                       # (seq_len,)
```

### 11.5. Mutual Information — Epistemic Uncertainty

$$MI_t = H[\bar{p}_t] - \bar{E}_t$$

**This is the gold metric.** Mutual information between the weight posterior and the predicted token. It measures how much the predictions *disagree across different weight samples*.

**Why is this epistemic (reducible) uncertainty?**
- If all weight samples produce the *same* softmax distribution, then $\bar{p}_t = p_t^{(i)}$ for all $i$, so $H[\bar{p}_t] = \bar{E}_t$, and $MI = 0$. No epistemic uncertainty.
- If weight samples produce *different* distributions, the mixture $\bar{p}_t$ is more spread out (higher entropy) than the individual predictions. The gap $MI > 0$ captures exactly this disagreement.
- This uncertainty is "reducible" because with more training data, the posterior $q(W)$ would tighten (smaller $\sigma$), weight samples would agree more, and MI would shrink.

**Intuition:** Imagine 10 experts (weight samples). If they all agree "the next word is 'the'" → MI ≈ 0 (no epistemic uncertainty, even if each expert individually assigns some probability to other words). If 5 say "the" and 5 say "a" → MI is high (they disagree because they were trained on limited data).

**In code:**
```python
mi = predictive_entropy - expected_entropy                             # (seq_len,)
```

### 11.6. Summary Table

| Metric | Formula | Measures | Interpretation |
|--------|---------|----------|---------------|
| Predictive entropy $H[\bar{p}]$ | $-\sum_v \bar{p}(v) \log \bar{p}(v)$ | Total uncertainty | High = model unsure (any reason) |
| Expected entropy $\bar{E}$ | $\frac{1}{N}\sum_i H[p^{(i)}]$ | Aleatoric uncertainty | High = task is inherently ambiguous |
| **Mutual information** $MI$ | $H[\bar{p}] - \bar{E}$ | **Epistemic uncertainty** | High = weight samples disagree = model lacks knowledge |

### 11.7. Additional Metrics

| Metric | Computation | Use |
|--------|-------------|-----|
| **Top-1 flip rate** | Fraction of $N$ samples where $\arg\max p_t^{(i)}$ differs from mode | Quick binary "is the model confused?" |
| **Top-k logit variance** | Variance of top-3 logit values across $N$ samples | Scale-sensitive complement to MI |
| **Sequence-level MI** | $\frac{1}{T}\sum_{t=1}^{T} MI_t$ | Per-sequence uncertainty score for downstream use |

---

## 12. Practical Measurement — How It Works

### 12.1. N Forward Passes

For a given input sequence of length $T$:

```python
model.eval()
probs_list = []
for i in range(N):
    logits, _ = model(x)                    # (batch, seq_len, vocab) — fresh weight sample
    probs = F.softmax(logits, dim=-1)       # Convert to probabilities
    probs_list.append(probs)

probs = torch.stack(probs_list, dim=0)       # (N, batch, seq_len, vocab)
```

**How many passes?** $N = 10$ for quick estimates, $N = 30$ for publication-quality results. For A1, **start with N=30** — we're on a small model, cost is trivial.

### 12.2. Important: A1 Efficiency

In A1, only the output head is Bayesian. The transformer body is deterministic. This means:

1. The hidden representation $h = \text{transformer}(x)$ is the **same across all $N$ passes**
2. Only `lm_head(h)` differs (different weight samples)

We can optimize: run the transformer body **once**, then run `lm_head` $N$ times with different samples:

```python
model.eval()
h = model.forward_body(x)                   # Run transformer once → hidden states
probs_list = []
for i in range(N):
    logits = model.lm_head(h)               # Fresh sample each time
    probs_list.append(F.softmax(logits, dim=-1))
```

This is ~1x cost + $N$ cheap matrix multiplies. Very efficient.

### 12.3. Evaluation Pipeline on AG News

```
AG News (4 categories):
├── ID: World (1) + Sports (2)
│   ├── train (80%) → used for training
│   ├── val (10%) → checkpoint selection
│   └── test_id (10%) → unbiased ID evaluation
└── OOD: Business (3) + Sci/Tech (4)
    └── test_ood (100%) → OOD evaluation
```

For uncertainty evaluation:
1. Take random sequences from `test_id` → compute MI per token → average → **ID MI score**
2. Take random sequences from `test_ood` → compute MI per token → average → **OOD MI score**
3. **Expected result:** OOD MI >> ID MI (the model is more epistemically uncertain about topics it's never seen)

### 12.4. What to Log to MLflow

| Metric | Source | MLflow key |
|--------|--------|------------|
| Test ID perplexity | Standard eval | `test_id_perplexity` |
| Test OOD perplexity | Standard eval | `test_ood_perplexity` |
| Mean MI (ID) | Uncertainty eval | `mi_id_mean` |
| Mean MI (OOD) | Uncertainty eval | `mi_ood_mean` |
| MI ratio (OOD/ID) | Computed | `mi_ood_id_ratio` |
| Mean predictive entropy (ID) | Uncertainty eval | `pred_entropy_id_mean` |
| Mean predictive entropy (OOD) | Uncertainty eval | `pred_entropy_ood_mean` |
| Mean expected entropy (ID) | Uncertainty eval | `expected_entropy_id_mean` |
| Mean expected entropy (OOD) | Uncertainty eval | `expected_entropy_ood_mean` |
| Top-1 flip rate (ID) | Uncertainty eval | `flip_rate_id` |
| Top-1 flip rate (OOD) | Uncertainty eval | `flip_rate_ood` |
| KL loss (final) | Training | `final_kl_loss` |
| ELBO loss (final) | Training | `final_elbo_loss` |
| N forward passes used | Config | logged as param |
| `kl_weight` | Config | logged as param |
| `prior_std` | Config | logged as param |

---

## 13. Production Considerations (Forward-Looking)

From `specs/practical-inference.md` — we don't implement all of this in A1, but we design with it in mind.

### 13.1. Strategy 2: Deterministic Generation + Separate Scoring

The production path for A1:
1. **Generate** using mean weights ($\mu$) — 1x cost, deterministic, like a normal LLM
2. **Score** the generated sequence with $N$ forward passes to compute per-token MI

This decouples generation from uncertainty estimation. We implement `mean_forward()` on `BayesianLinear` now so the mechanism is ready.

### 13.2. Strategy 3: Closed-Form Uncertainty (A1-Specific Opportunity)

For A1 specifically (only `lm_head` is Bayesian), we can compute logit variance **analytically** without any MC sampling:

Given fixed hidden state $h$ and output head $W \sim \mathcal{N}(M, \text{diag}(\Omega^2))$:

$$\text{Var}(\text{logit}_k) = \sum_j \Omega^2_{kj} \cdot h_j^2$$

This is **one forward pass** — same cost as deterministic inference. The variance per logit is a direct measure of epistemic uncertainty for that vocabulary token.

We should **benchmark this against MC-based MI** in A1 to see if it's a sufficient proxy. If it correlates well, it becomes the production uncertainty estimator.

### 13.3. What We Implement Now

- `BayesianLinear.mean_forward(x)` — uses $\mu$ only, returns logits
- `BayesianLinear.mean_forward_with_variance(x)` — uses $\mu$, also returns per-logit variance (closed-form)
- `MiniGPT.generate(..., use_mean=True)` — generation with mean weights
- MC-based uncertainty in `uncertainty.py` — the research-grade measurement

---

## 14. What to Expect

### 14.1. Perplexity

A1 must **match A0 test_id_ppl ≤ 49.11**. The Bayesian output head adds parameters and the KL regularizer acts as additional regularization — perplexity should be comparable or slightly better (KL prevents overfitting). If perplexity degrades, likely causes:
- KL weight too high (model underfits because it's penalized for deviating from prior)
- Initialization issue ($\rho$ too large → too much sampling noise early in training)

### 14.2. MI Separation

The primary A1 success metric: **MI on OOD > MI on ID**, with a clear gap. Specifically:
- ID MI should be low and relatively uniform (model "knows" these topics)
- OOD MI should be notably higher (model is uncertain because it's never seen Business/Sci/Tech)
- The ratio `mi_ood / mi_id` should be > 1, ideally > 2x

### 14.3. Potential Challenges

1. **KL collapse**: $\sigma$ shrinks to near-zero, model becomes effectively deterministic, MI ≈ 0 everywhere. Fix: KL annealing, larger `prior_std`, or minimum $\sigma$ floor.
2. **KL domination**: KL too large, model can't fit data. Fix: lower `kl_weight`, warm up KL gradually.
3. **No weight tying**: 2x more params in lm_head. May need adjusting learning rate or regularization.

---

## 15. Implementation Plan (Step by Step)

### Step 1: Architecture Changes
- `layers.py`: Add `mean_forward()` and `mean_forward_with_variance()` to `BayesianLinear`
- `model.py`: Make `bayes` config apply **only to lm_head**; skip weight tying when `bayes.enabled=True`; add `forward_body()` method for efficient multi-pass inference; add `use_mean` flag to `generate()`
- `config.py`: Add `kl_weight` to training config; update `DEFAULT_CONFIG` with Bayesian training params (N_samples, kl_weight)

### Step 2: Training Changes
- `train.py`: Compute ELBO loss (CE + KL/num_tokens); log KL loss and ELBO to MLflow; pass `kl_weight` and `num_train_tokens` to train loop

### Step 3: Uncertainty Module
- `uncertainty.py`: Implement `compute_uncertainty_metrics(model, data, N, ...)` — returns dict with MI, predictive entropy, expected entropy, flip rate per token position; aggregate to per-sequence and per-dataset means

### Step 4: Experiment Script
- `experiments/a1_bayes_output.py`: Train with ELBO, evaluate perplexity (ID + OOD), run uncertainty evaluation, log everything to MLflow, generate samples (mean weights + N Bayesian samples)
- `configs/a1_agnews.yaml`: Config with `bayes.enabled=true`, `kl_weight`, `prior_std`, `N_samples`

### Step 5: Unit Tests
- MI = 0 for deterministic model (all weight samples identical → no disagreement)
- MI ≥ 0 always (mathematical invariant)
- KL ≥ 0 always
- `mean_forward()` is deterministic (same input → same output)
- Bayesian forward produces variance (different calls → different outputs)

### Step 6: Run and Validate
- Train A1 on AG News with same hyperparameters as A0 (adjusting for ELBO)
- Compare test_id_ppl to A0 reference (49.11)
- Measure MI gap between ID and OOD
- Log everything to MLflow, analyze results

---

## 16. Qualitative Evaluation — Curated Prompt Panel

### Motivation

The quantitative metrics (aggregate MI over random batches) are necessary but not sufficient. A human looking at `mi_id=0.03, mi_ood=0.12` has to trust the numbers blindly. We want a **human-interpretable evaluation** where you can read the prompt, read the generated text, see the per-token uncertainty, and judge whether the model's confidence makes sense.

### Design

Pick real article openings from each AG News category as prompts. Generate continuations with mean weights (temp=0). Score each continuation with N forward passes to get per-token MI. Display everything together.

**Prompt selection:**
- From each ID category (World, Sports): pick 3-5 article title+description openings, truncated to ~30-50 tokens
- From each OOD category (Business, Sci/Tech): same, 3-5 prompts each
- Total: ~12-20 curated prompts
- Prompts should be stored in config or a small JSON file for reproducibility

**Per prompt, compute and log:**

| Field | Description |
|-------|-------------|
| `category` | AG News category (World / Sports / Business / Sci/Tech) |
| `split` | ID or OOD |
| `prompt` | The article opening (human-readable text) |
| `continuation` | Generated text (mean weights, temp=0, ~100 tokens) |
| `per_token_mi` | MI value for each generated token |
| `sequence_mi` | Mean MI across the continuation |
| `top1_flip_rate` | Fraction of tokens where argmax changes across N samples |

**Example output:**

```
[ID — World] "World leaders gathered in Geneva to discuss..."
  → "the ongoing conflict in the region. The UN secretary..."
  → Sequence MI: 0.028  |  Flip rate: 0.02
  → Per-token MI: [0.01, 0.03, 0.02, 0.04, 0.01, ...]

[OOD — Sci/Tech] "Researchers at MIT developed a new quantum..."
  → "technology that could revolutionize the way we..."
  → Sequence MI: 0.134  |  Flip rate: 0.18
  → Per-token MI: [0.08, 0.15, 0.12, 0.19, 0.11, ...]
```

### What Gets Logged to MLflow

- **Text artifact** (`qualitative_eval.txt`): Full formatted panel — all prompts, continuations, per-token MI, readable by a human
- **Per-prompt metrics** logged as a table or JSON artifact: category, split, sequence_mi, flip_rate — for later analysis
- **Bayesian samples** (optional, research mode): For 2-3 selected prompts, also generate 3 continuations with weight sampling (temp=0) to show how outputs diverge. Log as `bayesian_samples.txt`

### Expected Observations

- **ID prompts** (World, Sports): coherent continuations, low sequence MI, low flip rate — the model "knows" these topics
- **OOD prompts** (Business, Sci/Tech): possibly less coherent continuations, notably higher MI, higher flip rate — the model is guessing
- **Per-token patterns**: MI should spike on content words specific to the OOD domain (technical terms, company names) and stay low on common function words ("the", "of", "and") even in OOD context
- **Bayesian samples**: ID prompts should produce near-identical continuations across weight samples; OOD prompts should show more divergence
