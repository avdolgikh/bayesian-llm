# Production Epistemic Uncertainty: Approaches & Priority

**Context:** We have trained Bayesian models (C1 variational FFN, C3 BLoB LoRA, C4-TFB).
The question: how to estimate epistemic uncertainty in production with minimal overhead?

---

## Approaches

### 1. LoRA-only MC (PRIMARY)

**What:** Run N full forward passes, but only sample LoRA parameters (0.2% of weights). Base model weights are deterministic -- no reparameterization, no sigma storage for 99.8% of params.

**Why best for prod:**
- C3 already proved LoRA carries strong signal (AUROC 0.916)
- Sampling cost proportional to Bayesian param count, not total param count
- VRAM: no sigma tensors for base weights (~76M params saved)
- Strong separation (MI ratio 1.53x) suggests N=3-5 may suffice
- Same forward pass structure -- no architectural changes needed

**Cost:** N forward passes, but each is cheaper than full variational MC. Expected overhead: ~Nx wall-clock (compute-bound, not memory-bound).

**Risk:** Still N forward passes. If latency budget is strict (e.g., real-time generation), even N=3 may be too much.

---

### 2. TFB MC (ALSO AVAILABLE)

**What:** Same as #1 but with SVD-structured variance (no training). Already have C4-TFB checkpoint.

**Why consider:** AUROC 0.917 -- ties LoRA-only MC. Zero extra training cost. Same inference mechanism as #1.

**Cost:** Same as #1 (N forward passes with LoRA-only sampling).

**Priority:** Benchmark alongside #1 -- same mechanism, different variance source. Already supported.

---

### 3. Closed-form Variance Propagation (ZERO MC)

**What:** Propagate $\text{Var}(y)$ analytically through Bayesian linear layers instead of MC sampling. For LoRA ($\Delta W = BA$, $A \sim \mathcal{N}(\mu, \sigma^2)$), compute output variance from input and learned $\sigma$ directly.

**Proxies (cheap to better):**
- **Proxy 1:** $\text{trace}(\Sigma_h)$ at final hidden layer. One scalar per token. Cost: ~1 forward pass.
- **Proxy 2:** Linearize logits around mean: $\Sigma_{\text{logits}} \approx J \cdot \Sigma_h \cdot J^T$. One backward pass for Jacobian.
- **Proxy 3:** Approximate logit distribution as Gaussian, compute entropy via quadrature.

**Why consider:** Single pass. True 1x cost. If Proxy 1 correlates well with MC-based AUROC, it dominates everything else on efficiency.

**Risk:** Approximation error accumulates through deep nonlinear stack (16 layers). Linearization may underestimate uncertainty. Untested in our setup.

**Priority:** Stretch goal. Only pursue if LoRA-only MC overhead is unacceptable. Low-rank LoRA structure makes Proxy 1 especially cheap -- worth a quick experiment after D3.

---

### 4. Full Variational MC (BASELINE)

**What:** Sample ALL Bayesian weights (full FFN) N times. Our C1 method.

**Why not for prod:**
- All FFN weights have sigma tensors -- 2x memory for Bayesian layers
- Reparameterization on every weight in every pass
- C1 AUROC (0.876) is weaker than C3 (0.916) -- worse quality AND worse cost
- No structural advantage over LoRA-only MC

**Cost:** N forward passes with full reparameterization. Highest cost, lowest AUROC. Dominated by #1.

**Priority:** Benchmark as a baseline to show LoRA-only MC is strictly better (less cost, more signal).

---

### 5. Mean-Weights + Last-Layer MC

**What:** Run deterministic forward to get hidden states at the final layer, then only sample the output projection N times.

**Why consider:** Cheapest possible MC. Only one layer is sampled. Hidden state computation is shared.

**Risk:** Our A1 results showed output-head-only Bayesian gives MI ratio 1.36x (AUROC unknown but likely ~0.85). Most epistemic signal lives in intermediate layers (A2 > A1). Throwing away FFN-level uncertainty may lose too much.

**Cost:** ~1 forward pass + N cheap matrix multiplies.

**Priority:** Low. Only relevant if we need near-zero overhead AND accept weaker signal. Not applicable to LoRA methods (LoRA is on FFN, not output head).

---

### 6. Deep Ensembles

**What:** Train M independent models, aggregate predictions. Disagreement = uncertainty.

**Why not:** M models = Mx memory, Mx training cost. We have one GPU (RTX 4070, 12 GB). Even M=3 of our 76M model is infeasible. Also: not Bayesian (no weight posterior), philosophically different approach.

**Priority:** Out of scope. Mentioned for completeness.

---

### 7. MC Dropout

**What:** Enable dropout at inference, run N passes. Gal & Ghahramani (2016).

**Why not:** Our model has no dropout (deliberate choice -- clean miniGPT). Adding dropout post-hoc would require retraining. Also: MC Dropout is a crude variational approximation (Bernoulli posterior) -- weaker than learned Gaussian posteriors we already have.

**Priority:** Out of scope.

---

### 8. Deterministic Baselines (NO Bayesian)

**What:** Max softmax probability or predictive entropy from a single forward pass.

**Why include:** C0 deterministic already gives AUROC 0.591 (max-prob). This is the floor. Every Bayesian method must beat this to justify its cost.

**Cost:** Zero overhead. Already computed.

**Priority:** Always include as baseline.

---

## Priority Order for D3

| Priority | Approach | Expected AUROC | Expected Overhead | Status |
|----------|----------|---------------|-------------------|--------|
| 1 | LoRA-only MC (C3) | ~0.916 | ~Nx (N=3-5) | To implement |
| 2 | TFB MC (C4-TFB) | ~0.917 | ~Nx (N=3-5) | To benchmark |
| 3 | Full MC baseline (C1) | ~0.876 | ~Nx (expensive) | To benchmark |
| 4 | Deterministic (C0) | 0.591 | 1x | Already have |
| 5 | Closed-form propagation | Unknown | ~1x | Stretch goal |

**Decision logic:** If LoRA-only MC at N=5 gives <2x overhead and AUROC >0.80 -- ship it. If overhead is unacceptable -- investigate closed-form propagation (Proxy 1) as a fallback.
