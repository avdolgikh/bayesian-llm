# Bayesian LLMs: Comprehensive Research Report
**Date:** March 4, 2026
**Author:** Gemini CLI

## 1. Executive Summary
The field of Bayesian Large Language Models (LLMs) has matured significantly by 2026. The "all-or-nothing" debate between frequentist and Bayesian methods has settled into a pragmatic middle ground: **Bayesian Low-Rank Adaptation (Bayesian LoRA)** and **Post-hoc Laplace Approximations**.

Full-parameter Bayesian training for LLMs (e.g., 7B+ parameters) remains computationally prohibitive and arguably unnecessary. Instead, research focuses on treating specific, high-leverage subsets of parameters (adapters, heads, or "knowledge" layers) as random variables while keeping the vast majority of the pretrained "world model" deterministic.

## 2. Core Methodologies

### A. Bayesian LoRA (The "Sweet Spot")
This is the most viable path for the `bayesian-llm` project. It combines the parameter efficiency of LoRA with the uncertainty quantification of Bayesian inference.
*   **Mechanism:** Instead of learning a single update matrix $\Delta W = BA$, we learn a distribution over $A$ and $B$.
*   **Implementations:**
    *   **Laplace-LoRA (Yang et al., ICLR 2024):** Train standard LoRA (MAP estimate), then fit a Gaussian posterior using a Kronecker-factored Hessian approximation. **Pros:** Fast, works on existing checkpoints. **Cons:** Post-hoc approximation (can miss modes).
    *   **Variational LoRA (BLoB, Wang et al., NeurIPS 2024):** Train a distribution $q(A, B)$ directly using Variational Inference (ELBO). **Pros:** Better OOD detection, learns the "shape" of uncertainty during training. **Cons:** Harder to tune (prior/KL weighting).
    *   **Scalable Bayesian LoRA (ScalaBL):** Performs inference in a tiny subspace of the LoRA parameters themselves. Scaling to 30B+ models with negligible overhead.

### B. Post-hoc Laplace Approximation
A "shortcut" to uncertainty that requires no Bayesian training loop.
1.  Fine-tune a standard model (or LoRA).
2.  Compute the Hessian (curvature) of the loss on the validation set.
3.  Use the inverse Hessian as the covariance matrix for a Gaussian posterior.
*   **Tooling:** The `Laplace` library (Alex Immer et al.) is the industry standard here.

### C. Ensembles & "Pseudo-Bayesian" Methods
*   **Deep Ensembles:** Training 5 LoRA adapters with different seeds. **Verdict:** Often outperforms complex VI in calibration but has $5\times$ inference cost.
*   **MC-Dropout:** "Poor man's Bayesian." Easy to implement but often poorly calibrated for LLMs unless dropout rates are tuned per-layer.

---

## 3. Technical Deep Dive: Tractability Tricks

### Flipout Estimator (Variance Reduction)
*   **Problem:** Naive VI samples one set of weights per batch $\to$ all examples see the same "noise" $\to$ gradients are highly correlated $\to$ high variance.
*   **Solution (Flipout):** For a batch of inputs $X$, sample base weights $W$, but multiply the output by random sign vectors $r_i, s_i$:
    $$ y_i = \phi(x_i \cdot (W \circ r_i s_i^T)) $$
    This effectively decorrelates the noise for each example in the batch without performing $N$ matrix multiplications.
*   **Relevance:** Essential for training Bayesian FFNs (Milestone A2) without huge batch sizes.

### Kronecker-Factored Approximate Curvature (K-FAC)
*   **Problem:** The Hessian for a $4096 \times 4096$ weight matrix is $(4096^2)^2 \approx 281$ trillion elements.
*   **Solution:** Approximate the Hessian $H$ as the Kronecker product of input covariance $A$ and output gradient covariance $B$:
    $$ H \approx A \otimes B $$
    Storage drops from $O(D^2)$ to $O(D)$.
*   **Relevance:** The enabling technology for **Laplace-LoRA**.

---

## 4. Current Ecosystem & Tooling

### PyTorch Libraries
| Library | Status | Best For | Note |
| :--- | :--- | :--- | :--- |
| **`torchbnn`** | Active | Simple prototyping | Good for learning, but lacks advanced estimators (Flipout) for Transformers. |
| **`BLiTZ`** | Maintenance | Variational Layers | Has `BayesianLinear` with "Bayes by Backprop". Good reference implementation. |
| **`Laplace`** | **Standard** | Post-hoc UQ | The go-to for applying Laplace approximations to pre-trained models. |
| **`TyXe`** | Academic | Pyro wrapper | Connects PyTorch models to Pyro's advanced inference (HMC, NUTS). |

---

## 5. Strategic Recommendations for `bayesian-llm`

### 1. Milestone A2 (Bayesian FFN) $\to$ Keep it Pedagogical
*   Do not aim for SOTA performance with full Bayesian FFNs. The parameter overhead ($2\times$) is a dead end for production.
*   **Action:** Implement **Flipout** if training is unstable, but otherwise wrap up A2 quickly.

### 2. Milestone B1 (Bayesian LoRA) $\to$ The Real Goal
*   This is where scientific novelty meets practical utility.
*   **Plan:**
    *   Implement a `BayesianLoRALayer` that replaces `minigpt`'s linear layers.
    *   Start with **Diagonal VI** (learn $\mu$ and $\sigma$ for LoRA matrices $A$ and $B$).
    *   Compare against a **Deep Ensemble** of 3 standard LoRA adapters (strong baseline).

### 3. Metric: "OOD Detection" is King
*   NLL and Perplexity are insufficient.
*   **Validation:** You must create an OOD dataset (e.g., training on AG News, testing on Shakespeare or vice versa).
*   **Success:** The Bayesian model should have **high entropy** (uncertainty) on OOD text, while the deterministic model remains confidently wrong.

## 6. Conclusion
The `bayesian-llm` repo is on the right track. The shift from "Full Bayes" (A2) to "Bayesian LoRA" (B1) perfectly mirrors the field's trajectory.

**Final Advice:** "Uncertainty is not just a metric; it is a feature."
