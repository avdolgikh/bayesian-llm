# Epistemic Uncertainty in Language Models via Bayesian Weight Posteriors

## Abstract

We present a controlled comparison of four Bayesian approaches to epistemic uncertainty estimation in language models, organized as a 2×2 matrix: (variational vs post-hoc) × (full weights vs LoRA adapters), plus an MC Dropout baseline. All methods are evaluated on the same 76M-parameter GPT-2-style architecture trained on The Pile with domain-split OOD evaluation. We report three main findings. First, **LoRA-based Bayesian methods achieved the strongest OOD detection in our setup**: BLoB LoRA (AUROC 0.909) and TFB (0.917) outperform full-weight variational inference (0.874) with 17× fewer Bayesian parameters, though this comparison is confounded by differences in training procedure and backbone quality (§6). Second, **diagonal Laplace approximation fails** for language model OOD detection — the diagonal Fisher curvature is flat at convergence regardless of parameterization (full weights or LoRA). Third, **MC Dropout** applied to the deterministic checkpoint (zero extra training) achieves AUROC 0.898 — surprisingly competitive with trained Bayesian methods (overlapping confidence intervals with BLoB). **TFB**, also requiring no Bayesian training, matches the best trained methods at AUROC 0.917. Production benchmarks show N=3 MC samples capture 97% of the full signal at 50ms per sequence.

## 1. Introduction

Large language models produce confident predictions even on inputs far from their training distribution. Standard softmax probabilities conflate two distinct sources of uncertainty: **aleatoric** uncertainty (inherent ambiguity in the data) and **epistemic** uncertainty (what the model doesn't know). Distinguishing these is critical for safe deployment — a model should flag when it is uncertain due to lack of knowledge, not just when the next token is ambiguous.

Bayesian inference offers a principled framework: replace point-estimate weights with posterior distributions, then measure disagreement across weight samples. Mutual information (MI) between weights and predictions isolates the epistemic component. High MI signals "the model's weights disagree about this input" — a direct indicator of out-of-distribution data.

Recent work has introduced several practical methods for Bayesian weight posteriors in large models, including variational LoRA (BLoB; Wang et al., 2024), post-hoc SVD-based variance (TFB; Shi et al., 2025), and post-hoc Laplace on LoRA (Yang et al., 2024). However, these methods have been evaluated on different architectures, datasets, and metrics, making direct comparison impossible.

We address this gap with a **controlled 2×2 comparison** on a single architecture and evaluation protocol:

|  | Full weights | LoRA adapters |
|---|---|---|
| **Variational** (train-time) | Variational FFN | BLoB LoRA |
| **Post-hoc** (no Bayesian training) | Diag. Laplace FFN | TFB / Diag. Laplace LoRA |

All four cells are evaluated on the same 76M-parameter GPT-2-style model trained on The Pile with domain-split evaluation (in-distribution: HackerNews; OOD: ArXiv, FreeLaw, PubMed). We evaluate using AUROC, FPR@95, AUPRC, ECE, Brier score, NLL, and AURC, and report production inference costs.

**Contributions:**

1. First controlled head-to-head comparison of variational vs post-hoc Bayesian methods on the same LM architecture, dataset, and evaluation protocol, including an MC Dropout baseline.
2. LoRA-based methods (BLoB, TFB) achieved stronger OOD detection than full-weight variational inference at 76M parameters, though the comparison is confounded by training procedure and backbone quality differences (§6).
3. Definitive negative result for **diagonal Laplace** in LM OOD detection — both full-weight and LoRA parameterizations produce MI ratio 1.00×. This does not extend to KFAC or full-Hessian variants.
4. **MC Dropout** (zero extra training) achieves AUROC 0.898 — a surprisingly strong baseline whose confidence interval overlaps with trained BLoB LoRA (0.909). **TFB** (also zero Bayesian training) achieves the best AUROC (0.917) via SVD-structured variance on existing LoRA checkpoints.
5. Production deployment recipe: merged mean-weights for serving (zero overhead vs deterministic), N=3 MC for uncertainty scoring (97% of full signal at 50ms/sequence).

## 2. Related Work

**Bayesian neural networks.** Variational inference for neural network weights was introduced by Graves (2011) and scaled by Blundell et al. (2015) with Bayes by Backprop. Flipout (Wen et al., 2018) reduces gradient variance through pseudo-independent weight perturbations. These methods provide principled uncertainty but have historically been difficult to scale beyond small networks.

**Laplace approximation.** Post-hoc Laplace fits a Gaussian posterior at the MAP estimate using curvature information. Daxberger et al. (2021) systematized the approach with diagonal, KFAC, and full Hessian variants. While effective for image classifiers, diagonal Laplace has known limitations when Fisher curvature is flat — a condition we find is endemic to well-trained language models.

**Bayesian LoRA methods.** Low-Rank Adaptation (Hu et al., 2022) enables parameter-efficient fine-tuning. Several works extend LoRA with Bayesian posteriors. **BLoB** (Wang et al., 2024) places variational Gaussian posteriors on LoRA's A matrix during training, using a KL decomposition theorem to avoid full-space computation. **TFB** (Shi et al., 2025) converts trained LoRA checkpoints into Bayesian adapters post-hoc via SVD-structured isotropic variance with binary search calibration. **Laplace-LoRA** (Yang et al., 2024) applies Laplace approximation to LoRA parameters. **ScalaBL** (Samplawski et al., 2025) performs Bayesian inference in the r-dimensional subspace defined by LoRA rank, scaling to 32B parameters.

These methods report results on different model families (LLaMA, Qwen, GPT-2), different tasks (MMLU, commonsense QA, calibration), and different metrics. Our contribution is a controlled apples-to-apples comparison on a single setup.

**OOD detection in LMs.** Most work uses token-level predictive entropy or energy scores. We focus on epistemic uncertainty via mutual information, which requires a weight posterior and is theoretically grounded in Bayesian decision theory.

## 3. Methods

### 3.1. Uncertainty Quantification

Given a model with weight posterior $q(\theta)$, we draw $N$ weight samples $\{\theta^{(i)}\}_{i=1}^N$ and compute token-level predictive distributions $p_t^{(i)} = \text{softmax}(f_{\theta^{(i)}}(x)_t)$. Epistemic uncertainty is measured by **mutual information** (MI):

$$\text{MI} = H\left[\frac{1}{N}\sum_i p_t^{(i)}\right] - \frac{1}{N}\sum_i H[p_t^{(i)}]$$

The first term (predictive entropy) captures total uncertainty; the second (expected entropy) captures aleatoric uncertainty. Their difference isolates the epistemic component — uncertainty due to weight disagreement.

For OOD detection, we compute mean MI per sequence and threshold: high MI → OOD. We report AUROC as the primary metric, with FPR@95 as the production-relevant operating point.

### 3.2. Variational Full-Weight

Standard Bayes by Backprop applied to all FFN layers (fc and proj sublayers across all transformer blocks). Each weight $w_{ij}$ is parameterized as $\mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$ with $\sigma_{ij} = \log(1 + \exp(\rho_{ij}))$. Training minimizes the ELBO: reconstruction loss plus KL divergence to a standard normal prior. This doubles the FFN parameter count (33.6M Bayesian parameters on a 76M model).

### 3.3. BLoB LoRA

Bayesian Low-Rank Adaptation by Backpropagation (Wang et al., 2024). LoRA decomposes weight updates as $\Delta W = \frac{\alpha}{r} BA$, with rank $r=16$. BLoB places variational Gaussian posteriors on $A$ while keeping $B$ deterministic. Via a KL factorization theorem, the full-weight KL reduces to a diagonal computation in the low-rank A-space. Training uses Flipout for variance reduction. Total Bayesian parameters: 1.97M (2.5% of model).

### 3.4. Laplace Approximation

Post-hoc diagonal Laplace: fit a Gaussian posterior $\mathcal{N}(\hat{\theta}, (\text{diag}(F) + \lambda I)^{-1})$ at the MAP estimate using per-sample diagonal Fisher information. Applied to both full FFN weights (33.6M parameters) and LoRA parameters (1.97M parameters). No Bayesian training — operates on existing deterministic or LoRA checkpoints.

### 3.5. TFB

Training-Free Bayesianization (Shi et al., 2025). Converts a trained LoRA checkpoint into a Bayesian adapter without retraining. SVD of B yields singular values that define per-direction variance scaling: $\Omega_{ij} = \sigma_q / d_i$, where $d_i$ are singular values. A single scalar $\sigma_q$ controls the posterior width, found by binary search on a calibration set. Total fit time: 7 minutes.

### 3.6. MC Dropout

Monte Carlo Dropout (Gal & Ghahramani, 2016) interprets dropout at inference as approximate variational inference with a Bernoulli posterior over weights — each dropout mask corresponds to a sample from this posterior. We apply it to the deterministic baseline checkpoint (C0) with the training dropout rate (0.1) on attention and FFN dropout layers. At inference, N stochastic forward passes (each with a different random dropout mask) produce N predictions; MI is computed from their disagreement. No retraining or architectural changes are required — MC Dropout can be applied to any model that was trained with dropout.

## 4. Experimental Setup

**Architecture.** GPT-2-style transformer: 16 layers, 8 heads, 512 embedding dimension, 2048 FFN width. ~76M parameters. No modern tricks (RoPE, SwiGLU, GQA) — deliberately minimal to isolate Bayesian effects from architecture improvements.

**Dataset.** The Pile (Gao et al., 2020), domain-split for OOD evaluation. Training domains: StackExchange, Ubuntu IRC, EuroParl, HackerNews. OOD domains: ArXiv, FreeLaw, PubMed Central. BPE tokenization (GPT-2 vocabulary, 50,257 tokens). Training: ~100K steps, batch size 16, sequence length 256.

**Evaluation protocol.** 500 in-distribution sequences (HackerNews) and 500 OOD sequences (ArXiv + FreeLaw + PubMed) at block_size=256. N=20 MC weight samples (or dropout passes for MC Dropout). Metrics: AUROC, FPR@95 TPR, AUPRC, ECE, Brier score, NLL, AURC. Each Bayesian method uses MI as its uncertainty score; the deterministic baseline uses max-probability. MC Dropout uses the deterministic checkpoint with dropout enabled at inference. 95% bootstrap CIs (10,000 resamples) are reported for AUROC.

**Hardware.** Single NVIDIA RTX 4070 (12 GB VRAM). AMP fp16 enabled. PyTorch 2.x with Flash Attention.

## 5. Results

### 5.1. OOD Detection

**Table 1.** OOD detection performance (16L scale, 500 ID + 500 OOD sequences, N=20 MC samples).

| Method | Type | MI Ratio | AUROC [95% CI] | FPR@95 | AUPRC | ECE | Brier | NLL | AURC |
|---|---|---|---|---|---|---|---|---|---|
| Deterministic | — | — | 0.591 [0.556, 0.626] | 0.794 | 0.552 | 0.022 | 0.606 | 2.79 | 0.500 |
| MC Dropout | — | — | 0.898 [0.877, 0.917] | 0.368 | 0.870 | 0.012 | 0.608 | 2.79 | — |
| Variational FFN | Var × Full | 1.32× | 0.874 [0.852, 0.895] | 0.494 | 0.866 | 0.023 | 0.673 | 3.31 | 0.347 |
| Diag. Laplace FFN | Post × Full | 1.00× | 0.536 [0.500, 0.572] | 0.934 | 0.533 | 0.033 | 1.000 | 9.10 | 0.988 |
| **BLoB LoRA** | **Var × LoRA** | **1.53×** | **0.909 [0.890, 0.925]** | **0.424** | **0.909** | 0.044 | 0.658 | 3.06 | **0.330** |
| **TFB LoRA** | **Post × LoRA** | **1.35×** | **0.917 [0.900, 0.933]** | **0.384** | **0.918** | **0.022** | 0.658 | 3.06 | 0.337 |
| Diag. Laplace LoRA | Post × LoRA | 1.00× | 0.494 [0.459, 0.529] | 0.956 | 0.495 | 0.034 | 0.998 | 9.73 | 0.963 |

The two LoRA methods — BLoB (variational) and TFB (post-hoc) — achieve the strongest OOD detection (AUROC 0.909 and 0.917, with overlapping 95% bootstrap CIs). MC Dropout, a zero-training baseline applied to the deterministic checkpoint, is surprisingly competitive at AUROC 0.898 with the best calibration (ECE=0.012). Full-weight variational inference reaches 0.874 but requires 17× more Bayesian parameters than LoRA methods. Both diagonal Laplace variants produce near-random OOD detection (AUROC ≈ 0.5) with severely degraded calibration (Brier ≈ 1.0, NLL > 9.0). All 95% CIs are computed via 10,000 bootstrap resamples of per-sequence scores.

MI is the only effective uncertainty score. Predictive entropy and max-probability AUROC remain near 0.5 for all methods (Table 2), confirming that token-level uncertainty is insufficient for OOD detection — weight-level disagreement (MI) is required.

**Table 2.** Uncertainty score comparison (AUROC by score type).

| Method | MI | Pred. Entropy | Max-Prob |
|---|---|---|---|
| Deterministic | — | 0.545 | 0.591 |
| MC Dropout | 0.898 | 0.561 | 0.615 |
| Variational FFN | 0.874 | 0.506 | 0.548 |
| BLoB LoRA | 0.909 | 0.532 | 0.569 |
| TFB LoRA | 0.917 | 0.553 | 0.589 |

**Key takeaways from Table 1.** (1) MC Dropout — a zero-cost Bayesian baseline requiring no extra training — achieves AUROC 0.898, surprisingly competitive with trained methods. Its 95% CI [0.877, 0.917] overlaps with BLoB LoRA [0.890, 0.925], so the advantage of trained variational LoRA over MC Dropout is not statistically significant at this sample size. Practitioners should consider whether the modest AUROC gain justifies the additional training cost. (2) MC Dropout achieves the best calibration (ECE=0.012) and tied-lowest NLL (2.79), suggesting that dropout's implicit regularization benefits prediction quality alongside uncertainty estimation. (3) Diagonal Laplace is definitively negative — the result is consistent across four independent experiments (B1, B3-LAP, C2, C4-LAP), two scales (4L, 16L), and both parameterizations (full weights, LoRA). (4) TFB (AUROC 0.917, also zero Bayesian training) is the strongest single method, though its CI overlaps with both BLoB and MC Dropout.

### 5.2. Production Viability

**Table 3.** Inference benchmarks (RTX 4070, sequence length 256, batch size 1, AMP fp16).

| Method | N | Latency (ms) | Overhead | VRAM (MB) |
|---|---|---|---|---|
| Deterministic | 1 | 8.1 | 1.0× | 470 |
| Full Var. MC | 5 | 71.0 | 8.8× | 534 |
| Full Var. MC | 20 | 286.9 | 35.4× | 534 |
| BLoB LoRA MC | 3 | 50.5 | 6.2× | 382 |
| BLoB LoRA MC | 5 | 84.4 | 10.4× | 382 |
| TFB LoRA MC | 5 | 133.0 | 16.4× | 389 |

Each MC sample requires a full forward pass because the residual stream diverges after each Bayesian FFN layer — there is no shared computation to amortize. Overhead scales linearly with N. LoRA MC uses 28% less VRAM than full variational MC (382 vs 534 MB) because only 2.5% of parameters carry variance.

**Table 4.** AUROC vs MC samples N (200 ID + 200 OOD sequences).

| Method | N=1 | N=3 | N=5 | N=10 | N=20 |
|---|---|---|---|---|---|
| Full Var. MC | 0.500 | 0.850 | 0.855 | 0.866 | 0.869 |
| BLoB LoRA MC | 0.500 | **0.861** | **0.879** | 0.880 | 0.888 |
| TFB LoRA MC | 0.500 | 0.847 | 0.859 | 0.881 | 0.886 |

**N=3 is the knee.** AUROC jumps from 0.50 (N=1, no disagreement possible) to ~0.86 (N=3) — 97% of the N=20 signal. Further samples add diminishing returns: N=5→20 contributes <3 AUROC points. The production sweet spot is **N=3 at 50ms/sequence with 382 MB VRAM**.

### 5.3. Mean-Weights Inference

For serving predictions (without uncertainty scoring), mean-weights inference uses the posterior mean $\mu$ as a deterministic point estimate. This avoids MC sampling entirely.

**Table 5.** Mean-weights PPL vs MC-averaged PPL (N=20).

| Method | Mean-Wt PPL | MC-Avg PPL | Relative Diff |
|---|---|---|---|
| Variational FFN | 24.48 | 24.41 | 0.29% |
| BLoB LoRA | 16.12 | 16.78 | 3.93% |

Mean-weights perplexity matches MC-averaged perplexity within 4%. For LoRA methods, the posterior mean $\mu_A$ can be merged into the base weights ($W' = W + \frac{\alpha}{r} B \mu_A$), yielding a standard dense model with **zero overhead** — identical architecture and latency to the deterministic baseline. This enables a two-tier deployment: use merged mean-weights for serving predictions (no overhead, deterministic) and reserve MC sampling only for uncertainty estimation as a post-processing step.

## 6. Discussion

**LoRA vs full-weight: an observational comparison.** LoRA-based methods (BLoB, TFB) achieved stronger OOD detection than full-weight variational inference in our experiments (AUROC 0.909–0.917 vs 0.874). However, this comparison is confounded by at least three factors: (1) **training procedure** — variational FFN trains the entire model from scratch with ELBO, while LoRA methods fine-tune adapters on a pre-trained deterministic backbone; (2) **backbone quality** — LoRA methods inherit a well-converged base model (ppl=14.3), while full-weight variational must learn representations and posteriors jointly (ppl=21.9); (3) **parameter count** — 1.97M vs 33.6M Bayesian parameters. One hypothesis is that LoRA's rank-16 subspace constrains posteriors to directions that matter for adaptation, producing more meaningful weight disagreement. But disentangling this subspace effect from the training-procedure and backbone-quality confounds requires ablations (e.g., variational fine-tuning on the pre-trained backbone, or BLoB LoRA from random initialization) that we leave to future work.

**Diagonal Laplace: a hypothesis for the failure.** Diagonal Laplace approximates the posterior covariance as the inverse of the diagonal Fisher information matrix. We observe that at convergence, the per-parameter Fisher values are near-zero for our well-trained language models — the loss landscape appears flat in most coordinate directions. We hypothesize that this produces an overly diffuse posterior that generates no meaningful prediction disagreement. The failure is consistent across full weights (33.6M params) and LoRA weights (1.97M params) at both 4L and 16L scale, suggesting the pattern may be general for diagonal Laplace on language models — though we note this does not extend to KFAC or full-Hessian variants, which capture off-diagonal curvature.

**TFB vs diagonal Laplace: a possible explanation.** Both are post-hoc methods operating on LoRA parameters, yet TFB succeeds (AUROC 0.917) where diagonal Laplace fails (0.494). The key difference appears to be variance structure: diagonal Laplace uses curvature (diagonal Fisher), which in our experiments carries no directional information at convergence. TFB uses SVD of the B matrix, which captures the geometric structure of the LoRA subspace — singular values encode how much each direction contributes to the weight update. We conjecture that this geometric information, being independent of the loss landscape's curvature, remains informative at convergence. Testing this hypothesis would require comparing TFB's SVD-structured variance against alternative variance structures (e.g., random isotropic, Fisher-weighted).

**MC Dropout as a competitive baseline.** MC Dropout (Gal & Ghahramani, 2016) — enabling dropout at inference on the deterministic checkpoint — achieves AUROC 0.898 with zero extra training cost. Its confidence interval [0.877, 0.917] overlaps with BLoB LoRA [0.890, 0.925], meaning we cannot claim statistical significance for the trained method's advantage at this sample size. MC Dropout also achieves the best calibration (ECE=0.012) among all methods. This suggests that for applications where marginal AUROC gains are not critical, MC Dropout may be the most cost-effective choice.

**Limitations.** Our comparison is conducted at 76M parameters on a GPT-2-style architecture. Scaling behavior to billion-parameter models remains an open question, though ScalaBL (Samplawski et al., 2025) reports positive results for Bayesian LoRA at 32B. We evaluate only autoregressive language modeling with domain-split OOD; other tasks (classification, QA) and OOD types (semantic shift, adversarial) may yield different rankings. Our diagonal Laplace negative result does not extend to KFAC or full-Hessian Laplace variants, which capture off-diagonal curvature — though these are computationally expensive at scale. The LoRA vs full-weight comparison is observational; the three confounds listed above prevent causal attribution. All results are from single training runs — bootstrap CIs capture evaluation variance but not training variance.

## 7. Conclusion

In a controlled 2×2 comparison of Bayesian uncertainty methods for language models, LoRA-based approaches (BLoB and TFB) achieved the strongest OOD detection at 76M-parameter scale (AUROC 0.909–0.917 vs 0.874 for full-weight variational), though this advantage may partly reflect differences in training procedure and backbone quality rather than parameterization alone. MC Dropout — a zero-cost baseline — proved surprisingly competitive (AUROC 0.898, overlapping CIs with BLoB). TFB is particularly notable: it achieves the best AUROC (0.917) with zero Bayesian training — only a 7-minute binary search on an existing LoRA checkpoint. Diagonal Laplace approximation fails consistently across all configurations.

For production deployment, we recommend: (1) merged mean-weights for serving predictions (zero overhead — LoRA means fold into base weights, producing a standard dense model), (2) N=3 MC sampling for uncertainty scoring when needed (50ms/sequence, 382 MB VRAM, 97% of full signal). For applications where marginal AUROC gains over MC Dropout are not critical, MC Dropout on any existing checkpoint offers the lowest-cost path to epistemic uncertainty.

## References

Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. *ICML*.

Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2021). Laplace Redux — Effortless Bayesian deep learning. *NeurIPS*.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., ... & Leahy, C. (2020). The Pile: An 800GB dataset of diverse text for language modeling. *arXiv:2101.00027*.

Graves, A. (2011). Practical variational inference for neural networks. *NeurIPS*.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.

Samplawski, C., Cobb, A. D., Acharya, M., Kaur, R., & Jha, S. (2025). Scalable Bayesian low-rank adaptation of large language models via stochastic variational subspace inference. *UAI*.

Shi, H., Wang, Y., Han, L., Zhang, H., & Wang, H. (2025). Training-free Bayesianization for low-rank adapters of large language models. *NeurIPS*.

Wang, Y., Shi, H., Han, L., Metaxas, D., & Wang, H. (2024). BLoB: Bayesian low-rank adaptation by backpropagation for large language models. *NeurIPS*.

Wen, Y., Vicol, P., Ba, J., Tran, D., & Grosse, R. (2018). Flipout: Efficient pseudo-independent weight perturbations on mini-batches. *ICLR*.

Yang, A. X., Robeyns, M., Wang, X., & Aitchison, L. (2024). Bayesian low-rank adaptation for large language models. *ICLR*.
