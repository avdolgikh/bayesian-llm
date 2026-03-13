# Comparative Study: Bayesian Uncertainty Methods for Language Models

## Thesis

A controlled, end-to-end comparison of four Bayesian uncertainty estimation approaches on the same architecture, dataset, and evaluation protocol — to determine which method best captures epistemic uncertainty for OOD detection in language models.

## 2x2 Comparison Matrix

|  | **Full weights** | **LoRA adapters** |
|---|---|---|
| **Variational (train-time)** | A-series (done) — MI ratio 1.43x | BLoB-style (new) |
| **Post-hoc (no Bayesian training)** | B1 Laplace (in progress) | TFB / Laplace-LoRA (new) |

Same model. Same dataset. Same eval protocol. Same target params (FFN). Four approaches, one controlled comparison.

## Why This Works Scientifically

1. **Controlled variable.** Every method shares miniGPT 4L/4H/256d, AG News topic-split, MI evaluation. The *only* variable is the Bayesian treatment. This is how you draw real conclusions.

2. **Covers the full taxonomy.** Train-time vs post-hoc x full-weight vs low-rank. These are the four main approaches in the 2024-2025 Bayesian LLM literature. BLoB (NeurIPS 2024) and TFB (NeurIPS 2025) are from the *same lab* (Wang-ML-Lab) and represent the state of the art.

3. **Negative results are publishable.** If Laplace gives MI ratio 1.00x while variational gives 1.43x -- that's a finding. It says something about post-hoc vs train-time Bayesianization for OOD detection. If LoRA matches full-weight with 10x fewer params -- also a finding.

4. **Already have 80% of the infrastructure.** Evaluation pipeline, MLflow, test suite, data loading, MC metrics -- all shared. Adding LoRA is incremental.

## Controlled Setup

- **Architecture:** miniGPT 4L/4H/256d (16M params)
- **Dataset:** AG News topic-split (ID: World+Sports, OOD: Business+Sci/Tech)
- **Tokenizer:** BPE via tiktoken (GPT-2 encoding, vocab_size=50257)
- **Target params:** FFN layers (MLP.fc + MLP.proj) across all methods
- **Evaluation protocol:** MI (predictive_entropy - expected_entropy), flip rate, perplexity (ID/OOD), qualitative prompt-panel scoring
- **MC sampling:** N=20 forward passes with weight sampling, temperature=0
- **Tracking:** Local MLflow (sqlite backend)

## Methods

### M1: Variational Bayesian (A-series) -- DONE

Full Bayesian layers via variational inference (ELBO training). BayesianLinear replaces selected nn.Linear modules. Weights parameterized as (mu, rho), sampled via reparameterization trick.

- **A1:** Bayesian output head. MI ratio 1.36x. Ceiling: vocabulary-level uncertainty only.
- **A2:** Bayesian FFN. MI ratio **1.43x batch / 1.70x qualitative**. Topic-level OOD detection.
- **A3:** Bayesian FFN + attention V. Negative result vs A2 (closed).
- **Best result:** A2 with init_rho=-2, 100K steps, sigma mean=0.147.

### M2: Post-hoc Laplace (B1) -- DONE (NEGATIVE)

Diagonal Fisher curvature on FFN params of a deterministic checkpoint. No Bayesian training. MC sampling via Gaussian posterior: `theta ~ N(theta_MAP, diag(1/(curvature + damping)))`.

- **R1 (100K steps):** MI ratio 1.00x. Fisher computation bug (batch-averaged gradients).
- **Approach A (sample_scale sweep):** MI ratio 1.00x at all scales (0.05-0.3). Identity-curvature Laplace fails for LM OOD detection.
- **Approach B (per-sample Fisher):** Correct Fisher (mean=0.000018), MI ratio still 1.00x. Curvature too small vs damping — loss landscape is flat at convergence.
- **Conclusion:** Diagonal Laplace on FFN params cannot substitute for train-time variational inference for LM OOD detection.

### M3: BLoB-style Bayesian LoRA -- PLANNED

Bayesian Low-Rank Adaptation by Backpropagation (BLoB, NeurIPS 2024). Variational inference during LoRA training. Asymmetric design: Bayesianize A matrix, fix B matrix.

- Freeze deterministic base model
- Add LoRA adapters to FFN layers
- Train LoRA with variational inference (mu + rho for A matrix)
- Evaluate MI via MC sampling (sample LoRA A, compute forward pass)

### M4: TFB / Laplace-LoRA -- PLANNED

Post-hoc Bayesianization of trained LoRA adapters. Two variants:
- **TFB (NeurIPS 2025):** Training-Free Bayesianization. Systematic variance search on trained LoRA. No gradients needed -- inference only.
- **Laplace-LoRA:** Fit diagonal Laplace on LoRA params (reuses B1 pipeline directly).

- Train standard LoRA on FFN layers (deterministic)
- Apply post-hoc variance estimation
- Evaluate MI via MC sampling

## How LoRA Fits on miniGPT

1. **Freeze** the B1 deterministic checkpoint (100K steps, already trained)
2. **Add LoRA** adapters to FFN layers (same target as A2 and B1 Laplace)
3. **BLoB approach:** train LoRA with variational inference (mu + rho for LoRA A matrix)
4. **TFB approach:** train standard LoRA, then post-hoc variance search (inference only)
5. **Laplace-LoRA approach:** train standard LoRA, then fit Laplace on LoRA params

### Base Model Concern

The base model is already trained on ID data. LoRA would learn near-zero residual corrections. Two options:

- **Option A (recommended):** Pretrain miniGPT on TinyShakespeare (general domain), then LoRA fine-tune on AG News ID. Clean pretrain -> finetune split. Matches how LoRA is used in practice.
- **Option B:** Use existing setup, accept small LoRA corrections. Bayesian signal should still emerge from uncertainty in those corrections.

## Comparison Table (Target)

| Method | Type | Bayesian Params | MI Ratio (batch) | MI Ratio (qual) | Test ID PPL | Training Cost |
|---|---|---|---|---|---|---|
| A0 Deterministic | baseline | 0 | -- | -- | 49.1 | baseline |
| A2 Variational FFN | train-time, full | 4.2M | 1.43x | 1.70x | 53.5 | 1x |
| B1 Laplace FFN | post-hoc, full | 2.1M (post-hoc) | 1.00x | 0.99x | 41.0 | 0.01x fit |
| B2 BLoB LoRA | train-time, LoRA | 163K (rank 16) | 1.13x | 1.02x | 226.9 | 0.1x |
| B3 TFB LoRA | post-hoc, LoRA | 82K (post-hoc) | 1.10x | TBD | 224.6 | 0.01x search |
| B3 Laplace-LoRA | post-hoc, LoRA | 82K (post-hoc) | 1.00x | 1.00x | 224.6 | 0.001x fit |

## Concerns and Mitigations

1. **Scale criticism from reviewers.** "16M params, does this generalize to real LLMs?" Mitigated by: (a) the contribution is *methodology comparison*, not scaling claims; (b) BLoB and TFB papers include small-model experiments; (c) can add one HuggingFace experiment as a "scaling check" appendix -- not the main contribution.

2. **LoRA rank on a small model.** With n_embd=256, LoRA rank r=4-16 is reasonable. But the ratio of LoRA params to base params is different than on a 7B model. Need to discuss this in the paper.

3. **Scope discipline.** The 2x2 matrix is clean. Resist adding more methods (ensembles, MC dropout, etc.) or the paper becomes unfocused. Four methods, one table.

## Execution Plan

1. **Finish B1** (Approach B Fisher fix). Get the Laplace number, even if it's weak. A weak result is a valid data point.
2. **Implement LoRA adapter** on miniGPT -- minimal: just a LoRALinear wrapper. Follow BLoB's asymmetric design (Bayesianize A, fix B).
3. **Run BLoB and TFB** on the same setup. Reference their papers explicitly for hyperparameter choices.
4. **Write the comparison.** Table + analysis + conclusion about which approach works best for epistemic uncertainty in language models.

## References

- **BLoB** (NeurIPS 2024) -- Bayesian LoRA by backprop. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- **TFB** (NeurIPS 2025) -- Training-Free Bayesianization for LoRA. [arXiv:2412.05723](https://arxiv.org/abs/2412.05723)
- **Wang-ML-Lab bayesian-peft** -- Reference implementation. [GitHub](https://github.com/Wang-ML-Lab/bayesian-peft)
- **Laplace-LoRA** (2023) -- Laplace approximation on LoRA params. [arXiv:2308.13111](https://arxiv.org/abs/2308.13111)
- **Bayesian-LoRA** (2025) -- Probabilistic Low-Rank Adaptation. [arXiv:2601.21003](https://arxiv.org/html/2601.21003)
- **ICLA** (WACV 2025) -- Identity Curvature Laplace for OOD Detection. [arXiv:2312.10464](https://arxiv.org/abs/2312.10464)
- **Laplace Redux** (NeurIPS 2021) -- Effortless Bayesian Deep Learning. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)
