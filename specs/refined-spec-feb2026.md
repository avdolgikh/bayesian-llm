# Refined Spec — February 2026

## Overview

Build a Bayesian mini-GPT from scratch to explore epistemic uncertainty in LLMs. Train it, add Bayesian layers, and build a framework for measuring epistemic uncertainty. Later, explore Bayesian LoRA on an existing open-weight LLM (separate track).

---

## Principles

- **Start from scratch.** The old `src/bayesian_llm/` structure is discarded.
- **Keep it simple.** No modern transformer tricks (RoPE, SwiGLU, sliding window, MoE, GQA). Basic architecture, small numbers, focus on Bayesian.
- **PyTorch only.** Use `torch.distributions` for Bayesian layers. No extra Bayesian libraries. Manual work is fine.
- **No notebooks.** Only runnable `.py` scripts.
- **Uncertainty from day one.** Build the epistemic uncertainty evaluation framework alongside the Bayesian extension, not as an afterthought.

---

## Repo Structure (flat, minimal)

```
bayesian-llm/
├── docs/                  # PDF papers (theory source of truth)
├── specs/                 # Planning docs
├── minigpt/               # Python package — all model-related code
│   ├── __init__.py
│   ├── model.py           # MiniGPT architecture (deterministic)
│   ├── layers.py          # Bayesian layers (BayesianLinear, etc.)
│   ├── data.py            # Dataset loading + topic splitting
│   ├── train.py           # Training loop (cross-entropy + ELBO)
│   ├── evaluate.py        # Standard eval (perplexity, generation)
│   └── uncertainty.py     # Epistemic uncertainty measurement
├── experiments/           # Runnable scripts
│   ├── a0_baseline.py
│   ├── a1_bayes_output.py
│   ├── a2_bayes_ffn.py
│   └── ...
├── tests/
├── pyproject.toml
├── README.md
├── NOTES.md
└── AGENTS.md
```

---

## MiniGPT Architecture

Simple decoder-only transformer with small hyperparameters:

- Learned positional embeddings (no RoPE)
- Standard multi-head self-attention with causal mask
- Vanilla FFN: Linear -> GELU -> Linear
- Pre-LayerNorm
- No modern additions (no SwiGLU, no sliding window, no MoE, no GQA)

Typical config:

| Parameter       | Value     |
|-----------------|-----------|
| Layers          | 4-6       |
| Heads           | 4-8       |
| Embedding dim   | 128-256   |
| FFN hidden dim  | 512-1024  |
| Context length  | 128-256   |
| Vocab           | char-level (~100) or small BPE (~1k-5k) |

Fits in <1 GB VRAM, trains in minutes.

---

## Datasets

### A0 smoke test
**TinyShakespeare** (~1 MB, character-level). Standard "hello world" for miniGPT. Just prove the model trains and generates.

### A1+ uncertainty evaluation
A **multi-topic corpus** with clear topic boundaries so we can train on some topics and hold out others as OOD. Top pick: **AG News** (4 categories: World, Sports, Business, Sci/Tech). Alternatives: curated Wikipedia subsets by category, 20 Newsgroups.

---

## Which Layers to Make Bayesian

Based on mechanistic interpretability research: FFN layers store factual knowledge, attention layers route information.

| Layer type                  | What it captures                      | Priority                         |
|-----------------------------|---------------------------------------|----------------------------------|
| **Output head**             | Uncertainty in final prediction       | A1 — simplest, proves pipeline   |
| **FFN layers** (both linears) | Uncertainty about stored knowledge  | A2 — main event, strongest signal|
| **Q/K/V projections**       | Uncertainty about attention routing   | Optional / later                 |
| **Embeddings**              | Uncertainty about token representations | Optional / later               |

Progression: output head first (A1), then FFN layers (A2).

---

## Epistemic Uncertainty Measurement

### Core idea
All stochasticity comes from Bayesian weights, not from decoding. Use **temperature = 0** (greedy). Run N forward passes (10-30), each sampling fresh weights from learned posteriors. Measure disagreement.

### Procedure (per token position t)

1. Run N forward passes, each sampling weights from q(w)
2. Collect logits: z_t^(1), ..., z_t^(N)
3. Convert to probabilities: p_t^(i) = softmax(z_t^(i))
4. Mean predictive: p_bar_t = (1/N) sum p_t^(i)

### Metrics

| Metric                | Formula                                        | Measures                     |
|-----------------------|------------------------------------------------|------------------------------|
| Predictive entropy    | H[p_bar] = -sum_v p_bar(v) log p_bar(v)       | Total uncertainty            |
| Expected entropy      | E_bar = (1/N) sum_i H[p_t^(i)]                | Aleatoric uncertainty        |
| **Mutual information**| MI = H[p_bar] - E_bar                          | **Epistemic uncertainty**    |

**Mutual information is the gold metric.** It captures pure epistemic uncertainty — the disagreement between weight samples.

### Additional practical metrics
- **Top-k logit variance:** variance of top-3 logit values across N runs
- **Top-1 flip rate:** how often argmax token changes across runs
- **Sequence-level MI:** average per-token MI across the sequence

### Evaluation framework
1. **Train** on topics A, B (in-distribution)
2. **Evaluate** on held-out samples from A, B -> expect low MI
3. **Evaluate** on topics C, D (never seen) -> expect high MI
4. **Report:** MI gap between in-distribution and OOD

---

## Phased Plan

### Phase 1 — MiniGPT Baseline (A0)
1. Clean up repo (remove old `src/bayesian_llm/`, set up flat `minigpt/` structure)
2. Implement decoder-only transformer in `minigpt/model.py`
3. Implement TinyShakespeare data loader in `minigpt/data.py`
4. Implement training loop (cross-entropy) in `minigpt/train.py`
5. Implement basic evaluation (loss, perplexity, text generation) in `minigpt/evaluate.py`
6. Wire up `experiments/a0_baseline.py` — train, evaluate, generate
7. Verify: loss decreases, generation produces coherent text, fits on GPU

### Phase 2 — Bayesian Output Head + Uncertainty Framework (A1)
8. Implement `BayesianLinear` in `minigpt/layers.py` (mean-field Gaussian, reparameterization trick, `torch.distributions`)
9. Add ELBO loss (data log-likelihood + KL divergence) to `minigpt/train.py`
10. Create `BayesianMiniGPT` variant with Bayesian output head only
11. Wire up `experiments/a1_bayes_output.py` — train with ELBO, verify convergence
12. Implement multi-pass inference + uncertainty metrics (MI, entropy, logit variance) in `minigpt/uncertainty.py`
13. First uncertainty sanity check on TinyShakespeare

### Phase 3 — Bayesian FFN + Topic-Split Evaluation (A2)
14. Extend `BayesianMiniGPT`: replace FFN layers with Bayesian linear layers
15. Implement topic-split data loader (AG News or similar) in `minigpt/data.py`
16. Train on subset of topics, hold out others
17. Run full uncertainty evaluation: in-distribution vs OOD
18. Report MI gap — does epistemic uncertainty clearly separate ID from OOD?

### Phase 4 (later, separate track) — Bayesian LoRA (B1)
- Take existing open-weight LLM (e.g., small Llama or Phi)
- Implement Bayesian LoRA adapters (Bayesian layers only within the adapter)
- Separate codebase section, different data, different evaluation
- Will be planned when Phases 1-3 stabilize

---

## Open Questions

- Character-level vs BPE tokenizer for miniGPT?
- Exact topic split for uncertainty evaluation (which categories to train on, which to hold out)?
- Papers the user will share — may refine which layers to make Bayesian and how
- Mean-field (diagonal covariance) vs full covariance for weight posteriors? (Start mean-field, revisit later)
