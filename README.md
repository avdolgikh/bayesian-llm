# Bayesian LLM

Estimating **epistemic uncertainty** in language models via Bayesian inference over weights.

Replace point-estimate weights with learned posterior distributions (mean + variance per weight). Sample weights multiple times, measure how much predictions disagree — that disagreement (mutual information) is the model's epistemic uncertainty. High MI on a given input means the model knows it doesn't know.

## Approach

Start with a small GPT (4-layer, 256-dim) trained on a topic-split corpus (AG News). Train on World + Sports articles; hold out Business + Sci/Tech as OOD. Progressively make layers Bayesian, measure whether MI separates ID from OOD text.

## Results so far

| Milestone | What's Bayesian | MI ratio (OOD/ID) | Status |
|-----------|----------------|-------------------|--------|
| **A0** | Nothing (baseline) | — | Done |
| **A1** | Output head | 1.36x | Done |
| **A2** | FFN layers | **1.70x** | Tuning |

**A2 highlights:** 4.2M Bayesian params (out of 20M total). Posteriors learned meaningful structure — sigma ranges from 0.04 (confident weights) to 0.97 (uncertain, near prior). MI cleanly separates all four categories. FFN captures topic-level uncertainty, not just vocabulary.

## Quick start

```bash
uv sync
python experiments/a0_baseline.py --config configs/a0_agnews.yaml        # deterministic baseline
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml       # Bayesian FFN
uv run pytest tests/ -v                                                   # 28 tests
```

Requires Python 3.12+ and CUDA-enabled PyTorch for GPU training.

## Next steps

- Further A2 tuning (prior_std, kl_weight)
- **B1:** Bayesian LoRA on an open-weight LLM — scale from toy model to real
