# Bayesian LLM

Estimating **epistemic uncertainty** in language models via Bayesian inference over weights.

Replace point-estimate weights with learned posterior distributions (mean + variance per weight). Sample weights multiple times, measure how much predictions disagree — that disagreement (mutual information) is the model's epistemic uncertainty. High MI on a given input means the model knows it doesn't know.

## Approach

Small GPT (4-layer, 256-dim) trained on topic-split AG News. Train on World + Sports; hold out Business + Sci/Tech as OOD. Progressively make layers Bayesian and compare variational vs post-hoc methods across full weights and LoRA.

## Results

| Milestone | Method | MI ratio (OOD/ID) | Status |
|-----------|--------|-------------------|--------|
| **A0** | Deterministic baseline | — | Done |
| **A1** | Variational output head | 1.36x | Done |
| **A2** | Variational FFN | **1.70x** | Done |
| **A3** | Variational FFN + attn V | < A2 | Closed (negative) |
| **B1** | Post-hoc Laplace FFN | 1.00x | Closed (negative) |
| **B2** | BLoB LoRA (variational) | 1.13x | Done (weak positive) |

**Key findings:** Variational FFN (A2) is the strongest — 4.2M Bayesian params, MI cleanly separates ID/OOD. Post-hoc Laplace fails: diagonal Fisher curvature too flat at convergence. BLoB LoRA detects OOD at batch level but weaker than full variational.

## Quick start

```bash
uv sync
python experiments/a0_baseline.py --config configs/a0_agnews.yaml   # deterministic baseline
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml  # Bayesian FFN
python experiments/b2_blob_lora.py --config configs/b2_blob.yaml    # BLoB LoRA
uv run ruff check minigpt/ experiments/ tests/                      # lint
uv run pytest tests/ -v                                              # 84 tests
```

Requires Python 3.12+ and CUDA-enabled PyTorch for GPU training.

## Next steps

- **B3:** Laplace-LoRA (post-hoc on trained LoRA params)
- **C:** Scaled replication (16L miniGPT) of full A1–B3 suite + Flipout — final comparison paper
