# Bayesian LLM

Estimating **epistemic uncertainty** in language models via Bayesian inference over weights.

Replace point-estimate weights with learned posterior distributions (mean + variance per weight). Sample weights multiple times, measure prediction disagreement via **mutual information** (MI). High MI on a given input = the model knows it doesn't know.

## What This Is

A **comparative study** of 4 Bayesian methods in a 2x2 matrix — (variational vs post-hoc) x (full weights vs LoRA) — tested at two scales on the same architecture, dataset, and evaluation protocol:

- **4-layer** (4L/4H/256d, ~16M params) on AG News (topic-split OOD)
- **16-layer** (16L/8H/512d, ~76M params) on The Pile (domain-split OOD)

The 16L experiments are run by an **agentic HP optimization pipeline** — an LLM agent (Claude) that automatically tunes hyperparameters, diagnoses failures, and proposes adjustments.

## Results

### 4-Layer Scale (AG News)

| Milestone | Method | Type | MI Ratio | Test ID PPL | Bayesian Params |
|-----------|--------|------|----------|-------------|-----------------|
| A0 | Deterministic baseline | — | — | 49.1 | 0 |
| A1 | Variational output head | Variational x Full | 1.36x | 56.3 | 25.7M |
| **A2** | **Variational FFN** | **Variational x Full** | **1.43x** | 53.5 | 4.2M |
| A3 | Variational FFN + Attn V | Variational x Full | <1.29x | 59.2 | 4.7M |
| B1 | Laplace FFN | Post-hoc x Full | 1.00x | 41.0 | 2.1M |
| B2 | BLoB LoRA | Variational x LoRA | 1.13x | 226.9 | 163K |
| B3-TFB | TFB LoRA | Post-hoc x LoRA | 1.10x | 224.6 | 82K |
| B3-LAP | Laplace LoRA | Post-hoc x LoRA | 1.00x | 224.6 | 82K |

### 16-Layer Scale (The Pile)

| Milestone | Method | Type | MI Ratio | Test ID PPL | Training Time |
|-----------|--------|------|----------|-------------|---------------|
| C0 | Deterministic baseline | — | — | 14.3 | 4.5 hrs |
| C1 | Variational FFN | Variational x Full | 1.32x | 21.9 | 3.3 hrs |
| C2 | Laplace FFN | Post-hoc x Full | 1.00x | 12.7 | 8s fit |
| **C3** | **BLoB LoRA** | **Variational x LoRA** | **1.53x** | 64.9 | 27 min |
| C4-TFB | TFB LoRA | Post-hoc x LoRA | 1.35x | 66.3 | 7 min fit |
| C4-LAP | Laplace LoRA | Post-hoc x LoRA | 1.00x | 65.4 | 17s fit |

### Cross-Scale Comparison

| Method | 4L MI Ratio | 16L MI Ratio | Scales? |
|--------|-------------|--------------|---------|
| Variational full (A2 / C1) | 1.43x | 1.32x | Slight decrease |
| BLoB LoRA (B2 / C3) | 1.13x | **1.53x** | Strong increase |
| TFB post-hoc LoRA (B3 / C4) | 1.10x | **1.35x** | Strong increase |
| Laplace full (B1 / C2) | 1.00x | 1.00x | Dead |
| Laplace LoRA (B3 / C4) | 1.00x | 1.00x | Dead |

## Key Findings

1. **Scaling inversion.** At 4L, full-weight variational (1.43x) beats LoRA (1.13x). At 16L, reversed — LoRA (1.53x) beats full-weight (1.32x). LoRA's rank-16 subspace constrains posteriors to meaningful directions.

2. **TFB (zero training) matches variational full-weight.** C4-TFB 1.35x ~ C1 1.32x, but TFB needs only a 7-minute binary search on a trained checkpoint vs 3.3 hours of Bayesian training.

3. **Diagonal Laplace is dead for LM OOD detection.** Four independent experiments (B1, B3-LAP, C2, C4-LAP) all produce MI ratio 1.00x. Diagonal Fisher curvature is flat at convergence.

4. **SVD-structured variance works where curvature fails.** TFB (1.35x) and Laplace (1.00x) are both post-hoc on LoRA. TFB succeeds because SVD captures geometric structure; Laplace fails because diagonal curvature carries no directional information.

5. **Post-hoc methods need subspace structure.** Post-hoc on full weights = 1.00x. Post-hoc on LoRA with SVD = 1.35x. The LoRA subspace makes post-hoc methods viable.

## Agentic Pipeline

The 16L experiments use an autonomous HP optimization pipeline (`experiments/c_pipeline.py`):

- **State machine:** CONFIGURE -> RUN -> ANALYZE -> DECIDE
- **LLM agent** (Claude/Codex via Provider pattern) diagnoses failures and proposes HP adjustments
- **Success gates** per milestone (e.g., MI ratio > 1.2x for C1)
- **Patience early-stop** + NaN detection + OOM recovery

The agent receives full context (AGENTS.md + briefing) and produces evidence-based reasoning — citing prior results, KL scaling math, and cross-experiment history.

See [`agents/pipeline-guide.md`](agents/pipeline-guide.md) and [`specs/c-pipeline-spec.md`](specs/c-pipeline-spec.md) for details.

## Skills

Reusable agent skills in [`agents/skills/`](agents/skills/):

- **[`check-paper-refs`](agents/skills/check-paper-refs/skill.md)** — Verify paper references against arXiv/Scholar. Catches wrong authors, outdated citations.
- **[`convert-md-to-pdf`](agents/skills/convert-md-to-pdf/skill.md)** — Markdown to PDF with LaTeX math, Mermaid diagrams, and Puppeteer rendering.
- **[`build-latex-pdf`](agents/skills/build-latex-pdf/skill.md)** — LaTeX to PDF via Docker + TeX Live. NeurIPS preprint format. Portable — copy the skill directory to any project.

## Repository Structure

```
minigpt/       Model, training, Bayesian layers, LoRA, Laplace, TFB
experiments/   Experiment scripts (A0-B3) + agentic pipeline (C0-C4)
tests/         217 tests (134 core + 83 pipeline)
configs/       YAML configs per experiment
specs/         Design documents
scripts/       Utilities (MLflow inspection, GPU profiling, checkpoint eval)
agents/        Detail documents (research findings, pipeline guide)
```

## Quick Start

```bash
uv sync                                          # install dependencies
uv run pytest tests/ -v                          # 217 tests
uv run ruff check minigpt/ experiments/ tests/   # lint

# Example experiments (require CUDA):
python experiments/a0_baseline.py --config configs/a0_agnews.yaml       # deterministic baseline
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml      # variational FFN (best at 4L)
python experiments/c_pipeline.py --milestone c3 --provider claude       # agentic pipeline (16L BLoB LoRA)
python experiments/c_pipeline.py --compare                              # cross-scale comparison report
```

Requires Python 3.11+ and CUDA-enabled PyTorch for GPU training.

## Tech Stack

- **PyTorch** + `torch.distributions` (no extra Bayesian libraries)
- **tiktoken** BPE tokenization (GPT-2 encoding, vocab 50257)
- **MLflow** local tracking (`sqlite:///mlflow.db`)
- **uv** for dev tooling; global Python for CUDA training
- Target hardware: RTX 4070 (~12 GB VRAM)

## References

- **BLoB** (NeurIPS 2024) — Bayesian LoRA by backprop. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- **TFB** (NeurIPS 2025) — Training-Free Bayesianization. [arXiv:2412.05723](https://arxiv.org/abs/2412.05723)
- **Laplace-LoRA** (2023) — Laplace approximation on LoRA params. [arXiv:2308.13111](https://arxiv.org/abs/2308.13111)
- **Laplace Redux** (NeurIPS 2021) — Effortless Bayesian Deep Learning. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)
- **ICLA** (WACV 2025) — Identity Curvature Laplace for OOD. [arXiv:2312.10464](https://arxiv.org/abs/2312.10464)
