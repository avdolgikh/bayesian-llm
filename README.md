# Bayesian LLM

Estimating **epistemic uncertainty** in language models via Bayesian inference over weights.

Replace point-estimate weights with learned posterior distributions (mean + variance per weight). Sample weights multiple times, measure prediction disagreement via mutual information (MI). High MI on a given input = the model knows it doesn't know.

## Study Design

Comparative study: **4 Bayesian methods** in a 2×2 matrix — (variational vs post-hoc) × (full weights vs LoRA). Tested at two scales:

- **4-layer** (4L/4H/256d, ~16M params) on AG News (topic-split)
- **16-layer** (16L/8H/512d, ~76M params) on The Pile (domain-split)

The 16L experiments are run by an **agentic HP optimization pipeline** — an LLM agent (Claude) that automatically tunes hyperparameters, diagnoses failures, and proposes adjustments. See `experiments/c_pipeline.py`.

Results and cross-scale comparison: [`report.md`](report.md).

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

Requires Python 3.12+ and CUDA-enabled PyTorch for GPU training.

## Repository Structure

```
minigpt/       Model, training, Bayesian layers, LoRA, Laplace, TFB
experiments/   Experiment scripts (A0–B3) + agentic pipeline (C0–C4)
tests/         217 tests (134 core + 83 pipeline)
configs/       YAML configs per experiment
specs/         Design documents
scripts/       Utilities (MLflow inspection, GPU profiling, checkpoint eval)
```
