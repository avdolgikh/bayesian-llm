# Bayesian LLM

Estimating **epistemic uncertainty** in language models via Bayesian inference over weights.

Replace point-estimate weights with learned posterior distributions (mean + variance per weight). Sample weights multiple times, measure prediction disagreement via **mutual information** (MI). High MI on a given input = the model knows it doesn't know.

## Paper

**Epistemic Uncertainty in Language Models via Bayesian Inference over Weights: A Comparative Study**

A comparative study of 4 Bayesian methods in a 2x2 matrix — (variational vs post-hoc) x (full weights vs LoRA) — tested at two scales on the same architecture, dataset, and evaluation protocol.

- [`paper/paper.pdf`](paper/paper.pdf) — compiled paper (NeurIPS preprint format, 8 pages)
- [`paper/paper.tex`](paper/paper.tex) — LaTeX source
- [`paper/arxiv-submission.zip`](paper/arxiv-submission.zip) — arXiv submission package

## Results

### OOD Detection (16-Layer, The Pile, 76M params)

Community-standard evaluation: 500 ID + 500 OOD sequences, N=20 MC samples, 95% bootstrap CIs.

| Method | Type | AUROC [95% CI] | MI Ratio | FPR@95 | ECE |
|--------|------|-----------------|----------|--------|-----|
| TFB LoRA | Post-hoc x LoRA | **0.917** [0.900, 0.933] | 1.35x | 0.384 | 0.022 |
| BLoB LoRA | Variational x LoRA | **0.909** [0.890, 0.925] | 1.53x | 0.424 | 0.044 |
| MC Dropout | Baseline | **0.898** [0.877, 0.917] | -- | 0.368 | 0.012 |
| Variational FFN | Variational x Full | 0.874 [0.852, 0.895] | 1.32x | 0.494 | 0.023 |
| Deterministic | -- | 0.591 [0.556, 0.626] | -- | 0.794 | 0.022 |
| Diag. Laplace FFN | Post-hoc x Full | 0.536 [0.500, 0.572] | 1.00x | 0.934 | 0.033 |
| Diag. Laplace LoRA | Post-hoc x LoRA | 0.494 [0.459, 0.529] | 1.00x | 0.956 | 0.034 |

### Cross-Scale Comparison (MI Ratio)

| Method | 4L (16M params) | 16L (76M params) | Scales? |
|--------|------------------|-------------------|---------|
| Variational full | 1.43x | 1.32x | Slight decrease |
| BLoB LoRA | 1.13x | **1.53x** | Strong increase |
| TFB post-hoc LoRA | 1.10x | **1.35x** | Strong increase |
| Diag. Laplace full | 1.00x | 1.00x | Dead |
| Diag. Laplace LoRA | 1.00x | 1.00x | Dead |

Full results including production benchmarks and AUROC-vs-N tradeoffs: [`report.md`](report.md).

## Key Findings

1. **Scaling inversion (observational).** At 4L, full-weight variational (1.43x) beats LoRA (1.13x). At 16L, reversed — LoRA (1.53x) beats full-weight (1.32x). One hypothesis: LoRA's rank-16 subspace constrains posteriors to meaningful directions. Confounded by training procedure, backbone quality, and parameter count.

2. **TFB (zero training) matches variational full-weight.** C4-TFB AUROC 0.917 vs C1 0.874, but TFB needs only a 7-minute binary search on a trained checkpoint vs 3.3 hours of Bayesian training.

3. **Diagonal Laplace is dead for LM OOD detection.** Four independent experiments all produce MI ratio 1.00x. Diagonal Fisher curvature is flat at convergence. Does not extend to KFAC or full-Hessian variants (untested).

4. **SVD-structured variance works where curvature fails.** TFB succeeds (AUROC 0.917) because SVD captures geometric structure; diagonal Laplace fails (AUROC ~0.5) because curvature carries no directional information. Both are post-hoc on LoRA.

5. **Post-hoc methods need subspace structure.** Post-hoc on full weights = dead. Post-hoc on LoRA with SVD = 0.917 AUROC. The LoRA subspace makes post-hoc methods viable.

6. **MC Dropout is surprisingly competitive.** Zero extra training, just dropout-at-inference: AUROC 0.898 — overlapping CIs with trained Bayesian methods. Best calibration (ECE=0.012).

7. **N=3 MC samples capture most of the signal.** AUROC jumps 0.50 -> 0.86 at N=3 (97% of N=20 signal). Diminishing returns beyond N=5. Production sweet spot: N=3 at 50ms/seq, 382 MB VRAM.

8. **Mean-weights inference is production-ready.** Posterior mean perplexity matches MC-averaged within 4%. Use mean weights for serving (zero overhead), N=3 MC for uncertainty scoring.

9. **Bootstrap CIs show overlapping top methods.** TFB, BLoB, and MC Dropout all overlap — none is statistically significantly better at this sample size. Variational FFN is clearly below. Diagonal Laplace is clearly dead.

## Agentic Pipeline

The 16L experiments use an autonomous HP optimization pipeline (`experiments/c_pipeline.py`):

- **State machine:** CONFIGURE -> RUN -> ANALYZE -> DECIDE
- **LLM agent** (Claude/Codex via Provider pattern) diagnoses failures and proposes HP adjustments
- **Success gates** per milestone (e.g., MI ratio > 1.2x for C1)
- **Patience early-stop** + NaN detection + OOM recovery

The agent receives full context (AGENTS.md + briefing) and produces evidence-based reasoning — citing prior results, KL scaling math, and cross-experiment history.

See [`agents/pipeline-guide.md`](agents/pipeline-guide.md) and [`specs/c-pipeline-spec.md`](specs/c-pipeline-spec.md) for details.

## Skills

Reusable agent skills in [`agents/skills/`](agents/skills/). Two-layer architecture: portable `agents/skills/*/skill.md` (repo-agnostic, full docs) + thin `.claude/skills/*/SKILL.md` wrappers (project defaults).

- **[`check-paper-refs`](agents/skills/check-paper-refs/skill.md)** — Verify paper references: ground truth check (CI) + live arXiv API verification + multi-agent cross-review. Script: `scripts/verify_references.py`.
- **[`convert-md-to-pdf`](agents/skills/convert-md-to-pdf/skill.md)** — Markdown to PDF with LaTeX math, Mermaid diagrams, and Puppeteer rendering.
- **[`build-latex-pdf`](agents/skills/build-latex-pdf/skill.md)** — LaTeX to PDF via Docker + TeX Live. NeurIPS preprint format. Portable — copy the skill directory to any project.
- **[`build-arxiv-submission`](agents/skills/build-arxiv-submission/skill.md)** — LaTeX to arXiv submission zip. Auto-detects figures/bib/sty, rewrites paths, verifies compilation via Docker.

## Repository Structure

```
minigpt/       Model, training, Bayesian layers, LoRA, Laplace, TFB, uncertainty metrics
experiments/   Experiment scripts (A0-B3) + agentic pipeline (C0-C4)
tests/         288 tests (134 core + 83 pipeline + 71 metrics/eval)
configs/       YAML configs per experiment
paper/         LaTeX paper, compiled PDF, arXiv submission zip
figures/       Generated paper figures (PDF/PNG) — from scripts/generate_figures.py
scripts/       Utilities (MLflow inspection, GPU profiling, checkpoint eval, figure generation)
specs/         Design documents
docs/          arXiv requirements, metrics guide, reference PDFs
agents/        Detail documents, portable skills, pipeline guide
```

## Quick Start

```bash
uv sync                                          # install dependencies
uv run pytest tests/ -v                          # 288 tests
uv run ruff check minigpt/ experiments/ tests/   # lint

# Example experiments (require CUDA):
python experiments/a0_baseline.py --config configs/a0_agnews.yaml       # deterministic baseline
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml      # variational FFN (best at 4L)
python experiments/c_pipeline.py --milestone c3 --provider claude       # agentic pipeline (16L BLoB LoRA)
python experiments/c_pipeline.py --compare                              # cross-scale comparison report

# Evaluation & benchmarks:
python scripts/eval_c_checkpoints.py --from-scores data/d1_scores.pt --bootstrap  # AUROC + CIs
python scripts/benchmark_inference.py                                              # latency/VRAM
python scripts/eval_mc_dropout.py                                                  # MC Dropout baseline
python scripts/generate_figures.py --png                                           # paper figures
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

## License

MIT — see [LICENSE](LICENSE).
