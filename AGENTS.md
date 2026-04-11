# Repository Guidelines

## Goal
**Estimate epistemic uncertainty in LLMs via Bayesian inference over weights.**

The core idea: replace point-estimate weights with learned posterior distributions, then measure how much the model's predictions disagree across weight samples (mutual information). High MI = "the model knows what it doesn't know" — it's uncertain about out-of-distribution inputs at the weight level, not just the token level.

**Comparative study:** 2×2 matrix — (variational vs post-hoc) × (full weights vs LoRA) — at two scales (4L and 16L). Active spec: `specs/comparative-bayesian-llm-study.md`. Full results: `report.md`.

## Project Structure

```
minigpt/          # Python package — all model code
  model.py        # MiniGPT architecture (deterministic + selective Bayesian)
  layers.py       # Bayesian layers (BayesianLinear, context managers, sigma stats)
  data.py         # Dataset loading + BPE tokenization — TinyShakespeare, AG News, Pile
  train.py        # Training loop (cross-entropy + ELBO)
  evaluate.py     # Perplexity, text generation
  config.py       # YAML config ↔ dataclass bridge
  uncertainty.py  # Epistemic uncertainty (MI via MC sampling)
  laplace.py      # Post-hoc Laplace: curvature fitting, sampling, context manager
  lora.py         # LoRA: BLoBLoRALinear, DeterministicLoRALinear, LoRAConfig, inject_lora()
  tfb.py          # TFB: Training-Free Bayesianization (SVD variance search, sampling)
configs/          # YAML config files per experiment (a0–b3 AG News, c0–c4 Pile)
experiments/      # Runnable scripts + pipeline
  experiment_setup.py  # Shared setup: CLI parsing, config, data, model, device
  eval_utils.py        # Shared eval: perplexity suite, MI suite, qualitative eval
  mlflow_utils.py      # Shared MLflow: context, logging helpers
  runner.py            # A-series orchestrator — A1/A2/A3 are thin wrappers
  pipeline_runner.py   # Generic pipeline engine: PipelineRunnerBase, MilestonePolicy, RuntimeHooks
  c_milestones.py      # C-specific: templates, gates, knobs, comparison report
  c_pipeline.py        # C pipeline CLI: wires hooks+policy, providers, entry point
  agent_briefing.md    # HP tuning playbook injected into agent prompts
scripts/          # Utilities (dump_mlflow_run, compare_runs, profile_gpu, eval_checkpoint, generate_figures, benchmark_inference, eval_mc_dropout, verify_mean_weights, verify_references)
tests/            # pytest (288 tests: 134 core + 83 pipeline + 71 metrics/eval)
data/             # Local datasets + saved scores (gitignored)
paper/            # LaTeX paper, compiled PDF, arXiv submission zip, figures/
figures/          # Generated paper figures (PDF/PNG) — from scripts/generate_figures.py
specs/            # Design documents
docs/             # arXiv requirements, metrics guide, reference PDFs
agents/           # Detail documents, portable skills (read on demand, not every task)
```

## Hard Rules
- **No notebooks.** Only `.py` scripts.
- **No extra Bayesian libraries.** Only `torch.distributions`. Manual implementation preferred.
- **No modern transformer tricks** (RoPE, SwiGLU, MoE, etc.). Keep miniGPT basic.
- **Document on the fly** in this file — during implementation, not after.
- **No unsolicited AGENTS.md cleanup.** Do not reformat or rewrite text unless explicitly requested.
- **Explicit configs.** Every parameter in YAML — never rely on code defaults.
- **LaTeX formulas in specs.** All formulas in `specs/` documents must use LaTeX.
- **Keep repo docs fresh.** When any of the following change, update the corresponding files immediately — not after, not in a batch:
  - **New skill created** → update AGENTS.md Skills section + README.md Skills section.
  - **Test count changes** → update AGENTS.md project structure + README.md (both the structure block and Quick Start commands).
  - **Milestone completes** → update `report.md` first (source of truth), then README.md summary, then AGENTS.md milestones.
  - **Paper status changes** (new build, figure change, submission) → update AGENTS.md Paper Publishing section.
  - **New script added** → update AGENTS.md project structure scripts line + README.md Quick Start if user-facing.
  - **Reusable workflow emerges** (build, verify, convert, check) → wrap as portable skill in `agents/skills/` + thin wrapper in `.claude/skills/`. Update AGENTS.md + README.md Skills sections.
- **Keep AGENTS.md lean.** This file is the entry point, not the encyclopedia. When a section grows beyond a concise summary, extract the detail to an `agents/*.md` file immediately and leave a summary + link here. Rule of thumb: if a section exceeds ~10 lines of dense content, it belongs in a detail doc.

## Detail Documents
**AGENTS.md is the entry point — not the whole picture.** It contains rules, structure, milestones, and status. The detail documents below extend it with deep technical content. Read them on demand (not every task). **Keep them fresh using the same on-the-fly principle as AGENTS.md itself:**
- **Technical change** (new dataset, new metric, config change, new tests, figures) → update `agents/technical-reference.md`.
- **Design decision made** (why method X, why not Y, tradeoff rationale) → update `agents/design-rationale.md`.
- **Pipeline change** (new provider, new hook, CLI change, template HPs) → update `agents/pipeline-guide.md`.
- **Milestone completes or new detail surfaces** (run IDs, bug fixes, interim findings) → update `agents/milestone-history.md`.

| Document | What it covers | When to read |
|----------|---------------|--------------|
| [`agents/technical-reference.md`](agents/technical-reference.md) | Datasets, tokenization, experiment tracking, uncertainty, config, tests, figures, references | Working on data, configs, metrics, or need API details |
| [`agents/design-rationale.md`](agents/design-rationale.md) | Why specific technical decisions were made (A1-B3, C scaling hypothesis, infra fixes) | Need to understand *why* something was built a certain way |
| [`agents/pipeline-guide.md`](agents/pipeline-guide.md) | Agentic HP optimization pipeline (architecture, providers, features, CLI, template HPs) | Working on or running the C pipeline |
| [`agents/milestone-history.md`](agents/milestone-history.md) | Detailed record of every milestone — results, MLflow run IDs, bug fixes, pipeline events | Investigating what happened in a specific milestone or tracing a result |

## Skills
Two-layer architecture: **portable** `agents/skills/*/skill.md` (YAML frontmatter + full docs, repo-agnostic) + **thin wrappers** `.claude/skills/*/SKILL.md` (frontmatter + 1 line with project defaults). Portable skills use `<skill-dir>` placeholders — copy the whole directory to any project.

- [`agents/skills/check-paper-refs/`](agents/skills/check-paper-refs/skill.md) — Verify paper references: ground truth check (CI) + live arXiv verification + multi-agent cross-review. Script: `scripts/verify_references.py`. Ground truth: `paper/references_ground_truth.json`.
- [`agents/skills/convert-md-to-pdf/`](agents/skills/convert-md-to-pdf/skill.md) — Markdown -> PDF with MathJax + Mermaid + Puppeteer. Scripts: `build-pdf.ps1`/`.sh`, `md-to-pdf.config.js`. Prereqs: `npm install -g md-to-pdf @mermaid-js/mermaid-cli`.
- [`agents/skills/build-latex-pdf/`](agents/skills/build-latex-pdf/skill.md) — LaTeX -> PDF via Docker + TeX Live + pdflatex. NeurIPS preprint format, Times fonts, booktabs, natbib. Self-contained (`.sty` + build scripts bundled). Prereqs: Docker.
- [`agents/skills/build-arxiv-submission/`](agents/skills/build-arxiv-submission/skill.md) — LaTeX -> arXiv submission zip. Auto-detects figures/bib/sty, rewrites paths, verifies compilation via Docker. Scripts: `build.sh`/`build.ps1`. Prereqs: Docker (optional, for verification).

## Milestones

All milestones complete. Detail: [`agents/milestone-history.md`](agents/milestone-history.md). Full results: [`report.md`](report.md).

| Milestone | Method | Status | Key Result |
|-----------|--------|--------|------------|
| A0 | Deterministic baseline (4L) | DONE | test_id_ppl=49.1 |
| A2 | Variational FFN (4L) | DONE | MI **1.43x**, best at 4L |
| A3 | Variational FFN + Attn V (4L) | CLOSED | Negative vs A2 |
| B1 | Laplace FFN (4L) | DONE | MI 1.00x, diagonal Fisher flat |
| B2 | BLoB LoRA (4L) | DONE | MI **1.13x**, weak positive |
| B3 | Post-hoc LoRA (4L) | DONE | TFB 1.10x, Laplace 1.00x |
| C0-C4 | All methods at 16L scale | DONE | **Scaling inversion: LoRA > full-weight** |
| D0-D3 | Metrics, eval, benchmarks | DONE | AUROC + CIs, N=3 knee, production recipe |
| P1-P4 | Bootstrap CIs, MC Dropout, paper polish | DONE | MC Dropout AUROC 0.898 |
| Paper | LaTeX + arXiv submission | DONE | 8 pages, 3 figs, 5 tables |

Paper specs: `specs/paper-improvements.md`, `specs/paper-reviewer-concerns.md`.

### Paper Publishing (arXiv)
- **Paper: COMPLETE** — `paper/paper.tex` (NeurIPS preprint format, 8 pages, 7 sections, 5 tables, 3 figures, 11 references). `paper/references.bib`.
- **PDF: BUILT** — `paper/paper.pdf`. Skill: `/build-latex-pdf paper/paper.tex`. Docker + TeX Live + pdflatex. Times fonts, booktabs, natbib.
- **arXiv submission: BUILT** — `paper/arxiv-submission.zip`. Skill: `/build-arxiv-submission paper/paper.tex`. Auto-detects figures/bib/sty, rewrites paths, verifies compilation.
- **arXiv requirements:** `docs/arxiv-requirements.md`. Target categories: `cs.LG`, cross-list `stat.ML`.
- **Figures in paper:** 3 PNG figures in `paper/figures/` (conceptual posteriors, method matrix, BLoB architecture). Source: `scripts/generate_figures.py`.
- **Postponed experiments:** Deep Ensembles, LoRA ablation, multi-seed runs, KFAC Laplace — all documented, none blocking publication.

## Environment & Tooling
- **`uv`** for dev tooling (lint, test, deps). **Global Python** (CUDA PyTorch) for GPU training only.
- Target: RTX 4070 (~12 GB VRAM). AMP auto-enabled on CUDA. If uv hits permission errors: `UV_CACHE_DIR=.uv-cache`
- PyTorch + `torch.distributions`. No JAX, no TensorFlow.

## Build & Dev Commands
```bash
uv sync                                          # install deps
uv run pytest tests/ -v                          # 288 tests
uv run ruff check minigpt/ experiments/ tests/   # lint
python experiments/a0_baseline.py --config configs/a0_agnews.yaml          # A0
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml         # A2
python experiments/b3_post_hoc_lora.py --phase full --config configs/b3_lora_agnews.yaml  # B3
python experiments/c_pipeline.py --milestone c0 --provider claude --provider-model sonnet  # C pipeline
python experiments/c_pipeline.py --compare                                                  # comparison
python scripts/dump_mlflow_run.py <run_id>       # inspect MLflow run
```

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`): `ruff check` → `pytest`. No GPU in CI.
**Always run lint + tests locally before committing:** `uv run ruff check minigpt/ experiments/ tests/ && uv run pytest tests/ -x`

## Coding Style
4 spaces, 100-char lines (ruff). `snake_case` functions, `PascalCase` classes. Type hints on public APIs.

## Commit Guidelines
One-line messages, Conventional Commits (`feat:`, `fix:`, `docs:`, etc.).
**Never mention AI assistants** in commits, comments, PRs, or code.

## Security
No secrets or large binaries in git. `data/`, `mlflow.db`, `mlruns/` are gitignored.

## Future Work (Parked)
Non-Bayesian: RoPE, SwiGLU, KV-cache, RMSNorm, GQA. Done: AMP, Flash Attention.
Pipeline refactoring: `specs/pipeline-refactoring-spec.md`.
