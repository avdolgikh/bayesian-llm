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
scripts/          # Utilities (dump_mlflow_run, compare_runs, profile_gpu, eval_checkpoint)
tests/            # pytest (217 tests: 134 core + 83 pipeline)
data/             # Local datasets (gitignored)
specs/            # Design documents
docs/             # PDF papers (theory reference)
agents/           # Detail documents (read on demand, not every task)
```

## Hard Rules
- **No notebooks.** Only `.py` scripts.
- **No extra Bayesian libraries.** Only `torch.distributions`. Manual implementation preferred.
- **No modern transformer tricks** (RoPE, SwiGLU, MoE, etc.). Keep miniGPT basic.
- **Document on the fly** in this file — during implementation, not after.
- **No unsolicited AGENTS.md cleanup.** Do not reformat or rewrite text unless explicitly requested.
- **Explicit configs.** Every parameter in YAML — never rely on code defaults.
- **LaTeX formulas in specs.** All formulas in `specs/` documents must use LaTeX.

## Detail Documents
- [`agents/technical-reference.md`](agents/technical-reference.md) — Datasets, tokenization, experiment tracking, uncertainty measurement, config system, tests listing, references
- [`agents/design-rationale.md`](agents/design-rationale.md) — Why specific technical decisions were made (A1-B3, C scaling hypothesis, infra fixes)
- [`agents/pipeline-guide.md`](agents/pipeline-guide.md) — Agentic HP optimization pipeline (architecture, providers, features, CLI, template HPs)

## Milestones

- **A0: DONE** — Deterministic baseline (4L/4H/256d, 16M params). test_id_ppl=49.11.
- **A1: DONE** — Bayesian output head. MI ratio **1.36x**. Ceiling: vocabulary-level only.
- **A2: DONE** — Bayesian FFN. MI ratio **1.43x batch / 1.70x qual**. Best method at 4L.
- **A3: CLOSED** — Bayesian FFN + attn V. Negative result vs A2 (4 runs, all worse).
- **B1: DONE (NEGATIVE)** — Post-hoc Laplace on FFN. MI ratio 1.00x. Diagonal Fisher too flat.
- **B2: DONE (WEAK POSITIVE)** — BLoB LoRA. MI ratio **1.13x**. Weaker than A2 at 4L.
- **B3: DONE (MIXED)** — Post-hoc LoRA. **B3-TFB: 1.10x** (SVD works). **B3-LAP: 1.00x** (Laplace fails).
- **C: DONE** — Scaled replication at 16L/8H/512d (~76M params) on The Pile (domain-split). C0 ppl=14.3, C1 MI **1.32x**, C2 MI 1.00x, C3 MI **1.53x**, C4-TFB MI **1.35x**, C4-LAP MI 1.00x. **Scaling inversion: LoRA > full-weight at 16L.**
- **D0: DONE** — Metrics framework (AUROC, FPR@95, AUPRC, ECE, Brier, NLL, AURC) in `minigpt/uncertainty.py`. 45 tests.
- **D1: DONE** — Eval of all 6 C checkpoints. C3 AUROC=0.916, C4-TFB=0.917, C1=0.876. Laplace dead (C2=0.536, C4-LAP=0.494). Script: `scripts/eval_c_checkpoints.py`. Spec: `specs/d1-eval-c-checkpoints.md`.
- **D2: DONE** — Mean-weights inference. `compute_perplexity_mc()` in `evaluate.py`. Verified mean-weights PPL ≈ MC-averaged PPL: C1 diff=0.29%, C3 diff=3.93% (gate <5%). 9 tests. Script: `scripts/verify_mean_weights.py`. Spec: `specs/d2-mean-weights-spec.md`.
- **D3: DONE** — Production benchmarks (RTX 4070). LoRA MC N=5: AUROC=0.879, 84ms, 382 MB. **N=3 is the knee** — AUROC jumps 0.50→0.86 (97% of N=20 signal). LoRA MC uses 28% less VRAM than full variational (382 vs 534 MB). Script: `scripts/benchmark_inference.py`. Spec: `specs/prod-uncertainty-approaches.md`.
- **P1: DONE** — Bootstrap 95% CIs for all methods. `bootstrap_ci()` in `uncertainty.py`. 11 tests (288 total). `eval_c_checkpoints.py --bootstrap --save-scores`. Saved: `data/d1_scores.pt`. Fast recompute: `--from-scores data/d1_scores.pt --bootstrap`.
- **P2: DONE** — MC Dropout baseline. `enable_dropout()` CM in `layers.py`. 6 tests. Script: `scripts/eval_mc_dropout.py`. **AUROC 0.898 [0.877, 0.917]** — surprisingly competitive with trained Bayesian methods (zero extra training). Saved: `data/mc_dropout_scores.pt`.
- **P3: DONE** — Narrow "Laplace" → "diagonal Laplace" throughout paper. Table labels, 2×2 matrix, body text.
- **P4: DONE** — Reframe LoRA vs full-weight claim: observational (not causal), list 3 confounds. All Section 6 "why" paragraphs rewritten as hypotheses. MC Dropout discussion added.
- **Paper tables: DONE** — Tables 1-2 updated with bootstrap CIs + MC Dropout row. Point estimates from `data/d1_scores.pt` (self-consistent with CIs). Section 3.6 (MC Dropout method) added.
- **References: FIXED** — 4/11 references had wrong authors (BLoB, TFB, Laplace-LoRA, ScalaBL). All verified against arXiv/proceedings. Orphaned Lakshminarayanan ref removed.

Full results and cross-scale comparison: `report.md`. Paper improvements spec: `specs/paper-improvements.md`. Reviewer concerns: `specs/paper-reviewer-concerns.md`.

## Environment & Tooling
- **`uv`** for dev tooling (lint, test, deps). **Global Python** (CUDA PyTorch) for GPU training only.
- Target: RTX 4070 (~12 GB VRAM). AMP auto-enabled on CUDA. If uv hits permission errors: `UV_CACHE_DIR=.uv-cache`
- PyTorch + `torch.distributions`. No JAX, no TensorFlow.

## Build & Dev Commands
```bash
uv sync                                          # install deps
uv run pytest tests/ -v                          # 217 tests
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
