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
```

## Hard Rules
- **No notebooks.** Only `.py` scripts.
- **No extra Bayesian libraries.** Only `torch.distributions`. Manual implementation preferred.
- **No modern transformer tricks** (RoPE, SwiGLU, MoE, etc.). Keep miniGPT basic.
- **Document on the fly** in this file — during implementation, not after.
- **No unsolicited AGENTS.md cleanup.** Do not reformat or rewrite text unless explicitly requested.
- **Explicit configs.** Every parameter in YAML — never rely on code defaults.
- **LaTeX formulas in specs.** All formulas in `specs/` documents must use LaTeX.

## Datasets

**TinyShakespeare:** ~1MB, ~304K BPE tokens. Single-domain. `load_shakespeare()` auto-downloads.

**AG News:** 127.6K articles, 4 categories (1=World, 2=Sports, 3=Business, 4=Sci/Tech). Topic split: ID (default World+Sports) → train/val/test_id (80/10/10); OOD (default Business+Sci/Tech) → test_ood. ~2.4M train tokens. Configurable via `data.id_categories`, `data.ood_categories`.

**The Pile (C milestone):** Domain-split from `ArmelR/the-pile-splitted` (HuggingFace). ID: Wikipedia + StackExchange (~200M tokens). OOD: ArXiv, FreeLaw, PubMed. LoRA fine-tune: HackerNews. Cache: `data/pile/{domain}_{token_limit}.pt`.

**Dispatcher:** `load_dataset(cfg, tokenizer)` → `{"train", "val", "test_id", "test_ood"}`.

## Tokenization
BPE via `tiktoken` (GPT-2 encoding, vocab_size=50257).

## Experiment Tracking
- **MLflow** (local, `sqlite:///mlflow.db`). `mlflow.db` and `mlruns/` are gitignored.
- Logs: hyperparams, train/val loss, perplexity, LR, test perplexity, MI metrics, sigma stats.
- Tags: `dataset`, `milestone`, `gpu`.
- `--no-mlflow` flag to disable. UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Environment & Tooling
- **`uv`** for dev tooling (lint, test, deps). **Global Python** (CUDA PyTorch) for GPU training only.
- Target: RTX 4070 (~12 GB VRAM). AMP auto-enabled on CUDA.
- If uv hits permission errors: `UV_CACHE_DIR=.uv-cache`

## Framework
PyTorch + `torch.distributions`. No JAX, no TensorFlow.

## Milestones

- **A0: DONE** — Deterministic baseline (4L/4H/256d, 16M params). test_id_ppl=49.11.
- **A1: DONE** — Bayesian output head. MI ratio **1.36x**. Ceiling: vocabulary-level only.
- **A2: DONE** — Bayesian FFN. MI ratio **1.43x batch / 1.70x qual**. Best method at 4L.
- **A3: CLOSED** — Bayesian FFN + attn V. Negative result vs A2 (4 runs, all worse).
- **B1: DONE (NEGATIVE)** — Post-hoc Laplace on FFN. MI ratio 1.00x. Diagonal Fisher too flat.
- **B2: DONE (WEAK POSITIVE)** — BLoB LoRA. MI ratio **1.13x**. Weaker than A2 at 4L.
- **B3: DONE (MIXED)** — Post-hoc LoRA. **B3-TFB: 1.10x** (SVD works). **B3-LAP: 1.00x** (Laplace fails).
- **C: DONE** — Scaled replication at 16L/8H/512d (~76M params) on The Pile (domain-split). C0 ppl=14.3, C1 MI **1.32x**, C2 MI 1.00x, C3 MI **1.53x**, C4-TFB MI **1.35x**, C4-LAP MI 1.00x. **Scaling inversion: LoRA > full-weight at 16L.**

Full results and cross-scale comparison: `report.md`.

## Bayesian Layer Strategy
1. Output head (A1) — simplest, proves pipeline
2. FFN layers (A2) — strongest epistemic signal (FFN stores factual knowledge)
3. Attention V (A3) — closed (negative, adds noise without improving discrimination)

## Epistemic Uncertainty Measurement
- Temperature = 0. All stochasticity from Bayesian weights.
- N forward passes (10–30) with weight sampling.
- Primary metric: **MI** (predictive_entropy − expected_entropy) — pure epistemic uncertainty.
- Secondary: flip rate, per-token MI scores.
- MI ratio: MI_OOD / MI_ID. Values >1.0 indicate OOD detection.

## Configuration System
Pipeline: `DEFAULT_CONFIG → YAML file → CLI --set overrides → validate`.

Key conventions:
- `vocab_size` always from tokenizer, never config.
- `model.bayes_head` / `model.bayes_ffn` / `model.bayes_attn_v` for Bayesian component configs.
- `train.kl_weight` is global for all enabled Bayesian components.
- `posthoc_method: "laplace"|"tfb"` disambiguates post-hoc dispatch.
- **PyYAML gotcha:** write `3.0e-4` not `3e-4` (parsed as string without decimal).
- Checkpoint resume: saves full config, `best_val_loss`, RNG states. Best-checkpoint: ELBO for Bayesian, CE for deterministic.

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

## Tests (217 total: 134 core + 83 pipeline)

```
tests/
  test_model.py               # Model architecture, weight tying, perplexity
  test_data.py                # Data loading, split sizes
  test_reproducibility.py     # Seed determinism
  test_bayesian.py            # Bayesian layers, KL, sampling, MI, architecture paths
  test_lora.py                # LoRA injection, BLoB forward/KL, config, context managers, grads
  test_laplace.py             # Laplace fitting, sampling, selection modes, roundtrip, LoRA selection
  test_b3_deterministic_lora.py  # DeterministicLoRALinear forward, no-KL, param count
  test_tfb.py                 # TFB SVD, variance structure, search convergence, sampling
  test_pile_data.py           # Pile data loader: splits, domains, config, caching, multiprocessing
  test_c_pipeline.py          # Pipeline: CLI, config gen, helpers, runner state/loop, policies,
                              #   posthoc templates/fit/integration, BLoB→DeterministicLoRA conversion
```

## Commit Guidelines
One-line messages, Conventional Commits (`feat:`, `fix:`, `docs:`, etc.).
**Never mention AI assistants** in commits, comments, PRs, or code.

## Security
No secrets or large binaries in git. `data/`, `mlflow.db`, `mlruns/` are gitignored.

---

## Key Technical Decisions

### A1: Sigma window and output head ceiling
Optimal σ ≈ 0.1–0.3. Below → posterior collapse. Above → uniform noise. Output head ceiling at MI ratio 1.2–1.4x (vocabulary-level uncertainty only, factual knowledge lives in FFN).

### A2: init_rho=-2 unlocks posterior learning
init_rho=-3: sigma frozen at init. init_rho=-2: sigma spread [0.036, 0.966], model differentiates which weights need certainty. FFN detects topic-level uncertainty (Business ≈ Sci/Tech), not vocabulary-level (A1: Sci/Tech >> Business). 4.2M Bayesian params outperform A1's 25.7M.

### B1: Diagonal Laplace fails for language models
Curvature at convergence is flat (~10⁻⁵). With damping=1.0: posterior variance ≈ 1.0 everywhere → isotropic perturbations → MI ratio 1.00x. Both identity-curvature (ICLA) and per-sample Fisher produce the same failure. ICLA works for image classification (10–200 classes) but fails for LM (50K vocab).

### B2: BLoB LoRA design
Category-split: pretrain World → LoRA fine-tune Sports → OOD Business/Sci-Tech. This ensures OOD is truly unseen by both base and adapter. Key HPs: rank=16, alpha=32, lr=3e-4, init_g=0.1, prior_std=0.2. MI ratio 1.13x (163K Bayesian params).

### B3: SVD-structured variance vs curvature-based
TFB uses SVD of B matrix: Ω_ij = σ_q / S_i (directions with large singular values get small variance). Diagonal Laplace uses loss curvature (flat at convergence). SVD captures geometric structure of LoRA subspace → works (1.10x). Curvature carries no directional information → fails (1.00x).

---

## C Milestone (16L Scaled Replication)

### Architecture
16L / 8H / 512d (~76.3M params). Same `model.py`, config-only changes. Weight tying preserved.
GPU: bs=16/accum=2 for full-weight (~50% VRAM), bs=32 for LoRA (~73% VRAM). Flipout not needed.

### Agentic Pipeline (Auto-Research)

`experiments/c_pipeline.py` — autonomous HP optimization pipeline. State machine: CONFIGURE → RUN → ANALYZE → DECIDE. On failure, invokes LLM agent (Claude/Codex via Provider pattern as subprocess) to diagnose problems and propose HP adjustments.

Architecture:
- `pipeline_runner.py` — Generic engine: `PipelineRunnerBase`, `MilestonePolicy`, `RuntimeHooks`. State persistence, budget tracking, OOM/divergence recovery, agent prompt construction, mechanical fallback.
- `c_milestones.py` — C-specific: config templates (C0–C4), gate functions (`check_gate`), tunable knobs (`TUNABLE_KNOBS`), comparison report, helper predicates.
- `c_pipeline.py` — CLI wiring: providers (`ClaudeProvider`, `CodexProvider`), hooks (`_prepare_model`, `_posthoc_fit`, `_sigma_std_extractor`), entry point.
- `agent_briefing.md` — HP tuning playbook injected into every agent prompt.

Key features:
- Full AGENTS.md + briefing injected into agent prompt (no file reading needed).
- Provider pattern: `ClaudeProvider` → `claude -p -` (stdin), `CodexProvider` → `codex exec`.
- Patience early-stop: `patience_evals=10`, `min_delta=0.001` (0.1% relative).
- NaN detection: checked every step + every eval interval.
- Diagnostic summary: gap analysis, step efficiency, convergence analysis, run-over-run deltas.
- Post-hoc support: `posthoc_fit_fn` hook dispatches Laplace/TFB for steps=0 milestones.
- BLoB→DeterministicLoRA conversion in `_prepare_model`: maps `lora_A_mu`→`lora_A` for C4 post-hoc milestones.
- Record-only milestones (C2, C4_TFB, C4_LAP): `max_runs=1`, no agent retry.

CLI: `--milestone {c0|c1|c2|c3|c4_tfb|c4_lap}`, `--provider {claude|codex}`, `--provider-model`, `--max-runs`, `--compare`, `--dry-run`, `--no-agent`, `--no-mlflow`, `--state-dir`, `--initial-agent`.

### C0 Template HPs
lr=6e-4, warmup=4000, steps=100K, bs=16, accum=2, dropout=0.1.

### C1 Template HPs
Same as C0 + `bayes_ffn: {enabled: True, init_rho: -2.0, prior_std: 1.0}`, kl_weight=0.1 (reduced from 0.2 — KL pressure scales with 64x more Bayesian params at 16L).

### C3 Template HPs (Phase 2 BLoB LoRA)
lr=3e-4, warmup=500, steps=10K, bs=32, accum=1, kl_weight=1.0 (matched B2's effective KL/data ratio), weight_decay=0.0. LoRA: rank=16, alpha=32, target=ffn, prior_std=0.2, init_g=0.1. Data: HackerNews (ID).

### Infrastructure Fixes During C (Summary)
- Token-level shuffle in Pile loader destroyed sequential structure → removed
- Residual projection scaling for deep models (`std / √(2·n_layer)`) → added
- Flash Attention (`F.scaled_dot_product_attention`) → replaced manual attention
- Patience early-stop + per-step NaN detection → added to train loop
- Agent subprocess: max-turns 5, structured_output envelope, stdin prompt delivery, UTF-8 encoding
- torch.quantile overflow for >16M elements → random subsample

---

## Future Work (Parked)
Non-Bayesian: RoPE, SwiGLU, KV-cache, RMSNorm, GQA.
Done: AMP (auto-enabled), Flash Attention.
Pipeline refactoring: `specs/pipeline-refactoring-spec.md` (split c_pipeline.py into providers.py + c_hooks.py).

## References
- **BLoB** (NeurIPS 2024) — Bayesian LoRA by backprop. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- **TFB** (NeurIPS 2025) — Training-Free Bayesianization. [arXiv:2412.05723](https://arxiv.org/abs/2412.05723)
- **Laplace-LoRA** (2023) — Laplace approximation on LoRA params. [arXiv:2308.13111](https://arxiv.org/abs/2308.13111)
- **Laplace Redux** (NeurIPS 2021) — Effortless Bayesian Deep Learning. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)
- **ICLA** (WACV 2025) — Identity Curvature Laplace for OOD. [arXiv:2312.10464](https://arxiv.org/abs/2312.10464)
