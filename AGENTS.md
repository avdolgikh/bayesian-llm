# Repository Guidelines

## Project Structure & Source of Truth
`docs/` holds PDF papers — the theory baseline. Implementations should align with those docs.
`specs/` holds planning documents. The active spec is `specs/refined-spec-feb2026.md`. Other specs: `held-out-test-split.md`, `vital-unit-tests.md`, `a0-baseline-checklist.md`, `ml-infrastructure.md`.

Layout (flat, minimal — no deep nesting):
```
minigpt/          # Python package — all model-related code
  model.py        # MiniGPT architecture (deterministic)
  layers.py       # Bayesian layers (BayesianLinear, etc.)
  data.py         # Dataset loading + tokenization (BPE via tiktoken) — TinyShakespeare, AG News
  train.py        # Training loop (cross-entropy + ELBO)
  evaluate.py     # Standard eval (perplexity, generation)
  config.py       # YAML config ↔ dataclass bridge
  uncertainty.py  # Epistemic uncertainty measurement (Phase 2)
configs/          # YAML config files for experiments
experiments/      # Runnable .py scripts (a0_baseline, a1_bayes_output, etc.)
tests/            # pytest tests
data/             # Local datasets (gitignored; document provenance in README.md)
```

## Hard Rules
- **No notebooks.** Only runnable `.py` scripts. Never create `.ipynb` files.
- **No extra Bayesian libraries.** Use only `torch.distributions` for probabilistic layers. Manual implementation is preferred over adding dependencies.
- **No modern transformer tricks** in miniGPT: no RoPE, SwiGLU, sliding window attention, MoE, GQA. Keep it basic — the focus is Bayesian, not architecture.
- **Document on the fly.** Always log findings, decisions, implementation steps, and user requests in this file and NOTES.md as work progresses. Do NOT wait for user to ask.

## Datasets

### TinyShakespeare
- ~1MB text, ~304K BPE tokens. Single-domain (literary English).
- `load_shakespeare()` downloads on first run to `data/tinyshakespeare.txt`.
- No OOD split — `test_ood` is `None`.

### AG News
- Source: `mhjabreel/CharCnn_Keras` GitHub repo (CSVs, no headers).
- 127.6K articles (120K train + 7.6K test), 4 categories: 1=World, 2=Sports, 3=Business, 4=Sci/Tech.
- `load_agnews()` downloads to `data/agnews/{train,test}.csv`, parses with stdlib `csv`, returns `list[tuple[int, str, str]]` (category, title, description).
- **Topic split** (`prepare_agnews_data()`):
  - ID categories (default: World=1, Sports=2) → shuffled (seeded), joined with `\n\n`, tokenized → train/val/test_id split (80/10/10).
  - OOD categories (default: Business=3, Sci/Tech=4) → same pipeline → `test_ood` tensor.
  - ~2.4M train, ~300K val, ~300K test_id, ~3.5M OOD tokens with default split.
- Category splits configurable via config: `data.id_categories`, `data.ood_categories`.

### Dataset Dispatcher
`load_dataset(cfg, tokenizer)` — top-level entry point. Reads `cfg["data"]["dataset"]` (`"tinyshakespeare"` or `"agnews"`), calls the appropriate loader, returns `{"train": tensor, "val": tensor, "test_id": tensor, "test_ood": tensor | None}`.

## Tokenization
- **BPE tokenization** via `tiktoken` (GPT-2 encoding, vocab_size=50257).
- Character-level tokenizer was used in early prototype but is now replaced.

## Experiment Tracking
- **MLflow** (local, sqlite backend: `sqlite:///mlflow.db`) for all experiment tracking.
- Every training run logs: hyperparameters, train/val loss, perplexity, generated samples, learning rate (every step), `test_id_perplexity`, `test_ood_perplexity` (AG News only), `best_val_loss`, `best_val_step`, `train_time_sec`, `tokens_per_sec`.
- Tags: `dataset` (tinyshakespeare/agnews), `milestone` (a0/a1/a2), `gpu` (device name).
- `train()` returns `tuple[MiniGPT, dict]` — metadata dict with `best_val_loss`, `best_val_step`, `train_time_sec`, `tokens_per_sec`.
- Launch MLflow UI with: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`
- `--no-mlflow` flag available on experiment scripts to disable tracking.
- **Model artifacts**: `mlflow.pytorch.log_model(model, "model")` + `mlflow.log_artifact("ckpt_best.pt")` logged per run. Run summary stored in `mlflow.note.content` tag.
- `mlflow.db` and `mlruns/` are **committed to git** (lightweight — metrics + text artifacts only, no model weights).

## Environment & Tooling
- **`uv`** for all dev tooling: linting, testing, dependency management. Use `uv run` for pytest, ruff, etc.
- **Global Python environment** (Python 3.11, CUDA-enabled PyTorch) **only for GPU training**. Training scripts use `python experiments/...` from global env because CUDA/PyTorch must be installed globally.
- Dev dependencies (pytest, ruff) are in `[dependency-groups] dev` in `pyproject.toml`.
- Document setup in `README.md`
- Target GPU runs (RTX 4070, ~10-12 GB VRAM)
- If uv hits permission errors, set `UV_CACHE_DIR` to a local folder (e.g., `.uv-cache`)

## Framework Decision
PyTorch + `torch.distributions` for all milestones. No JAX for now. No TensorFlow.

## Milestones (order matters)
- **A0:** Deterministic miniGPT baseline — TinyShakespeare + AG News, cross-entropy, verify loss/generation/VRAM, ID vs OOD perplexity baseline
- **A1:** Bayesian output head — BayesianLinear on final projection, ELBO training, first uncertainty metrics
- **A2:** Bayesian FFN layers — replace FFN linears, in-distribution vs OOD epistemic uncertainty evaluation
- **B1 (later, separate track):** Bayesian LoRA on an existing open-weight LLM

## Bayesian Layer Strategy
Which layers to make Bayesian, in order:
1. Output head (A1) — simplest, proves pipeline
2. FFN layers in transformer blocks (A2) — strongest epistemic signal (FFN stores factual knowledge)
3. Q/K/V projections — optional, later
4. Embeddings — optional, later

## Epistemic Uncertainty Measurement
- Temperature = 0 (greedy decoding). All stochasticity from Bayesian weights.
- N forward passes (10-30) with weight sampling per input.
- Primary metric: **mutual information** (MI = predictive_entropy - expected_entropy) — pure epistemic uncertainty.
- Secondary metrics: top-k logit variance, top-1 flip rate, sequence-level MI.
- Evaluation: train on subset of topics, measure MI on in-distribution vs OOD topics. Expect clear MI gap.
- **Note:** The spec notation `p_t^(i) = softmax(z_t^(i))` needs clarification — discussed, to be revisited in Phase 2.

## Configuration System
Experiments use a YAML config pipeline: `DEFAULT_CONFIG → YAML file → CLI overrides`.

- **`minigpt/config.py`** — `DEFAULT_CONFIG` dict with all defaults, plus helpers: `load_yaml`, `deep_merge`, `apply_overrides` (dot-notation), `build_gpt_config`, `build_train_config`, `validate_config`, `config_to_flat_params`.
- **`configs/a0_baseline.yaml`** — reference config for A0 baseline (TinyShakespeare, matches defaults).
- **`configs/a0_agnews.yaml`** — AG News config (fully explicit — all settings, 5000 steps, 500 warmup, ID=[1,2], OOD=[3,4]).
- `vocab_size` always comes from the tokenizer, never from config files.
- **PyYAML gotcha:** scientific notation without a decimal point (e.g. `3e-4`) is parsed as a **string**, not float. Always write `3.0e-4` or `0.0003` in YAML files.

### Experiment CLI
```bash
# TinyShakespeare (defaults)
python experiments/a0_baseline.py
python experiments/a0_baseline.py --config configs/a0_baseline.yaml

# AG News (ID=World+Sports, OOD=Business+Sci/Tech)
python experiments/a0_baseline.py --config configs/a0_agnews.yaml

# AG News with custom category split
python experiments/a0_baseline.py --config configs/a0_agnews.yaml \
  --set data.id_categories="[1,3]" --set data.ood_categories="[2,4]"

# Config + overrides
python experiments/a0_baseline.py --config configs/a0_baseline.yaml --set train.lr=1e-3 --set model.n_layer=6

# Resume interrupted run
python experiments/a0_baseline.py --config configs/a0_baseline.yaml --resume data/checkpoints/ckpt_step500.pt
```

### Checkpoint Resume
Checkpoints save full config dict, `best_val_loss`, torch/CUDA RNG states. On resume, training continues from `ckpt["step"] + 1` with restored RNG and optimizer state. The LR schedule is stateless (computed from step number) so it resumes automatically. Old checkpoints (pre-config) are backward-compatible via `.get()` defaults.

## Build, Test, and Development Commands
- `uv sync` (install/update all dependencies including dev)
- `uv run pytest tests/ -v` (run all unit tests)
- `uv run ruff check minigpt/ experiments/ tests/` (lint)
- `python experiments/a0_baseline.py` (training — global env, GPU)
- `mlflow ui --backend-store-uri sqlite:///mlflow.db` (view experiment results)

## CI/CD
- **GitHub Actions** (`.github/workflows/ci.yml`): runs on push/PR. Uses `astral-sh/setup-uv@v5`.
- Pipeline: `uv run ruff check` → `uv run pytest tests/ -v`. No GPU, no training in CI.
- **Ruff** for linting (replaces flake8 + isort). Config in `pyproject.toml` under `[tool.ruff.lint]`.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; line length up to 100 (enforced by ruff)
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Prefer type hints and short docstrings for public APIs and experiment entry points

## Testing & Evaluation Guidelines
Use `pytest` with `test_*.py` naming. Keep tests deterministic (seed randomness). For experiments, report calibration error, NLL, Brier score, and OOD detection; note whether results target epistemic vs aleatoric uncertainty.

### Vital Unit Tests (ML Guardrails)
Sanity checks on mathematical and methodological invariants. Spec: `specs/vital-unit-tests.md`.

```
tests/
  test_model.py          # P0: weight tying pointer equality (3 tests)
                         # P1: perplexity bounds at init (2 tests)
  test_data.py           # P0: category isolation — AG News (4 tests)
                         # P1: split sizes sum to total (3 tests)
  test_reproducibility.py # P2: same seed = identical losses (2 tests)
```

**Deferred to Phase 2** (needs `uncertainty.py`):
- P0: MI = 0 for deterministic model
- P1: Bayesian sampling produces variance
- P2: KL/MI non-negativity

## Commit & Pull Request Guidelines
- Commit messages: **one line only**, concise. Use Conventional Commits prefix (e.g., `feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- **Never mention AI code assistants** in commits, code comments, PRs, or any repo content. No `Co-Authored-By` AI lines.
- PRs should include a short problem/solution summary, links to relevant papers or issues, test evidence, and experiment metadata (model size, method, dataset, metrics).

## Security & Data
Do not commit secrets or large binaries. Use `.env` (ignored) and provide `.env.example` when needed. Keep `data/` out of version control and document licensing constraints in `README.md`.

## Implementation Log
- **2026-02-21:** Phase 1 restructuring complete. Flat `minigpt/` package replaces `src/bayesian_llm/`. BPE tokenization (tiktoken/GPT-2). MLflow tracking (sqlite, git-tracked). TinyShakespeare dataset. Checkpoints save to `data/checkpoints/`. Smoke test passed: loss decreases 10.66→7.96 in 50 steps, 13.7M params. Added `*.egg-info/` to .gitignore.
- **2026-02-23:** Training loop hardened for A0 polish:
  - **Weight decay 0.1** with proper param groups (decay on 2-D weights only, not biases/layernorms). AdamW betas=(0.9, 0.95).
  - **Cosine LR schedule** with linear warmup (default 200 steps, min_lr=1e-5). LR logged to MLflow every step.
  - **Gradient clipping** (max_norm=1.0).
  - **Weight tying** (GPT-2 style): `lm_head.linear.weight = token_emb.weight`. Cuts ~12.9M params with vocab 50257 and n_embd=256.
  - **Best checkpoint tracking**: saves `ckpt_best.pt` on val loss improvement, reloads it after training for evaluation/generation.
  - New defaults: n_embd=256, block_size=256, dropout=0.2, n_layer=4 (was already default).
  - New CLI flags: `--dropout`, `--weight-decay`, `--warmup-steps`, `--min-lr`, `--grad-clip`.
  - Observation: 304K BPE tokens is very small vs 50257 vocab — embedding table dominates params. Weight tying is critical.
- **2026-02-23:** YAML config system + checkpoint resume:
  - **Bug fix:** Weight tying was broken — `self.lm_head.weight = ...` set an attribute on the `DeterministicLinear` wrapper but the forward pass uses `self.lm_head.linear.weight`. Fixed to `self.lm_head.linear.weight = self.token_emb.weight`. This doubled param count (29M → ~16M with n_embd=256).
  - **YAML config pipeline**: `DEFAULT_CONFIG → YAML → CLI --set overrides → validate → run`. New `minigpt/config.py` module. Reference config: `configs/a0_baseline.yaml`.
  - **Checkpoint resume**: `--resume path` restores model, optimizer, RNG states, best_val_loss. LR schedule is stateless.
  - **Simplified CLI**: replaced 20 argparse flags with `--config`, `--resume`, `--set key=value`, `--no-mlflow`.
  - Added `pyyaml` dependency.
- **2026-02-23:** AG News dataset (A0.1) — non-Bayesian baseline on topic-split data:
  - **AG News loader**: `load_agnews()` downloads train+test CSVs (127.6K articles, 4 categories). Parsed with stdlib `csv`.
  - **Topic split**: ID categories (default: World, Sports) → train/val; OOD categories (default: Business, Sci/Tech) → test_ood. Articles joined with `\n\n`, tokenized with BPE.
  - **Dataset dispatcher**: `load_dataset(cfg, tokenizer)` returns `{"train", "val", "test_ood"}` dict. TinyShakespeare returns `test_ood=None`.
  - **OOD evaluation**: after training, if `test_ood` exists, computes and logs `ood_perplexity` to MLflow. Prints ID vs OOD comparison.
  - **Config**: `configs/a0_agnews.yaml` (5000 steps, 500 warmup). Category splits configurable via `--set data.id_categories="[1,3]"`.
  - **`_coerce_type` extended**: now parses JSON lists/objects from CLI `--set` overrides.
  - **Bug fix**: `configs/a0_baseline.yaml` had `lr: 3e-4` and `min_lr: 1e-5` — PyYAML (YAML 1.1) parses these as strings. Fixed to `3.0e-4` / `1.0e-5`.
- **2026-02-24:** Held-out test set (ID + OOD) — methodological fix for Phase 2:
  - **Problem:** val was used both for checkpoint selection AND as "ID test set" for final evaluation. Biased — best checkpoint was selected to minimize val loss.
  - **Fix:** Three-way ID split: `[train 80% | val 10% | test_id 10%]`. Val is for checkpoint selection only. `test_id` is held-out for unbiased ID vs OOD comparison.
  - Data pipeline now returns `{"train", "val", "test_id", "test_ood"}`. Both `prepare_data()` (TinyShakespeare) and `prepare_agnews_data()` (AG News) updated.
  - New config key: `data.test_fraction` (default 0.1). Validation: `val_fraction + test_fraction < 1.0`.
  - Experiment script evaluates and logs `test_id_perplexity` separately from `final_val_perplexity`. ID vs OOD comparison uses `test_id` vs `test_ood`.
  - `configs/a0_agnews.yaml` made fully explicit (all settings spelled out, no reliance on defaults).
  - Spec: `specs/held-out-test-split.md`. Vital unit tests proposed: `specs/vital-unit-tests.md`.
- **2026-02-24:** MLflow tracking improvements:
  - `train()` now returns `tuple[MiniGPT, dict]` — metadata dict with `best_val_loss`, `best_val_step`, `train_time_sec`, `tokens_per_sec`.
  - Experiment script logs 4 new metrics (`best_val_loss`, `best_val_step`, `train_time_sec`, `tokens_per_sec`) and 3 tags (`dataset`, `milestone`, `gpu`).
  - Spec: `specs/a0-baseline-checklist.md` — what to verify before moving to A1. Spec: `specs/ml-infrastructure.md` — tracking, model registry, HPO pipeline, CI/CD plans.
- **2026-02-24:** Vital unit tests implemented (14 tests, all passing):
  - `tests/test_model.py` — P0 weight tying pointer equality (init/forward/backward), P1 perplexity bounds at init.
  - `tests/test_data.py` — P0 category isolation (4 tests), P1 split sizes sum to total (3 tests).
  - `tests/test_reproducibility.py` — P2 same seed = identical losses.
  - Phase 2 tests deferred: MI=0 for deterministic, Bayesian sampling variance, KL/MI non-negativity.
- **2026-02-24:** ML infrastructure — model artifacts, CI/CD, linting:
  - **Model artifacts**: experiment script logs `mlflow.pytorch.log_model()` + `mlflow.log_artifact(ckpt_best.pt)` + `mlflow.note.content` run summary.
  - **GitHub Actions CI**: `.github/workflows/ci.yml` — `astral-sh/setup-uv@v5`, runs ruff + pytest. No GPU in CI.
  - **Ruff linting**: config in `pyproject.toml`, dev deps in `[dependency-groups] dev`.
  - **Environment rule**: `uv` for all dev tooling (lint, test, deps). Global Python env only for GPU training.

## Future Work (Non-Bayesian — Parked)
These are architectural improvements to revisit **after** Bayesian milestones (A1/A2) are done. Not in scope now — the current miniGPT is intentionally basic to keep focus on Bayesian aspects.

- **RoPE** — Rotary Position Embeddings (replace learned positional embeddings)
- **SwiGLU** — gated FFN activation (replace GELU)
- **Pre-Norm vs Post-Norm** — current model uses pre-norm (GPT-2 style); evaluate post-norm or sandwich-norm
- **KV-cache** — inference-time optimization for autoregressive generation
- **PEFT** — Parameter-Efficient Fine-Tuning (LoRA, adapters) to squeeze more params into VRAM budget
- **Mixed precision (FP16 / BF16)** — faster training and reduced VRAM via `torch.cuda.amp`

## Papers to Consider (Parked)
Bayesian + LLM papers for later review, likely relevant to B1 (Bayesian LoRA on open-weight LLM).

- **BLoB: Bayesian Low-Rank Adaptation by Backpropagation for LLMs** (NeurIPS 2024) — jointly adjusts mean and covariance of LLM parameters during fine-tuning for better generalization and uncertainty estimation. [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d53575463291ea6b5a23cf6e571f59b-Paper-Conference.pdf)
- **Training-Free Bayesianization for Low-Rank Adapters of LLMs** (2024) — converts trained LoRA adapters into Bayesian ones without additional training by searching for optimal weight variance. [arXiv:2412.05723](https://arxiv.org/pdf/2412.05723)
- **Bayesian Low-rank Adaptation for Large Language Models (Laplace-LoRA)** (2023) — applies Laplace approximation to LoRA parameters, improving calibration and reducing overconfidence. [arXiv:2308.13111](https://arxiv.org/pdf/2308.13111)
- **ScalaBL: Scalable Bayesian Low-Rank Adaptation via Stochastic Variational Subspace Inference** (2025) — Bayesian inference in a low-dimensional subspace (~1000 extra params), scales to larger models than prior approaches. [arXiv:2506.21408](https://arxiv.org/pdf/2506.21408)
