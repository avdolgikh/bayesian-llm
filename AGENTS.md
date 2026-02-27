# Repository Guidelines

## Project Structure & Source of Truth
`docs/` holds PDF papers — the theory baseline. Implementations should align with those docs.
`specs/` holds planning documents. The active spec is `specs/refined-spec-feb2026.md`. Other specs: `held-out-test-split.md`, `vital-unit-tests.md`, `a0-baseline-checklist.md`, `ml-infrastructure.md`, `a1-bayesian-output-head.md`, `a1-kl-tuning.md`, `a1-tuning-round2.md`, `gpu-acceleration.md`, `a2-bayesian-ffn.md`.

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
experiments/      # Runnable .py scripts (a0_baseline, a1_bayes_output, a2_bayes_ffn, etc.)
scripts/          # Utility scripts (dump_mlflow_run, profile_gpu, etc.)
tests/            # pytest tests
data/             # Local datasets (gitignored; document provenance in README.md)
```

## Hard Rules
- **No notebooks.** Only runnable `.py` scripts. Never create `.ipynb` files.
- **No extra Bayesian libraries.** Use only `torch.distributions` for probabilistic layers. Manual implementation is preferred over adding dependencies.
- **No modern transformer tricks** in miniGPT: no RoPE, SwiGLU, sliding window attention, MoE, GQA. Keep it basic — the focus is Bayesian, not architecture.
- **Document on the fly.** Always log findings, decisions, implementation steps, and user requests in this file (AGENTS.md) as work progresses — DURING implementation, not at the end. Each meaningful change should be documented immediately after it's made. Do NOT batch documentation to the end of a session.
- **Explicit configs.** YAML config files must list every parameter explicitly — never rely on silent code defaults. When a new config key is added, it must appear in all relevant YAML files. The config is the single source of truth for experiment reproducibility.

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
- A1 runs additionally log: `mi_id_mean`, `mi_ood_mean`, `mi_ood_id_ratio`, `flip_rate_id`, `flip_rate_ood`, `pred_entropy_id_mean`, `pred_entropy_ood_mean`, `expected_entropy_id_mean`, `expected_entropy_ood_mean`, `final_kl_loss`, `n_bayes_params`, sigma stats (`sigma_mean`, `sigma_std`, `sigma_min`, `sigma_max`, `sigma_median`, `sigma_p5/p25/p75/p95`).
- `train()` returns `tuple[MiniGPT, dict]` — metadata dict with `best_val_loss`, `best_val_step`, `train_time_sec`, `tokens_per_sec`.
- Launch MLflow UI with: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`
- `--no-mlflow` flag available on experiment scripts to disable tracking.
- **Model artifacts**: opt-in via `--log-model` flag (off by default — heavy). Logs `mlflow.pytorch.log_model()` + `mlflow.log_artifact(ckpt_best.pt)`. Run summary note (`mlflow.note.content`) always logged.
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
- **A0: DONE** — Deterministic miniGPT baseline on AG News. Reference: test_id_ppl=49.11, test_ood_ppl=540.28 (run `5dc45450e7b6458fbad2ec07dfd91ce3`). See `specs/a0-baseline-checklist.md`.
- **A1: DONE** — Bayesian output head — BayesianLinear on lm_head, ELBO training, uncertainty metrics (MI). Model: 41.8M params (25.7M Bayesian). Three training runs explored kl_weight (0.01–0.2), init_rho (-5 to -2), steps (5K–50K). **Best MI ratio: 1.36x** (R1 step 40K, σ=0.22). Conclusion: output head detects OOD at vocabulary level (1.2–1.4x MI ratio ceiling); FFN layers needed for semantic OOD detection. All infrastructure (ELBO, MC sampling, MI eval, sigma logging) validated and reusable. Specs: `a1-bayesian-output-head.md`, `a1-kl-tuning.md`, `a1-tuning-round2.md`, `gpu-acceleration.md`.
  - **First run (2026-02-25)**: OOM crash during uncertainty eval — fixed by streaming MC sampling (`_stream_metrics`). KL dominated training: 57M KL / 2.7M tokens = 21 nats swamping CE of ~6 (root cause: `weight_rho=-5` → σ≈0.007 vs `prior_std=1.0` → 4.5 nats/param × 12.8M params). Fix: cold posterior (`kl_weight=0.01`) + optional linear KL annealing (`kl_annealing_steps=200`). Sanity run with fixes confirmed: ELBO = CE(5.96) + weighted_KL(0.21) — 100x reduction. Results identical at 400 steps (too few to differentiate); bumped to 5000 steps for real run.
  - **MLflow fix**: stepless `log_metric` calls created single-point bar charts in UI. Moved all summary values (best_val_loss, perplexities, MI metrics, etc.) to `log_param` in both A0 and A1 scripts. Only step-aware time-series (train_loss, val_loss, lr, effective_kl_weight, etc.) stay as metrics.
  - **Config audit**: moved 8 categories of hardcoded values to YAML config: `eval.temperature`, `eval.sample_tokens` (wiring gap), `model.bayes.init_rho`, `eval.n_perplexity_batches`, `experiment.mlflow_uri`, `train.adam_beta1/adam_beta2`, `eval.qualitative_*` (prompts_per_category, max_new_tokens, seed), `train.checkpoint_dir`. All experiment scripts now read from config, no magic numbers.
  - **train.py refactor**: replaced vague `num_train_tokens > 0` guard with explicit `is_bayesian = kl_weight > 0` flag. Collapsed `kl_weight/num_train_tokens` into `kl_scale` in `estimate_loss`. Added validation: `num_train_tokens` must be > 0 when `kl_weight > 0`. Deterministic path (A0) now has zero KL overhead — no `model.kl_loss()` call.
- **A2: CODE READY** — Bayesian FFN layers (MLP.fc + MLP.proj, 4 blocks). ~20M params (4.2M Bayesian, weight-tied head). Dual-path MC sampling auto-detects body vs head Bayesian. 15 new tests (53 total). Spec: `a2-bayesian-ffn.md`. Config: `init_rho=-3`, `kl_weight=0.2`, `kl_annealing=10K`, `steps=50K`, `num_samples=20`. Pending training run.
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
- **`configs/a1_agnews.yaml`** — A1 Bayesian output head on AG News. `bayes_head.enabled: true`, `prior_std: 1.0`, `kl_weight: 0.2`, `init_rho: -2.0`, `steps: 50000`, `kl_annealing_steps: 10000`, `eval.num_samples: 30`, `eval_interval: 2000`, `checkpoint_interval: 5000`.
- **`configs/a2_agnews.yaml`** — A2 Bayesian FFN on AG News. `bayes_head.enabled: false` (weight-tied), `bayes_ffn.enabled: true`, `prior_std: 1.0`, `kl_weight: 0.2`, `init_rho: -3.0`, `steps: 50000`, `kl_annealing_steps: 10000`, `eval.num_samples: 20`.
- `vocab_size` always comes from the tokenizer, never from config files.
- Config key `model.bayes` was renamed to `model.bayes_head` in A2 for clarity. `model.bayes_ffn` added for FFN-layer Bayesian config.
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
- `python experiments/a1_bayes_output.py --config configs/a1_agnews.yaml` (A1 Bayesian training)
- `python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml` (A2 Bayesian FFN training)
- `python scripts/profile_gpu.py --config configs/a1_agnews.yaml` (GPU throughput profiling)
- `python scripts/dump_mlflow_run.py <run_id>` (inspect MLflow run)
- `python scripts/eval_checkpoint.py <ckpt_path> --config <yaml>` (uncertainty + sigma stats on any checkpoint)
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
  test_bayesian.py       # A1: KL non-negativity (4), sampling variance (4),
                         #     frozen sampling (4), selective Bayesian (6),
                         #     MI invariants (6)
                         # A2: FFN Bayesian (8), combined (3),
                         #     body detection (4) — 39 tests total
```

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
- **2026-02-24:** A0 baseline DONE — AG News 10K steps, reference metrics locked:
  - MLflow run: `5dc45450e7b6458fbad2ec07dfd91ce3`. Config: `configs/a0_agnews.yaml`.
  - **Reference:** test_id_ppl=49.11, test_ood_ppl=540.28, OOD/ID=11.0x, best_val_loss=3.8874 (step 9400).
  - Model: 4L/4H/256d, 16M params, 2.7M train tokens, 2.8hrs on RTX 4070.
  - Convergence: plateau by ~8K steps, train-val gap 0.75 (moderate overfitting, expected).
  - Full analysis: `specs/a0-baseline-checklist.md`.
  - Utility script: `scripts/dump_mlflow_run.py <run_id>` — dumps params/metrics/tags.
  - **Ready for A1** (Bayesian output head).
- **2026-02-25:** A1 Bayesian output head — code infrastructure complete (pending training run):
  - **`layers.py`**: `BayesianLinear` extended with `freeze_sample()`/`unfreeze_sample()` (cached weights for coherent autoregressive generation), `mean_forward()` (deterministic μ-only), `mean_forward_with_variance()` (μ + closed-form logit variance). Context managers: `frozen_bayesian_sample(model)`, `use_mean_weights(model)`.
  - **`model.py`**: Selective Bayesian via Option A — `Block`/`CausalSelfAttention`/`MLP` take explicit `bayes: BayesConfig` param. `MiniGPT` passes `BayesConfig(enabled=False)` to all blocks, `config.bayes` only to `lm_head`. Weight tying skipped when `bayes.enabled=True`. Added `forward_body()` for A1 efficiency (run transformer once, `lm_head` N times). `generate()` gains `use_mean` param.
  - **`train.py`**: ELBO training with auto-detection — `loss = ce + kl_weight * kl / num_train_tokens`. When no BayesianLinear layers, KL=0 → pure CE path. Logs `kl_loss`, `ce_loss`, `elbo_loss` to MLflow when KL > 0. Same `train()` function for A0 and A1.
  - **`uncertainty.py`**: Full implementation — `compute_uncertainty_metrics()` (aggregate MI/entropy/flip_rate over batches, uses `forward_body` + N `lm_head` calls for A1 efficiency), `score_sequence()` (per-token MI for a single sequence), `_compute_token_metrics()` (core MI/entropy/flip_rate computation from stacked probs).
  - **`experiments/a1_bayes_output.py`**: Train with ELBO, eval perplexity (mean weights, ID+OOD), uncertainty eval (MI/entropy/flip_rate on test_id+test_ood), qualitative prompt panel (curated AG News article openings, generate+score per-token MI), log everything to MLflow.
  - **`configs/a1_agnews.yaml`**: Same as a0_agnews but `bayes.enabled: true`, `prior_std: 1.0`, `kl_weight: 1.0`, `eval.num_samples: 30`.
  - **`tests/test_bayesian.py`**: 24 new tests — KL non-negativity (4), sampling variance (4), frozen sampling (4), selective Bayesian architecture (6), MI invariants (6). All 38 tests passing (14 old + 24 new).
  - **Spec**: `specs/a1-bayesian-output-head.md` — full design doc (16 sections, LaTeX math, code mappings).
  - **Next**: GPU training run on AG News, compare test_id_ppl to A0 reference (49.11), measure MI gap.
- **2026-02-25:** A1 first full run — posterior collapse diagnosed (MLflow `686e3201`):
  - **Config**: `kl_weight=0.01`, `kl_annealing_steps=2000`, `batch_size=64`, 5K steps, fp32.
  - **Results**: test_id_ppl=65.62, test_ood_ppl=630.46 (9.6x ratio). MI_id=0.0236, MI_ood=0.0253, **MI ratio=1.07x** — no separation. Flip rate: ID=12.7%, OOD=13.0% — also flat.
  - **Diagnosis**: `kl_weight=0.01` is too cold. Effective KL per token = `0.01 * 52.4M / 2.7M` = 0.19 nats — only 4.5% of ELBO. Posterior σ collapsed to near-zero → MC samples are near-identical → MI ≈ 0.
  - **Fix**: increase `kl_weight` from 0.01 → 0.1 (32% of ELBO). Keep annealing at 2000 steps. Spec: `specs/a1-kl-tuning.md`.
- **2026-02-25:** GPU profiling — silent VRAM overflow discovered:
  - **Problem**: A1 model is 41.8M params (no weight tying + μ/ρ doubling in BayesianLinear). At `batch_size=64` fp32, peak VRAM = 14.9 GB — exceeds RTX 4070's 12.3 GB. CUDA on Windows silently spills to system RAM → **10x throughput collapse** (8K tok/s vs 82K tok/s at B=32).
  - **Profiling script**: `scripts/profile_gpu.py`. Best config: AMP + B=32 → 111K tok/s, 48% VRAM.
  - **Plan**: add AMP (mixed precision) to train.py + reduce batch_size to 32. Spec: `specs/gpu-acceleration.md`.
- **2026-02-25:** GPU acceleration implemented + KL weight tuning applied:
  - **AMP (mixed precision)**: `torch.amp.autocast` + `GradScaler` added to `train.py` (training loop + `estimate_loss`), `uncertainty.py` (`_stream_metrics`, `compute_uncertainty_metrics`, `score_sequence`). Auto-enabled on CUDA (`use_amp = device.type == "cuda"`). Entropy accumulation stays in fp32 to avoid numerical issues.
  - **Gradient accumulation**: `gradient_accumulation_steps` added to `TrainConfig` (default 1) and `DEFAULT_CONFIG`. Training loop runs `accum_steps` micro-batches per optimizer step with CE scaled by `1/accum_steps`. KL added only on the last micro-step (KL is a weight property, not data-dependent — adding per micro-batch would double-count). Infrastructure for A2 when model grows larger.
  - **Batch size**: `a1_agnews.yaml` already had `batch_size: 32` (updated in previous session). No change needed.
  - **KL weight tuning**: `kl_weight` bumped from `0.01` → `0.1` in `configs/a1_agnews.yaml`. Expected to fix posterior collapse (MI ratio 1.07x → target >1.5x). KL annealing kept at 4000 steps (was 2000, config already had 4000).
  - **Config audit**: `gradient_accumulation_steps: 1` added to all YAML configs (`a0_baseline.yaml`, `a0_agnews.yaml`, `a1_agnews.yaml`) per explicit-config rule.
  - All 38 tests passing, ruff clean. No changes to model architecture or test code.
- **2026-02-26:** A1 second run analysis (MLflow `4e1662e6`, 40K steps):
  - **Actual config** (via CLI overrides): `kl_weight=0.2`, `kl_annealing_steps=5000`, `steps=40000`, `batch_size=32`, `eval_interval=1000`, `checkpoint_interval=2000`, `warmup_steps=1000`. AMP active, 105K tok/s on RTX 4070 (~52 min).
  - **Best val loss at step 8000** (CE=4.18). Val loss plateau after 8K, train continues to ~2.82. Classic overfitting after step 8K.
  - **KL trajectory**: Monotonically decreasing 58M→15.7M. NOT collapse — sigmas are GROWING (from 0.007→0.224), approaching prior (σ=1.0). KL gradient `-1/σ + σ` pushes small sigmas up.
  - **Sigma statistics** (step 40K): mean=0.224, median=0.239, range=[0.015, 0.441]. Bimodal: 5th pct=0.064, 95th pct=0.370. Some vocab tokens retain low sigma (confident), others high.
  - **Checkpoint comparison** (via `scripts/eval_checkpoint.py`):

    | Step | KL | σ mean | MI (ID) | MI (OOD) | Ratio | ID ppl | OOD ppl |
    |------|-----|--------|---------|----------|-------|--------|---------|
    | 8000 (best CE) | 34.1M | 0.049 | 0.146 | 0.188 | 1.29x | 60.1 | 733 |
    | 20000 | 19.8M | 0.161 | 0.274 | 0.362 | 1.32x | — | — |
    | 40000 (final) | 15.7M | 0.224 | 0.293 | 0.398 | 1.36x | 56.3 | 1477 |

  - **Step 40K is the better Bayesian model**: MI 2x higher, ratio 1.36x (vs 1.29x), ID ppl slightly better (56 vs 60), OOD ppl higher (more separation).
  - **MI discrepancy (1.29x global vs 1.02x qualitative)**: Global MI scores real OOD text (Business/Sci-Tech). Qualitative MI scores model-generated continuations (with `use_mean=True`), which are always in-distribution regardless of prompt topic. The model ignores OOD prompts and generates what it knows → MI is flat. **Global ratio is the real OOD signal; qualitative ratio is a methodological artifact.**
  - **Checkpoint selection issue**: Best-val-loss criterion selects step 8000 (best CE), but Bayesian quality improves through step 40K (sigma grows, MI improves, ELBO improves). **Recommendation**: select by best val ELBO when `is_bayesian=True`. The ELBO was still improving at 40K (5.44 vs 6.70 at step 8K).
  - **init_rho=-5 problem**: Starting sigma=0.007 means 40K steps just to reach σ≈0.22. Softplus gradient through `sigmoid(rho)` at rho=-5 is 0.007 — 143x attenuated. This is the **softplus saturation bottleneck**. Starting with `init_rho=-2` (σ≈0.13) or `-1` (σ≈0.31) would skip the slow climb and let training focus on the right sigma region immediately.
  - **Scripts**: `scripts/eval_checkpoint.py` — loads any checkpoint, prints sigma stats + perplexity + MI. Usage: `python scripts/eval_checkpoint.py data/checkpoints/ckpt_stepN.pt --config configs/a1_agnews.yaml --set train.batch_size=32`.
  - **Open questions for next run**: (1) Try `init_rho=-1` or `-2`; (2) Switch checkpoint selection to best ELBO; (3) Investigate sigma distribution per vocab token — do rare tokens have higher σ? (4) Grad clipping impact: `grad_clip=1.0` clips total norm, per-param gradient is tiny with 41.8M params, but softplus saturation is the bigger bottleneck.
- **2026-02-26:** A1 tuning round 2 — four fixes implemented (spec: `specs/a1-tuning-round2.md`):
  - **Change 1 — qualitative MI fix**: `_run_qualitative_eval` now scores MI on **prompt tokens** (the real ID/OOD text), not on model-generated continuations. Generation still displayed for human inspection but no longer used for MI scoring. Variable naming: `cont_mi` → `prompt_mi`, `sequence_mi` → `prompt_mi` in results dict.
  - **Change 2 — best-ELBO checkpoint**: `train.py` now selects best checkpoint by `elbo_loss` when `is_bayesian=True`, by `loss` (CE) otherwise. Prints criterion at training start. `best_val_loss` in checkpoint dict holds ELBO for Bayesian runs (consistent within same run + resume).
  - **Change 3 — init_rho default**: Code default changed from `-5.0` to `-1.0` in `BayesConfig`, `BayesianLinear.__init__`, `DEFAULT_CONFIG`. Toy run with `-1.0` showed mean-weight ppl=4004 at 1K steps — σ=0.313 is too noisy for mu to learn CE task. **Config shipped with init_rho=-2.0** (σ=0.127, 8.4x attenuation) as the compromise.
  - **Change 4 — sigma logging**: New `sigma_summary(model)` function in `layers.py` — returns 9 aggregate stats (mean/std/min/max/median/p5/p25/p75/p95). Called from `a1_bayes_output.py` after training, logged to MLflow as params. `scripts/eval_checkpoint.py::_sigma_stats()` refactored to use `sigma_summary()` internally (per-layer detail still printed).
  - **Qualitative eval output**: console now prints summary line only (`Qualitative MI — ID: ... OOD: ... Ratio: ...`). Full per-prompt report still saved to MLflow artifact `qualitative_eval.txt`.
  - **Config for round 2** (`configs/a1_agnews.yaml`): `init_rho=-2.0`, `steps=50000`, `eval_interval=2000`, `checkpoint_interval=5000`, `kl_annealing_steps=10000`. Rest unchanged from round 1.
  - All 38 tests passing, ruff clean. **Ready for round 2 training run**.
- **2026-02-26:** A1 tuning round 2 — training results (MLflow `5dff1029689942c4a08b0136c4647298`, 50K steps):
  - **Config**: `init_rho=-2.0` (σ₀=0.127), `kl_weight=0.2`, `kl_annealing_steps=10000`, `steps=50000`, `batch_size=32`, `eval_interval=2000`, `checkpoint_interval=5000`, `warmup_steps=1000`. AMP active, 106K tok/s on RTX 4070 (~64 min).
  - **Best ELBO at step 48000** (val CE=4.4875, val ELBO=4.7732). ELBO checkpoint selection working correctly — CE kept improving slightly past step 48K but ELBO (which includes KL) peaked there.
  - **KL trajectory**: 20.2M → 3.85M (monotonically decreasing, as expected — posterior shrinking from init toward optimal). Much lower than round 1's final 15.7M because init_rho=-2 starts with larger σ (closer to prior), so KL starts lower.
  - **Sigma statistics** (step 48K): mean=0.637, std=0.213, median=0.706, range=[0.111, 0.919], p5=0.298, p95=0.870. Posterior is ~64% as wide as the prior (σ=1.0) — much wider than round 1's 0.224.
  - **Uncertainty metrics** (N=30 MC samples):
    - MI: ID=0.317, OOD=0.387, **ratio=1.22x**
    - Predictive entropy: ID=3.70, OOD=4.59
    - Expected entropy: ID=3.38, OOD=4.20
    - Flip rate: ID=28.5%, OOD=31.7%
  - **Perplexity** (mean weights): test_id=65.83, test_ood=1069.33 (16.2x ratio)
  - **Qualitative MI** (prompt-token scoring, fixed methodology):
    - Sci/Tech: avg MI ≈ 0.42 (highest — most specialized vocabulary)
    - Business: avg MI ≈ 0.37
    - World: avg MI ≈ 0.30
    - Sports: avg MI ≈ 0.31
    - Overall: ID=0.307, OOD=0.394, ratio=1.28x
  - **Qualitative MI shows a real category gradient** — Sci/Tech (jargon-heavy) is most detectable, Business intermediate, World/Sports (seen in training) lowest. This confirms the output head detects unfamiliar *vocabulary*, not unfamiliar *concepts*.

- **2026-02-26:** A1 conclusion — cross-round analysis and architectural ceiling:

  **Three-run comparison table:**

  | Run | init_rho | σ final | MI (ID) | MI (OOD) | MI ratio | ID ppl | OOD ppl | Notes |
  |-----|----------|---------|---------|----------|----------|--------|---------|-------|
  | R0 (first run) | -5 | — | 0.024 | 0.025 | 1.07x | 65.6 | 630 | kl_weight=0.01, posterior collapse |
  | R1 (step 40K) | -5 | 0.224 | 0.293 | 0.398 | **1.36x** | 56.3 | 1477 | kl_weight=0.2, best MI ratio |
  | R1 (step 8K) | -5 | 0.049 | 0.146 | 0.188 | 1.29x | 60.1 | 733 | CE-selected checkpoint |
  | **R2 (step 48K)** | **-2** | **0.637** | **0.317** | **0.387** | **1.22x** | **65.8** | **1069** | **ELBO-selected checkpoint** |

  **Finding 1 — Optimal sigma window.** There is a sweet spot around σ ≈ 0.1–0.3 for the output head. Below it (R0, σ → 0): posterior collapse, MI ≈ 0. Above it (R2, σ = 0.64): noise is uniform across all tokens, MI is high everywhere (both ID and OOD), so the ratio drops. R1 at step 40K (σ = 0.22) hit the sweet spot accidentally — enough variance for MC samples to disagree on OOD tokens, but not so much that ID predictions become noisy too.

  **Finding 2 — Higher sigma hurts ID perplexity.** R2 (σ=0.64): test_id_ppl=65.8, R1 (σ=0.22): test_id_ppl=56.3, A0 (deterministic): test_id_ppl=49.1. Each increase in posterior width costs prediction quality. The posterior is too diffuse to make confident predictions — the KL term pushes σ toward the prior (1.0), overpowering the CE signal that wants narrow posteriors.

  **Finding 3 — The output head has a structural ceiling for OOD detection.** Across three runs spanning very different hyperparameters (kl_weight 0.01–0.2, init_rho -5 to -2, steps 5K–50K), the MI ratio stays in the **1.2–1.4x band** (excluding the collapsed R0). The output head is a linear projection from hidden states to vocabulary — it can only express uncertainty about the token-level mapping, not about content semantics. The factual knowledge about *what topics the model has seen* lives in the FFN layers (and to some extent attention), not in the vocabulary projection.

  **Finding 4 — Qualitative MI confirms the vocabulary-level limitation.** The category gradient (Sci/Tech > Business > World ≈ Sports) tracks vocabulary specialization, not conceptual novelty. Sci/Tech articles contain domain-specific terms (technical jargon) that the output head has rarely mapped from hidden states → high MI. Business uses more general vocabulary → lower MI. This is consistent with the output head detecting OOD at the lexical level only.

  **Finding 5 — Infrastructure validated.** All A1 machinery is working correctly and reusable for A2:
  - ELBO training with KL annealing ✓
  - Best-ELBO checkpoint selection ✓
  - MC sampling uncertainty eval (streaming, AMP-compatible) ✓
  - MI / flip rate / entropy metrics ✓
  - Qualitative prompt-panel scoring (prompt tokens, not generations) ✓
  - Sigma summary logging ✓
  - Gradient accumulation infrastructure ✓

  **Decision: A1 is DONE. Move to A2 (Bayesian FFN layers).**

  Best A1 result: **MI ratio 1.36x** (R1 step 40K, σ=0.22). This is a positive but modest OOD signal from the output head alone. The hypothesis — that Bayesian weight uncertainty produces higher MI on OOD text — is confirmed directionally. The magnitude should increase substantially with A2, where FFN layers (which store factual/topical knowledge) become Bayesian.

- **2026-02-26:** A2 Bayesian FFN — code infrastructure complete (pending training run):
  - **Spec**: `specs/a2-bayesian-ffn.md` — full design doc (14 sections).
  - **`model.py`**: `GPTConfig.bayes` renamed to `GPTConfig.bayes_head` for clarity; new `bayes_ffn` field added. `Block` signature changed to `(config, bayes_attn, bayes_ffn)` (Option A — explicit params) — attention always deterministic, FFN configurable. `MiniGPT` wires `config.bayes_ffn` to blocks, `config.bayes_head` to head. Weight tying active when head is deterministic (`config.bayes_head.enabled=False`).
  - **`config.py`**: `DEFAULT_CONFIG` `model.bayes` renamed to `model.bayes_head`; `bayes_ffn` section added (default disabled). `build_gpt_config()` refactored — shared `_build_bayes_config()` helper reads both `bayes_head` and `bayes_ffn`.
  - **`uncertainty.py`**: Auto-detecting dual-path MC sampling. `_has_bayesian_body(model)` checks for BayesianLinear in transformer blocks. `_stream_metrics_full()` does N full forward passes (A2+ path). `compute_uncertainty_metrics()` and `score_sequence()` auto-select A1 (body once, head N times) vs A2 (full pass N times) path.
  - **`train.py`**: `_configure_optimizer()` now excludes `_rho` params from weight decay — rho is regularized by KL, not weight decay.
  - **`configs/a2_agnews.yaml`**: FFN Bayesian (`bayes_ffn.enabled: true`, `init_rho: -3.0`), head deterministic (weight-tied). `eval.num_samples: 20` (reduced from 30 — full passes are more expensive). `kl_weight: 0.2`, `kl_annealing: 10K`, `steps: 50K`.
  - **`experiments/a2_bayes_ffn.py`**: Adapted from A1 script. KL weight sourced from active Bayesian config (FFN or head). Prints which components are Bayesian. Milestone tag `a2`.
  - **Existing YAML configs**: `bayes_ffn: { enabled: false }` added to `a0_baseline.yaml`, `a0_agnews.yaml`, `a1_agnews.yaml` (explicit-config rule).
  - **Tests**: 15 new tests (53 total, all passing): `TestFFNBayesian` (8 — architecture, weight tying, stochastic body, MI), `TestCombinedBayesian` (3 — FFN+head config), `TestHasBayesianBody` (4 — path detection). Ruff clean.
  - **Model size**: ~20M params (4.2M Bayesian in FFN), ~half of A1's 42M — weight tying restored.
  - **Next**: GPU training run: `python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml`. Target: MI ratio > 1.5x, test_id_ppl < 55.

## Future Work (Non-Bayesian — Parked)
These are architectural improvements to revisit **after** Bayesian milestones (A1/A2) are done. Not in scope now — the current miniGPT is intentionally basic to keep focus on Bayesian aspects.

- **RoPE** — Rotary Position Embeddings (replace learned positional embeddings)
- **SwiGLU** — gated FFN activation (replace GELU)
- **Pre-Norm vs Post-Norm** — current model uses pre-norm (GPT-2 style); evaluate post-norm or sandwich-norm
- **KV-cache** — inference-time optimization for autoregressive generation
- **PEFT** — Parameter-Efficient Fine-Tuning (LoRA, adapters) to squeeze more params into VRAM budget
- **Mixed precision (FP16 / BF16)** — **DONE** (AMP auto-enabled on CUDA in train.py + uncertainty.py)

## Papers to Consider (Parked)
Bayesian + LLM papers for later review, likely relevant to B1 (Bayesian LoRA on open-weight LLM).

- **BLoB: Bayesian Low-Rank Adaptation by Backpropagation for LLMs** (NeurIPS 2024) — jointly adjusts mean and covariance of LLM parameters during fine-tuning for better generalization and uncertainty estimation. [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d53575463291ea6b5a23cf6e571f59b-Paper-Conference.pdf)
- **Training-Free Bayesianization for Low-Rank Adapters of LLMs** (2024) — converts trained LoRA adapters into Bayesian ones without additional training by searching for optimal weight variance. [arXiv:2412.05723](https://arxiv.org/pdf/2412.05723)
- **Bayesian Low-rank Adaptation for Large Language Models (Laplace-LoRA)** (2023) — applies Laplace approximation to LoRA parameters, improving calibration and reducing overconfidence. [arXiv:2308.13111](https://arxiv.org/pdf/2308.13111)
- **ScalaBL: Scalable Bayesian Low-Rank Adaptation via Stochastic Variational Subspace Inference** (2025) — Bayesian inference in a low-dimensional subspace (~1000 extra params), scales to larger models than prior approaches. [arXiv:2506.21408](https://arxiv.org/pdf/2506.21408)
