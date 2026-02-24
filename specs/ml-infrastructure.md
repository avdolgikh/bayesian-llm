# ML Infrastructure — Tracking, Registry, Tuning, CI/CD

**Date:** 2026-02-24
**Status:** Proposed

---

## 1. MLflow Tracking Improvements

### Current state

- Hyperparameters logged as params via `config_to_flat_params()`
- Train/val loss logged per eval interval
- LR logged every step
- `final_val_perplexity`, `test_id_perplexity`, `test_ood_perplexity` logged as metrics
- Generated samples logged as text artifacts

### Additions

#### Best val_loss step

Track when the best checkpoint occurred — overfitting signal. If best is at step 200/5000, the model is massively overfitting past that point.

```python
mlflow.log_metric("best_val_loss", best_val_loss)
mlflow.log_metric("best_val_step", best_step)
```

#### Tags (metadata, not params)

MLflow distinguishes params (hyperparameters) from tags (metadata). Dataset and milestone are metadata — filterable in UI, show as columns.

```python
mlflow.set_tag("dataset", cfg["data"]["dataset"])  # "tinyshakespeare" or "agnews"
mlflow.set_tag("milestone", "a0")                   # phase identifier
mlflow.set_tag("gpu", torch.cuda.get_device_name()) # reproducibility
```

#### Additional metrics

| Metric | Why |
|---|---|
| `train_time_sec` | Compare wall-clock cost across configs |
| `tokens_per_sec` | Throughput — matters when scaling to A1/A2 (Bayesian layers are slower) |

#### What to skip

System metrics (CPU/RAM via `mlflow.system_metrics`) — noise for a research project. Keep the signal clean.

---

## 2. Model Registry

### Problem it solves

"Which checkpoint is the current best for milestone X?" — not serving, not deployment, just lifecycle management for research.

### Two layers in MLflow

1. **Run artifacts** — `mlflow.log_artifact()` or `mlflow.pytorch.log_model()`. Tied to a specific run. Just storage.
2. **Model Registry** — Named models with versions, stages (Staging/Production/Archived), descriptions. Cross-run. Lifecycle management.

### Approach for this project

#### During training — log both formats

```python
# Raw checkpoint (current workflow, fast to reload)
mlflow.log_artifact("data/checkpoints/ckpt_best.pt")

# MLflow model format (enables mlflow.pytorch.load_model() later)
mlflow.pytorch.log_model(model, "model")
```

#### Promote milestone models manually

```bash
# "This is the best A0 baseline"
mlflow models create -n "minigpt-a0"
mlflow models create-version -n "minigpt-a0" --source "runs:/<run_id>/model"
```

Use registry when comparing across milestones (A0 vs A1 vs A2) — each milestone gets a named model with versions.

#### Model card = the run itself

MLflow already captures all hyperparameters, metrics, dataset tag, config YAML (as artifact). Add a human-readable summary:

```python
mlflow.set_tag("mlflow.note.content", "A0 baseline, 4L/4H/256d, TinyShakespeare, best_val=X.XX")
```

No separate model card format needed.

#### Lineage

Automatic — every registered model version points back to its source run, which has all params/metrics/artifacts.

#### What to skip

- Docker images — not serving
- Custom model flavors — `mlflow.pytorch` is sufficient
- Model signatures — useful for serving APIs, not research

---

## 3. Hyperparameter Tuning Pipeline

### Option A: Simple sweep script (recommended first)

Pure Python + MLflow. Zero extra dependencies.

```python
# experiments/sweep_a0.py
import itertools
import mlflow

grid = {
    "model.n_layer": [4, 6],
    "model.n_embd": [256, 384],
    "train.lr": [1e-4, 3e-4, 1e-3],
    "train.dropout": [0.1, 0.2, 0.3],
}

with mlflow.start_run(run_name="sweep-a0") as parent:
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        with mlflow.start_run(run_name="...", nested=True):
            # build config with overrides from params
            # train model
            # log metrics
            ...
```

Key concept: **MLflow nested runs** — parent run groups the sweep, child runs are individual trials. The UI shows them as a collapsible tree.

Feasibility: grid of 2x2x3x3 = 36 combos, ~5000 steps each at ~2 min/run = ~1 hour on RTX 4070.

### Option B: Optuna (if search space grows)

Add Optuna when doing 50+ trial sweeps. Benefits over grid search:

- **TPE (Bayesian search)** — finds good configs in fewer trials than grid
- **Pruning** — kills bad runs early (if val loss is terrible at step 500, skip remaining 4500)
- **MLflow integration** — `optuna.integration.MLflowCallback`

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    n_layer = trial.suggest_int("n_layer", 4, 8)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    # ... build config, train, return best_val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

### Recommendation

Start with Option A (sweep script). The search space for A0/A1 is small enough that grid/random works. Add Optuna later if needed — it's a one-line install (`pip install optuna`).

The infrastructure choice that matters: **parent/child MLflow runs**. This organizes sweeps cleanly regardless of search strategy.

---

## 4. Unit Tests, Guardrails, CI/CD

### Unit test organization

Tests from `specs/vital-unit-tests.md`, organized by module:

```
tests/
  test_model.py       # weight tying, perplexity bounds at init
  test_data.py        # category isolation, split sizes
  test_uncertainty.py  # MI=0 for deterministic, sampling variance (Phase 2)
```

All tests run on CPU with small tensors — no GPU required, fast execution.

### GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: python -m pytest tests/ -v
```

### What goes in CI vs what doesn't

| In CI (fast, CPU) | NOT in CI (slow, GPU) |
|---|---|
| Unit tests (P0/P1/P2) | Training runs |
| Lint (ruff) | Model evaluation |
| Import smoke test | Sweep runs |
| Config validation | MLflow integration tests |

No GPU in GitHub Actions — runners are CPU-only (GPU runners exist but are expensive and overkill for this project). The unit tests are designed to validate invariants, not train models.

### Linting

**Ruff** over flake8 — faster, replaces both flake8 and isort, single config in `pyproject.toml`.

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W"]
```

---

## Priority Order

| Priority | What | Value |
|---|---|---|
| Now (before A1) | P0 unit tests + MLflow tracking improvements | Prevents silent bugs, better experiment visibility |
| Soon | Sweep script with nested runs + GitHub Actions CI | Systematic tuning, automated quality gate |
| Later | Model Registry promotion | When comparing A0 vs A1 vs A2 milestone models |
| If needed | Optuna | When search space grows beyond grid feasibility |
