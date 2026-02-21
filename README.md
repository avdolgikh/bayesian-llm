# Bayesian LLM Experiments

Minimal, hands-on experiments for Bayesian methods in LLMs. The two PDFs in `docs/` are the theory baseline and should guide implementation choices.

## Quickstart (uv + Python 3.12)
```bash
uv venv
uv sync
uv run python experiments/a0_baseline.py
```

If uv reports permission errors, set a repo-local cache:
```powershell
$env:UV_CACHE_DIR=".uv-cache"; uv venv; uv sync
```

## Dataset
TinyShakespeare (~1.1 MB, ~304k BPE tokens) — auto-downloaded on first run to `data/tinyshakespeare.txt`.

## Tokenizer
GPT-2 BPE via `tiktoken` (vocab_size=50,257).

## Experiment Tracking (MLflow)
All runs are logged to a local SQLite-backed MLflow store (`mlflow.db`).

View results:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://127.0.0.1:5000. Use `--no-mlflow` flag on experiment scripts to disable tracking.

## Framework Choice
PyTorch with `torch.distributions` for Bayesian layers and ELBO/KL wiring. Dependencies managed with uv.

## Experiments
- **A0:** Deterministic miniGPT baseline (`uv run python experiments/a0_baseline.py`)
- **A1:** Bayesian output head (planned)
- **A2:** Bayesian FFN layers + OOD evaluation (planned)
- **B1:** Bayesian LoRA on existing model (later)

## Repo Structure
```
minigpt/          # Python package
  layers.py       # Bayesian layers (BayesianLinear, reparameterization trick, KL)
  model.py        # MiniGPT (CausalSelfAttention, MLP, Block, MiniGPT)
  data.py         # TinyShakespeare download + BPE tokenization
  train.py        # Training loop with perplexity, checkpoints, MLflow
  evaluate.py     # Perplexity computation + text generation
  uncertainty.py  # Epistemic uncertainty (Phase 2)
experiments/      # Runnable .py scripts
tests/            # pytest tests
data/             # Local datasets + checkpoints (gitignored)
docs/             # Theory references
specs/            # Project spec and decision notes
```

## Notes
See `NOTES.md` for running decisions and experiment logs.
