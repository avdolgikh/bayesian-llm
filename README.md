# Bayesian LLM Experiments

Minimal, hands-on experiments for Bayesian methods in LLMs. The two PDFs in `docs/` are the theory baseline and should guide implementation choices.

## Quickstart (uv + Python 3.12)
```bash
uv venv
uv sync
uv run python -m bayesian_llm.experiments.a0_minigpt
```

If uv reports permission errors, set a repo-local cache:
```powershell
$env:UV_CACHE_DIR=".uv-cache"; uv venv; uv sync
```

## Framework Choice (current)
For milestones A0-A2, the repo uses PyTorch with `torch.distributions` to keep Bayesian layers and ELBO/KL wiring simple and explicit. JAX can be revisited later if it provides lower boilerplate.

Dependencies are managed exclusively with uv (`uv add`, `uv sync`, `uv run`).

## Experiments
- A0: Deterministic mini-GPT baseline (`python -m bayesian_llm.experiments.a0_minigpt`)
- A1: First Bayesian component (planned)
- A2: Broader Bayesian coverage (planned)
- B1: Bayesian LoRA on a small existing model (later)

## Repo Structure (initial)
- `docs/` theory references (source of truth)
- `specs/` project spec and decision notes
- `src/bayesian_llm/` code
  - `model/` transformer + Bayesian variants
  - `train/` training loops and objectives
  - `data/` tiny dataset loaders
  - `experiments/` runnable scripts
- `tests/` unit and smoke tests
- `data/` local datasets (gitignored)

## Notes
See `NOTES.md` for running decisions and experiment logs.
