# Repository Guidelines

## Project Structure & Source of Truth
`docs/` holds PDF papers — the theory baseline. Implementations should align with those docs.
`specs/` holds planning documents. The active spec is `specs/refined-spec-feb2026.md`.

Layout (flat, minimal — no deep nesting):
```
minigpt/          # Python package — all model-related code
  model.py        # MiniGPT architecture (deterministic)
  layers.py       # Bayesian layers (BayesianLinear, etc.)
  data.py         # Dataset loading + topic splitting
  train.py        # Training loop (cross-entropy + ELBO)
  evaluate.py     # Standard eval (perplexity, generation)
  uncertainty.py  # Epistemic uncertainty measurement
experiments/      # Runnable .py scripts (a0_baseline, a1_bayes_output, etc.)
tests/            # pytest tests
data/             # Local datasets (gitignored; document provenance in README.md)
```

The old `src/bayesian_llm/` structure is deprecated and will be removed.

## Hard Rules
- **No notebooks.** Only runnable `.py` scripts. Never create `.ipynb` files.
- **No extra Bayesian libraries.** Use only `torch.distributions` for probabilistic layers. Manual implementation is preferred over adding dependencies.
- **No modern transformer tricks** in miniGPT: no RoPE, SwiGLU, sliding window attention, MoE, GQA. Keep it basic — the focus is Bayesian, not architecture.

## Environment & Tooling
- Python 3.12 with `uv` and a `pyproject.toml`
- Use uv only; avoid pip (`uv add`, `uv sync`, `uv run`)
- Document setup in `README.md`
- Target GPU runs (RTX 4070, ~10-12 GB VRAM)
- If uv hits permission errors, set `UV_CACHE_DIR` to a local folder (e.g., `.uv-cache`)

## Framework Decision
PyTorch + `torch.distributions` for all milestones. No JAX for now. No TensorFlow.

## Milestones (order matters)
- **A0:** Deterministic miniGPT baseline — TinyShakespeare, cross-entropy, verify loss/generation/VRAM
- **A1:** Bayesian output head — BayesianLinear on final projection, ELBO training, first uncertainty metrics
- **A2:** Bayesian FFN layers — replace FFN linears, topic-split dataset (AG News), in-distribution vs OOD epistemic uncertainty evaluation
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

## Build, Test, and Development Commands
- `uv venv` and `uv sync`
- `uv run python -m pytest`
- `uv run python experiments/a0_baseline.py` (etc.)

## Coding Style & Naming Conventions
- Indentation: 4 spaces; line length up to 100
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Prefer type hints and short docstrings for public APIs and experiment entry points

## Testing & Evaluation Guidelines
Use `pytest` with `test_*.py` naming. Keep tests deterministic (seed randomness). For experiments, report calibration error, NLL, Brier score, and OOD detection; note whether results target epistemic vs aleatoric uncertainty.

## Commit & Pull Request Guidelines
This directory is not currently a Git repository. If/when Git is initialized, use Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `chore:`, `test:`). PRs should include a short problem/solution summary, links to relevant papers or issues, test evidence, and experiment metadata (model size, method, dataset, metrics).

## Security & Data
Do not commit secrets or large binaries. Use `.env` (ignored) and provide `.env.example` when needed. Keep `data/` out of version control and document licensing constraints in `README.md`.
