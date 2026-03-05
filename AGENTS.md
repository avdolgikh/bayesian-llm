# Repository Guidelines

## Goal
**Estimate epistemic uncertainty in LLMs via Bayesian inference over weights.**

The core idea: replace point-estimate weights with learned posterior distributions, then measure how much the model's predictions disagree across weight samples (mutual information). High MI = "the model knows what it doesn't know" — it's uncertain about out-of-distribution inputs at the weight level, not just the token level.

**Current approach:** Start small (miniGPT on AG News), progressively make layers Bayesian (A1: output head → A2: FFN → A3: attention V), validate that MI separates ID from OOD text, then scale to real LLMs via Bayesian LoRA (B1).

## Project Structure
`docs/` — PDF papers (theory baseline). `specs/` — planning docs (active: `specs/refined-spec-feb2026.md`).

```
minigpt/          # Python package — all model code
  model.py        # MiniGPT architecture (deterministic + selective Bayesian)
  layers.py       # Bayesian layers (BayesianLinear, context managers, sigma stats)
  data.py         # Dataset loading + BPE tokenization — TinyShakespeare, AG News
  train.py        # Training loop (cross-entropy + ELBO)
  evaluate.py     # Perplexity, text generation
  config.py       # YAML config ↔ dataclass bridge
  uncertainty.py  # Epistemic uncertainty (MI via MC sampling)
configs/          # YAML config files per experiment
experiments/      # Runnable scripts (a0_baseline, a1_bayes_output, a2_bayes_ffn; a3 planned)
  runner.py       # Shared runner for Bayesian milestones — A1/A2 are thin wrappers (A3 planned)
scripts/          # Utilities (dump_mlflow_run, profile_gpu, eval_checkpoint)
tests/            # pytest (28 tests)
data/             # Local datasets (gitignored)
```

## Hard Rules
- **No notebooks.** Only `.py` scripts.
- **No extra Bayesian libraries.** Only `torch.distributions`. Manual implementation preferred.
- **No modern transformer tricks** (RoPE, SwiGLU, MoE, etc.). Keep miniGPT basic.
- **Document on the fly** in this file — during implementation, not after.
- **Explicit configs.** Every parameter in YAML — never rely on code defaults.

## Datasets

**TinyShakespeare:** ~1MB, ~304K BPE tokens. Single-domain. `load_shakespeare()` auto-downloads. No OOD split.

**AG News:** 127.6K articles, 4 categories (1=World, 2=Sports, 3=Business, 4=Sci/Tech). Topic split: ID (default World+Sports) → train/val/test_id (80/10/10); OOD (default Business+Sci/Tech) → test_ood. ~2.4M train tokens, ~3.5M OOD tokens. Configurable via `data.id_categories`, `data.ood_categories`.

**Dispatcher:** `load_dataset(cfg, tokenizer)` → `{"train", "val", "test_id", "test_ood"}`.

## Tokenization
BPE via `tiktoken` (GPT-2 encoding, vocab_size=50257).

## Experiment Tracking
- **MLflow** (local, `sqlite:///mlflow.db`). `mlflow.db` and `mlruns/` are **gitignored** (kept local only).
- Logs: hyperparams, train/val loss, perplexity, LR, test_id/test_ood perplexity, MI metrics, sigma stats, generated samples.
- Tags: `dataset`, `milestone` (a0/a1/a2/a3), `gpu`.
- `--no-mlflow` flag to disable. Launch UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Environment & Tooling
- **`uv`** for dev tooling (lint, test, deps). **Global Python** (CUDA PyTorch) for GPU training only.
- Target: RTX 4070 (~12 GB VRAM). AMP auto-enabled on CUDA.
- If uv hits permission errors: `UV_CACHE_DIR=.uv-cache`

## Framework
PyTorch + `torch.distributions`. No JAX, no TensorFlow.

## Milestones

- **A0: DONE** — Deterministic miniGPT on AG News. 4L/4H/256d, 16M params. test_id_ppl=49.11, test_ood_ppl=540.28. MLflow `5dc45450`.
- **A1: DONE** — Bayesian output head (BayesianLinear on lm_head). 42M params (25.7M Bayesian). Best MI ratio **1.36x** (σ=0.22). Ceiling at 1.2–1.4x — detects OOD at vocabulary level only. All ELBO/MI/sigma infrastructure validated. See [A1 Report](#a1-report).
- **A2: DONE** — Bayesian FFN (MLP.fc + MLP.proj, 4 blocks). 20M params (4.2M Bayesian, weight-tied head). Best MI ratio **1.43x batch / 1.70x qualitative** (σ mean=0.147, posteriors learned). Confirmation run (seed=2352) reproduced separation at **1.36x batch / 1.55x qualitative**. See [A2 Report](#a2-report).
- **A3: PLANNED** — Bayesian FFN + Bayesian attention value projection (`V`) with diagonal posteriors. Keep `Q/K` deterministic for stability and attribution. Spec: `specs/a3-bayesian-ffn-attention-v.md`.
- **B1 (future):** Bayesian LoRA on open-weight LLM.

## Bayesian Layer Strategy (order)
1. Output head (A1) — simplest, proves pipeline
2. FFN layers (A2) — strongest epistemic signal (FFN stores factual knowledge)
3. Attention value projection `V` (A3) — add content-path uncertainty with stable routing
4. Q/K projections — optional, later
5. Embeddings — optional, later

## Epistemic Uncertainty Measurement
- Temperature = 0. All stochasticity from Bayesian weights.
- N forward passes (10–30) with weight sampling.
- Primary metric: **MI** (predictive_entropy − expected_entropy) — pure epistemic uncertainty.
- Secondary: flip rate, per-token MI scores.
- Evaluation: train on ID topics, compare MI on ID vs OOD.

## Configuration System
Pipeline: `DEFAULT_CONFIG → YAML file → CLI --set overrides → validate`.

Configs: `a0_baseline.yaml` (TinyShakespeare), `a0_agnews.yaml`, `a1_agnews.yaml`, `a2_agnews.yaml`, `a3_agnews.yaml` (planned).

Key conventions:
- `vocab_size` always from tokenizer, never config.
- `model.bayes_head` for output head config, `model.bayes_ffn` for FFN config.
- **PyYAML gotcha:** write `3.0e-4` not `3e-4` (parsed as string without decimal).

### Experiment CLI
```bash
python experiments/a0_baseline.py --config configs/a0_agnews.yaml
python experiments/a1_bayes_output.py --config configs/a1_agnews.yaml
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml
python experiments/a0_baseline.py --config configs/a0_agnews.yaml --set train.lr=1e-3
python experiments/a0_baseline.py --config configs/a0_agnews.yaml --resume data/checkpoints/ckpt_step500.pt
```

### Checkpoint Resume
Saves full config, `best_val_loss`, RNG states. LR schedule is stateless (computed from step). Best-checkpoint criterion: ELBO for Bayesian, CE for deterministic.

## Build & Dev Commands
```bash
uv sync                                          # install deps
uv run pytest tests/ -v                          # 28 unit tests
uv run ruff check minigpt/ experiments/ tests/   # lint
python experiments/a0_baseline.py                # A0 training (GPU)
python experiments/a1_bayes_output.py --config configs/a1_agnews.yaml  # A1
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml     # A2
python scripts/dump_mlflow_run.py <run_id>       # inspect run
python scripts/eval_checkpoint.py <ckpt> --config <yaml>  # eval any checkpoint
```

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`): `ruff check` → `pytest`. No GPU in CI.

## Coding Style
4 spaces, 100-char lines (ruff). `snake_case` functions, `PascalCase` classes. Type hints on public APIs.

## Tests (28 total)
```
tests/
  test_model.py          # Weight tying (2), perplexity bounds (1)
  test_data.py           # Split sizes + nonempty (3)
  test_reproducibility.py # Same seed = identical losses (1)
  test_bayesian.py       # KL invariants (2), sampling (2), context managers (2),
                         # A1 architecture (4), MI invariants (4),
                         # A2 FFN architecture (5), path detection (2)
```

## Commit Guidelines
- One-line messages, Conventional Commits (`feat:`, `fix:`, `docs:`, etc.).
- **Never mention AI assistants** in commits, comments, PRs, or code.

## Security
No secrets or large binaries in git. `data/`, `mlflow.db`, `mlruns/` are gitignored.

---

## A1 Report

### A1 Training Runs

| Run | init_rho | σ final | MI (ID) | MI (OOD) | MI ratio | ID ppl | OOD ppl | Notes |
|-----|----------|---------|---------|----------|----------|--------|---------|-------|
| R0 | -5 | ~0 | 0.024 | 0.025 | 1.07x | 65.6 | 630 | kl_weight=0.01, posterior collapse |
| R1 (40K) | -5 | 0.224 | 0.293 | 0.398 | **1.36x** | 56.3 | 1477 | kl_weight=0.2, best MI ratio |
| R2 (48K) | -2 | 0.637 | 0.317 | 0.387 | 1.22x | 65.8 | 1069 | ELBO-selected, σ too wide |

MLflow runs: R0=`686e3201`, R1=`4e1662e6`, R2=`5dff1029`.

### A1 Findings

1. **Optimal sigma window: σ ≈ 0.1–0.3.** Below → posterior collapse (MI ≈ 0). Above → uniform noise across all tokens, MI ratio drops. R1 at σ=0.22 hit the sweet spot.

2. **Higher σ hurts ID perplexity.** Each increase in posterior width costs prediction quality (49.1 → 56.3 → 65.8). KL pushes σ toward prior, overpowering CE.

3. **Output head has a structural ceiling: MI ratio 1.2–1.4x.** The head is a linear projection — can only express vocabulary-level uncertainty. The factual knowledge about topics lives in FFN layers.

4. **Qualitative MI confirms vocabulary-level limitation.** Category gradient: Sci/Tech (jargon) > Business > World ≈ Sports. Tracks vocabulary specialization, not conceptual novelty.

5. **Key fixes discovered during A1:** OOM in uncertainty eval → streaming MC sampling. Posterior collapse with kl_weight=0.01 → raised to 0.2. CE-based checkpoint selection missed Bayesian optimum → switched to ELBO selection. Qualitative MI scored generated text (always ID) → fixed to score prompt tokens. Softplus saturation at init_rho=-5 → default changed to -1.0 (configs use -2.0).

### A1 Infrastructure Validated
ELBO training + KL annealing, best-ELBO checkpoint, streaming MC sampling (AMP-compatible), MI/entropy/flip rate metrics, qualitative prompt-panel scoring, sigma summary logging, gradient accumulation, `_rho` excluded from weight decay.

---

## A2 Report

### A2 Training Runs

| Metric | A0 (baseline) | A1 best (R1 40K) | A2 R1 (50K) | **A2 R2 (98K)** |
|--------|---------------|------------------|-------------|-----------------|
| Model | deterministic | bayes head | bayes FFN | **bayes FFN** |
| Params (Bayesian) | 16M (0) | 42M (25.7M) | 20M (4.2M) | **20M (4.2M)** |
| init_rho | — | -5 | -3 | **-2** |
| σ mean | — | 0.224 | 0.048 | **0.147** |
| σ range | — | [.015, .441] | [.027, .090] | **[.036, .966]** |
| MI ratio (batch) | — | 1.36x | 1.38x | **1.43x** |
| MI ratio (qual) | — | 1.28x | 1.57x | **1.70x** |
| Test ID ppl | 49.1 | 56.3 | 51.0 | **53.5** |
| Test OOD ppl | 540 | 1477 | 500 | **595** |
| KL (final) | — | 15.7M | 5.35M | **3.20M** |
| KL trend | — | ↓ (58→16M) | ↑ pathological | **↓ then flat (healthy)** |

MLflow runs: R1=`93ff41da`, R2=`76d049b7`, R3=`1238951099144292844c33258721fa80`.

**A2 R1** (init_rho=-3, 50K steps): Posteriors frozen at init (σ stuck at softplus(-3)=0.049). KL rising (structural, not informative). Despite this, MI ratio 1.38x batch / 1.57x qualitative — matching A1's best with 6x fewer Bayesian params.

**A2 R2** (init_rho=-2, 100K steps): Posteriors learned. σ range [0.036, 0.966] — model differentiated which weights need certainty vs uncertainty. KL decreased then stabilized (healthy). Best ELBO at step 98K.

**A2 R3 (confirmation run, seed=2352)** (init_rho=-2, 100K steps): Core behavior reproduced on a new run. Sigma stats remained healthy (mean=0.1509, range [0.0229, 0.9382]), MI separated ID/OOD (batch=1.36x, qualitative=1.55x), and KL remained stable (~3.17M). This run confirms A2 signal is robust, with expected variance vs the best-seed run.

### A2 R2 Qualitative MI (prompt-token scoring)
- World (ID): 0.050 | Sports (ID): 0.053 | Business (OOD): 0.090 | Sci/Tech (OOD): 0.088
- Overall: ID=0.052, OOD=0.089, **ratio=1.70x**. Clean category separation.

### A2 Findings

1. **init_rho=-2 unlocks posterior learning for FFN.** R1 (ρ=-3): σ frozen, std=0.004. R2 (ρ=-2): σ spread [0.036, 0.966], std=0.074. The sigmoid gradient at ρ=-2 (0.12) provides 2.4x more flow than ρ=-3 (0.05).

2. **Posterior learning adds modest but real signal.** Frozen posteriors (R1): 1.38x. Learned posteriors (R2): 1.43x batch, 1.70x qualitative. Curated prompts benefit more from posterior differentiation.

3. **Healthy KL trajectory confirms correct optimization.** R1 had pathological rising KL (structural log-ratio penalty). R2 starts closer to prior → KL decreases → equilibrium at ~3.2M.

4. **FFN uncertainty is topic-level, not vocabulary-level.** In A1: Sci/Tech >> Business (vocabulary jargon). In A2: Business ≈ Sci/Tech — FFN detects unfamiliar content patterns, not unfamiliar words.

5. **FFN is more parameter-efficient than output head.** 4.2M Bayesian params produce better MI separation than A1's 25.7M. Weight tying preserved (head deterministic).

6. **A2 is reproducible enough to close.** The seed-2352 confirmation run preserved the same qualitative behavior (OOD MI > ID MI, learned posteriors, healthy KL), so A2 is closed and B1 is the next milestone.

### Tuning Levers (if further runs needed)
- If posteriors frozen: increase init_rho (try -1) or reduce kl_weight (0.05–0.1).
- If σ too wide (uniform noise): decrease init_rho or reduce prior_std (0.3–0.5).
- One variable at a time.

---

## Future Work (Parked)
Non-Bayesian improvements — after A2 tuning complete:
- RoPE, SwiGLU, KV-cache, PEFT
- Mixed precision: **DONE** (AMP auto-enabled)

## Papers (B1-relevant)
- **BLoB** (NeurIPS 2024) — Bayesian LoRA by backprop. [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d53575463291ea6b5a23cf6e571f59b-Paper-Conference.pdf)
- **Training-Free Bayesianization for LoRA** (2024) — post-hoc variance search. [arXiv:2412.05723](https://arxiv.org/pdf/2412.05723)
- **Laplace-LoRA** (2023) — Laplace approximation on LoRA params. [arXiv:2308.13111](https://arxiv.org/pdf/2308.13111)
- **ScalaBL** (2025) — Bayesian inference in low-dim subspace. [arXiv:2506.21408](https://arxiv.org/pdf/2506.21408)
