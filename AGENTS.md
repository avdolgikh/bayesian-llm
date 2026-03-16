# Repository Guidelines

## Goal
**Estimate epistemic uncertainty in LLMs via Bayesian inference over weights.**

The core idea: replace point-estimate weights with learned posterior distributions, then measure how much the model's predictions disagree across weight samples (mutual information). High MI = "the model knows what it doesn't know" — it's uncertain about out-of-distribution inputs at the weight level, not just the token level.

**Current approach:** Controlled comparative study on miniGPT (AG News). Four Bayesian methods in a 2x2 matrix: (variational vs post-hoc) x (full weights vs LoRA). A-series (variational full weights) done. B-series: B1 Laplace (post-hoc full weights), B2 BLoB (variational LoRA), B3 TFB/Laplace-LoRA (post-hoc LoRA). Active spec: `specs/comparative-bayesian-llm-study.md`.

## Project Structure
`docs/` — PDF papers (theory baseline). `specs/` — planning docs (active: `specs/comparative-bayesian-llm-study.md`). B1 tech spec: `specs/b1-laplace-tech-spec-mar2026.md`. B1 analysis: `specs/b1-laplace-analysis.md`. B2 spec: `specs/b2-blob-lora-spec.md`. B2 R2 revised plan: `specs/b2-r2-revised-plan.md`. B2 code review: `specs/b2-blob-lora-review.md`. B3 spec: `specs/b3-post-hoc-lora-spec.md`.

```
minigpt/          # Python package — all model code
  model.py        # MiniGPT architecture (deterministic + selective Bayesian)
  layers.py       # Bayesian layers (BayesianLinear, context managers, sigma stats)
  data.py         # Dataset loading + BPE tokenization — TinyShakespeare, AG News
  train.py        # Training loop (cross-entropy + ELBO)
  evaluate.py     # Perplexity, text generation
  config.py       # YAML config ↔ dataclass bridge
  uncertainty.py  # Epistemic uncertainty (MI via MC sampling)
  laplace.py      # Post-hoc Laplace: curvature fitting, sampling, context manager
  lora.py         # LoRA: BLoBLoRALinear, DeterministicLoRALinear, LoRAConfig, inject_lora()
  tfb.py          # TFB: Training-Free Bayesianization (SVD variance search, sampling)
configs/          # YAML config files per experiment
experiments/      # Runnable scripts (a0_baseline, a1–a3 Bayesian, b1_laplace_baseline, b2_blob_lora, b3_post_hoc_lora)
  experiment_setup.py  # Shared setup: CLI parsing, config, data, model, device
  eval_utils.py        # Shared eval: perplexity suite, MI suite, qualitative eval
  mlflow_utils.py      # Shared MLflow: context, logging helpers
  runner.py            # A-series orchestrator — A1/A2/A3 are thin wrappers
scripts/          # Utilities (dump_mlflow_run, compare_runs, profile_gpu, eval_checkpoint, fit_laplace)
tests/            # pytest (84 tests)
data/             # Local datasets (gitignored)
```

## Hard Rules
- **No notebooks.** Only `.py` scripts.
- **No extra Bayesian libraries.** Only `torch.distributions`. Manual implementation preferred.
- **No modern transformer tricks** (RoPE, SwiGLU, MoE, etc.). Keep miniGPT basic.
- **Document on the fly** in this file — during implementation, not after.
- **No unsolicited AGENTS.md cleanup.** Do not reformat, normalize typography/encoding, or rewrite text unless explicitly requested; only apply the exact requested doc change.
- **Explicit configs.** Every parameter in YAML — never rely on code defaults.
- **LaTeX formulas in specs.** All formulas in `specs/` documents must use LaTeX (`$...$` inline, `$$...$$` display).

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
- **A3: CLOSED** — Bayesian FFN + Bayesian attention value projection (`V`) with diagonal posteriors. Four runs completed (`660914c3`, `1848a95f`, `c7855477`, `910b0b43`); all underperformed A2 on MI separation and ID perplexity. Final archived A3 config (no rerun): `bayes_ffn.init_rho=-2.0`, `bayes_ffn.prior_std=1.0`, `bayes_attn_v.init_rho=-2.0`, `bayes_attn_v.prior_std=1.0`, `train.kl_weight=0.2` (reference run `1848a95f`). Spec: `specs/a3-bayesian-ffn-attention-v.md`.
- **B1: DONE (NEGATIVE)** — Post-hoc Laplace on deterministic checkpoint. Approach A (identity-curvature sweep): MI ratio 1.00x at all scales. Approach B (per-sample Fisher): curvature non-zero but still MI ratio 1.00x. **Conclusion: diagonal Laplace on FFN params does not produce OOD-discriminative uncertainty in language models.** See [B1 Implementation Notes](#b1-implementation-notes).
- **B2: DONE (WEAK POSITIVE)** — BLoB-style Bayesian LoRA (variational, train-time). R1 inconclusive (TinyShakespeare pretrain too small). R2: category-split pretrain (cat 1 World, val ppl=46.5), LoRA fine-tune (cat 2 Sports), OOD eval (cats 3+4). MI ratio **1.13x batch / 1.02x qual**. Weak but positive — BLoB LoRA detects OOD at batch level with 163K Bayesian params (25x fewer than A2's 4.2M). Signal weaker than A2 (1.43x/1.70x). See [B2 Implementation Notes](#b2-implementation-notes).
- **B3: DONE (MIXED)** — Post-hoc Bayesianization of deterministic LoRA. **B3-TFB: MI ratio 1.10x** (σ_q=0.013, SVD-structured variance works). **B3-LAP: MI ratio 1.00x** (diagonal curvature too flat, same failure as B1). TFB succeeds post-hoc because it uses structural information (SVD of B), not curvature. See [B3 Implementation Notes](#b3-implementation-notes). Spec: `specs/b3-post-hoc-lora-spec.md`.

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

Configs: `a0_baseline.yaml` (TinyShakespeare), `a0_agnews.yaml`, `a1_agnews.yaml`, `a2_agnews.yaml`, `a3_agnews.yaml`, `b1_laplace_agnews.yaml`, `b2_pretrain_shakespeare.yaml` (obsolete — R1), `b2_pretrain_agnews.yaml` (R2), `b2_blob_agnews.yaml`, `b3_lora_agnews.yaml` (B3 — deterministic LoRA + TFB/Laplace).

Key conventions:
- `vocab_size` always from tokenizer, never config.
- `model.bayes_head` for output head config, `model.bayes_ffn` for FFN config,
  `model.bayes_attn_v` for attention value projection config.
- `train.kl_weight` is global for all enabled Bayesian components.
- **PyYAML gotcha:** write `3.0e-4` not `3e-4` (parsed as string without decimal).

### Experiment CLI
```bash
python experiments/a0_baseline.py --config configs/a0_agnews.yaml
python experiments/a1_bayes_output.py --config configs/a1_agnews.yaml
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml
python experiments/a3_bayes_ffn_attn_v.py --config configs/a3_agnews.yaml
python experiments/b1_laplace_baseline.py --config configs/b1_laplace_agnews.yaml
python experiments/b1_laplace_baseline.py --config configs/b1_laplace_agnews.yaml --skip-train --laplace-state data/checkpoints/laplace_state.pt
python experiments/b2_blob_lora.py --phase pretrain --pretrain-config configs/b2_pretrain_agnews.yaml
python experiments/b2_blob_lora.py --phase finetune --config configs/b2_blob_agnews.yaml
python experiments/b2_blob_lora.py --phase full --pretrain-config configs/b2_pretrain_agnews.yaml --config configs/b2_blob_agnews.yaml
python experiments/b3_post_hoc_lora.py --phase train --config configs/b3_lora_agnews.yaml
python experiments/b3_post_hoc_lora.py --phase tfb --config configs/b3_lora_agnews.yaml
python experiments/b3_post_hoc_lora.py --phase laplace --config configs/b3_lora_agnews.yaml
python experiments/b3_post_hoc_lora.py --phase full --config configs/b3_lora_agnews.yaml
python experiments/a0_baseline.py --config configs/a0_agnews.yaml --set train.lr=1e-3
python experiments/a0_baseline.py --config configs/a0_agnews.yaml --resume data/checkpoints/ckpt_step500.pt
```

### Checkpoint Resume
Saves full config, `best_val_loss`, RNG states. LR schedule is stateless (computed from step). Best-checkpoint criterion: ELBO for Bayesian, CE for deterministic.

## Build & Dev Commands
```bash
uv sync                                          # install deps
uv run pytest tests/ -v                          # 103 unit tests
uv run ruff check minigpt/ experiments/ tests/   # lint
python experiments/a0_baseline.py                # A0 training (GPU)
python experiments/a1_bayes_output.py --config configs/a1_agnews.yaml  # A1
python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml     # A2
python experiments/a3_bayes_ffn_attn_v.py --config configs/a3_agnews.yaml  # A3
python experiments/b1_laplace_baseline.py --config configs/b1_laplace_agnews.yaml  # B1
python experiments/b2_blob_lora.py --phase pretrain --pretrain-config configs/b2_pretrain_agnews.yaml  # B2 pretrain
python experiments/b2_blob_lora.py --phase finetune --config configs/b2_blob_agnews.yaml              # B2 finetune
python experiments/b3_post_hoc_lora.py --phase train --config configs/b3_lora_agnews.yaml             # B3 det LoRA
python experiments/b3_post_hoc_lora.py --phase tfb --config configs/b3_lora_agnews.yaml               # B3 TFB
python experiments/b3_post_hoc_lora.py --phase laplace --config configs/b3_lora_agnews.yaml           # B3 Laplace
python scripts/dump_mlflow_run.py <run_id>       # inspect run
python scripts/dump_mlflow_run.py latest          # inspect most recent run
uv run python scripts/compare_runs.py --runs <run_id...> --baseline <run_id>  # compare runs vs gates
UV_CACHE_DIR=.uv-cache uv run python scripts/compare_runs.py --runs <run_id...> --baseline <run_id>  # if uv cache permission errors
python scripts/eval_checkpoint.py <ckpt> --config <yaml>  # eval any checkpoint
```

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`): `ruff check` → `pytest`. No GPU in CI.
**Always run lint + tests locally before committing:** `uv run ruff check minigpt/ experiments/ tests/ && uv run pytest tests/ -x`

## Coding Style
4 spaces, 100-char lines (ruff). `snake_case` functions, `PascalCase` classes. Type hints on public APIs.

## Tests (103 total)
```
tests/
  test_model.py          # Weight tying (2), perplexity bounds (1)
  test_data.py           # Split sizes + nonempty (3)
  test_reproducibility.py # Same seed = identical losses (1)
  test_bayesian.py       # KL invariants (2), sampling (2), context managers (2),
                         # A1 architecture (4), MI invariants (4),
                         # A2 FFN architecture (5), A3 Attn-V architecture (3),
                         # path detection (3)
  test_lora.py           # B2 LoRA: injection (3), BLoBLoRALinear forward/KL/sampling (5),
                         # config (3), context managers (3), sigma_summary (2),
                         # freeze/grad (2), inject_lora validation (2), optimizer (2),
                         # uncertainty detection (2), checkpoint roundtrip (2),
                         # prior_std/init_g wiring (2), alpha scaling (1), rank sweep (1)
  test_laplace.py        # B1 Laplace: curvature shapes (1), damping (1),
                         # sampling reproducibility (2), zero scale (1),
                         # logit variation (1), pipeline integration (2),
                         # checkpoint roundtrip (1), param scope (2),
                         # selection modes (4), curvature invariants (3),
                         # scale/damping modulation (2), fit side effects (2),
                         # loaded state sampling (1), e2e smoke (1),
                         # B3 Laplace-LoRA: lora selection (1), curvature shapes (1),
                         # sampling changes logits (1)
  test_b3_deterministic_lora.py  # B3 DeterministicLoRALinear: forward matches base (1),
                         # no kl (1), params trainable (1), forward deterministic (1),
                         # inject_lora bayesian=False (1), freeze base (1),
                         # param count (1)
  test_tfb.py            # B3 TFB: SVD shapes (1), variance structure (1),
                         # sampling reproducible (1), zero sigma (1),
                         # search converges (1), save/load roundtrip (1),
                         # search respects tolerance (1),
                         # sampling changes logits (1), MC metrics protocol (1)
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

## A3 Implementation Notes

- Implemented architecture delta only: attention `V` projection is now optionally Bayesian; `Q/K` and
  attention output projection remain deterministic.
- Added config surface `model.bayes_attn_v` (defaults + YAML wiring).
- Refactored attention from fused `qkv` to `q_proj/k_proj/v_proj` so `v_proj` can be Bayesian in isolation.
- Added A3 experiment entrypoint: `experiments/a3_bayes_ffn_attn_v.py`.
- Added A3 config: `configs/a3_agnews.yaml`.
- Updated shared runner KL resolution: single global `train.kl_weight` (no per-component KL weight).
- Updated tests for attention split and A3 invariants.

Validation:
- `uv run pytest tests/ -q` → `32 passed`
- `uv run ruff check minigpt/ experiments/ tests/` → passed

First training run:
- Command: `python experiments/a3_bayes_ffn_attn_v.py --config configs/a3_agnews.yaml`
- MLflow run: `660914c3a2744188bd197afa3b4901ae`
- Seed: `1337`
- Model params: `18,456,320` (Bayesian: `4,730,880`)
- Best val loss (ELBO criterion): `4.3734` at step `90,000`
- Sigma stats: mean=`0.1329`, std=`0.0594`, min=`0.0244`, max=`0.9043`
- Test ID/OOD perplexity: `59.20` / `554.70`
- MI (ID/OOD): `0.0582` / `0.0753` → batch ratio `1.29x`
- Qualitative MI ratio: `1.39x`
- Final KL loss: `3.83M`

Conclusion:
- A3 architecture works end-to-end, but this initial run is a regression vs A2 on MI separation and
  ID perplexity. A3 remains in tuning.
- Likely issue: attention `v_proj` posteriors are under-learning with `init_rho=-3.0`
  (sigma means stayed low, ~`0.04–0.07` per block) compared to FFN posteriors.

Updated tuning history:
- Run 2 already tested `model.bayes_attn_v.init_rho=-2.0` (from `-3.0`).
- Next experiment tested lower KL pressure via `train.kl_weight=0.15`.

Second training run:
- MLflow run: `1848a95f9d494a6baca5e41a5cd65829`
- Key config change vs run 1: `model.bayes_attn_v.init_rho=-2.0` (from `-3.0`)
- Best val loss (ELBO criterion): `4.4310` at step `90,000`
- Sigma stats: mean=`0.1405`, std=`0.0559`, min=`0.0365`, max=`0.9036`
- Test ID/OOD perplexity: `61.97` / `573.95`
- MI (ID/OOD): `0.0621` / `0.0789` → batch ratio `1.27x`
- Qualitative MI ratio: `1.41x`
- Final KL loss: `3.63M`

Third training run:
- Command: `python experiments/a3_bayes_ffn_attn_v.py --config configs/a3_agnews.yaml`
- MLflow run: `c7855477d3bf4ae09bd66fcb3949351a`
- Key config change vs run 2: `train.kl_weight=0.15` (from `0.2`)
- Best val loss (ELBO criterion): `4.3571` at step `90,000`
- Sigma stats: mean=`0.1253`, std=`0.0338`, min=`0.0375`, max=`0.7410`
- Test ID/OOD perplexity: `61.23` / `551.91`
- MI (ID/OOD): `0.0618` / `0.0795` -> batch ratio `1.29x`
- Qualitative MI ratio: `1.41x`
- Final KL loss: `3.85M`

Fourth training run:
- Command: `python experiments/a3_bayes_ffn_attn_v.py --config configs/a3_agnews.yaml --set model.bayes_attn_v.prior_std=0.7`
- MLflow run: `910b0b43a53a48518936f0ac41972e7a`
- Key config change vs run 2: `model.bayes_attn_v.prior_std=0.7` (from `1.0`, with `train.kl_weight=0.2`)
- Best val loss (ELBO criterion): `4.4204` at step `80,000`
- Sigma stats: mean=`0.1388`, std=`0.0502`, min=`0.0385`, max=`0.8372`
- Test ID/OOD perplexity: `62.33` / `558.66`
- MI (ID/OOD): `0.0625` / `0.0786` -> batch ratio `1.26x`
- Qualitative MI ratio: `1.35x`
- Final KL loss: `3.56M`

KL trajectory diagnostics (MLflow metric history, 51 eval points each):
- A2 best (`76d049b7...`): KL `3.306M` -> `3.200M` (overall decreasing).
- A2 repro (`12389510...`): KL `3.306M` -> `3.172M` (overall decreasing).
- A3 run 1 (`660914c3...`, `kl_weight=0.2`): KL `3.971M` -> `3.832M` (overall decreasing).
- A3 run 2 (`1848a95f...`, `kl_weight=0.2`): KL `3.720M` -> `3.633M` (overall decreasing).
- A3 run 3 (`c7855477...`, `kl_weight=0.15`): KL `3.720M` -> `3.848M` (overall increasing).
- A3 run 4 (`910b0b43...`, `attn_v prior_std=0.7`): KL `3.628M` -> `3.562M` (overall decreasing).

Interpretation:
- Rising raw KL does **not** mean posterior is not training. It means posterior moved farther from prior.
- Training optimizes `CE + (effective_kl_weight / num_train_tokens) * KL`.
- Lowering `train.kl_weight` weakens KL pressure and can allow CE-driven posterior drift (higher raw KL).
- KL is the variational inference term `KL(q(w)||p(w))`, not only optional regularization.

Cross-run A3 conclusion:
- Run 2 confirmed that `init_rho=-2.0` unlocks attention-`v_proj` posterior learning
  (per-block sigma means increased into ~`0.10–0.14`), but uncertainty separation did not improve.
- Run 3 (`kl_weight=0.15`) did not improve MI separation or ID perplexity and introduced an
  overall increasing KL trajectory.
- Run 4 (`attn_v prior_std=0.7`) reduced KL and narrowed sigma tails, but further regressed
  MI separation (batch `1.26x`, qualitative `1.35x`) and ID perplexity (`62.33`).
- A3 remains below A2:
  - A2 best: batch `1.42x`, qualitative `1.70x`, test_id_ppl `53.53`
  - A2 repro: batch `1.36x`, qualitative `1.55x`, test_id_ppl `55.08`
  - A3 run 1: batch `1.29x`, qualitative `1.39x`, test_id_ppl `59.20`
  - A3 run 2: batch `1.27x`, qualitative `1.41x`, test_id_ppl `61.97`
  - A3 run 3: batch `1.29x`, qualitative `1.41x`, test_id_ppl `61.23`
  - A3 run 4: batch `1.26x`, qualitative `1.35x`, test_id_ppl `62.33`
- Working hypothesis: current Bayesian `V` adds broad uncertainty (raises MI on both ID and OOD)
  but does not improve discriminative epistemic separation.

A3 closure decision:
1. A3 is closed (negative result vs A2).
2. Final archived A3 hyperparameters: `model.bayes_ffn.init_rho=-2.0`, `model.bayes_ffn.prior_std=1.0`, `model.bayes_attn_v.init_rho=-2.0`, `model.bayes_attn_v.prior_std=1.0`, `train.kl_weight=0.2`.
3. Reference run for archived config: `1848a95f9d494a6baca5e41a5cd65829`.
4. No further A3 reruns are planned.
5. Next milestone: B1.
---

## B1 Implementation Notes

### Architecture
- **Post-hoc Laplace** — no training loop changes. Fits diagonal Fisher curvature from a deterministic checkpoint, samples from Gaussian posterior for MI evaluation.
- New module: `minigpt/laplace.py` with `LaplaceState` dataclass, `select_params`, `fit_laplace`, `sample_laplace_params`, `apply_sampled_params` (context manager), `compute_laplace_uncertainty`, `score_sequence_laplace`, save/load utilities.
- Core MC metrics loop (`mc_metrics_single` in `uncertainty.py`) is shared between A-series and Laplace — no duplication.
- `compute_laplace_uncertainty` mirrors `compute_uncertainty_metrics` protocol (MI, entropy, flip rate) but uses Laplace sampling.

### Config surface
- New YAML section `laplace:` with `checkpoint`, `selection_mode`, `damping`, `sample_scale`, `n_curvature_batches`.
- Config: `configs/b1_laplace_agnews.yaml`.

### Scripts
- `experiments/b1_laplace_baseline.py` — full pipeline: train (or load checkpoint) → fit Laplace → evaluate MI → qualitative panel → MLflow logging. Uses shared infrastructure from `experiment_setup.py`, `eval_utils.py`, `mlflow_utils.py`. Supports `--skip-train`.
### Key implementation details
- Curvature = empirical Fisher (average squared gradients over mini-batches). Diagonal only (no K-FAC).
- Posterior variance = `1 / (curvature + damping)`. Damping mandatory for numerical stability.
- Sampling: `phi ~ N(phi_hat, diag(sample_scale^2 / (curvature + damping)))`. CPU generator for reproducibility, then `.to(device)`.
- `apply_sampled_params` context manager: temporarily patches model params, restores on exit (including on exception).
- Selection modes: `ffn` (MLP fc/proj weights), `head` (lm_head weight), `all` (all weight matrices excluding embeddings/layernorm).

### Validation
- 24 unit tests in `tests/test_laplace.py`, all green.
- Smoke test completed on GPU (200 training steps): end-to-end pipeline runs without errors.

### B1 R1 — Full training run (100K steps, damping=1.0, sample_scale=1.0)
- Command: `python experiments/b1_laplace_baseline.py --config configs/b1_laplace_agnews.yaml`
- MLflow run: `4da8958d2ad84c5a9d58547289f8bf29`
- Training: 100K steps, best val loss 3.6656 (step 94K), training time 5955s
- Test ID/OOD perplexity: **40.99 / 476.42** (strong separation — model trained well)
- Laplace: 8 FFN params selected (2,097,152 elements), fit time 8s
- Curvature: mean=0.000001, std=0.000048, min=0.000000, max=0.050442
- MI (ID/OOD): 1.7077 / 1.7079 → **MI ratio 1.00x** (no OOD signal)
- Qualitative MI ratio: **0.99x**
- Flip rate: 0.89, predictive entropy: 9.13 (near-uniform over vocab)

**Root cause: two bugs identified.**

1. **Fisher computation bug** — `fit_laplace()` computes `(mean_gradient)²` instead of `mean(gradient²)`. Loss is `reduction='mean'` over batch×seq (32×256=8192 tokens). At convergence, per-token gradients cancel in the mean → (mean_grad)² ≈ 0. Correct diagonal Fisher requires per-sample squared gradients. See `specs/b1-laplace-analysis.md`.

2. **MLflow duplicate params bug** — `log_common_mlflow` flattens entire config (including `laplace.*` keys), then B1 script logged `laplace.damping`/`sample_scale`/etc. explicitly → MLflow threw on duplicate keys, silently losing all downstream logging (MI, perplexity, qualitative). Fixed: removed duplicate explicit params, kept only computed values (`laplace.num_selected_params`, `skip_train`, `laplace_state_path`).

**Why MI ratio = 1.00x:** With curvature ≈ 0, posterior variance = `1/(0 + damping)` = 1.0 for ALL params. Effective σ ≈ 1.0 — catastrophic noise (10-100x weight magnitude). Every MC sample outputs near-random predictions. Noise is equally random for ID and OOD → MI identical.

### B1 Approach A — sample_scale sweep (identity-curvature regime) — NEGATIVE

Since curvature ≈ 0, posterior is effectively `N(θ_MAP, (sample_scale²/damping) × I)`.
Four runs with damping=1.0, varying sample_scale:

| sample_scale | eff. σ | MI ID | MI OOD | Ratio | Flip Rate | MLflow run |
|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 1.881 | 1.879 | 1.00x | 0.900 | `497423b8` |
| 0.10 | 0.10 | 1.795 | 1.795 | 1.00x | 0.904 | `c9966171` |
| 0.20 | 0.20 | 1.724 | 1.724 | 1.00x | 0.897 | `44efc13d` |
| 0.30 | 0.30 | 1.710 | 1.711 | 1.00x | 0.894 | `c76e81cb` |

All perplexity identical (40.05 / 488.33) — same MAP model, expected.

**Conclusion: uniform (identity-curvature) noise fundamentally cannot differentiate ID from OOD in language models, at any scale.** Flip rate ≈ 0.90 even at σ=0.05 — 2M uniformly perturbed params accumulate to catastrophic output changes. MI is high but identical for ID and OOD because the noise has no notion of which parameters matter for what. ICLA (WACV 2025) works for image classification (10–200 classes) but fails for language modeling (50K vocab, high-dimensional output space).

### B1 Approach B — per-sample Fisher computation — NEGATIVE
Fix `fit_laplace()` to process sequences one-at-a-time instead of batch-averaged gradients. This gives correct diagonal Fisher: `mean(gradient²)` instead of `(mean_gradient)²`.

- Command: `python experiments/b1_laplace_baseline.py --config configs/b1_laplace_agnews.yaml --skip-train`
- MLflow run: `8774fa910ffc455fa7c734f933a44761`
- Curvature: mean=0.000018, std=0.000139, min=0.000000, max=0.086192 (non-zero — fix worked)
- Fit time: 20.6s (vs 8s before — per-sample loop is 2.5x slower)
- Test ID/OOD perplexity: 41.06 / 481.58
- MI (ID/OOD): 1.7072 / 1.7078 → **MI ratio 1.00x** (still no separation)
- Qualitative MI ratio: **0.99x**
- Flip rate: 0.89 (unchanged)

**Why still 1.00x:** Curvature is non-zero but still tiny (mean ~10⁻⁵) vs damping=1.0. Posterior variance = `1/(0.000018 + 1.0) ≈ 1.0` — damping dominates completely. Even the max-curvature param: `1/(0.086 + 1.0) = 0.92` vs `1/(0 + 1.0) = 1.0` — only 8% difference. Lowering damping to match curvature scale (e.g., 0.00001) would give variance ~55,000 — catastrophic noise that destroys all predictions equally. The fundamental issue is that diagonal Fisher curvature at convergence for a well-trained LM is too flat to provide informative posterior structure.

### B1 Final Conclusion

**Diagonal post-hoc Laplace on FFN params of a converged language model does not produce OOD-discriminative uncertainty.** Two independent approaches confirm this:

1. **Approach A (identity curvature):** Uniform noise at any scale gives MI ratio 1.00x. ICLA works for image classification (10-200 classes) but fails for LM (50K vocab).
2. **Approach B (correct per-sample Fisher):** Real curvature is too small (~10⁻⁵) to overcome any reasonable damping. The loss landscape is flat in FFN weight directions at convergence.

This is a scientifically valuable negative result: it demonstrates that post-hoc Bayesianization via diagonal Laplace cannot substitute for train-time variational inference (A2: 1.43x) for epistemic uncertainty in language models. The curvature structure from a converged model lacks the informative variance differentiation that variational posteriors learn during training.

### B3 Implementation Notes

- **Architecture:** Implemented `DeterministicLoRALinear` in `minigpt/lora.py` and updated `inject_lora` to support both BLoB (Variational) and Deterministic adapters via a `bayesian` flag.
- **TFB (Training-Free Bayesianization):**
  - Implemented in `minigpt/tfb.py`.
  - Structures posterior variance in LoRA $A$ matrix based on SVD of deterministic $B$ matrix: $\Omega_{ij} = \sigma_q / s_i$.
  - Binary search for optimal noise scale $\sigma_q$ on fixed anchor batches within performance tolerance $\epsilon$.
  - Max iteration safeguard (`max_iterations=100`) prevents infinite loops.
- **Laplace-LoRA:**
  - Extended `minigpt/laplace.py` with `"lora"` selection mode targeting `lora_A` parameters.
  - Fits diagonal Fisher curvature specifically on adapter weights.
- **Unified Pipeline:** Created `experiments/b3_post_hoc_lora.py` supporting three phases: `train` (deterministic baseline), `tfb` (post-hoc fitting), and `laplace` (post-hoc fitting). CLI: `--phase`, `--lora-checkpoint`, `--base-checkpoint`, `--no-mlflow`.

### B3 Code Review (2026-03-13) — ALL FIXED

Review doc: `specs/b3-implementation-review.md`

Critical bugs found and fixed:
1. **Base checkpoint loaded after LoRA injection** — FFN weights silently dropped (`fc.linear.weight` vs `fc.base_linear.weight` mismatch). Fix: load base BEFORE `inject_lora()`, matching B2's pattern.
2. **`torch.device("auto")` crash** — main config has `device: auto` which is invalid. Fix: use shared `resolve_device()`.
3. **MLflow URI/experiment never set** — runs went to default `mlruns/` instead of sqlite. Fix: use shared `mlflow_context()`.

Medium issues found and fixed:
4. **TFB binary search used different random batches per step** — now draws fixed anchor batches once and reuses them.
5. **`--no-mlflow` flag ignored** — now threaded through all phase functions via `mlflow_context`.
6. **No max iteration limit on binary search** — added `max_iterations=100`.
7. **2 of 19 spec tests missing** — added `test_tfb_search_respects_tolerance` and `test_tfb_sampling_changes_logits`.
8. **`load_tfb_state`/`load_laplace_state` lacked `map_location`** — added to both.
9. **Seed reuse across batches in MC eval** — fixed in both `compute_tfb_uncertainty` and `compute_laplace_uncertainty` to use `(batch_idx * batch_size + b) * n_samples`.

Verification:
- 19 new unit tests added (103 total).
- Smoke test (100 steps, `--phase full --no-mlflow`) completes end-to-end on CPU.
- All 3 standalone phases (`train`, `tfb`, `laplace`) work with MLflow enabled.

### B3 Phase 1 — Deterministic LoRA Training

- Command: `python experiments/b3_post_hoc_lora.py --phase train --config configs/b3_lora_agnews.yaml`
- MLflow run: `d6c513442f7e45aea034956f90e42745`
- Base checkpoint: `data/checkpoints/b2_pretrain/ckpt_best.pt` (cat 1 World, val ppl=46.5)
- LoRA: rank=16, alpha=32, target=ffn. 163,840 total LoRA params (81,920 in A matrices).
- Training: 10K steps, lr=3e-4, batch_size=32. Training time: ~556s.
- Best val loss: 5.4188 (val ppl=225.6)
- **Test ID ppl: 224.6** / **Test OOD ppl: 531.7**
- Very close to B2 R2 (226.9 / 533.7) — same base, same LoRA capacity, deterministic vs Bayesian training has negligible effect on point-estimate quality.

### B3 Phase 2a — TFB (Training-Free Bayesianization)

- Command: `python experiments/b3_post_hoc_lora.py --phase tfb --config configs/b3_lora_agnews.yaml`
- MLflow run: `0081274d19cd4567abca58300cfb2ac0`
- Config: epsilon=0.1, n_search_samples=10, n_anchor_batches=20
- Anchor loss: 5.3519
- **σ_q = 0.0131** — non-trivial! Binary search found real noise level within ε=0.1 tolerance.
- Fit time: 112s (~2 min)
- **MI (ID/OOD): 0.0917 / 0.1006 → MI ratio 1.10x**
- Flip rate (ID/OOD): 0.215 / 0.221
- Predictive entropy (ID/OOD): 5.151 / 5.349

Interpretation:
- Post-hoc LoRA (TFB) **works** — unlike post-hoc on full weights (B1: 1.00x), adding SVD-structured noise to LoRA A matrices produces OOD-discriminative uncertainty.
- σ_q=0.013 is small but meaningful. The SVD structure ($\Omega_{ij} = \sigma_q / s_i$) concentrates variance on directions with small singular values in B — this is informative structure that uniform noise (B1) lacks.
- MI ratio 1.10x approaches variational LoRA (B2: 1.13x) with zero Bayesian training cost — just a 2-minute binary search on a trained checkpoint.
- Potential room to improve: wider ε or different search range may yield higher σ_q and stronger MI signal.

### B3 Cross-Method Comparison (so far)

| Method | Type | Bayesian params | MI ratio (batch) | Test ID ppl | Cost |
|--------|------|----------------|-----------------|-------------|------|
| A2 | variational, full | 4.2M | **1.43x** | 53.5 | 1x (100K steps) |
| B2 | variational, LoRA | 163K | **1.13x** | 226.9 | 0.1x (10K steps) |
| B3-TFB | post-hoc, LoRA | 82K (post-hoc) | **1.10x** | 224.6 | 0.01x (2 min search) |
| B1 | post-hoc, full | 2.1M (post-hoc) | **1.00x** | 41.0 | 0.001x (8s fit) |

### B3 Phase 2b — Laplace-LoRA (NEGATIVE)

- Command: `python experiments/b3_post_hoc_lora.py --phase laplace --config configs/b3_lora_agnews.yaml`
- MLflow run: `10261b6765c245f488aabf0c1a92b0c1`
- Config: damping=1.0, sample_scale=1.0, selection_mode=lora, n_curvature_batches=20
- Curvature: mean=2.509e-5, std=3.758e-5, max=0.00242
- Fit time: 4.0s
- **MI (ID/OOD): 1.7444 / 1.7440 → MI ratio 1.00x**
- Flip rate (ID/OOD): 0.884 / 0.884
- Predictive entropy (ID/OOD): 8.677 / 8.677

**Same failure mode as B1.** Curvature on LoRA A params is as flat as on full FFN params (B1: mean=1.8e-5, B3-LAP: mean=2.5e-5). With damping=1.0, posterior variance ≈ 1.0 everywhere — catastrophic uniform noise. Flip rate 0.88 confirms near-random MC samples. MI is high but identical for ID/OOD because the noise has no structure.

**Why TFB works and Laplace doesn't:** TFB uses the SVD of the deterministic $B$ matrix to structure variance ($\Omega_{ij} = \sigma_q / s_i$). Directions with large singular values get small variance — this is an *architectural* prior baked into the LoRA structure, not a data-derived curvature estimate. Diagonal Laplace relies on curvature from the loss landscape, which is uniformly flat at convergence for both full weights and LoRA.

### B3 Final Results — Complete 2x2 Matrix

| Method | Type | Bayesian params | MI ratio (batch) | Test ID ppl | Cost |
|--------|------|----------------|-----------------|-------------|------|
| A2 | variational, full | 4.2M | **1.43x** | 53.5 | 1x (100K steps) |
| B2 | variational, LoRA | 163K | **1.13x** | 226.9 | 0.1x (10K steps) |
| B3-TFB | post-hoc, LoRA | 82K (post-hoc) | **1.10x** | 224.6 | 0.01x (2 min search) |
| B3-LAP | post-hoc, LoRA | 82K (post-hoc) | **1.00x** | 224.6 | 0.001x (4s fit) |
| B1 | post-hoc, full | 2.1M (post-hoc) | **1.00x** | 41.0 | 0.001x (8s fit) |

Key findings from the complete matrix:
1. **Variational methods work.** Both full-weight (A2: 1.43x) and LoRA (B2: 1.13x) produce OOD-discriminative uncertainty. Train-time posterior learning is the gold standard.
2. **Diagonal Laplace fails universally.** Both on full weights (B1) and LoRA params (B3-LAP). The curvature at convergence is too flat for informative posteriors — this is a fundamental limitation of diagonal post-hoc Laplace for LM uncertainty.
3. **TFB is the exception.** It works post-hoc (1.10x) because it uses structural information from the LoRA architecture (SVD of B), not curvature from the loss landscape. This is a meaningful contribution: SVD-structured noise > curvature-informed noise for post-hoc LM Bayesianization.
4. **LoRA subspace matters.** Post-hoc fails on full weights (B1: 1.00x) but succeeds on LoRA with the right variance structure (TFB: 1.10x). The low-rank constraint provides enough inductive bias for meaningful posterior estimation.

---

## Research Plan

**Comparative study:** 2x2 matrix — (variational vs post-hoc) x (full weights vs LoRA). See `specs/comparative-bayesian-llm-study.md`.

| | Full weights | LoRA adapters |
|---|---|---|
| **Variational (train-time)** | A-series (done, **1.43x**) | B2 BLoB (done, **1.13x**) |
| **Post-hoc (no training)** | B1 Laplace (done, **1.00x** negative) | B3-TFB (**1.10x**) / B3-LAP (**1.00x** negative) |

Execution order: ~~B1 (finish)~~ → ~~B2 (BLoB LoRA)~~ → ~~B3 (TFB + Laplace-LoRA)~~ → C (scaled replication) → comparison paper.

### Milestone Numbering (aligned 2026-03-10)
- **B2 = BLoB** — variational Bayesian LoRA (train-time). Asymmetric: Bayesianize A matrix, fix B.
- **B3 = TFB / Laplace-LoRA** — post-hoc Bayesianization of trained LoRA params.
- **C = Scaled replication** — re-run full A1–A3, B1–B3 suite on 16L miniGPT. Final comparison paper.

Old B-milestone-roadmap (`specs/B-milestone-roadmap.md`) is superseded by this numbering.

---

## B2 Plan (BLoB-style Bayesian LoRA)

### Design Decision: Category-Split Pretrain → LoRA Fine-Tune

**R1 approach (TinyShakespeare pretrain) failed.** TinyShakespeare (~304K tokens) is 50x too small for 16M params. Catastrophic overfitting: best val ppl=160 at step 800–1000, then diverges to val ppl=1646 by 15K steps. The resulting base was too poor for LoRA to bridge Shakespeare→News.

**R2 approach: AG News category-split.**

| Phase | Categories | Purpose |
|-------|-----------|---------|
| Pretrain (deterministic) | 1 = World | Base LM learns news language in one domain |
| LoRA fine-tune (BLoB) | 2 = Sports | Adapter specializes for a related but different domain |
| OOD eval | 3+4 = Business, Sci/Tech | Truly unseen by both base and adapter |

**Why not pretrain on ALL AG News categories:** If the base already knows Business/Sci-Tech, the base output dominates LoRA's small delta at OOD eval time. MC LoRA samples create tiny perturbations on a confident base → MI signal suppressed. We'd measure uncertainty in an irrelevant residual, not genuine epistemic uncertainty.

**Why category-split works:**
- OOD is truly unseen by both base AND adapter → clean uncertainty attribution.
- LoRA does genuine work (World→Sports adaptation) → meaningful gradients → posteriors properly trained.
- Mirrors real-world pretrain→finetune: pretrain domain ≠ finetune domain ≠ OOD domain.

Full rationale and proof: `specs/b2-r2-revised-plan.md`.

### Flipout Decision
**Not used for B2.** Current reparameterization trick works fine for 4L/batch_size=32 — A2 trained successfully, posteriors learned, KL healthy, MI separation reproduced across seeds. Flipout is reserved for C milestone (16L/batch_size=8) where gradient variance from shared weight samples becomes a real bottleneck.

## B2 Implementation Notes

### Implementation status
- Added `minigpt/lora.py` with `LoRAConfig`, `BLoBLoRALinear`, and `inject_lora()`.
- Added configs: `configs/b2_pretrain_agnews.yaml` (R2 pretrain) and `configs/b2_blob_agnews.yaml` (R2 finetune). Old `configs/b2_pretrain_shakespeare.yaml` is obsolete (R1).
- Added experiment entrypoint: `experiments/b2_blob_lora.py`.
- Added tests: `tests/test_lora.py`.
- Extended integration points:
  - `minigpt/layers.py` context managers now work with LoRA layers via capability checks (`freeze_sample`, `_use_mean`).
  - `sigma_summary()` now includes LoRA `G^2` values.
  - `minigpt/uncertainty.py::_has_bayesian_body()` now detects `BLoBLoRALinear`.
  - `minigpt/config.py` now builds `LoRAConfig`.
  - `minigpt/train.py::_configure_optimizer()` excludes `_g` params from weight decay, like `_rho`.

Validation:
- `uv run pytest tests/ -q` → `84 passed`

### B2 code review findings (2026-03-10) — ALL FIXED (2026-03-11)
Review doc: `specs/b2-blob-lora-review.md`

1. **~~Invalid LoRA config can silently freeze the entire model.~~** FIXED.
   - `inject_lora()` now raises `ValueError` for unsupported `lora.target` values.
   - `validate_config()` now validates LoRA section when present: `target` in allowed set, `rank > 0` (int), `alpha > 0`, `prior_std > 0`, `init_g > 0`.

2. **~~B2 phase-2 CLI can silently run with the wrong config if `--config` is omitted.~~** FIXED.
   - `b2_blob_lora.py` now requires `--config` for `--phase finetune` and `--phase full`.
   - Missing `--config` raises a clear `ValueError` with usage hint.

3. **~~Generated sample is mislabeled as "mean weights".~~** FIXED.
   - Sample generation is now wrapped in `use_mean_weights(model)` context manager.
   - The "mean weights" label is now accurate.

None of these fixes affect B2 R1 metrics (configs were valid, `--config` was passed, generation label is cosmetic).

### MLflow duplicate-param bug discovered in B2
First attempted finetune command:

```bash
python experiments/b2_blob_lora.py --phase finetune --config configs/b2_blob_agnews.yaml
```

Observed failure:
- MLflow run: `a42eac1756f847aca76cf9a506e3d89d`
- Crash happened before training started.
- Error:
  - `Changing param values is not allowed`
  - offending key: `lora.base_checkpoint`
  - old value: `data/checkpoints/b2_pretrain/ckpt_best.pt`
  - new value: `data\checkpoints\b2_pretrain\ckpt_best.pt`

Root cause:
- `log_common_mlflow()` flattens and logs the full config, including `lora.base_checkpoint`.
- `experiments/b2_blob_lora.py` then logs `lora.rank`, `lora.alpha`, `lora.prior_std`, `lora.init_g`, and `lora.base_checkpoint` again explicitly.
- On Windows, the explicit `str(Path(...))` form uses backslashes, so MLflow sees a second value for the same key and throws.

Specific redundancy:
- First log source: `log_common_mlflow()` via flattened config.
- Second log source: explicit `mlflow.log_params({...})` in `experiments/b2_blob_lora.py`.
- The second log is redundant for all `lora.*` keys because they are already included in the flattened config.

### B2 R1 — First completed finetune run (negative / inconclusive)
- Command: `python experiments/b2_blob_lora.py --phase finetune --config configs/b2_blob_agnews.yaml`
- MLflow run: `e1f60bfcd2bc41aaa33a6839e453fedf`
- Status: `FINISHED`
- Dataset: AG News
  - ID categories: World + Sports
  - OOD categories: Business + Sci/Tech
  - Train/val/test_id/test_ood tokens: `2,701,999 / 337,750 / 337,750 / 3,498,051`
- Base checkpoint: `data/checkpoints/b2_pretrain/ckpt_best.pt`
- Base model params before injection: `16,090,880`
- LoRA params after injection:
  - total LoRA params: `122,880`
  - Bayesian LoRA params: `81,920`
- Training:
  - steps: `10,000`
  - best val loss (ELBO criterion): `8.1399` at step `8,000`
  - training time: `641.3s`
- Final eval:
  - val perplexity: `3388.95`
  - test ID perplexity: `3487.13`
  - test OOD perplexity: `4360.15`
  - MI (ID/OOD): `0.000576 / 0.000600`
  - batch MI ratio: `1.04x`
  - flip rate (ID/OOD): `0.0211 / 0.0210`
  - final KL: `137,856`
- Sigma stats:
  - mean=`0.004775`, std=`0.002177`, min=`0.000527`, max=`0.019830`
  - median=`0.004324`, p5=`0.001986`, p25=`0.003129`, p75=`0.006137`, p95=`0.008738`

Metric history:
- `train_loss`: `11.2686` at step 1 → `8.1493` at step 10,000
- `val_loss`: `11.2481` at step 1 → `8.1378` at step 10,000
- `val_loss` plateaued around `8.13–8.15` after the early phase; best checkpoint was step `8,000`
- `kl_loss`: `172,664.875` at step 1 → `137,310.875` at step 10,000 (overall decreasing)
- `effective_kl_weight`: linearly annealed to `0.2`, then stayed there
- `lr`: cosine-decayed to `1.0e-5` by step `10,000`

Qualitative eval:
- Average MI — ID: `0.0004`, OOD: `0.0006`, ratio: `1.32x`
- Absolute MI values remained tiny.
- Generated text and qualitative continuations were gibberish / non-news-like, indicating the model did not adapt to AG News well enough for the uncertainty signal to be scientifically meaningful.

Interpretation:
- The Bayesian LoRA machinery appears to be functioning:
  - KL decreases over training
  - sigma values move above the tiny init scale
  - MI is non-zero
- But the mean model quality is catastrophically poor:
  - perplexity remains in the `3.4k–4.4k` range
  - samples are incoherent
- Therefore B2 R1 is **not** evidence that Bayesian LoRA fails in principle.
- More likely, the current setup is too weak:
  - TinyShakespeare pretrain is too mismatched to AG News
  - rank-8 FFN-only LoRA on a frozen 4L miniGPT is insufficient to recover a competent news model
- As a result, this run is not yet a fair scientific comparison to A2.

B2 R1 conclusion:
1. B2 implementation is landed and test-covered (84 tests).
2. Code review issues (config validation, CLI footgun, sample label) — all fixed 2026-03-11.
3. MLflow duplicate-param bug — fixed earlier (commented out redundant explicit logging).
4. B2 R1 (`e1f60bfc`) is inconclusive: TinyShakespeare pretrain was structurally inadequate (304K tokens for 16M params). Not evidence that BLoB LoRA fails.
5. B2 R2 plan: category-split pretrain (cat 1 World → cat 2 Sports LoRA → cats 3+4 OOD). Revised hyperparams. Full spec: `specs/b2-r2-revised-plan.md`.

### B2 Pretrain Runs (TinyShakespeare — ABANDONED)

| Run | Steps | Train ppl | Val ppl | Status | MLflow |
|-----|-------|-----------|---------|--------|--------|
| `9ae3327a` | 15,000 | **1.37** | **1,646** | Catastrophic overfit | Prior run |
| `1fae7c40` | 1,000 | 65.7 | 161.0 | Best achievable (early stop) | Used for R1 |

TinyShakespeare pretrain is abandoned. 304K tokens for 16M params → structural overfitting ceiling at val ppl ~160. Not viable as a base for LoRA adaptation.

### B2 R2 — Revised Hyperparameters

**Pretrain (Phase 1):** AG News category 1 (World), ~1.08M train tokens.
- Config: `configs/b2_pretrain_agnews.yaml`
- Steps: 15,000. Same architecture/optimizer as R1 pretrain (lr=3e-4, dropout=0.2, weight_decay=0.1).
- Expected val ppl: 60–100.

**Finetune (Phase 2):** AG News category 2 (Sports). OOD: categories 3+4 (Business + Sci/Tech).

| Parameter | R1 | R2 | Rationale |
|-----------|----|----|-----------|
| `data.id_categories` | [1, 2] | **[2]** | Sports only (LoRA trains on this) |
| `lora.rank` | 8 | **16** | More capacity for cross-category adaptation |
| `lora.alpha` | 16.0 | **32.0** | Keep alpha/rank=2 |
| `train.lr` | 1e-4 | **3e-4** | Standard LoRA LR; matches A-series and BLoB paper |
| `lora.init_g` | 0.05 | **0.1** | Initial σ=0.01 (4x wider → more room for posterior differentiation) |
| `lora.prior_std` | 0.2 | 0.2 | Keep |
| `train.kl_weight` | 0.2 | 0.2 | Keep |
| `train.steps` | 10,000 | 10,000 | Keep |

### B2 R2 Pretrain — AG News Cat 1 (World)

- Command: `python experiments/b2_blob_lora.py --phase pretrain --pretrain-config configs/b2_pretrain_agnews.yaml`
- MLflow run: `96c2a7862d0b4025b0fa3ea30b0deb8a`
- Config: `configs/b2_pretrain_agnews.yaml`
- Dataset: AG News category 1 (World), ~1.08M train tokens
- Steps: 15,000
- Best val loss: `3.8385` at step `15,000` (no overfitting — kept improving throughout)
- Val ppl: **46.46** (better than expected 60–100 range)
- Test ID ppl: **44.84**
- Train ppl: 11.81 (moderate gap = healthy regularization, not catastrophic overfit)
- Training time: 942s

Comparison to prior pretrains:

| Corpus | Train tokens | Ratio | Val ppl | Outcome |
|--------|-------------|-------|---------|---------|
| TinyShakespeare | 243K | 1:66 | 161 | Structural overfitting |
| AG News cat 1 (World) | ~1.08M | 1:15 | **46.5** | Functional base LM |
| AG News cats 1+2 (A0) | ~2.16M | 1:7 | 49.1 | A0 baseline |

Cat-1-only pretrain achieved comparable quality to the full A0 baseline (46.5 vs 49.1), confirming AG News provides sufficient signal even from a single category.

### B2 R2 Finetune — BLoB LoRA on Cat 2 (Sports)

- Command: `python experiments/b2_blob_lora.py --phase finetune --config configs/b2_blob_agnews.yaml`
- MLflow run: `8ddcdf26503043e794de1a7bbbac6e5a`
- Status: `FINISHED`
- Dataset: AG News
  - ID categories: Sports [2]
  - OOD categories: Business + Sci/Tech [3, 4]
- Base checkpoint: `data/checkpoints/b2_pretrain/ckpt_best.pt` (val ppl=46.5)
- LoRA params after injection:
  - total LoRA params: `245,760`
  - Bayesian LoRA params: `163,840`
- Training:
  - steps: `10,000`
  - best val loss (ELBO criterion): `5.4515` at step `6,500`
  - training time: `637s`
- Final eval:
  - test ID perplexity: `226.85`
  - test OOD perplexity: `533.69`
  - MI (ID/OOD): `0.00601 / 0.00679`
  - **batch MI ratio: 1.13x**
  - flip rate (ID/OOD): `0.0605 / 0.0624`
  - final KL: `226,988`
- Sigma stats:
  - mean=`0.0083`, std=`0.0027`, min=`0.0004`, max=`0.0316`
  - median=`0.0081`, p5=`0.0044`, p25=`0.0064`, p75=`0.0100`, p95=`0.0132`

Qualitative MI (prompt-token scoring, 5 prompts per category):
- Sports (ID): 0.0064
- Business (OOD): 0.0069
- Sci/Tech (OOD): 0.0063
- Overall: ID=0.0064, OOD=0.0065, **qualitative ratio=1.02x**

Generated text (mean weights): semi-coherent news-style fragments with sports vocabulary leaking into other topics. Better than R1 gibberish but still low quality (ppl=227).

### B2 R2 Cross-Run Comparison

| Metric | B2 R1 (TinyShakespeare) | **B2 R2 (AG News split)** | A2 best |
|--------|------------------------|--------------------------|---------|
| Base val ppl | 161 | **46.5** | 49.1 (same model) |
| Test ID ppl | 3,487 | **226.9** | 53.5 |
| Test OOD ppl | 4,360 | **533.7** | 595 |
| MI ratio (batch) | 1.04x | **1.13x** | 1.43x |
| MI ratio (qual) | 1.32x | **1.02x** | 1.70x |
| Sigma mean | 0.0048 | **0.0083** | 0.147 |
| Bayesian params | 81,920 | **163,840** | 4,200,000 |
| KL (final) | 137K | **227K** | 3.2M |

### B2 Closure Decision

**B2 is DONE — weak positive result.**

1. **BLoB LoRA produces OOD-discriminative uncertainty** at the batch level (MI ratio 1.13x), clearing the >1.1x target. This is directional evidence that variational LoRA can detect OOD via epistemic uncertainty.

2. **The signal is much weaker than full-weight variational inference** (A2: 1.43x batch, 1.70x qual). This is expected: rank-16 LoRA has 163K Bayesian params vs A2's 4.2M — a 25x reduction in posterior expressiveness.

3. **Qualitative MI ratio (1.02x) is flat.** The per-prompt MI values are too small (0.006) for reliable separation at 5 prompts per category. Batch-level aggregation over thousands of tokens is needed to detect the signal.

4. **Posteriors are constrained.** Sigma mean=0.0083 (close to init σ=0.01), range [0.0004, 0.032]. Posteriors moved but didn't differentiate as dramatically as A2's [0.036, 0.966]. The low-rank subspace constrains how much posterior variance the model can express.

5. **ID ppl=227 is higher than expected** (target was 80–200). World→Sports is a genuine domain shift. The LoRA adapted but not fully — there's room for improvement at larger scale.

6. **Key takeaway for the paper:** Variational LoRA (BLoB) works but produces weaker OOD signals than full-weight variational inference at 4L miniGPT scale. The question of whether this gap narrows or widens at 16L scale (C milestone) is the central contribution of the comparison paper.

7. **No further B2 runs planned.** The current result is sufficient for the 2x2 comparison. Scaling experiments (C milestone) will test whether BLoB LoRA's MI ratio improves with a larger model and more data.

---

## C Milestone (Scaled Replication)

Full spec: `specs/c-milestone-spec.md`.

### Concept
Re-run the 2×2 matrix on a scaled-up miniGPT to validate whether findings transfer across model size. Addresses the reviewer concern: "does this generalize beyond a toy model?"

### Architecture (CONFIRMED)
- **16L / 8H / 512d** (~76.3M params)
- Same `model.py`, only config changes. Weight tying preserved.

### GPU Profiling (2026-03-13)

Script: `scripts/profile_c_gpu.py`. RTX 4070 (12GB), AMP, seq=256.

| Variant | Params | Best batch | Peak VRAM | VRAM% | Tok/sec | Accum |
|---------|--------|-----------|-----------|-------|---------|-------|
| Deterministic | 76.3M | 16 | 5,653 MB | 46% | 35,584 | 2 |
| Bayesian FFN | 109.9M | 16 | 6,422 MB | 52% | 28,420 | 2 |
| Bayesian FFN+AttnV | 114.1M | 16 | 6,518 MB | 53% | 26,468 | 2 |
| BLoB LoRA r=16 | 78.3M (2.0M train) | 32 | 9,021 MB | 73% | 41,869 | 1 |
| Det LoRA r=16 | 77.6M (1.3M train) | 32 | 9,006 MB | 73% | 46,974 | 1 |

**Flipout NOT needed** — batch_size=16 fits all full-weight variants with ~50% headroom. Standard reparameterization trick is sufficient. Gradient checkpointing also not needed.

### Dataset: The Pile (domain-split)
AG News (~5M tokens) is structurally inadequate for 76M params. Using The Pile (uncopyrighted) via `ArmelR/the-pile-splitted` on HuggingFace.

- **ID train:** Wikipedia + StackExchange (~200M tokens total, subsampled)
- **OOD eval:** ArXiv (scientific), FreeLaw (legal), PubMed (biomedical)
- **LoRA fine-tune:** HackerNews (tech-adjacent, different style from ID)
- Token/param ratio: 200M / 76M = 2.6 (vs 0.15 at 4L — much healthier)

### Sub-Milestones
- **C0:** Deterministic baseline on Pile ID
- **C1:** Variational full-weight (A2-equiv), batch=16, accum=2
- **C2:** Post-hoc Laplace on C0 checkpoint (expected negative)
- **C3:** BLoB LoRA, batch=32, no accum
- **C4:** Post-hoc LoRA: TFB + Laplace-LoRA

### Pipeline — Agentic Experiment Optimization (Auto-Research)
`experiments/c_pipeline.py` — **autonomous HP optimization pipeline** (the "auto-research" layer). Run AFTER each sub-milestone's method code is implemented and tested via the usual BDD→TDD→Code process. The pipeline runs the implemented code, sends MLflow results to an LLM agent (Claude/Codex) that reasons about failures and proposes HP adjustments, then re-runs. **Each milestone is a separate invocation** — no batch mode. Workflow per sub-milestone: `BDD → TDD → Code (all green) → c_pipeline.py --milestone cN (auto-research)`. CLI: `--milestone {c0|c1|c2|c3|c4}`, `--resume <milestone>`, `--compare`, `--agent-provider`, `--no-agent`. State in `.pipeline-state/`. Max 4 runs/milestone, 12h GPU budget per milestone.

### Implementation Progress (2026-03-13)

**Phase 0 (Preparation) — DONE:**
- GPU profiling script: `scripts/profile_c_gpu.py`
- C milestone spec: `specs/c-milestone-spec.md`
- Pile data loader BDD: `specs/pile-data-loader-spec.md`
- Pile data loader TDD: `tests/test_pile_data.py` (31 tests, frozen)
- Codex handoff: `specs/pile-data-loader-tdd-handoff.md`

**Pile data loader: DONE.** `load_pile_data()` in `minigpt/data.py`, Pile validation in `minigpt/config.py`, dispatcher updated. 31/31 tests green, 134/134 full suite. New dep: `datasets` (HuggingFace, lazy-imported on cache miss only).

Pipeline orchestrator BDD: `specs/c-pipeline-spec.md`.

**Next: TDD for the C pipeline orchestrator** — write unit tests in `tests/test_c_pipeline.py` against `specs/c-pipeline-spec.md`. Awaiting user sign-off on the BDD spec before proceeding.

### Paper Structure
Table 1: 4L results (existing). Table 2: 16L results (C milestone). Analysis: which methods scale? Which findings transfer? The 2×2 matrix at two scales gives a clean, publishable comparison.

---

## Future Work (Parked)
Non-Bayesian improvements:
- RoPE, SwiGLU, KV-cache
- Mixed precision: **DONE** (AMP auto-enabled)

## References
- **BLoB** (NeurIPS 2024) — Bayesian LoRA by backprop. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- **TFB** (NeurIPS 2025) — Training-Free Bayesianization for LoRA. [arXiv:2412.05723](https://arxiv.org/abs/2412.05723)
- **Laplace-LoRA** (2023) — Laplace approximation on LoRA params. [arXiv:2308.13111](https://arxiv.org/abs/2308.13111)
- **ScalaBL** (2025) — Bayesian inference in low-dim subspace. [arXiv:2506.21408](https://arxiv.org/abs/2506.21408)
- **ICLA** (WACV 2025) — Identity Curvature Laplace for OOD. [arXiv:2312.10464](https://arxiv.org/abs/2312.10464)
- **Laplace Redux** (NeurIPS 2021) — Effortless Bayesian Deep Learning. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)
