# Repository Guidelines

## Goal
**Estimate epistemic uncertainty in LLMs via Bayesian inference over weights.**

The core idea: replace point-estimate weights with learned posterior distributions, then measure how much the model's predictions disagree across weight samples (mutual information). High MI = "the model knows what it doesn't know" — it's uncertain about out-of-distribution inputs at the weight level, not just the token level.

**Current approach:** Controlled comparative study on miniGPT (AG News). Four Bayesian methods in a 2x2 matrix: (variational vs post-hoc) x (full weights vs LoRA). A-series (variational full weights) done. B-series: B1 Laplace (post-hoc full weights), B2 BLoB (variational LoRA), B3 TFB/Laplace-LoRA (post-hoc LoRA). Active spec: `specs/comparative-bayesian-llm-study.md`.

## Project Structure
`docs/` — PDF papers (theory baseline). `specs/` — planning docs (active: `specs/comparative-bayesian-llm-study.md`). B1 tech spec: `specs/b1-laplace-tech-spec-mar2026.md`. B1 analysis: `specs/b1-laplace-analysis.md`. B2 spec: `specs/b2-blob-lora-spec.md`. B2 R2 revised plan: `specs/b2-r2-revised-plan.md`. B2 code review: `specs/b2-blob-lora-review.md`. B3 spec: `specs/b3-post-hoc-lora-spec.md`. C milestone: `specs/c-milestone-spec.md`. C pipeline BDD: `specs/c-pipeline-spec.md`. C pipeline TDD handoff: `specs/c-pipeline-tdd-handoff.md`. C pipeline review: `specs/c-pipeline-implementation-review.md`.

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
configs/          # YAML config files per experiment (a0–b3 AG News, c0 Pile)
experiments/      # Runnable scripts (a0_baseline, a1–a3 Bayesian, b1_laplace_baseline, b2_blob_lora, b3_post_hoc_lora)
  experiment_setup.py  # Shared setup: CLI parsing, config, data, model, device
  eval_utils.py        # Shared eval: perplexity suite, MI suite, qualitative eval
  mlflow_utils.py      # Shared MLflow: context, logging helpers
  runner.py            # A-series orchestrator — A1/A2/A3 are thin wrappers
  pipeline_runner.py   # Generic pipeline engine: PipelineRunnerBase, MilestonePolicy, RuntimeHooks
  c_milestones.py      # C-specific: templates, gates, knobs, comparison report
  c_pipeline.py        # C pipeline CLI: wires hooks+policy, providers, entry point
  agent_briefing.md    # HP tuning playbook injected into agent prompts (no file reading needed)
scripts/          # Utilities (dump_mlflow_run, compare_runs, profile_gpu, eval_checkpoint, fit_laplace)
tests/            # pytest (191 tests: 134 core + 57 pipeline)
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
uv run pytest tests/ -v                          # 134 unit tests (+ 57 pipeline tests pending implementation)
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

## Tests (134 passing + 57 pipeline pending)
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
  test_pile_data.py      # Pile data loader: split shapes (5), domain content (3),
                         # config validation (6), tokenization (3), caching (2),
                         # edge cases (4), dispatcher integration (3), determinism (2),
                         # multiprocessing (3) — 31 tests
  test_c_pipeline.py     # C pipeline orchestrator (57 tests, pending implementation):
                         # TestCli (5), TestConfigGeneration (9),
                         # TestPureHelpers (8), TestRunnerStateAndLoop (22),
                         # TestPolicies (13)
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
- **C0:** Deterministic baseline on Pile ID — **DONE** (PPL 14.3)
- **C1:** Variational full-weight Bayesian FFN (A2-equiv), batch=16, accum=2 — **DONE** (MI ratio 1.32x)
- **C2:** Post-hoc Laplace on C0 checkpoint (B1-equiv). Depends on C0. Expected negative.
- **C3:** BLoB LoRA (B2-equiv), batch=32, no accum. Phase 1 pretrain + Phase 2 BLoB fine-tune. Independent.
- **C4-TFB:** Post-hoc TFB on C3 checkpoint (B3-TFB-equiv). Depends on C3. Expected positive (~1.10x).
- **C4-LAP:** Post-hoc Laplace-LoRA on C3 checkpoint (B3-LAP-equiv). Depends on C3. Expected negative.

### Pipeline — Agentic Experiment Optimization (Auto-Research)
`experiments/c_pipeline.py` — **autonomous HP optimization pipeline** (the "auto-research" layer). Run AFTER each sub-milestone's method code is implemented and tested via the usual BDD→TDD→Code process. The pipeline is a state machine (CONFIGURE → RUN → ANALYZE → DECIDE) that runs the implemented code, checks success gates, and on failure invokes an **agent via the Provider pattern** (Claude Code / Codex as subprocess — NOT API calls). The agent reads the repo, AGENTS.md, MLflow results, and proposes structured HP adjustments. Uses the same Provider adapter pattern as `vla-game-agent/pipeline`: `ClaudeProvider` → `claude -p`, `CodexProvider` → `codex exec`. **Each milestone is a separate invocation** — no batch mode. Workflow per sub-milestone: `BDD → TDD → Code (all green) → c_pipeline.py --milestone cN (auto-research)`. CLI: `--milestone {c0|c1|c2|c3|c4}`, `--resume <milestone>`, `--compare`, `--provider {claude|codex}`, `--no-agent`, `--dry-run`, `--budget`, `--state-dir`, `--no-mlflow`, `--provider-model`. State in `.pipeline-state/`. Max 4 runs/milestone, 12h GPU budget per milestone. Full spec: `specs/c-pipeline-spec.md`. TDD handoff: `specs/c-pipeline-tdd-handoff.md`.

### Implementation Progress (2026-03-16)

**Phase 0 (Preparation) — DONE:**
- GPU profiling script: `scripts/profile_c_gpu.py`
- C milestone spec: `specs/c-milestone-spec.md`
- Pile data loader BDD: `specs/pile-data-loader-spec.md`
- Pile data loader TDD: `tests/test_pile_data.py` (31 tests, frozen)
- Codex handoff: `specs/pile-data-loader-tdd-handoff.md`

**Pile data loader: DONE.** `load_pile_data()` in `minigpt/data.py`, Pile validation in `minigpt/config.py`, dispatcher updated. 31/31 tests green, 134/134 full suite. Dep: `datasets` (HuggingFace) added to `pyproject.toml`.

**Pipeline orchestrator BDD: APPROVED (2026-03-16).** `specs/c-pipeline-spec.md` — rewritten to use Provider pattern (Claude Code / Codex as subprocess, not API calls). Architecture follows `vla-game-agent/pipeline`. ~310 lines (67% shorter than original draft). Covers behaviors B-1 through B-16: CLI, config generation, state management, resume, orchestration loop, run/analyze phases, agent invocation, success gates, tunable HP knobs, budget tracking, OOM recovery, divergence detection, comparison reporting.

**Pipeline orchestrator TDD: DONE (2026-03-16).** `tests/test_c_pipeline.py` — 57 tests across 5 classes (TestCli, TestConfigGeneration, TestPureHelpers, TestRunnerStateAndLoop, TestPolicies). Written by Codex via `specs/c-pipeline-tdd-handoff.md`. Ruff clean. 134/134 existing tests unaffected. Uses FakeProvider mock, `tmp_path` for state, monkeypatched training/eval stack.

**Pipeline orchestrator CODE: DONE (2026-03-16).** Implementation split into 3 files per user's "generic orchestration" requirement:
- `experiments/pipeline_runner.py` — Generic orchestration engine: `PipelineRunnerBase`, `MilestonePolicy` (dataclass), `RuntimeHooks` (dataclass). State machine loop, JSON persistence, resume, budget tracking, OOM/divergence recovery, agent prompt construction, mechanical fallback. Zero C-specific code.
- `experiments/c_milestones.py` — C-specific config: templates (C0–C4), gate functions, tunable knobs, 4L reference results, comparison report (reads actual state files), policy helper functions (`needs_phase1`, `needs_mi_eval`, `dependency_for`, etc.).
- `experiments/c_pipeline.py` — Thin CLI wiring: `PipelineRunner` subclass wires `RuntimeHooks` + `MilestonePolicy`, real providers (`ClaudeProvider`, `CodexProvider`), re-exports all test-expected names. `_OSProxy` for test monkeypatching, `_sigma_std_extractor` uses `sigma_summary()`.
- 57/57 pipeline tests green, 134/134 other tests green, ruff clean.
- Smoke tested: `scripts/smoke_pipeline.py` — 2L/2H/64d on TinyShakespeare, 50 steps, MLflow. 4 runs, correct failure (toy model can't pass C0 gate). Qualitative eval wrapped in try/except (supplementary).
- Implementation handoff: `specs/c-pipeline-implementation-handoff.md`. Review: `specs/c-pipeline-implementation-review.md`.

**Pile end-to-end validated (2026-03-16):**
- `scripts/smoke_pile.py` — 2L/2H/64d on real Pile data, 10 steps, CPU. Downloads 5 domains (wikipedia_en, stackexchange, arxiv, freelaw, pubmed_abstracts) via HF streaming → tokenize → cache.
- Pile loader bug fixed: was materializing entire domain into memory (`docs = [item["text"] for item in stream]`). Now uses `stream.shuffle(buffer_size=10_000)` for HF IterableDataset, falls back to collect+shuffle for test mocks.
- `setup_data()` fixed: handles Pile per-domain OOD keys (`test_ood_arxiv`, etc.) alongside AG News single `test_ood`.
- `datasets` added to `pyproject.toml` dependencies (was lazy-imported but never declared). **Also needs `pip install datasets` in global env for CUDA runs** (bare `python` uses global env, not uv venv).
- YAML config: `configs/c0_pile.yaml` — C0 template values for consistency with A/B series configs.
- Pile cache is per-token-target: `data/pile/{domain}_{token_limit}.pt`. Smoke test cached 50K-token files; real C0 (100M tokens) will re-download.
- 191/191 tests green, ruff clean.

**YAML-centric refactor (2026-03-16):** Pipeline now reads config from YAML files (`configs/{milestone_key}.yaml`) as template baseline. YAML is generated once from Python template; overrides are applied in-memory only (never saved back to YAML).
- `config_path_fn` parameter on `PipelineRunnerBase` — when set, pipeline loads/saves YAML; when `None` (tests), falls back to programmatic `build_milestone_config`.
- `save_yaml()` and `apply_dict_overrides()` added to `minigpt/config.py`.
- `config_path_for()` added to `c_milestones.py` — convention: `configs/{milestone_key}.yaml`.
- `--dry-run` now prints YAML content (reads file if exists, else generates from template).
- CLI `main()` passes `config_path_fn`; tests don't — so 57/57 frozen tests unaffected.
- If YAML doesn't exist on first run, pipeline auto-generates it from the Python template.

**Production fixes (2026-03-16):**
- `log_perplexity_mlflow` fixed: handles per-domain OOD dict (`{"arxiv": 2245.4, ...}`) not just single float.
- `MAX_RUNS` set to 3 (default). Overridable via `--max-runs` CLI param. Record-only milestones always 1.

**Known limitations (deferred to production):** `train.steps` range for C1 is `(1, 300K)` — frozen test sends `train.steps=1`, so can't narrow to `(50K, 300K)` without test change. `run_c3_phase1` is a stub returning hardcoded path. `eval_perplexity_suite` expects single OOD tensor — pipeline handles per-domain eval manually.

**Pipeline improvements — agent UX & reliability (2026-03-17):**

*Problem:* First agentic C0 run revealed 5 critical issues:
1. **Agent returned empty adjustments** — Claude CLI returned JSON envelope `{"type":"result","result":"Done."}` but `parse_agent_response` tried `json.loads()` on the envelope, getting no `diagnosis`/`adjustment` keys → empty defaults.
2. **Zero visibility** — no logging between runs (no banners, no gate results, no agent decisions).
3. **Empty-adjustment → wasted run** — pipeline ran identical config twice (1K steps, same PPL ~1628).
4. **C0 mechanical fallback empty** — only c1 (init_rho) and c3 (init_g) had fallback logic. C0 returned `{}`.
5. **NaN training ran all 50K steps** — model diverged at step ~1700 (warmup_steps=200 too short for 3e-4 LR), but training continued for hours with NaN loss.

*Fixes applied:*
- **Logging** (`pipeline_runner.py`): Run banners (`RUN {n}/{max} -- {milestone}`), config summary (steps/lr/warmup/bs), gate result with key metrics, agent response (diagnosis/reasoning/adjustment), mechanical fallback display, OOM/divergence messages. Inspired by VLA pipeline's `PipelineLogger`.
- **Envelope extraction** (`c_pipeline.py`): `_run_provider_command` now parses Claude CLI JSON envelope and extracts `result` field. Also strips `CLAUDECODE` env vars to prevent recursion (VLA pipeline pattern).
- **Robust response parsing** (`c_milestones.py`): `parse_agent_response` tries 3 strategies: (1) direct JSON, (2) fenced JSON (```json...```), (3) balanced-brace extraction from prose. Falls back to empty dict.
- **Empty-adjustment safeguard** (`pipeline_runner.py`): If agent returns empty `adjustment`, falls through to mechanical fallback with warning.
- **C0 mechanical fallback** (`pipeline_runner.py`): Doubles `train.steps` (min 50K from tunable range, max 300K). Also scales `train.warmup_steps` proportionally (4% of steps, clamped to range [500, 10K]).
- **NaN early-stop** (`minigpt/train.py`): Checks `math.isnan(train_loss) or math.isnan(val_loss)` at every eval interval. Breaks training loop immediately, saving GPU hours.
- **Agent prompt improved** (`pipeline_runner.py`): Includes knob ranges with `[min, max]`, directive tone ("You MUST return non-empty adjustment"), explicit JSON example, "no markdown, no prose" instruction.
- **Schema passed** to provider: `_AGENT_RESPONSE_SCHEMA` constant in `pipeline_runner.py`, passed to `run_role(schema=...)`.

*Issues from 2026-03-17 — all resolved 2026-03-20:*
1. ~~Claude CLI subprocess timeout~~ → FIXED: `--max-turns 1` + `timeout=300` + `TimeoutExpired` handling.
2. ~~Frozen test breakage~~ → FIXED: `test_four_consecutive_failures` updated to match MAX_RUNS (now 3).
3. ~~warmup_steps=200 in YAML~~ → FIXED: patience early-stop + agent briefing playbook diagnoses this.
4. ~~Config path convention~~ → Confirmed: `configs/{milestone_key}.yaml` (no `_pile` suffix).

**Pipeline improvements — agent intelligence & reliability (2026-03-20):**

*Problem:* Agent always returned empty responses. Pipeline was a glorified mechanical fallback. Three root causes found:

1. **JSON envelope type mismatch** — Claude CLI with `--json-schema` returns `result` as a parsed dict, not a string. Code did `str(envelope["result"])` → Python repr with single quotes → invalid JSON → `parse_agent_response` returned empty. Fix: check `isinstance(result, dict)` → use `json.dumps()`.
2. **YAML config corruption** — `_build_config_from_yaml` saved overrides back to the YAML file (`save_yaml(yaml_path, cfg)` after applying overrides). Mechanical fallback overrides persisted across runs, corrupting the template (e.g., LR mutated from 3e-4 to 3e-3). Fix: removed `save_yaml` after overrides — overrides are now in-memory only, YAML stays as template baseline.
3. **Thin agent prompt** — agent only saw raw result JSON and knob ranges. No context about model size, corpus, convergence expectations, or diagnostic reasoning.

*Fixes applied:*
- **Claude CLI** (`c_pipeline.py`): Added `--max-turns 1` (prevents multi-turn repo exploration), `timeout=300` with `TimeoutExpired` handling. Raw output logging for debugging (`[provider] raw stdout`).
- **JSON envelope fix** (`c_pipeline.py`): `_run_provider_command` now handles `result` as dict (→ `json.dumps`) or string (→ `str`).
- **YAML immutability** (`pipeline_runner.py`): Removed `save_yaml` after override application. YAML is generated once from template, never mutated by overrides.
- **Patience early-stop** (`minigpt/train.py`): New `patience_evals` param (default 10). After warmup, if val loss doesn't improve for N consecutive evals → breaks early with `early_stop_reason="patience"`. Returns `early_stop_reason` in metadata.
- **Enriched result dict** (`pipeline_runner.py`): `_execute_run` now includes `best_val_step`, `steps_planned`, `early_stop_reason` in result — agent sees step efficiency and why training stopped.
- **Agent briefing system** (`experiments/agent_briefing.md`): Comprehensive HP tuning playbook loaded into every agent prompt. Contains: project goal, model specs, 4L reference results, symptom→diagnosis→prescription lookup table, known pitfalls, decision rules. Agent doesn't need to read files — entire research brain is in the prompt.
- **Diagnostic summary** (`pipeline_runner.py`): `_build_diagnostic_summary()` computes and injects: gap analysis (current PPL vs target), step efficiency (best step / planned), early-stop reason with hints, run-over-run deltas showing effect of each adjustment.
- **Pre-run agent reasoning** (`pipeline_runner.py`): `--initial-agent` flag triggers agent call BEFORE first run. Agent sees template defaults, corpus/model specs, and briefing → reasons about optimal starting params instead of blindly using template defaults.
- **`--max-runs` CLI param** (`c_pipeline.py`): Configurable max runs per milestone (default from `MAX_RUNS` constant, currently 3). Record-only milestones (c2/c4) always stay at 1.
- **C-scale eval intervals** (`c_milestones.py`): `eval_interval=1000`, `checkpoint_interval=5000` in C0_TEMPLATE (was 200/500 from DEFAULT_CONFIG — too frequent for 100K+ step runs).

**Critical bugs found and fixed (2026-03-21):**

Three bugs prevented C0 from learning beyond unigram token frequencies (PPL stuck at ~1600 across all 5 runs, regardless of LR/steps/warmup).

*Bug 1 — Token-level shuffle destroyed training data (CRITICAL):*
`minigpt/data.py:205` had `all_id = all_id[torch.randperm(len(all_id))]`. This shuffled 200M **individual tokens** into random positions. Every 256-token training window became random gibberish — the model could only learn which tokens are frequent (unigram distribution), never what comes next. The observed loss=7.37 / PPL=1600 is exactly the unigram entropy of a GPT-2 BPE vocabulary on English text. **Intent:** mix Wikipedia + StackExchange domains so train/val/test splits are representative. **Reality:** documents were already shuffled during HuggingFace streaming (`stream.shuffle(buffer_size=10_000)` in `_load_domain_cached`). The `randperm` was redundant and destructive. Contrast with AG News (`prepare_agnews_data`): shuffles **articles** (line 101), not tokens — sequential structure preserved within documents. **Fix:** removed the `randperm` line entirely. Updated corresponding test helper `expected_id_splits()` in `tests/test_pile_data.py`.

*Bug 2 — Missing residual projection scaling (SIGNIFICANT):*
`minigpt/model.py:_init_weights` used flat `std=0.02` for ALL `nn.Linear` layers. GPT-2 and nanoGPT scale residual output projections (`CausalSelfAttention.proj`, `MLP.proj`) by `1/sqrt(2*n_layer)` to prevent signal variance from growing with depth. At 16 layers (32 residual additions), projection init should be `std ≈ 0.02/sqrt(32) ≈ 0.0035` — a 5.7x difference. Without this, deep models suffer from activation variance blowup, leading to saturated softmax, degraded gradient flow, and collapse toward unigram predictions. At 4 layers (8 additions), the effect is mild and tolerable; at 16 layers, it's damaging. **Fix:** added `_scale_proj()` helper that handles `DeterministicLinear` (`.linear.weight`), `BayesianLinear` (`.weight_mu`), and raw `nn.Linear` (`.weight`). Applied to both `attn.proj` and `mlp.proj` in every block after `self.apply(self._init_weights)`.

*Bug 3 — Dropout 0.2 too aggressive for 16 layers (MODERATE):*
C0 inherited `dropout=0.2` from `DEFAULT_CONFIG` (designed for 4L AG News). With 16 layers and dropout at 5 points per layer (3 in attention + 1 in MLP + 1 at embedding), gradient signal through the MLP residual path retains only `(0.8)^16 ≈ 2.8%` — extreme regularization. GPT-2 small (12L/117M) uses `dropout=0.1`. On a 100M-token Pile corpus, overfitting is not a concern for a 76M model. **Fix:** C0 template now explicitly sets `dropout=0.1`.

**Architecture improvement — Flash Attention (2026-03-21):**
Replaced manual attention computation in `CausalSelfAttention.forward()` with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. This is PyTorch 2.0+'s built-in SDPA which dispatches to Flash Attention kernels on CUDA. Benefits: (1) 2-4x faster attention, (2) O(N) memory instead of O(N²), (3) no manual causal mask buffer needed. The `mask` buffer and `attn_dropout` layer are removed — SDPA handles both internally via `is_causal=True` and `dropout_p` parameter. Not an architectural change — same computation, optimized implementation. Tests passed and got slightly faster (3.92s vs 4.15s).

**Pipeline improvements — automated diagnosis (2026-03-21):**

Additional pipeline hardening applied alongside the bug fixes:

- **Patience min_delta** (`train.py`): 0.1% relative improvement threshold. Noise-level improvements (val loss improving by 0.007 over 4000 steps) no longer reset the patience counter. Saves ~30 min of wasted GPU time per plateau.
- **Eval history in metadata** (`train.py`): Training loop now collects `{step, val_loss, train_loss, lr}` at every eval point and returns it in metadata. Pipeline builds a sampled loss curve summary for the agent prompt, including "90% of improvement happened by step X (Y% of training)" convergence analysis.
- **Fast NaN detection** (`train.py`): `torch.isnan(loss)` checked every step, not just at eval_interval. Saves up to 3.5 min of wasted GPU time per NaN event. Also guards `load_checkpoint(best_path)` against FileNotFoundError if NaN occurs before any checkpoint is saved.
- **`steps_completed` in metadata** (`train.py`): Exact step count when training stopped. Pipeline includes it in the result dict for convergence analysis.
- **Effective params in result** (`pipeline_runner.py`): Agent sees the actual lr/warmup/bs/dropout/grad_accum used, not just the override diff.
- **Training scale in initial prompt** (`pipeline_runner.py`): Computed effective batch size, tokens/step, total tokens, data passes — helps agent reason about whether training scale is adequate.
- **Gap severity classification** (`pipeline_runner.py`): PPL gap labels — EXTREME (>10x), LARGE (>3x), MODERATE (>1.5x), CLOSE (<1.5x) — with actionable guidance.
- **Wasted training detection** (`pipeline_runner.py`): If >50% of completed steps were post-best with no meaningful improvement, diagnostic explicitly warns "model is stuck."
- **Smarter C0 mechanical fallback** (`pipeline_runner.py`): When patience fires → increase LR (model is stuck in local minimum). When all steps complete → increase steps. Old behavior (always double steps regardless of stop reason) was wrong for plateaus.
- **Resilient knob validation** (`pipeline_runner.py`): Unknown knobs from agent are warned and skipped (not raised). Out-of-range values are clamped to bounds. One bad key from agent no longer kills the entire adjustment.
- **Empty adjustment fallback** (`pipeline_runner.py`): If `_validate_adjustment` strips all agent knobs (all unknown), falls back to mechanical instead of applying empty config.
- **Repeat detection** (`pipeline_runner.py`): If agent suggests identical params to previous run, augments with mechanical fallback to avoid wasting a run.
- **Agent briefing enhanced** (`agent_briefing.md`): Expanded plateau guidance (most common failure mode), convergence-aware decision rules ("patience + best_step < 20% → increase LR").

**C0 template HP changes (2026-03-21):**
- `lr: 6e-4` (was 3e-4 from DEFAULT_CONFIG). GPT-2 small used 2.5e-4 with 64x larger effective batch. Our small batch (32 sequences) benefits from higher peak LR.
- `warmup_steps: 4000` (was 2000). 4% of total steps — proportional to LR increase. Prevents NaN from aggressive early gradient steps.
- `dropout: 0.1` (was 0.2). GPT-2 standard for this depth.

**Agent provider fixes (2026-03-21):**

*Bug 4 — `--max-turns 1` too restrictive:*
Claude Code CLI's `--max-turns` controls how many reasoning/tool cycles the agent subprocess gets per invocation (NOT to be confused with `--max-runs` which is our pipeline flag for GPU training runs). With `--max-turns 1`, Claude spent its single turn on a tool call (reading files), got cut off (`"subtype":"error_max_turns"`, `"stop_reason":"tool_use"`), never produced text → always returned empty. Fixed: `--max-turns 5`.

*Bug 5 — `structured_output` envelope extraction:*
When `--json-schema` is passed, Claude CLI returns the result in `structured_output` field (not `result`). `_run_provider_command` only checked `envelope.get("result")` → fell through → `parse_agent_response` got the raw envelope string (with session_id etc.) → no `diagnosis` key at top level → always empty. Fixed: check `structured_output` first, then `result`.

*Bug 6 — Windows command-line length limit:*
Injecting full AGENTS.md (~91KB) + agent_briefing.md (~5KB) into the prompt made the `claude -p <prompt>` command line ~96KB, exceeding Windows' ~32K character limit for `CreateProcess`. `subprocess.run` raises `FileNotFoundError` on Windows when this happens (misleading error — the binary IS found, the argument list is too long). Fixed: pass prompt via stdin (`-p -` + `input=prompt` in `subprocess.run`). No length limit via stdin.

*Bug 7 — Windows charmap encoding:*
`subprocess.run(text=True)` uses the system default encoding on Windows (cp1252/charmap), which can't handle Unicode characters like `→` in AGENTS.md. Error: `'charmap' codec can't encode character '\u2192'`. Fixed: `encoding="utf-8"` in `subprocess.run`.

*Bug 8 — PATH resolution:*
`claude` CLI at `C:\Users\alexe\.local\bin\claude.EXE` was sometimes not found by subprocess PATH resolution. Fixed: `shutil.which("claude")` at provider init to resolve full path upfront.

*Subprocess timeout:* Increased to 600s (was 300s) to accommodate multi-turn agent.

**Agent context injection (2026-03-21):**
Key insight: `claude -p` is the same Claude model as interactive Claude Code — same reasoning, same capabilities. The gap was purely informational: the subprocess started fresh with only the prompt content, while interactive Claude has full conversation history.

Solution: inject the full AGENTS.md content (complete project history, all 4L results, bug history, decisions, expectations — ~1100 lines) directly into the agent prompt alongside `agent_briefing.md` (HP tuning playbook — 95 lines). This eliminates the need for the agent to spend turns reading files and gives it the same context depth as interactive Claude.

Architecture of a single agent invocation:
```
Prompt = task description
       + current config / run history / diagnostics
       + agent_briefing.md (HP playbook)
       + AGENTS.md (full project history)
       + response format instructions
```

The agent now has 5 turns purely for reasoning and (if needed) inspecting runtime artifacts like MLflow results or checkpoint files — no turns wasted discovering project context.

**`--max-turns` vs `--max-runs` (important distinction):**
- `--max-turns N` — Claude CLI flag. How many reasoning/tool cycles the agent *subprocess* gets per invocation. Internal to Claude. Default: 5.
- `--max-runs N` — Our pipeline flag. How many full *GPU training runs* (each ~2-3 hours) the pipeline attempts before giving up. Default: 3 (MAX_RUNS in c_milestones.py).
- One pipeline execution with `--max-runs 5` may invoke the agent 4-5 times (once initial + once per failed run), and each invocation gets `--max-turns 5` internally.

**Checkpoint reuse between runs — deliberately NOT done:**
When the agent adjusts HPs after a failed run, the next run trains from scratch (random init), not from the previous run's best checkpoint. Reasons:
1. LR schedule (warmup + cosine decay) designed from step 0 — resuming mid-schedule is invalid.
2. Adam optimizer state calibrated to old LR — new LR + old state = instability.
3. KL annealing (C1/C3) ramps from 0 — restarting on partially-trained posteriors is wrong.
4. Confounded results — can't attribute success to new HPs vs head start.
Exception: C2/C4 deliberately reuse C0/C3 checkpoints (post-hoc methods don't retrain).

**C0 GPU run history (2026-03-16 → 2026-03-21):**
- Session 1 (2026-03-17): 3 runs, all agent-empty → mechanical fallback only. LR corrupted to 3e-3 via YAML persistence bug. NaN at 50K steps. No progress.
- Session 2 (2026-03-20): YAML immutability fixed, patience early-stop added, agent briefing system added. 5 runs: agent still returned empty (max-turns=1 too restrictive). Mechanical fallback escalated LR 3e-4→6e-4→1e-3→NaN→doubled steps. ALL runs plateaued at PPL ~1600. Root cause: data bug (token-level shuffle).
- Session 3 (2026-03-21): All bugs fixed. Template HPs (lr=6e-4, warmup=4000, dropout=0.1). **Results: PPL 75 by step 2000, PPL 17 by step 26K.** C0 gate (PPL<80) passed at step 2000. Model still converging. Agent initial call still returned empty (structured_output bug — fixed mid-run, will take effect on next invocation). Training speed: ~167s per 1K steps.

**C0 FINAL RESULTS (2026-03-21) — PASSED:**

| Metric | Value |
|--------|-------|
| Status | completed, Run 1/1 |
| Best val loss | 2.437 at step 88K |
| Test ID PPL | **14.3** |
| Test OOD PPL | arxiv=29.3, freelaw=75.8, pubmed=140.4 |
| Gate | 14.3 < 80 ✓ AND freelaw(75.8) & pubmed(140.4) > 2×14.3=28.6 ✓ |
| Early stop | patience at step 98K (best at 88K, last 10K wasted) |
| Training time | 4.5 hours (16,276s) |
| MLflow run | `6215391b` |
| Checkpoint | `data/checkpoints/c0/ckpt_best.pt` |
| Pipeline state | `.pipeline-state/c0.json` |

Convergence trajectory (val loss → PPL):
- Step 1: 10.86 → 51970 (random init)
- Step 1K: 4.94 → 140 (data fix confirmed — sequential patterns learned)
- Step 2K: 4.32 → 75 (**C0 gate passed here**)
- Step 5K: 3.36 → 29
- Step 19K: 2.91 → 18
- Step 37K: 2.78 → 16
- Step 60K: 2.70 → 15
- Step 70K: 2.57 → 13
- Step 88K: 2.44 → 11.4 (**best**)
- Step 98K: 2.54 → 12.6 (patience fired, 10 evals no >0.1% improvement)

Key observations:
- 76M model converges well on Pile domain-split data. PPL 14.3 is strong for this corpus.
- OOD discrimination clear: pubmed (140.4) is 10x ID PPL, arxiv (29.3) is 2x, freelaw (75.8) is 5x.
- Training speed: ~167s per 1K steps on CUDA (Flash Attention).
- Patience mechanism worked correctly: saved ~10% of compute by stopping at 98K instead of 100K.
- For reference: 4L model on AG News reached test_id_ppl=49. 16L on Pile reaching 14.3 is expected — bigger model, more diverse data, longer training.

**Ideas considered but deferred:**

| Idea | Why not now | When |
|------|-----------|------|
| RoPE position encoding | User decision: "keep miniGPT basic." Learned absolute positions are adequate for block_size=256. | Future work |
| RMSNorm | ~10-15% speedup in norm ops, but pre-norm LayerNorm is correct and stable. Not a bottleneck. | Future work |
| SwiGLU activation | Better than GELU but adds complexity (3 weight matrices vs 2 in MLP). Not critical for research conclusions. | Future work |
| GQA (grouped-query attention) | Only matters at scale (>1B params) or very long sequences. 8 heads at 512d is fine. | Future work |
| KV-cache for generation | Only helps qualitative eval (token-by-token generation). Qualitative suite is <1% of total runtime. | Future work |
| Gradient checkpointing | GPU profiling showed 46% VRAM usage for deterministic 16L. Plenty of headroom. Not needed. | If OOM |
| Anthropic API provider | Direct SDK call (no tools, no CLI overhead). Considered as fix for agent-empty issue. User prefers Claude Code CLI as smart multi-turn agent. | If agent still fails |
| Document-level interleaving for Pile splits | Currently train/val/test are contiguous slices of concatenated domains. Wikipedia tokens come first, then StackExchange. A document-level interleave would give each split a mix of both domains. Low priority — the model will learn from whatever text it sees, and domain mixing mainly affects eval representativeness. | After C0 passes |

**C1 readiness analysis (2026-03-21):**

C1 = Variational Bayesian FFN. Same 16L/8H/512d architecture, but FFN layers (MLP.fc + MLP.proj in all 16 blocks) are BayesianLinear. Training minimizes ELBO = CE + kl_weight × KL. Gate: mi_ratio_mean > 1.2.

Code review (thorough, all 10 components checked):
- ✓ BayesianLinear: proper closed-form KL, weight sampling (μ + softplus(ρ) × ε)
- ✓ KL in training: added once per optimizer step (not per micro-batch), annealed per-step
- ✓ Checkpoint criterion: ELBO for Bayesian models (CE for deterministic)
- ✓ MI evaluation pipeline: complete (MC sampling → per-domain MI → ratio → gate)
- ✓ Config validation: enforces kl_weight > 0 when Bayesian enabled
- ✓ Pipeline wiring: all hooks connected (train, eval_mi_suite, uncertainty_eval_fn, sigma_std)
- ✓ Unit tests: 50+ covering Bayesian layers, uncertainty invariants, path selection
- No red flags found.

KL scaling analysis (4L → 16L):
- Bayesian weight elements: 16L has ~33.6M (16 blocks × 2 layers × 512×2048) vs 4L's ~524K → 64x more KL terms
- KL-to-data ratio: 33.6M/160M = 0.21 at 16L vs 524K/5M = 0.105 at 4L → **2x stronger KL pressure** with same kl_weight
- Decision: **Reduced kl_weight from 0.2 to 0.1** in C1_TEMPLATE to compensate. Keeps effective KL pressure ~same as 4L A2 (which achieved 1.43x MI ratio)

C1 template HPs (final):
- lr=6e-4, warmup=4000, steps=100K, bs=16, accum=2 (same as C0)
- bayes_ffn: enabled=True, init_rho=-2.0, prior_std=1.0
- **kl_weight=0.1** (was 0.2 — reduced for 16L scaling)
- kl_annealing_steps=5000 (5% ramp)
- dropout=0.1

Expected training time: ~6-7 hours (1.5x C0 due to BayesianLinear overhead in FFN)

**Agent first successful reasoning (2026-03-21) — C1 initial call:**

After fixing bugs 4-8, the agent produced its first intelligent response. Key qualities:
- **Evidence-based**: cited C0 results (PPL 14.3, convergence at 88K), 4L A2 results (MI ratio 1.43x)
- **Understood KL scaling math**: correctly identified 33.6M/160M = 0.21 ratio vs 4L's 0.105
- **Cited failure modes**: referenced init_rho=-3 freezing posteriors from A2 experiments
- **Compared to literature**: GPT-2-small scaling (lr=2.5e-4, batch 64x larger)
- **Correct conclusion**: no adjustments needed — template defaults are optimal
- **Identified diagnostics to watch**: "if posteriors collapse, σ_std < 0.01 is the diagnostic"
- Cost: $0.097 (Sonnet), 2 turns, 140s

This validates the pipeline architecture: Claude Code CLI as subprocess with full AGENTS.md context produces research-quality reasoning. The agent is not a mechanical fallback — it's an informed researcher.

Total bugs fixed to get agent working: 8 (max-turns, structured_output envelope, YAML persistence, mechanical fallback logic, prompt quality, Windows cmd-line length, charmap encoding, PATH resolution).

**C1 GPU run (2026-03-21):**
- 109.9M params (76M deterministic + 33.6M Bayesian μ/ρ). ELBO criterion.
- Agent approved template defaults (no adjustments) — first successful agent reasoning.
- Training speed: ~205s per 1K steps (1.23x slower than C0's 167s — BayesianLinear overhead modest).
- KL evolution: 52.82M (step 1) → 52.89M (step 5K, annealing completing) → 54.53M (step 58K). Steady growth indicates posteriors diverging from prior.

*Bug 9 — `torch.quantile()` input tensor too large:*
`sigma_summary()` concatenates all Bayesian sigma values (33.6M elements) and calls `torch.quantile()`, which fails on tensors > 2^24 (~16.7M) elements. Crashed after MI evaluation completed but before results were saved. Fixed: random subsample to 2^24 elements for quantile computation. Full mean/std/min/max still computed on all elements.

Recovery: loaded best checkpoint, computed sigma stats with fixed code, manually constructed `.pipeline-state/c1.json` from training output. MLflow run `a1071de8e6984a2a8fd612549f7644a0` was left in FAILED status with MI metrics missing (crash happened after MI computation but before MLflow logging). Patched via `mlflow.tracking.MlflowClient`: logged mi_id, mi_ratio per domain, mi_ratio_mean, sigma_std, sigma_mean as metrics; changed status FAILED→FINISHED. All C1 data now consistent across pipeline state, MLflow, and checkpoint.

**C1 FINAL RESULTS (2026-03-21) — PASSED:**

| Metric | Value |
|--------|-------|
| Status | completed, Run 1/1 |
| Best val loss (ELBO) | 3.057 at step 48K |
| Test ID PPL | **21.9** |
| Test OOD PPL | arxiv=53.4, freelaw=121.5, pubmed=275.6 |
| MI ratio | arxiv=1.34x, freelaw=1.28x, pubmed=1.34x |
| **MI ratio mean** | **1.32x** (gate >1.2 ✓) |
| Sigma stats | mean=0.123, std=0.016, range=[0.047, 0.243] |
| Early stop | patience at step 58K (best at 48K) |
| Training time | 3.3 hours (11,820s) |
| Checkpoint | `data/checkpoints/c1/ckpt_best.pt` |
| Pipeline state | `.pipeline-state/c1.json` |
| MLflow run | `a1071de8e6984a2a8fd612549f7644a0` (patched post-crash) |

Key observations:
- **MI ratio 1.32x at 16L vs 1.43x at 4L.** The signal is weaker at scale, but still clearly above gate. Possible reasons: (1) kl_weight=0.1 may be slightly conservative — posteriors aren't as differentiated as at 4L with kl_weight=0.2. (2) Pile is more diverse than AG News — OOD distinction is harder. (3) 16L model has more capacity to memorize, reducing the ID/OOD gap.
- **Posteriors NOT collapsed:** sigma_std=0.016 >> 0.01 threshold. Sigma range [0.047, 0.243] shows meaningful differentiation across layers — some layers learned tighter posteriors (more certain), others wider (less certain).
- **Test ID PPL 21.9** vs C0's 14.3 — higher as expected. The KL regularization trades CE accuracy for uncertainty estimation. The gap (21.9 vs 14.3) is moderate, suggesting kl_weight=0.1 is well-balanced.
- **Patience stopped at 58K** (vs C0's 98K) — Bayesian training converges faster in ELBO terms because both CE and KL can improve. 42% of compute saved.
- **OOD PPL gradient:** arxiv(53.4) < freelaw(121.5) < pubmed(275.6). Same ordering as C0 but with different magnitudes. The model discriminates domains even with Bayesian layers.

Comparison with 4L reference:
| Metric | 4L A2 | 16L C1 | Notes |
|--------|-------|--------|-------|
| MI ratio | 1.43x | 1.32x | Weaker at scale, but above gate |
| Sigma std | 0.147 | 0.016 | Posteriors less differentiated at 16L |
| Sigma mean | ~0.13 | 0.123 | Similar starting point |
| ID PPL | 49 | 21.9 | Better model, more data |
| kl_weight | 0.2 | 0.1 | Scaled for 2x KL ratio |

**Next steps — ordered plan:**

Ordering rationale: C4 depends on C3 (needs C3's deterministic LoRA checkpoint from Phase 1). C2 is independent (uses C0 checkpoint, already available). Running C3 first unblocks C4 immediately. C2 can slot in anywhere — it's a quick single run with expected negative result.

1. **Run C3** (BLoB LoRA — variational LoRA, 16L scaled B2):
   - **What:** Two-phase training. Phase 1: pretrain deterministic model on HackerNews domain (ID for LoRA). Phase 2: inject BLoB LoRA adapters into FFN layers, fine-tune with ELBO loss. `run_c3_phase1` is currently a stub — needs implementation before running.
   - **Template:** rank=16, alpha=32, init_g=0.1, prior_std=0.2, kl_weight=0.2, steps=10K, bs=32
   - **Gate:** mi_ratio_mean > 1.05. At 4L B2 achieved 1.13x (weak positive).
   - **Dependency:** None (standalone). But its checkpoint is needed by C4.
   - **Command:** `python experiments/c_pipeline.py --milestone c3 --provider claude --provider-model sonnet --initial-agent --max-runs 5`
2. **Run C2** (Post-hoc Laplace on C0 deterministic checkpoint — 16L scaled B1):
   - **What:** Take the trained C0 model (76M params, deterministic), fit diagonal Laplace approximation to FFN weights post-hoc (no retraining). Compute Fisher curvature from training data, then sample perturbed weights for MI evaluation.
   - **Template:** laplace selection_mode=ffn, damping=1.0, sample_scale=1.0, n_curvature_batches=30
   - **Gate:** Record-only (no gate). Has `should_early_abort`: if curvature_mean < 1e-4 AND mi_ratio_mean < 1.02, abort early.
   - **Dependency:** C0 checkpoint (`data/checkpoints/c0/ckpt_best.pt`) — already available.
   - **Expected result:** MI ratio ~1.00x (negative). At 4L B1, diagonal Fisher curvature was too flat at convergence — no OOD signal. Same physics applies at 16L.
   - **Command:** `python experiments/c_pipeline.py --milestone c2 --provider claude --provider-model sonnet --no-agent`
3. **Run C4-TFB** (Post-hoc TFB on C3's deterministic LoRA checkpoint — 16L scaled B3-TFB):
   - **What:** Take C3's Phase 1 checkpoint (deterministic LoRA), apply Training-Free Bayesianization: SVD of LoRA B matrix → variance search → weight sampling.
   - **Template:** epsilon=0.1, n_search_samples=10, n_anchor_batches=20
   - **Gate:** Record-only. Always passes (gate returns True).
   - **Dependency:** C3 Phase 1 checkpoint.
   - **Expected result:** MI ratio ~1.10x (positive, same as B3-TFB at 4L). TFB succeeds post-hoc because it uses structural SVD information, not curvature.
4. **Run C4-LAP** (Post-hoc Laplace-LoRA on C3's deterministic LoRA checkpoint — 16L scaled B3-LAP):
   - **What:** Take C3's Phase 1 checkpoint, fit diagonal Laplace to LoRA parameters.
   - **Template:** laplace selection_mode=lora, damping=1.0, sample_scale=1.0, n_curvature_batches=30
   - **Gate:** Record-only. Has `should_early_abort` (same as C2).
   - **Dependency:** C3 Phase 1 checkpoint.
   - **Expected result:** MI ratio ~1.00x (negative, same as B3-LAP at 4L). Diagonal curvature fails on LoRA params too.
5. **Generate comparison report:** `python experiments/c_pipeline.py --compare`. Write up Table 2 for the paper.

**C3 readiness analysis (2026-03-21):**

C3 = BLoB LoRA (variational LoRA fine-tuning). Same 16L/8H/512d architecture, but FFN layers wrapped with BLoBLoRALinear adapters. Base model frozen, only LoRA params trainable. Two-phase: Phase 1 pretrain (reuses C0 checkpoint), Phase 2 BLoB LoRA fine-tune on HackerNews. Gate: mi_ratio_mean > 1.05.

Code changes for C3 support:
- **`run_c3_phase1` implemented** (`c_pipeline.py`): No longer a stub. Reuses C0 checkpoint (same 16L/8H/512d architecture) — copies `data/checkpoints/c0/ckpt_best.pt` to `data/checkpoints/c3/base/ckpt_best.pt`. Saves ~4.5 hours of redundant pretraining.
- **`prepare_model` hook** (`pipeline_runner.py`): New optional hook in `RuntimeHooks` (default=None). Called between `setup_model()` and `train()` in `_execute_run()`. For C3: loads base checkpoint, then injects BLoB LoRA adapters. Gracefully skips for test mocks (checks `isinstance(model, nn.Module)`).
- **`_prepare_model` function** (`c_pipeline.py`): Wired as `prepare_model` hook. If `cfg["train"]["resume_checkpoint"]` exists, loads checkpoint. If `cfg["lora"]` section exists, calls `inject_lora()`.

Parameter counts (rank=16, alpha=32, target=ffn):
- Total LoRA params: **1,966,080** (1.97M trainable)
- Bayesian params (lora_A_mu + lora_A_g): **1,310,720** (1.31M)
- lora_B (deterministic): 655,360
- Base model: 76.3M (frozen)
- Compare B2 (4L): 163K Bayesian params (8x fewer)

KL scaling analysis (critical):
- C3: 1.31M Bayesian elements / 80M train tokens = 0.016 KL/data ratio
- B2: 163K / 2M = 0.082 KL/data ratio
- With same kl_weight=0.2: C3 has **5x WEAKER** KL pressure than B2
- B2 achieved σ_mean=0.0083 (constrained) with its KL pressure. Even weaker KL at C3 would collapse posteriors further.
- **Fix: kl_weight raised from 0.2 to 1.0** — matches B2's effective KL pressure (verified mathematically)

Template HP fixes (C3_PHASE2_TEMPLATE):
- `lr: 3e-4` (was 6e-4 inherited from C0 pretraining — too high for fine-tuning)
- `warmup_steps: 500` (was 4000 = 40% of 10K steps — way too long. B2 used 500 = 5%)
- `weight_decay: 0.0` (was 0.1 — KL already regularizes. B2 used 0.0)
- `eval_interval: 500` (was 1000 — shorter runs need more frequent eval)
- `kl_weight: 1.0` (was 0.2 — increased to match B2's effective KL/data pressure)

C3 template HPs (final):
- Base: 16L/8H/512d, deterministic base from C0 (frozen)
- LoRA: rank=16, alpha=32.0, target=ffn, prior_std=0.2, init_g=0.1
- Training: lr=3e-4, warmup=500, steps=10K, bs=32, accum=1, weight_decay=0.0
- KL: kl_weight=1.0, kl_annealing_steps=1000
- Data: HackerNews (ID), arxiv/freelaw/pubmed (OOD), pile_id_tokens=100M
- dropout=0.1

Smoke test results (2026-03-21):
- Config generation: all HPs correct ✓
- Policy functions: gates, dependencies, phase1 — all correct ✓
- Phase 1 checkpoint reuse: C0 copied to C3/base ✓
- Full flow (tiny model): checkpoint load → LoRA inject → forward → KL → ELBO → sigma_summary → freeze/unfreeze — all correct ✓
- No variation at init: expected (lora_B=0 at init, variation emerges during training) ✓
- CLI --dry-run: correct YAML output ✓
- 191/191 tests pass, ruff clean ✓

**C3 GPU run (2026-03-21):**
- Phase 1: C0 checkpoint reused (no additional GPU time).
- Phase 2: BLoB LoRA fine-tune on HackerNews, 10K steps, bs=32.
- 78.3M total params (76.3M frozen + 1.97M LoRA trainable, 1.31M Bayesian).
- Expected training time: ~30-60 min (10K steps at bs=32, LoRA is lighter than full Bayesian).
- Agent initial reasoning invoked (`--initial-agent`).
- Watching for: sigma (G²) collapse (σ_mean < 0.001 = bad), MI ratio > 1.05 (gate), patience behavior.

### Paper Structure
Table 1: 4L results (existing). Table 2: 16L results (C milestone). Analysis: which methods scale? Which findings transfer? The 2×2 matrix at two scales gives a clean, publishable comparison.

---

## Future Work (Parked)
Non-Bayesian improvements:
- RoPE, SwiGLU, KV-cache, RMSNorm, GQA
- Mixed precision: **DONE** (AMP auto-enabled)
- Flash Attention: **DONE** (`F.scaled_dot_product_attention`, 2026-03-21)

## References
- **BLoB** (NeurIPS 2024) — Bayesian LoRA by backprop. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- **TFB** (NeurIPS 2025) — Training-Free Bayesianization for LoRA. [arXiv:2412.05723](https://arxiv.org/abs/2412.05723)
- **Laplace-LoRA** (2023) — Laplace approximation on LoRA params. [arXiv:2308.13111](https://arxiv.org/abs/2308.13111)
- **ScalaBL** (2025) — Bayesian inference in low-dim subspace. [arXiv:2506.21408](https://arxiv.org/abs/2506.21408)
- **ICLA** (WACV 2025) — Identity Curvature Laplace for OOD. [arXiv:2312.10464](https://arxiv.org/abs/2312.10464)
- **Laplace Redux** (NeurIPS 2021) — Effortless Bayesian Deep Learning. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)
