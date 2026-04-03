# Milestone History

Detailed record of every milestone — results, MLflow run IDs, bug fixes, pipeline events, and interim findings. Read when investigating what happened in a specific milestone or tracing a result back to its run.

---

## 4-Layer Scale (AG News, 4L/4H/256d, ~16M params)

- **A0: DONE** — Deterministic baseline. test_id_ppl=49.11, test_ood_ppl=540.28. MLflow `5dc45450`.
- **A1: DONE** — Bayesian output head. MI ratio **1.36x** (sigma=0.22). Ceiling: vocabulary-level only.
- **A2: DONE** — Bayesian FFN. MI ratio **1.43x batch / 1.70x qual** (sigma=0.147). MLflow `76d049b7`. Best method at 4L.
- **A3: CLOSED** — Bayesian FFN + attn V. Negative result vs A2 (4 runs, all worse).
- **B1: DONE (NEGATIVE)** — Post-hoc Laplace on FFN. Both approaches: MI ratio 1.00x. Diagonal Fisher curvature too flat at convergence.
- **B2: DONE (WEAK POSITIVE)** — BLoB LoRA. R2: MI ratio 1.13x batch / 1.02x qual (163K Bayesian params). Sigmas constrained (mean=0.0083). ID ppl=227. Weaker than A2 but directionally positive.
  - Category-split design: pretrain AG News cat 1 (World), LoRA fine-tune cat 2 (Sports), OOD cats 3+4.
  - R2 hyperparams: rank=16, alpha=32, lr=3e-4, init_g=0.1.
  - No Flipout for 4L miniGPT — reserved for C milestone (16L/batch_size=8).
- **B3: DONE (MIXED)** — Post-hoc LoRA. **B3-TFB: MI ratio 1.10x** (sigma_q=0.013, SVD-structured variance works). **B3-LAP: MI ratio 1.00x** (diagonal curvature fails, same as B1). 2x2 matrix complete.

## 16-Layer Scale (The Pile, 16L/8H/512d, ~76M params) — C Milestone

Pipeline = Agentic HP Optimization (THE MAIN GOAL): LLM agent (Claude/Codex as subprocess via Provider pattern) runs experiments 1-by-1, checks success gates, invokes agent on failure for HP adjustments. Agent replaces manual researcher babysitting.

- **Phase 0: DONE** — GPU profiled: bs=16/accum=2 full-weight, bs=32 LoRA.
- **Pile data loader: DONE** — 31/31 tests, 134/134 full suite. `load_pile_data()` in `minigpt/data.py`.
- **Pipeline BDD: APPROVED** (2026-03-16). `specs/c-pipeline-spec.md`.
- **Pipeline TDD: DONE** (2026-03-16). `tests/test_c_pipeline.py` — 57 tests.
- **Pipeline Code: DONE.** 57/57 tests green (191/191 total).
- **Pipeline agent intelligence: DONE** (2026-03-20). Agent briefing system, diagnostic summaries, pre-run reasoning, patience early-stop, JSON envelope fix.
- **9 bugs fixed (2026-03-21):** (1) token-level shuffle, (2) residual projection scaling, (3) Flash Attention, (4) structured_output envelope, (5) max-turns 1->5, (6) AGENTS.md injection, (7) Windows cmd-line length (stdin fix), (8) charmap encoding (UTF-8 fix), (9) torch.quantile overflow.
- **Agent WORKING** (2026-03-21): First successful reasoning on C1 initial call. Evidence-based, cited C0 results + KL scaling math + 4L A2 history. Cost $0.097/call (Sonnet).

### Sub-milestones

- **C0 DONE** (2026-03-21): test_id_ppl=14.3, OOD: arxiv=29.3, freelaw=75.8, pubmed=140.4. Best val loss 2.437 at step 88K. Patience stopped at 98K. MLflow `6215391b`.
- **C1 DONE** (2026-03-21): MI ratio **1.32x** (gate >1.2 pass). test_id_ppl=21.9, sigma_std=0.016. Best val 3.057 at step 48K, patience at 58K. MLflow `a1071de8` (patched post-crash).
- **C2 DONE (NEGATIVE)** (2026-03-22): MI ratio **1.00x** (gate >1.05 failed). Curvature mean=0.000000. Confirms B1 at 16L scale. Diagonal Laplace on full weights fails at every scale.
- **C3 DONE** (2026-03-22): MI ratio **1.53x** (gate >1.05 pass). test_id_ppl=64.9 (hackernews), OOD: arxiv=34.8, freelaw=91.3, pubmed=156.6. sigma_std=0.0066. 27 min training. MLflow `3b22bfcf`. **KEY FINDING: BLoB LoRA scales better than full-weight variational** (1.53x vs C1's 1.32x — reversed from 4L where B2 1.13x < A2 1.43x).
- **C4-TFB DONE (POSITIVE)** (2026-03-22): MI ratio **1.35x** (sigma_q=0.030). TFB (zero training, 7 min fit) matches C1 variational full-weight (1.32x). SVD-structured variance scales strongly: 4L 1.10x -> 16L 1.35x.
- **C4-LAP DONE (NEGATIVE)** (2026-03-22): MI ratio **1.00x**. Diagonal Laplace fails on LoRA (same as B3-LAP at 4L). Curvature flat at convergence.

### Post-hoc pipeline & infrastructure

- **Post-hoc pipeline DONE** (2026-03-22): `posthoc_fit_fn` hook, `_posthoc_fit` dispatch (laplace/tfb), `posthoc_method` config field, steps=0 for C2/C4_TFB/C4_LAP.
- **BLoB->DeterministicLoRA conversion DONE** (2026-03-22): `_prepare_model` maps BLoB checkpoint keys (`lora_A_mu`->`lora_A`) for C4 post-hoc milestones. 5 new tests.
- **Bug fix (2026-03-22):** `comparison_report()` used C4-LAP for "Laplace full" row — fixed to use C2. Added `_fmt()` for float formatting.

## Evaluation & Production — D/P Milestones

- **D0 DONE** (2026-03-27): Metrics framework — AUROC, FPR@95, AUPRC, ECE, Brier, NLL, AURC in `minigpt/uncertainty.py`. 45 tests. BDD->TDD->Code.
- **D1 DONE** (2026-03-27): Eval of all 6 C checkpoints. Script: `scripts/eval_c_checkpoints.py`. Spec: `specs/d1-eval-c-checkpoints.md`. 500 ID + 500 OOD seqs, N=20 MC, block_size=256.
  - C3 BLoB LoRA: AUROC **0.909 [0.890, 0.925]**, FPR@95=0.424, ECE=0.0438, NLL=3.06
  - C4-TFB LoRA: AUROC **0.917 [0.900, 0.933]**, FPR@95=0.384, ECE=0.0215, NLL=3.06
  - C1 Variational FFN: AUROC **0.874 [0.852, 0.895]**, FPR@95=0.494, ECE=0.0237, NLL=3.31
  - MC Dropout (P2): AUROC **0.898 [0.877, 0.917]**, FPR@95=0.368, ECE=0.0119, NLL=2.79
  - C0 Deterministic: AUROC 0.591 [0.556, 0.626] (max-prob), ECE=0.0224, NLL=2.79
  - C2 Diag. Laplace FFN: AUROC 0.536, NLL=9.10, Brier=1.000
  - C4-LAP Diag. Laplace LoRA: AUROC 0.494, NLL=9.73, Brier=0.998
  - **Key finding:** TFB ties BLoB on AUROC despite lower MI ratio (CIs overlap). MI ratio overstates gap.
  - `docs/metrics-guide.md` — plain-English metric explanations. `specs/d-production-inference.md` — full D milestone spec.
- **D2 DONE** (2026-03-28): Mean-weights inference. `compute_perplexity_mc()` in `evaluate.py`. 9 tests. Mean-weights PPL ~ MC-averaged PPL: C1 diff=0.29%, C3 diff=3.93% (gate <5% PASS). Script: `scripts/verify_mean_weights.py`. Spec: `specs/d2-mean-weights-spec.md`.
- **D3 DONE** (2026-03-28): Production benchmarks on RTX 4070. Script: `scripts/benchmark_inference.py`. Spec: `specs/prod-uncertainty-approaches.md`.
  - C3 LoRA MC N=5: AUROC=0.879, 84ms, 382 MB VRAM. AUROC gate (>0.80) PASS.
  - **N=3 is the knee:** AUROC 0.50->0.86 (97% of N=20 signal). Diminishing returns beyond N=5.
  - LoRA MC uses 28% less VRAM than full variational (382 vs 534 MB).
  - Production recipe: merged mean-weights for serving (zero overhead), N=3 MC for uncertainty scoring (6x overhead, 50ms/seq).
- **P1 DONE** (2026-03-28): Bootstrap 95% CIs. `bootstrap_ci()` in `uncertainty.py`, 11 tests. `eval_c_checkpoints.py --bootstrap --save-scores`. Saved: `data/d1_scores.pt`.
  - TFB: AUROC **0.917 [0.900, 0.933]**, BLoB: **0.909 [0.890, 0.925]** (CIs overlap).
  - Rankings stable across bootstrap resamples.
- **P2 DONE** (2026-03-28): MC Dropout baseline. `enable_dropout()` CM in `layers.py`, 6 tests. Script: `scripts/eval_mc_dropout.py`. Saved: `data/mc_dropout_scores.pt`.
  - MC Dropout AUROC **0.898 [0.877, 0.917]** — zero extra training. Overlaps BLoB CI.
  - ECE=0.0119 (best calibration), NLL=2.79, Brier=0.608.

## Paper Polish — P3/P4

- **P3 DONE** (2026-03-29): Narrow "Laplace" -> "diagonal Laplace" throughout paper. Table labels, 2x2 matrix, body text.
- **P4 DONE** (2026-03-29): Reframe all Section 6 "why" paragraphs as hypotheses. LoRA claim observational with 3 confounds. MC Dropout discussion added.
- **Paper tables DONE** (2026-03-29): CIs in Tables 1-2, MC Dropout row, Section 3.6, updated eval protocol. Point estimates from `data/d1_scores.pt`.
- **References FIXED** (2026-03-29): 4/11 had wrong authors (BLoB, TFB, Laplace-LoRA, ScalaBL). All verified against arXiv. Orphaned Lakshminarayanan removed.
