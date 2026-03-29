# Paper Improvements Spec

Four targeted improvements to `docs/paper.md` based on simulated ICML/NeurIPS review.

---

## P1. Bootstrap Confidence Intervals

**Goal.** Add 95% CIs to all AUROC values in Tables 1-2 so that method rankings are statistically grounded.

**Approach.** Given per-sequence uncertainty scores (MI, max-prob) and binary labels (ID=0, OOD=1) already produced by `scripts/eval_c_checkpoints.py`:

1. Implement `bootstrap_ci(scores, labels, metric_fn, n_bootstrap=10000, ci=0.95)` in `minigpt/uncertainty.py`. Returns `(point_estimate, ci_low, ci_high)`.
2. Add a `--bootstrap` flag to `scripts/eval_c_checkpoints.py` that reports CIs for AUROC (primary) and optionally FPR@95, AUPRC.
3. Update paper Tables 1-2 with format `0.916 [0.89, 0.94]`.

**Key design decisions:**
- Resample *sequences* (not tokens) — the unit of independence is a sequence.
- Use the existing `auroc()` function from `uncertainty.py` as `metric_fn`.
- Seed the bootstrap RNG for reproducibility.
- 10,000 resamples is standard; percentile method for CI bounds.

**Acceptance criteria:**
- `bootstrap_ci()` is a tested, reusable function in `uncertainty.py`.
- Running `eval_c_checkpoints.py --bootstrap` prints CIs alongside point estimates.
- Paper tables show CIs for AUROC at minimum.

---

## P2. MC Dropout Baseline

**Goal.** Add MC Dropout as a fifth Bayesian method row in Table 1. MC Dropout is a variational approximation (Bernoulli posterior over weights) and fits the paper's weight-posterior framing.

**Approach.**

1. Take the C0 deterministic checkpoint (already trained, ppl=14.3).
2. At inference, enable dropout (keep `model.train()` for dropout layers only, or explicitly set dropout layers to training mode). Standard rate: 0.1 (GPT-2 default).
3. Run $N=20$ forward passes with dropout active. Compute MI from prediction disagreement across passes.
4. Evaluate on the same 500 ID + 500 OOD protocol. Report AUROC, FPR@95, AUPRC, ECE, Brier, NLL, AURC — same as all other methods.
5. Also benchmark latency/VRAM for N=3,5 (Table 3) and AUROC vs N (Table 4).

**Key design decisions:**
- No retraining — the point is that MC Dropout is a zero-cost Bayesian approximation applied to any existing checkpoint.
- Dropout rate 0.1 (the rate already in the model architecture). If the model was trained with dropout, this is the correct rate. If not, try 0.1 and 0.3.
- Enable dropout only on attention/FFN dropout layers, not embedding dropout.

**Acceptance criteria:**
- MC Dropout row appears in Tables 1-4 of the paper.
- Script is either integrated into `eval_c_checkpoints.py` or a standalone `scripts/eval_mc_dropout.py`.

---

## P3. Narrow "Laplace" to "Diagonal Laplace"

**Goal.** Every unqualified mention of "Laplace" in `docs/paper.md` that refers to our negative result must say "diagonal Laplace". KFAC Laplace is untested and should not be implicated.

**Scope.** Text edits only — abstract, contributions, Section 6 discussion, conclusion. The method description (Section 3.4) already says "diagonal" but the claims elsewhere don't.

**Acceptance criteria:**
- No unqualified "Laplace fails" / "Laplace approximation fails" remains in the paper.
- Contribution 3 explicitly says "diagonal Laplace".
- Conclusion explicitly says "diagonal Laplace".
- Limitations paragraph mentions KFAC as an open question (already present — verify unchanged).

---

## P4. Reframe LoRA vs Full-Weight Claim

**Goal.** Replace causal claims ("LoRA outperforms because...") with observational claims ("LoRA-based methods achieved stronger results in our setup") and list confounds explicitly.

**Scope.** Text edits only — abstract, Section 6 discussion ("Why LoRA outperforms full-weight"), conclusion.

**Confounds to list:**
1. Training procedure: from-scratch ELBO vs pretrain-then-finetune.
2. Base model quality: C0 ppl=14.3 backbone vs C1 ppl=21.9 joint training.
3. Parameter count: 33.6M vs 1.97M Bayesian params.

**Acceptance criteria:**
- No causal "LoRA outperforms because" phrasing remains.
- Section 6 paragraph lists all three confounds and frames the subspace interpretation as a hypothesis, not a conclusion.
- Abstract uses observational language.
