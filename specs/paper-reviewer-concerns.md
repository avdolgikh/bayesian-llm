# Paper Reviewer Concerns — Future Research Candidates

Simulated ICML/NeurIPS review of `docs/paper.md`. Each concern includes the reviewer objection, its severity, and a concrete remediation path.

---

## 1. Missing baselines: Deep Ensembles and MC Dropout

**Objection.** The paper cites Lakshminarayanan et al. (2017) and Gal & Ghahramani (2016) but never compares against them. Deep ensembles are the de facto gold standard for uncertainty estimation; MC Dropout is the simplest Bayesian approximation and the most widely deployed in practice. Without these baselines the reader cannot judge whether any of the four Bayesian methods offer value over trivially available alternatives.

**Severity.** High — most likely single reason for desk rejection.

**Remediation.**

- **Deep Ensembles (M=3, M=5).** Train 3-5 independent C0-class models (different seeds, same hyperparameters). Score MI via prediction disagreement across ensemble members. Report AUROC, FPR@95, latency, VRAM.
- **MC Dropout.** Take the C0 deterministic checkpoint, enable dropout at inference (standard rates 0.1-0.3), run N forward passes. Score MI from dropout-induced disagreement. Report same metrics.
- **Expected outcome.** Ensembles will likely be competitive on AUROC but at M$\times$ the parameter/VRAM cost. MC Dropout will likely underperform trained variational methods (known weak posteriors). Either outcome strengthens the paper: if ensembles win, the contribution becomes "LoRA Bayesian methods match ensembles at fraction of cost"; if they lose, the contribution is clean.

**Effort.** Low-medium. No new code beyond a thin ensemble wrapper and a dropout-at-inference flag. 3 C0 re-trains (~3 hours each on RTX 4070). MC Dropout is a single-afternoon experiment.

---

## 2. Confounded LoRA vs full-weight comparison

**Objection.** The central claim — "LoRA-based methods dominate full-weight" — attributes the AUROC gap to parameterization (rank-16 subspace vs full FFN). But the two settings differ in multiple dimensions simultaneously:

1. **Training procedure.** Variational FFN (C1) trains the entire model from scratch with ELBO. BLoB LoRA (C3) pre-trains a deterministic base (C0), then fine-tunes LoRA adapters. Different optimization problems.
2. **Base model quality.** LoRA methods inherit a well-converged backbone (C0 ppl=14.3). Variational FFN must learn representations and posteriors jointly (C1 ppl=21.9). The PPL gap alone could explain the AUROC gap, independent of any Bayesian effect.
3. **Number of Bayesian parameters.** 33.6M vs 1.97M — unclear whether "fewer is better" or "the specific LoRA subspace is better."

The discussion offers one interpretation ("LoRA constrains posteriors to meaningful directions") but the design does not isolate this mechanism.

**Severity.** High — undermines the paper's strongest claim.

**Remediation options (pick one or more):**

- **Ablation A: Variational on importance-selected subset.** Apply variational inference to only ~2M parameters chosen by Fisher information magnitude (the most important full-weight parameters). If this matches LoRA, the effect is "fewer well-chosen params"; if it doesn't, the effect is specifically about LoRA's geometric subspace.
- **Ablation B: BLoB LoRA from random init.** Train BLoB LoRA without a pre-trained C0 base (random initialization + LoRA from step 0). If AUROC drops significantly, the advantage is in the pre-trained backbone, not LoRA parameterization.
- **Ablation C: Variational FFN fine-tune.** Take the C0 checkpoint, freeze most layers, apply variational inference to only FFN layers during a short fine-tune phase (matching the LoRA training budget). This controls for the pretrain-then-finetune vs from-scratch confound.
- **Minimum viable fix.** If ablations are too expensive, reframe the claim. Replace "LoRA outperforms full-weight" with "LoRA-based Bayesian methods achieve stronger OOD detection in our setup" and explicitly list the confounds as open questions.

**Effort.** Ablation A or C: medium (1-2 days). Ablation B: 1 C3-class training run (~30 min). Reframing: trivial (text edits only).

---

## 3. No error bars — single-run results

**Objection.** Every number in Tables 1-5 comes from a single training run with a single seed. Key comparisons rest on small gaps:

- BLoB (0.916) vs TFB (0.917): $\Delta$ = 0.001
- TFB (0.917) vs Variational FFN (0.876): $\Delta$ = 0.041

Without confidence intervals or multi-seed runs, the entire ranking could shuffle with a different random seed. The claim that TFB "matches" BLoB is statistically unsupported.

**Severity.** Medium-high — standard expectation at top venues.

**Remediation options:**

- **Option A: Multi-seed training (ideal).** Re-train each method with 3 seeds. Report mean $\pm$ std for AUROC, FPR@95. Expensive but definitive.
- **Option B: Bootstrap CIs on existing eval data (cheap).** The evaluation already produces per-sequence MI scores for 500 ID + 500 OOD sequences. Bootstrap-resample these 1000 scores (e.g., 10,000 resamples), recompute AUROC each time, report 95% CI. This captures evaluation variance (not training variance), but is better than nothing and costs zero compute.
- **Option C: Permutation test.** Test whether the AUROC difference between two methods is significant via a paired permutation test on per-sequence scores.

**Recommended approach.** Option B immediately (1 hour of scripting), Option A if time permits (days of GPU).

**Effort.** Option B: trivial — ~50 lines of code. Option A: 3$\times$ the full training budget.

---

## 4. Diagonal Laplace is a strawman for "Laplace fails"

**Objection.** The paper frames the result broadly: "Laplace approximation fails for LM OOD detection" (abstract, contribution 3, Section 6, conclusion). But only *diagonal* Laplace is tested — the weakest variant. The Laplace-LoRA paper (Yang et al., 2024) specifically uses KFAC, which captures off-diagonal block curvature. The limitations section mentions this in one sentence, but the rest of the paper repeatedly says "Laplace fails" without qualification.

The finding that diagonal Fisher is flat at convergence is unsurprising and well-known. The interesting open question — whether KFAC Laplace also fails for LMs — is not addressed.

**Severity.** Medium — fixable by either narrowing language or adding the experiment.

**Remediation options:**

- **Option A: Implement KFAC Laplace.** Fit a block-diagonal (KFAC) posterior on LoRA parameters (matching Yang et al., 2024). Report AUROC. If KFAC also fails for LM OOD, the negative result is genuinely strong. If KFAC succeeds, the paper's story changes but becomes more nuanced and interesting.
- **Option B: Narrow the language (minimum viable fix).** Replace every instance of "Laplace fails" / "Laplace approximation" with "diagonal Laplace" throughout the paper. Explicitly state that KFAC Laplace is an open question, not a tested claim. Adjust contribution 3 accordingly.

**Recommended approach.** Option B is a 30-minute text edit and should be done regardless. Option A is a significant implementation effort (KFAC requires per-layer Kronecker factors) but would be a strong addition.

**Effort.** Option B: trivial. Option A: 3-5 days implementation + 1 day eval.
