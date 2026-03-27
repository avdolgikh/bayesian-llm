# Metrics Guide

Plain-English explanations of every metric used in this project.

---

## Epistemic Uncertainty Metrics

**MI — Mutual Information**

The model runs N forward passes with different weight samples. MI = how much the predictions disagree across those samples. High MI = "the weights themselves are uncertain about this input." Formally: total uncertainty minus aleatoric uncertainty. This isolates the part of uncertainty that comes from not knowing the right weights — the epistemic part.

**MI Ratio**

MI_OOD / MI_ID. "Is the model more epistemically uncertain on out-of-distribution inputs than in-distribution ones?" Values >1.0 mean the method detects OOD. Our custom summary metric — useful but non-standard (compares means, ignores variance).

**Predictive Entropy**

Total uncertainty in the averaged prediction. Combines both epistemic (weight) uncertainty and aleatoric (inherent data) uncertainty. A token that's genuinely ambiguous in context will have high predictive entropy even with a deterministic model.

**Expected Entropy**

Average uncertainty of individual weight samples. This is the aleatoric part — the uncertainty that remains even if you knew the exact right weights. Subtracted from predictive entropy to get MI.

**Flip Rate**

Fraction of MC samples where the top-1 predicted token differs from the mode. Intuitive: "how often does the model change its mind when you resample the weights?" 0 = always agrees. 1 = different answer every time.

---

## OOD Detection Metrics

**AUROC — Area Under ROC Curve**

You have a pile of ID sequences and a pile of OOD sequences. Each gets an uncertainty score. AUROC = "if I pick one random ID and one random OOD, what's the probability the OOD one scores higher?" 1.0 = perfect separation. 0.5 = coin flip. The single most reported metric in OOD detection papers.

**FPR@95TPR — False Positive Rate at 95% True Positive Rate**

"I set my threshold so I catch 95% of OOD inputs. How many ID inputs do I accidentally flag?" 0% = no false alarms. 100% = flagging everything. This is the production metric — it tells you the cost of a safety net.

**AUPRC — Area Under Precision-Recall Curve**

Like AUROC but cares about class imbalance. "Of the inputs I flagged as OOD, how many actually were?" Matters when OOD is rare (1 in 1000) — AUROC can look great while precision is terrible.

---

## Calibration Metrics

**ECE — Expected Calibration Error**

"When the model says 80% confident, is it actually right 80% of the time?" Bin predictions by confidence, compare predicted vs actual accuracy in each bin. 0 = perfectly calibrated. High ECE = the model's probabilities are lying to you.

**NLL — Negative Log-Likelihood**

Cross-entropy loss on held-out data. Equivalent to -log(probability assigned to the correct token), averaged. Perplexity = exp(NLL). A proper scoring rule — it rewards models that assign high probability to truth and penalizes confident wrong answers harshly.

**Brier Score**

Squared error between the predicted probability vector and the one-hot truth. Like MSE for probabilities. 0 = perfect. Captures both calibration AND sharpness — a model that always says 50/50 gets punished even if it's calibrated.

---

## Selective Prediction Metrics

**AURC — Area Under Risk-Coverage Curve**

"If I let the model abstain on its most uncertain inputs, how does accuracy improve?" Sort by uncertainty, progressively remove the worst. Plot error rate vs fraction served. Low AURC = the model's uncertainty is useful for knowing when to say "I don't know." The most production-relevant metric.

---

## Perplexity

**Perplexity (PPL)**

exp(NLL). "How many tokens is the model effectively choosing between at each step?" A perplexity of 14 means the model is as uncertain as if it were uniformly guessing among 14 tokens. Lower = better. Our baseline: C0 deterministic = 14.3 (ID), A0 4-layer = 49.1 (ID).
