# D Milestone: Production Inference & Community-Standard Evaluation

**Date:** 2026-03-27
**Status:** PROPOSED
**Prerequisite:** C milestone (all sub-milestones complete)
**Context:** The 2x2 matrix is done. Results are strong. But two critical gaps block a publishable paper:
1. **No community-standard metrics.** MI ratio is custom. Reviewers will demand AUROC.
2. **No production inference path.** N=20 MC passes per token is unusable. Need LoRA-only MC with shared base computation.

---

## The Two Goals

### Goal 1: Community-Standard Evaluation Framework

Retrofit all existing methods (A0-C4) with metrics the community actually uses. This transforms "MI ratio 1.53x" into "AUROC 0.93, FPR@95 = 12%."

### Goal 2: Production Inference Mechanism

Implement and benchmark the efficient LoRA-only MC path. Prove that epistemic uncertainty estimation is deployable with <10% overhead.

**Together, these produce the paper's Section 5 (Practical Deployment) and upgrade all results tables to publication standard.**

---

## Why MI Ratio Is Not Enough

MI ratio = $\frac{\text{mean}(\text{MI}_{\text{OOD}})}{\text{mean}(\text{MI}_{\text{ID}})}$

This compares means. It says nothing about distributional separation:

- **Method A:** MI_ID = 0.10 +/- 0.05, MI_OOD = 0.14 +/- 0.05. MI ratio = 1.40x. Distributions overlap heavily. AUROC ~ 0.70.
- **Method B:** MI_ID = 0.10 +/- 0.01, MI_OOD = 0.12 +/- 0.01. MI ratio = 1.20x. Distributions barely overlap. AUROC ~ 0.95.

Method B is vastly better for deployment despite lower MI ratio. The scaling inversion finding (C3 > C1) might look even more dramatic -- or different -- in AUROC terms. We don't know until we compute it.

---

## Part 1: Community-Standard Metrics

### 1.1 OOD Detection Metrics (The "Big Three")

Treat OOD detection as binary classification: each test sequence gets an uncertainty score, threshold sweeps produce curves.

**AUROC** (Area Under ROC Curve)
- Probability that a random OOD sample scores higher uncertainty than a random ID sample.
- Threshold-free. Range: 0.5 (random) to 1.0 (perfect).
- **The single most expected metric.** Used by: Laplace Redux (NeurIPS 2021), ICLA (WACV 2025), OpenOOD benchmark, LM-Polygraph (TACL 2025).

**FPR@95TPR** (False Positive Rate at 95% True Positive Rate)
- At the threshold where 95% of OOD is detected, what fraction of ID is falsely flagged?
- Range: 0% (perfect) to 100%. Lower is better.
- **Production-critical.** Answers: "If I catch 95% of OOD, how much false alarm?"
- Used by: ICLA, OpenOOD, most OOD detection papers since Hendrycks & Gimpel (2017).

**AUPRC** (Area Under Precision-Recall Curve)
- Like AUROC but accounts for class imbalance.
- Report AUPR-Out (OOD as positive class).
- Used by: OpenOOD. Standard completeness metric.

### 1.2 Calibration Metrics

**ECE** (Expected Calibration Error)
- Binned |accuracy - confidence| weighted by bin size. M=15 bins.
- Shows whether predicted probabilities are trustworthy.
- **Used by: BLoB, TFB, Laplace-LoRA, ScalaBL, Laplace Redux.** All papers in our comparison space.

**NLL** (Negative Log-Likelihood)
- Already computed (perplexity = exp(NLL)). Report raw NLL alongside perplexity.
- Proper scoring rule. Shows Bayesian model averaging benefit.

**Brier Score**
- $\frac{1}{N} \sum (1 - 2 p(y_{\text{true}}) + \sum_k p_k^2)$
- Proper scoring rule. Decomposes into calibration + refinement.
- **Used by: BLoB, TFB.** Aligns directly with those papers.

### 1.3 Selective Prediction

**Risk-Coverage Curve & AURC**
- Sort sequences by uncertainty (descending). Progressively remove the most uncertain.
- At each coverage level (fraction retained), compute error rate.
- AURC = area under curve. Lower is better.
- **The most production-relevant framing.** Answers: "If I only serve the 80% of inputs I'm most confident about, what quality do I get?"
- Used by: LM-Polygraph (TACL 2025), selective prediction literature.

### 1.4 Uncertainty Scores to Evaluate

For each method, compute all metrics using multiple uncertainty scores:

| Score | Definition | Source |
|-------|-----------|--------|
| MI (mutual information) | $H[\bar{p}] - \frac{1}{N}\sum H[p^{(i)}]$ | Primary. Our main metric. |
| Predictive entropy | $H[\bar{p}]$ | Total uncertainty (epistemic + aleatoric). |
| Max softmax probability | $\max_k \bar{p}_k$ (negated for "uncertainty") | Simplest baseline. Hendrycks & Gimpel (2017). |

This lets us benchmark MI against simpler alternatives. If MI beats MaxProb on AUROC, that proves the Bayesian treatment adds value beyond softmax confidence.

### 1.5 Granularity: Token-Level vs Sequence-Level

**Token-level:** Per-position MI, entropy, max-prob. Fine-grained.

**Sequence-level (for OOD detection):** Aggregate token scores into one number per sequence:
- Mean across positions (standard)
- Max across positions (flags worst-case tokens)
- Proportion above threshold (fraction of "uncertain" tokens)

Report sequence-level for AUROC/FPR95/AUPRC (one score per sequence needed).
Report token-level for selective prediction and qualitative analysis.

---

## Part 2: Production Inference Mechanism

### 2.1 Mean-Weights Forward Pass

For generation: use $\mu$ weights (no sampling). Deterministic, same speed as standard inference.

**Implementation:** `BayesianLinear` and `BLoBLoRALinear` get a `use_mean` mode that skips reparameterization and uses $\mu$ directly. No KL computation.

**What it proves:** Generation quality doesn't degrade when using mean weights (perplexity should be close to or better than sampled).

### 2.2 LoRA-Only MC Sampling

For uncertainty estimation: fix base model weights, sample only LoRA parameters.

**Key insight:** Base model forward pass (99%+ of parameters) is deterministic and computed once. Only the LoRA delta ($\Delta W = BA$ where $B$ or $A$ has Bayesian parameters) varies between MC samples.

**Architecture for efficient LoRA-only MC:**

```
Input sequence
    |
    v
[Base model forward: deterministic, computed ONCE]
    |
    v
Hidden states at each LoRA injection point (cached)
    |
    +---> LoRA sample 1 --> logits_1
    +---> LoRA sample 2 --> logits_2
    +---> ...
    +---> LoRA sample N --> logits_N
    |
    v
Aggregate: MI, entropy, AUROC scores
```

**Critical detail:** This only works cleanly when attention Q/K/V are NOT Bayesian (true for BLoB LoRA on FFN). The residual stream diverges after each Bayesian FFN, so hidden states at layer $l+1$ depend on the LoRA sample at layer $l$. Therefore the "compute base once" optimization applies **per-layer** -- within each layer, the base linear is shared, but the full forward pass must run N times. The savings come from the base weights not needing sampling/reparameterization.

**More aggressive optimization (LoRA-only, base frozen):**
If LoRA is only on FFN (our C3 setup), and attention is deterministic:
1. Run full forward pass N times, but base `nn.Linear` layers use cached weights (no sampling overhead)
2. Only `BLoBLoRALinear` layers sample new weights per pass
3. Savings: no reparameterization for 99.8% of parameters, reduced memory (no sigma storage for base)

### 2.3 What to Benchmark

**Timing measurements** (GPU, RTX 4070, sequence length 512, batch size 1 and 8):

| Configuration | Description |
|---------------|-------------|
| Deterministic | C0 checkpoint, single forward pass |
| Mean-weights | C3 checkpoint, $\mu$ weights, single pass |
| Full MC (N) | C1 checkpoint, all weights sampled, N passes |
| LoRA MC (N) | C3 checkpoint, LoRA-only sampling, N passes |
| TFB MC (N) | C4-TFB checkpoint, TFB sampling, N passes |

N values: 1, 3, 5, 10, 20

**Report:**
- Wall-clock latency per sequence (ms)
- Throughput (sequences/sec)
- Peak VRAM (MB)
- Overhead ratio vs deterministic baseline

### 2.4 Quality-Cost Tradeoff Curve

The central production question: **how few MC samples can you get away with?**

For each method and each N in {1, 2, 3, 5, 10, 20, 30}:
- Compute AUROC, FPR@95, AUPRC
- Plot AUROC vs N (quality-cost tradeoff curve)
- Find the "knee" -- the N where more samples stop helping

**Hypothesis:** For LoRA-only MC, N=3-5 should retain most of the signal. C3's strong MI ratio (1.53x) suggests the separation is large enough that even few samples detect it.

### 2.5 Comparison with Closed-Form Propagation (Stretch)

From `specs/practical-inference.md`, Proxy 1: $\text{trace}(\Sigma)$ at the final hidden layer as a scalar uncertainty score. Zero MC cost.

**If time permits:**
- Implement Proxy 1 for BLoB LoRA (low-rank variance propagation)
- Compute AUROC using trace($\Sigma$) as uncertainty score
- Compare with MC-based AUROC at matched compute budget

**This is a stretch goal.** LoRA-only MC with small N is the primary story.

---

## Part 3: Retroactive Evaluation

Apply the full metrics suite to **all existing checkpoints** (A0-C4). This upgrades every row in `report.md`.

### 3.1 Updated Results Table (Target)

```
| Method          | MI Ratio | AUROC | FPR@95 | AUPRC | ECE   | Brier | NLL  |
|-----------------|----------|-------|--------|-------|-------|-------|------|
| A0 Deterministic|    --    |  0.50 |  100%  |  ...  |  ...  |  ...  | ...  |
| A2 Var. FFN     |  1.43x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |
| B1 Laplace FFN  |  1.00x   |  ~0.5 |  ~100% |  ...  |  ...  |  ...  | ...  |
| B2 BLoB LoRA    |  1.13x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |
| ...             |  ...     |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |
| C3 BLoB LoRA    |  1.53x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |
| C4-TFB          |  1.35x   |  ...  |  ...   |  ...  |  ...  |  ...  | ...  |
```

### 3.2 Selective Prediction Table (Target)

```
| Method          | AURC  | Acc@80% cov | Acc@90% cov |
|-----------------|-------|-------------|-------------|
| A0 Deterministic|  ...  |    ...      |    ...      |
| C3 BLoB LoRA    |  ...  |    ...      |    ...      |
| ...             |  ...  |    ...      |    ...      |
```

### 3.3 Production Inference Table (Target)

```
| Method         | N  | Latency (ms) | Overhead | AUROC | VRAM (MB) |
|----------------|----|-------------|----------|-------|-----------|
| Deterministic  | 1  |     ...     |   1.0x   |  0.50 |    ...    |
| LoRA MC (C3)   | 3  |     ...     |   ...x   |  ...  |    ...    |
| LoRA MC (C3)   | 5  |     ...     |   ...x   |  ...  |    ...    |
| LoRA MC (C3)   | 10 |     ...     |   ...x   |  ...  |    ...    |
| TFB (C4)       | 5  |     ...     |   ...x   |  ...  |    ...    |
| Full MC (C1)   | 5  |     ...     |   ...x   |  ...  |    ...    |
```

---

## Part 4: Implementation Plan

### D0: Metrics Framework
- Implement AUROC, FPR@95, AUPRC computation (use `sklearn.metrics`)
- Implement ECE (binned, M=15)
- Implement Brier score for LM (efficient: $1 - 2p(y) + \sum p_k^2$)
- Implement selective prediction (risk-coverage curve, AURC)
- All in `minigpt/uncertainty.py` as reusable functions
- Sequence-level aggregation: mean-MI, mean-entropy, mean-max-prob

### D1: Retroactive Evaluation
- Load each checkpoint (A0, A2, B1, B2, B3-TFB, B3-LAP, C0, C1, C2, C3, C4-TFB, C4-LAP)
- Run full metrics suite on test_id + test_ood
- Populate the tables above
- Script: `scripts/eval_all_checkpoints.py`

### D2: Mean-Weights Forward
- Add `use_mean` flag to `BayesianLinear.forward()` and `BLoBLoRALinear.forward()`
- Verify: perplexity with mean weights ~ perplexity with N=20 MC average
- Context manager: `with mean_weights(model):` for clean API

### D3: Production Benchmark
- Implement LoRA-only MC path (sample only LoRA params, base frozen)
- Wall-clock benchmarking script: `scripts/benchmark_inference.py`
- Measure: latency, throughput, VRAM for each configuration x N
- Quality-cost tradeoff: AUROC vs N curve

### D4: Paper Tables
- Generate final publication-ready tables
- Comparison: full MC vs LoRA-only MC vs TFB at matched compute
- Key number: "LoRA-only MC with N=5 adds X% overhead and achieves Y AUROC"

---

## Success Criteria

| Sub-milestone | Gate |
|---------------|------|
| D0 | All metrics implemented, tested, produce correct output on synthetic data |
| D1 | All checkpoints evaluated, tables populated, MI ratio correlates with AUROC ranking |
| D2 | Mean-weights perplexity within 5% of MC-averaged perplexity |
| D3 | LoRA-only MC (N=5) overhead < 2x vs deterministic; AUROC > 0.80 for C3 |
| D4 | Three publication-ready tables in `report.md` |

---

## What This Does NOT Cover

- Closed-form uncertainty propagation (Proxy 2/3 from practical-inference.md) -- stretch goal only
- New Bayesian methods or architectures
- Scaling beyond 76M params
- Conference submission logistics

---

## References

- Hendrycks & Gimpel (2017) — "A Baseline for Detecting Misclassified and Out-of-Distribution Examples." Introduced max-softmax-probability OOD baseline and FPR@95TPR.
- OpenOOD v1.5 — Standard benchmark suite. Reports AUROC, AUPRC, FPR@95.
- LM-Polygraph (TACL 2025) — UQ benchmark for NLP. Uses rejection-verification curves (= selective prediction).
- BLoB (NeurIPS 2024) — Reports ECE, NLL, Brier on LLaMA-2-7B.
- TFB (NeurIPS 2025) — Reports ECE, NLL, Brier. Same protocol as BLoB.
- Laplace Redux (NeurIPS 2021) — Reports AUROC, ECE, NLL on image classification.
- ICLA (WACV 2025) — Reports AUROC, FPR@95, ECE on image classification.
- Laplace-LoRA (ICLR 2024) — Reports ECE, NLL on LLaMA-2-7B.
