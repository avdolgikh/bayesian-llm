# B2 R2: Revised BLoB Bayesian LoRA Plan

Date: 2026-03-11

## 1. Problem: Why B2 R1 Failed

B2 R1 (MLflow `e1f60bfc`) produced val ppl=3389, gibberish output, MI ratio=1.04x. The Bayesian
LoRA machinery functioned (KL decreased, sigmas moved), but the mean model was catastrophically
poor. Root cause: the pretrain base was unusable.

### 1.1 TinyShakespeare Overfitting

TinyShakespeare has ~304K BPE tokens. Our 16M-param miniGPT memorizes it completely:

| Run | Steps | Train ppl | Val ppl | Outcome |
|-----|-------|-----------|---------|---------|
| `9ae3327a` | 15,000 | **1.37** | **1,646** | Catastrophic overfit |
| `1fae7c40` | 1,000 | 65.7 | 161.0 | Best achievable (early stop) |

Best val loss occurs at step 800--1000. The model hits a val ppl floor of ~160 --- this is the
**structural ceiling** for 16M params on 304K tokens. No amount of tuning (dropout, LR, weight
decay) can fix this: the dataset is 50x too small for the model.

### 1.2 Domain Gap Is Unbridgeable

Val ppl=160 means the base model barely knows English. Rank-8 LoRA (122K params, 0.76% of base)
cannot bridge Shakespeare-to-news --- that requires rewriting most of the model's knowledge.
The LoRA learned tiny residuals on garbage representations, producing garbage outputs with
garbage uncertainty.

### 1.3 What B2 R1 Was NOT

B2 R1 is **not evidence that BLoB LoRA fails**. It is evidence that the experimental setup
(pretrain corpus, hyperparameters) was insufficient. The Bayesian machinery itself showed
correct behavior: KL decreased (172K $\to$ 137K), sigmas doubled from init (0.0025 $\to$ 0.0048),
MI was non-zero. The problem was upstream of the Bayesian layer.

## 2. Revised Pretrain Strategy

### 2.1 Why Not TinyShakespeare

304K tokens for 16M params = guaranteed overfitting. The best checkpoint is barely functional
(val ppl=160). Additionally, Shakespeare-to-news is a catastrophic domain shift that LoRA at
this scale cannot bridge.

### 2.2 Why Not AG News ALL (4 Categories)

Pretraining on all 4 AG News categories would contaminate the base model with OOD knowledge.
If the base model already knows Business and Sci/Tech:

$$\mathbf{y} = \underbrace{\mathbf{W}_0 \mathbf{x}}_{\text{confident on OOD}} + \underbrace{\frac{\alpha}{r}\mathbf{B}\hat{\mathbf{A}}\mathbf{x}}_{\text{uncertain on OOD}}$$

The base output **dominates** (LoRA is a small delta). MC samples of $\hat{\mathbf{A}}$ create
small perturbations on a confident base prediction. The MI signal is suppressed because the
LoRA's uncertainty is a residual that doesn't matter --- the base already handles OOD well.

This fundamentally undermines the experiment: we'd be measuring uncertainty in an irrelevant
residual, not epistemic uncertainty about unseen domains.

### 2.3 Chosen Design: Category-Split Pretrain

**Pretrain on AG News category 1 (World) only.** LoRA fine-tune on category 2 (Sports). Evaluate
OOD on categories 3+4 (Business + Sci/Tech).

| Phase | Categories | Purpose |
|-------|-----------|---------|
| Pretrain (deterministic) | 1 = World | Base LM learns news language in one domain |
| LoRA fine-tune (BLoB) | 2 = Sports | Adapter specializes for a related but different domain |
| OOD eval | 3+4 = Business, Sci/Tech | Truly unseen by both base and adapter |

### 2.4 Why This Design Is Correct

**Principle 1: OOD must be truly unseen.** Neither the base model nor the LoRA adapter should
have trained on categories 3+4. If either component has seen OOD data, the uncertainty signal
is contaminated. In this design, the base trained on World only, and the LoRA trained on
Sports only. Business and Sci/Tech are completely novel to the entire model.

**Principle 2: LoRA must do genuine work.** If the base model already handles the fine-tune
domain well, LoRA learns near-zero residuals and its posterior has no informative structure.
World $\to$ Sports is a genuine domain shift within news: different vocabulary (players,
scores, leagues vs. diplomacy, politics, conflict), different sentence patterns, different
named entities. The LoRA has real adaptation work to do.

**Principle 3: The pretrain domain should provide useful but insufficient representations.**
World news shares general news language structure (article format, reporting style, sentence
complexity) with Sports but not the content. The base model provides a reasonable foundation
that the LoRA can build on --- unlike Shakespeare, which shares almost nothing with news.

**Principle 4: The uncertainty attribution is clean.** At OOD eval time:

- Base model: uncertain (never saw Business/Sci-Tech)
- LoRA adapter: uncertain (never saw Business/Sci-Tech)
- Both components contribute to MI $\Rightarrow$ strong, clean signal

At ID eval time (Sports):

- Base model: partial knowledge (World $\to$ Sports transfer)
- LoRA adapter: confident (trained on Sports)
- LoRA suppresses base uncertainty $\Rightarrow$ low MI

The **contrast** between these two cases is maximized, giving the best chance for a strong
MI ratio.

**Principle 5: This mirrors real-world pretrain $\to$ finetune.** In practice, you pretrain
on one corpus, fine-tune on your task, and want to detect OOD at inference. The pretrain domain
$\neq$ the fine-tune domain $\neq$ the OOD domain. This is exactly our setup.

### 2.5 Data Feasibility

AG News category 1 (World) contains ~32K articles. With `\n\n` joining and BPE tokenization,
this produces ~1.35M tokens. After val/test split (80/10/10), the train set is ~1.08M tokens.

Comparison to previous pretrains:

| Corpus | Train tokens | Model params | Ratio | Result |
|--------|-------------|-------------|-------|--------|
| TinyShakespeare | 243K | 16M | 1:66 | Overfits at step 800 |
| AG News cat 1 (World) | ~1.08M | 16M | 1:15 | Expected: reasonable convergence |
| AG News cats 1+2 (A0) | ~2.16M | 16M | 1:7 | val ppl=49, no overfitting |

At 1:15 ratio, we expect more overfitting than A0 but far less than Shakespeare. With
dropout=0.2 and weight_decay=0.1, the model should reach val ppl in the 60--100 range ---
good enough for a functional base LM.

## 3. Revised Hyperparameters

### 3.1 Pretrain (Phase 1)

| Parameter | B2 R1 value | Revised | Rationale |
|-----------|-------------|---------|-----------|
| `data.dataset` | tinyshakespeare | **agnews** | 4x more tokens, same domain family as finetune |
| `data.id_categories` | n/a | **[1]** | World only |
| `data.ood_categories` | n/a | **[]** | No OOD for pretrain |
| `train.steps` | 1,000 | **15,000** | ~1M tokens, batch=32$\times$256=8K $\to$ ~114 epochs |
| `train.warmup_steps` | 100 | **1,000** | Standard 6.7% warmup |
| `train.eval_interval` | 200 | **1,000** | 15 eval points over 15K steps |
| `train.checkpoint_interval` | 100 | **5,000** | 3 checkpoints |
| `train.checkpoint_dir` | b2_pretrain | **b2_pretrain** | Keep |
| All other params | (same) | (same) | lr=3e-4, dropout=0.2, weight_decay=0.1 |

Monitor val loss curve. If overfitting appears before 15K steps, the best checkpoint (ELBO
criterion) will catch it automatically. Expect best val step around 5K--10K.

### 3.2 Finetune (Phase 2)

| Parameter | B2 R1 value | Revised | Rationale |
|-----------|-------------|---------|-----------|
| `data.id_categories` | [1, 2] | **[2]** | Sports only (LoRA trains on this) |
| `data.ood_categories` | [3, 4] | **[3, 4]** | Keep (Business + Sci/Tech) |
| `lora.rank` | 8 | **16** | More adaptation capacity for cross-category shift |
| `lora.alpha` | 16.0 | **32.0** | Keep alpha/rank=2 (standard LoRA scaling) |
| `lora.init_g` | 0.05 | **0.1** | Initial $\sigma = G^2 = 0.01$ (4x wider than R1's 0.0025) |
| `train.lr` | 1e-4 | **3e-4** | Standard LoRA LR; BLoB paper uses 2e-4; A-series used 3e-4 |
| `lora.prior_std` | 0.2 | **0.2** | Keep (BLoB default) |
| `train.kl_weight` | 0.2 | **0.2** | Keep (validated in A-series) |
| `train.steps` | 10,000 | **10,000** | Keep (review after first run) |
| `train.kl_annealing_steps` | 2,000 | **2,000** | Keep |
| `train.weight_decay` | 0.0 | **0.0** | Keep (BLoB convention: KL regularizes) |

#### Rationale for Each Change

**`lora.rank` 8 $\to$ 16:** World $\to$ Sports requires more adaptation capacity than
fine-tuning within the same domain. Rank 16 gives the LoRA ~245K params (vs 122K at rank 8),
still 17x fewer Bayesian params than A2's 4.2M.

**`lora.alpha` 16 $\to$ 32:** Maintains the standard `alpha/rank = 2.0` scaling factor.
This ensures the LoRA contribution scales consistently regardless of rank choice.

**`lora.init_g` 0.05 $\to$ 0.1:** In B2 R1, initial $\sigma = 0.05^2 = 0.0025$, and sigmas
barely doubled to 0.0048 by end of training. Starting at $\sigma = 0.1^2 = 0.01$ gives 4x
more room for posterior differentiation. The gradient of $\sigma = G^2$ w.r.t. $G$ is
$2G$, so larger $G$ also means stronger gradient flow for variance learning.

**`train.lr` 1e-4 $\to$ 3e-4:** Standard LoRA fine-tuning uses higher LR than full
fine-tuning (only adapter params update, so can afford more aggressive steps). 3e-4 matches
our A-series and is within the typical LoRA range of 2e-4 to 1e-3.

## 4. Expected Outcomes

### 4.1 Pretrain

- Val ppl: 60--100 (functional base LM on World news)
- Train ppl: 10--30 (some overfitting expected at 1:15 token ratio, but manageable)
- No catastrophic overfitting (val loss should plateau, not diverge)

### 4.2 Finetune

- Val ppl on Sports: 80--200 (reasonable given cross-category transfer)
- Test OOD ppl on Business+Sci/Tech: higher than ID (expected)
- MI ratio: target > 1.1x batch, > 1.2x qualitative (meaningful signal)
- Sigma stats: mean > 0.01, range showing differentiation (not collapsed)
- KL trajectory: decreasing or stable (healthy)

### 4.3 Comparison Targets

| Metric | A2 best | B1 (negative) | B2 R2 target |
|--------|---------|---------------|-------------|
| MI ratio (batch) | 1.43x | 1.00x | > 1.1x |
| MI ratio (qual) | 1.70x | 0.99x | > 1.2x |
| Bayesian params | 4.2M | 2.1M | ~164K |

Note: B2 MI ratios are not directly comparable to A-series because of the different
category split (A-series: ID=[1,2], OOD=[3,4]; B2 R2: ID=[2], OOD=[3,4]). The comparison
is directional --- does BLoB LoRA produce *any* OOD-discriminative uncertainty?

## 5. Execution Plan

### Step 1: Update Pretrain Config

Replace `b2_pretrain_shakespeare.yaml` with AG News category 1 pretrain. New file:
`configs/b2_pretrain_agnews.yaml`.

### Step 2: Update Finetune Config

Update `configs/b2_blob_agnews.yaml` with revised hyperparameters (rank=16, alpha=32,
lr=3e-4, init_g=0.1, id_categories=[2]).

### Step 3: Run Pretrain

```bash
python experiments/b2_blob_lora.py --phase pretrain \
    --pretrain-config configs/b2_pretrain_agnews.yaml
```

Verify: val ppl in 60--100 range, no catastrophic overfitting.

### Step 4: Run Finetune

```bash
python experiments/b2_blob_lora.py --phase finetune \
    --config configs/b2_blob_agnews.yaml
```

### Step 5: Analyze Results

- If MI ratio > 1.1x: B2 produces OOD-discriminative uncertainty. Document and compare to A2.
- If MI ratio = 1.00x: investigate (see Section 6).
- Log all results to MLflow with tag `b2-r2`.

## 6. If MI Is Still Flat

If B2 R2 still shows MI ratio ~1.0x with a functional base model, investigate in order:

1. **Sigma diagnostic.** Are posteriors collapsed ($\sigma \ll$ init)? $\to$ Increase init_g
   or decrease kl_weight.
2. **Flip rate diagnostic.** Is flip rate ~0? $\to$ LoRA variation is invisible at output level.
   Try larger rank (32) or higher init_g (0.2).
3. **LoRA contribution magnitude.** Are LoRA outputs negligible vs base? $\to$ Increase rank
   or alpha. Check `||LoRA(x)|| / ||W_0 x||` ratio.
4. **LR too low.** Are LoRA mean weights still near init? $\to$ Try lr=5e-4 or 1e-3.
5. **Fundamental limitation.** If rank-16 LoRA on 4L miniGPT genuinely cannot produce
   OOD-discriminative uncertainty, document as negative result. This would mean the LoRA
   subspace is too small to capture the relevant epistemic structure. Proceed to B3 and C.

## 7. Relation to Prior Work

### BLoB Paper Setup

The BLoB paper (NeurIPS 2024) fine-tunes pretrained LLMs (LLaMA-7B, RoBERTa) with
Bayesian LoRA on downstream NLP tasks. Our setup adapts this to miniGPT scale:

| | BLoB paper | Our B2 R2 |
|---|---|---|
| Base model | LLaMA-7B (pretrained) | miniGPT-16M (pretrained on AG News cat 1) |
| Fine-tune task | GLUE, CommonsenseQA | AG News cat 2 (Sports) |
| LoRA target | attention Q, V | FFN (fc, proj) |
| Eval | accuracy, calibration, OOD AUROC | MI ratio (ID vs OOD) |

The key difference is scale (16M vs 7B) and eval metric (MI ratio vs AUROC). Our setup
directly measures epistemic uncertainty via weight-space disagreement, while BLoB uses
predictive uncertainty for downstream tasks.
