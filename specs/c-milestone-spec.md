# C Milestone — Scaled Replication Specification

## 1. Goal

Validate whether the 4L miniGPT findings transfer to a larger model. This is the central contribution of the comparison paper: a 2x2 matrix (variational vs post-hoc) x (full weights vs LoRA) at **two scales** on the **same evaluation protocol**.

### Key Questions

1. Does variational full-weight (A2) remain the gold standard at scale?
2. Does BLoB LoRA's gap vs full-weight narrow at 16L? (Practical implication for large model deployment.)
3. Does diagonal Laplace still fail? (Expected yes — confirms fundamental limitation.)
4. Does TFB maintain its performance relative to BLoB?
5. Does A3 (FFN+AttnV) still underperform A2, or does it benefit from depth?

### Paper Structure

| | 4L/4H/256d (Table 1) | 16L/8H/512d (Table 2) |
|---|---|---|
| A2 (variational, full) | **1.43x** | ? |
| B2 (variational, LoRA) | **1.13x** | ? |
| B3-TFB (post-hoc, LoRA) | **1.10x** | ? |
| B1 (post-hoc, full) | **1.00x** | ? |
| B3-LAP (post-hoc, LoRA) | **1.00x** | ? |

---

## 2. Model Architecture

**16L / 8H / 512d miniGPT** (~76M parameters).

| Parameter | 4L (current) | 16L (C milestone) |
|---|---|---|
| `n_layer` | 4 | 16 |
| `n_head` | 4 | 8 |
| `n_embd` | 256 | 512 |
| `block_size` | 256 | 256 |
| `dropout` | 0.2 | 0.2 |
| `vocab_size` | 50257 | 50257 |
| **Total params** | ~16M | ~76M |
| **FFN Bayesian params** (A2) | 4.2M | ~33.6M |

Architecture is identical (same `model.py`), only config changes. No new layers, no new tricks. Weight tying preserved when head is deterministic.

### GPU Fit (RTX 4070, 12GB) — PROFILED

Actual measurements from `python scripts/profile_c_gpu.py` (AMP, seq=256):

| Variant | Params | Best batch | Peak VRAM | VRAM% | Tok/sec | Grad accum |
|---|---|---|---|---|---|---|
| Deterministic | 76.3M | 16 | 5,653 MB | 46% | 35,584 | 2 |
| Bayesian FFN | 109.9M | 16 | 6,422 MB | 52% | 28,420 | 2 |
| Bayesian FFN+AttnV | 114.1M | 16 | 6,518 MB | 53% | 26,468 | 2 |
| BLoB LoRA r=16 | 78.3M (2.0M trainable) | 32 | 9,021 MB | 73% | 41,869 | 1 |
| Det LoRA r=16 | 77.6M (1.3M trainable) | 32 | 9,006 MB | 73% | 46,974 | 1 |

**Key findings:**
- batch_size=16 fits all full-weight variants with ~50% headroom
- LoRA variants fit at batch_size=32 with no gradient accumulation
- **Flipout NOT needed** — batch_size=16 provides sufficient gradient diversity
- Gradient checkpointing NOT needed — all variants well under 12GB at optimal batch
- batch_size=32 for full-weight technically runs but throughput drops ~35% (memory pressure)

---

## 3. Dataset

### Problem with AG News at Scale

AG News has ~5M total tokens across 4 categories. At 4L (16M params), the ratio is 0.15 tokens/param — already very low. At 76M params, this ratio drops to 0.07 — structurally inadequate.

### Solution: The Pile (domain-split)

Use **The Pile** (uncopyrighted version) with domain-based ID/OOD splits. Each document has a `meta.pile_set_name` label identifying its source domain.

**Available via:** `ArmelR/the-pile-splitted` on HuggingFace (per-domain subsets, no need to download 800GB).

### Domain Assignment

| Role | Domain | Source | Est. Size | Why |
|---|---|---|---|---|
| **ID train** | Wikipedia | `en_wikipedia` subset | ~6.4 GiB (~1.5B tokens) | Encyclopedic, factual, clean prose |
| **ID train** | StackExchange | `stack_exchange` subset | ~32 GiB (~7B tokens) | Technical Q&A, well-structured |
| **Fine-tune** (LoRA) | HackerNews | `hacker_news` subset | ~3.7 GiB (~850M tokens) | Tech-adjacent but different style |
| **OOD eval** | ArXiv | `arxiv` subset | ~56 GiB (~13B tokens) | Scientific/mathematical |
| **OOD eval** | FreeLaw | `free_law` subset | ~51 GiB (~12B tokens) | Legal text |
| **OOD eval** | PubMed | `pubmed_abstracts` subset | ~19 GiB (~4.5B tokens) | Biomedical |

### Subsample Strategy

We do **not** need the full Pile. Subsample to manageable sizes:

| Split | Source domain(s) | Target tokens | Purpose |
|---|---|---|---|
| **train** | Wikipedia + StackExchange | ~200M tokens (100M each) | Base LM training |
| **val** | Wikipedia + StackExchange (held-out) | ~10M tokens | Validation during training |
| **test_id** | Wikipedia + StackExchange (held-out) | ~10M tokens | ID perplexity + MI eval |
| **test_ood_arxiv** | ArXiv | ~10M tokens | OOD eval — scientific |
| **test_ood_law** | FreeLaw | ~10M tokens | OOD eval — legal |
| **test_ood_med** | PubMed | ~10M tokens | OOD eval — biomedical |

For LoRA experiments (C3, C4):

| Split | Source | Target tokens | Purpose |
|---|---|---|---|
| **pretrain** | Wikipedia | ~100M tokens | Base model pretraining |
| **lora_train** | HackerNews | ~20M tokens | LoRA fine-tune domain |
| **lora_val** | HackerNews (held-out) | ~2M tokens | LoRA validation |
| **test_id** | HackerNews (held-out) | ~5M tokens | ID perplexity + MI |
| **test_ood** | ArXiv + FreeLaw + PubMed | ~10M tokens each | OOD eval |

### Token/Param Ratio

- Training: 200M tokens / 76M params = **2.6 tokens/param** — much better than 4L (0.15)
- Still below Chinchilla-optimal (20x) but adequate with regularization (dropout=0.2, weight_decay=0.1)

### Implementation

Add `load_pile_domains()` to `minigpt/data.py`:
- Uses HuggingFace `datasets` library with streaming
- Downloads and caches subsets locally in `data/pile/`
- Returns `{"train", "val", "test_id", "test_ood_arxiv", "test_ood_law", "test_ood_med"}`
- BPE tokenization via tiktoken (same as AG News path)
- Configurable via `data.dataset: pile`, `data.pile_id_domains`, `data.pile_ood_domains`, `data.pile_tokens_per_domain`

---

## 4. Sub-Milestones

### C0 — Deterministic Baseline

Train a deterministic 16L miniGPT on Pile ID domains.

| Parameter | Value |
|---|---|
| Architecture | 16L/8H/512d |
| Dataset | Pile Wikipedia + StackExchange (~200M tokens) |
| Training steps | TBD (profile first — likely 50K–100K) |
| Batch size | TBD (`profile_c_gpu.py` output) |
| Gradient accumulation | TBD (effective batch ~32) |
| LR | 3e-4 (cosine decay to 1e-5) |
| Warmup | 2000 steps |
| AMP | Yes |

**Success gate:** test_id_ppl reasonable for the domain, test_ood_ppl clearly higher.

### C1 — Variational Full-Weight (A2-equivalent)

Bayesian FFN layers, trained with ELBO.

| Parameter | Starting value | Tuning range |
|---|---|---|
| `bayes_ffn.enabled` | True | — |
| `bayes_ffn.init_rho` | -2.0 | [-3.0, -1.0] |
| `bayes_ffn.prior_std` | 1.0 | [0.5, 2.0] |
| `train.kl_weight` | 0.2 | [0.05, 0.5] |

**Success gate:** MI ratio $> 1.2$x on at least one OOD domain.

### C2 — Post-hoc Laplace Full-Weight (B1-equivalent)

Fit diagonal Fisher on C0 checkpoint. No training.

| Parameter | Value |
|---|---|
| `laplace.selection_mode` | ffn |
| `laplace.damping` | 1.0 |
| `laplace.sample_scale` | 1.0 |
| `laplace.n_curvature_batches` | 30 |

**Expected result:** MI ratio ~1.00x (same failure as B1). Early-abort gate: if curvature mean $< 10^{-4}$ and MI ratio $< 1.02$x, log negative result and skip further runs.

### C3 — BLoB LoRA (B2-equivalent)

Variational LoRA fine-tuning.

**Phase 1:** Pretrain deterministic model on Wikipedia (~100M tokens). May reuse C0 checkpoint if domain split is compatible.

**Phase 2:** BLoB LoRA fine-tune on HackerNews.

| Parameter | Starting value | Tuning range |
|---|---|---|
| `lora.rank` | 16 | [8, 32, 64] |
| `lora.alpha` | 32.0 | 2 $\times$ rank |
| `lora.prior_std` | 0.2 | [0.1, 0.5] |
| `lora.init_g` | 0.1 | [0.05, 0.2] |
| `train.kl_weight` | 0.2 | [0.05, 0.5] |
| `train.lr` | 3e-4 | [1e-4, 5e-4] |
| `train.steps` | 10,000 | [5K, 20K] |

**Success gate:** MI ratio $> 1.05$x.

### C4 — Post-hoc LoRA (B3-equivalent)

**Phase 1:** Deterministic LoRA training on HackerNews (same base as C3 Phase 1).

**Phase 2a — TFB:**
| Parameter | Value |
|---|---|
| `tfb.epsilon` | 0.1 |
| `tfb.n_search_samples` | 10 |
| `tfb.n_anchor_batches` | 20 |

**Phase 2b — Laplace-LoRA:**
| Parameter | Value |
|---|---|
| `laplace.selection_mode` | lora |
| `laplace.damping` | 1.0 |
| `laplace.sample_scale` | 1.0 |

**Expected:** TFB works ($\geq 1.05$x), Laplace-LoRA fails (~1.00x).

---

## 5. Automated Research Pipeline

### Overview

A single CLI orchestrator that runs C0–C4 autonomously. Each sub-milestone follows: **CONFIGURE → RUN → ANALYZE → DECIDE (accept/retry/abort)**.

```
python experiments/c_pipeline.py --milestone c0
python experiments/c_pipeline.py --milestone c1
python experiments/c_pipeline.py --milestone c2
python experiments/c_pipeline.py --milestone c3
python experiments/c_pipeline.py --milestone c4
python experiments/c_pipeline.py --milestone all       # run everything
python experiments/c_pipeline.py --compare             # generate comparison table
python experiments/c_pipeline.py --resume              # resume from last incomplete
```

### Pipeline Phases

```
Phase 0: GPU Profiling
  └→ Run scripts/profile_c_gpu.py
  └→ Determine batch_size, gradient_accumulation
  └→ Store in .pipeline-state/gpu_profile.json

Phase 1: Dataset Preparation
  └→ Download Pile subsets (streaming, cached)
  └→ Tokenize and store in data/pile/
  └→ Verify token counts

Phase 2: C0 — Deterministic Baseline
  └→ Train 16L model on Pile ID domains
  └→ Evaluate perplexity on all splits
  └→ Gate: test_id_ppl < target

Phase 3: C1 — Variational Full-Weight
  └→ Train with ELBO (up to 4 runs with HP tuning)
  └→ Evaluate MI on all OOD domains
  └→ Gate: MI ratio > 1.2x on at least one OOD domain

Phase 4: C2 — Post-hoc Laplace
  └→ Fit curvature on C0 checkpoint
  └→ Evaluate MI
  └→ Gate: early-abort if curvature flat

Phase 5: C3 — BLoB LoRA
  └→ Phase 5a: Pretrain base (or reuse C0)
  └→ Phase 5b: BLoB LoRA fine-tune (up to 4 runs)
  └→ Gate: MI ratio > 1.05x

Phase 6: C4 — Post-hoc LoRA
  └→ Phase 6a: Deterministic LoRA train
  └→ Phase 6b: TFB
  └→ Phase 6c: Laplace-LoRA
  └→ Gate: log results (TFB expected positive, Laplace expected negative)

Phase 7: Comparison
  └→ Generate cross-method comparison table
  └→ Compare 4L vs 16L results
  └→ Produce final summary report
```

### State Management

```
.pipeline-state/
  pipeline.json          # global state: current phase, start time, budget used
  gpu_profile.json       # Phase 0 output
  c0.json                # per-milestone: runs, configs, results, decision
  c1.json
  c2.json
  c3.json
  c4.json
  comparison.json        # final comparison table
  *.log                  # audit trail per milestone
```

### Guardrails

1. **Budget cap:** 48h total GPU time. Pipeline tracks cumulative training time.
2. **OOM detection:** If OOM, reduce batch_size by 2x and retry. Does not count against retry budget.
3. **Divergence detection:** If loss > 100 or NaN, kill run and retry with adjusted hyperparameters.
4. **Max retries:** 4 runs per method (1 initial + 3 retries).
5. **No code changes:** Pipeline only tunes hyperparameters. Code must be frozen before Phase 0.
6. **Human checkpoint:** After Phase 3 (C1), pipeline pauses for human review before proceeding to LoRA experiments.
7. **Idempotent:** Pipeline can be killed and resumed at any point. State is persisted to disk.

### Hyperparameter Tuning Strategy

Each retry adjusts one variable at a time based on diagnostic signals:

| Signal | Diagnosis | Action |
|---|---|---|
| Posteriors frozen (σ stuck at init) | init_rho too negative | Increase init_rho by 1.0 |
| σ too wide (uniform noise) | init_rho too high or prior_std too large | Decrease init_rho or prior_std |
| KL rising monotonically | kl_weight too low | Increase kl_weight by 1.5x |
| ID ppl much worse than C0 | KL pressure too high | Decrease kl_weight by 0.5x |
| MI ratio < 1.05x but σ healthy | Insufficient training | Increase steps by 1.5x |

---

## 6. Evaluation Protocol

### Metrics (same as 4L)

- **Primary:** MI ratio = MI(OOD) / MI(ID)
  - Computed per OOD domain: MI ratio (arxiv), MI ratio (law), MI ratio (med)
  - Aggregate: mean MI ratio across OOD domains
- **Secondary:** Flip rate, predictive entropy, per-token MI
- **Perplexity:** test_id_ppl, test_ood_ppl (per domain)
- **Sigma stats:** mean, std, min, max, percentiles
- **KL trajectory:** final KL, direction (increasing/decreasing/stable)

### MC Sampling

- $N = 20$ forward passes (same as 4L)
- Temperature = 0 (all stochasticity from Bayesian weights)
- Batch elements processed one at a time during MC sampling (memory-safe)

### Qualitative Evaluation

- Select 5 text samples per domain
- Score per-token MI
- Report domain-wise qualitative MI ratios

### Cross-Scale Comparison

After all C sub-milestones complete:

| Method | 4L MI ratio | 16L MI ratio | Direction |
|---|---|---|---|
| A2 (variational, full) | 1.43x | ? | ↑ / ↓ / ≈ |
| B2 (variational, LoRA) | 1.13x | ? | ↑ / ↓ / ≈ |
| B3-TFB (post-hoc, LoRA) | 1.10x | ? | ↑ / ↓ / ≈ |
| B1 (post-hoc, full) | 1.00x | ? | ≈ (expected) |
| B3-LAP (post-hoc, LoRA) | 1.00x | ? | ≈ (expected) |

Key analysis questions:
1. Does relative ordering change?
2. Does LoRA gap vs full-weight narrow at scale?
3. Do absolute MI values increase (more capacity → more uncertainty)?
4. Does multiple OOD domains reveal method-dependent sensitivity?

---

## 7. New Code Required

### Before Pipeline (BDD → TDD → Code)

1. **`minigpt/data.py` — Pile loader**
   - `load_pile_domains(cfg, tokenizer)` → `{"train", "val", "test_id", "test_ood_*"}`
   - Streaming download from HuggingFace
   - Local caching in `data/pile/`
   - Configurable domains and token counts

2. **`experiments/c_pipeline.py` — Pipeline orchestrator**
   - CLI with `--milestone`, `--resume`, `--compare`
   - State management (JSON files in `.pipeline-state/`)
   - Per-milestone logic: configure → run → analyze → decide
   - Calls existing training/evaluation infrastructure
   - HP tuning logic based on diagnostic signals

3. ~~**Flipout**~~ — **NOT NEEDED**. GPU profiling confirmed batch_size=16 for all
   full-weight variants. Standard reparameterization trick is sufficient at this batch size.

### Reused As-Is

- `minigpt/model.py` — architecture scales via config
- `minigpt/layers.py` — BayesianLinear is param-agnostic
- `minigpt/train.py` — training loop handles gradient accumulation, AMP
- `minigpt/uncertainty.py` — MC sampling works at any scale
- `minigpt/laplace.py` — curvature fitting is param-agnostic
- `minigpt/tfb.py` — SVD search works at any scale
- `minigpt/lora.py` — LoRA injection is param-agnostic
- `experiments/eval_utils.py` — evaluation utilities
- `experiments/mlflow_utils.py` — MLflow logging

---

## 8. Execution Plan

### Phase 0: Preparation — DONE (2026-03-13)
- [x] GPU profiling script (`scripts/profile_c_gpu.py`)
- [x] Run profiler → batch_size=16/accum=2 (full-weight), batch_size=32/accum=1 (LoRA)
- [x] Write C milestone spec (this document)
- [x] Dataset choice: The Pile (domain-split) via `ArmelR/the-pile-splitted`
- [x] Flipout: NOT NEEDED (batch_size=16 sufficient)
- [ ] Review and finalize pipeline design

### Phase 1: Implementation (BDD → TDD → Code)
- [x] BDD: Data loader spec (`specs/pile-data-loader-spec.md`)
- [x] TDD: Data loader tests (`tests/test_pile_data.py` — 31 tests, reviewed & frozen)
- [ ] Code: `load_pile_data()` implementation
- [ ] Code: `load_pile_domains()` implementation
- [ ] BDD: Pipeline orchestrator spec
- [ ] TDD: Pipeline tests
- [ ] Code: `c_pipeline.py` implementation
- [ ] Configs: `configs/c0_baseline.yaml` through `configs/c4_post_hoc_lora.yaml`

### Phase 2: Execution
- [ ] Run C0 (deterministic baseline)
- [ ] Run C1 (variational full-weight)
- [ ] Human checkpoint: review C0/C1 results
- [ ] Run C2 (post-hoc Laplace)
- [ ] Run C3 (BLoB LoRA)
- [ ] Run C4 (post-hoc LoRA: TFB + Laplace)

### Phase 3: Analysis
- [ ] Cross-method comparison at 16L
- [ ] Cross-scale comparison (4L vs 16L)
- [ ] Final report / paper draft

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 76M model doesn't converge on 200M tokens | Low | High | Increase data (Pile has unlimited supply), increase regularization |
| Variational training diverges at 16L | Medium | High | Flipout, lower kl_weight, gradient clipping (already in place) |
| GPU OOM at batch_size=8 | Low | Low | Profile script determines actual limits; accumulation handles any batch_size |
| Pile download slow/fails | Low | Medium | Streaming mode; local cache; fallback to SlimPajama (Apache 2.0) |
| Pipeline state corruption | Low | Medium | JSON state files; each run logged to MLflow independently |
| All methods show 1.00x at 16L | Low | High | Would be a valid (if disappointing) result; still publishable as negative |

---

## 10. Timeline Estimate

Not providing time estimates per AGENTS.md guidelines. The execution is sequential and GPU-bound. The pipeline will track cumulative GPU hours and compare against the 48h budget cap.
