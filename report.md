# Results

Comparative study of Bayesian methods for epistemic uncertainty in language models. Four methods in a 2×2 matrix — (variational vs post-hoc) × (full weights vs LoRA) — tested at two scales.

## 4-Layer Scale (AG News, 4L/4H/256d, ~16M params)

| Milestone | Method | Type | MI Ratio | Test ID PPL | Bayesian Params |
|-----------|--------|------|----------|-------------|-----------------|
| A0 | Deterministic baseline | — | — | 49.1 | 0 |
| A1 | Variational output head | Variational × Full | 1.36x | 56.3 | 25.7M |
| **A2** | **Variational FFN** | **Variational × Full** | **1.43x** | 53.5 | 4.2M |
| A3 | Variational FFN + Attn V | Variational × Full | <1.29x | 59.2 | 4.7M |
| B1 | Laplace FFN | Post-hoc × Full | 1.00x | 41.0 | 2.1M |
| B2 | BLoB LoRA | Variational × LoRA | 1.13x | 226.9 | 163K |
| B3-TFB | TFB LoRA | Post-hoc × LoRA | 1.10x | 224.6 | 82K |
| B3-LAP | Laplace LoRA | Post-hoc × LoRA | 1.00x | 224.6 | 82K |

## 16-Layer Scale (The Pile, 16L/8H/512d, ~76M params)

| Milestone | Method | Type | MI Ratio | Test ID PPL | Training Time |
|-----------|--------|------|----------|-------------|---------------|
| C0 | Deterministic baseline | — | — | 14.3 | 4.5 hrs |
| C1 | Variational FFN | Variational × Full | 1.32x | 21.9 | 3.3 hrs |
| C2 | Laplace FFN | Post-hoc × Full | 1.00x | 12.7 | 8s fit |
| **C3** | **BLoB LoRA** | **Variational × LoRA** | **1.53x** | 64.9 | 27 min |
| C4-TFB | TFB LoRA | Post-hoc × LoRA | 1.35x | 66.3 | 7 min fit |
| C4-LAP | Laplace LoRA | Post-hoc × LoRA | 1.00x | 65.4 | 17s fit |

## Extended Evaluation (D1)

16L C checkpoints evaluated with AUROC, FPR@95, AUPRC, ECE, Brier, NLL, AURC.
500 ID sequences, 500 OOD sequences, block_size=256, N=20 MC samples.

### OOD Detection (primary uncertainty score per method)

| Milestone | Method          | MI Ratio | AUROC | FPR@95 | AUPRC | ECE    | Brier | NLL  | AURC  |
|-----------|-----------------|----------|-------|--------|-------|--------|-------|------|-------|
| C0        | Deterministic   |       -- | 0.591 | 0.794  | 0.552 | 0.0224 | 0.606 | 2.79 | 0.4995 |
| C1        | Variational FFN |    1.32x | 0.876 | 0.500  | 0.870 | 0.0228 | 0.673 | 3.31 | 0.3470 |
| C2        | Laplace FFN     |    1.00x | 0.536 | 0.934  | 0.533 | 0.0329 | 1.000 | 9.10 | 0.9876 |
| C3        | BLoB LoRA       |    1.53x | 0.916 | 0.398  | 0.920 | 0.0436 | 0.658 | 3.06 | 0.3302 |
| C4-TFB    | TFB LoRA        |    1.35x | 0.917 | 0.384  | 0.918 | 0.0215 | 0.658 | 3.06 | 0.3373 |
| C4-LAP    | Laplace LoRA    |    1.00x | 0.494 | 0.956  | 0.495 | 0.0340 | 0.998 | 9.73 | 0.9634 |

C0 uses max-prob uncertainty (deterministic — MI=0). All others use MI.

### Uncertainty Score Comparison (AUROC)

| Milestone | MI AUROC | Pred. Entropy AUROC | Max-Prob AUROC |
|-----------|----------|---------------------|----------------|
| C0        |       -- |               0.545 |          0.591 |
| C1        |    0.876 |               0.506 |          0.548 |
| C2        |    0.536 |               0.474 |          0.494 |
| C3        |    0.916 |               0.532 |          0.569 |
| C4-TFB    |    0.917 |               0.553 |          0.589 |
| C4-LAP    |    0.494 |               0.499 |          0.486 |

## Cross-Scale Comparison

| Method | 4L MI Ratio | 16L MI Ratio | Scales? |
|--------|-------------|--------------|---------|
| Variational full (A2 / C1) | 1.43x | 1.32x | Slight decrease |
| BLoB LoRA (B2 / C3) | 1.13x | **1.53x** | Strong increase |
| TFB post-hoc LoRA (B3 / C4) | 1.10x | **1.35x** | Strong increase |
| Laplace full (B1 / C2) | 1.00x | 1.00x | Dead |
| Laplace LoRA (B3 / C4) | 1.00x | 1.00x | Dead |

## D2: Mean-Weights Inference Verification

Mean-weights (posterior mean $\mu$) forward pass produces deterministic inference with quality matching MC-averaged perplexity. Gate: <5% relative difference.

| Method | Mean-Wt PPL | MC-Avg PPL (N=20) | Relative Diff |
|--------|-------------|--------------------|--------------:|
| C1 Variational FFN | 24.48 | 24.41 | 0.29% |
| C3 BLoB LoRA | 16.12 | 16.78 | 3.93% |

**Production path:** Use $\mu$ weights for serving predictions (deterministic, no sampling overhead). Reserve MC sampling only for uncertainty estimation.

## D3: Production Inference Benchmarks

RTX 4070, sequence length 256, batch size 1, AMP fp16.

### Latency / VRAM / Throughput

| Method | N | Latency (ms) | Overhead | Throughput | VRAM (MB) |
|--------|---|-------------|----------|------------|-----------|
| C0 Deterministic | 1 | 8.1 | 1.00x | 123.2/s | 470 |
| C3 Mean-Weights | 1 | 14.8 | 1.82x | 67.6/s | 382 |
| C1 Full Var. MC | 1 | 14.0 | 1.73x | 71.2/s | 534 |
| C1 Full Var. MC | 5 | 71.0 | 8.75x | 14.1/s | 534 |
| C1 Full Var. MC | 20 | 286.9 | 35.35x | 3.5/s | 534 |
| C3 LoRA MC | 1 | 17.2 | 2.12x | 58.2/s | 382 |
| C3 LoRA MC | 3 | 50.5 | 6.22x | 19.8/s | 382 |
| C3 LoRA MC | 5 | 84.4 | 10.41x | 11.8/s | 382 |
| C3 LoRA MC | 20 | 315.6 | 38.89x | 3.2/s | 382 |
| C4-TFB MC | 1 | 21.2 | 2.62x | 47.1/s | 389 |
| C4-TFB MC | 5 | 133.0 | 16.39x | 7.5/s | 389 |
| C4-TFB MC | 20 | 468.6 | 57.76x | 2.1/s | 389 |

### AUROC vs N (Quality-Cost Tradeoff)

200 ID + 200 OOD sequences, block_size=256. MI as uncertainty score.

| Method | N=1 | N=3 | N=5 | N=10 | N=20 |
|--------|-----|-----|-----|------|------|
| C0 Deterministic (MaxProb) | 0.495 | — | — | — | — |
| C1 Full Var. MC | 0.500 | 0.850 | 0.855 | 0.866 | 0.869 |
| C3 LoRA MC | 0.500 | **0.861** | **0.879** | 0.880 | 0.888 |
| C4-TFB MC | 0.500 | 0.847 | 0.859 | 0.881 | 0.886 |

**N=3 is the knee.** Going from N=1→3 jumps AUROC from 0.50 to ~0.86. Further increases (N=5→20) add <3 points. For production: **N=3 at 50ms/seq with 382 MB VRAM** is the sweet spot.

## Key Findings

1. **Scaling inversion.** At 4L: full-weight variational (1.43x) > LoRA (1.13x). At 16L: reversed — LoRA (1.53x) > full-weight (1.32x). LoRA's rank-16 subspace constrains posteriors to meaningful directions rather than spreading uncertainty across all parameters.

2. **TFB (zero training) matches variational full-weight.** C4-TFB 1.35x ≈ C1 1.32x, but TFB requires zero gradient computation — only a 7-minute binary search on a trained checkpoint vs 3.3 hours of Bayesian training.

3. **Diagonal Laplace is dead for LM OOD detection.** Four independent experiments (B1, B3-LAP, C2, C4-LAP) all produce MI ratio 1.00x. The failure is fundamental: diagonal Fisher curvature is flat at convergence for well-trained language models. Neither increasing parameters (82K → 33.5M) nor scaling models (4L → 16L) helps.

4. **SVD-structured variance works where curvature-based fails.** Both TFB and Laplace are post-hoc and operate on LoRA. TFB succeeds (1.35x) because SVD of B captures the geometric structure of the LoRA subspace. Laplace fails (1.00x) because diagonal curvature at convergence carries no directional information.

5. **Post-hoc methods need subspace structure.** Post-hoc on full weights (Laplace) = 1.00x. Post-hoc on LoRA with SVD structure (TFB) = 1.35x. The LoRA subspace provides the geometric structure that makes post-hoc methods viable.

6. **N=3 MC samples capture most of the signal.** AUROC jumps from 0.50 (N=1) to ~0.86 (N=3) — 97% of the N=20 signal. Further samples add diminishing returns (<3 AUROC points from N=3→20). The cost is ~6x vs deterministic. LoRA MC uses 28% less VRAM than full variational MC (382 vs 534 MB) with equal or better AUROC at every N.

7. **Mean-weights inference is production-ready.** Posterior mean ($\mu$) perplexity matches MC-averaged perplexity within 4% (C1: 0.29%, C3: 3.93%). For serving predictions: use $\mu$ weights (deterministic, 1.8x overhead vs vanilla). For uncertainty scoring: use N=3 MC as a post-processing step on generated text.
