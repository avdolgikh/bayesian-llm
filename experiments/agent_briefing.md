# Agent Briefing: Bayesian LLM HP Optimization

Full project history is in AGENTS.md.

## Project Goal
Estimate epistemic uncertainty in LLMs via Bayesian inference over weights.
Measure mutual information (MI) between weight samples: high MI ratio
(OOD/ID) = model knows what it doesn't know.

## Model Architecture
- MiniGPT: 16 layers, 8 heads, 512 dim, ~76M params
- Tokenizer: BPE via tiktoken (GPT-2, vocab_size=50257)
- Dataset: The Pile (domain-split). ID: wikipedia_en + stackexchange.
  OOD: arxiv, freelaw, pubmed_abstracts.

## Milestones
- C0: Deterministic baseline. Gate: test_id_ppl < 80 AND any OOD ppl > 2x ID.
- C1: Variational Bayesian FFN. Gate: mi_ratio_mean > 1.2.
- C3: BLoB LoRA (variational). Gate: mi_ratio_mean > 1.05.
- C2/C4: Post-hoc (Laplace/TFB). Record-only, single run.

## 4-Layer Reference Results (what to expect at 16L)
| Method              | MI ratio | Notes                          |
|---------------------|----------|--------------------------------|
| A0 deterministic    | N/A      | test_id_ppl=49, test_ood_ppl=540 |
| A2 variational FFN  | 1.43x    | Best. init_rho=-2, kl_weight=0.2 |
| B2 BLoB LoRA        | 1.13x    | rank=16, alpha=32, init_g=0.1  |
| B3 TFB post-hoc     | 1.10x    | sigma_q=0.013, SVD-structured  |
| B1 Laplace full     | 1.00x    | NEGATIVE. Diagonal curvature too flat |
| B3 Laplace LoRA     | 1.00x    | NEGATIVE. Same failure as B1   |

## HP Tuning Playbook

### Symptom: PPL >> 80 (e.g., >500)
- **Cause 1: Too few steps.** 76M params need 100K-200K steps to converge.
  Fix: train.steps=150000 or 200000.
- **Cause 2: LR too low during critical phase.** Default lr=3e-4 is reasonable
  but may need increase to 6e-4 for faster convergence.
  Fix: train.lr=6e-4 (max 1e-3).
- **Cause 3: Warmup too long relative to steps.** If warmup is >10% of
  total steps, the model spends too long at low LR.
  Fix: warmup = 4% of steps (e.g., 4000 for 100K steps).

### Symptom: Loss plateaus early (patience early-stop)
- **Cause 1: LR too low.** Model gets stuck in a flat region.
  Fix: increase train.lr by 2-3x.
- **Cause 2: Warmup too short.** LR jumps too fast, optimizer overshoots
  then gets stuck. Fix: increase warmup_steps.
- **Cause 3: Dropout too high.** Excessive regularization prevents learning.
  Fix: reduce model.dropout (try 0.1).

### Symptom: NaN loss
- **Cause 1: LR too high.** Gradients explode.
  Fix: halve train.lr.
- **Cause 2: Warmup too short.** LR ramps up too fast.
  Fix: increase warmup_steps to 4% of total steps.

### Symptom: MI ratio < 1.2 (C1 gate)
- **Cause 1: Posterior collapse (sigma_std < 0.01).** Posteriors stuck at init.
  Fix: increase model.bayes_ffn.init_rho (try -1.0 from -2.0).
- **Cause 2: KL too strong.** Posteriors crushed toward prior.
  Fix: decrease train.kl_weight (try 0.1 from 0.2).
- **Cause 3: Not enough training.** Posteriors need time to differentiate.
  Fix: increase train.steps.

### Symptom: MI ratio < 1.05 (C3/LoRA gate)
- **Cause 1: init_g too small.** LoRA sigma too constrained.
  Fix: increase lora.init_g (try 0.2 from 0.1).
- **Cause 2: Rank too low.** Not enough capacity for uncertainty.
  Fix: increase lora.rank (try 32 from 16).

## Key Pitfalls (learned from prior experiments)
1. warmup_steps=200 with steps=50000 = warmup is 0.4%, way too short -> NaN.
2. init_rho=-3 freezes posteriors (sigmoid gradient too small). Use -2 or -1.
3. kl_weight=0.01 causes posterior collapse. Use 0.2 as default.
4. Doubling steps without fixing warmup wastes hours on the same plateau.
5. Always change warmup proportionally when changing steps (~4%).

## Decision Rules
- If PPL > 500: focus on steps and LR, not regularization.
- If PPL 80-200: close to gate, fine-tune LR/dropout/warmup.
- If previous adjustment didn't help: try a DIFFERENT knob, not the same one.
- If two runs show same PPL: something structural is wrong (LR/warmup ratio).
- Prefer changing one knob at a time for clear signal, but if gap is >10x,
  change multiple knobs aggressively.
