# Vital Unit Tests — ML Guardrails

**Date:** 2026-02-24
**Status:** Proposed

## Goal

Catch silent, hidden-nature ML bugs that corrupt results without raising errors. These are not software tests — they are sanity checks on mathematical and methodological invariants.

---

## P0 — Must have before Phase 2 conclusions

### Weight tying pointer equality

If broken silently, the model gets 50257x256 ~ 13M extra free parameters in the head. It will memorize instead of generalize, and the overfitting gets attributed to something else.

```python
assert model.lm_head.weight is model.transformer.token_emb.weight
```

### Category isolation (AG News)

If even one ID article leaks into the OOD tensor, the MI comparison (ID vs OOD) is compromised — OOD perplexity looks "slightly better than expected" and nobody notices.

- Verify category counts before tokenization: ID articles only from id_categories, OOD only from ood_categories
- Zero overlap between the two sets

### MI = 0 for deterministic model

THE Phase 2 cornerstone. N forward passes on a non-Bayesian model with temperature=0 produce identical outputs, so MI must be exactly 0. If not, the MI calculation has a bug and every conclusion built on it is wrong.

```python
# Run MI estimation on a deterministic model (bayes.enabled=False)
# Assert MI == 0.0
```

---

## P1 — Should have

### Split sizes sum to total

Off-by-one in `int(len * fraction)` can silently shift thousands of tokens between splits.

```python
# Given val_fraction=0.1, test_fraction=0.1:
# len(train) + len(val) + len(test_id) == len(all_tokens)
# Each split ~80%, ~10%, ~10% (within ±1 token of expected)
```

### Perplexity bounds at init

At random init, perplexity should be ~vocab_size (50257). If it's 150 instead, the loss computation is broken but training still runs and "converges." Always: perplexity >= 1.0 (mathematical lower bound, since perplexity = exp(cross_entropy_loss)).

```python
# model = MiniGPT(config)  # random init, no training
# ppl = compute_perplexity(model, data, ...)
# assert ppl >= 1.0
# assert ppl > 10000  # should be ~vocab_size at init
```

### Bayesian sampling produces variance

N forward passes with weight sampling must produce different logits. Complement to the MI=0 test: if sampling is broken, all passes are identical, MI is always 0, and you'd conclude "no epistemic uncertainty" when really the code just isn't sampling.

```python
# model with bayes.enabled=True
# logits_1 = model(x, sample_weights=True)
# logits_2 = model(x, sample_weights=True)
# assert not torch.allclose(logits_1, logits_2)
```

---

## P2 — Nice to have

### Reproducibility

Same config + same seed -> identical loss after N steps. Catches accidental non-determinism (dropout state, data sampling order). Matters because Phase 2 compares metrics across runs — non-reproducible runs make ID vs OOD comparisons noisy.

```python
# Run 2 steps with seed=42 twice -> identical loss values
```

### KL / MI non-negativity

Both are KL divergences — mathematically non-negative. A negative value means a numerical or implementation bug.

```python
# assert kl_loss >= 0
# assert mi >= 0
```

---

## What to skip

- **Training loop convergence** — too slow, too flaky for CI
- **Model architecture details** — PyTorch handles this
- **Config parsing** — low blast radius, errors are loud and immediate

---

## Summary table

| Priority | Test | Catches |
|---|---|---|
| P0 | Weight tying pointer equality | Silent 13M extra params, memorization |
| P0 | Category isolation (AG News) | Leaked ID data in OOD, false MI results |
| P0 | MI = 0 for deterministic | Broken MI calculation, all Phase 2 invalid |
| P1 | Split sizes sum to total | Off-by-one, silent data loss between splits |
| P1 | Perplexity bounds at init | Broken loss computation |
| P1 | Bayesian sampling produces variance | Dead sampling, MI always 0 |
| P2 | Reproducibility | Non-determinism across comparison runs |
| P2 | KL/MI non-negativity | Numerical / implementation bugs |
