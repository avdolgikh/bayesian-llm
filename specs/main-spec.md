# Bayesian LLM Experiments — Repo Starter Spec

## Goal

Build a minimal, hands-on experimental codebase for exploring **Bayesian methods in LLMs**. Priorities:

- Fast iteration on core Bayesian ideas (uncertainty, Bayesian layers / Bayesian adapters)
- Minimal boilerplate
- Runs locally on a single GPU (RTX 4070 class, ~10–12GB VRAM)

This repo is intentionally researchy + iterative and will evolve as experiments clarify what works.

---

## Canonical theory sources

- `docs/` contains **two PDF papers** (must-read references).
- The AI coding assistant should read and re-check these PDFs periodically and ensure implementations align with what they recommend/assume.

---

## Environment & tooling requirements

### Language & packaging

- **Python 3.12**
- Dependency management: **uv**
- Repo must have:
  - `pyproject.toml`
  - uv-managed local virtual environment (standard `uv` workflow)
- Keep setup friction low: prefer a simple “one command” dev setup.

---

## Milestone plan (order matters)

### Milestone A (first): From-scratch mini-GPT → Bayesian extensions

Primary path: implement a **small modern GPT-style decoder-only transformer** (few layers) that trains/runs locally.  
Then extend it with Bayesian components in a way that avoids rewrites.

Implementation intent:
- Keep model clean and modular so Bayesian variants are “swappable”:
  - deterministic baseline
  - Bayesian layers (some subset)
  - Bayesian adapters (later)

Key point: design with “Bayesian later” in mind so we don’t paint ourselves into a corner.

---

### Milestone B (later): Start from an existing small model → Bayesian LoRA

Secondary path after A stabilizes:

- Take an existing small GPT/LLM codebase (small enough to run locally)
- Add **Bayesian LoRA** (or similar Bayesian adapter approach)
- Investigate whether “Bayesian LoRA” exists in literature / prior art and reuse ideas where possible

---

## Framework decision policy (important)

You want to prefer **JAX**, but only if it doesn’t force lots of Bayesian boilerplate.

### Preferred if feasible: JAX stack

Use JAX **if** we can find a practical stack that provides:
- Probability distributions + reparameterization
- Correct gradients / autodiff integration
- Ready-to-use losses like negative log likelihood (NLL)
- Tools for Bayesian layers / variational inference without excessive custom math

If the JAX Bayesian stack is too clunky, **fall back to PyTorch**.

### Acceptable fallback: PyTorch stack

If PyTorch makes Bayesian layers/adapters significantly easier with less glue code, use PyTorch.

### Not preferred (but not forbidden): TensorFlow

TensorFlow Probability is strong, but TensorFlow ecosystem is less desirable for this repo unless it becomes clearly the best low-boilerplate option.

---

## Design constraints to reduce rework

When implementing Milestone A (mini-GPT), structure code so later Bayesian variants don’t require major refactors:

- Encapsulate layers so “weight-as-point-estimate” can become “weight-as-distribution”
- Keep forward pass and parameter definitions cleanly separated
- Make training loop flexible enough to support:
  - standard cross-entropy
  - ELBO / variational objectives
  - KL regularization terms (if used)

---

## Experiments (initial set)

### A0 — Deterministic baseline

- Minimal GPT training on a tiny dataset
- Verify:
  - loss decreases
  - generation works
  - runs on GPU within VRAM limits

### A1 — First Bayesian component (smallest viable step)

Pick the smallest Bayesian addition that teaches something quickly, e.g.:
- Bayesian output layer
- Bayesian embeddings
- Bayesian MLP weights
- Simple variational treatment for one module

Goal: confirm we can train and get reasonable uncertainty behavior.

### A2 — Broader Bayesian coverage

Extend Bayesian treatment to more of the transformer, but keep it small enough to iterate.

### B1 — Bayesian LoRA exploration (later)

- Identify prior work / papers / repos
- Implement Bayesian LoRA-style adapters on a small existing model

---

## What the AI coding assistant must do

1. Create repo skeleton aligned to this spec:
   - uv project, Python 3.12, clean structure
   - `docs/` stays source of truth for theory
2. Choose framework (JAX first, PyTorch if needed) based on:
   - minimal Bayesian boilerplate
   - availability of usable probabilistic tooling
3. Keep code minimal and hackable:
   - avoid “framework tourism”
   - avoid heavy abstractions early
4. When unsure, push back and propose:
   - the simplest next step
   - the smallest Bayesian component to implement first
   - tradeoffs of JAX vs PyTorch in this repo specifically

---

## Open questions (keep explicit)

- JAX vs PyTorch: which yields the least boilerplate for Bayesian layers/adapters?
- What are the best Bayesian/probabilistic libraries for the chosen framework?
- What Bayesian approach is best for first experiments?
  - variational inference (mean-field)
  - Laplace approximations
  - ensembles as a baseline comparison
  - Bayesian adapters vs full Bayesian weights
- Does “Bayesian LoRA” exist as known prior art, and what’s the cleanest way to implement it?

---

## Repo structure suggestion (lightweight)

- `docs/` — papers (source of truth)
- `src/`
  - `model/` — transformer + Bayesian variants
  - `train/` — training loop(s)
  - `data/` — tiny dataset loader(s)
  - `experiments/` — runnable experiment scripts (A0/A1/A2/B1)
- `README.md` — quickstart + how to run each experiment
- `NOTES.md` — running log of decisions, findings, and links to relevant papers
