# Agentic Pipeline Guide

`experiments/c_pipeline.py` — autonomous HP optimization pipeline for the C milestone (16L scaled replication).

---

## Architecture

16L / 8H / 512d (~76.3M params). Same `model.py`, config-only changes. Weight tying preserved.
GPU: bs=16/accum=2 for full-weight (~50% VRAM), bs=32 for LoRA (~73% VRAM). Flipout not needed.

## Pipeline Design

State machine: CONFIGURE -> RUN -> ANALYZE -> DECIDE. On failure, invokes LLM agent (Claude/Codex via Provider pattern as subprocess) to diagnose problems and propose HP adjustments.

Components:
- `pipeline_runner.py` — Generic engine: `PipelineRunnerBase`, `MilestonePolicy`, `RuntimeHooks`. State persistence, budget tracking, OOM/divergence recovery, agent prompt construction, mechanical fallback.
- `c_milestones.py` — C-specific: config templates (C0-C4), gate functions (`check_gate`), tunable knobs (`TUNABLE_KNOBS`), comparison report, helper predicates.
- `c_pipeline.py` — CLI wiring: providers (`ClaudeProvider`, `CodexProvider`), hooks (`_prepare_model`, `_posthoc_fit`, `_sigma_std_extractor`), entry point.
- `agent_briefing.md` — HP tuning playbook injected into every agent prompt.

## Key Features

- Full AGENTS.md + briefing injected into agent prompt (no file reading needed).
- Provider pattern: `ClaudeProvider` -> `claude -p -` (stdin), `CodexProvider` -> `codex exec`.
- Patience early-stop: `patience_evals=10`, `min_delta=0.001` (0.1% relative).
- NaN detection: checked every step + every eval interval.
- Diagnostic summary: gap analysis, step efficiency, convergence analysis, run-over-run deltas.
- Post-hoc support: `posthoc_fit_fn` hook dispatches Laplace/TFB for steps=0 milestones.
- BLoB->DeterministicLoRA conversion in `_prepare_model`: maps `lora_A_mu`->`lora_A` for C4 post-hoc milestones.
- Record-only milestones (C2, C4_TFB, C4_LAP): `max_runs=1`, no agent retry.

## CLI

```bash
python experiments/c_pipeline.py --milestone c0 --provider claude --provider-model sonnet
python experiments/c_pipeline.py --compare
```

Options: `--milestone {c0|c1|c2|c3|c4_tfb|c4_lap}`, `--provider {claude|codex}`, `--provider-model`, `--max-runs`, `--compare`, `--dry-run`, `--no-agent`, `--no-mlflow`, `--state-dir`, `--initial-agent`.

## Template HPs

**C0:** lr=6e-4, warmup=4000, steps=100K, bs=16, accum=2, dropout=0.1.

**C1:** Same as C0 + `bayes_ffn: {enabled: True, init_rho: -2.0, prior_std: 1.0}`, kl_weight=0.1 (reduced from 0.2 — KL pressure scales with 64x more Bayesian params at 16L).

**C3 (BLoB LoRA):** lr=3e-4, warmup=500, steps=10K, bs=32, accum=1, kl_weight=1.0, weight_decay=0.0. LoRA: rank=16, alpha=32, target=ffn, prior_std=0.2, init_g=0.1. Data: HackerNews (ID).
