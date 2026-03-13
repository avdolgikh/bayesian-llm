# C Milestone: Automated Research Pipeline

## Goal

Fully automated ML research pipeline that scales the 4L comparative study (A0–A3, B1–B3) to 16L miniGPT, using Claude Code CLI as the reasoning agent for hyperparameter tuning and experiment interpretation.

**The human provides the spec. The machine does the rest.**

## Why Automation

At 4L scale, we ran experiments manually: inspect results → reason about what to change → edit config → rerun. This worked for 6 milestones over weeks. At 16L, the same suite has ~15-20 experiment runs across 7 methods. Manual iteration doesn't scale.

But this isn't a grid search problem. With a GPU budget of 3-5 runs per method, each run must count. The agent needs to **reason** — understand the papers, the math, the prior 4L results — to make informed decisions about what to try next. Optuna finds the best lr in a range. Claude Code decides whether Laplace needs KFAC instead of diagonal, or whether $\epsilon=0.1$ is too tight for TFB at this scale.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                          │
│              (Python script + bash)                      │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │  Claude   │    │  MLflow  │    │  Experiment       │  │
│  │  Code CLI │◄──►│  (sqlite)│◄───│  Runner (GPU)     │  │
│  │  (-p mode)│    │          │    │  (bare python)    │  │
│  └──────────┘    └──────────┘    └──────────────────┘  │
│       │                                    ▲             │
│       ▼                                    │             │
│  ┌──────────┐                              │             │
│  │  Config   │──────────────────────────────┘             │
│  │  (YAML)   │                                           │
│  └──────────┘                                            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  State Manager (.pipeline-state/)                 │   │
│  │  - Current experiment, iteration, history         │   │
│  │  - Resume after crash / GPU OOM / timeout         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Components

1. **Orchestrator** — Python script (`scripts/run_research_pipeline.py`). Drives the loop. No ML logic — pure coordination.
2. **Claude Code CLI** — Reasoning agent invoked via `claude -p`. Reads specs, papers, AGENTS.md, MLflow dumps. Outputs structured JSON (config YAML + reasoning).
3. **Experiment Runner** — Existing experiment scripts (`experiments/*.py`). Invoked with `python` (CUDA). Writes results to MLflow.
4. **MLflow** — Feedback interface. `scripts/dump_mlflow_run.py` extracts metrics for Claude Code to analyze.
5. **State Manager** — JSON state files for resumability. Tracks: current experiment, iteration count, run history, decisions made.

## Pipeline Loop

For each experiment in the suite:

```
INIT → CONFIGURE → RUN → ANALYZE → DECIDE
                    ▲                  │
                    │    ◄─ REVISE ◄───┤
                    │                  │
                    └──────────────────┘ (if retry)
                                       │
                                ACCEPT ─┘ (if done)
```

### Stage: INIT

Claude Code receives:
- The experiment spec (method description, paper reference, math)
- 4L results for this method (from AGENTS.md)
- 16L architecture constraints (GPU memory, batch size)
- Prior experiments at 16L already completed (comparison context)

Claude Code outputs:
- Initial YAML config with reasoning for each hyperparameter choice
- Expected outcome range (e.g., "MI ratio should be 1.3-1.6x based on 4L=1.43x")
- Key risks (e.g., "posterior collapse if init_rho too low at this scale")

### Stage: CONFIGURE

Orchestrator writes the YAML config to `configs/c_<method>.yaml`.

### Stage: RUN

Orchestrator invokes the experiment script:
```bash
python experiments/<script>.py --config configs/c_<method>.yaml
```

This is the only stage that uses GPU. All other stages are CPU-only (Claude Code reasoning).

Timeout: configurable per experiment (default 4h for training, 1h for post-hoc fitting).

### Stage: ANALYZE

Orchestrator dumps MLflow results:
```bash
python scripts/dump_mlflow_run.py latest > .pipeline-state/<method>_run<N>.txt
```

Claude Code receives:
- MLflow dump (metrics, params, artifacts)
- The config that was used
- History of all prior runs for this method
- The spec's acceptance criteria

Claude Code outputs structured JSON:
```json
{
  "decision": "accept" | "retry" | "abort",
  "analysis": "Free-text reasoning about what happened and why",
  "metrics_summary": {
    "mi_ratio_batch": 1.35,
    "mi_ratio_qual": 1.52,
    "test_id_ppl": 42.0,
    "key_observation": "Posteriors learned, sigma range healthy"
  },
  "next_config": { ... },  // only if decision=retry
  "abort_reason": "..."     // only if decision=abort
}
```

### Stage: DECIDE

- **accept**: Log final results, move to next experiment.
- **retry**: Apply config changes, go back to RUN. Decrement retry budget.
- **abort**: Method is fundamentally broken at this scale. Log negative result. Move on.

### Retry Budget

Each method gets a **max_retries** budget (default: 3). This means up to 4 total runs per method (1 initial + 3 retries). The budget forces the agent to make each run count — no blind grid search.

If retries exhausted without acceptance: log best result so far, flag for human review, continue to next method.

## Experiment Suite

### Phase 0: Architecture Profiling

Before any experiments, Claude Code profiles the 16L architecture:

```bash
python scripts/profile_gpu.py --n-layer 16 --n-head 8 --n-embd 512 --batch-size 8
```

This determines: max batch size, whether AMP is needed, memory headroom for Bayesian layers. Claude Code adjusts the architecture if needed (e.g., reduce n_embd from 512 to 384 if OOM).

### Phase 1: Deterministic Baseline (A0)

Train deterministic 16L miniGPT on AG News (cats 1+2 ID, cats 3+4 OOD). This is the anchor for all comparisons.

**Acceptance:** val loss converges, test_id_ppl < 60 (scaling from 4L's 49.1), no OOM.

### Phase 2: Variational Full-Weight Methods (A1, A2, A3)

Run in order A2 → A1 → A3 (A2 is the strongest signal, prioritize it):

| Method | Script | Key HPs to tune | 4L reference |
|--------|--------|-----------------|--------------|
| A2 | `a2_bayes_ffn.py` | `init_rho`, `kl_weight`, `prior_std` | 1.43x batch, 1.70x qual |
| A1 | `a1_bayes_output.py` | `init_rho`, `kl_weight` | 1.36x batch |
| A3 | `a3_bayes_ffn_attn_v.py` | same + `attn_v` HPs | negative at 4L |

**A3 gate:** If A3 is still negative at 16L after 2 runs, abort (confirmed negative across scales).

**Flipout:** All variational methods use Flipout at 16L scale (deeper network + smaller batch → gradient variance matters). Implementation needed before Phase 2.

### Phase 3: Post-hoc Full-Weight (B1)

Run diagonal Laplace on the A0 checkpoint. Expected: still negative (flat curvature). But must confirm at 16L for completeness.

**Early abort:** If curvature mean < $10^{-4}$ and MI ratio < 1.02x on first run, abort immediately. We have strong prior from 4L.

### Phase 4: LoRA Base Training

Train deterministic pretrain on cat 1 (World) — shared base for B2 and B3. Then train deterministic LoRA on cat 2 (Sports) — shared base for B3.

### Phase 5: Variational LoRA (B2)

BLoB LoRA at 16L. Key question: does the MI gap vs A2 narrow with scale?

| HP | 4L value | 16L starting point | Reasoning |
|----|----------|-------------------|-----------|
| `rank` | 16 | 32 | Scale with model width (512 vs 256) |
| `alpha` | 32 | 64 | Keep alpha/rank=2 |
| `init_g` | 0.1 | 0.1 | Start same, let agent adjust |
| `kl_weight` | 0.2 | 0.2 | Start same |

### Phase 6: Post-hoc LoRA (B3)

TFB and Laplace-LoRA on the B3 deterministic LoRA checkpoint.

**TFB key HPs:** `epsilon` (tolerance), `n_search_samples`, `n_anchor_batches`.
**Laplace-LoRA key HPs:** `damping`, `sample_scale`.

These are the most uncertain — 4L results not yet available (B3 training in progress). Agent will need to reason from papers + first-run results.

### Phase 7: Comparison Table

After all experiments, Claude Code generates:
- Full comparison table (Table 2 in the paper)
- Cross-scale analysis (4L vs 16L trends)
- Key findings summary
- Recommendation for paper narrative

## Orchestrator Prompt Templates

### INIT Prompt

```
You are a Bayesian ML researcher. You are designing experiment {METHOD}
for the C milestone of a comparative study.

## Context
- Architecture: 16L/8H/512d miniGPT (~75M params)
- Dataset: AG News topic-split (same as 4L)
- GPU: RTX 4070 (12GB VRAM)

## 4L Results for This Method
{4L_RESULTS from AGENTS.md}

## Paper Reference
{PAPER_EXCERPT from docs/}

## Method Spec
{METHOD_SPEC from specs/comparative-bayesian-llm-study.md}

## Already Completed 16L Experiments
{COMPLETED_RUNS summary}

## Task
Design the initial config for {METHOD} at 16L scale.
- Choose hyperparameters with reasoning grounded in the paper and 4L results.
- State expected outcome ranges.
- Identify key risks.
- Output a YAML config.

Output JSON with schema: {INIT_SCHEMA}
```

### ANALYZE Prompt

```
You are a Bayesian ML researcher analyzing experiment results.

## Experiment: {METHOD}, Run {N}/{MAX}
## Config Used
{YAML_CONFIG}

## MLflow Results
{MLFLOW_DUMP}

## Run History (prior attempts)
{HISTORY}

## Acceptance Criteria
{CRITERIA}

## 4L Reference
{4L_RESULTS}

## Task
Analyze these results. Decide: accept, retry (with new config), or abort.
If retry: explain what went wrong and what specific HP changes will fix it.
Ground your reasoning in the paper's theory, not just "try a different number."

Output JSON with schema: {ANALYZE_SCHEMA}
```

## State Management

```
.pipeline-state/
  pipeline.json          # Global state: current phase, current method
  a0_baseline.json       # Per-method: runs, configs, decisions, final result
  a2_bayes_ffn.json
  ...
  history.json           # Chronological log of all decisions (for context)
```

Each method state file:
```json
{
  "method": "a2_bayes_ffn",
  "status": "in_progress" | "accepted" | "aborted" | "exhausted",
  "runs": [
    {
      "iteration": 0,
      "config_path": "configs/c_a2_v0.yaml",
      "mlflow_run_id": "abc123",
      "mlflow_dump_path": ".pipeline-state/a2_run0.txt",
      "decision": "retry",
      "analysis": "Posterior collapse — init_rho=-3 too low at 16L...",
      "duration_sec": 7200
    }
  ],
  "max_retries": 3,
  "final_metrics": { ... }
}
```

Resume: orchestrator reads `pipeline.json`, finds the last incomplete method, resumes from the last completed stage.

## Guardrails

1. **Budget cap.** Total GPU hours configurable (default: 48h). Orchestrator tracks cumulative time and stops if exceeded.
2. **OOM detection.** If experiment exits with CUDA OOM, Claude Code is asked to reduce batch size or model size before retry. Does not count against retry budget.
3. **Divergence detection.** If training loss is NaN or > 100 after 1000 steps, kill early. Does not count against retry budget.
4. **No code changes.** The pipeline runs existing experiment scripts with different configs. Claude Code tunes hyperparameters, not code. Code changes (e.g., Flipout) are prerequisites, done before the pipeline starts.
5. **Human checkpoint.** After Phase 2 (variational methods), pause for human review before proceeding to post-hoc methods. This is the "does the 16L model work at all?" gate.
6. **Logging.** All Claude Code prompts and responses logged to `.pipeline-state/<method>.log`. Full audit trail.

## Prerequisites (Before Running Pipeline)

1. **Flipout implementation** — needed for variational methods at 16L. Separate BDD→TDD→Code task.
2. **GPU profiling script** — `scripts/profile_gpu.py` to determine architecture constraints.
3. **B3 results at 4L** — needed as baseline for Phase 6 reasoning.
4. **All existing tests green** — `uv run pytest tests/ -q` must pass.

## CLI Interface

```bash
# Run full pipeline
python scripts/run_research_pipeline.py --spec specs/comparative-bayesian-llm-study.md

# Resume after interrupt
python scripts/run_research_pipeline.py --resume

# Run specific method only
python scripts/run_research_pipeline.py --method a2_bayes_ffn

# Dry run (Claude Code reasons but doesn't execute experiments)
python scripts/run_research_pipeline.py --dry-run

# Set GPU budget
python scripts/run_research_pipeline.py --gpu-hours 48

# Override max retries
python scripts/run_research_pipeline.py --max-retries 5
```

## Success Criteria

1. All 7 methods (A0, A1, A2, A3, B1, B2, B3) have final results (accepted, aborted, or exhausted) at 16L.
2. Comparison table populated with 16L metrics alongside 4L reference.
3. Cross-scale trends documented (which methods scale, which don't).
4. Total GPU budget < 48h.
5. Human intervention required only at the Phase 2 checkpoint — everything else automated.

## Reference

- Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — similar concept for automated ML research.
- Existing BDD→TDD→Code pipeline (`scripts/run_pipeline.sh`) — patterns for state management, bounded loops, structured review output.
- Claude Code CLI: `claude -p` for non-interactive prompts, `--output-format json --json-schema` for structured output, `--permission-mode bypassPermissions` for headless operation.
