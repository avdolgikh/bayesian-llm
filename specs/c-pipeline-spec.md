# C Pipeline: Agentic Experiment Optimization

## Context

### The Core Idea

The pipeline is a **state machine** that automates the HP tuning loop for C-milestone experiments. For each sub-milestone (C0–C4):

1. **CONFIGURE** — generate config (template + prior agent adjustments)
2. **RUN** — execute training + evaluation (pure subprocess, no agent)
3. **ANALYZE** — extract metrics, check success gate (pure Python, no agent)
4. **DECIDE** — if gate passed: accept. If failed: invoke **agent** to diagnose and propose HP adjustment, then loop back to CONFIGURE.

Steps 1–3 are **programmatic** — plain Python and subprocess calls. Only step 4 (diagnosis + HP reasoning) requires the agent. The agent is **Claude Code or Codex**, invoked as a subprocess (`claude -p` / `codex exec`), not an API call. The agent has full repo access — it reads `AGENTS.md`, inspects MLflow dumps, understands the research context, and returns a structured HP adjustment. This is exactly what a human researcher does today, but automated.

### Two-Layer Workflow

| Layer | Process | What it does | When |
|-------|---------|-------------|------|
| **Method code** | BDD → TDD → Code (manual, stage-gated) | Implements training/eval code for each method | Before auto-research |
| **Auto-research** | `c_pipeline.py --milestone cN` (automated) | Runs the implemented code with agentic HP optimization | After method code is tested & green |

**This spec describes the auto-research layer only.** Method code already exists and passes all tests.

### Research Context

The C milestone replicates the 4L 2×2 Bayesian comparison matrix on 16L/8H/512d miniGPT (~76M params) using Pile domain-split data. Parent spec: `specs/c-milestone-spec.md`.

---

## 1. User Stories

**US-1:** Run one sub-milestone (`--milestone c1`) and have the pipeline autonomously find best HP via the agent loop (up to 4 runs).

**US-2:** Resume an interrupted milestone with `--resume` — completed runs preserved, loop picks up where it left off.

**US-3:** Generate a cross-method comparison table with `--compare` after running milestones.

**US-4:** Track cumulative GPU time per milestone; stop if budget exceeded.

**US-5:** When a success gate fails, the agent reasons about what went wrong and proposes an HP adjustment — driven by intelligent analysis, not hardcoded rules.

**US-6:** The agent sees the **full history** of prior attempts (all runs, not just the latest) when diagnosing.

---

## 2. Architecture

```
c_pipeline.py (state machine, pure Python)
    │
    ├── CONFIGURE ──── pure Python: merge template + agent adjustments
    ├── RUN ────────── subprocess: python experiments/cN_*.py (training + eval)
    ├── ANALYZE ────── pure Python: dump_mlflow_run.py, parse metrics, check gate
    └── DECIDE ─────── if gate failed: invoke Provider (claude -p / codex exec)
                        Agent reads repo, AGENTS.md, MLflow dump,
                        returns structured HP adjustment
                        ↓
                       loop back to CONFIGURE (up to 4 runs)

Provider (adapter pattern, like vla-game-agent):
    ├── ClaudeProvider  →  claude -p <prompt> --model <model> --output-format json
    └── CodexProvider   →  codex exec <prompt> --model <model> --output-schema <schema>
```

The pipeline **composes** existing building blocks. It does NOT contain training loops, evaluation math, or model code. Its unique responsibilities: **state machine orchestration**, **provider-based agent invocation** for HP tuning, config generation, success gate evaluation, budget tracking, and reporting.

### Provider Pattern

Following the `vla-game-agent/pipeline` pattern:

```python
class Provider(Protocol):
    name: str
    def run_role(self, *, role: str, prompt: str, repo_root: Path,
                 schema: dict | None = None) -> ProviderExecution

class ClaudeProvider:
    """Runs agent via: claude -p <prompt> --model <model> --permission-mode bypassPermissions"""

class CodexProvider:
    """Runs agent via: codex exec <prompt> --model <model> --sandbox danger-full-access"""
```

Single role needed: `"researcher"` — the HP diagnosis role. The agent is invoked **only** when a success gate fails and the pipeline needs a reasoned HP adjustment.

---

## 3. Walkthrough

**Scenario: C1 (Variational FFN) — agent finds best HP in 3 runs.**

```
RUN 1: Initial config (init_rho=-2.0, kl_weight=0.2)
  → python experiments/c1_*.py → MLflow → mi_ratio_mean=1.08, sigma_std=0.004
  → Gate: FAIL (need >1.2)
  → Agent invoked (claude -p / codex exec):
    Agent reads AGENTS.md, MLflow dump, sees sigma_std=0.004 → "posteriors frozen"
    → adjustment: {"model.bayes_ffn.init_rho": -1.0}

RUN 2: Agent adjustment applied (init_rho=-1.0)
  → mi_ratio_mean=1.15, kl_direction=increasing
  → Gate: FAIL (improved from 1.08 but still <1.2)
  → Agent sees full history (run 1 + run 2), prior diagnosis helped
    → adjustment: {"train.kl_weight": 0.3}

RUN 3: Agent adjustment applied (init_rho=-1.0, kl_weight=0.3)
  → mi_ratio_mean=1.28
  → Gate: PASS (1.28 > 1.2)
  → ACCEPT. Save checkpoint. Done.
```

---

## 4. Behaviors

### B-1: CLI

```
python experiments/c_pipeline.py --milestone c0
python experiments/c_pipeline.py --milestone c1
python experiments/c_pipeline.py --resume c1
python experiments/c_pipeline.py --compare
```

Optional flags:
- `--state-dir <path>` — default: `.pipeline-state`
- `--budget <hours>` — GPU hours per milestone (default: 12)
- `--no-mlflow` — disable MLflow logging
- `--dry-run` — print config without executing
- `--provider {claude|codex}` — agent provider (default: `claude`)
- `--provider-model <model>` — agent model (default: `sonnet`)
- `--no-agent` — skip agent, use mechanical fallback only

`--milestone` and `--compare` are mutually exclusive. `--resume` requires prior state with incomplete milestone.

### B-2: Config generation

Pipeline generates config by merging:
1. `DEFAULT_CONFIG` (from `minigpt/config.py`)
2. Milestone-specific template (hardcoded in pipeline, see Section 6)
3. HP adjustments from agent (on retry runs)

Validated via `validate_config()` before use.

### B-3: State management

State directory: `.pipeline-state/`
- `pipeline.json` — global state (milestones completed, results summary)
- `{milestone}.json` — per-milestone state (runs, metrics, agent responses)

State is persisted after every run (atomic write: temp file → rename). Survives process termination.

### B-4: State schema

Per-milestone state (`{milestone}.json`):
```json
{
  "status": "pending | running | completed | failed",
  "runs": [
    {
      "run_number": 1,
      "mlflow_run_id": "abc123...",
      "config_overrides": {},
      "training_time_seconds": 3600.0,
      "result": {
        "test_id_ppl": 45.2,
        "test_ood_ppl": {"arxiv": 234.5, "freelaw": 312.1, "pubmed_abstracts": 198.7},
        "mi_id": 0.062,
        "mi_ood": {"arxiv": 0.089, "freelaw": 0.075, "pubmed_abstracts": 0.081},
        "mi_ratio": {"arxiv": 1.44, "freelaw": 1.21, "pubmed_abstracts": 1.31},
        "mi_ratio_mean": 1.32,
        "sigma_mean": 0.147,
        "sigma_std": 0.074,
        "kl_final": 3200000,
        "kl_direction": "decreasing"
      },
      "decision": "accept | retry | abort",
      "agent_response": {
        "diagnosis": "posteriors_frozen",
        "reasoning": "...",
        "adjustment": {"model.bayes_ffn.init_rho": -1.0}
      },
      "checkpoint_path": "data/checkpoints/c1/run1/ckpt_best.pt"
    }
  ],
  "accepted_run": 1,
  "checkpoint_path": "data/checkpoints/c1/run1/ckpt_best.pt"
}
```

### B-5: Resume

`--resume <milestone>` loads state, identifies last incomplete run, resumes loop from that point (re-running the interrupted run from scratch). Completed runs preserved. If milestone already completed, prints result and exits.

### B-6: Orchestration loop

```
for run_number in 1..4:
    CONFIGURE: generate config (template + agent adjustments from prior run)
    RUN:       subprocess: training + evaluation + MLflow logging
    ANALYZE:   run dump_mlflow_run.py, parse metrics, check success gate
    DECIDE:
      ├─ gate passed  → ACCEPT, save state, exit
      ├─ diverged     → mechanical fix (halve lr), retry (doesn't count as agent retry)
      ├─ OOM          → halve batch_size, double grad_accum, retry (max 2 OOM retries, doesn't count)
      ├─ gate failed  → invoke agent via Provider, get HP adjustment, retry
      └─ max retries  → mark failed, record best run, exit
```

### B-7: RUN phase

Executes training + evaluation as a subprocess (or in-process function call using existing `experiment_setup.py` + `train_model()` + `eval_utils.py`). Logs everything to MLflow. Returns: MLflow run ID, training time, checkpoint path.

### B-8: ANALYZE phase

1. Run `scripts/dump_mlflow_run.py <run_id>` to extract metrics
2. Parse result into structured dict (perplexity, MI, sigma stats, KL)
3. Check success gate (see B-12)
4. Detect divergence: NaN loss or `best_val_loss > 100`
5. Save run result to milestone state

### B-9: Agent invocation (the DECIDE phase when gate fails)

The pipeline:
1. Writes a prompt file containing: milestone context, success gate, tunable HP knobs, full run history (all prior runs with metrics + prior agent responses), current run's MLflow dump
2. Invokes the Provider: `claude -p <prompt> --output-format json` (or `codex exec`)
3. The agent has **full repo access** — can read `AGENTS.md`, inspect code, run commands
4. Agent returns structured JSON: `{"diagnosis": "...", "reasoning": "...", "adjustment": {"key": value}}`
5. Pipeline records agent response in milestone state, applies adjustment to next run's config

The agent prompt is constructed programmatically but the agent **reads the repo itself** for full context. The prompt only needs to provide: what milestone, what happened (metrics dump), what the gate is, and what HP knobs are tunable.

### B-10: Agent prompt content

The prompt passed to the Provider includes:

- **Milestone:** which method is being tested (e.g., "C1: Variational FFN")
- **Success gate:** metric threshold (e.g., `mi_ratio_mean > 1.2`)
- **Tunable knobs:** which HP the agent may adjust, with valid ranges (see B-13)
- **Run history:** for each prior run — config overrides, MLflow metrics dump, prior agent diagnosis
- **Current run:** full MLflow metrics dump, gate result (pass/fail, distance from threshold)
- **Instructions:** read `AGENTS.md` for full research context; diagnose the failure; propose exactly one HP change; return JSON with `diagnosis`, `reasoning`, `adjustment`; do not repeat a failed adjustment

The prompt does NOT embed the full research context — the agent reads `AGENTS.md` directly from the repo.

### B-11: Agent fallback

If the Provider fails (subprocess error, timeout):
- Log warning: `"Agent unavailable, using mechanical fallback"`
- Diverged → halve `train.lr`
- `sigma_std < 0.01` → `init_rho += 1.0` (or `init_g *= 2` for LoRA)
- Otherwise → no adjustment, retry with same config

Fallback is deliberately minimal. Only handles obvious mechanical cases.

### B-12: Success gates

| Milestone | Gate | Metric |
|-----------|------|--------|
| C0 | `test_id_ppl < 80` AND `test_ood_ppl > 2 * test_id_ppl` (any domain) | Perplexity separation |
| C1 | `mi_ratio_mean > 1.2` | MI separation |
| C2 | No gate (expected negative). Early-abort: curvature mean $< 10^{-4}$ AND `mi_ratio_mean < 1.02` | Record only |
| C3 | `mi_ratio_mean > 1.05` | MI separation |
| C4-TFB | No gate (record result) | Record MI |
| C4-LAP | No gate (expected negative). Early-abort: same as C2 | Record only |

### B-13: Tunable HP knobs per milestone

| Milestone | Tunable knobs | Valid ranges |
|-----------|--------------|--------------|
| C0 | `train.lr`, `train.steps`, `train.warmup_steps`, `model.dropout` | lr: [1e-5, 1e-3], steps: [50K, 300K], warmup: [500, 10K], dropout: [0.0, 0.5] |
| C1 | `bayes_ffn.init_rho`, `bayes_ffn.prior_std`, `train.kl_weight`, `train.kl_annealing_steps`, `train.lr`, `train.steps` | init_rho: [-5, 0], prior_std: [0.1, 5], kl_weight: [0.01, 2], kl_annealing: [0, 20K], lr: [1e-5, 1e-3], steps: [50K, 300K] |
| C3 Phase 2 | `lora.init_g`, `lora.prior_std`, `lora.rank`, `train.kl_weight`, `train.lr`, `train.steps` | init_g: [0.01, 1.0], prior_std: [0.05, 2.0], rank: [4, 64], kl_weight: [0.01, 2.0], lr: [1e-5, 1e-3], steps: [5K, 50K] |

C2 and C4 are post-hoc methods — no training HP to tune.

### B-14: Budget tracking

Before each run: check `budget_used + estimated_run_time <= budget_limit`. Estimate = most recent run's training time (or 6h default). If exceeded: mark milestone `"failed"` with reason `"budget_exceeded"`.

### B-15: OOM recovery

OOM (`RuntimeError` with `"out of memory"` or `"CUDA"`) → halve batch_size, double grad_accum. Max 2 OOM retries per milestone. Does NOT count against 4-run retry budget.

### B-16: Divergence detection

NaN loss or `best_val_loss > 100` → mark run `"diverged"`, halve `train.lr`, retry. This is mechanical (no agent needed). Counts against retry budget.

---

## 5. Per-Milestone Specifics

### C0 — Deterministic baseline

Train 16L/8H/512d on Pile ID domains. Evaluate perplexity on test_id + all test_ood domains. No MI eval (no Bayesian components). Agent tunes: `lr`, `steps`, `warmup_steps`, `dropout`.

### C1 — Variational full-weight

Train with ELBO (Bayesian FFN). Evaluate perplexity + MI on all splits. Extract sigma stats and KL trajectory for diagnostics. Starting HP: `bayes_ffn.init_rho=-2.0`, `prior_std=1.0`, `kl_weight=0.2`.

### C2 — Post-hoc Laplace

Load C0 checkpoint (requires C0 completed). Fit diagonal Fisher curvature on FFN params. Evaluate MI. Expected negative result (MI ~1.00x). One run, no retries.

### C3 — BLoB LoRA (two phases)

**Phase 1:** Reuse C0 checkpoint if domain-compatible, else pretrain on Wikipedia only. Run once.
**Phase 2:** BLoB LoRA fine-tune on HackerNews. Evaluate perplexity + MI. Agent tunes Phase 2 HP only (Phase 1 not repeated). Starting HP: `rank=16`, `alpha=32`, `prior_std=0.2`, `init_g=0.1`, `kl_weight=0.2`.

### C4 — Post-hoc LoRA (three phases)

**Phase 1:** Deterministic LoRA on HackerNews (reuses C3's base). Run once.
**Phase 2a (TFB):** Fit SVD-structured variance. Evaluate MI. One run, no retries.
**Phase 2b (Laplace-LoRA):** Fit diagonal Laplace on LoRA A params. Evaluate MI. Expected negative. One run.

### Checkpoint dependencies

```
C0 ──────> C2 (uses C0 checkpoint)
C0 ──────> C3 Phase 1 (reuses C0 if compatible)
C3 Ph.1 ─> C3 Phase 2, C4 Phase 1 (shared base)
C4 Ph.1 ─> C4 Phase 2a/2b (uses LoRA checkpoint)
```

C1 has no dependencies (trains from scratch).

### Multi-OOD evaluation

- Perplexity: evaluate each OOD domain independently → `{"arxiv": float, "freelaw": float, "pubmed_abstracts": float}`
- MI: compute per-domain MI ratios → `mi_ratio_mean = mean(mi_ratio_domain for all domains)`
- MLflow: log per-domain metrics with suffix: `mi_ood_arxiv`, `test_ood_ppl_freelaw`, `mi_ratio_mean`, etc.

---

## 6. Config Templates

### C0 (deterministic baseline)
```yaml
data:
  dataset: pile
  pile_id_domains: [wikipedia_en, stackexchange]
  pile_ood_domains: [arxiv, freelaw, pubmed_abstracts]
  pile_id_tokens: 100000000
  pile_ood_tokens: 10000000
  val_fraction: 0.05
  test_fraction: 0.05
model:
  block_size: 256
  n_layer: 16
  n_head: 8
  n_embd: 512
  dropout: 0.2
  bias: true
  bayes_head: {enabled: false}
  bayes_ffn: {enabled: false}
  bayes_attn_v: {enabled: false}
train:
  steps: 100000
  batch_size: 16
  block_size: 256
  lr: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 2000
  min_lr: 1.0e-5
  grad_clip: 1.0
  eval_interval: 2000
  eval_iters: 20
  checkpoint_interval: 10000
  checkpoint_dir: data/checkpoints/c0
  gradient_accumulation_steps: 2
  kl_weight: 0.0
  kl_annealing_steps: 0
  seed: 1337
  device: auto
eval:
  num_samples: 20
  n_perplexity_batches: 50
```

### C1 (variational full-weight)
Same as C0 except:
```yaml
model:
  bayes_ffn: {enabled: true, prior_std: 1.0, init_rho: -2.0}
train:
  kl_weight: 0.2
  kl_annealing_steps: 5000
  checkpoint_dir: data/checkpoints/c1
```

### C2 (post-hoc Laplace)
Uses C0 checkpoint. No training:
```yaml
laplace:
  selection_mode: ffn
  damping: 1.0
  sample_scale: 1.0
  n_curvature_batches: 30
```

### C3 Phase 1 (pretrain — if not reusing C0)
Same as C0 except:
```yaml
data:
  pile_id_domains: [wikipedia_en]
  pile_id_tokens: 100000000
train:
  steps: 50000
  checkpoint_dir: data/checkpoints/c3/base
```

### C3 Phase 2 (BLoB LoRA fine-tune)
```yaml
data:
  pile_id_domains: [hackernews]
  pile_ood_domains: [arxiv, freelaw, pubmed_abstracts]
  pile_id_tokens: 20000000
  pile_ood_tokens: 10000000
lora:
  rank: 16
  alpha: 32.0
  target: ffn
  prior_std: 0.2
  init_g: 0.1
  base_checkpoint: data/checkpoints/c3/base/ckpt_best.pt
train:
  steps: 10000
  batch_size: 32
  gradient_accumulation_steps: 1
  lr: 3.0e-4
  kl_weight: 0.2
  kl_annealing_steps: 1000
  checkpoint_dir: data/checkpoints/c3/lora
```

### C4 Phase 1 (deterministic LoRA)
Same as C3 Phase 2 except: no `lora.prior_std`/`lora.init_g` (deterministic), `kl_weight: 0.0`.

### C4 Phase 2a (TFB)
```yaml
tfb:
  epsilon: 0.1
  n_search_samples: 10
  n_anchor_batches: 20
```

### C4 Phase 2b (Laplace-LoRA)
```yaml
laplace:
  selection_mode: lora
  damping: 1.0
  sample_scale: 1.0
  n_curvature_batches: 30
```

---

## 7. Reporting

### Comparison report (`--compare`)

```
=== C Milestone: 16L/8H/512d Results ===

Method               | MI ratio (mean) | MI arxiv | MI law   | MI med   | ID ppl  | Runs
---------------------|-----------------|----------|----------|----------|---------|-----
C0 (deterministic)   | —               | —        | —        | —        | 45.2    | 1
C1 (variational FFN) | 1.32x           | 1.44x    | 1.21x    | 1.31x    | 58.1    | 2
C2 (Laplace full)    | 1.00x           | 1.00x    | 1.00x    | 1.00x    | 45.2    | 1
C3 (BLoB LoRA)       | 1.08x           | 1.12x    | 1.05x    | 1.07x    | 189.3   | 3
C4-TFB (post-hoc)    | 1.06x           | 1.09x    | 1.04x    | 1.05x    | 187.1   | 1
C4-LAP (post-hoc)    | 1.00x           | 1.00x    | 1.00x    | 1.00x    | 187.1   | 1

=== Cross-Scale Comparison ===

Method               | 4L MI ratio | 16L MI ratio | Direction
---------------------|-------------|--------------|----------
Variational full     | 1.43x       | 1.32x        | ↓
BLoB LoRA            | 1.13x       | 1.08x        | ↓
TFB (post-hoc LoRA)  | 1.10x       | 1.06x        | ↓
Laplace full         | 1.00x       | 1.00x        | ≈
Laplace LoRA         | 1.00x       | 1.00x        | ≈
```

4L results are constants embedded in the pipeline (from completed A/B-series). Report saved to `.pipeline-state/comparison.{txt,json}`.

---

## 8. Acceptance Criteria

1. Each milestone is a separate invocation. No `--milestone all`.
2. `--milestone cN` runs the full CONFIGURE → RUN → ANALYZE → DECIDE loop (up to 4 runs).
3. Agent is invoked via Provider (`claude -p` / `codex exec`) — NOT via API. Agent has full repo access.
4. Agent receives full run history (all prior attempts + metrics) when diagnosing.
5. Agent's diagnosis, reasoning, and adjustment are recorded in milestone state and printed to stdout.
6. `--resume` picks up incomplete milestone from prior state.
7. `--compare` generates cross-method and cross-scale comparison tables.
8. `--dry-run` prints planned config without executing.
9. State files persisted after every run, survive process termination.
10. Budget tracking stops milestone before exceeding limit.
11. Max 4 runs per milestone; OOM retries don't count.
12. Diverged runs trigger mechanical LR halving (no agent needed).
13. Multi-OOD metrics computed and logged per domain with mean aggregate.
14. Milestone dependencies validated before execution (C2 requires C0, etc.).
15. If agent Provider fails, pipeline falls back to minimal mechanical adjustments.

---

## 9. Out of Scope

- **`--milestone all`** — no batch mode. Researcher reviews between milestones.
- **Pile-specific qualitative evaluation** — quantitative MI (batch-level) is sufficient.
- **Automatic config file generation** — configs generated in-memory from templates.
- **Flipout** — confirmed unnecessary at batch_size=16.
- **A3-equivalent** — A3 was negative at 4L; not included.
