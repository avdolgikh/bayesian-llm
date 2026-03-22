# Pipeline Refactoring Spec: Decouple Experiment Logic from Generic Engine

**Date:** 2026-03-22
**Status:** PROPOSED
**Priority:** Low (after C milestone completion)

## Problem

`c_pipeline.py` (~615 lines) mixes two concerns:

1. **Generic pipeline infrastructure** — provider wiring (Claude/Codex), CLI entry point, subprocess management, JSON envelope handling
2. **C-experiment-specific hooks** — `_prepare_model` (BLoB→DeterministicLoRA conversion), `_posthoc_fit` (Laplace/TFB dispatch), `_sigma_std_extractor`, `run_c3_phase1`

The pipeline's *main goal* is a **smart reasoning agent that tunes hyperparameters automatically**. The experiment-specific details (which model to load, how to inject LoRA, how to fit Laplace) are implementation details that should be pluggable, not hard-wired.

## Current Architecture

```
pipeline_runner.py          # Generic engine (PipelineRunnerBase, MilestonePolicy, RuntimeHooks)
├── State machine (run loop, gate checks, retries)
├── Agent orchestration (prompt building, diagnostic summaries, fallbacks)
├── Budget control, OOM handling, state persistence
└── Evaluation orchestration (perplexity, MI, qualitative)

c_milestones.py             # C-specific policy (pure functions, no side effects)
├── Config templates (C0–C4)
├── Gate criteria, knob ranges, dependency graph
├── Comparison report generation
└── Helper predicates (needs_phase1, record_only_for, etc.)

c_pipeline.py               # MIXED: generic CLI + C-specific hooks
├── Provider implementations (ClaudeProvider, CodexProvider)     ← generic
├── _run_provider_command() subprocess wrapper                   ← generic
├── CLI argparse entry point                                     ← generic
├── _prepare_model() with BLoB→LoRA conversion                  ← C-specific
├── _posthoc_fit() Laplace/TFB dispatch                         ← C-specific
├── _sigma_std_extractor()                                       ← C-specific
├── run_c3_phase1()                                              ← C-specific
└── PipelineRunner(PipelineRunnerBase) wiring class              ← thin glue
```

## Proposed Architecture

### File Split

```
pipeline_runner.py           # UNCHANGED — generic engine
c_milestones.py              # UNCHANGED — C-specific policy

c_hooks.py (NEW)             # C-specific hook implementations
├── _prepare_model()         # Checkpoint loading + LoRA injection + BLoB conversion
├── _posthoc_fit()           # Laplace / TFB dispatch
├── _sigma_std_extractor()   # Sigma summary extraction
├── run_c3_phase1()          # C3 Phase 1 pretraining / C0 checkpoint reuse
└── build_c_hooks()          # Factory: returns RuntimeHooks wired with all above

providers.py (NEW)           # Generic provider implementations
├── ClaudeProvider           # Claude CLI subprocess
├── CodexProvider            # Codex CLI subprocess
├── _run_provider_command()  # Shared subprocess wrapper + JSON envelope extraction
├── AGENT_RESPONSE_SCHEMA    # JSON schema for structured output
└── make_provider()          # Factory: name → provider instance

c_pipeline.py (SIMPLIFIED)   # Thin CLI entry point + wiring
├── PipelineRunner class     # Just calls build_c_hooks() + _make_policy()
├── CLI argparse (main)      # --milestone, --provider, etc.
└── _write_compare_outputs() # Comparison mode
```

### Key Principles

1. **`c_hooks.py` is the ONLY file that imports from `minigpt.*`** among pipeline files. It contains all domain knowledge about models, checkpoints, LoRA, Laplace, TFB.

2. **`providers.py` is experiment-agnostic.** It knows how to call Claude/Codex and parse responses. It does NOT know about milestones, configs, or Bayesian methods.

3. **`c_pipeline.py` becomes a thin shell** (~100 lines): imports hooks from `c_hooks`, policy from `c_milestones`, providers from `providers`, wires them into `PipelineRunnerBase`.

4. **A hypothetical "D experiment"** would only need `d_hooks.py` + `d_milestones.py`. It reuses `pipeline_runner.py`, `providers.py`, and the same CLI pattern.

### Detailed Changes

#### `c_hooks.py` (new, ~200 lines)

Move from `c_pipeline.py`:
- `run_c3_phase1()` — as-is
- `_prepare_model()` — as-is (includes BLoB→DeterministicLoRA conversion)
- `_posthoc_fit()` — as-is (Laplace/TFB dispatch)
- `_sigma_std_extractor()` — as-is

Add factory function:
```python
def build_c_hooks() -> RuntimeHooks:
    """Wire all C-experiment hook implementations into a RuntimeHooks instance."""
    return RuntimeHooks(
        os_replace=os.replace,
        setup_data=setup_data,
        setup_model=setup_model,
        resolve_device=resolve_device,
        build_train_config=build_train_config,
        train=train,
        eval_perplexity_suite=eval_perplexity_suite,
        eval_mi_suite=eval_mi_suite,
        run_qualitative_suite=run_qualitative_suite,
        mlflow_context=mlflow_context,
        log_common_mlflow=log_common_mlflow,
        log_train_meta_mlflow=log_train_meta_mlflow,
        log_perplexity_mlflow=log_perplexity_mlflow,
        log_mi_mlflow=log_mi_mlflow,
        log_qualitative_mlflow=log_qualitative_mlflow,
        uncertainty_eval_fn=compute_uncertainty_metrics,
        sigma_std_extractor=_sigma_std_extractor,
        prepare_model=_prepare_model,
        posthoc_fit_fn=_posthoc_fit,
    )
```

#### `providers.py` (new, ~150 lines)

Move from `c_pipeline.py`:
- `ClaudeProvider` class
- `CodexProvider` class
- `_run_provider_command()` helper
- `_NullProvider` class
- `AGENT_RESPONSE_SCHEMA` constant
- `make_provider()` factory

No changes to implementation — pure relocation.

#### `c_pipeline.py` (simplified, ~100 lines)

```python
"""C milestone pipeline CLI — thin wiring layer."""

from experiments.c_hooks import build_c_hooks, run_c3_phase1
from experiments.c_milestones import (
    OOD_DOMAINS, build_milestone_config, check_gate, ...,
)
from experiments.providers import make_provider, _NullProvider
from experiments.pipeline_runner import PipelineRunnerBase, MilestonePolicy


class PipelineRunner(PipelineRunnerBase):
    def __init__(self, *, repo_root, milestone, provider, state_dir, ...):
        super().__init__(
            hooks=build_c_hooks(),
            policy=_make_policy(...),
            run_phase1=run_c3_phase1,
            ood_domains=OOD_DOMAINS,
            ...
        )


def main():
    # argparse + PipelineRunner(...).run()
```

### What Does NOT Change

- `pipeline_runner.py` — zero modifications
- `c_milestones.py` — zero modifications
- All test files — imports update only (e.g., `pipeline_module._posthoc_fit` → test still accesses via the module, which re-exports from `c_hooks`)
- CLI interface — identical flags and behavior
- Agent prompting, diagnostic summaries, mechanical fallbacks — all in `pipeline_runner.py`, unchanged

### Migration Plan

1. Create `providers.py` — move provider classes (pure relocation, no logic changes)
2. Create `c_hooks.py` — move hook functions (pure relocation, no logic changes)
3. Simplify `c_pipeline.py` — import from new modules, remove moved code
4. Update test imports if needed (the `pipeline_module` fixture reimports `c_pipeline`, so if `c_pipeline` re-exports the moved functions, tests may need no changes)
5. Run full test suite — 217/217 must pass
6. Lint check

### Scope Boundaries

**In scope:**
- File split (pure relocation of existing code)
- Factory functions for clean wiring

**Out of scope (future work):**
- Generalizing `MilestonePolicy` to a config-driven schema (JSON/YAML gate definitions)
- Making mechanical fallback pluggable per experiment family
- CLI abstraction (auto-discovering experiment modules)
- Multi-experiment comparison across different experiment families

### Risk Assessment

**Low risk.** This is a pure code reorganization with no behavioral changes. The same functions move between files. Tests validate behavior is preserved.

### Success Criteria

1. `c_pipeline.py` is ≤120 lines (currently ~615)
2. `providers.py` has zero imports from `minigpt.*`
3. `c_hooks.py` is the only pipeline file importing `minigpt.*`
4. 217/217 tests pass unchanged
5. CLI invocations produce identical results
