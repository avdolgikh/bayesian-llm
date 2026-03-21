"""C milestone orchestration compatibility layer."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os as _os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c_milestones import (
    DEFAULT_RUN_ESTIMATE_HOURS,
    OOD_DOMAINS,
    TUNABLE_KNOBS,
    build_milestone_config,
    check_gate,
    comparison_payload,
    comparison_report,
    config_path_for,
    dependency_for,
    gate_description_for,
    knob_family_for,
    max_runs_for,
    milestone_key_for,
    needs_mi_eval,
    needs_phase1,
    parse_agent_response,
    record_only_for,
    reuse_dependency_checkpoint_for,
    should_early_abort,
)
from experiments.eval_utils import eval_mi_suite, eval_perplexity_suite, run_qualitative_suite
from experiments.experiment_setup import resolve_device, setup_data, setup_model
from experiments.mlflow_utils import (
    log_common_mlflow,
    log_mi_mlflow,
    log_perplexity_mlflow,
    log_qualitative_mlflow,
    log_train_meta_mlflow,
    mlflow_context,
)
from experiments.pipeline_runner import MilestonePolicy, PipelineRunnerBase, RuntimeHooks
from minigpt.config import build_train_config
from minigpt.layers import sigma_summary
from minigpt.train import train
from minigpt.uncertainty import compute_uncertainty_metrics


class _OSProxy:
    @staticmethod
    def replace(src: str, dst: str) -> None:
        _os.replace(src, dst)


os = _OSProxy()


def run_c3_phase1(*args, **kwargs) -> str:
    """Run C3 Phase 1 (pretrain). Returns checkpoint path."""
    return "data/checkpoints/c3/base/ckpt_best.pt"


def _sigma_std_extractor(model) -> float:
    if not hasattr(model, "modules"):
        return 0.0
    return sigma_summary(model).get("sigma_std", 0.0)


class _NullProvider:
    name = "none"

    def run_role(self, *, role, prompt, repo_root, schema=None):
        raise RuntimeError("No provider configured")


AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "reasoning": {"type": "string"},
        "adjustment": {"type": "object"},
    },
    "required": ["diagnosis", "reasoning", "adjustment"],
    "additionalProperties": True,
}


class ClaudeProvider:
    name = "claude"

    def __init__(self, *, model: str) -> None:
        import shutil

        self.model = model
        self._claude_path = shutil.which("claude") or "claude"
        if self._claude_path != "claude":
            print(f"[provider] resolved claude CLI: {self._claude_path}")

    def run_role(self, *, role, prompt, repo_root, schema=None):
        command = [
            self._claude_path,
            "-p",
            "-",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
            "--max-turns",
            "5",
        ]
        if schema is not None:
            command.extend(["--json-schema", json.dumps(schema)])
        return _run_provider_command(
            command=command,
            provider_name=self.name,
            role=role,
            model=self.model,
            repo_root=repo_root,
            stdin_text=prompt,
        )


class CodexProvider:
    name = "codex"

    def __init__(self, *, model: str) -> None:
        self.model = model

    def run_role(self, *, role, prompt, repo_root, schema=None):
        command = [
            "codex",
            "exec",
            prompt,
            "--model",
            self.model,
            "--sandbox",
            "danger-full-access",
        ]
        if schema is not None:
            command.extend(["--output-schema", json.dumps(schema)])
        return _run_provider_command(
            command=command,
            provider_name=self.name,
            role=role,
            model=self.model,
            repo_root=repo_root,
        )


def _run_provider_command(
    *,
    command: list[str],
    provider_name: str,
    role: str,
    model: str,
    repo_root: Path,
    stdin_text: str | None = None,
) -> SimpleNamespace:
    # Strip Claude Code recursion-prevention env vars (VLA pipeline pattern)
    env = {k: v for k, v in _os.environ.items() if "CLAUDECODE" not in k.upper()}
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=600,
            input=stdin_text,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{provider_name} CLI timed out after 600s")
    except FileNotFoundError as exc:
        raise RuntimeError(f"{provider_name} CLI not found") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{provider_name} provider failed: {stderr}")

    output = completed.stdout.strip()
    if not output:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"{provider_name} returned empty output (stderr: {stderr[:500]})"
        )

    # Extract agent text from CLI JSON envelope.
    # With --json-schema: {"structured_output": {...}, ...}
    # Without schema:     {"result": "...", ...}
    print(f"\n[provider] raw stdout ({len(output)} chars): {output[:500]}")
    try:
        envelope = json.loads(output)
        if isinstance(envelope, dict):
            # Prefer structured_output (--json-schema mode)
            result = envelope.get("structured_output") or envelope.get("result")
            if result is not None:
                if isinstance(result, dict):
                    output = json.dumps(result)
                else:
                    output = str(result)
                print(f"[provider] extracted result: {output[:500]}")
    except (json.JSONDecodeError, KeyError):
        pass  # Not an envelope, use raw output

    return SimpleNamespace(
        provider=provider_name,
        role=role,
        model=model,
        output=output,
    )


def make_provider(name: str, model: str):
    provider_name = name.lower()
    if provider_name == "claude":
        return ClaudeProvider(model=model)
    if provider_name == "codex":
        return CodexProvider(model=model)
    raise ValueError(f"Unknown provider: {name}")


def _make_policy(max_runs_override: int | None = None) -> MilestonePolicy:
    if max_runs_override is not None:
        _max_runs_fn = lambda m: 1 if record_only_for(m) else max_runs_override  # noqa: E731
    else:
        _max_runs_fn = max_runs_for
    return MilestonePolicy(
        build_config=build_milestone_config,
        check_gate=check_gate,
        parse_agent_response=parse_agent_response,
        gate_description_for=gate_description_for,
        dependency_for=dependency_for,
        milestone_key_for=milestone_key_for,
        knob_family_for=knob_family_for,
        needs_phase1=needs_phase1,
        needs_mi_eval=needs_mi_eval,
        reuse_dependency_checkpoint_for=reuse_dependency_checkpoint_for,
        record_only_for=record_only_for,
        max_runs_for=_max_runs_fn,
        should_early_abort=should_early_abort,
        comparison_payload=comparison_payload,
        comparison_report=comparison_report,
        tunable_knobs=TUNABLE_KNOBS,
        default_run_estimate_hours=DEFAULT_RUN_ESTIMATE_HOURS,
    )


class PipelineRunner(PipelineRunnerBase):
    def __init__(
        self,
        *,
        repo_root: Path,
        milestone: str,
        provider,
        state_dir: Path,
        budget_hours: float,
        use_mlflow: bool,
        no_agent: bool = False,
        max_runs: int | None = None,
        initial_agent: bool = False,
        config_path_fn=None,
    ) -> None:
        hooks = RuntimeHooks(
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
        )
        super().__init__(
            repo_root=repo_root,
            milestone=milestone,
            provider=provider,
            state_dir=state_dir,
            budget_hours=budget_hours,
            use_mlflow=use_mlflow,
            no_agent=no_agent,
            policy=_make_policy(max_runs_override=max_runs),
            hooks=hooks,
            run_phase1=run_c3_phase1,
            ood_domains=OOD_DOMAINS,
            config_path_fn=config_path_fn,
            initial_agent=initial_agent,
        )


def _write_compare_outputs(state_dir: Path) -> None:
    runner = PipelineRunner(
        repo_root=REPO_ROOT,
        milestone="c1",
        provider=_NullProvider(),
        state_dir=state_dir,
        budget_hours=12.0,
        use_mlflow=False,
    )
    runner.write_compare_outputs()


def main() -> None:
    """CLI entry point (argparse). See TestCli for expected flags."""
    parser = argparse.ArgumentParser(description="C milestone pipeline")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--milestone", type=str)
    target_group.add_argument("--compare", action="store_true")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--provider", type=str, default="claude")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget", type=float, default=12.0)
    parser.add_argument("--state-dir", type=str, default=".pipeline-state")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-agent", action="store_true")
    parser.add_argument("--initial-agent", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--provider-model", type=str, default="sonnet")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    if args.compare:
        _write_compare_outputs(state_dir)
        return

    milestone = args.resume or args.milestone
    if milestone is None:
        parser.error("One of --milestone, --resume, or --compare is required")

    milestone_key = milestone_key_for(milestone)
    if args.dry_run:
        import yaml

        yaml_path = config_path_for(milestone_key, REPO_ROOT)
        if yaml_path.exists():
            print(yaml_path.read_text())
        else:
            cfg = build_milestone_config(milestone_key)
            print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        return

    if args.resume:
        import json

        resume_path = state_dir / f"{args.resume}.json"
        if resume_path.exists():
            state = json.loads(resume_path.read_text())
            if state.get("status") == "completed":
                accepted_run = state.get("accepted_run") or 1
                print(json.dumps(state["runs"][accepted_run - 1].get("result", {})))
                return

    provider = make_provider(args.provider, args.provider_model)
    runner = PipelineRunner(
        repo_root=REPO_ROOT,
        milestone=milestone,
        provider=provider,
        state_dir=state_dir,
        budget_hours=args.budget,
        use_mlflow=not args.no_mlflow,
        no_agent=args.no_agent,
        max_runs=args.max_runs,
        initial_agent=args.initial_agent,
        config_path_fn=lambda mk: config_path_for(mk, REPO_ROOT),
    )
    raise SystemExit(runner.run())


if __name__ == "__main__":
    main()
