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
from minigpt.config import build_lora_config, build_train_config
from minigpt.layers import sigma_summary
from minigpt.lora import inject_lora
from minigpt.train import load_checkpoint, train
from minigpt.uncertainty import compute_uncertainty_metrics


class _OSProxy:
    @staticmethod
    def replace(src: str, dst: str) -> None:
        _os.replace(src, dst)


os = _OSProxy()


def run_c3_phase1(*, repo_root: Path, state_dir: Path) -> str:
    """Run C3 Phase 1 (pretrain). Returns checkpoint path.

    C3 Phase 1 = deterministic 16L model on Pile ID domains.
    This is the same architecture and data as C0, so we reuse the C0
    checkpoint if available (saves ~4.5 hours of GPU time).
    """
    c3_phase1_dir = Path("data/checkpoints/c3/base")
    c3_phase1_ckpt = c3_phase1_dir / "ckpt_best.pt"

    if c3_phase1_ckpt.exists():
        print(f"[phase1] C3 Phase 1 checkpoint already exists: {c3_phase1_ckpt}")
        return str(c3_phase1_ckpt)

    # Reuse C0 checkpoint — same 16L/8H/512d architecture, same Pile ID data
    c0_ckpt = Path("data/checkpoints/c0/ckpt_best.pt")
    c0_state = state_dir / "c0.json"
    if c0_ckpt.exists() and c0_state.exists():
        import json as _json
        import shutil

        c0_data = _json.loads(c0_state.read_text())
        if c0_data.get("status") == "completed":
            c3_phase1_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(c0_ckpt), str(c3_phase1_ckpt))
            print(f"[phase1] Reused C0 checkpoint for C3 Phase 1: {c0_ckpt} -> {c3_phase1_ckpt}")
            return str(c3_phase1_ckpt)

    raise FileNotFoundError(
        "C3 Phase 1 requires a pretrained deterministic 16L model. "
        "Run C0 first: python experiments/c_pipeline.py --milestone c0"
    )


def _prepare_model(model, cfg):
    """Load base checkpoint and inject LoRA if config requires it.

    Called between setup_model() and train() for milestones that need
    a pretrained base (C3 Phase 2, C2, C4).
    """
    from torch import nn

    if not isinstance(model, nn.Module):
        return model  # test mock — skip preparation

    resume_path = cfg["train"].get("resume_checkpoint")
    if resume_path:
        ckpt_path = Path(resume_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
        print(f"[prepare] Loading base checkpoint: {ckpt_path}")
        load_checkpoint(ckpt_path, model)

    lora_section = cfg.get("lora")
    if lora_section and lora_section.get("rank"):
        lora_cfg = build_lora_config(cfg)
        model = inject_lora(model, lora_cfg)
        n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_bayes = sum(
            p.numel() for name, p in model.named_parameters()
            if p.requires_grad and ("lora_A_mu" in name or "lora_A_g" in name)
        )
        print(f"[prepare] LoRA injected: rank={lora_cfg.rank}, alpha={lora_cfg.alpha}, "
              f"target={lora_cfg.target}")
        print(f"[prepare] LoRA params: {n_lora:,} trainable, {n_bayes:,} Bayesian")

    return model


def _sigma_std_extractor(model) -> float:
    if not hasattr(model, "modules"):
        return 0.0
    return sigma_summary(model).get("sigma_std", 0.0)


def _posthoc_fit(model, cfg, data, device):
    """Fit post-hoc Bayesian method (Laplace or TFB) if config requires it.

    Dispatches on cfg["posthoc_method"]: "laplace" or "tfb".
    Only post-hoc milestone templates set this field.

    Returns (mi_eval_fn, extra_result_fields) or None if not applicable.
    """
    import time
    from functools import partial

    import torch
    from torch import nn

    if not isinstance(model, nn.Module):
        return None  # test mock — skip

    method = cfg.get("posthoc_method")
    if not method:
        return None

    train_data = data.get("train")
    if train_data is None:
        raise ValueError("No training data for post-hoc curvature fitting")

    block_size = cfg["train"]["block_size"]
    batch_size = cfg["train"]["batch_size"]

    lap_cfg = cfg.get("laplace", {})
    tfb_cfg = cfg.get("tfb", {})

    if method == "laplace":
        from minigpt.laplace import (
            compute_laplace_uncertainty,
            fit_laplace,
            save_laplace_state,
            select_params,
        )

        selection_mode = lap_cfg.get("selection_mode", "ffn")
        damping = lap_cfg.get("damping", 1.0)
        sample_scale = lap_cfg.get("sample_scale", 1.0)
        n_curv_batches = lap_cfg.get("n_curvature_batches", 30)

        selected = select_params(model, mode=selection_mode)
        n_selected = sum(p.numel() for p in selected.values())
        print(f"[laplace] Selected {len(selected)} param groups "
              f"({n_selected:,} elements), mode={selection_mode}")

        fit_start = time.time()
        state = fit_laplace(
            model, train_data.to(device),
            block_size=block_size,
            batch_size=batch_size,
            selection=selected,
            n_batches=n_curv_batches,
            damping=damping,
            sample_scale=sample_scale,
        )
        print(f"[laplace] Fit time: {time.time() - fit_start:.1f}s")

        ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_laplace_state(state, ckpt_dir / "laplace_state.pt")

        all_curv = torch.cat(
            [state.curvature[n].flatten() for n in state.param_names]
        )
        curvature_mean = all_curv.mean().item()
        print(f"[laplace] Curvature: mean={curvature_mean:.6f} "
              f"std={all_curv.std().item():.6f} "
              f"min={all_curv.min().item():.6f} max={all_curv.max().item():.6f}")

        mi_fn = partial(compute_laplace_uncertainty, state=state)
        return mi_fn, {"curvature_mean": curvature_mean}

    if method == "tfb":
        from minigpt.tfb import compute_tfb_uncertainty, fit_tfb, save_tfb_state

        epsilon = tfb_cfg.get("epsilon", 0.1)
        n_search_samples = tfb_cfg.get("n_search_samples", 10)
        n_anchor_batches = tfb_cfg.get("n_anchor_batches", 20)

        fit_start = time.time()
        state = fit_tfb(
            model, train_data.to(device),
            block_size=block_size,
            batch_size=batch_size,
            n_batches=n_anchor_batches,
            epsilon=epsilon,
            n_search_samples=n_search_samples,
        )
        print(f"[tfb] Fit time: {time.time() - fit_start:.1f}s")
        print(f"[tfb] sigma_q={state.sigma_q:.6f}, anchor_loss={state.anchor_loss:.4f}")

        ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_tfb_state(state, ckpt_dir / "tfb_state.pt")

        mi_fn = partial(compute_tfb_uncertainty, state=state)
        return mi_fn, {"sigma_q": state.sigma_q, "anchor_loss": state.anchor_loss}

    return None


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
            prepare_model=_prepare_model,
            posthoc_fit_fn=_posthoc_fit,
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
