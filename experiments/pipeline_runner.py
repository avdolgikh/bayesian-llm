"""Generic research pipeline orchestration."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "reasoning": {"type": "string"},
        "adjustment": {"type": "object"},
    },
    "required": ["diagnosis", "reasoning", "adjustment"],
}


@dataclass(frozen=True)
class RuntimeHooks:
    os_replace: Callable[[str, str], None]
    setup_data: Callable[[dict[str, Any]], tuple[Any, dict[str, Any]]]
    setup_model: Callable[[dict[str, Any], Any], tuple[Any, Any, int]]
    resolve_device: Callable[[dict[str, Any]], Any]
    build_train_config: Callable[[dict[str, Any]], Any]
    train: Callable[..., tuple[Any, dict[str, Any]]]
    eval_perplexity_suite: Callable[..., dict[str, Any]]
    eval_mi_suite: Callable[..., tuple[dict[str, Any], dict[str, Any] | None, float | None]]
    run_qualitative_suite: Callable[..., tuple[str, list[dict[str, Any]]]]
    mlflow_context: Callable[..., Any]
    log_common_mlflow: Callable[..., None]
    log_train_meta_mlflow: Callable[..., None]
    log_perplexity_mlflow: Callable[..., None]
    log_mi_mlflow: Callable[..., None]
    log_qualitative_mlflow: Callable[..., None]
    uncertainty_eval_fn: Callable[..., dict[str, float]]
    sigma_std_extractor: Callable[[Any], float]


@dataclass(frozen=True)
class MilestonePolicy:
    build_config: Callable[[str, dict | None], dict]
    check_gate: Callable[[str, dict], bool]
    parse_agent_response: Callable[[str], dict]
    gate_description_for: Callable[[str], str]
    dependency_for: Callable[[str], str | None]
    milestone_key_for: Callable[[str], str]
    knob_family_for: Callable[[str], str]
    needs_phase1: Callable[[str], bool]
    needs_mi_eval: Callable[[str], bool]
    reuse_dependency_checkpoint_for: Callable[[str], bool]
    record_only_for: Callable[[str], bool]
    max_runs_for: Callable[[str], int]
    should_early_abort: Callable[[str, dict], bool]
    comparison_payload: Callable[[Path], dict[str, Any]]
    comparison_report: Callable[[Path], str]
    tunable_knobs: dict[str, dict[str, tuple[float, float]]]
    default_run_estimate_hours: float = 1.0


class PipelineRunnerBase:
    def __init__(
        self,
        *,
        repo_root: Path,
        milestone: str,
        provider,
        state_dir: Path,
        budget_hours: float,
        use_mlflow: bool,
        no_agent: bool,
        policy: MilestonePolicy,
        hooks: RuntimeHooks,
        run_phase1: Callable[..., str],
        ood_domains: tuple[str, ...],
        config_path_fn: Callable[[str], Path] | None = None,
        initial_agent: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.milestone = milestone
        self.provider = provider
        self.state_dir = Path(state_dir)
        self.budget_hours = budget_hours
        self.use_mlflow = use_mlflow
        self.no_agent = no_agent
        self.policy = policy
        self.hooks = hooks
        self.run_phase1 = run_phase1
        self.initial_agent = initial_agent
        self.ood_domains = ood_domains
        self.config_path_fn = config_path_fn

    def run(self) -> int:
        pipeline_state = self._load_pipeline_state()
        state = self._load_milestone_state()
        if state.get("status") == "completed":
            self._print_completed_result(state)
            return 0

        dependency_checkpoint = self._resolve_dependency_checkpoint()
        phase1_checkpoint = None
        if self.policy.needs_phase1(self.milestone):
            phase1_checkpoint = self.run_phase1(repo_root=self.repo_root, state_dir=self.state_dir)

        state["status"] = "running"
        overrides = self._initial_overrides_from_state(state)
        oom_retries = 0
        max_runs = self.policy.max_runs_for(self.milestone)

        # Pre-run: let agent reason about initial params
        if not state["runs"] and self.initial_agent and not self.no_agent:
            initial_cfg = self._build_current_config(
                overrides, dependency_checkpoint, phase1_checkpoint,
            )
            initial_overrides = self._request_initial_params(initial_cfg)
            if initial_overrides:
                overrides.update(initial_overrides)

        while len(state["runs"]) < max_runs:
            if self._budget_exceeded(pipeline_state, state):
                state["status"] = "failed"
                state["reason"] = "budget_exceeded"
                self._persist(state, pipeline_state)
                return 1

            cfg = self._build_current_config(overrides, dependency_checkpoint, phase1_checkpoint)
            run_number = len(state["runs"]) + 1

            self._log_run_banner(run_number, max_runs, cfg, overrides)

            try:
                run_record = self._execute_run(run_number, cfg, overrides, dependency_checkpoint)
            except RuntimeError as exc:
                if not self._is_oom(exc):
                    raise
                if oom_retries >= 2:
                    state["status"] = "failed"
                    state["reason"] = "oom_retries_exhausted"
                    self._persist(state, pipeline_state)
                    return 1
                oom_retries += 1
                overrides["train.batch_size"] = max(1, cfg["train"]["batch_size"] // 2)
                grad_accum = cfg["train"].get("gradient_accumulation_steps", 1)
                overrides["train.gradient_accumulation_steps"] = grad_accum * 2
                print(f"OOM — halving batch to {overrides['train.batch_size']}, "
                      f"doubling accum to {overrides['train.gradient_accumulation_steps']}")
                continue

            oom_retries = 0
            state["runs"].append(run_record)
            pipeline_state["budget_used_hours"] = (
                pipeline_state.get("budget_used_hours", 0.0)
                + run_record["training_time_seconds"] / 3600.0
            )

            result = run_record["result"]
            passed = self.policy.check_gate(self.milestone, result)
            diverged = self._is_diverged(result.get("best_val_loss"))

            self._log_gate_result(result, passed)

            if self.policy.record_only_for(self.milestone):
                run_record["decision"] = (
                    "abort" if self.policy.should_early_abort(self.milestone, result) else "accept"
                )
                state["status"] = "completed"
                state["checkpoint_path"] = run_record["checkpoint_path"]
                self._mark_completed(pipeline_state)
                self._persist(state, pipeline_state)
                return 0

            if passed:
                run_record["decision"] = "accept"
                state["status"] = "completed"
                state["accepted_run"] = run_number
                state["checkpoint_path"] = run_record["checkpoint_path"]
                self._mark_completed(pipeline_state)
                self._persist(state, pipeline_state)
                return 0

            if self.policy.should_early_abort(self.milestone, result):
                run_record["decision"] = "abort"
                state["status"] = "completed"
                state["checkpoint_path"] = run_record["checkpoint_path"]
                self._mark_completed(pipeline_state)
                self._persist(state, pipeline_state)
                return 0

            if diverged:
                run_record["decision"] = "retry"
                overrides["train.lr"] = cfg["train"]["lr"] / 2.0
                print(f"\nDiverged — halving LR to {overrides['train.lr']:.2e}")
                self._persist(state, pipeline_state)
                continue

            if self.no_agent:
                fallback = self._mechanical_fallback(cfg, result)
                run_record["decision"] = "retry"
                overrides.update(fallback)
                self._log_fallback(fallback)
                self._persist(state, pipeline_state)
                continue

            try:
                agent_response = self._request_agent_adjustment(state)
            except Exception as exc:
                print(f"\nAgent error: {exc}")
                fallback = self._mechanical_fallback(cfg, result)
                run_record["decision"] = "retry"
                overrides.update(fallback)
                self._log_fallback(fallback)
                self._persist(state, pipeline_state)
                continue

            self._log_agent_response(agent_response)
            run_record["agent_response"] = agent_response
            adjustment = agent_response.get("adjustment", {})
            if adjustment:
                adjustment = self._validate_adjustment(adjustment)
            if not adjustment:
                print(
                    "WARNING: Agent returned empty/invalid "
                    "adjustment, using mechanical fallback"
                )
                adjustment = self._mechanical_fallback(cfg, result)
                self._log_fallback(adjustment)
            # Detect if agent is repeating the same overrides
            candidate = {**overrides, **adjustment}
            prev_overrides = (
                state["runs"][-1].get("config_overrides", {})
                if state["runs"] else {}
            )
            if candidate == prev_overrides and adjustment:
                print(
                    "WARNING: Agent suggested identical params "
                    "to previous run. Augmenting with mechanical "
                    "fallback."
                )
                fallback = self._mechanical_fallback(cfg, result)
                adjustment.update(fallback)
                self._log_fallback(fallback)
            run_record["decision"] = "retry"
            overrides.update(adjustment)
            self._persist(state, pipeline_state)

        state["status"] = "failed"
        self._persist(state, pipeline_state)
        return 1

    # -- Logging helpers (inspired by VLA pipeline) --

    def _log_run_banner(
        self, run_number: int, max_runs: int, cfg: dict[str, Any], overrides: dict[str, Any],
    ) -> None:
        print(f"\n{'=' * 60}")
        print(f"RUN {run_number}/{max_runs} -- {self.milestone}")
        print(f"{'=' * 60}")
        train = cfg.get("train", {})
        print(f"  steps={train.get('steps')}  lr={train.get('lr'):.2e}  "
              f"warmup={train.get('warmup_steps')}  bs={train.get('batch_size')}")
        if overrides:
            print(f"  overrides: {json.dumps(overrides)}")

    def _log_gate_result(self, result: dict[str, Any], passed: bool) -> None:
        gate_desc = self.policy.gate_description_for(self.milestone)
        status = "PASSED" if passed else "FAILED"
        print(f"\n--- GATE {status}: {gate_desc} ---")
        for key in ("test_id_ppl", "test_ood_ppl", "mi_ratio_mean", "sigma_std", "best_val_loss"):
            if key in result:
                val = result[key]
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")

    def _log_agent_response(self, response: dict[str, Any]) -> None:
        print(f"\n--- Agent Response (provider={self.provider.name}) ---")
        print(f"  Diagnosis: {response.get('diagnosis') or '(empty)'}")
        print(f"  Reasoning: {response.get('reasoning') or '(empty)'}")
        adj = response.get("adjustment", {})
        if adj:
            for k, v in adj.items():
                print(f"  {k}: {v}")
        else:
            print("  Adjustment: (none)")

    def _log_fallback(self, fallback: dict[str, Any]) -> None:
        print("\n--- Mechanical Fallback ---")
        if fallback:
            for k, v in fallback.items():
                print(f"  {k}: {v}")
        else:
            print("  (no adjustments available)")

    def write_compare_outputs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        report = self.policy.comparison_report(self.state_dir)
        payload = self.policy.comparison_payload(self.state_dir)
        (self.state_dir / "comparison.txt").write_text(report)
        (self.state_dir / "comparison.json").write_text(json.dumps(payload, indent=2))
        print(report)

    def _build_current_config(
        self,
        overrides: dict[str, Any],
        dependency_checkpoint: str | None,
        phase1_checkpoint: str | None,
    ) -> dict[str, Any]:
        milestone_key = self.policy.milestone_key_for(self.milestone)
        if self.config_path_fn is not None:
            cfg = self._build_config_from_yaml(milestone_key, overrides)
        else:
            cfg = self.policy.build_config(milestone_key, overrides)
        if dependency_checkpoint is not None:
            cfg["train"]["resume_checkpoint"] = dependency_checkpoint
        if phase1_checkpoint is not None:
            cfg["train"]["resume_checkpoint"] = phase1_checkpoint
        return cfg

    def _build_config_from_yaml(
        self, milestone_key: str, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        from minigpt.config import apply_dict_overrides, load_yaml, save_yaml, validate_config

        yaml_path = self.config_path_fn(milestone_key)
        if not yaml_path.exists():
            # Generate initial YAML from template
            cfg = self.policy.build_config(milestone_key, None)
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(yaml_path, cfg)

        cfg = load_yaml(yaml_path)
        if overrides:
            apply_dict_overrides(cfg, overrides)

        validate_config(cfg)
        return cfg

    def _execute_run(
        self,
        run_number: int,
        cfg: dict[str, Any],
        overrides: dict[str, Any],
        dependency_checkpoint: str | None,
    ) -> dict[str, Any]:
        tokenizer, data = self.hooks.setup_data(cfg)
        model, _model_cfg, n_params = self.hooks.setup_model(cfg, tokenizer)
        device = self.hooks.resolve_device(cfg)
        train_cfg = self.hooks.build_train_config(cfg)
        train_data = data.get("train") if isinstance(data, dict) else None
        val_data = data.get("val") if isinstance(data, dict) else None
        num_train_tokens = len(train_data) if train_data is not None else 0

        with self.hooks.mlflow_context(cfg, self.use_mlflow) as mlflow_run:
            if self.use_mlflow:
                self.hooks.log_common_mlflow(cfg, tokenizer, n_params, self.milestone)

            model, train_meta = self.hooks.train(
                model,
                train_data,
                val_data,
                train_cfg,
                mlflow_run=mlflow_run,
                config_dict=cfg,
                kl_weight=cfg["train"].get("kl_weight", 0.0),
                num_train_tokens=num_train_tokens,
            )

            if self.use_mlflow:
                self.hooks.log_train_meta_mlflow(train_meta)

            ppl_result = self._evaluate_perplexity(model, cfg, data, device)
            if self.use_mlflow:
                self.hooks.log_perplexity_mlflow(ppl_result)

            result = {
                "test_id_ppl": ppl_result["test_id_ppl"],
                "test_ood_ppl": ppl_result.get("test_ood_ppl") or self._empty_ood_result(),
                "best_val_loss": train_meta["best_val_loss"],
                "best_val_step": train_meta.get("best_val_step", 0),
                "steps_completed": train_meta.get("steps_completed", cfg["train"].get("steps", 0)),
                "steps_planned": cfg["train"].get("steps", 0),
                "early_stop_reason": train_meta.get("early_stop_reason"),
                "training_time_seconds": train_meta["train_time_sec"],
                "effective_params": {
                    "lr": cfg["train"].get("lr"),
                    "warmup_steps": cfg["train"].get("warmup_steps"),
                    "batch_size": cfg["train"].get("batch_size"),
                    "dropout": cfg["model"].get("dropout"),
                    "gradient_accumulation_steps": cfg["train"].get(
                        "gradient_accumulation_steps", 1
                    ),
                },
                "eval_history": train_meta.get("eval_history", []),
            }

            if self.policy.needs_mi_eval(self.milestone):
                mi_result = self._evaluate_mi(model, cfg, data, device, result["test_ood_ppl"])
                result.update(mi_result)
                if self.use_mlflow:
                    self.hooks.log_mi_mlflow(
                        {"mi_mean": result["mi_id"], "flip_rate": 0.0},
                        {"mi_mean": result["mi_id"] * result["mi_ratio_mean"], "flip_rate": 0.0},
                        result["mi_ratio_mean"],
                    )

            if self.use_mlflow:
                try:
                    report, qual_rows = self.hooks.run_qualitative_suite(
                        model,
                        tokenizer,
                        cfg,
                        device,
                        cfg["eval"].get("num_samples", 20),
                    )
                    self.hooks.log_qualitative_mlflow(report, qual_rows)
                except Exception as exc:
                    print(f"Warning: qualitative eval skipped: {exc}")

        checkpoint_path = str(Path(cfg["train"]["checkpoint_dir"]) / "ckpt_best.pt")
        if dependency_checkpoint is not None and self.policy.reuse_dependency_checkpoint_for(
            self.milestone
        ):
            checkpoint_path = dependency_checkpoint

        mlflow_run_id = ""
        if mlflow_run is not None and getattr(mlflow_run, "info", None) is not None:
            mlflow_run_id = getattr(mlflow_run.info, "run_id", "")

        return {
            "run_number": run_number,
            "mlflow_run_id": mlflow_run_id,
            "config_overrides": copy.deepcopy(overrides),
            "training_time_seconds": train_meta["train_time_sec"],
            "result": result,
            "decision": "",
            "agent_response": {},
            "checkpoint_path": checkpoint_path,
        }

    def _evaluate_perplexity(
        self,
        model: Any,
        cfg: dict[str, Any],
        data: dict[str, Any],
        device: Any,
    ) -> dict[str, Any]:
        n_batches = cfg["eval"].get("n_perplexity_batches", 20)
        ood_data = self._extract_ood_data(data)
        if not isinstance(ood_data, dict):
            return self.hooks.eval_perplexity_suite(
                model,
                cfg,
                data.get("test_id"),
                data.get("test_ood"),
                device,
                n_batches,
                val_data=data.get("val"),
            )

        id_result = self.hooks.eval_perplexity_suite(
            model,
            cfg,
            data.get("test_id"),
            None,
            device,
            n_batches,
            val_data=data.get("val"),
        )
        domain_ppls = {}
        for domain, domain_data in ood_data.items():
            domain_result = self.hooks.eval_perplexity_suite(
                model,
                cfg,
                data.get("test_id"),
                domain_data,
                device,
                n_batches,
                val_data=data.get("val"),
            )
            domain_ppls[domain] = domain_result.get("test_ood_ppl")
        id_result["test_ood_ppl"] = domain_ppls
        return id_result

    def _evaluate_mi(
        self,
        model: Any,
        cfg: dict[str, Any],
        data: dict[str, Any],
        device: Any,
        test_ood_ppl: dict[str, Any],
    ) -> dict[str, Any]:
        n_batches = cfg["eval"].get("n_perplexity_batches", 20)
        n_samples = cfg["eval"].get("num_samples", 20)
        ood_data = self._extract_ood_data(data)
        if isinstance(ood_data, dict):
            ratio_by_domain: dict[str, float] = {}
            mi_id = None
            for domain, domain_data in ood_data.items():
                domain_mi_id, _domain_mi_ood, domain_ratio = self.hooks.eval_mi_suite(
                    self.hooks.uncertainty_eval_fn,
                    model,
                    cfg,
                    data.get("test_id"),
                    domain_data,
                    device,
                    n_samples,
                    n_batches,
                )
                if mi_id is None:
                    mi_id = domain_mi_id
                ratio_by_domain[domain] = domain_ratio
            assert mi_id is not None
            mi_ratio_mean = sum(ratio_by_domain.values()) / max(len(ratio_by_domain), 1)
        else:
            mi_id, _mi_ood, mi_ratio_mean = self.hooks.eval_mi_suite(
                self.hooks.uncertainty_eval_fn,
                model,
                cfg,
                data.get("test_id"),
                data.get("test_ood"),
                device,
                n_samples,
                n_batches,
            )
            ratio_by_domain = {domain: mi_ratio_mean for domain in test_ood_ppl}
        sigma_std = self.hooks.sigma_std_extractor(model)
        return {
            "mi_id": mi_id["mi_mean"],
            "mi_ratio": ratio_by_domain,
            "mi_ratio_mean": mi_ratio_mean,
            "sigma_std": sigma_std,
        }

    def _extract_ood_data(self, data: dict[str, Any]) -> dict[str, Any] | Any:
        ood_dict = {
            key.removeprefix("test_ood_"): value
            for key, value in data.items()
            if key.startswith("test_ood_")
        }
        if ood_dict:
            return ood_dict
        if isinstance(data.get("test_ood"), dict):
            return data["test_ood"]
        return data.get("test_ood")

    def _empty_ood_result(self) -> dict[str, Any]:
        return {domain: None for domain in self.ood_domains}

    def _load_json(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return copy.deepcopy(default)
        return json.loads(path.read_text())

    def _save_json(self, path: Path, payload: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        self.hooks.os_replace(str(tmp_path), str(path))

    def _load_pipeline_state(self) -> dict[str, Any]:
        return self._load_json(
            self.state_dir / "pipeline.json",
            {"completed": [], "budget_used_hours": 0.0},
        )

    def _load_milestone_state(self) -> dict[str, Any]:
        return self._load_json(
            self.state_dir / f"{self.milestone}.json",
            {
                "status": "pending",
                "runs": [],
                "accepted_run": None,
                "checkpoint_path": None,
            },
        )

    def _persist(self, state: dict[str, Any], pipeline_state: dict[str, Any]) -> None:
        self._save_json(self.state_dir / f"{self.milestone}.json", state)
        self._save_json(self.state_dir / "pipeline.json", pipeline_state)

    def _mark_completed(self, pipeline_state: dict[str, Any]) -> None:
        if self.milestone not in pipeline_state["completed"]:
            pipeline_state["completed"].append(self.milestone)

    def _print_completed_result(self, state: dict[str, Any]) -> None:
        accepted_run = state.get("accepted_run") or 1
        run = state.get("runs", [])[accepted_run - 1]
        print(json.dumps(run.get("result", {})))

    def _resolve_dependency_checkpoint(self) -> str | None:
        prerequisite = self.policy.dependency_for(self.milestone)
        if prerequisite is None:
            return None
        prereq = self._load_json(self.state_dir / f"{prerequisite}.json", {})
        if prereq.get("status") != "completed" or not prereq.get("checkpoint_path"):
            raise FileNotFoundError(f"Missing {prerequisite} checkpoint")
        return prereq["checkpoint_path"]

    def _initial_overrides_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        runs = state.get("runs", [])
        if not runs:
            return {}
        return copy.deepcopy(runs[-1].get("config_overrides", {}))

    def _request_agent_adjustment(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_agent_prompt(state)
        print(f"\n[provider] launching provider={self.provider.name} role=researcher")
        response = self.provider.run_role(
            role="researcher",
            prompt=prompt,
            repo_root=self.repo_root,
            schema=_AGENT_RESPONSE_SCHEMA,
        )
        return self.policy.parse_agent_response(response.output)

    def _load_agent_briefing(self) -> str:
        briefing_path = Path(__file__).parent / "agent_briefing.md"
        if briefing_path.exists():
            return briefing_path.read_text(encoding="utf-8")
        return ""

    def _request_initial_params(
        self, cfg: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_initial_prompt(cfg)
        print(
            f"\n[provider] pre-run reasoning: "
            f"provider={self.provider.name}"
        )
        try:
            response = self.provider.run_role(
                role="researcher",
                prompt=prompt,
                repo_root=self.repo_root,
                schema=_AGENT_RESPONSE_SCHEMA,
            )
            parsed = self.policy.parse_agent_response(response.output)
            self._log_agent_response(parsed)
            adjustment = parsed.get("adjustment", {})
            if adjustment:
                return self._validate_adjustment(adjustment)
        except Exception as exc:
            print(f"\nInitial agent call failed: {exc}")
            print("Using template defaults.")
        return {}

    def _build_initial_prompt(self, cfg: dict[str, Any]) -> str:
        knob_family = self.policy.knob_family_for(self.milestone)
        knobs = self.policy.tunable_knobs.get(knob_family, {})
        knob_lines = "\n".join(
            f"  {k}: [{lo}, {hi}]"
            for k, (lo, hi) in sorted(knobs.items())
        )
        train = cfg.get("train", {})
        data = cfg.get("data", {})
        model = cfg.get("model", {})
        briefing = self._load_agent_briefing()

        # Compute training scale for agent reasoning
        bs = train.get("batch_size", 16)
        accum = train.get("gradient_accumulation_steps", 1)
        block = train.get("block_size", 256)
        steps = train.get("steps", 100_000)
        eff_batch = bs * accum
        tokens_per_step = eff_batch * block
        total_tokens = tokens_per_step * steps
        id_tokens = data.get("pile_id_tokens", 100_000_000)
        data_passes = total_tokens / id_tokens if id_tokens else 0

        return (
            "You are a research assistant planning the FIRST "
            "run of a Bayesian LLM experiment.\n"
            "\n"
            f"Milestone: {self.milestone}\n"
            f"Success gate: "
            f"{self.policy.gate_description_for(self.milestone)}\n"
            "\n"
            "Current template defaults:\n"
            f"  train.steps: {steps}\n"
            f"  train.lr: {train.get('lr')}\n"
            f"  train.warmup_steps: {train.get('warmup_steps')}\n"
            f"  train.batch_size: {bs}\n"
            f"  train.gradient_accumulation_steps: {accum}\n"
            f"  model.dropout: {model.get('dropout')}\n"
            f"  model.n_layer: {model.get('n_layer')}\n"
            f"  model.n_head: {model.get('n_head')}\n"
            f"  model.n_embd: {model.get('n_embd')}\n"
            f"  data.dataset: {data.get('dataset')}\n"
            f"  data.pile_id_domains: {data.get('pile_id_domains')}\n"
            f"  data.pile_id_tokens: {id_tokens}\n"
            "\n"
            "Derived training scale:\n"
            f"  effective_batch_size: {eff_batch} "
            f"({bs} × {accum})\n"
            f"  tokens_per_step: {tokens_per_step:,}\n"
            f"  total_training_tokens: {total_tokens:,}\n"
            f"  data_passes: {data_passes:.1f}x over "
            f"{id_tokens:,} ID tokens\n"
            "\n"
            f"Tunable knobs (name: [min, max]):\n{knob_lines}\n"
            "\n"
            "=== REFERENCE: Research context and HP tuning "
            "playbook (from AGENTS.md and prior experiments) "
            "===\n"
            f"{briefing}\n"
            "=== END REFERENCE ===\n"
            "\n"
            "This is the FIRST run. No history yet.\n"
            "\n"
            "Think step by step as a deep learning researcher:\n"
            "1. For a 76M-param GPT on this corpus, what LR "
            "and schedule will converge to PPL < 80?\n"
            "2. Compare the defaults to GPT-2-small "
            "(117M, lr=2.5e-4, warmup=2000, batch=512). "
            "Our batch is much smaller — should LR be higher?\n"
            "3. Is dropout appropriate for this depth?\n"
            "4. Are 100K steps sufficient given the "
            "tokens/step and data volume?\n"
            "\n"
            "Read AGENTS.md in this repo for full experiment "
            "history, 4-layer reference results, and prior "
            "bugs. Use that context to make informed "
            "decisions.\n"
            "\n"
            "If the defaults look good, set adjustment to {}. "
            "Otherwise, specify the knobs to change.\n"
            "\n"
            "Respond with a JSON object with this shape:\n"
            '{"diagnosis": "your analysis of template '
            'defaults (be specific)",'
            ' "reasoning": "why you chose these params '
            '(cite evidence from AGENTS.md or ML literature)",'
            ' "adjustment": {"knob.name": new_value}}\n'
        )

    def _build_diagnostic_summary(self, state: dict[str, Any]) -> str:
        runs = state.get("runs", [])
        if not runs:
            return "No runs yet."

        lines = []
        gate_desc = self.policy.gate_description_for(self.milestone)
        latest = runs[-1].get("result", {})

        # Effective params used
        eff = latest.get("effective_params", {})
        if eff:
            lines.append(
                f"Effective params: lr={eff.get('lr')}, "
                f"warmup={eff.get('warmup_steps')}, "
                f"bs={eff.get('batch_size')}, "
                f"dropout={eff.get('dropout')}, "
                f"grad_accum={eff.get('gradient_accumulation_steps')}"
            )

        # Gap analysis
        ppl = latest.get("test_id_ppl")
        if ppl is not None:
            gap = ppl / 80
            severity = (
                "EXTREME — needs aggressive LR/steps changes"
                if gap > 10
                else "LARGE — significant HP changes needed"
                if gap > 3
                else "MODERATE — fine-tuning may suffice"
                if gap > 1.5
                else "CLOSE — minor adjustments"
            )
            lines.append(
                f"Current test_id_ppl={ppl:.1f}, target <80, "
                f"gap={gap:.1f}x. Severity: {severity}."
            )

        mi = latest.get("mi_ratio_mean")
        if mi is not None:
            target = 1.2 if "1.2" in gate_desc else 1.05
            lines.append(
                f"Current mi_ratio_mean={mi:.3f}, "
                f"target >{target}."
            )

        # Step efficiency & convergence analysis
        best_step = latest.get("best_val_step", 0)
        steps_completed = latest.get("steps_completed", 0)
        planned = latest.get("steps_planned", 0)
        if planned > 0 and best_step > 0:
            util = best_step / planned * 100
            lines.append(
                f"Best val loss at step {best_step}/{planned} "
                f"planned ({util:.0f}% utilization)."
            )
        if steps_completed > 0 and planned > 0:
            lines.append(
                f"Training completed {steps_completed}/{planned} "
                f"steps ({steps_completed / planned * 100:.0f}%)."
            )
        # Convergence pattern
        if best_step > 0 and steps_completed > 0:
            wasted = steps_completed - best_step
            if wasted > 0:
                wasted_pct = wasted / steps_completed * 100
                if wasted_pct > 50:
                    lines.append(
                        f"WARNING: {wasted_pct:.0f}% of training "
                        f"was wasted after best step ({wasted} "
                        f"steps with no meaningful improvement). "
                        f"This is a strong signal that the model "
                        f"is stuck. Increase LR or reduce steps."
                    )

        # Early stop
        reason = latest.get("early_stop_reason")
        if reason == "patience":
            lines.append(
                "EARLY STOP: Loss plateaued (patience "
                "exhausted). The model converged to a local "
                "minimum. Primary fix: increase LR to escape "
                "the plateau. Secondary: adjust warmup ratio "
                "or dropout."
            )
        elif reason == "nan":
            lines.append(
                "EARLY STOP: NaN loss detected. LR is too "
                "high or warmup too short. Halve LR or "
                "double warmup_steps."
            )

        # Loss curve summary (from eval history)
        eval_hist = latest.get("eval_history", [])
        if len(eval_hist) >= 3:
            lines.append("")
            lines.append("Loss curve (val_loss @ step):")
            # Show first, best, last, and a few samples
            sampled = self._summarize_eval_history(eval_hist)
            for entry in sampled:
                marker = ""
                if entry["step"] == best_step:
                    marker = " <-- best"
                lines.append(
                    f"  step {entry['step']:>6d}: "
                    f"val={entry['val_loss']:.4f} "
                    f"train={entry['train_loss']:.4f} "
                    f"lr={entry['lr']:.2e}{marker}"
                )
            # Convergence rate
            first_loss = eval_hist[0]["val_loss"]
            best_loss = latest.get("best_val_loss", first_loss)
            total_drop = first_loss - best_loss
            if total_drop > 0 and len(eval_hist) >= 2:
                # Where did 90% of the drop happen?
                drop_90 = first_loss - 0.9 * total_drop
                for entry in eval_hist:
                    if entry["val_loss"] <= drop_90:
                        pct_of_training = (
                            entry["step"] / eval_hist[-1]["step"]
                            * 100
                        )
                        lines.append(
                            f"90% of loss improvement happened "
                            f"by step {entry['step']} "
                            f"({pct_of_training:.0f}% of training)."
                        )
                        break

        # Run-over-run deltas
        if len(runs) >= 2:
            lines.append("")
            lines.append("Run-over-run changes:")
            for i in range(1, len(runs)):
                prev_r = runs[i - 1].get("result", {})
                curr_r = runs[i].get("result", {})
                overrides = runs[i].get("config_overrides", {})
                prev_ppl = prev_r.get("test_id_ppl")
                curr_ppl = curr_r.get("test_id_ppl")
                delta = ""
                if prev_ppl and curr_ppl:
                    if curr_ppl < prev_ppl:
                        delta = (
                            f"PPL {prev_ppl:.0f}->"
                            f"{curr_ppl:.0f} (improved)"
                        )
                    elif curr_ppl > prev_ppl:
                        delta = (
                            f"PPL {prev_ppl:.0f}->"
                            f"{curr_ppl:.0f} (regressed)"
                        )
                    else:
                        delta = f"PPL unchanged at {curr_ppl:.0f}"
                override_str = ", ".join(
                    f"{k}={v}" for k, v in overrides.items()
                )
                prev_reason = prev_r.get("early_stop_reason", "")
                reason_note = (
                    f" [prev stopped: {prev_reason}]"
                    if prev_reason else ""
                )
                lines.append(
                    f"  Run {i}->{i + 1}: "
                    f"changed [{override_str}] -> "
                    f"{delta}{reason_note}"
                )

        return "\n".join(lines)

    def _build_agent_prompt(self, state: dict[str, Any]) -> str:
        knob_family = self.policy.knob_family_for(self.milestone)
        knobs = self.policy.tunable_knobs.get(knob_family, {})
        knob_lines = "\n".join(
            f"  {k}: [{lo}, {hi}]" for k, (lo, hi) in sorted(knobs.items())
        )
        history_lines = []
        for run in state.get("runs", []):
            # Exclude eval_history from verbose dump (already
            # summarized in diagnostics)
            result_for_prompt = {
                k: v for k, v in run.get("result", {}).items()
                if k != "eval_history"
            }
            history_lines.append(
                f"Run {run['run_number']}: "
                f"result={json.dumps(result_for_prompt, indent=2)}\n"
                f"  overrides={run.get('config_overrides', {})}"
            )
        history_text = (
            "\n".join(history_lines) if history_lines
            else "No previous runs."
        )
        diagnostic = self._build_diagnostic_summary(state)
        briefing = self._load_agent_briefing()
        return (
            "You are a research assistant optimizing "
            "hyperparameters for a Bayesian LLM experiment.\n"
            "\n"
            f"Milestone: {self.milestone}\n"
            f"Success gate: "
            f"{self.policy.gate_description_for(self.milestone)}\n"
            "\n"
            f"Tunable knobs (name: [min, max]):\n{knob_lines}\n"
            "\n"
            f"Run history:\n{history_text}\n"
            "\n"
            f"Diagnostic summary:\n{diagnostic}\n"
            "\n"
            "=== REFERENCE: Research context and HP tuning "
            "playbook (from AGENTS.md and prior experiments) "
            "===\n"
            f"{briefing}\n"
            "=== END REFERENCE ===\n"
            "\n"
            "Read AGENTS.md in this repo for full experiment "
            "history, 4-layer reference results, and known "
            "failure patterns. Use that context.\n"
            "\n"
            "Think step by step as a deep learning researcher:\n"
            "1. Diagnose the root cause. Look at the loss "
            "curve, early_stop_reason, steps_completed vs "
            "steps_planned, and gap severity.\n"
            "2. Check what was already tried. Do NOT repeat "
            "an adjustment that failed before.\n"
            "3. Propose a SPECIFIC fix. Cite evidence from "
            "the run history or playbook.\n"
            "\n"
            "IMPORTANT: You MUST return a non-empty "
            "adjustment dict with at least one knob change.\n"
            "\n"
            "Respond with a JSON object with this shape:\n"
            '{"diagnosis": "root cause of gate failure '
            '(be specific, cite metrics)",'
            ' "reasoning": "what to change and why '
            '(cite prior results or ML principles)",'
            ' "adjustment": {"knob.name": new_value}}\n'
        )

    def _validate_adjustment(self, adjustment: dict[str, Any]) -> dict[str, Any]:
        if not adjustment:
            return {}
        knob_family = self.policy.knob_family_for(self.milestone)
        allowed = self.policy.tunable_knobs.get(knob_family, {})
        validated: dict[str, Any] = {}
        for key, value in adjustment.items():
            if key not in allowed:
                print(f"WARNING: agent suggested unknown knob "
                      f"'{key}', skipping")
                continue
            lower, upper = allowed[key]
            if value < lower or value > upper:
                clamped = max(lower, min(value, upper))
                print(f"WARNING: agent value {key}={value} "
                      f"out of range [{lower}, {upper}], "
                      f"clamping to {clamped}")
                validated[key] = clamped
            else:
                validated[key] = value
        return validated

    @staticmethod
    def _summarize_eval_history(
        history: list[dict[str, float]], max_points: int = 10,
    ) -> list[dict[str, float]]:
        """Sample key points from eval history for the agent prompt."""
        if len(history) <= max_points:
            return history
        # Always include first and last
        indices = {0, len(history) - 1}
        # Find best val_loss entry
        best_idx = min(
            range(len(history)),
            key=lambda i: history[i]["val_loss"],
        )
        indices.add(best_idx)
        # Evenly space remaining points
        remaining = max_points - len(indices)
        if remaining > 0:
            step = len(history) / (remaining + 1)
            for i in range(1, remaining + 1):
                indices.add(int(i * step))
        return [history[i] for i in sorted(indices)]

    def _mechanical_fallback(
        self,
        cfg: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if self._is_diverged(result.get("best_val_loss")):
            return {"train.lr": cfg["train"]["lr"] / 2.0}

        knob_family = self.policy.knob_family_for(self.milestone)
        early_reason = result.get("early_stop_reason")

        if knob_family == "c0":
            current_lr = cfg["train"].get("lr", 3e-4)
            lr_range = self.policy.tunable_knobs.get("c0", {}).get(
                "train.lr", (1e-5, 1e-3),
            )
            # Patience plateau → model is stuck → increase LR
            if early_reason == "patience":
                new_lr = min(current_lr * 2.0, lr_range[1])
                if new_lr != current_lr:
                    return {"train.lr": new_lr}
            # Completed all steps but PPL still high → more steps
            current_steps = cfg["train"].get("steps", 100_000)
            step_range = self.policy.tunable_knobs.get("c0", {}).get(
                "train.steps", (50_000, 300_000),
            )
            new_steps = min(
                max(current_steps * 2, step_range[0]), step_range[1],
            )
            if new_steps != current_steps:
                warmup_range = self.policy.tunable_knobs.get("c0", {}).get(
                    "train.warmup_steps", (500, 10_000),
                )
                new_warmup = int(min(
                    max(new_steps * 0.04, warmup_range[0]),
                    warmup_range[1],
                ))
                return {
                    "train.steps": new_steps,
                    "train.warmup_steps": new_warmup,
                }
            # LR increase as last resort
            new_lr = min(current_lr * 2.0, lr_range[1])
            if new_lr != current_lr:
                return {"train.lr": new_lr}
            return {}

        if knob_family == "c1" and result.get("sigma_std", 0.0) < 0.01:
            init_rho = cfg["model"]["bayes_ffn"].get("init_rho", -2.0)
            return {"model.bayes_ffn.init_rho": init_rho + 1.0}

        if knob_family == "c3" and result.get("sigma_std", 0.0) < 0.01:
            init_g = cfg.get("lora", {}).get("init_g", 0.1)
            return {"lora.init_g": min(init_g * 2.0, 1.0)}

        return {}

    def _is_oom(self, exc: RuntimeError) -> bool:
        return "out of memory" in str(exc).lower()

    def _is_diverged(self, best_val_loss: Any) -> bool:
        try:
            return math.isnan(best_val_loss) or best_val_loss > 100
        except TypeError:
            return False

    def _budget_exceeded(
        self,
        pipeline_state: dict[str, Any],
        state: dict[str, Any],
    ) -> bool:
        used_hours = pipeline_state.get("budget_used_hours", 0.0)
        if state.get("runs"):
            estimate_seconds = state["runs"][-1].get("training_time_seconds", 3600.0)
            estimate_hours = estimate_seconds / 3600.0
        else:
            estimate_hours = self.policy.default_run_estimate_hours
        return used_hours + estimate_hours > self.budget_hours
