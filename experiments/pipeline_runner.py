"""Generic research pipeline orchestration."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


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
        self.ood_domains = ood_domains

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

        while len(state["runs"]) < self.policy.max_runs_for(self.milestone):
            if self._budget_exceeded(pipeline_state, state):
                state["status"] = "failed"
                state["reason"] = "budget_exceeded"
                self._persist(state, pipeline_state)
                return 1

            cfg = self._build_current_config(overrides, dependency_checkpoint, phase1_checkpoint)
            run_number = len(state["runs"]) + 1

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
                self._persist(state, pipeline_state)
                continue

            if self.no_agent:
                run_record["decision"] = "retry"
                overrides.update(self._mechanical_fallback(cfg, result))
                self._persist(state, pipeline_state)
                continue

            try:
                agent_response = self._request_agent_adjustment(state)
            except Exception:
                run_record["decision"] = "retry"
                overrides.update(self._mechanical_fallback(cfg, result))
                self._persist(state, pipeline_state)
                continue

            adjustment = self._validate_adjustment(agent_response.get("adjustment", {}))
            run_record["agent_response"] = agent_response
            run_record["decision"] = "retry"
            overrides.update(adjustment)
            self._persist(state, pipeline_state)

        state["status"] = "failed"
        self._persist(state, pipeline_state)
        return 1

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
        cfg = self.policy.build_config(milestone_key, overrides)
        if dependency_checkpoint is not None:
            cfg["train"]["resume_checkpoint"] = dependency_checkpoint
        if phase1_checkpoint is not None:
            cfg["train"]["resume_checkpoint"] = phase1_checkpoint
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
                "training_time_seconds": train_meta["train_time_sec"],
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
        response = self.provider.run_role(
            role="researcher",
            prompt=prompt,
            repo_root=self.repo_root,
            schema=None,
        )
        return self.policy.parse_agent_response(response.output)

    def _build_agent_prompt(self, state: dict[str, Any]) -> str:
        knob_family = self.policy.knob_family_for(self.milestone)
        allowed = ", ".join(sorted(self.policy.tunable_knobs.get(knob_family, {})))
        history_lines = []
        for run in state.get("runs", []):
            history_lines.append(
                f"Run {run['run_number']}: result={run.get('result', {})} "
                f"overrides={run.get('config_overrides', {})}"
            )
        history_text = "\n".join(history_lines) if history_lines else "No previous runs."
        return (
            f"Milestone: {self.milestone}\n"
            f"Success gate: {self.policy.gate_description_for(self.milestone)}\n"
            f"Tunable knobs: {allowed}\n"
            "Run history:\n"
            f"{history_text}\n"
            "Read AGENTS.md before proposing any adjustment.\n"
            "Return JSON with diagnosis, reasoning, and adjustment.\n"
        )

    def _validate_adjustment(self, adjustment: dict[str, Any]) -> dict[str, Any]:
        if not adjustment:
            return {}
        knob_family = self.policy.knob_family_for(self.milestone)
        allowed = self.policy.tunable_knobs.get(knob_family, {})
        for key, value in adjustment.items():
            if key not in allowed:
                raise ValueError(f"{key} not allowed")
            lower, upper = allowed[key]
            if value < lower or value > upper:
                raise ValueError(f"{key} out of range")
        return adjustment

    def _mechanical_fallback(
        self,
        cfg: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if self._is_diverged(result.get("best_val_loss")):
            return {"train.lr": cfg["train"]["lr"] / 2.0}

        knob_family = self.policy.knob_family_for(self.milestone)
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
