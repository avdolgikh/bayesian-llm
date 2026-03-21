import copy
import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from minigpt.config import DEFAULT_CONFIG, validate_config


class FakeProvider:
    name = "fake"

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def run_role(self, *, role, prompt, repo_root, schema=None):
        self.calls.append(
            {"role": role, "prompt": prompt, "repo_root": repo_root, "schema": schema}
        )
        if self.error is not None:
            raise self.error
        payload = self.responses.pop(0) if self.responses else {}
        return SimpleNamespace(
            provider="fake",
            role=role,
            model="fake-model",
            output=json.dumps(payload),
        )


@pytest.fixture
def pipeline_module(monkeypatch):
    experiments_dir = Path(__file__).resolve().parents[1] / "experiments"
    monkeypatch.syspath_prepend(str(experiments_dir))
    sys.modules.pop("c_pipeline", None)
    return importlib.import_module("c_pipeline")


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / ".pipeline-state"


def make_runner(module, state_dir, *, milestone="c1", provider=None, budget_hours=12.0):
    return module.PipelineRunner(
        repo_root=Path(__file__).resolve().parents[1],
        milestone=milestone,
        provider=provider or FakeProvider(),
        state_dir=state_dir,
        budget_hours=budget_hours,
        use_mlflow=False,
    )


def result_dict(**overrides):
    result = {
        "test_id_ppl": 52.0,
        "test_ood_ppl": {"arxiv": 220.0, "freelaw": 250.0, "pubmed_abstracts": 210.0},
        "mi_id": 0.05,
        "mi_ratio": {"arxiv": 1.40, "freelaw": 1.20, "pubmed_abstracts": 1.30},
        "mi_ratio_mean": 1.25,
        "sigma_std": 0.07,
        "best_val_loss": 4.2,
        "training_time_seconds": 3600.0,
    }
    result.update(overrides)
    return result


def load_json(path):
    return json.loads(Path(path).read_text())


def install_runner_stubs(module, monkeypatch, results, gates, seen_cfgs=None):
    seen_cfgs = seen_cfgs if seen_cfgs is not None else []
    run_idx = {"value": 0}

    def fake_build_milestone_config(milestone, overrides=None):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["data"]["dataset"] = "pile"
        cfg["experiment"]["name"] = milestone
        cfg["train"]["checkpoint_dir"] = f"data/checkpoints/{milestone}"
        for key, value in (overrides or {}).items():
            cursor = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        seen_cfgs.append(copy.deepcopy(cfg))
        return cfg

    def fake_train(*args, **kwargs):
        run_idx["value"] += 1
        current = results[run_idx["value"] - 1]
        return object(), {
            "best_val_loss": current["best_val_loss"],
            "best_val_step": 1000,
            "train_time_sec": current["training_time_seconds"],
            "tokens_per_sec": 1234.0,
        }

    def fake_eval_perplexity_suite(*args, **kwargs):
        current = results[run_idx["value"] - 1]
        return {
            "test_id_ppl": current["test_id_ppl"],
            "test_ood_ppl": current["test_ood_ppl"],
        }

    def fake_eval_mi_suite(*args, **kwargs):
        current = results[run_idx["value"] - 1]
        mi_id = {"mi_mean": current["mi_id"], "flip_rate": 0.1, "predictive_entropy_mean": 3.0}
        mi_ood = {
            "mi_mean": current["mi_id"] * current["mi_ratio_mean"],
            "flip_rate": 0.2,
            "predictive_entropy_mean": 4.0,
        }
        return mi_id, mi_ood, current["mi_ratio_mean"]

    gate_iter = iter(gates)
    monkeypatch.setattr(module, "build_milestone_config", fake_build_milestone_config)
    monkeypatch.setattr(
        module,
        "setup_data",
        lambda cfg: (SimpleNamespace(n_vocab=50257), {}),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "setup_model",
        lambda cfg, tok: (object(), object(), 76000000),
        raising=False,
    )
    monkeypatch.setattr(module, "resolve_device", lambda cfg: "cpu", raising=False)
    monkeypatch.setattr(module, "train", fake_train, raising=False)
    monkeypatch.setattr(module, "eval_perplexity_suite", fake_eval_perplexity_suite, raising=False)
    monkeypatch.setattr(module, "eval_mi_suite", fake_eval_mi_suite, raising=False)
    monkeypatch.setattr(module, "check_gate", lambda milestone, result: next(gate_iter))
    monkeypatch.setattr(
        module,
        "mlflow_context",
        lambda cfg, use_mlflow: nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="run-1"))),
        raising=False,
    )
    for name in [
        "log_common_mlflow",
        "log_train_meta_mlflow",
        "log_perplexity_mlflow",
        "log_mi_mlflow",
        "log_qualitative_mlflow",
    ]:
        monkeypatch.setattr(
            module,
            name,
            lambda *args, **kwargs: None,
            raising=False,
        )
    monkeypatch.setattr(
        module,
        "run_qualitative_suite",
        lambda *args, **kwargs: ("", []),
        raising=False,
    )


class TestCli:
    @pytest.mark.parametrize(
        "flag",
        [
            "--milestone",
            "--compare",
            "--resume",
            "--provider",
            "--dry-run",
            "--budget",
            "--state-dir",
            "--no-mlflow",
            "--no-agent",
            "--provider-model",
        ],
    )
    def test_help_mentions_expected_flag(self, pipeline_module, monkeypatch, capsys, flag):
        monkeypatch.setattr(sys, "argv", ["c_pipeline.py", "--help"])
        with pytest.raises(SystemExit):
            pipeline_module.main()
        assert flag in capsys.readouterr().out

    def test_milestone_and_compare_are_mutually_exclusive(
        self, pipeline_module, monkeypatch,
    ):
        monkeypatch.setattr(sys, "argv", ["c_pipeline.py", "--milestone", "c0", "--compare"])
        with pytest.raises(SystemExit):
            pipeline_module.main()


class TestConfigGeneration:
    def test_c0_template_matches_16_layer_pile_baseline(self, pipeline_module):
        cfg = pipeline_module.build_milestone_config("c0")
        assert cfg["data"]["dataset"] == "pile"
        assert cfg["model"]["n_layer"] == 16
        assert cfg["model"]["n_head"] == 8
        assert cfg["model"]["n_embd"] == 512
        assert cfg["model"]["bayes_ffn"]["enabled"] is False

    def test_c1_template_enables_bayesian_ffn(self, pipeline_module):
        cfg = pipeline_module.build_milestone_config("c1")
        assert cfg["model"]["bayes_ffn"]["enabled"] is True
        assert cfg["model"]["bayes_ffn"]["init_rho"] == pytest.approx(-2.0)
        assert cfg["train"]["kl_weight"] == pytest.approx(0.2)

    def test_c3_phase2_template_contains_lora_defaults(self, pipeline_module):
        cfg = pipeline_module.build_milestone_config("c3_phase2")
        assert cfg["data"]["pile_id_domains"] == ["hackernews"]
        assert cfg["lora"]["rank"] == 16
        assert cfg["lora"]["alpha"] == pytest.approx(32.0)
        assert cfg["lora"]["init_g"] == pytest.approx(0.1)

    def test_overrides_are_applied_to_generated_config(self, pipeline_module):
        cfg = pipeline_module.build_milestone_config("c0", {"train.lr": 1.0e-4})
        assert cfg["train"]["lr"] == pytest.approx(1.0e-4)

    @pytest.mark.parametrize("milestone", ["c0", "c1", "c2", "c3_phase2", "c4_tfb"])
    def test_generated_templates_validate(self, pipeline_module, milestone):
        validate_config(pipeline_module.build_milestone_config(milestone))


class TestPureHelpers:
    def test_parse_agent_response_returns_structured_adjustment(self, pipeline_module):
        parsed = pipeline_module.parse_agent_response(
            json.dumps(
                {
                    "diagnosis": "posteriors_frozen",
                    "reasoning": "sigma std is too low",
                    "adjustment": {"model.bayes_ffn.init_rho": -1.0},
                }
            )
        )
        assert parsed["diagnosis"] == "posteriors_frozen"
        assert parsed["adjustment"] == {"model.bayes_ffn.init_rho": -1.0}

    @pytest.mark.parametrize(
        ("milestone", "result", "expected"),
        [
            (
                "c0",
                {"test_id_ppl": 45.0, "test_ood_ppl": {"arxiv": 120.0, "freelaw": 80.0}},
                True,
            ),
            (
                "c0",
                {"test_id_ppl": 45.0, "test_ood_ppl": {"arxiv": 70.0, "freelaw": 80.0}},
                False,
            ),
            ("c1", {"mi_ratio_mean": 1.21}, True),
            ("c1", {"mi_ratio_mean": 1.19}, False),
            ("c3", {"mi_ratio_mean": 1.06}, True),
            ("c4_tfb", {"mi_ratio_mean": 1.00}, True),
        ],
    )
    def test_check_gate_matches_spec(self, pipeline_module, milestone, result, expected):
        assert pipeline_module.check_gate(milestone, result) is expected

class TestRunnerStateAndLoop:
    def test_first_success_creates_state_files_and_accepts_run(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [True])
        exit_code = make_runner(pipeline_module, state_dir).run()
        assert exit_code == 0
        assert state_dir.exists()
        state = load_json(state_dir / "c1.json")
        assert state["status"] == "completed"
        assert state["accepted_run"] == 1
        assert len(state["runs"]) == 1

    def test_pipeline_json_tracks_completed_milestones(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [True])
        make_runner(pipeline_module, state_dir, milestone="c1").run()
        pipeline_state = load_json(state_dir / "pipeline.json")
        assert "c1" in pipeline_state["completed"]

    def test_failed_gate_invokes_provider_and_retries(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        provider = FakeProvider(
            responses=[
                {
                    "diagnosis": "posteriors_frozen",
                    "reasoning": "increase init_rho",
                    "adjustment": {"model.bayes_ffn.init_rho": -1.0},
                }
            ]
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.05, sigma_std=0.004), result_dict(mi_ratio_mean=1.24)],
            [False, True],
        )
        make_runner(pipeline_module, state_dir, provider=provider).run()
        assert len(provider.calls) == 1
        assert provider.calls[0]["role"] == "researcher"

    def test_agent_prompt_includes_milestone_gate_knobs_history_and_agents_hint(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        provider = FakeProvider(
            responses=[
                {
                    "diagnosis": "posteriors_frozen",
                    "reasoning": "increase init_rho",
                    "adjustment": {"model.bayes_ffn.init_rho": -1.0},
                },
                {
                    "diagnosis": "low_mi",
                    "reasoning": "increase kl",
                    "adjustment": {"train.kl_weight": 0.3},
                },
            ]
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [
                result_dict(mi_ratio_mean=1.05, sigma_std=0.004),
                result_dict(mi_ratio_mean=1.11, sigma_std=0.02),
                result_dict(mi_ratio_mean=1.24),
            ],
            [False, False, True],
        )
        make_runner(pipeline_module, state_dir, provider=provider).run()
        prompt = provider.calls[1]["prompt"]
        assert "c1" in prompt.lower()
        assert "mi_ratio_mean" in prompt
        assert "1.2" in prompt
        assert "init_rho" in prompt
        assert "kl_weight" in prompt
        assert "run 1" in prompt.lower()
        assert "run 2" in prompt.lower()
        assert "AGENTS.md" in prompt

    def test_agent_adjustment_is_applied_to_next_config(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        seen_cfgs = []
        provider = FakeProvider(
            responses=[
                {
                    "diagnosis": "posteriors_frozen",
                    "reasoning": "increase init_rho",
                    "adjustment": {"model.bayes_ffn.init_rho": -1.0},
                }
            ]
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.05, sigma_std=0.004), result_dict(mi_ratio_mean=1.24)],
            [False, True],
            seen_cfgs,
        )
        make_runner(pipeline_module, state_dir, provider=provider).run()
        assert seen_cfgs[1]["model"]["bayes_ffn"]["init_rho"] == pytest.approx(-1.0)

    def test_agent_response_is_persisted_in_milestone_state(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        provider = FakeProvider(
            responses=[
                {
                    "diagnosis": "posteriors_frozen",
                    "reasoning": "increase init_rho",
                    "adjustment": {"model.bayes_ffn.init_rho": -1.0},
                }
            ]
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [
                result_dict(mi_ratio_mean=1.05, sigma_std=0.004),
                result_dict(mi_ratio_mean=1.24),
            ],
            [False, True],
        )
        make_runner(pipeline_module, state_dir, provider=provider).run()
        state = load_json(state_dir / "c1.json")
        assert state["runs"][0]["agent_response"]["diagnosis"] == "posteriors_frozen"
        assert state["runs"][0]["agent_response"]["adjustment"] == {
            "model.bayes_ffn.init_rho": -1.0
        }

    def test_four_consecutive_failures_mark_milestone_failed(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        provider = FakeProvider(
            responses=[
                {
                    "diagnosis": "low_mi",
                    "reasoning": "raise steps",
                    "adjustment": {"train.steps": 1},
                },
                {
                    "diagnosis": "low_mi",
                    "reasoning": "raise kl",
                    "adjustment": {"train.kl_weight": 0.3},
                },
            ]
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.01)] * 3,
            [False] * 3,
        )
        exit_code = make_runner(pipeline_module, state_dir, provider=provider).run()
        assert exit_code != 0
        assert load_json(state_dir / "c1.json")["status"] == "failed"

    def test_resume_completed_milestone_prints_existing_result(
        self, pipeline_module, monkeypatch, capsys, state_dir,
    ):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "c1.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "runs": [{"run_number": 1, "result": {"mi_ratio_mean": 1.28}}],
                    "accepted_run": 1,
                    "checkpoint_path": "data/checkpoints/c1/run1/ckpt_best.pt",
                }
            )
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["c_pipeline.py", "--resume", "c1", "--state-dir", str(state_dir)],
        )
        pipeline_module.main()
        assert "1.28" in capsys.readouterr().out

    def test_resume_incomplete_milestone_preserves_history_and_continues(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "c1.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "runs": [
                        {
                            "run_number": 1,
                            "config_overrides": {"train.lr": 3.0e-4},
                            "decision": "retry",
                        }
                    ],
                    "accepted_run": None,
                    "checkpoint_path": None,
                }
            )
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.24)],
            [True],
        )
        make_runner(pipeline_module, state_dir).run()
        state = load_json(state_dir / "c1.json")
        assert len(state["runs"]) == 2
        assert state["runs"][0]["run_number"] == 1
        assert state["runs"][1]["run_number"] == 2

    def test_dry_run_prints_config_and_does_not_create_state(
        self, pipeline_module, monkeypatch, capsys, state_dir,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            ["c_pipeline.py", "--milestone", "c1", "--dry-run", "--state-dir", str(state_dir)],
        )
        pipeline_module.main()
        assert "bayes_ffn" in capsys.readouterr().out
        assert not state_dir.exists()

    def test_compare_writes_text_and_json_reports(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        state_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            sys,
            "argv",
            ["c_pipeline.py", "--compare", "--state-dir", str(state_dir)],
        )
        pipeline_module.main()
        assert (state_dir / "comparison.txt").exists()
        assert (state_dir / "comparison.json").exists()

    def test_compare_output_includes_cross_scale_table_content(
        self, pipeline_module, monkeypatch, capsys, state_dir,
    ):
        state_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            sys,
            "argv",
            ["c_pipeline.py", "--compare", "--state-dir", str(state_dir)],
        )
        pipeline_module.main()
        output = capsys.readouterr().out
        assert "Cross-Scale Comparison" in output
        assert "4L MI ratio" in output
        assert "16L MI ratio" in output

    def test_state_writes_are_atomic_with_temp_file_and_rename(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        rename_calls = []
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [True])

        def fake_replace(src, dst):
            rename_calls.append((Path(src).suffix, Path(dst).name))

        monkeypatch.setattr(pipeline_module.os, "replace", fake_replace, raising=False)
        make_runner(pipeline_module, state_dir).run()
        assert rename_calls
        assert rename_calls[0][0] == ".tmp"


class TestPolicies:
    def test_provider_failure_uses_lr_halving_fallback(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        seen_cfgs = []
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(best_val_loss=101.0), result_dict()],
            [False, True],
            seen_cfgs,
        )
        make_runner(
            pipeline_module,
            state_dir,
            provider=FakeProvider(error=RuntimeError("provider unavailable")),
        ).run()
        assert seen_cfgs[1]["train"]["lr"] == pytest.approx(seen_cfgs[0]["train"]["lr"] / 2)

    def test_provider_failure_uses_init_rho_fallback_for_frozen_posteriors(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        seen_cfgs = []
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(sigma_std=0.004), result_dict()],
            [False, True],
            seen_cfgs,
        )
        make_runner(
            pipeline_module,
            state_dir,
            provider=FakeProvider(error=RuntimeError("provider unavailable")),
        ).run()
        assert seen_cfgs[1]["model"]["bayes_ffn"]["init_rho"] == pytest.approx(
            seen_cfgs[0]["model"]["bayes_ffn"]["init_rho"] + 1.0
        )

    def test_unknown_knob_is_skipped_not_raised(
        self, pipeline_module, monkeypatch, state_dir, capsys,
    ):
        """Unknown knobs from agent are warned and skipped, not raised."""
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.04), result_dict()],
            [False, True],
        )
        provider = FakeProvider(
            responses=[{
                "diagnosis": "bad", "reasoning": "bad",
                "adjustment": {"train.not_a_real_knob": 1},
            }]
        )
        # Should NOT raise — unknown knob is silently dropped
        make_runner(pipeline_module, state_dir, provider=provider).run()
        captured = capsys.readouterr()
        assert "unknown knob" in captured.out.lower()

    def test_out_of_range_knob_is_clamped(
        self, pipeline_module, monkeypatch, state_dir, capsys,
    ):
        """Out-of-range values from agent are clamped, not raised."""
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.04), result_dict()],
            [False, True],
        )
        provider = FakeProvider(
            responses=[{
                "diagnosis": "bad", "reasoning": "bad",
                "adjustment": {"train.kl_weight": 10.0},
            }]
        )
        # Should NOT raise — value is clamped to range
        make_runner(pipeline_module, state_dir, provider=provider).run()
        captured = capsys.readouterr()
        assert "clamping" in captured.out.lower()

    def test_budget_exceeded_fails_before_next_run(self, pipeline_module, monkeypatch, state_dir):
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [True])
        assert make_runner(pipeline_module, state_dir, budget_hours=0.5).run() != 0

    def test_no_agent_mode_skips_provider_invocation(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        provider = FakeProvider(
            responses=[
                {
                    "diagnosis": "posteriors_frozen",
                    "reasoning": "increase init_rho",
                    "adjustment": {"model.bayes_ffn.init_rho": -1.0},
                }
            ]
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.05), result_dict(mi_ratio_mean=1.24)],
            [False, True],
        )
        runner = pipeline_module.PipelineRunner(
            repo_root=Path(__file__).resolve().parents[1],
            milestone="c1",
            provider=provider,
            state_dir=state_dir,
            budget_hours=12.0,
            use_mlflow=False,
            no_agent=True,
        )
        runner.run()
        assert provider.calls == []

    def test_oom_halves_batch_and_doubles_accumulation(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        seen_cfgs = []
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [True], seen_cfgs)
        calls = {"value": 0}

        def fake_train(*args, **kwargs):
            calls["value"] += 1
            if calls["value"] == 1:
                raise RuntimeError("CUDA out of memory")
            return object(), {
                "best_val_loss": 4.2,
                "best_val_step": 1000,
                "train_time_sec": 3600.0,
                "tokens_per_sec": 1234.0,
            }

        monkeypatch.setattr(pipeline_module, "train", fake_train, raising=False)
        make_runner(pipeline_module, state_dir).run()
        assert seen_cfgs[1]["train"]["batch_size"] == seen_cfgs[0]["train"]["batch_size"] // 2
        assert seen_cfgs[1]["train"]["gradient_accumulation_steps"] == (
            seen_cfgs[0]["train"]["gradient_accumulation_steps"] * 2
        )

    def test_third_oom_retry_aborts(self, pipeline_module, monkeypatch, state_dir):
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [False])
        monkeypatch.setattr(
            pipeline_module,
            "train",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
            raising=False,
        )
        assert make_runner(pipeline_module, state_dir).run() != 0

    @pytest.mark.parametrize("best_val_loss", [float("nan"), 200.0])
    def test_divergence_halves_lr(self, pipeline_module, monkeypatch, state_dir, best_val_loss):
        seen_cfgs = []
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(best_val_loss=best_val_loss), result_dict()],
            [False, True],
            seen_cfgs,
        )
        make_runner(pipeline_module, state_dir).run()
        assert seen_cfgs[1]["train"]["lr"] == pytest.approx(seen_cfgs[0]["train"]["lr"] / 2)

    def test_c0_skips_mi_evaluation(self, pipeline_module, monkeypatch, state_dir):
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [True])
        monkeypatch.setattr(
            pipeline_module,
            "eval_mi_suite",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MI should not run")),
            raising=False,
        )
        make_runner(pipeline_module, state_dir, milestone="c0").run()

    def test_c2_requires_c0_checkpoint(self, pipeline_module, state_dir):
        with pytest.raises(FileNotFoundError, match="c0|checkpoint"):
            make_runner(pipeline_module, state_dir, milestone="c2").run()

    def test_c2_early_abort_is_triggered_for_flat_curvature_and_no_signal(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "c0.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "runs": [{"run_number": 1}],
                    "accepted_run": 1,
                    "checkpoint_path": "data/checkpoints/c0/run1/ckpt_best.pt",
                }
            )
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.01, curvature_mean=1.0e-5)],
            [False],
        )
        make_runner(pipeline_module, state_dir, milestone="c2").run()
        state = load_json(state_dir / "c2.json")
        assert state["status"] == "completed"
        assert len(state["runs"]) == 1

    def test_c4_laplace_early_abort_uses_same_negative_gate(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "c3.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "runs": [{"run_number": 1}],
                    "accepted_run": 1,
                    "checkpoint_path": "data/checkpoints/c3/base/ckpt_best.pt",
                }
            )
        )
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.01, curvature_mean=1.0e-5)],
            [False],
        )
        make_runner(pipeline_module, state_dir, milestone="c4_lap").run()
        state = load_json(state_dir / "c4_lap.json")
        assert state["status"] == "completed"
        assert len(state["runs"]) == 1

    def test_c3_phase1_runs_once_even_if_phase2_retries(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        phase1_calls = []
        install_runner_stubs(
            pipeline_module,
            monkeypatch,
            [result_dict(mi_ratio_mean=1.03), result_dict(mi_ratio_mean=1.08)],
            [False, True],
        )
        monkeypatch.setattr(
            pipeline_module,
            "run_c3_phase1",
            lambda *args, **kwargs: phase1_calls.append("phase1")
            or "data/checkpoints/c3/base/ckpt_best.pt",
            raising=False,
        )
        provider = FakeProvider(
            responses=[
                {
                    "diagnosis": "low_mi",
                    "reasoning": "raise rank",
                    "adjustment": {"lora.rank": 32},
                }
            ]
        )
        make_runner(pipeline_module, state_dir, milestone="c3", provider=provider).run()
        assert phase1_calls == ["phase1"]

    def test_c4_reuses_c3_base_checkpoint(self, pipeline_module, monkeypatch, state_dir):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "c3.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "runs": [{"run_number": 1}],
                    "accepted_run": 1,
                    "checkpoint_path": "data/checkpoints/c3/base/ckpt_best.pt",
                }
            )
        )
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [True])
        make_runner(pipeline_module, state_dir, milestone="c4_tfb").run()
        assert load_json(state_dir / "c4_tfb.json")["checkpoint_path"].endswith("ckpt_best.pt")

    def test_pipeline_result_uses_expected_multi_ood_domain_keys(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        install_runner_stubs(pipeline_module, monkeypatch, [result_dict()], [True])
        make_runner(pipeline_module, state_dir).run()
        result = load_json(state_dir / "c1.json")["runs"][0]["result"]
        assert set(result["test_ood_ppl"]) == {"arxiv", "freelaw", "pubmed_abstracts"}
        assert set(result["mi_ratio"]) == {"arxiv", "freelaw", "pubmed_abstracts"}
