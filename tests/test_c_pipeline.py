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
        assert cfg["train"]["kl_weight"] == pytest.approx(0.1)

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


# ============================================================
# Post-hoc pipeline tests (Laplace / TFB integration)
# ============================================================


class TestPosthocTemplates:
    """Config templates for post-hoc milestones (C2, C4_TFB, C4_LAP)."""

    @pytest.mark.parametrize("milestone", ["c2", "c4_tfb", "c4_lap"])
    def test_posthoc_templates_have_steps_zero(self, pipeline_module, milestone):
        cfg = pipeline_module.build_milestone_config(milestone)
        assert cfg["train"]["steps"] == 0

    @pytest.mark.parametrize(
        "milestone,expected_method",
        [("c2", "laplace"), ("c4_tfb", "tfb"), ("c4_lap", "laplace")],
    )
    def test_posthoc_templates_have_correct_method(
        self, pipeline_module, milestone, expected_method,
    ):
        cfg = pipeline_module.build_milestone_config(milestone)
        assert cfg["posthoc_method"] == expected_method

    @pytest.mark.parametrize("milestone", ["c0", "c1", "c3_phase2"])
    def test_training_templates_have_no_posthoc_method(self, pipeline_module, milestone):
        cfg = pipeline_module.build_milestone_config(milestone)
        assert cfg.get("posthoc_method") is None

    def test_validation_accepts_steps_zero(self):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["train"]["steps"] = 0
        validate_config(cfg)  # should not raise

    @pytest.mark.parametrize("milestone", ["c2", "c4_tfb", "c4_lap"])
    def test_posthoc_templates_pass_validation(self, pipeline_module, milestone):
        validate_config(pipeline_module.build_milestone_config(milestone))


class TestPosthocFit:
    """Unit tests for _posthoc_fit dispatch function."""

    def test_returns_none_for_mock_model(self, pipeline_module):
        cfg = {"posthoc_method": "laplace", "train": {"steps": 0}}
        result = pipeline_module._posthoc_fit(object(), cfg, {}, "cpu")
        assert result is None

    def test_returns_none_without_posthoc_method(self, pipeline_module):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        result = pipeline_module._posthoc_fit(object(), cfg, {}, "cpu")
        assert result is None

    def test_laplace_fit_returns_callable_and_curvature(self, pipeline_module, tmp_path):
        import torch

        from minigpt.layers import BayesConfig
        from minigpt.model import GPTConfig, MiniGPT

        torch.manual_seed(42)
        config = GPTConfig(
            vocab_size=256, block_size=32, n_layer=2, n_head=2,
            n_embd=64, dropout=0.0, bias=True,
            bayes_head=BayesConfig(enabled=False),
        )
        model = MiniGPT(config)
        data = {"train": torch.randint(0, 256, (512,))}
        cfg = {
            "posthoc_method": "laplace",
            "train": {
                "block_size": 32, "batch_size": 4,
                "checkpoint_dir": str(tmp_path / "ckpt"), "steps": 0,
            },
            "laplace": {
                "selection_mode": "ffn", "damping": 1.0,
                "sample_scale": 1.0, "n_curvature_batches": 2,
            },
        }
        result = pipeline_module._posthoc_fit(
            model, cfg, data, torch.device("cpu"),
        )
        assert result is not None
        mi_fn, extras = result
        assert callable(mi_fn)
        assert "curvature_mean" in extras
        assert isinstance(extras["curvature_mean"], float)
        # Laplace state should be saved
        assert (tmp_path / "ckpt" / "laplace_state.pt").exists()

    def test_laplace_mi_fn_produces_valid_metrics(self, pipeline_module, tmp_path):
        import torch

        from minigpt.layers import BayesConfig
        from minigpt.model import GPTConfig, MiniGPT

        torch.manual_seed(42)
        config = GPTConfig(
            vocab_size=256, block_size=32, n_layer=2, n_head=2,
            n_embd=64, dropout=0.0, bias=True,
            bayes_head=BayesConfig(enabled=False),
        )
        model = MiniGPT(config)
        data_tensor = torch.randint(0, 256, (512,))
        cfg = {
            "posthoc_method": "laplace",
            "train": {
                "block_size": 32, "batch_size": 4,
                "checkpoint_dir": str(tmp_path / "ckpt"), "steps": 0,
            },
            "laplace": {
                "selection_mode": "ffn", "damping": 1.0,
                "sample_scale": 1.0, "n_curvature_batches": 2,
            },
        }
        mi_fn, _ = pipeline_module._posthoc_fit(
            model, cfg, {"train": data_tensor}, torch.device("cpu"),
        )
        # Call the returned mi_fn — same signature as compute_uncertainty_metrics
        result = mi_fn(
            model, data_tensor, 32, 4, torch.device("cpu"),
            n_samples=3, n_batches=1,
        )
        assert "mi_mean" in result
        assert "flip_rate" in result
        assert result["mi_mean"] >= 0.0


class TestPosthocPipelineIntegration:
    """Integration: posthoc_fit results flow through the pipeline."""

    def _setup_c0_state(self, state_dir):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "c0.json").write_text(json.dumps({
            "status": "completed",
            "runs": [{"run_number": 1}],
            "accepted_run": 1,
            "checkpoint_path": "data/checkpoints/c0/ckpt_best.pt",
        }))

    def test_posthoc_extras_appear_in_pipeline_result(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        self._setup_c0_state(state_dir)
        install_runner_stubs(
            pipeline_module, monkeypatch,
            [result_dict(mi_ratio_mean=1.01)], [False],
        )

        def fake_posthoc(*args, **kwargs):
            def fake_mi_fn(*a, **kw):
                return {"mi_mean": 0.001, "flip_rate": 0.01,
                        "predictive_entropy_mean": 3.0}
            return fake_mi_fn, {"curvature_mean": 1.23e-5}

        monkeypatch.setattr(pipeline_module, "_posthoc_fit", fake_posthoc)
        make_runner(pipeline_module, state_dir, milestone="c2").run()
        state = load_json(state_dir / "c2.json")
        assert state["runs"][0]["result"]["curvature_mean"] == pytest.approx(1.23e-5)

    def test_posthoc_mi_fn_is_passed_to_eval_mi_suite(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        self._setup_c0_state(state_dir)
        captured_mi_fns = []

        def capturing_eval_mi_suite(mi_fn, *args, **kwargs):
            captured_mi_fns.append(mi_fn)
            mi_id = {"mi_mean": 0.05, "flip_rate": 0.1,
                      "predictive_entropy_mean": 3.0}
            mi_ood = {"mi_mean": 0.06, "flip_rate": 0.2,
                       "predictive_entropy_mean": 4.0}
            return mi_id, mi_ood, 1.01

        sentinel_fn = lambda *a, **kw: None  # noqa: E731

        def fake_posthoc(*args, **kwargs):
            return sentinel_fn, {"curvature_mean": 1e-5}

        install_runner_stubs(
            pipeline_module, monkeypatch,
            [result_dict(mi_ratio_mean=1.01)], [False],
        )
        monkeypatch.setattr(pipeline_module, "eval_mi_suite",
                            capturing_eval_mi_suite)
        monkeypatch.setattr(pipeline_module, "_posthoc_fit", fake_posthoc)
        make_runner(pipeline_module, state_dir, milestone="c2").run()
        # The sentinel should have been passed as the mi_fn argument
        assert sentinel_fn in captured_mi_fns

    def test_without_posthoc_default_mi_fn_is_used(
        self, pipeline_module, monkeypatch, state_dir,
    ):
        """When posthoc_fit returns None, eval_mi_suite receives the default fn."""
        captured_mi_fns = []

        def capturing_eval_mi_suite(mi_fn, *args, **kwargs):
            captured_mi_fns.append(mi_fn)
            mi_id = {"mi_mean": 0.05, "flip_rate": 0.1,
                      "predictive_entropy_mean": 3.0}
            mi_ood = {"mi_mean": 0.06, "flip_rate": 0.2,
                       "predictive_entropy_mean": 4.0}
            return mi_id, mi_ood, 1.25

        install_runner_stubs(
            pipeline_module, monkeypatch,
            [result_dict()], [True],
        )
        monkeypatch.setattr(pipeline_module, "eval_mi_suite",
                            capturing_eval_mi_suite)
        make_runner(pipeline_module, state_dir, milestone="c1").run()
        # The default uncertainty_eval_fn should be used (compute_uncertainty_metrics)
        from minigpt.uncertainty import compute_uncertainty_metrics
        assert all(fn is compute_uncertainty_metrics for fn in captured_mi_fns)


class TestPrepareModelPosthocLoRA:
    """_prepare_model: BLoB checkpoint → DeterministicLoRA conversion for C4."""

    def _make_blob_checkpoint(self, tmp_path):
        """Create a tiny model, inject BLoB LoRA, save checkpoint."""
        import torch

        from minigpt.config import build_lora_config
        from minigpt.layers import BayesConfig
        from minigpt.lora import inject_lora
        from minigpt.model import GPTConfig, MiniGPT
        from minigpt.train import save_checkpoint

        torch.manual_seed(42)
        config = GPTConfig(
            vocab_size=256, block_size=32, n_layer=2, n_head=2,
            n_embd=64, dropout=0.0, bias=True,
            bayes_head=BayesConfig(enabled=False),
        )
        model = MiniGPT(config)
        lora_cfg_dict = {
            "lora": {"rank": 4, "alpha": 8.0, "target": "ffn",
                     "prior_std": 0.2, "init_g": 0.1},
        }
        lora_cfg = build_lora_config(lora_cfg_dict)
        model = inject_lora(model, lora_cfg, bayesian=True)

        # Capture BLoB MAP weights for later comparison
        blob_a_mu = {}
        blob_b = {}
        for name, module in model.named_modules():
            from minigpt.lora import BLoBLoRALinear
            if isinstance(module, BLoBLoRALinear):
                blob_a_mu[name] = module.lora_A_mu.data.clone()
                blob_b[name] = module.lora_B.data.clone()

        ckpt_dir = tmp_path / "ckpt_blob"
        ckpt_dir.mkdir()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(model, optimizer, step=100, best_val_loss=3.0,
                        config_dict={}, path=ckpt_dir / "ckpt_best.pt")
        return ckpt_dir / "ckpt_best.pt", blob_a_mu, blob_b, config

    def test_posthoc_lora_injects_deterministic_not_blob(self, pipeline_module, tmp_path):
        import torch

        from minigpt.lora import BLoBLoRALinear, DeterministicLoRALinear
        from minigpt.model import MiniGPT

        ckpt_path, _, _, gpt_config = self._make_blob_checkpoint(tmp_path)

        # Fresh vanilla model (no LoRA yet)
        torch.manual_seed(99)
        model = MiniGPT(gpt_config)

        cfg = {
            "posthoc_method": "tfb",
            "train": {"resume_checkpoint": str(ckpt_path), "checkpoint_dir": str(tmp_path)},
            "lora": {"rank": 4, "alpha": 8.0, "target": "ffn",
                     "prior_std": 0.2, "init_g": 0.1},
        }
        model = pipeline_module._prepare_model(model, cfg)

        # Should have DeterministicLoRALinear, NOT BLoBLoRALinear
        has_det = any(isinstance(m, DeterministicLoRALinear) for m in model.modules())
        has_blob = any(isinstance(m, BLoBLoRALinear) for m in model.modules())
        assert has_det, "Expected DeterministicLoRALinear modules"
        assert not has_blob, "Should NOT have BLoBLoRALinear modules"

    def test_posthoc_lora_maps_blob_a_mu_to_lora_a(self, pipeline_module, tmp_path):
        import torch

        from minigpt.lora import DeterministicLoRALinear
        from minigpt.model import MiniGPT

        ckpt_path, blob_a_mu, blob_b, gpt_config = self._make_blob_checkpoint(tmp_path)

        torch.manual_seed(99)
        model = MiniGPT(gpt_config)
        cfg = {
            "posthoc_method": "laplace",
            "train": {"resume_checkpoint": str(ckpt_path), "checkpoint_dir": str(tmp_path)},
            "lora": {"rank": 4, "alpha": 8.0, "target": "ffn",
                     "prior_std": 0.2, "init_g": 0.1},
        }
        model = pipeline_module._prepare_model(model, cfg)

        # Verify lora_A values match BLoB's lora_A_mu
        for name, module in model.named_modules():
            if isinstance(module, DeterministicLoRALinear):
                assert name in blob_a_mu, f"Unexpected LoRA module: {name}"
                torch.testing.assert_close(
                    module.lora_A.data, blob_a_mu[name],
                    msg=f"lora_A mismatch at {name}",
                )
                torch.testing.assert_close(
                    module.lora_B.data, blob_b[name],
                    msg=f"lora_B mismatch at {name}",
                )

    def test_posthoc_lora_tfb_fit_succeeds(self, pipeline_module, tmp_path):
        """End-to-end: BLoB ckpt → DeterministicLoRA → TFB fit finds sigma_q."""
        import torch

        from minigpt.model import MiniGPT

        ckpt_path, _, _, gpt_config = self._make_blob_checkpoint(tmp_path)

        torch.manual_seed(99)
        model = MiniGPT(gpt_config)
        cfg = {
            "posthoc_method": "tfb",
            "train": {"resume_checkpoint": str(ckpt_path),
                      "checkpoint_dir": str(tmp_path / "c4_tfb"),
                      "block_size": 32, "batch_size": 4, "steps": 0},
            "lora": {"rank": 4, "alpha": 8.0, "target": "ffn",
                     "prior_std": 0.2, "init_g": 0.1},
            "tfb": {"epsilon": 0.1, "n_search_samples": 2, "n_anchor_batches": 2},
        }
        model = pipeline_module._prepare_model(model, cfg)

        # TFB fit should succeed (finds DeterministicLoRALinear modules)
        data = {"train": torch.randint(0, 256, (512,))}
        result = pipeline_module._posthoc_fit(model, cfg, data, torch.device("cpu"))
        assert result is not None
        mi_fn, extras = result
        assert callable(mi_fn)
        assert "sigma_q" in extras
        assert extras["sigma_q"] > 0

    def test_posthoc_lora_laplace_fit_succeeds(self, pipeline_module, tmp_path):
        """End-to-end: BLoB ckpt → DeterministicLoRA → Laplace fit on LoRA params."""
        import torch

        from minigpt.model import MiniGPT

        ckpt_path, _, _, gpt_config = self._make_blob_checkpoint(tmp_path)

        torch.manual_seed(99)
        model = MiniGPT(gpt_config)
        cfg = {
            "posthoc_method": "laplace",
            "train": {"resume_checkpoint": str(ckpt_path),
                      "checkpoint_dir": str(tmp_path / "c4_lap"),
                      "block_size": 32, "batch_size": 4, "steps": 0},
            "lora": {"rank": 4, "alpha": 8.0, "target": "ffn",
                     "prior_std": 0.2, "init_g": 0.1},
            "laplace": {"selection_mode": "lora", "damping": 1.0,
                        "sample_scale": 1.0, "n_curvature_batches": 2},
        }
        model = pipeline_module._prepare_model(model, cfg)

        data = {"train": torch.randint(0, 256, (512,))}
        result = pipeline_module._posthoc_fit(model, cfg, data, torch.device("cpu"))
        assert result is not None
        mi_fn, extras = result
        assert callable(mi_fn)
        assert "curvature_mean" in extras

    def test_non_posthoc_lora_still_injects_blob(self, pipeline_module, tmp_path):
        """C3 (non-posthoc) should still get BLoBLoRALinear."""
        import torch

        from minigpt.layers import BayesConfig
        from minigpt.lora import BLoBLoRALinear
        from minigpt.model import GPTConfig, MiniGPT
        from minigpt.train import save_checkpoint

        torch.manual_seed(42)
        config = GPTConfig(
            vocab_size=256, block_size=32, n_layer=2, n_head=2,
            n_embd=64, dropout=0.0, bias=True,
            bayes_head=BayesConfig(enabled=False),
        )
        model = MiniGPT(config)
        # Save a base checkpoint (no LoRA)
        ckpt_dir = tmp_path / "base_ckpt"
        ckpt_dir.mkdir()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(model, optimizer, step=100, best_val_loss=3.0,
                        config_dict={}, path=ckpt_dir / "ckpt_best.pt")

        # Fresh model, no posthoc_method → should inject BLoB
        model2 = MiniGPT(config)
        cfg = {
            "train": {"resume_checkpoint": str(ckpt_dir / "ckpt_best.pt"),
                      "checkpoint_dir": str(tmp_path)},
            "lora": {"rank": 4, "alpha": 8.0, "target": "ffn",
                     "prior_std": 0.2, "init_g": 0.1},
        }
        model2 = pipeline_module._prepare_model(model2, cfg)
        has_blob = any(isinstance(m, BLoBLoRALinear) for m in model2.modules())
        assert has_blob, "Non-posthoc LoRA should use BLoBLoRALinear"


class TestTrainStepsZero:
    """train() with steps=0 must not crash and return valid metadata."""

    def test_train_steps_zero_returns_valid_metadata(self, tmp_path):
        import torch

        from minigpt.config import build_train_config
        from minigpt.layers import BayesConfig
        from minigpt.model import GPTConfig, MiniGPT
        from minigpt.train import train

        torch.manual_seed(42)
        config = GPTConfig(
            vocab_size=256, block_size=32, n_layer=2, n_head=2,
            n_embd=64, dropout=0.0, bias=True,
            bayes_head=BayesConfig(enabled=False),
        )
        model = MiniGPT(config)
        data = torch.randint(0, 256, (512,))
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["train"]["steps"] = 0
        cfg["train"]["block_size"] = 32
        cfg["train"]["batch_size"] = 4
        cfg["train"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        cfg["train"]["device"] = "cpu"
        train_cfg = build_train_config(cfg)
        model, meta = train(model, data, data, train_cfg, config_dict=cfg)
        assert meta["steps_completed"] == 0
        assert meta["train_time_sec"] >= 0
        assert meta["best_val_loss"] == float("inf")
        assert meta["eval_history"] == []
