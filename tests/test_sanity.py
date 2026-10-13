"""S9 trust checks: scripts/sanity.py (spec agents/specs/2026-10/i2-trust-checks.md, T1-T9).

Each check's right answer comes from outside the code under test: math, saved results,
sklearn or a brute-force pair loop. These tests run on the CPU and read no file under data/.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sanity.py"


@pytest.fixture(scope="module")
def sanity():
    spec = importlib.util.spec_from_file_location("sanity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sanity"] = module
    spec.loader.exec_module(module)
    return module


# T1: GPU and opt-in. Without --with-gpu it must SKIP and say why.
def test_t1_skips_with_reason_without_gpu_flag(sanity):
    r = sanity.check_t1(with_gpu=False)
    assert r.status == "SKIP"
    assert "--with-gpu" in r.reason


def test_t1_skip_is_printed_and_exit_is_zero(sanity, capsys):
    assert sanity.main(["--only", "T1"]) == 0
    out = capsys.readouterr().out
    assert "T1" in out and "SKIP" in out and "--with-gpu" in out


# T2: N identical weight samples give MI = 0 and G = 0.
def test_t2_identical_samples_give_zero_mi_and_g(sanity, monkeypatch):
    monkeypatch.delenv(sanity.PLANT_ENV, raising=False)
    r = sanity.check_t2()
    assert r.status == "PASS", r.reason


# T3: g by hand, log 0.5 - (log 0.9 + log 0.1) / 2 = 0.511 nats.
def test_t3_gap_by_hand(sanity):
    expected = math.log(0.5) - (math.log(0.9) + math.log(0.1)) / 2
    assert abs(expected - 0.5108) < 1e-4
    r = sanity.check_t3()
    assert r.status == "PASS", r.reason


# T4: B times the v2 TFB noise has covariance sigma^2 U U^T; v1_legacy must fail it.
def test_t4_v2_covariance_matches_and_v1_legacy_does_not(sanity):
    assert sanity.tfb_cov_error("v2") < 0.05
    assert sanity.tfb_cov_error("v1_legacy") > 0.5
    r = sanity.check_t4()
    assert r.status == "PASS", r.reason


# T5: diagonal Laplace precision N T^2 F_hat equals the exact Hessian diagonal of the summed NLL.
def test_t5_laplace_precision_matches_exact_hessian(sanity):
    ratios = sanity.laplace_scale_ratios()
    assert len(ratios) >= 2
    for name, ratio in ratios.items():
        assert abs(ratio - 1.0) < 0.05, (name, ratio)
    r = sanity.check_t5()
    assert r.status == "PASS", r.reason


def test_t5_catches_a_scale_off_by_t(sanity):
    # A precision built with T = 1 instead of T = block_size is off by T^2.
    ratios = sanity.laplace_scale_ratios(tokens_per_seq=1)
    assert all(ratio < 0.1 for ratio in ratios.values()), ratios


# T6: our weighted AUROC equals a brute-force pair loop (and sklearn) on data with ties.
def test_t6_brute_force_auroc_by_hand(sanity):
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert sanity.brute_force_auroc(scores, labels, np.ones(4)) == pytest.approx(0.75)
    # One tie between an ID and an OOD row counts one half.
    assert sanity.brute_force_auroc(np.array([0.5, 0.5]), np.array([0, 1]),
                                    np.ones(2)) == pytest.approx(0.5)
    # Weight 2 on the misranked ID row: right pair weight 1, wrong pair weight 2, so 1 / 3.
    w = np.array([1.0, 2.0, 1.0])
    assert sanity.brute_force_auroc(np.array([0.2, 0.9, 0.5]), np.array([0, 0, 1]),
                                    w) == pytest.approx(1.0 / 3.0)


def test_t6_auroc_matches_outside_code(sanity):
    r = sanity.check_t6()
    assert r.status == "PASS", r.reason


# T7: health gate on stored posteriors.
def test_t7_gate_rule(sanity):
    budget = 0.003
    # The old C2 state: NLL 9.10 against 2.79 for the plain model.
    assert not sanity.within_budget(delta_nll=9.10 - 2.79, ell0=2.79, budget_rel=budget)
    assert sanity.within_budget(delta_nll=0.0, ell0=2.79, budget_rel=budget)
    assert sanity.within_budget(delta_nll=budget * 2.79, ell0=2.79, budget_rel=budget)


def test_t7_skips_with_reason_when_states_are_absent(sanity, tmp_path):
    missing = tmp_path / "no_such_state.pt"
    cases = [sanity.GateCase("c2 (pre-fix)", REPO_ROOT / "configs" / "i2_posthoc_c2.yaml",
                             missing, False)]
    r = sanity.check_t7("T7a", cases)
    assert r.status == "SKIP"
    assert "no_such_state.pt" in r.reason


# T8: the real builder feeds the real refit end to end.
def test_t8_builder_to_refit_end_to_end(sanity):
    r = sanity.check_t8()
    assert r.status == "PASS", r.reason


# T9: a planted error makes the table show FAIL and the exit code 1.
def test_t9_planted_error_fails_the_run(sanity, monkeypatch, capsys):
    monkeypatch.setenv(sanity.PLANT_ENV, "1")
    assert sanity.check_t2().status == "FAIL"
    assert sanity.main(["--only", "T2,T3"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_clean_run_exits_zero_and_prints_one_table(sanity, monkeypatch, capsys):
    monkeypatch.delenv(sanity.PLANT_ENV, raising=False)
    assert sanity.main(["--only", "T2,T3,T6"]) == 0
    out = capsys.readouterr().out
    for word in ("check", "what it proves", "status", "T2", "T3", "T6", "PASS"):
        assert word in out
    assert "FAIL" not in out
