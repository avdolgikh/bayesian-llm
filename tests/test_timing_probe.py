"""S0 timing probe (specs/i2-timing-probe.md): CPU-only tests.

The probe lives in scripts/, which is not a package, so it is imported through
``syspath_prepend`` like tests/test_c_pipeline.py imports the pipeline. No test reads
data/: every fixture is built here or under ``tmp_path``.
"""

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from minigpt.model import GPTConfig, MiniGPT

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_YAML = REPO_ROOT / "configs" / "i2_timing_probe.yaml"
DET_LORA_YAML = REPO_ROOT / "configs" / "i2_timing_probe_det_lora.yaml"
C3_YAML = REPO_ROOT / "configs" / "c3_phase2.yaml"

# CA10 batch-1 rates in seconds per block (spec Section 5 fixture).
CA10 = {
    "c0": 0.0081,
    "c1": 0.2869,
    "c2": 0.2869,
    "mc_dropout": 0.2869,
    "c3": 0.3156,
    "c4_tfb": 0.4686,
    "c4_lap": 0.4686,
}


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    sys.modules.pop("timing_probe", None)
    return importlib.import_module("timing_probe")


def _cell(method, batch_size, s_per_block, mib, over_budget=False):
    measured = not over_budget
    return {
        "method": method,
        "batch_size": batch_size,
        "n_samples": 20,
        "ms_per_block": s_per_block * 1000.0 if measured else None,
        "n_blocks_timed": 64 if measured else None,
        "peak_reserved_mib": mib if measured else None,
        "peak_allocated_mib": mib if measured else None,
        "over_budget": over_budget,
        "oom": False,
        "sampler_ms_per_sample": None,
    }


def _rate_cells(rates, scale=1.0, batch_size=1):
    return [_cell(m, batch_size, r * scale, 1000.0) for m, r in rates.items()]


def _header(probe, created_utc):
    header = {field: None for field in probe.HEADER_FIELDS}
    header["schema"] = probe.SCHEMA
    header["created_utc"] = created_utc
    return header


def _rescore_section():
    over = _cell("c2", 32, 0.0, 0.0, over_budget=True)
    over["sampler_ms_per_sample"] = 102.0
    return {
        "anchor": {"attempts_ms": [8.3], "anchor_pass": True, "other_compute_procs": [[]]},
        "cells": [_cell("c0", 1, 0.0081, 900.0), over],
        "nll_check": {"max_abs_diff_nats": {"8": 0.001, "32": 0.002}, "pass": True},
    }


def _write_probe_yaml(tmp_path, mutate):
    cfg = yaml.safe_load(PROBE_YAML.read_text(encoding="utf-8"))
    mutate(cfg)
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# S0-T1(d): batched scorer consistency on a 2-layer CPU fixture (fp32)
# ---------------------------------------------------------------------------

def test_batched_scorer_consistency(probe):
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True,
    )
    model = MiniGPT(cfg).eval()
    model.requires_grad_(False)
    tokens = torch.randint(0, cfg.vocab_size, (16, cfg.block_size + 1))
    xs, ys = tokens[:, :-1], tokens[:, 1:]
    cpu = torch.device("cpu")

    nll_b1 = probe.per_block_nll(model, xs, ys, batch_size=1, device=cpu, use_amp=False)
    nll_b8 = probe.per_block_nll(model, xs, ys, batch_size=8, device=cpu, use_amp=False)
    assert nll_b1.shape == (16,)
    assert (nll_b8 - nll_b1).abs().max().item() <= 1e-5

    # The batched path is not only self-consistent: it matches a plain per-block CE.
    with torch.no_grad():
        ref = torch.stack([
            F.cross_entropy(model(xs[i : i + 1])[0][0], ys[i]) for i in range(xs.size(0))
        ])
    assert (nll_b1 - ref.double()).abs().max().item() <= 1e-5

    # Three identical draws: no disagreement, so MI and g_t vanish.
    stats = probe.score_blocks(
        model, xs, ys, batch_size=8, n_samples=3, mode="deterministic", draw_fn=None,
        device=cpu, use_amp=False,
    )
    assert stats["mi"].shape == (16, 16)
    assert stats["ll"].shape == (3, 16, 16)
    assert stats["mi"].abs().max().item() <= 1e-6
    assert stats["g"].abs().max().item() <= 1e-6
    assert torch.all(stats["max_prob"] > 0) and torch.all(stats["max_prob"] <= 1)
    # pred_entropy is H[p-bar] and must match the N=1 entropy of the same model.
    one = probe.score_blocks(
        model, xs, ys, batch_size=16, n_samples=1, mode="deterministic", draw_fn=None,
        device=cpu, use_amp=False,
    )
    assert (stats["pred_entropy"] - one["pred_entropy"]).abs().max().item() <= 1e-5


# ---------------------------------------------------------------------------
# S0-T1(d): JSON merge and schema validation
# ---------------------------------------------------------------------------

def test_json_merge_schema(probe, tmp_path):
    out = tmp_path / "i2" / "timing_probe.json"
    probe.merge_section(out, "shape355m", {"skipped": "Q1=a"}, _header(probe, "first"))
    probe.merge_section(out, "rescore", _rescore_section(), _header(probe, "second"))

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["shape355m"] == {"skipped": "Q1=a"}
    assert doc["rescore"]["cells"][0]["method"] == "c0"
    for field in probe.HEADER_FIELDS:
        assert field in doc["header"], field
    assert doc["header"]["schema"] == "i2-timing-probe/1"
    assert doc["header"]["created_utc"] == "first"
    assert not list(tmp_path.rglob("*.tmp"))
    probe.validate_schema(doc, require=("shape355m", "rescore"))

    no_cells = copy.deepcopy(doc)
    del no_cells["rescore"]["cells"]
    with pytest.raises(probe.SchemaError, match="cells"):
        probe.validate_schema(no_cells)

    no_rescore = copy.deepcopy(doc)
    del no_rescore["rescore"]
    with pytest.raises(probe.SchemaError, match="rescore"):
        probe.validate_schema(no_rescore, require=("rescore",))

    bad_header = copy.deepcopy(doc)
    del bad_header["header"]["git_head"]
    with pytest.raises(probe.SchemaError, match="git_head"):
        probe.validate_schema(bad_header)

    # An invalid section is refused before anything is written.
    before = out.read_text(encoding="utf-8")
    broken = _rescore_section()
    del broken["cells"]
    with pytest.raises(probe.SchemaError):
        probe.merge_section(out, "rescore", broken, _header(probe, "third"))
    assert out.read_text(encoding="utf-8") == before
    assert not list(tmp_path.rglob("*.tmp"))


# ---------------------------------------------------------------------------
# S0-T4: estimate arithmetic and batch choice
# ---------------------------------------------------------------------------

def test_estimate_arithmetic(probe):
    cfg = probe.load_config(PROBE_YAML)

    est = probe.estimate(cfg, {"cells": _rate_cells(CA10)}, {"wall_min": 20.0})
    m4 = est["m4"]
    assert m4["sum_rate_s"] == pytest.approx(2.1216, abs=1e-9)
    assert m4["sum_rate_lora_s"] == pytest.approx(1.2528, abs=1e-9)
    assert m4["gpu_h"]["1"] == pytest.approx(3.29, abs=0.01)
    assert m4["gpu_h"]["2"] == pytest.approx(2 * 3.2947, abs=0.01)
    assert m4["gpu_h"]["3"] == pytest.approx(9.88, abs=0.01)
    assert m4["cap_applied"] is False
    assert m4["blocks_per_doc"] == 3
    assert m4["m4_nights_needed"] == 2
    assert m4["low_end_holds"] is False
    assert m4["escalate_g0"] is False
    assert m4["batch_size"] == {m: 1 for m in CA10}
    assert m4["speedup_vs_ca10"] == pytest.approx(1.0)
    assert m4["speedup_measured"] == pytest.approx(1.0)
    assert est["repro"]["gpu_h_b1"] == pytest.approx(0.75, abs=0.01)
    assert est["repro"]["gpu_h_bstar"] == pytest.approx(0.75, abs=0.01)
    assert est["s5_finetune_gpu_h"]["n_blob_2"] == pytest.approx(1.897, abs=0.005)
    assert est["s5_finetune_gpu_h"]["n_blob_3"] == pytest.approx(2.345, abs=0.005)

    up = probe.estimate(cfg, {"cells": _rate_cells(CA10, 1.02)}, None)["m4"]
    assert up["gpu_h"]["3"] == pytest.approx(10.08, abs=0.01)
    assert up["cap_applied"] is True
    assert up["blocks_per_doc"] == 1
    assert up["m4_nights_needed"] == 1
    assert up["escalate_g0"] is False

    down = probe.estimate(cfg, {"cells": _rate_cells(CA10, 1 / 3.5)}, None)
    assert down["m4"]["gpu_h"]["1"] == pytest.approx(0.94, abs=0.01)
    assert down["m4"]["low_end_holds"] is True
    assert down["s5_finetune_gpu_h"] is None

    # Batching: the fastest fitting batch wins and both speed-ups follow.
    batched = _rate_cells(CA10) + _rate_cells(CA10, 0.25, batch_size=8)
    fast = probe.estimate(cfg, {"cells": batched}, None)
    assert fast["m4"]["batch_size"] == {m: 8 for m in CA10}
    assert fast["m4"]["speedup_measured"] == pytest.approx(4.0)
    assert fast["m4"]["speedup_vs_ca10"] == pytest.approx(4.0)
    assert fast["m4"]["gpu_h"]["1"] == pytest.approx(3.2947 / 4, abs=0.01)
    assert fast["repro"]["gpu_h_b1"] == pytest.approx(0.754, abs=0.01)
    assert fast["repro"]["gpu_h_bstar"] == pytest.approx(0.754 / 4, abs=0.01)

    # t_det(n, i) rescales with the evals inside the timed region (spec Section 5):
    # C3's clock (1614.1 s, 21 evals of 10 s) round-trips to 26.9 min at 10,000 steps.
    t_step = (1614.1 - 21 * 10.0) / 10_000
    assert probe.t_det_min(10_000, 500, t_step, 10.0) == pytest.approx(1614.1 / 60)
    assert probe.t_det_min(5_000, 500, t_step, 10.0) == pytest.approx(
        (5_000 * t_step + 11 * 10.0) / 60
    )


def test_batch_choice(probe):
    budget = 9728
    cells = [
        _cell("a", 1, 0.30, 500.0),
        _cell("a", 8, 0.05, 2000.0),
        _cell("a", 32, 0.03, 10000.0),  # faster, but above the VRAM budget
        _cell("b", 1, 0.30, 500.0),
        _cell("b", 8, 0.0, 0.0, over_budget=True),
        _cell("b", 32, 0.04, 9000.0),
        _cell("c", 1, 0.0, 0.0, over_budget=True),
        _cell("c", 8, 0.0, 0.0, over_budget=True),
        _cell("c", 32, 0.0, 0.0, over_budget=True),
    ]
    assert probe.choose_batch(cells, "a", budget) == 8
    assert probe.choose_batch(cells, "b", budget) == 32
    with pytest.raises(probe.NoFeasibleBatchError, match="'c'"):
        probe.choose_batch(cells, "c", budget)
    with pytest.raises(probe.NoFeasibleBatchError, match="'d'"):
        probe.choose_batch(cells, "d", budget)

    # --part estimate stops on a set with no feasible cell and names it.
    cfg = probe.load_config(PROBE_YAML)
    infeasible = [
        _cell(m, 1, r, 1000.0, over_budget=(m == "c2")) for m, r in CA10.items()
    ]
    with pytest.raises(probe.NoFeasibleBatchError, match="'c2'"):
        probe.estimate(cfg, {"cells": infeasible}, None)


# ---------------------------------------------------------------------------
# S0-T5: hand-off check before m4
# ---------------------------------------------------------------------------

def _plan_files(probe, tmp_path, *, json_blocks, yaml_blocks, escalate=False, write_json=True,
                nested=False):
    out = tmp_path / "timing_probe.json"
    if out.exists():
        out.unlink()
    config = _write_probe_yaml(
        tmp_path, lambda cfg: cfg["probe"].__setitem__("output", str(out)),
    )
    if write_json:
        est = probe.estimate(probe.load_config(config), {"cells": _rate_cells(CA10)}, None)
        est["m4"]["blocks_per_doc"] = json_blocks
        est["m4"]["escalate_g0"] = escalate
        probe.merge_section(out, "estimates", est, _header(probe, "t"))
    m4_yaml = tmp_path / "m4.yaml"
    body = ({"eval_set": {"max_blocks_per_doc": yaml_blocks}} if nested
            else {"blocks_per_doc": yaml_blocks})
    m4_yaml.write_text(yaml.safe_dump(body), encoding="utf-8")
    return ["--config", str(config), "--part", "check", "--m4-config", str(m4_yaml)]


def test_check_plan(probe, tmp_path, capsys):
    args = _plan_files(probe, tmp_path, json_blocks=1, yaml_blocks=3)
    assert probe.main(args) == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "blocks_per_doc" in text and "3" in text and "1" in text

    assert probe.main(_plan_files(probe, tmp_path, json_blocks=1, yaml_blocks=1)) == 0
    assert probe.main(_plan_files(probe, tmp_path, json_blocks=3, yaml_blocks=1)) == 0
    # S1's YAML nests the value as eval_set.max_blocks_per_doc; the check finds it.
    nested = _plan_files(probe, tmp_path, json_blocks=1, yaml_blocks=3, nested=True)
    assert probe.main(nested) == 2

    escalated = _plan_files(probe, tmp_path, json_blocks=1, yaml_blocks=1, escalate=True)
    assert probe.main(escalated) == 3
    assert probe.main(escalated + ["--g0-ack"]) == 0

    missing = _plan_files(probe, tmp_path, json_blocks=1, yaml_blocks=1, write_json=False)
    assert probe.main(missing) == 4

    # The pure function gives the same codes.
    code, _ = probe.check_plan(tmp_path / "absent.json", tmp_path / "m4.yaml", g0_ack=False)
    assert code == 4


# ---------------------------------------------------------------------------
# Configs and the CPU dry run of every GPU part
# ---------------------------------------------------------------------------

def test_det_lora_config_matches_c3(probe):
    det = yaml.safe_load(DET_LORA_YAML.read_text(encoding="utf-8"))
    c3 = yaml.safe_load(C3_YAML.read_text(encoding="utf-8"))
    changed = {
        ("experiment", "name"): "i2_probe_det_lora",
        ("experiment", "run_name"): "i2_probe_det_lora",
        ("train", "checkpoint_dir"): "data/checkpoints/i2_probe/det_lora",
        ("train", "checkpoint_interval"): 0,
        ("train", "kl_weight"): 0.0,
        ("train", "kl_annealing_steps"): 0,
        ("train", "patience_evals"): 0,
        ("train", "patience_min_delta"): 0.001,
    }
    for (section, key), value in changed.items():
        assert det[section][key] == value, (section, key)
    assert set(det) == set(c3)
    for section, body in c3.items():
        for key, value in body.items():
            if (section, key) not in changed:
                assert det[section][key] == value, (section, key)
        extra = set(det[section]) - set(body)
        assert extra <= {"patience_evals", "patience_min_delta"}, (section, extra)
    digest = probe.config_sha256(DET_LORA_YAML)
    assert len(digest) == 64 and digest == probe.config_sha256(DET_LORA_YAML)


def test_dry_run_parts(probe, tmp_path):
    out = tmp_path / "dry" / "timing_probe.json"
    ckpt_dir = tmp_path / "dry" / "det_lora"

    def mutate(cfg):
        cfg["dry_run"]["overrides"]["probe.output"] = str(out)
        cfg["dry_run"]["checkpoint_dir"] = str(ckpt_dir)

    config = _write_probe_yaml(tmp_path, mutate)
    for part in ("shape355m", "rescore", "finetune", "estimate"):
        assert probe.main(["--config", str(config), "--part", part, "--dry-run"]) == 0, part

    doc = json.loads(out.read_text(encoding="utf-8"))
    probe.validate_schema(doc, require=("shape355m", "rescore", "finetune", "estimates"))
    assert doc["header"]["dry_run"] is True
    assert not list(tmp_path.rglob("*.tmp"))

    rescore = doc["rescore"]
    methods = ["c0", "c1", "c2", "c3", "c4_tfb", "c4_lap", "mc_dropout"]
    assert [c["method"] for c in rescore["cells"]] == [m for m in methods for _ in (1, 4)]
    assert rescore["checks"]["t1a_pass"] is True
    assert rescore["nll_check"]["pass"] is True
    assert len(rescore["anchor"]["attempts_ms"]) in (1, 2)
    for cell in rescore["cells"]:
        assert cell["n_blocks_timed"] == {1: 4, 4: 8}[cell["batch_size"]]
        if cell["method"] in ("c2", "c4_tfb", "c4_lap"):
            assert cell["sampler_ms_per_sample"] > 0

    shapes = doc["shape355m"]["shapes"]
    assert set(shapes) == {"355m", "76m"}
    assert shapes["355m"]["trainable_params"] > 0
    assert shapes["355m"]["median_step_ms"] > 0

    finetune = doc["finetune"]
    assert finetune["steps_completed"] == 20
    assert finetune["n_evals"] == 3
    assert finetune["trainable_params"] == 2560
    assert finetune["seed_applied"] == 1337
    for name in ("steps_completed", "n_evals", "trainable_params", "seed_applied"):
        assert finetune["checks"][name] is True, name
    assert (ckpt_dir / "ckpt_best.pt").exists()

    m4 = doc["estimates"]["m4"]
    assert set(m4["gpu_h"]) == {"1", "2", "3"}
    assert set(m4["batch_size"]) == set(methods)
    assert set(doc["estimates"]["s5_finetune_gpu_h"]) == {"n_blob_2", "n_blob_3"}
