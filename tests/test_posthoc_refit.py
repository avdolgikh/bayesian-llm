"""S2-T2 and S2-T4: the post-hoc refit module and script (specs/i2-posthoc-fixes.md).

Everything runs on the CPU on small fixtures built here: a 2-layer MiniGPT trained on a
period-50 token pattern, a deterministic LoRA whose B has a random rotation, cached "Pile"
tensors under ``tmp_path`` (``minigpt.data.DATA_DIR`` is patched) and a synthetic manifest
in S1's format with 20 ``val`` and 20 ``test`` documents. No test reads data/.

The real refits (C4-TFB, C4-LAP, C2) are night jobs; their end-of-run checks are the same
functions these tests call.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import minigpt.data as data_mod
from minigpt import evalset
from minigpt import posthoc_refit as pr
from minigpt.laplace import fit_laplace, load_laplace_state, save_laplace_state, select_params
from minigpt.lora import DeterministicLoRALinear, LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT
from minigpt.tfb import load_tfb_state

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------------
# Fixture constants (spec S2-T2)
# --------------------------------------------------------------------------------------------
BLOCK = 16
VOCAB = 100
PERIOD = 50
ID_TOKENS = 10_000            # one ID domain: train [0, 8000), val [8000, 9000), test [9000, 10000)
OOD_TOKENS = 1_000
VAL_RANGE = (8_000, 9_000)
N_VAL_DOCS = 20
N_TEST_DOCS = 20
N_FIT_BLOCKS = 32
FIT_BATCH = 16
RANK = 4
ALPHA = 8.0
GRID = [1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pattern_stream() -> torch.Tensor:
    pat = torch.randint(0, VOCAB, (PERIOD,), generator=torch.Generator().manual_seed(2))
    return pat.repeat(ID_TOKENS // PERIOD)


def _gpt_config() -> GPTConfig:
    return GPTConfig(n_layer=2, n_head=1, n_embd=32, block_size=BLOCK, vocab_size=VOCAB)


def _train_base() -> MiniGPT:
    """200 AdamW steps (lr 3e-3, batch 16) on the train slice of the pattern stream."""
    torch.manual_seed(0)
    model = MiniGPT(_gpt_config())
    seq = _pattern_stream()[: VAL_RANGE[0]]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(200):
        ix = torch.randint(0, len(seq) - BLOCK - 1, (16,))
        x = torch.stack([seq[i:i + BLOCK] for i in ix])
        y = torch.stack([seq[i + 1:i + BLOCK + 1] for i in ix])
        opt.zero_grad()
        model(x, y)[1].backward()
        opt.step()
    model.eval()
    return model


def _rotate_lora_b(model: MiniGPT) -> None:
    """Set every lora_B to U diag(d) V^T (QRs of Gaussian draws, generator seed 1)."""
    g = torch.Generator().manual_seed(1)
    d = torch.logspace(0, -0.5, RANK)
    for module in model.modules():
        if isinstance(module, DeterministicLoRALinear):
            u, _ = torch.linalg.qr(torch.randn(module.lora_B.shape[0], RANK, generator=g))
            v, _ = torch.linalg.qr(torch.randn(RANK, RANK, generator=g))
            with torch.no_grad():
                module.lora_B.copy_(u @ torch.diag(d) @ v.T)


def _doc_row(domain: str, idx: int, start: int, n_tokens: int, split: str) -> dict:
    sha = hashlib.sha1(f"{domain}-{idx}".encode()).hexdigest()
    k = min(3, (n_tokens - 1) // BLOCK) if split == "test" else None
    return {
        "doc_id": f"{domain}/{idx:09d}/{sha[:12]}",
        "domain": domain,
        "split": split,
        "variant": "raw",
        "stream_index": idx,
        "c_i": start,
        "L_d": n_tokens,
        "text_sha1": sha,
        "token_sha1": sha,
        "offset": 0 if split == "test" else None,
        "n_blocks": k,
    }


def _manifest_rows() -> list[dict]:
    """20 val documents that tile HackerNews [8000, 9000); 20 test documents after the cut."""
    rows = []
    start = VAL_RANGE[0]
    for i in range(N_VAL_DOCS):
        n = 45 if i % 2 == 0 else 55
        rows.append(_doc_row("hackernews", 100 + i, start, n, "val"))
        start += n
    assert start == VAL_RANGE[1]
    for j in range(12):
        rows.append(_doc_row("hackernews", 400 + j, ID_TOKENS + 120 * j, 100, "test"))
    for j in range(8):
        rows.append(_doc_row("arxiv", 50 + j, OOD_TOKENS + 120 * j, 100, "test"))
    assert sum(r["split"] == "test" for r in rows) == N_TEST_DOCS
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _save_ckpt(model: torch.nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "step": 0}, path)
    return path


def _base_cfg(fx, *, method: str, cell: str, base: Path, kind: str) -> dict:
    cfg = {
        "method": method,
        "cell": cell,
        "out_dir": str(fx.root / "out" / cell),
        "device": "cpu",
        "autocast": "none",
        "base": {"base_checkpoint": str(base), "base_kind": kind},
        "model": {"block_size": BLOCK, "n_layer": 2, "n_head": 1, "n_embd": 32,
                  "dropout": 0.1, "bias": True},
        "data": {"dataset": "pile", "pile_id_domains": ["hackernews"],
                 "pile_ood_domains": ["arxiv"], "pile_id_tokens": ID_TOKENS,
                 "pile_ood_tokens": OOD_TOKENS, "val_fraction": 0.1, "test_fraction": 0.1,
                 "data_seed": 1337},
        "fit": {"manifest_path": str(fx.manifest), "fit_split": "val",
                "fit_domain": "hackernews", "n_fit_blocks": N_FIT_BLOCKS,
                "max_blocks_per_doc": 3, "fit_seed": 0, "batch_size": FIT_BATCH},
    }
    if kind in ("blob_mean", "det_lora"):
        cfg["lora"] = {"rank": RANK, "alpha": ALPHA, "target": "ffn"}
    return cfg


def _tfb_cfg(fx, *, sampler_version: str = "v2", cell: str = "c4_tfb_v2") -> dict:
    cfg = _base_cfg(fx, method="tfb", cell=cell, base=fx.blob_ckpt, kind="blob_mean")
    cfg["tfb"] = {"sampler_version": sampler_version, "epsilon_rel": 0.003,
                  "n_search_samples": 10, "search_min": 1.0e-4, "search_max": 1.0,
                  "search_precision": 1.0e-5, "n_delta_samples": 20, "se_threshold": 0.05,
                  "se_max_doublings": 1}
    return cfg


def _lap_cfg(fx, *, cell: str, base: Path, kind: str, selection: str, reference: Path) -> dict:
    cfg = _base_cfg(fx, method="laplace", cell=cell, base=base, kind=kind)
    cfg["laplace"] = {"selection_mode": selection, "n_curvature_batches": 4,
                      "curvature_batch_size": 8, "curvature_seed": 0,
                      "n_data_seqs": VAL_RANGE[0] // BLOCK, "tokens_per_seq": BLOCK,
                      "prior_prec_grid": list(GRID), "grid_extension": [1.0e8],
                      "bisect_max_steps": 8, "match_tol": 0.10,
                      "tfb_record_path": str(fx.root / "out" / "c4_tfb_v2" / "fit_record.json"),
                      "n_delta_samples": 20, "se_threshold": 0.05, "se_max_doublings": 1,
                      "reference_state_path": str(reference)}
    return cfg


def _script():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(REPO_ROOT / "scripts"))
        return importlib.import_module("refit_posthoc")


def _run(fx, cfg: dict, name: str) -> tuple[int, dict | None]:
    """Write the YAML, run the script's main() with DATA_DIR patched, return (exit, record)."""
    path = fx.root / "cfg" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "DATA_DIR", fx.data_dir)
        code = _script().main(["--config", str(path)])
    record_path = Path(cfg["out_dir"]) / "fit_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
    return code, record


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("refit")
    data_dir = root / "data"
    (data_dir / "pile").mkdir(parents=True)
    torch.save(_pattern_stream(), data_dir / "pile" / f"hackernews_{ID_TOKENS}.pt")
    ood = torch.randint(0, VOCAB, (OOD_TOKENS,), generator=torch.Generator().manual_seed(3))
    torch.save(ood, data_dir / "pile" / f"arxiv_{OOD_TOKENS}.pt")

    base = _train_base()
    full_ckpt = _save_ckpt(base, root / "base" / "ckpt_best.pt")
    decoy = copy.deepcopy(base)
    with torch.no_grad():
        decoy.token_emb.weight.add_(0.5)
    full_decoy = _save_ckpt(decoy, root / "base" / "ckpt_decoy.pt")

    det = inject_lora(copy.deepcopy(base), LoRAConfig(rank=RANK, alpha=ALPHA, target="ffn"),
                      bayesian=False)
    _rotate_lora_b(det)
    det.eval()
    det_ckpt = _save_ckpt(det, root / "adapter" / "det_lora.pt")

    blob = inject_lora(copy.deepcopy(base),
                       LoRAConfig(rank=RANK, alpha=ALPHA, target="ffn", init_g=0.1),
                       bayesian=True)
    det_sd = det.state_dict()
    with torch.no_grad():
        for name, param in blob.named_parameters():
            if name.endswith(".lora_A_mu"):
                param.copy_(det_sd[name.replace(".lora_A_mu", ".lora_A")])
            elif name.endswith(".lora_B"):
                param.copy_(det_sd[name])
    blob_ckpt = _save_ckpt(blob, root / "adapter" / "ckpt_best.pt")
    blob_decoy = copy.deepcopy(blob)
    with torch.no_grad():
        blob_decoy.blocks[0].mlp.fc.lora_B.mul_(2.0)
    blob_decoy_ckpt = _save_ckpt(blob_decoy, root / "adapter" / "ckpt_decoy.pt")

    manifest = _write_manifest(root / "eval" / "manifest.jsonl", _manifest_rows())

    # Reference (pre-fix style) states for the median-curvature check: same windows and seed.
    train = _pattern_stream()[: VAL_RANGE[0]]
    refs = {}
    for key, model, mode in (("lora", det, "lora"), ("ffn", base, "ffn")):
        m = copy.deepcopy(model)
        torch.manual_seed(0)
        state = fit_laplace(m, train, block_size=BLOCK, batch_size=8,
                            selection=select_params(m, mode), n_batches=4, damping=1.0)
        refs[key] = root / "ref" / f"laplace_state_{key}.pt"
        refs[key].parent.mkdir(parents=True, exist_ok=True)
        save_laplace_state(state, refs[key])

    return SimpleNamespace(root=root, data_dir=data_dir, base=base, det=det,
                           full_ckpt=full_ckpt, full_decoy=full_decoy, det_ckpt=det_ckpt,
                           blob_ckpt=blob_ckpt, blob_decoy=blob_decoy_ckpt, manifest=manifest,
                           refs=refs)


@pytest.fixture(scope="module")
def tfb_runs(fx):
    """The TFB search on the fixture, once with the v2 sampler and once with v1_legacy."""
    code_v2, rec_v2 = _run(fx, _tfb_cfg(fx), "tfb_v2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the legacy sampler warns on every draw
        code_leg, rec_leg = _run(fx, _tfb_cfg(fx, sampler_version="v1_legacy",
                                              cell="c4_tfb_legacy"), "tfb_legacy")
    return SimpleNamespace(code_v2=code_v2, v2=rec_v2, code_legacy=code_leg, legacy=rec_leg)


@pytest.fixture(scope="module")
def lap_runs(fx, tfb_runs):
    """Laplace end to end: C4-LAP-like (blob_mean, LoRA A) and C2-like (full, FFN)."""
    assert tfb_runs.code_v2 == 0
    out = {}
    for cell, base, kind, selection, ref in (
        ("c4_lap", fx.blob_ckpt, "blob_mean", "lora", fx.refs["lora"]),
        ("c2", fx.full_ckpt, "full", "ffn", fx.refs["ffn"]),
    ):
        cfg = _lap_cfg(fx, cell=cell, base=base, kind=kind, selection=selection, reference=ref)
        out[cell] = _run(fx, cfg, f"lap_{cell}")
    return out


# --------------------------------------------------------------------------------------------
# S2-T2 (a): the search log
# --------------------------------------------------------------------------------------------

def test_t2a_v2_search_log_complete_and_consistent(tfb_runs):
    assert tfb_runs.code_v2 == 0
    rec = tfb_runs.v2
    log = rec["tfb"]["search_log"]
    assert len(log) == 18  # pre-check + 17 bisection steps from [1e-4, 1] to 1e-5
    for step in log:
        for key in ("sigma_q", "avg_loss", "anchor_loss", "accepted", "sampler_version"):
            assert key in step
    sigma_star = rec["tfb"]["sigma_q_star"]
    accepted = [s["sigma_q"] for s in log if s["accepted"]]
    rejected = [s["sigma_q"] for s in log if not s["accepted"]]
    assert accepted and sigma_star == max(accepted)
    assert all(s > sigma_star for s in rejected)
    assert {s["sampler_version"] for s in log} == {"v2"}
    assert rec["sampler_version"] == "v2"
    assert rec["checks"] == {"t2a_search_log": True, "t2b_precheck": True,
                             "t2c_fit_isolation": True}
    assert rec["failures"] == [] and rec["exit_code"] == 0


def test_t2a_legacy_run_records_legacy_version(tfb_runs):
    assert tfb_runs.code_legacy == 0
    log = tfb_runs.legacy["tfb"]["search_log"]
    assert {s["sampler_version"] for s in log} == {"v1_legacy"}
    assert tfb_runs.legacy["checks"]["t2a_search_log"] is True


def test_t2_tfb_record_budget_and_state(fx, tfb_runs):
    rec = tfb_runs.v2
    assert rec["rho_tfb"] > 0
    assert math.isclose(rec["rho_tfb"], rec["delta_nll_tfb"] / rec["ell0"], rel_tol=1e-12)
    final = rec["delta_measurements"][-1]
    assert final["delta_nll"] == rec["delta_nll_tfb"]
    assert rec["delta_measurements"][0]["n_samples"] == 20
    assert rec["delta_measurements"][0]["seeds"] == list(range(20))
    state = load_tfb_state(rec["state_path"])  # S2-T5(d): the fit state reads back as v2
    assert state.sampler_version == "v2" and state.sigma_q == rec["tfb"]["sigma_q_star"]
    assert rec["state_sha256"] == _sha256(Path(rec["state_path"]))


# --------------------------------------------------------------------------------------------
# S2-T2 (b): pre-check and the two search errors
# --------------------------------------------------------------------------------------------

def test_t2b_precheck_rejects_search_max(tfb_runs):
    first = tfb_runs.v2["tfb"]["search_log"][0]
    assert first["stage"] == "precheck" and first["sigma_q"] == 1.0
    assert first["accepted"] is False
    assert abs(first["avg_loss"] - first["anchor_loss"]) > first["tolerance"]


def test_t2b_search_range_too_small_raises(fx):
    cfg = _tfb_cfg(fx, cell="tfb_small_range")
    cfg["tfb"].update({"search_min": 1.0e-7, "search_max": 1.0e-6, "search_precision": 1.0e-7})
    with pytest.raises(ValueError, match="search range too small"):
        _run(fx, cfg, "tfb_small_range")


def test_t2b_no_sigma_accepted_raises(fx):
    cfg = _tfb_cfg(fx, cell="tfb_none_accepted")
    cfg["tfb"].update({"search_min": 0.5, "search_max": 1.0})
    with pytest.raises(ValueError, match="no sigma_q accepted"):
        _run(fx, cfg, "tfb_none_accepted")


# --------------------------------------------------------------------------------------------
# S2-T2 (c): the fit reads val documents only
# --------------------------------------------------------------------------------------------

def test_t2c_blocks_inside_val_docs_and_no_test_ids(tfb_runs):
    rec = tfb_runs.v2
    rows = _manifest_rows()
    by_id = {r["doc_id"]: r for r in rows if r["split"] == "val"}
    test_ids = {r["doc_id"] for r in rows if r["split"] == "test"}
    blocks = rec["fit_set"]["blocks"]
    assert len(blocks) == N_FIT_BLOCKS and rec["fit_set"]["units"] == "document"
    per_doc: dict[str, int] = {}
    for doc_id, start in blocks:
        doc = by_id[doc_id]
        assert doc["c_i"] <= start
        assert start + BLOCK + 1 <= doc["c_i"] + doc["L_d"]
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
    assert max(per_doc.values()) <= 3
    assert not set(per_doc) & test_ids
    starts = sorted(start for _, start in blocks)
    assert all(b - a >= BLOCK for a, b in zip(starts, starts[1:]))  # non-overlapping
    ids_hash = hashlib.sha256("\n".join(sorted(per_doc)).encode()).hexdigest()
    assert rec["fit_set"]["doc_ids_sha256"] == ids_hash


def test_t2c_fit_tokens_match_the_cached_val_slice(fx):
    cfg = _tfb_cfg(fx)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "DATA_DIR", fx.data_dir)
        data = pr.load_refit_data(cfg)
    fit = pr.build_fit_blocks(_manifest_rows(), data, cfg)
    stream = _pattern_stream()
    for (doc_id, start), x, y in zip(fit.blocks, fit.x, fit.y):
        assert torch.equal(x, stream[start:start + BLOCK])
        assert torch.equal(y, stream[start + 1:start + BLOCK + 1])


@pytest.mark.parametrize("corruption", ["shared_id", "overlapping_range"])
def test_t2c_eval_document_in_fit_set_exits_nonzero(fx, corruption, capsys):
    rows = _manifest_rows()
    val = [r for r in rows if r["split"] == "val"]
    test_hn = [r for r in rows if r["split"] == "test" and r["domain"] == "hackernews"]
    if corruption == "shared_id":
        test_hn[0]["doc_id"] = val[3]["doc_id"]
    else:
        test_hn[0]["c_i"] = val[5]["c_i"] + 10
    cfg = _tfb_cfg(fx, cell=f"tfb_{corruption}")
    cfg["fit"]["manifest_path"] = str(
        _write_manifest(fx.root / "eval" / f"manifest_{corruption}.jsonl", rows))
    code, record = _run(fx, cfg, f"tfb_{corruption}")
    assert code != 0 and record is None
    assert "eval document in fit set" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# S2-T2 (d): every key is explicit
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("section,key", [("tfb", "epsilon_rel"), ("lora", "rank")])
def test_t2d_missing_key_raises_keyerror_before_any_helper(fx, section, key, monkeypatch):
    def _must_not_load(*args, **kwargs):
        raise AssertionError("data was loaded before the config check")

    monkeypatch.setattr("minigpt.refit_data.load_pile_data", _must_not_load)
    cfg = _tfb_cfg(fx, cell="tfb_missing_key")
    del cfg[section][key]
    with pytest.raises(KeyError, match=key):
        _run(fx, cfg, f"tfb_missing_{key}")


def test_t2d_string_number_is_rejected(fx):
    cfg = _tfb_cfg(fx)
    cfg["tfb"]["search_min"] = "1e-4"  # what PyYAML makes of 1e-4 without a dot
    with pytest.raises(ValueError, match="search_min"):
        pr.check_config(cfg)


# --------------------------------------------------------------------------------------------
# S2-T2 (e): the two samplers find different sigma_q*
# --------------------------------------------------------------------------------------------

def test_t2e_v2_and_legacy_sigma_star_differ(tfb_runs):
    s_v2 = tfb_runs.v2["tfb"]["sigma_q_star"]
    s_leg = tfb_runs.legacy["tfb"]["sigma_q_star"]
    assert abs(s_v2 - s_leg) > 10 * 1.0e-5


# --------------------------------------------------------------------------------------------
# One weight draw per seed (Section 5.6)
# --------------------------------------------------------------------------------------------

def test_measure_delta_nll_draws_once_per_seed(fx):
    model = copy.deepcopy(fx.det)
    xs = _pattern_stream()[:4 * 8 * BLOCK + 1]
    batches = [(xs[i * 8 * BLOCK:(i + 1) * 8 * BLOCK].view(8, BLOCK),
                xs[i * 8 * BLOCK + 1:(i + 1) * 8 * BLOCK + 1].view(8, BLOCK)) for i in range(4)]
    calls = []

    def sample_fn(seed):
        calls.append(seed)
        return {}

    out = pr.measure_delta_nll(model, sample_fn, batches, list(range(5)), autocast="none")
    assert calls == [0, 1, 2, 3, 4]
    assert abs(out["delta_nll"]) < 1e-6 and out["se"] < 1e-6  # no perturbation
    assert math.isclose(out["ell_bma"], out["ell0"], rel_tol=1e-5)
    assert len(out["per_sample_nll"]) == 5


# --------------------------------------------------------------------------------------------
# S2-T4 (a)-(b): match_prior_precision on synthetic Delta-NLL functions
# --------------------------------------------------------------------------------------------

def _match(fn, target):
    return pr.match_prior_precision(fn, target, GRID, [1.0e8], 8, 0.10, 0.05,
                                    n_samples=20, se_max_doublings=1)


def _smooth(se_frac=0.0, target=1.0):
    return lambda lam, n: (1.0 / (1.0 + lam / 100.0), se_frac * target)


def test_t4a_matched_within_band():
    res = _match(_smooth(), 0.05)
    assert res["status"] == "matched" and res["exit_code"] == 0
    assert abs(res["delta_nll"] - 0.05) <= 0.10 * 0.05
    assert res["monotone"] is True
    assert [e["lam"] for e in res["evaluations"][:8]] == GRID


def test_t4a_tighter_than_budget():
    res = _match(_smooth(), 2.0)
    assert res["status"] == "tighter_than_budget" and res["exit_code"] == 0
    assert res["lambda_star"] == 1.0


def test_t4a_not_matched():
    res = _match(_smooth(), 1.0e-7)
    assert res["status"] == "not_matched" and res["exit_code"] == 0
    assert res["lambda_star"] is None
    assert res["evaluations"][-1]["lam"] == 1.0e8


def test_t4a_step_function_bisection_failed():
    target = 0.05

    def step(lam, n):
        return (2 * target if lam < 2000.0 else 0.5 * target), 0.0

    res = _match(step, target)
    assert res["status"] == "bisection_failed" and res["exit_code"] != 0
    assert sum(e["stage"] == "bisect" for e in res["evaluations"]) == 8
    assert res["bracket"] == [1.0e3, 1.0e4]


def test_t4b_one_doubling_then_matched():
    target = 0.05
    calls = []

    def fn(lam, n):
        calls.append(n)
        return 1.0 / (1.0 + lam / 100.0), (0.08 if n == 20 else 0.03) * target

    res = _match(fn, target)
    assert res["status"] == "matched" and res["exit_code"] == 0
    assert calls.count(40) == 1 and set(calls) == {20, 40}
    assert res["n_samples"] == 40


def test_t4b_still_noisy_after_one_doubling():
    target = 0.05
    calls = []

    def fn(lam, n):
        calls.append(n)
        return 1.0 / (1.0 + lam / 100.0), 0.08 * target

    res = _match(fn, target)
    assert res["status"] == "matched_noisy" and res["exit_code"] == 0
    assert calls.count(40) == 1 and 80 not in calls
    assert res["delta_nll_first"] is not None and res["se_first"] == 0.08 * target


def test_t4_non_monotone_grid_is_recorded():
    target = 0.05
    values = {1.0: 0.9, 10.0: 0.95, 100.0: 0.5}

    def fn(lam, n):
        return values.get(lam, 1.0 / (1.0 + lam / 100.0)), 0.0

    res = _match(fn, target)
    assert res["monotone"] is False and res["status"] == "matched"


# --------------------------------------------------------------------------------------------
# S2-T4 (c): base loading and the Laplace fit record
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["full", "blob_mean", "det_lora"])
def test_t4c_loaded_weights_equal_file_weights(fx, kind):
    ckpt = {"full": fx.full_ckpt, "blob_mean": fx.blob_ckpt, "det_lora": fx.det_ckpt}[kind]
    cfg = _base_cfg(fx, method="laplace", cell="load", base=ckpt, kind=kind)
    model, info = pr.load_posthoc_base(cfg)
    file_sd = torch.load(ckpt, weights_only=False)["model_state_dict"]
    for name, tensor in model.state_dict().items():
        src = name.replace(".lora_A", ".lora_A_mu") if kind == "blob_mean" and name.endswith(
            ".lora_A") else name
        assert torch.equal(tensor, file_sd[src]), name
    assert info["sha256"] == _sha256(ckpt) and info["kind"] == kind
    if kind == "blob_mean":
        assert info["dropped_keys"] and all(k.endswith(".lora_A_g") for k in info["dropped_keys"])
        assert len(info["dropped_keys"]) == 4
    else:
        assert info["dropped_keys"] == []


def test_t4c_blob_mean_missing_lora_a_mu_raises_keyerror(fx):
    sd = torch.load(fx.blob_ckpt, weights_only=False)["model_state_dict"]
    del sd["blocks.1.mlp.proj.lora_A_mu"]
    path = fx.root / "adapter_broken" / "ckpt_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": sd}, path)
    cfg = _base_cfg(fx, method="laplace", cell="load", base=path, kind="blob_mean")
    with pytest.raises(KeyError, match="lora_A_mu"):
        pr.load_posthoc_base(cfg)


def test_t4c_full_missing_key_raises_keyerror(fx):
    sd = torch.load(fx.full_ckpt, weights_only=False)["model_state_dict"]
    del sd["ln_f.weight"]
    path = fx.root / "base_broken" / "ckpt_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": sd}, path)
    cfg = _base_cfg(fx, method="laplace", cell="load", base=path, kind="full")
    with pytest.raises(KeyError, match="ln_f.weight"):
        pr.load_posthoc_base(cfg)


@pytest.mark.parametrize("cell", ["c4_lap", "c2"])
def test_t4c_laplace_record_fields(fx, lap_runs, cell):
    code, rec = lap_runs[cell]
    assert rec is not None
    base, decoy = {"c4_lap": (fx.blob_ckpt, fx.blob_decoy),
                   "c2": (fx.full_ckpt, fx.full_decoy)}[cell]
    assert rec["base"]["path"] == str(base)
    assert rec["base"]["kind"] == {"c4_lap": "blob_mean", "c2": "full"}[cell]
    assert rec["base"]["sha256"] == _sha256(base) != _sha256(decoy)
    assert rec["n_data_seqs"] == VAL_RANGE[0] // BLOCK and rec["tokens_per_seq"] == BLOCK
    sweep = rec["sweep"]
    assert [e["lam"] for e in sweep[:8]] == GRID
    for e in sweep:
        for key in ("lam", "n_samples", "stage", "delta_nll", "se", "ell_bma", "seeds"):
            assert key in e
    assert rec["lambdas_tried"] == [e["lam"] for e in sweep]
    assert rec["match_status"] in pr.MATCH_EXIT_CODES
    assert code == rec["exit_code"] == pr.MATCH_EXIT_CODES[rec["match_status"]]
    assert rec["match_status"] != "bisection_failed"
    assert rec["fit_set"]["doc_ids_sha256"] == pr.doc_ids_sha256(
        [d for d, _ in rec["fit_set"]["blocks"]])
    assert rec["seeds"]["fit_seed"] == 0 and rec["seeds"]["curvature_seed"] == 0
    assert rec["seeds"]["delta_seeds"] == list(range(20))
    assert isinstance(rec["git_commit"], str) and len(rec["git_commit"]) == 40
    assert rec["sampler_version"] == "v2"
    assert rec["curvature"]["median_fhat"] > 0
    assert rec["curvature"]["median_ratio"] == pytest.approx(1.0)
    assert rec["checks"]["curvature_median_vs_reference"] is True
    assert rec["target_delta_nll"] == pytest.approx(rec["rho_tfb"] * rec["ell0"])
    if rec["match_status"] in ("matched", "matched_noisy", "tighter_than_budget"):
        state = load_laplace_state(rec["state_path"])  # S2-T5(d)
        assert state.sampler_version == "v2" and state.prior_prec == rec["lambda_star"]
        assert state.base_sha256 == rec["base"]["sha256"]


def test_t4c_c4_lap_target_equals_tfb_delta(lap_runs, tfb_runs):
    _, rec = lap_runs["c4_lap"]
    assert rec["same_fit_as_tfb"] is True
    assert rec["target_delta_nll"] == pytest.approx(tfb_runs.v2["delta_nll_tfb"], rel=1e-9)


def test_laplace_n_data_seqs_mismatch_exits_nonzero(fx, tfb_runs, capsys):
    cfg = _lap_cfg(fx, cell="lap_bad_nseq", base=fx.blob_ckpt, kind="blob_mean",
                   selection="lora", reference=fx.refs["lora"])
    cfg["laplace"]["n_data_seqs"] += 1
    code, record = _run(fx, cfg, "lap_bad_nseq")
    assert code != 0 and record is None
    assert "n_data_seqs" in capsys.readouterr().err


def test_curvature_isolation_detects_eval_doc_in_train_slice():
    rows = _manifest_rows()
    rows[-1]["domain"] = "hackernews"
    rows[-1]["c_i"] = 100
    with pytest.raises(pr.RefitCheckError, match="eval document in curvature set"):
        pr.check_curvature_isolation(rows, {"hackernews": (0, VAL_RANGE[0])})
    pr.check_curvature_isolation(_manifest_rows(), {"hackernews": (0, VAL_RANGE[0])})


# --------------------------------------------------------------------------------------------
# Data ranges with two ID domains (the C2 layout)
# --------------------------------------------------------------------------------------------

def test_load_refit_data_two_id_domains(tmp_path, monkeypatch):
    (tmp_path / "pile").mkdir()
    wiki = torch.arange(10_000) % VOCAB
    se = (torch.arange(10_000) * 7) % VOCAB
    torch.save(wiki, tmp_path / "pile" / "wikipedia_en_10000.pt")
    torch.save(se, tmp_path / "pile" / "stackexchange_10000.pt")
    torch.save(torch.zeros(500, dtype=torch.long), tmp_path / "pile" / "arxiv_500.pt")
    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
    cfg = {"data": {"dataset": "pile", "pile_id_domains": ["wikipedia_en", "stackexchange"],
                    "pile_ood_domains": ["arxiv"], "pile_id_tokens": 10_000,
                    "pile_ood_tokens": 500, "val_fraction": 0.1, "test_fraction": 0.1,
                    "data_seed": 1337},
           "fit": {"fit_domain": "stackexchange"}}
    data = pr.load_refit_data(cfg)
    assert data.fit_range == (6_000, 8_000)
    assert torch.equal(data.fit_tokens, se[6_000:8_000])
    assert data.train_ranges == {"wikipedia_en": (0, 10_000), "stackexchange": (0, 6_000)}
    assert len(data.train) == 16_000


def test_load_refit_data_refuses_a_cache_miss(tmp_path, monkeypatch):
    (tmp_path / "pile").mkdir()
    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path)
    cfg = {"data": {"dataset": "pile", "pile_id_domains": ["hackernews"],
                    "pile_ood_domains": [], "pile_id_tokens": 10_000, "pile_ood_tokens": 500,
                    "val_fraction": 0.1, "test_fraction": 0.1, "data_seed": 1337},
           "fit": {"fit_domain": "hackernews"}}
    with pytest.raises(FileNotFoundError, match="hackernews_10000"):
        pr.load_refit_data(cfg)


# --------------------------------------------------------------------------------------------
# S2-T4 (d), last check: score files record the base the fit used
# --------------------------------------------------------------------------------------------

def test_check_scores_compares_base_sha(tmp_path):
    records, scores = [], []
    for cell, score_set in (("c2", "c2_refit"), ("c4_lap", "c4_lap_refit"),
                            ("c4_tfb", "c4_tfb_fixed")):
        sha = hashlib.sha256(cell.encode()).hexdigest()
        rec = tmp_path / f"{cell}_fit_record.json"
        rec.write_text(json.dumps({"cell": cell, "base": {"sha256": sha, "path": "x.pt"}}))
        records.append(rec)
        score = tmp_path / f"{score_set}__test__main.pt"
        torch.save({"meta": {"score_set": score_set,
                             "checkpoint_sha256": {"x.pt": sha, "state.pt": "ab" * 32}}}, score)
        scores.append(score)
    assert pr.check_scores(scores, records) == []
    torch.save({"meta": {"score_set": "c2_refit", "checkpoint_sha256": {"x.pt": "00" * 32}}},
               scores[0])
    failures = pr.check_scores(scores, records)
    assert len(failures) == 1 and "c2_refit" in failures[0]
    assert _script().main(["--check-scores", *map(str, scores),
                           "--records", *map(str, records)]) != 0


# --------------------------------------------------------------------------------------------
# S2 re-score end to end (spec Section 2, S1 decision D6): the fixture refit outputs, scored by
# S1's scorer under the names c2_refit, c4_lap_refit and c4_tfb_fixed, then T4(d)'s last check
# --------------------------------------------------------------------------------------------

RESCORE_DOCS = 3               # per domain, one block each
RESCORE_SEED_OFFSET = 100_000  # the `test` offset of configs/i2_eval.yaml
RESCORE_CELLS = {"c2_refit": "c2", "c4_lap_refit": "c4_lap", "c4_tfb_fixed": "c4_tfb"}


def _scorer():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(REPO_ROOT / "scripts"))
        return importlib.import_module("eval_c_checkpoints")


def _rescore_eval_cfg(root: Path, eval_dir: Path) -> dict:
    """A small eval YAML; the S2 labels and N come from configs/i2_eval.yaml itself."""
    real = yaml.safe_load((REPO_ROOT / "configs" / "i2_eval.yaml").read_text(encoding="utf-8"))
    sc = real["scoring"]
    return {
        "eval_set": {
            "out_dir": str(eval_dir), "block_size": BLOCK,
            "domains": [{"key": "stackexchange", "role": "id_base", "stripped_copy": False},
                        {"key": "hackernews", "role": "id_adapter", "stripped_copy": False},
                        {"key": "arxiv", "role": "ood", "stripped_copy": False}],
        },
        "scoring": {
            "seed_base": 0,
            "seed_offsets": dict(sc["seed_offsets"]),
            "n_samples": {s: sc["n_samples"][s] for s in RESCORE_CELLS},
            "score_sets": {"legacy_d1": [], "test": [s for s in sc["score_sets"]["test"]
                                                     if s in RESCORE_CELLS],
                           "test_hn": [], "arxiv_stripped": []},
            "labels": {s: sc["labels"][s] for s in RESCORE_CELLS},
            "batch_size": {"legacy_d1": 1, "test": 1, "test_hn": 1, "arxiv_stripped": 1},
            "amp_fp16": True,
            "determinism": dict(sc["determinism"]),
            "out_dir": str(root / "scores"),
        },
        "analysis": {"ood_domains": ["arxiv"], "fixture": "rescore"},
        "checks": {"repro": {"spread_seed_bases": [1_000_000]}},
    }


def _rescore_blocks(fx) -> dict:
    stream = _pattern_stream()
    ood = torch.load(fx.data_dir / "pile" / f"arxiv_{OOD_TOKENS}.pt")
    rows, doc_ids, domains = [], [], []
    for domain, src, start in (("hackernews", stream, 9_000), ("arxiv", ood, 0)):
        for j in range(RESCORE_DOCS):
            s = start + 100 * j
            rows.append(src[s:s + BLOCK + 1])
            doc_ids.append(f"{domain}/{j:09d}/rescore")
            domains.append(domain)
    n = len(rows)
    return {"tokens": torch.stack(rows).to(torch.int32), "doc_id": doc_ids, "domain": domains,
            "offset": torch.zeros(n, dtype=torch.int64),
            "weight": torch.ones(n, dtype=torch.float32)}


@pytest.fixture(scope="module")
def rescore(fx, tfb_runs, lap_runs):
    # A TFB refit under the real cell name, so SCORE_SET_CELLS finds its record.
    code_tfb, _ = _run(fx, _tfb_cfg(fx, cell="c4_tfb"), "tfb_c4")
    assert code_tfb == 0 and all(lap_runs[c][0] == 0 for c in ("c4_lap", "c2"))
    refit_yamls = {"c2_refit": fx.root / "cfg" / "lap_c2.yaml",
                   "c4_lap_refit": fx.root / "cfg" / "lap_c4_lap.yaml",
                   "c4_tfb_fixed": fx.root / "cfg" / "tfb_c4.yaml"}
    root = fx.root / "rescore"
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True)
    blocks = _rescore_blocks(fx)
    torch.save(blocks, eval_dir / "blocks_test.pt")
    (eval_dir / "manifest.jsonl").write_text("".join(
        json.dumps({"doc_id": d, "domain": m, "split": "test"}) + "\n"
        for d, m in zip(blocks["doc_id"], blocks["domain"])), encoding="utf-8")
    cfg = _rescore_eval_cfg(root, eval_dir)
    cfg_path = root / "i2_eval_rescore.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    scores_dir = root / "scores"
    scores_dir.mkdir()
    sha = hashlib.sha256(json.dumps(cfg["analysis"], sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()
    (scores_dir / "prereg.json").write_text(json.dumps(
        {"analysis_sha256": sha, "frozen_at": "2026-09-26T00:00:00+00:00"}), encoding="utf-8")

    ecc = _scorer()
    det = (torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ecc, "POSTHOC_REFIT_CONFIGS", refit_yamls)
            # The scorer sets this variable; setenv makes the context restore the old value.
            mp.setenv("CUBLAS_WORKSPACE_CONFIG",
                      cfg["scoring"]["determinism"]["cublas_workspace_config"])
            ecc.main(["--eval-config", str(cfg_path), "--eval-set", "test",
                      "--run-tag", "main", "--device", "cpu"])
    finally:
        torch.use_deterministic_algorithms(det[0], warn_only=det[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = det[2], det[3]
    records = {s: fx.root / "out" / c / "fit_record.json" for s, c in RESCORE_CELLS.items()}
    files = {s: scores_dir / f"{s}__test__main.pt" for s in RESCORE_CELLS}
    return SimpleNamespace(files=files, records=records, refit_yamls=refit_yamls, blocks=blocks,
                           n_samples=cfg["scoring"]["n_samples"])


def test_s2_rescore_score_files_pass_check_scores(rescore):
    files = [rescore.files[s] for s in RESCORE_CELLS]
    records = [rescore.records[s] for s in RESCORE_CELLS]
    assert pr.check_scores(files, records) == []
    assert _script().main(["--check-scores", *map(str, files),
                           "--records", *map(str, records)]) == 0


def test_s2_rescore_meta_records_v2_and_the_fitted_files(rescore):
    for score_set in RESCORE_CELLS:
        meta = torch.load(rescore.files[score_set], weights_only=True)["meta"]
        record = json.loads(rescore.records[score_set].read_text(encoding="utf-8"))
        assert meta["score_set"] == score_set and meta["sampler"] == "v2"
        assert meta["n_samples"] == 20
        # Every warning raised while scoring lands here; a v1_legacy draw would warn.
        assert not [w for w in meta["determinism_warnings"] if "v1_legacy" in w]
        assert meta["checkpoint_sha256"] == {
            Path(record["base"]["path"]).as_posix(): record["base"]["sha256"],
            Path(record["state_path"]).as_posix(): record["state_sha256"],
        }


def test_s2_rescore_draws_are_the_v2_sampler_at_the_scorer_seed(rescore):
    """Row b of the score file equals N v2 draws at seed (offset + b) * N + s, by hand."""
    from minigpt.laplace import apply_sampled_params, sample_laplace_params
    from minigpt.tfb import sample_tfb_params

    for score_set in RESCORE_CELLS:
        refit_cfg = yaml.safe_load(rescore.refit_yamls[score_set].read_text(encoding="utf-8"))
        model, _ = pr.load_posthoc_base(refit_cfg)
        record = json.loads(rescore.records[score_set].read_text(encoding="utf-8"))
        if refit_cfg["method"] == "tfb":
            state, sample = load_tfb_state(record["state_path"]), sample_tfb_params
        else:
            state, sample = load_laplace_state(record["state_path"]), sample_laplace_params
        assert state.sampler_version == "v2"
        payload = torch.load(rescore.files[score_set], weights_only=True)
        n = rescore.n_samples[score_set]
        for b in (0, RESCORE_DOCS):  # one HackerNews and one arXiv block
            tok = rescore.blocks["tokens"][b].long()
            seed_b = RESCORE_SEED_OFFSET + b
            ref = []
            with torch.no_grad():
                for s in range(n):
                    with apply_sampled_params(model, sample(state, seed=seed_b * n + s)):
                        logits, _ = model(tok[None, :-1])
                    lp = torch.log_softmax(logits.float(), dim=-1)[0]
                    ref.append(lp.gather(-1, tok[1:, None]).squeeze(-1))
            assert torch.allclose(payload["logp_real"][b], torch.stack(ref), atol=1e-6), (
                score_set, b)


# --------------------------------------------------------------------------------------------
# The three real configs carry the frozen settings (Section 3)
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cell,method,kind,domain", [
    ("c4_tfb", "tfb", "blob_mean", "hackernews"),
    ("c4_lap", "laplace", "blob_mean", "hackernews"),
    ("c2", "laplace", "full", "stackexchange"),
])
def test_real_configs_pass_the_check_and_hold_the_frozen_values(cell, method, kind, domain):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / f"i2_posthoc_{cell}.yaml").read_text())
    pr.check_config(cfg)
    assert cfg["method"] == method and cfg["cell"] == cell
    assert cfg["out_dir"] == f"data/checkpoints/i2_posthoc/{cell}"
    assert cfg["base"]["base_kind"] == kind
    assert cfg["base"]["base_checkpoint"] == (
        "data/checkpoints/c0/ckpt_best.pt" if kind == "full" else
        "data/checkpoints/c3/ckpt_best.pt")
    assert cfg["fit"] == {"manifest_path": "data/eval_i2/manifest.jsonl", "fit_split": "val",
                          "fit_domain": domain, "n_fit_blocks": 640, "max_blocks_per_doc": 3,
                          "fit_seed": 0, "batch_size": 32}
    assert cfg["data"]["data_seed"] == 1337
    if kind == "blob_mean":
        assert cfg["lora"] == {"rank": 16, "alpha": 32.0, "target": "ffn"}
    if method == "tfb":
        assert cfg["tfb"] == {"sampler_version": "v2", "epsilon_rel": 0.003,
                              "n_search_samples": 10, "search_min": 1.0e-4, "search_max": 1.0,
                              "search_precision": 1.0e-5, "n_delta_samples": 20,
                              "se_threshold": 0.05, "se_max_doublings": 1}
    else:
        lap = cfg["laplace"]
        assert lap["prior_prec_grid"] == [10.0 ** k for k in range(8)]  # 7 decades
        assert lap["grid_extension"] == [1.0e8]
        assert (lap["bisect_max_steps"], lap["match_tol"]) == (8, 0.10)
        assert (lap["n_delta_samples"], lap["se_threshold"], lap["se_max_doublings"]) == (
            20, 0.05, 1)
        assert lap["tokens_per_seq"] == 256 and lap["curvature_seed"] == 0
        assert lap["tfb_record_path"] == "data/checkpoints/i2_posthoc/c4_tfb/fit_record.json"
        assert lap["reference_state_path"] == f"data/checkpoints/{cell}/laplace_state.pt"
        n_train = (80_000_000 if cell == "c4_lap" else 160_000_000)
        assert lap["n_data_seqs"] == n_train // 256
        assert lap["n_curvature_batches"] == 30
        assert lap["curvature_batch_size"] == (32 if cell == "c4_lap" else 16)
        assert lap["selection_mode"] == ("lora" if cell == "c4_lap" else "ffn")


# Manifest schema shared with the eval-set builder (S1 spec section 5.2: c_i, L_d). Rows come from
# the builder's own row function, so the refit and the builder cannot drift apart again (defect D8).
def _builder_test_row(domain: str, index: int, start: int, n_tokens: int) -> dict:
    cand = evalset.Candidate(key=domain, index=index, start=start,
                             doc_id=f"{domain}/{index:09d}/builder", text_sha1="0" * 40,
                             tokens=np.zeros(n_tokens, dtype=np.int32), n_blocks=1, offset=0)
    return evalset._test_rows(cand, "main", "raw")


def test_fit_isolation_reads_builder_manifest_rows():
    test_rows = [_builder_test_row("hackernews", 7, start=1000, n_tokens=500)]
    fit_row = dict(test_rows[0], doc_id="hackernews/000000003/val", split="val", eval_set=None,
                   c_i=1200, L_d=300)
    with pytest.raises(pr.RefitCheckError, match="eval document in fit set"):
        pr.check_fit_isolation([fit_row], test_rows)
    clear_row = dict(fit_row, c_i=5000)
    pr.check_fit_isolation([clear_row], test_rows)


def test_curvature_isolation_reads_builder_manifest_rows():
    rows = [_builder_test_row("hackernews", 7, start=1000, n_tokens=500)]
    with pytest.raises(pr.RefitCheckError, match="eval document in curvature set"):
        pr.check_curvature_isolation(rows, {"hackernews": (0, 1200)})
