"""Tests for scripts/check_eval_rebuild.py (spec specs/i2-eval-rebuild.md, S1-T5b).

The checker re-derives the result tables from score files with numpy, torch.load and
sklearn only. These tests build synthetic score files in the section 5.5 format, and
reproduce the section 5.6 reference values of the T4 fixtures (F-i, F-ii).
"""

import ast
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_eval_rebuild.py"

# Fixture constants (section 2: test fixtures keep their constants in the test module).
T_FIX = 8
N_FIX = 3
DOCS_PER_DOMAIN = 10
RESAMPLES_FIX = 20
TOL_FIX = 1.0e-6
CREATED_AT_FIX = "2026-09-26T12:00:00+00:00"
FROZEN_AT_FIX = "2026-09-26T10:00:00+00:00"
DOMAIN_KEYS = ["stackexchange", "hackernews", "arxiv", "freelaw", "pubmed_abstracts"]
OOD_KEYS = ["arxiv", "freelaw", "pubmed_abstracts"]
SCORE_SETS_FIX = {"test": ["c0", "c1", "c3"], "test_hn": ["c3"], "arxiv_stripped": ["c1", "c3"]}
N_SAMPLES_FIX = {"c0": 1, "c1": N_FIX, "c3": N_FIX}
SCORES = ["blk_g", "blk_mi", "blk_tu", "blk_au", "blk_nll", "blk_maxprob_unc"]


@pytest.fixture(scope="module")
def cer():
    spec = importlib.util.spec_from_file_location("check_eval_rebuild", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_eval_rebuild"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Section 3: realized-token Jensen gap
# ---------------------------------------------------------------------------


def test_block_gap_matches_hand_value(cer):
    lp = np.log(np.array([[[0.2, 0.5], [0.8, 0.5]]], dtype=np.float64)).astype(np.float32)
    out = cer.block_scores_from_logp(lp)
    g0 = math.log(0.5) - 0.5 * (math.log(0.2) + math.log(0.8))
    assert out["blk_g"][0] == pytest.approx(0.5 * g0, abs=1e-6)
    assert out["blk_nll"][0] == pytest.approx(math.log(2.0), abs=1e-6)
    assert out["g_tok_min"] >= -1e-6


def test_block_gap_is_zero_for_one_sample_and_identical_samples(cer):
    rng = np.random.default_rng(0)
    lp1 = -rng.uniform(0.1, 9.0, size=(4, 1, 16)).astype(np.float32)
    assert np.abs(cer.block_scores_from_logp(lp1)["blk_g"]).max() <= 1e-6
    lp3 = np.repeat(lp1, 3, axis=1)
    assert np.abs(cer.block_scores_from_logp(lp3)["blk_g"]).max() <= 1e-6


def test_block_gap_matches_torch_logsumexp(cer):
    rng = np.random.default_rng(1)
    lp = (-rng.uniform(0.1, 12.0, size=(5, 7, 32))).astype(np.float32)
    t = torch.from_numpy(lp).double()
    log_pbar = torch.logsumexp(t, dim=1) - math.log(7)
    g = (log_pbar - t.mean(dim=1)).mean(dim=1).numpy()
    np.testing.assert_allclose(cer.block_scores_from_logp(lp)["blk_g"], g, atol=1e-12)


# ---------------------------------------------------------------------------
# Section 5.6: weighted AUROC and weighted FPR@95
# ---------------------------------------------------------------------------


def _pairwise_auroc(y, s, w):
    i_idx, o_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    num = 0.0
    for i in i_idx:
        for j in o_idx:
            num += w[i] * w[j] * ((s[j] > s[i]) + 0.5 * (s[j] == s[i]))
    return num / (w[i_idx].sum() * w[o_idx].sum())


def _threshold_fpr(y, s, w, target):
    ood, idm = y == 1, y == 0
    best = None
    for tau in np.unique(s):
        tpr = w[ood & (s >= tau)].sum() / w[ood].sum()
        if tpr >= target and (best is None or tau > best):
            best = tau
    return w[idm & (s >= best)].sum() / w[idm].sum()


def test_weighted_auroc_matches_pairwise_formula(cer):
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(40), np.ones(50)].astype(int)
    s = np.round(rng.normal(size=90) + 0.7 * y, 1)  # rounding makes ties
    w = rng.uniform(0.1, 1.0, size=90)
    assert cer.weighted_auroc(y, s, w) == pytest.approx(_pairwise_auroc(y, s, w), abs=1e-12)


def test_weighted_fpr_matches_threshold_definition(cer):
    rng = np.random.default_rng(4)
    for rep in range(5):
        y = np.r_[np.zeros(60), np.ones(70)].astype(int)
        s = np.round(rng.normal(size=130) + y, 1)
        w = rng.uniform(0.1, 1.0, size=130)
        got = cer.weighted_fpr_at_tpr(y, s, w, 0.95)
        assert got == pytest.approx(_threshold_fpr(y, s, w, 0.95), abs=1e-12), rep


def _fixture_fi():
    rng = np.random.default_rng(101)
    y_doc = np.r_[np.zeros(300), np.ones(300)].astype(int)
    k = rng.integers(1, 4, size=600)
    s_doc = rng.normal(loc=1.0 * y_doc, scale=1.0)
    names = np.array([f"d{i:04d}" for i in range(600)])
    doc_of_block = np.repeat(np.arange(600), k)
    return {
        "y_doc": y_doc, "s_doc": s_doc, "names": names, "k": k,
        "y": y_doc[doc_of_block], "s": s_doc[doc_of_block],
        "doc_ids": names[doc_of_block], "w": 1.0 / k[doc_of_block],
    }


def _fixture_fii():
    rng = np.random.default_rng(202)
    y_doc = np.r_[np.zeros(200), np.ones(200)].astype(int)
    u = rng.normal(0, np.sqrt(0.9), size=400)
    e = rng.normal(0, np.sqrt(0.1), size=1200)
    doc_of_block = np.repeat(np.arange(400), 3)
    y = y_doc[doc_of_block]
    names = np.array([f"d{i:04d}" for i in range(400)])
    return {
        "y": y, "s": u[doc_of_block] + e + 0.8 * y,
        "doc_ids": names[doc_of_block], "w": np.full(1200, 1.0 / 3.0),
    }


def _boot(cer, scores, y, doc_ids, w, n_resamples, seed=0):
    return cer.doc_bootstrap(
        scores, y, doc_ids, w, n_resamples=n_resamples, seed=seed, level=0.95,
        target_tpr=0.95, percentile_method="linear",
    )


def test_t4a_weighted_block_auroc_equals_document_auroc(cer):
    f = _fixture_fi()
    assert len(f["y"]) == 1182
    a_blk = cer.weighted_auroc(f["y"], f["s"], f["w"])
    a_doc = roc_auc_score(f["y_doc"], f["s_doc"])
    assert a_blk == pytest.approx(a_doc, abs=1e-12)
    assert a_blk == pytest.approx(0.729278, abs=1e-6)
    assert roc_auc_score(f["y"], f["s"]) == pytest.approx(0.731459, abs=1e-6)
    fpr_blk = cer.weighted_fpr_at_tpr(f["y"], f["s"], f["w"], 0.95)
    fpr_doc = cer.weighted_fpr_at_tpr(f["y_doc"], f["s_doc"], np.ones(600), 0.95)
    assert fpr_blk == pytest.approx(fpr_doc, abs=1e-12)
    assert fpr_blk == pytest.approx(0.723333, abs=1e-6)


def test_t4b_block_ci_equals_document_ci_and_concatenation_form(cer):
    f = _fixture_fi()
    blk = _boot(cer, {"s": f["s"]}, f["y"], f["doc_ids"], f["w"], 2000)
    doc = _boot(cer, {"s": f["s_doc"]}, f["y_doc"], f["names"], np.ones(600), 2000)
    ci_blk, ci_doc = blk["ci"]["s"]["auroc_w"], doc["ci"]["s"]["auroc_w"]
    np.testing.assert_allclose(ci_blk, ci_doc, atol=1e-9)
    np.testing.assert_allclose(ci_blk, [0.6881, 0.7667], atol=1e-4)
    assert blk["point"]["s"]["auroc_w"] == pytest.approx(0.729278, abs=1e-6)

    # Concatenation form: take every block of each drawn document, with multiplicity.
    rng = np.random.default_rng(0)
    id_docs = np.unique(f["doc_ids"][f["y"] == 0])
    ood_docs = np.unique(f["doc_ids"][f["y"] == 1])
    rows_of = {d: np.flatnonzero(f["doc_ids"] == d) for d in f["names"]}
    for r in range(100):
        di = rng.integers(0, len(id_docs), size=len(id_docs))
        do = rng.integers(0, len(ood_docs), size=len(ood_docs))
        rows = np.concatenate([rows_of[d] for d in np.r_[id_docs[di], ood_docs[do]]])
        a_cat = roc_auc_score(f["y"][rows], f["s"][rows], sample_weight=f["w"][rows])
        assert blk["resamples"]["s"]["auroc_w"][r] == pytest.approx(a_cat, abs=1e-9)


def test_t4c_same_seed_gives_identical_resamples(cer):
    f = _fixture_fi()
    a = _boot(cer, {"s": f["s"]}, f["y"], f["doc_ids"], f["w"], 200)
    b = _boot(cer, {"s": f["s"]}, f["y"], f["doc_ids"], f["w"], 200)
    assert np.array_equal(a["resamples"]["s"]["auroc_w"], b["resamples"]["s"]["auroc_w"])
    assert np.array_equal(a["resamples"]["s"]["fpr95_w"], b["resamples"]["s"]["fpr95_w"])


def test_t4d_paired_delta_identical_and_shifted_scores(cer):
    f = _fixture_fii()
    n_res = 2000
    same = _boot(cer, {"a": f["s"], "b": f["s"].copy()}, f["y"], f["doc_ids"], f["w"], n_res)
    d_same = cer.paired_delta(same, "a", "b", level=0.95, percentile_method="linear")
    assert d_same["delta"] == 0.0
    assert d_same["delta_ci"] == [0.0, 0.0]
    assert d_same["p"] == 1.0

    shifted = {"a": f["s"], "b": f["s"] - 0.8 * f["y"]}
    boot = _boot(cer, shifted, f["y"], f["doc_ids"], f["w"], n_res)
    d = cer.paired_delta(boot, "a", "b", level=0.95, percentile_method="linear")
    deltas = boot["resamples"]["a"]["auroc_w"] - boot["resamples"]["b"]["auroc_w"]
    assert (deltas > 0).all()
    assert deltas.min() == pytest.approx(0.185, abs=5e-4)
    assert d["p"] == pytest.approx(2.0 / (n_res + 1), abs=1e-12)
    assert d["p"] == pytest.approx(0.0009995, abs=1e-7)


def test_p_value_formula(cer):
    assert cer.bootstrap_p_value(np.array([0.1, 0.2, -0.1, 0.0])) == pytest.approx(
        min(1.0, 2 * (1 + 2) / 5)
    )
    assert cer.bootstrap_p_value(np.full(9, 0.3)) == pytest.approx(0.2)
    assert cer.bootstrap_p_value(np.zeros(9)) == 1.0


def test_t4f_holm(cer):
    np.testing.assert_allclose(cer.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06], atol=1e-12)
    np.testing.assert_allclose(cer.holm([0.5, 0.01, 0.01]), [0.5, 0.03, 0.03], atol=1e-12)
    np.testing.assert_allclose(cer.holm([0.9, 0.8]), [1.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Synthetic score files (section 5.5 format)
# ---------------------------------------------------------------------------


def _fixture_cfg():
    domains = [
        {"key": "stackexchange", "role": "id_base", "cache_tokens": 1000,
         "max_stream_tokens": 2000},
        {"key": "hackernews", "role": "id_adapter", "cache_tokens": 1000,
         "max_stream_tokens": 2000},
        {"key": "arxiv", "role": "ood", "cache_tokens": 100, "max_stream_tokens": 500,
         "stripped_copy": True},
        {"key": "freelaw", "role": "ood", "cache_tokens": 100, "max_stream_tokens": 500},
        {"key": "pubmed_abstracts", "role": "ood", "cache_tokens": 100,
         "max_stream_tokens": 500},
    ]
    return {
        "eval_set": {"name": "fixture", "domains": domains, "block_size": T_FIX},
        "scoring": {
            "seed_base": 0,
            "seed_offsets": {"legacy_d1": 0, "test": 100000, "test_hn": 200000,
                             "arxiv_stripped": 300000},
            "n_samples": dict(N_SAMPLES_FIX),
            "score_sets": {"legacy_d1": ["c0", "c1", "c3"], **copy.deepcopy(SCORE_SETS_FIX)},
            "out_dir": "unused_in_tests",
        },
        "analysis": {
            "primary_score": "blk_g",
            "contrast_score": "blk_mi",
            "secondary_scores": ["blk_tu", "blk_au", "blk_nll", "blk_maxprob_unc"],
            "aggregation": "mean_all_positions",
            "ood_domains": list(OOD_KEYS),
            "fpr_target_tpr": 0.95,
            "bootstrap": {"resamples": RESAMPLES_FIX, "seed": 0, "level": 0.95,
                          "unit": "document", "stratify_by_class": True,
                          "percentile_method": "linear"},
            "margin_auroc": 0.02,
            "alpha": 0.05,
            "families": {
                "F1_fix": [["c1", "stackexchange"], ["c3", "hackernews"]],
                "F2_fix": [["c3", "stackexchange"]],
                "F3_fix_s2": [["c2_refit", "stackexchange"]],
            },
            "descriptive": {
                "rows": [["c3", "stackexchange"]],
                "stripped_rows": [["c1", "stackexchange"], ["c3", "hackernews"]],
                "replication": {"id": "stackexchange", "ood": "arxiv",
                                "old_point": {"c1": 0.87},
                                "old_cluster_ci": {"c1": [0.80, 0.90]}},
            },
        },
        "checks": {"rederive_tol": TOL_FIX},
    }


def _analysis_hash(analysis):
    text = json.dumps(analysis, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _documents():
    rng = np.random.default_rng(7)
    docs = {}
    for key in DOMAIN_KEYS:
        rows = []
        for i in range(DOCS_PER_DOMAIN):
            sha = hashlib.sha1(f"{key}-{i}".encode()).hexdigest()[:12]
            rows.append((f"{key}/{i:09d}/{sha}", int(rng.integers(1, 4)),
                         int(rng.integers(0, 50))))
        docs[key] = rows
    return docs


def _blocks(docs, domains):
    out = []
    for key in domains:
        for doc_id, k, o_d in docs[key]:
            for j in range(k):
                out.append((doc_id, key, o_d + j * T_FIX, k))
    return out


EVAL_SET_DOMAINS = {
    "test": ["stackexchange", "arxiv", "freelaw", "pubmed_abstracts"],
    "test_hn": ["hackernews"],
    "arxiv_stripped": ["arxiv"],
}


def _write_score_file(path, cfg, score_set, eval_set, blocks, seed):
    rng = np.random.default_rng(seed)
    n = N_SAMPLES_FIX[score_set]
    b = len(blocks)
    is_ood = np.array([dom in OOD_KEYS for _, dom, _, _ in blocks], dtype=float)
    doc_effect = rng.uniform(0.5, 1.5, size=b)
    spread = (0.2 + 0.3 * is_ood) * doc_effect
    base = -2.5 + 0.5 * rng.normal(size=(b, 1, T_FIX))
    lp = base + spread[:, None, None] * rng.normal(size=(b, n, T_FIX))
    lp = np.minimum(lp, -1e-3).astype(np.float32)
    if n == 1:
        tok_mi = np.zeros((b, T_FIX), dtype=np.float32)
    else:
        tok_mi = (rng.gamma(2.0, 0.01, size=(b, T_FIX)) * (1 + is_ood[:, None]))
        tok_mi = tok_mi.astype(np.float32)
    tok_au = rng.uniform(1.0, 4.0, size=(b, T_FIX)).astype(np.float32)
    tok_tu = tok_au + tok_mi
    tok_maxprob = rng.uniform(0.05, 0.95, size=(b, T_FIX)).astype(np.float32)

    lp_t = torch.from_numpy(lp)
    log_pbar = torch.logsumexp(lp_t, dim=1) - math.log(n)
    g = log_pbar - lp_t.mean(dim=1)
    counts = {}
    for doc_id, _, _, _ in blocks:
        counts[doc_id] = counts.get(doc_id, 0) + 1
    payload = {
        "logp_real": lp_t,
        "tok_mi": torch.from_numpy(tok_mi),
        "tok_tu": torch.from_numpy(tok_tu),
        "tok_au": torch.from_numpy(tok_au),
        "tok_maxprob": torch.from_numpy(tok_maxprob),
        "tok_sum_p_sq": torch.from_numpy(tok_maxprob * 0.9),
        "tok_correct": torch.from_numpy(rng.random((b, T_FIX)) < 0.4),
        "blk_g": g.mean(dim=1),
        "blk_mi": torch.from_numpy(tok_mi).mean(dim=1),
        "blk_tu": torch.from_numpy(tok_tu).mean(dim=1),
        "blk_au": torch.from_numpy(tok_au).mean(dim=1),
        "blk_nll": (-log_pbar).mean(dim=1),
        "blk_maxprob_unc": (1.0 - torch.from_numpy(tok_maxprob)).mean(dim=1),
        "block_index": torch.arange(b, dtype=torch.int64),
        "doc_id": [d for d, _, _, _ in blocks],
        "domain": [dom for _, dom, _, _ in blocks],
        "offset": torch.tensor([o for _, _, o, _ in blocks], dtype=torch.int64),
        "weight": torch.tensor([1.0 / counts[d] for d, _, _, _ in blocks],
                               dtype=torch.float32),
        "meta": {
            "score_set": score_set, "eval_set": eval_set, "run_tag": "main",
            "n_samples": n, "analysis_sha256": _analysis_hash(cfg["analysis"]),
            "created_at": CREATED_AT_FIX,
        },
    }
    torch.save(payload, path)


def _build_scores(root):
    cfg = _fixture_cfg()
    cfg_path = root / "i2_eval.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    scores_dir = root / "scores"
    scores_dir.mkdir()
    docs = _documents()
    seed = 1000
    for eval_set, score_sets in SCORE_SETS_FIX.items():
        blocks = _blocks(docs, EVAL_SET_DOMAINS[eval_set])
        for score_set in score_sets:
            seed += 1
            path = scores_dir / f"{score_set}__{eval_set}__main.pt"
            _write_score_file(path, cfg, score_set, eval_set, blocks, seed)
    return cfg_path, scores_dir


def _rederive(cer, cfg_path, scores_dir):
    code = cer.main(["--rederive", "--config", str(cfg_path), "--scores-dir", str(scores_dir)])
    report = json.loads((scores_dir / "rederive_check.json").read_text(encoding="utf-8"))
    return code, report


TABLE_FILES = ["table_test.json", "table_test_hn.json", "table_arxiv_stripped.json",
               "families.json"]


@pytest.fixture(scope="module")
def golden(cer, tmp_path_factory):
    """Score files plus 'scorer' tables copied from a first rederive run."""
    root = tmp_path_factory.mktemp("golden")
    cfg_path, scores_dir = _build_scores(root)
    code, report = _rederive(cer, cfg_path, scores_dir)
    first = {"code": code, "report": report}
    for name in TABLE_FILES:
        shutil.copy(scores_dir / "rederive" / name, scores_dir / name)
    return {"root": root, "cfg_path": cfg_path, "scores_dir": scores_dir, "first": first}


@pytest.fixture
def work(golden, tmp_path):
    root = tmp_path / "work"
    shutil.copytree(golden["root"], root)
    return {"cfg_path": root / "i2_eval.yaml", "scores_dir": root / "scores"}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path, obj):
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _cell(table, score_set, id_domain, ood_domain):
    for c in table["cells"]:
        if (c["score_set"], c["id_domain"], c["ood_domain"]) == (score_set, id_domain,
                                                                  ood_domain):
            return c
    raise KeyError((score_set, id_domain, ood_domain))


def test_rederive_without_scorer_tables_fails_but_writes_tables(golden):
    first = golden["first"]
    assert first["code"] == 1
    assert first["report"]["status"] == "FAIL"
    assert any("table_test.json" in f for f in first["report"]["failures"])
    assert first["report"]["block_checks_pass"] is True


def test_rederived_tables_have_expected_cells(golden):
    d = golden["scores_dir"] / "rederive"
    test = _load(d / "table_test.json")
    cells = {(c["score_set"], c["id_domain"], c["ood_domain"]) for c in test["cells"]}
    assert cells == {(s, "stackexchange", o) for s in SCORE_SETS_FIX["test"] for o in OOD_KEYS}
    hn = _load(d / "table_test_hn.json")
    assert {(c["score_set"], c["id_domain"], c["ood_domain"]) for c in hn["cells"]} == {
        ("c3", "hackernews", o) for o in OOD_KEYS}
    stripped = _load(d / "table_arxiv_stripped.json")
    assert {(c["score_set"], c["id_domain"], c["ood_domain"]) for c in stripped["cells"]} == {
        ("c1", "stackexchange", "arxiv_stripped"), ("c3", "stackexchange", "arxiv_stripped"),
        ("c3", "hackernews", "arxiv_stripped")}
    for c in test["cells"] + hn["cells"] + stripped["cells"]:
        assert set(c["scores"]) == set(SCORES)
        for q in ("auroc_w", "fpr95_w"):
            assert 0.0 <= c["scores"]["blk_g"][q] <= 1.0
            lo, hi = c["scores"]["blk_g"][f"{q}_ci"]
            assert lo <= hi
    # C0 has N = 1, so G is 0 on every block: all ties give AUROC 0.5 and FPR@95 1.
    c0 = _cell(test, "c0", "stackexchange", "arxiv")
    assert c0["scores"]["blk_g"]["auroc_w"] == pytest.approx(0.5, abs=1e-12)
    assert c0["scores"]["blk_g"]["fpr95_w"] == pytest.approx(1.0, abs=1e-12)


def test_cell_blocks_and_rng_follow_section_5_6(cer, golden):
    """ID = HackerNews comes from test_hn, OOD from test; RNG = [seed, i_id, i_ood]."""
    sd = golden["scores_dir"]
    hn = torch.load(sd / "c3__test_hn__main.pt", weights_only=True)
    te = torch.load(sd / "c3__test__main.pt", weights_only=True)
    ood_rows = [i for i, dom in enumerate(te["domain"]) if dom == "freelaw"]
    y = np.r_[np.zeros(len(hn["domain"])), np.ones(len(ood_rows))].astype(int)
    doc_ids = np.array(list(hn["doc_id"]) + [te["doc_id"][i] for i in ood_rows])
    k = {d: int((doc_ids == d).sum()) for d in doc_ids}
    w = np.array([1.0 / k[d] for d in doc_ids])
    scores = {name: np.r_[hn[name].numpy(), te[name].numpy()[ood_rows]].astype(np.float64)
              for name in SCORES}
    ref = cer.doc_bootstrap(scores, y, doc_ids, w, n_resamples=RESAMPLES_FIX, seed=[0, 1, 3],
                            level=0.95, target_tpr=0.95, percentile_method="linear")
    table = _load(sd / "rederive" / "table_test_hn.json")
    cell = _cell(table, "c3", "hackernews", "freelaw")
    assert cell["n_id_blocks"] == len(hn["domain"])
    assert cell["n_ood_blocks"] == len(ood_rows)
    assert cell["n_id_docs"] == DOCS_PER_DOMAIN
    for name in SCORES:
        for q in ("auroc_w", "fpr95_w"):
            assert cell["scores"][name][q] == pytest.approx(ref["point"][name][q], abs=1e-12)
            np.testing.assert_allclose(cell["scores"][name][f"{q}_ci"], ref["ci"][name][q],
                                       atol=1e-12)


def test_families_holm_over_complete_families_only(golden):
    fam = _load(golden["scores_dir"] / "rederive" / "families.json")["families"]
    f1 = fam["F1_fix"]
    assert f1["status"] == "scored"
    assert len(f1["cells"]) == 2 * len(OOD_KEYS)
    for c in f1["cells"]:
        assert c["p_holm"] >= c["p"] - 1e-15
        assert c["delta"] == pytest.approx(c["auroc_w_primary"] - c["auroc_w_contrast"],
                                           abs=1e-12)
    assert fam["F2_fix"]["status"] == "scored"
    assert len(fam["F2_fix"]["cells"]) == len(OOD_KEYS)
    assert fam["F3_fix_s2"]["status"] == "not_scored"
    assert fam["F3_fix_s2"]["cells"] == []


def test_rederive_passes_on_matching_tables(cer, work):
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert report["failures"] == []
    assert report["status"] == "PASS"
    assert code == 0
    assert report["n_values_compared"] > 0
    per_cell = report["per_cell_pass"]
    assert set(per_cell) == set(TABLE_FILES)
    assert len(per_cell["table_test.json"]) == len(SCORE_SETS_FIX["test"]) * len(OOD_KEYS)
    assert len(per_cell["families.json"]) == 3 * len(OOD_KEYS)  # F1 (2 rows) + F2 (1 row)
    assert all(ok for cells in per_cell.values() for ok in cells.values())


def test_rederive_tolerates_tiny_difference(cer, work):
    path = work["scores_dir"] / "table_test.json"
    table = _load(path)
    _cell(table, "c1", "stackexchange", "arxiv")["scores"]["blk_g"]["auroc_w"] += 5e-7
    _dump(path, table)
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert code == 0, report["failures"]


def _named(failures, *parts):
    return any(all(part in f for part in parts) for f in failures)


def test_rederive_flags_perturbed_table_values(cer, work):
    path = work["scores_dir"] / "table_test.json"
    table = _load(path)
    edits = [  # (cell, score, quantity, CI index or None)
        (("c1", "stackexchange", "freelaw"), "blk_mi", "auroc_w", None),
        (("c0", "stackexchange", "arxiv"), "blk_tu", "auroc_w_ci", 1),
        (("c3", "stackexchange", "pubmed_abstracts"), "blk_g", "fpr95_w", None),
        (("c1", "stackexchange", "arxiv"), "blk_nll", "fpr95_w_ci", 0),
    ]
    for cell, score, quantity, idx in edits:
        entry = _cell(table, *cell)["scores"][score]
        if idx is None:
            entry[quantity] += 1e-4
        else:
            entry[quantity][idx] += 1e-4
    _dump(path, table)
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert code == 1
    assert report["status"] == "FAIL"
    for cell, score, quantity, _ in edits:
        assert _named(report["failures"], "|".join(cell), f" {score} {quantity}:"), cell
    assert len(report["failures"]) == len(edits)
    per_cell = report["per_cell_pass"]["table_test.json"]
    assert {tag for tag, ok in per_cell.items() if not ok} == {"|".join(e[0]) for e in edits}


def test_rederive_flags_missing_cell(cer, work):
    path = work["scores_dir"] / "table_test_hn.json"
    table = _load(path)
    table["cells"] = [c for c in table["cells"] if c["ood_domain"] != "pubmed_abstracts"]
    _dump(path, table)
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert code == 1
    assert any("c3|hackernews|pubmed_abstracts" in f for f in report["failures"])


def test_rederive_flags_perturbed_family_values(cer, work):
    path = work["scores_dir"] / "families.json"
    fam = _load(path)
    cells = fam["families"]["F1_fix"]["cells"]
    quantities = ["delta", "delta_ci", "p", "p_holm"]
    for cell, quantity in zip(cells, quantities):
        if quantity == "delta_ci":
            cell[quantity][0] -= 1e-3
        else:
            cell[quantity] += 1e-3
    _dump(path, fam)
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert code == 1
    for cell, quantity in zip(cells, quantities):
        tag = "|".join((cell["score_set"], cell["id_domain"], cell["ood_domain"]))
        assert _named(report["failures"], "F1_fix", tag, f" {quantity}:"), quantity
    assert len(report["failures"]) == len(quantities)


def test_rederive_flags_missing_family(cer, work):
    path = work["scores_dir"] / "families.json"
    fam = _load(path)
    del fam["families"]["F2_fix"]
    _dump(path, fam)
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert code == 1
    assert any("F2_fix" in f for f in report["failures"])


def test_rederive_flags_wrong_block_g(cer, work):
    path = work["scores_dir"] / "c1__test__main.pt"
    sf = torch.load(path, weights_only=True)
    sf["blk_g"][3] += 1e-3
    torch.save(sf, path)
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert code == 1
    assert report["block_checks_pass"] is False
    assert any("c1__test__main.pt" in f and "blk_g" in f for f in report["failures"])


def test_rederive_reads_meta_with_non_tensor_objects(cer, work):
    """meta.torch_version = torch.__version__ is a TorchVersion, which weights_only refuses."""
    path = work["scores_dir"] / "c1__test__main.pt"
    sf = torch.load(path, weights_only=True)
    sf["meta"]["torch_version"] = torch.__version__
    sf["meta"]["seed_base"] = np.int64(0)
    torch.save(sf, path)
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert code == 0, report["failures"]
    result = next(r for r in report["block_checks"] if r["file"] == path.name)
    assert any("weights_only=False" in w for w in result["format_warnings"])


def test_rederive_flags_wrong_block_mi_and_weight(cer, work):
    path = work["scores_dir"] / "c3__test_hn__main.pt"
    sf = torch.load(path, weights_only=True)
    sf["blk_mi"][0] += 1e-4
    sf["weight"][1] = 1.0 if sf["weight"][1] < 1.0 else 0.5
    torch.save(sf, path)
    code, report = _rederive(cer, work["cfg_path"], work["scores_dir"])
    assert code == 1
    joined = " ".join(report["failures"])
    assert "blk_mi" in joined and "weight" in joined


# ---------------------------------------------------------------------------
# --freeze, --check prereg, --check align
# ---------------------------------------------------------------------------


def test_freeze_writes_hash_and_refuses_overwrite(cer, work):
    sd = work["scores_dir"]
    args = ["--freeze", "--config", str(work["cfg_path"]), "--scores-dir", str(sd)]
    assert cer.main(args) == 0
    prereg = _load(sd / "prereg.json")
    assert prereg["analysis_sha256"] == _analysis_hash(_fixture_cfg()["analysis"])
    assert "+00:00" in prereg["frozen_at"] or prereg["frozen_at"].endswith("Z")
    before = (sd / "prereg.json").read_bytes()
    assert cer.main(args) != 0
    assert (sd / "prereg.json").read_bytes() == before


def _write_prereg(sd, sha, frozen_at=FROZEN_AT_FIX):
    _dump(sd / "prereg.json", {"analysis_sha256": sha, "frozen_at": frozen_at})


def _prereg(cer, work):
    args = ["--check", "prereg", "--config", str(work["cfg_path"]),
            "--scores-dir", str(work["scores_dir"])]
    code = cer.main(args)
    return code, _load(work["scores_dir"] / "prereg_check.json")


def test_check_prereg_passes(cer, work):
    _write_prereg(work["scores_dir"], _analysis_hash(_fixture_cfg()["analysis"]))
    code, report = _prereg(cer, work)
    assert report["status"] == "PASS", report["failures"]
    assert code == 0
    assert report["n_test_score_files"] == sum(len(v) for v in SCORE_SETS_FIX.values())


def test_check_prereg_fails_on_missing_prereg(cer, work):
    code, report = _prereg(cer, work)
    assert code == 1 and report["status"] == "FAIL"


def test_check_prereg_fails_on_changed_analysis(cer, work):
    _write_prereg(work["scores_dir"], _analysis_hash(_fixture_cfg()["analysis"]))
    cfg = yaml.safe_load(work["cfg_path"].read_text(encoding="utf-8"))
    cfg["analysis"]["margin_auroc"] = 0.03
    work["cfg_path"].write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    code, report = _prereg(cer, work)
    assert code == 1
    assert any("prereg.json" in f for f in report["failures"])


def test_check_prereg_fails_on_score_file_hash_or_time(cer, work):
    sd = work["scores_dir"]
    _write_prereg(sd, _analysis_hash(_fixture_cfg()["analysis"]))
    p1 = sd / "c1__test__main.pt"
    sf = torch.load(p1, weights_only=True)
    sf["meta"]["analysis_sha256"] = "0" * 64
    torch.save(sf, p1)
    p2 = sd / "c3__test_hn__main.pt"
    sf = torch.load(p2, weights_only=True)
    sf["meta"]["created_at"] = "2026-09-26T09:00:00+00:00"
    torch.save(sf, p2)
    code, report = _prereg(cer, work)
    assert code == 1
    joined = " ".join(report["failures"])
    assert "c1__test__main.pt" in joined and "c3__test_hn__main.pt" in joined


def _align(cer, work):
    args = ["--check", "align", "--config", str(work["cfg_path"]),
            "--scores-dir", str(work["scores_dir"])]
    code = cer.main(args)
    return code, _load(work["scores_dir"] / "align_check.json")


def test_check_align_passes_with_subset_rerun(cer, work):
    sd = work["scores_dir"]
    sf = torch.load(sd / "c1__test__main.pt", weights_only=True)
    rows = [0, 1, 2, 5]
    sub = {key: (val[rows] if isinstance(val, torch.Tensor) else
                 [val[i] for i in rows] if isinstance(val, list) else val)
           for key, val in sf.items()}
    torch.save(sub, sd / "c1__test__rerun.pt")
    code, report = _align(cer, work)
    assert report["status"] == "PASS", report["failures"]
    assert code == 0


def test_check_align_flags_mismatch(cer, work):
    sd = work["scores_dir"]
    path = sd / "c3__test__main.pt"
    sf = torch.load(path, weights_only=True)
    sf["doc_id"][4] = "stackexchange/999999999/000000000000"
    sf["offset"][6] += 1
    torch.save(sf, path)
    code, report = _align(cer, work)
    assert code == 1
    joined = " ".join(report["failures"])
    assert "c3__test__main.pt" in joined and "doc_id" in joined and "offset" in joined


def test_missing_scores_dir_is_not_created(cer, work, tmp_path):
    missing = tmp_path / "no_such_dir"
    for mode in (["--rederive"], ["--check", "prereg"], ["--check", "align"]):
        args = [*mode, "--config", str(work["cfg_path"]), "--scores-dir", str(missing)]
        assert cer.main(args) == 2
    assert not missing.exists()


def test_unbuilt_checks_exit_with_code_2(cer, work):
    for check in ("repro", "stream", "manifest"):
        args = ["--check", check, "--config", str(work["cfg_path"]),
                "--scores-dir", str(work["scores_dir"])]
        assert cer.main(args) == 2


# ---------------------------------------------------------------------------
# T5b import check
# ---------------------------------------------------------------------------


def test_checker_imports_no_scorer_code():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    top = {n.split(".")[0] for n in names}
    assert not any(n.startswith("minigpt") for n in names)
    assert not top & {"eval_c_checkpoints", "eval_mc_dropout", "scripts", "experiments"}
    third_party = top - set(sys.stdlib_module_names) - {"__future__"}
    assert third_party <= {"numpy", "torch", "yaml", "sklearn"}
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "import_module" not in source and "__import__" not in source
