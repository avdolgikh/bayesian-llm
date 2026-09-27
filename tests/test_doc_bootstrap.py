"""S1-T4: weighted AUROC / FPR@95, document-clustered bootstrap, paired contrast, Holm.

Spec: specs/i2-eval-rebuild.md, sections 4 (S1-T4 checks T4a-T4f) and 5.6.
The fixtures F-i and F-ii are copied from the spec's section 5.6 table, so the
session-02 probe reference values reproduce. Every constant lives in this module.
Also covers the realized-token Jensen-gap helper named in section 5.5.
"""

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from minigpt.uncertainty import (
    auroc,
    bootstrap_ci,
    doc_bootstrap_auroc,
    fpr_at_tpr,
    holm,
    paired_doc_bootstrap,
    realized_token_gap,
)

# ---------------------------------------------------------------------------
# Fixture constants (spec section 5.6, "Fixtures for S1-T4")
# ---------------------------------------------------------------------------

F1_SEED = 101
F1_N_ID = 300
F1_N_OOD = 300
F1_N_BLOCKS = 1182
F2_SEED = 202
F2_N_ID = 200
F2_N_OOD = 200
F2_K = 3
F2_VAR_DOC = 0.9
F2_VAR_BLOCK = 0.1
F2_SHIFT = 0.8
N_RESAMPLES = 2000
BOOT_SEED = 0
LEVEL = 0.95
TARGET_TPR = 0.95

# Session-02 probe reference values (spec section 4, T4a-T4e; section 6.2)
REF_F1_AUROC_W = 0.729278
REF_F1_AUROC_UNWEIGHTED = 0.731459
REF_F1_FPR95 = 0.723333
REF_F1_CI = (0.6881, 0.7667)
REF_F2_MIN_WIDTH_RATIO = 1.4
REF_F2_WIDTH_RATIO = 1.587
REF_F2_MIN_DELTA = 0.185


def _fixture_i() -> dict:
    """F-i: 300 ID + 300 OOD documents, 1-3 blocks each, blocks share the doc score."""
    rng = np.random.default_rng(F1_SEED)
    n_docs = F1_N_ID + F1_N_OOD
    y_doc = np.r_[np.zeros(F1_N_ID, dtype=np.int64), np.ones(F1_N_OOD, dtype=np.int64)]
    names = np.array([f"d{i:04d}" for i in range(n_docs)])
    k = rng.integers(1, 4, size=n_docs)
    s_doc = rng.normal(loc=1.0 * y_doc, scale=1.0)
    doc_of_block = np.repeat(np.arange(n_docs), k)
    return {
        "s_doc": s_doc,
        "y_doc": y_doc,
        "names": names,
        "k": k,
        "doc_of_block": doc_of_block,
        "s": s_doc[doc_of_block],
        "y": y_doc[doc_of_block],
        "doc": names[doc_of_block],
        "w": 1.0 / k[doc_of_block],
    }


def _fixture_ii() -> dict:
    """F-ii: 200 ID + 200 OOD documents, 3 blocks each, within-doc correlation 0.9."""
    rng = np.random.default_rng(F2_SEED)
    n_docs = F2_N_ID + F2_N_OOD
    y_doc = np.r_[np.zeros(F2_N_ID, dtype=np.int64), np.ones(F2_N_OOD, dtype=np.int64)]
    names = np.array([f"d{i:04d}" for i in range(n_docs)])
    doc_of_block = np.repeat(np.arange(n_docs), F2_K)
    u = rng.normal(0.0, math.sqrt(F2_VAR_DOC), size=n_docs)
    e = rng.normal(0.0, math.sqrt(F2_VAR_BLOCK), size=n_docs * F2_K)
    y = y_doc[doc_of_block]
    return {
        "doc_of_block": doc_of_block,
        "s": u[doc_of_block] + e + F2_SHIFT * y,
        "y": y,
        "doc": names[doc_of_block],
        "w": np.full(n_docs * F2_K, 1.0 / F2_K),
    }


@pytest.fixture(scope="module")
def f1() -> dict:
    return _fixture_i()


@pytest.fixture(scope="module")
def f2() -> dict:
    return _fixture_ii()


def _boot(fx: dict, scores: dict | None = None, seed=BOOT_SEED) -> dict:
    return doc_bootstrap_auroc(
        scores if scores is not None else {"s": fx["s"]},
        fx["y"], fx["doc"], fx["w"],
        n_resamples=N_RESAMPLES, seed=seed, level=LEVEL, target_tpr=TARGET_TPR,
    )


def _reference_draws(fx: dict, seed: int, n_resamples: int):
    """Independent re-derivation of the draw rule of section 5.6.

    Yields (count per block, list of drawn document indices) per resample.
    Document order is numpy's lexicographic order of the unique IDs per class.
    """
    doc = fx["doc"]
    y = fx["y"]
    id_docs = np.unique(doc[y == 0])
    ood_docs = np.unique(doc[y == 1])
    all_docs = np.r_[id_docs, ood_docs]
    doc_pos = {d: i for i, d in enumerate(all_docs)}
    block_doc = np.array([doc_pos[d] for d in doc])
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        di = rng.integers(0, len(id_docs), size=len(id_docs))
        do = rng.integers(0, len(ood_docs), size=len(ood_docs))
        drawn = np.r_[di, len(id_docs) + do]
        counts = np.bincount(drawn, minlength=len(all_docs))
        yield counts[block_doc], drawn, block_doc


def _sk_fpr95(y, s, w) -> float:
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
    return float(fpr[np.argmax(tpr >= TARGET_TPR)])


# ---------------------------------------------------------------------------
# Fixture fidelity
# ---------------------------------------------------------------------------

class TestFixtures:
    def test_fixture_i_block_count(self, f1):
        """Spec: F-i has 1,182 blocks."""
        assert len(f1["s"]) == F1_N_BLOCKS

    def test_fixture_ii_block_count(self, f2):
        assert len(f2["s"]) == (F2_N_ID + F2_N_OOD) * F2_K


# ---------------------------------------------------------------------------
# T4a: weighted AUROC / FPR@95 equal one-row-per-document values
# ---------------------------------------------------------------------------

class TestT4aWeightedPoint:
    def test_weighted_auroc_equals_document_auroc(self, f1):
        a_blk = auroc(f1["s"], f1["y"], sample_weight=f1["w"])
        a_doc = auroc(f1["s_doc"], f1["y_doc"])
        assert abs(a_blk - a_doc) <= 1e-12
        assert abs(a_blk - REF_F1_AUROC_W) < 5e-7

    def test_unweighted_block_auroc_differs(self, f1):
        """Unweighted block AUROC is the probe's 0.731459, not the document value."""
        a_unw = auroc(f1["s"], f1["y"])
        assert abs(a_unw - REF_F1_AUROC_UNWEIGHTED) < 5e-7

    def test_weighted_fpr95_equals_document_fpr95(self, f1):
        f_blk = fpr_at_tpr(f1["s"], f1["y"], TARGET_TPR, sample_weight=f1["w"])
        f_doc_rule = fpr_at_tpr(
            f1["s_doc"], f1["y_doc"], TARGET_TPR, sample_weight=np.ones(len(f1["s_doc"])),
        )
        f_doc_old = fpr_at_tpr(f1["s_doc"], f1["y_doc"], TARGET_TPR)
        assert abs(f_blk - f_doc_rule) <= 1e-12
        assert abs(f_blk - f_doc_old) <= 1e-12
        assert abs(f_blk - REF_F1_FPR95) < 5e-7

    def test_weighted_fpr95_matches_spec_code(self, f1):
        """Spec 5.6: roc_curve(drop_intermediate=False), then fpr[argmax(tpr >= 0.95)]."""
        got = fpr_at_tpr(f1["s"], f1["y"], TARGET_TPR, sample_weight=f1["w"])
        assert got == _sk_fpr95(f1["y"], f1["s"], f1["w"])

    def test_sample_weight_none_keeps_old_values(self, f1):
        """With sample_weight=None both functions return today's values."""
        s, y = f1["s"], f1["y"]
        assert auroc(s, y, sample_weight=None) == float(roc_auc_score(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        idx = np.searchsorted(tpr, TARGET_TPR)
        old = float(fpr[-1]) if idx >= len(fpr) else float(fpr[idx])
        assert fpr_at_tpr(s, y, TARGET_TPR, sample_weight=None) == old
        assert fpr_at_tpr(s, y, TARGET_TPR) == old

    def test_bootstrap_point_uses_original_weights(self, f1):
        res = _boot(f1)["s"]
        assert res["auroc"] == auroc(f1["s"], f1["y"], sample_weight=f1["w"])
        assert res["fpr95"] == fpr_at_tpr(f1["s"], f1["y"], TARGET_TPR, sample_weight=f1["w"])
        assert res["n_id_docs"] == F1_N_ID
        assert res["n_ood_docs"] == F1_N_OOD


# ---------------------------------------------------------------------------
# T4b: block-level CI equals document-level CI; count form equals concatenation
# ---------------------------------------------------------------------------

class TestT4bDocumentCI:
    def test_block_ci_equals_document_ci(self, f1):
        blk = _boot(f1)["s"]
        doc_fx = {
            "s": f1["s_doc"], "y": f1["y_doc"], "doc": f1["names"],
            "w": np.ones(len(f1["s_doc"])),
        }
        doc = _boot(doc_fx)["s"]
        np.testing.assert_allclose(blk["auroc_ci"], doc["auroc_ci"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            blk["auroc_resamples"], doc["auroc_resamples"], rtol=0, atol=1e-9,
        )
        np.testing.assert_allclose(blk["auroc_ci"], REF_F1_CI, rtol=0, atol=5e-5)

    def test_ci_is_linear_percentile(self, f1):
        res = _boot(f1)["s"]
        lo, hi = np.percentile(res["auroc_resamples"], [2.5, 97.5])
        assert res["auroc_ci"] == (float(lo), float(hi))
        lo, hi = np.percentile(res["fpr95_resamples"], [2.5, 97.5])
        assert res["fpr95_ci"] == (float(lo), float(hi))

    def test_count_form_equals_concatenation_form(self, f1):
        """w_b * c_d(b) equals taking all blocks of each drawn document with multiplicity."""
        res = _boot(f1)["s"]
        blocks_of = [np.flatnonzero(f1["doc_of_block"] == d) for d in range(len(f1["names"]))]
        concat_vals = []
        for counts, drawn, _ in _reference_draws(f1, BOOT_SEED, N_RESAMPLES):
            idx = np.concatenate([blocks_of[d] for d in drawn])
            concat_vals.append(
                roc_auc_score(f1["y"][idx], f1["s"][idx], sample_weight=f1["w"][idx]),
            )
        concat_vals = np.asarray(concat_vals)
        np.testing.assert_allclose(res["auroc_resamples"], concat_vals, rtol=0, atol=1e-9)
        ci_concat = np.percentile(concat_vals, [2.5, 97.5])
        np.testing.assert_allclose(res["auroc_ci"], ci_concat, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Fast path equals sklearn per resample (spec 5.6 "Speed (optional)")
# ---------------------------------------------------------------------------

class TestFastPathMatchesSklearn:
    @pytest.mark.parametrize("name", ["f1", "f2"])
    def test_resamples_match_sklearn(self, name, request):
        fx = request.getfixturevalue(name)
        res = _boot(fx)["s"]
        sk_auc, sk_fpr = [], []
        for counts, _, _ in _reference_draws(fx, BOOT_SEED, N_RESAMPLES):
            w_r = fx["w"] * counts
            sk_auc.append(roc_auc_score(fx["y"], fx["s"], sample_weight=w_r))
            sk_fpr.append(_sk_fpr95(fx["y"], fx["s"], w_r))
        np.testing.assert_allclose(res["auroc_resamples"], sk_auc, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res["fpr95_resamples"], sk_fpr, rtol=0, atol=1e-12)

    def test_ties_across_classes(self):
        """Tied scores across ID and OOD count one half, as in roc_auc_score."""
        s = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 0.5, 0.2, 0.9])
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        doc = np.array(["a", "a", "b", "c", "c", "d", "e", "f"])
        w = np.array([0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0])
        fx = {"s": s, "y": y, "doc": doc, "w": w}
        res = doc_bootstrap_auroc(
            {"s": s}, y, doc, w, n_resamples=300, seed=7, level=LEVEL, target_tpr=TARGET_TPR,
        )["s"]
        sk_auc, sk_fpr = [], []
        for counts, _, _ in _reference_draws(fx, 7, 300):
            w_r = w * counts
            sk_auc.append(roc_auc_score(y, s, sample_weight=w_r))
            sk_fpr.append(_sk_fpr95(y, s, w_r))
        np.testing.assert_allclose(res["auroc_resamples"], sk_auc, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res["fpr95_resamples"], sk_fpr, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# T4c: determinism
# ---------------------------------------------------------------------------

class TestT4cDeterminism:
    def test_same_seed_identical_arrays(self, f1):
        r1 = _boot(f1)["s"]
        r2 = _boot(f1)["s"]
        assert np.array_equal(r1["auroc_resamples"], r2["auroc_resamples"])
        assert np.array_equal(r1["fpr95_resamples"], r2["fpr95_resamples"])

    def test_paired_same_seed_identical_arrays(self, f2):
        scores = {"a": f2["s"], "b": f2["s"] - F2_SHIFT * f2["y"]}
        kw = dict(n_resamples=N_RESAMPLES, seed=BOOT_SEED, level=LEVEL)
        p1 = paired_doc_bootstrap(scores, f2["y"], f2["doc"], f2["w"], [("a", "b")], **kw)
        p2 = paired_doc_bootstrap(scores, f2["y"], f2["doc"], f2["w"], [("a", "b")], **kw)
        assert np.array_equal(p1[("a", "b")]["resamples"], p2[("a", "b")]["resamples"])

    def test_other_seed_differs(self, f1):
        r0 = _boot(f1)["s"]
        r1 = _boot(f1, seed=1)["s"]
        assert not np.array_equal(r0["auroc_resamples"], r1["auroc_resamples"])

    def test_sequence_seed_matches_default_rng(self, f1):
        """seed = [seed, i_id, i_ood] feeds np.random.default_rng directly."""
        seq = [BOOT_SEED, 0, 2]
        res = doc_bootstrap_auroc(
            {"s": f1["s"]}, f1["y"], f1["doc"], f1["w"],
            n_resamples=50, seed=seq, level=LEVEL, target_tpr=TARGET_TPR,
        )["s"]
        ref = [
            roc_auc_score(f1["y"], f1["s"], sample_weight=f1["w"] * c)
            for c, _, _ in _reference_draws(f1, seq, 50)
        ]
        np.testing.assert_allclose(res["auroc_resamples"], ref, rtol=0, atol=1e-12)

    def test_all_scores_share_draws(self, f1):
        """Two scores in one call use the same resamples: equal inputs give equal arrays."""
        res = _boot(f1, scores={"x": f1["s"], "z": f1["s"].copy()})
        assert np.array_equal(res["x"]["auroc_resamples"], res["z"]["auroc_resamples"])
        solo = _boot(f1)["s"]
        assert np.array_equal(res["x"]["auroc_resamples"], solo["auroc_resamples"])


# ---------------------------------------------------------------------------
# T4d: paired contrast
# ---------------------------------------------------------------------------

class TestT4dPaired:
    def test_identical_scores_zero_delta(self, f1):
        scores = {"a": f1["s"], "b": f1["s"].copy()}
        res = paired_doc_bootstrap(
            scores, f1["y"], f1["doc"], f1["w"], [("a", "b")],
            n_resamples=N_RESAMPLES, seed=BOOT_SEED, level=LEVEL,
        )[("a", "b")]
        assert res["delta"] == 0.0
        assert res["ci"] == (0.0, 0.0)
        assert res["p"] == 1.0
        assert np.all(res["resamples"] == 0.0)

    def test_shift_gives_floor_p(self, f2):
        scores = {"a": f2["s"], "b": f2["s"] - F2_SHIFT * f2["y"]}
        res = paired_doc_bootstrap(
            scores, f2["y"], f2["doc"], f2["w"], [("a", "b")],
            n_resamples=N_RESAMPLES, seed=BOOT_SEED, level=LEVEL,
        )[("a", "b")]
        assert np.all(res["resamples"] > 0.0)
        assert abs(res["resamples"].min() - REF_F2_MIN_DELTA) < 5e-4
        assert abs(res["p"] - 2.0 / (N_RESAMPLES + 1)) <= 1e-12
        a_w = auroc(f2["s"], f2["y"], sample_weight=f2["w"])
        b_w = auroc(scores["b"], f2["y"], sample_weight=f2["w"])
        assert abs(res["delta"] - (a_w - b_w)) <= 1e-15
        lo, hi = np.percentile(res["resamples"], [2.5, 97.5])
        assert res["ci"] == (float(lo), float(hi))

    def test_paired_resamples_are_differences_of_marginals(self, f2):
        scores = {"a": f2["s"], "b": f2["s"] - F2_SHIFT * f2["y"]}
        marg = _boot(f2, scores=scores)
        res = paired_doc_bootstrap(
            scores, f2["y"], f2["doc"], f2["w"], [("a", "b"), ("b", "a")],
            n_resamples=N_RESAMPLES, seed=BOOT_SEED, level=LEVEL,
        )
        diff = marg["a"]["auroc_resamples"] - marg["b"]["auroc_resamples"]
        assert np.array_equal(res[("a", "b")]["resamples"], diff)
        assert np.array_equal(res[("b", "a")]["resamples"], -diff)
        assert res[("b", "a")]["p"] == res[("a", "b")]["p"]

    def test_p_value_formula(self, f2):
        """p = min(1, 2 (1 + min(#{d<=0}, #{d>=0})) / (B + 1)) on a mixed-sign contrast."""
        rng = np.random.default_rng(5)
        scores = {"a": f2["s"], "b": f2["s"] + rng.normal(0.0, 0.05, size=len(f2["s"]))}
        res = paired_doc_bootstrap(
            scores, f2["y"], f2["doc"], f2["w"], [("a", "b")],
            n_resamples=N_RESAMPLES, seed=BOOT_SEED, level=LEVEL,
        )[("a", "b")]
        d = res["resamples"]
        n_le, n_ge = int((d <= 0).sum()), int((d >= 0).sum())
        assert n_le > 0 and n_ge > 0
        expected = min(1.0, 2.0 * (1 + min(n_le, n_ge)) / (N_RESAMPLES + 1))
        assert res["p"] == expected


# ---------------------------------------------------------------------------
# T4e: design effect
# ---------------------------------------------------------------------------

class TestT4eDesignEffect:
    def test_document_ci_wider_than_iid(self, f2):
        doc = _boot(f2)["s"]
        _, lo, hi = bootstrap_ci(f2["s"], f2["y"], auroc, n_bootstrap=N_RESAMPLES, seed=BOOT_SEED)
        ratio = (doc["auroc_ci"][1] - doc["auroc_ci"][0]) / (hi - lo)
        assert ratio >= REF_F2_MIN_WIDTH_RATIO
        assert abs(ratio - REF_F2_WIDTH_RATIO) < 5e-3


# ---------------------------------------------------------------------------
# T4f: Holm
# ---------------------------------------------------------------------------

class TestT4fHolm:
    def test_spec_example(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06],
                                   rtol=0, atol=1e-12)

    def test_caps_at_one_and_monotone(self):
        p = [0.2, 0.5, 0.01, 0.6]
        adj = holm(p)
        # sorted: 0.01*4=0.04, 0.2*3=0.6, 0.5*2=1.0, 0.6*1=0.6 -> cummax 0.04, 0.6, 1, 1
        np.testing.assert_allclose(adj, [0.6, 1.0, 0.04, 1.0], rtol=0, atol=1e-12)
        assert np.all(adj <= 1.0)

    def test_ties_stable(self):
        np.testing.assert_allclose(holm([0.02, 0.02, 0.5]), [0.06, 0.06, 0.5],
                                   rtol=0, atol=1e-12)

    def test_single_and_empty(self):
        np.testing.assert_allclose(holm([0.3]), [0.3], rtol=0, atol=1e-12)
        assert holm([]).shape == (0,)

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            holm([0.1, float("nan")])
        with pytest.raises(ValueError):
            holm([0.1, 1.5])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _args(self):
        s = np.array([0.1, 0.2, 0.8, 0.9])
        y = np.array([0, 0, 1, 1])
        doc = np.array(["a", "b", "c", "d"])
        w = np.ones(4)
        return s, y, doc, w

    def _call(self, s, y, doc, w):
        return doc_bootstrap_auroc(
            {"s": s}, y, doc, w, n_resamples=10, seed=0, level=LEVEL, target_tpr=TARGET_TPR,
        )

    def test_doc_in_both_classes(self):
        s, y, doc, w = self._args()
        doc[2] = "a"
        with pytest.raises(ValueError):
            self._call(s, y, doc, w)

    def test_non_finite_score(self):
        s, y, doc, w = self._args()
        s[0] = np.nan
        with pytest.raises(ValueError):
            self._call(s, y, doc, w)

    def test_length_mismatch(self):
        s, y, doc, w = self._args()
        with pytest.raises(ValueError):
            self._call(s[:3], y, doc, w)

    def test_single_class(self):
        s, y, doc, w = self._args()
        with pytest.raises(ValueError):
            self._call(s, np.zeros(4, dtype=int), doc, w)

    def test_unknown_pair(self):
        s, y, doc, w = self._args()
        with pytest.raises(KeyError):
            paired_doc_bootstrap({"s": s}, y, doc, w, [("s", "x")],
                                 n_resamples=10, seed=0, level=LEVEL)

    def test_torch_inputs_accepted(self, f1):
        res_np = doc_bootstrap_auroc(
            {"s": f1["s"]}, f1["y"], f1["doc"], f1["w"],
            n_resamples=20, seed=0, level=LEVEL, target_tpr=TARGET_TPR,
        )["s"]
        res_t = doc_bootstrap_auroc(
            {"s": torch.from_numpy(f1["s"])}, torch.from_numpy(f1["y"]), list(f1["doc"]),
            torch.from_numpy(f1["w"]), n_resamples=20, seed=0, level=LEVEL, target_tpr=TARGET_TPR,
        )["s"]
        assert np.array_equal(res_np["auroc_resamples"], res_t["auroc_resamples"])


# ---------------------------------------------------------------------------
# Realized-token Jensen gap (spec sections 3 and 5.5)
# ---------------------------------------------------------------------------

class TestRealizedTokenGap:
    def test_matches_formula(self):
        gen = torch.Generator().manual_seed(0)
        logp = torch.log_softmax(torch.randn(4, 5, 7, 11, generator=gen), -1)[..., 0]
        logp = logp.float()  # [B=4, N=5, T=7]
        out = realized_token_gap(logp)
        lp = logp.double()
        log_pbar = torch.logsumexp(lp, dim=1) - math.log(5)
        g = log_pbar - lp.mean(dim=1)
        torch.testing.assert_close(out["log_pbar"].double(), log_pbar, rtol=0, atol=1e-6)
        torch.testing.assert_close(out["g"].double(), g, rtol=0, atol=1e-6)
        torch.testing.assert_close(out["G"].double(), g.mean(dim=-1), rtol=0, atol=1e-6)
        assert out["log_pbar"].shape == (4, 7)
        assert out["g"].shape == (4, 7)
        assert out["G"].shape == (4,)
        assert out["G"].dtype == torch.float32

    def test_gap_non_negative(self):
        gen = torch.Generator().manual_seed(1)
        logp = -20.0 * torch.rand(3, 20, 256, generator=gen)
        out = realized_token_gap(logp)
        assert out["g"].min().item() >= -1e-6

    def test_single_sample_is_zero(self):
        gen = torch.Generator().manual_seed(2)
        logp = -10.0 * torch.rand(2, 1, 32, generator=gen)
        out = realized_token_gap(logp)
        assert out["g"].abs().max().item() <= 1e-6
        assert out["G"].abs().max().item() <= 1e-6
        torch.testing.assert_close(out["log_pbar"], logp[:, 0], rtol=0, atol=1e-6)

    def test_identical_samples_is_zero(self):
        gen = torch.Generator().manual_seed(3)
        row = -15.0 * torch.rand(2, 1, 64, generator=gen)
        logp = row.expand(2, 20, 64).contiguous()
        out = realized_token_gap(logp)
        assert out["g"].abs().max().item() <= 1e-6
        assert out["G"].abs().max().item() <= 1e-6

    def test_unbatched_input(self):
        gen = torch.Generator().manual_seed(4)
        logp = -5.0 * torch.rand(6, 9, generator=gen)  # [N, T]
        out = realized_token_gap(logp)
        assert out["g"].shape == (9,)
        assert out["G"].shape == ()
