"""Section 3 and section 5.6 statistics of check_eval_rebuild.py.

Part of the independent checker (S1-T5b): it imports numpy, torch, PyYAML, sklearn
and its check_eval_* siblings, never minigpt or the eval scripts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from check_eval_common import LOGP_CHUNK_BLOCKS, CheckInputError, _np
from sklearn.metrics import auc, roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Section 3: realized-token Jensen gap
# ---------------------------------------------------------------------------


def block_scores_from_logp(logp_real: Any) -> dict[str, Any]:
    """Block G and NLL under p-bar from per-sample log-probs of shape [B, N, T].

    log pbar_t = logsumexp_s l_st - log N,  g_t = log pbar_t - mean_s l_st,
    G(b) = mean_t g_t,  NLL(b) = mean_t (-log pbar_t). Computed in float64.
    """
    lp_all = _np(logp_real)
    if lp_all.ndim != 3:
        raise CheckInputError(f"logp_real must be [B, N, T], got shape {lp_all.shape}")
    n_blocks, n_samples, _ = lp_all.shape
    blk_g = np.empty(n_blocks, dtype=np.float64)
    blk_nll = np.empty(n_blocks, dtype=np.float64)
    g_min = math.inf
    for start in range(0, n_blocks, LOGP_CHUNK_BLOCKS):
        lp = lp_all[start:start + LOGP_CHUNK_BLOCKS].astype(np.float64)
        top = lp.max(axis=1)
        lse = top + np.log(np.exp(lp - top[:, None, :]).sum(axis=1))
        log_pbar = lse - math.log(n_samples)
        g_tok = log_pbar - lp.mean(axis=1)
        blk_g[start:start + len(lp)] = g_tok.mean(axis=1)
        blk_nll[start:start + len(lp)] = (-log_pbar).mean(axis=1)
        if g_tok.size:
            g_min = min(g_min, float(g_tok.min()))
    return {"blk_g": blk_g, "blk_nll": blk_nll, "g_tok_min": g_min}


# ---------------------------------------------------------------------------
# Section 5.6: weighted metrics, document bootstrap, p-value, Holm
# ---------------------------------------------------------------------------


def weighted_auroc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """AUROC_w = roc_auc_score(y, s, sample_weight=w); label 1 = OOD, higher = more OOD."""
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def weighted_fpr_at_tpr(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, target_tpr: float
) -> float:
    """FPR_w at the largest threshold whose weighted TPR reaches target_tpr."""
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights, drop_intermediate=False)
    return float(fpr[np.argmax(tpr >= target_tpr)])


def _auroc_and_fpr(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, target_tpr: float
) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights, drop_intermediate=False)
    return float(auc(fpr, tpr)), float(fpr[np.argmax(tpr >= target_tpr)])


def percentile_ci(values: np.ndarray, level: float, percentile_method: str) -> list[float]:
    """Two-sided percentile interval, np.percentile with the given method."""
    lo = round(100.0 * (1.0 - level) / 2.0, 12)
    hi = round(100.0 - lo, 12)
    q = np.percentile(np.asarray(values, dtype=np.float64), [lo, hi], method=percentile_method)
    return [float(q[0]), float(q[1])]


def doc_bootstrap(
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    doc_ids: np.ndarray,
    weights: np.ndarray,
    *,
    n_resamples: int,
    seed: int | Sequence[int],
    level: float,
    target_tpr: float,
    percentile_method: str,
) -> dict[str, Any]:
    """Document-clustered, class-stratified bootstrap of AUROC_w and FPR95_w.

    Documents are np.unique(doc_ids[labels == c]) per class. Resample r draws the ID
    documents, then the OOD documents, from default_rng(seed); block weights become
    w_b * c_d. All scores share the same draws, so differences between them are paired.
    """
    labels = np.asarray(labels).astype(np.int64)
    doc_ids = np.asarray(doc_ids)
    weights = np.asarray(weights, dtype=np.float64)
    id_rows = np.flatnonzero(labels == 0)
    ood_rows = np.flatnonzero(labels == 1)
    if id_rows.size == 0 or ood_rows.size == 0:
        raise CheckInputError("a cell needs ID (label 0) and OOD (label 1) blocks")
    id_docs, id_inv = np.unique(doc_ids[id_rows], return_inverse=True)
    ood_docs, ood_inv = np.unique(doc_ids[ood_rows], return_inverse=True)
    n_id, n_ood = len(id_docs), len(ood_docs)

    point: dict[str, dict[str, float]] = {}
    path_gap = 0.0
    for name, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        a_exact = weighted_auroc(labels, s, weights)
        a_fast, fpr95 = _auroc_and_fpr(labels, s, weights, target_tpr)
        path_gap = max(path_gap, abs(a_exact - a_fast))
        point[name] = {"auroc_w": a_exact, "fpr95_w": fpr95}

    order = np.r_[id_rows, ood_rows]
    y_ord = labels[order]
    w_ord = weights[order]
    s_ord = {name: np.asarray(s, dtype=np.float64)[order] for name, s in scores.items()}
    count_index = np.r_[id_inv, n_id + ood_inv]
    res = {name: {"auroc_w": np.empty(n_resamples), "fpr95_w": np.empty(n_resamples)}
           for name in scores}
    rng = np.random.default_rng(seed)
    for r in range(n_resamples):
        di = rng.integers(0, n_id, size=n_id)
        do = rng.integers(0, n_ood, size=n_ood)
        counts = np.r_[np.bincount(di, minlength=n_id), np.bincount(do, minlength=n_ood)]
        w_r = w_ord * counts[count_index]
        keep = w_r > 0
        y_k, w_k = y_ord[keep], w_r[keep]
        for name in scores:
            a_r, f_r = _auroc_and_fpr(y_k, s_ord[name][keep], w_k, target_tpr)
            res[name]["auroc_w"][r] = a_r
            res[name]["fpr95_w"][r] = f_r

    ci = {name: {q: percentile_ci(res[name][q], level, percentile_method)
                 for q in ("auroc_w", "fpr95_w")} for name in scores}
    return {
        "point": point, "resamples": res, "ci": ci,
        "n_id_docs": n_id, "n_ood_docs": n_ood,
        "n_id_blocks": int(id_rows.size), "n_ood_blocks": int(ood_rows.size),
        "n_resamples": n_resamples, "auroc_path_gap": path_gap,
        "level": level, "percentile_method": percentile_method,
    }


def bootstrap_p_value(deltas: np.ndarray) -> float:
    """Two-sided p = min(1, 2 (1 + min(#{d <= 0}, #{d >= 0})) / (B + 1))."""
    deltas = np.asarray(deltas, dtype=np.float64)
    n_le = int((deltas <= 0).sum())
    n_ge = int((deltas >= 0).sum())
    return min(1.0, 2.0 * (1 + min(n_le, n_ge)) / (deltas.size + 1))


def paired_delta(
    boot: dict[str, Any], score_a: str, score_b: str, *, level: float,
    percentile_method: str,
) -> dict[str, Any]:
    """Delta = AUROC_w(a) - AUROC_w(b) with its paired bootstrap CI and p-value."""
    deltas = boot["resamples"][score_a]["auroc_w"] - boot["resamples"][score_b]["auroc_w"]
    return {
        "delta": boot["point"][score_a]["auroc_w"] - boot["point"][score_b]["auroc_w"],
        "delta_ci": percentile_ci(deltas, level, percentile_method),
        "p": bootstrap_p_value(deltas),
    }


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjustment, stable ascending sort, returned in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    stepped = np.minimum(1.0, (m - np.arange(m)) * p[order])
    stepped = np.maximum.accumulate(stepped)
    out = np.empty(m, dtype=np.float64)
    out[order] = stepped
    return [float(v) for v in out]


def _decision(delta: float, p_holm: float, margin: float, alpha: float) -> str:
    if delta >= margin and p_holm < alpha:
        return "primary_beats_contrast"
    if delta <= -margin and p_holm < alpha:
        return "contrast_beats_primary"
    return "no_difference"
