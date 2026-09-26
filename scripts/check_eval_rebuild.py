#!/usr/bin/env python
"""Independent checks for the iteration-2 eval rebuild (specs/i2-eval-rebuild.md).

Written from spec sections 3, 5.5 and 5.6 only, without reading the scorer. It imports
numpy, torch (for torch.load), PyYAML and sklearn, never minigpt or the eval scripts
(S1-T5b).

Subcommands:
  --rederive       Re-derive the block scores of every test score file, rebuild the result
                   tables and the families from the score files, write them to
                   <scores-dir>/rederive/, and compare them with the scorer's table files
                   (S1-T5b, S1-T5j). Report: <scores-dir>/rederive_check.json.
  --freeze         Write <scores-dir>/prereg.json with the sha256 of the YAML `analysis`
                   block and `frozen_at`. Refuses to overwrite an existing file.
  --check prereg   S1-T5i. Report: <scores-dir>/prereg_check.json.
  --check align    S1-T5k. Report: <scores-dir>/align_check.json.
  --check repro | stream | manifest   Not built yet; exit code 2.

Exit codes: 0 PASS, 1 FAIL, 2 check not built or bad input.

How --rederive computes (section 5.6):
  * A cell is (score set, ID domain, OOD domain). ID blocks have label 0 and OOD blocks
    label 1. ID = the `id_base` domain comes from `{s}__test__main.pt`, ID = the
    `id_adapter` domain from `{s}__test_hn__main.pt`. OOD blocks come from
    `{s}__test__main.pt`, or from `{s}__arxiv_stripped__main.pt` for `arxiv_stripped`.
  * Block weight w_b = 1 / k_d, with k_d the number of blocks of document d in the cell.
  * AUROC_w = roc_auc_score(y, s, sample_weight=w). FPR95_w = fpr[argmax(tpr >= 0.95)]
    from roc_curve(..., drop_intermediate=False).
  * One generator per (ID, OOD) pair: default_rng([seed, i_id, i_ood]), i = position in
    eval_set.domains, with arxiv_stripped after the last domain. Per resample: ID document
    draws, then OOD document draws; resampled weight w_b * c_d.
  * Inside the resample loop AUROC_w is auc(roc_curve(...)), the computation that
    roc_auc_score runs, so one sklearn call gives both metrics. The point estimates use
    roc_auc_score itself, and the report records the largest gap between the two paths.
  * Table AUROCs rank the stored blk_* scores. Separately, blk_g, blk_mi, blk_tu, blk_au
    and blk_maxprob_unc are re-derived from logp_real and the tok_* arrays and must match
    the stored values. blk_nll is re-derived and reported only.
  * A value matches when |table - rederived| <= tol * max(1, |rederived|), with tol =
    checks.rederive_tol.

Table file schema (the scorer's `--analyze` output is compared against it):
  table_<eval_set>.json = {"eval_set": str, "cells": [cell, ...]}
  cell = {"score_set", "id_domain", "ood_domain",
          "n_id_docs", "n_ood_docs", "n_id_blocks", "n_ood_blocks",   (compared if present)
          "scores": {score: {"auroc_w": x, "auroc_w_ci": [lo, hi],
                             "fpr95_w": x, "fpr95_w_ci": [lo, hi]}}}
  families.json = {"families": {name: {"status", "m", "cells": [fcell, ...]}}}
  fcell = {"score_set", "id_domain", "ood_domain", "delta", "delta_ci": [lo, hi], "p",
           "p_holm", "auroc_w_primary", "auroc_w_contrast" (both compared if present)}
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import auc, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "i2_eval.yaml"

MAIN_RUN_TAG = "main"
TABLE_EVAL_SETS = ("test", "test_hn", "arxiv_stripped")
PREREG_EVAL_SETS = ("test", "test_hn", "arxiv_stripped")
STRIPPED_OOD = "arxiv_stripped"
ID_ROLE_EVAL_SET = {"id_base": "test", "id_adapter": "test_hn"}
UNBUILT_CHECKS = ("repro", "stream", "manifest")

SCORE_FILE_KEYS = (
    "logp_real", "tok_mi", "tok_tu", "tok_au", "tok_maxprob", "tok_sum_p_sq", "tok_correct",
    "blk_g", "blk_mi", "blk_tu", "blk_au", "blk_nll", "blk_maxprob_unc",
    "block_index", "doc_id", "domain", "offset", "weight", "meta",
)
FLOAT32_KEYS = (
    "logp_real", "tok_mi", "tok_tu", "tok_au", "tok_maxprob", "tok_sum_p_sq",
    "blk_g", "blk_mi", "blk_tu", "blk_au", "blk_nll", "blk_maxprob_unc",
)
TOKEN_KEYS = ("tok_mi", "tok_tu", "tok_au", "tok_maxprob", "tok_sum_p_sq", "tok_correct")
BLOCK_KEYS = ("blk_g", "blk_mi", "blk_tu", "blk_au", "blk_nll", "blk_maxprob_unc")
ROW_KEYS = ("block_index", "doc_id", "domain", "offset", "weight")
ALIGN_KEYS = ("doc_id", "domain", "offset")
TABLE_QUANTITIES = ("auroc_w", "auroc_w_ci", "fpr95_w", "fpr95_w_ci")
CELL_COUNT_KEYS = ("n_id_docs", "n_ood_docs", "n_id_blocks", "n_ood_blocks")
FAMILY_QUANTITIES = ("delta", "delta_ci", "p")
FAMILY_OPTIONAL_QUANTITIES = ("auroc_w_primary", "auroc_w_contrast")
G_TOK_FLOOR = -1.0e-6  # S1-T5c: g_t >= -1e-6 on every token
AUROC_PATH_TOL = 1.0e-12  # section 5.6: a faster AUROC path must equal roc_auc_score to 1e-12
LOGP_CHUNK_BLOCKS = 256

Triple = tuple[str, str, str]


class CheckInputError(Exception):
    """A score or table file cannot be read the way the spec describes."""


# ---------------------------------------------------------------------------
# Config, hashing, file helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    """Load the eval YAML (configs/i2_eval.yaml or a fixture copy)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def analysis_sha256(analysis: dict[str, Any]) -> str:
    """sha256 of json.dumps(analysis, sort_keys=True, separators=(",", ":")) (section 5.5)."""
    text = json.dumps(analysis, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_path(scores_dir: Path, score_set: str, eval_set: str, run_tag: str) -> Path:
    """Score-file name from section 5.5: {score_set}__{eval_set}__{run_tag}.pt."""
    return Path(scores_dir) / f"{score_set}__{eval_set}__{run_tag}.pt"


def parse_score_name(path: Path) -> tuple[str, str, str] | None:
    """Split a score-file name into (score_set, eval_set, run_tag), or None."""
    parts = path.stem.split("__")
    if path.suffix != ".pt" or len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_time(value: str) -> dt.datetime:
    stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp


def _load_pt(path: Path, mmap: bool = False) -> tuple[dict[str, Any], bool]:
    """torch.load a local score file; returns (content, used_full_unpickle).

    weights_only=True first. Meta values such as torch.__version__ (TorchVersion), numpy
    scalars or datetimes make it refuse, so it falls back to a full unpickle of the file,
    which this project wrote itself.
    """
    for weights_only in (True, False):
        kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": weights_only}
        try:
            if mmap:
                try:
                    return torch.load(path, mmap=True, **kwargs), not weights_only
                except (RuntimeError, ValueError):
                    pass
            return torch.load(path, **kwargs), not weights_only
        except pickle.UnpicklingError:
            if not weights_only:
                raise
    raise AssertionError("unreachable")


def _np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _np_str(value: Any) -> np.ndarray:
    arr = _np(value)
    if arr.dtype.kind not in "US":
        if arr.dtype.kind != "O":
            raise CheckInputError(f"expected strings, got dtype {arr.dtype}")
        arr = arr.astype(str)
    return arr


def _dtype_name(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.dtype).replace("torch.", "")
    return str(_np(value).dtype)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _resolve_scores_dir(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    out = Path(cfg["scoring"]["out_dir"])
    return out if out.is_absolute() else REPO_ROOT / out


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


# ---------------------------------------------------------------------------
# Score files: block re-derivation (S1-T5b first half)
# ---------------------------------------------------------------------------


def _doc_weights(doc_ids: np.ndarray) -> np.ndarray:
    counts = Counter(doc_ids.tolist())
    return np.array([1.0 / counts[d] for d in doc_ids.tolist()], dtype=np.float64)


def _within_tol(table: np.ndarray, mine: np.ndarray, tol: float) -> np.ndarray:
    table = np.asarray(table, dtype=np.float64)
    mine = np.asarray(mine, dtype=np.float64)
    both_nan = np.isnan(table) & np.isnan(mine)
    return both_nan | (np.abs(table - mine) <= tol * np.maximum(1.0, np.abs(mine)))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def check_score_file(path: Path, tol: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Re-derive the block scores of one score file and compare with the stored ones.

    Returns the check result and the light fields (blk_*, doc_id, domain, ...) that the
    tables need, or None when the file cannot be used.
    """
    name = Path(path).name
    failures: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {"file": name, "failures": failures, "format_warnings": warnings}
    sf, full_unpickle = _load_pt(path)
    if full_unpickle:
        warnings.append(f"{name}: needs weights_only=False (non-tensor objects in the file)")
    missing = [k for k in SCORE_FILE_KEYS if k not in sf]
    if missing:
        failures.append(f"{name}: missing keys {missing}")
    if "logp_real" not in sf or any(k not in sf for k in ("doc_id", "domain")):
        result["pass"] = False
        return result, None

    lp = _np(sf["logp_real"])
    if lp.ndim != 3:
        failures.append(f"{name}: logp_real shape {list(lp.shape)} is not [B, N, T]")
        result["pass"] = False
        return result, None
    n_blocks, n_samples, n_tok = lp.shape
    result.update({"n_blocks": n_blocks, "n_samples": n_samples, "block_size": n_tok})
    for key in FLOAT32_KEYS:
        if key in sf and _dtype_name(sf[key]) != "float32":
            warnings.append(f"{name}: {key} dtype {_dtype_name(sf[key])}, spec says float32")
    if "tok_correct" in sf and _dtype_name(sf["tok_correct"]) != "bool":
        warnings.append(f"{name}: tok_correct dtype {_dtype_name(sf['tok_correct'])}")

    shapes_ok = True
    for key in TOKEN_KEYS:
        if key in sf and list(_np(sf[key]).shape) != [n_blocks, n_tok]:
            failures.append(f"{name}: {key} shape {list(_np(sf[key]).shape)} "
                            f"is not [{n_blocks}, {n_tok}]")
            shapes_ok = False
    for key in BLOCK_KEYS + ROW_KEYS:
        if key in sf and len(_np(sf[key])) != n_blocks:
            failures.append(f"{name}: {key} has {len(_np(sf[key]))} rows, expected {n_blocks}")
            shapes_ok = False
    meta = sf["meta"] if "meta" in sf else {}
    meta_n = meta["n_samples"] if "n_samples" in meta else None
    if meta_n is None or int(meta_n) != n_samples:
        failures.append(f"{name}: meta n_samples {meta_n} does not match logp_real "
                        f"N = {n_samples}")
    if not shapes_ok:
        result["pass"] = False
        return result, None

    derived = block_scores_from_logp(lp)
    result["g_tok_min"] = derived["g_tok_min"]
    if not derived["g_tok_min"] >= G_TOK_FLOOR:
        failures.append(f"{name}: g_t minimum {derived['g_tok_min']:.3g} is below {G_TOK_FLOOR}")
    rederived: dict[str, np.ndarray] = {"blk_g": derived["blk_g"]}
    for blk, tok in (("blk_mi", "tok_mi"), ("blk_tu", "tok_tu"), ("blk_au", "tok_au")):
        if tok in sf:
            rederived[blk] = _np(sf[tok]).astype(np.float64).mean(axis=1)
    if "tok_maxprob" in sf:
        rederived["blk_maxprob_unc"] = (1.0 - _np(sf["tok_maxprob"]).astype(np.float64)).mean(1)
    doc_ids = _np_str(sf["doc_id"])
    if "weight" in sf:
        rederived["weight"] = _doc_weights(doc_ids)

    diffs: dict[str, float] = {}
    for key, mine in rederived.items():
        if key not in sf:
            continue
        stored = _np(sf[key]).astype(np.float64)
        diffs[key] = _max_abs(stored, mine)
        if not _within_tol(stored, mine, tol).all():
            failures.append(f"{name}: {key} max |diff| {diffs[key]:.3g} exceeds tol {tol:g}")
    result["max_abs_diff"] = diffs
    if "blk_nll" in sf:
        stored_nll = _np(sf["blk_nll"]).astype(np.float64)
        result["report_only_blk_nll_max_abs_diff"] = _max_abs(stored_nll, derived["blk_nll"])
        result["report_only_blk_nll_within_tol"] = bool(
            _within_tol(stored_nll, derived["blk_nll"], tol).all())

    light: dict[str, Any] = {"doc_id": doc_ids, "domain": _np_str(sf["domain"])}
    for key in BLOCK_KEYS:
        if key in sf:
            light[key] = _np(sf[key]).astype(np.float64)
    result["pass"] = not failures
    return result, light


# ---------------------------------------------------------------------------
# Cells, tables and families
# ---------------------------------------------------------------------------


class Rederiver:
    """Builds cells from score files on demand, caching files and cells."""

    def __init__(self, cfg: dict[str, Any], scores_dir: Path, log: Callable[[str], None]):
        self.cfg = cfg
        self.scores_dir = Path(scores_dir)
        self.log = log
        an = cfg["analysis"]
        self.tol = float(cfg["checks"]["rederive_tol"])
        boot = an["bootstrap"]
        if boot["unit"] != "document" or boot["stratify_by_class"] is not True:
            raise CheckInputError("the checker implements only a document-unit, "
                                  "class-stratified bootstrap (section 5.6)")
        self.n_resamples = int(boot["resamples"])
        self.boot_seed = int(boot["seed"])
        self.level = float(boot["level"])
        self.percentile_method = str(boot["percentile_method"])
        self.target_tpr = float(an["fpr_target_tpr"])
        self.primary = an["primary_score"]
        self.contrast = an["contrast_score"]
        self.score_names = [self.primary, self.contrast, *an["secondary_scores"]]
        domains = cfg["eval_set"]["domains"]
        self.domain_index = {d["key"]: i for i, d in enumerate(domains)}
        self.domain_index[STRIPPED_OOD] = len(domains)
        self.roles = {d["key"]: d["role"] for d in domains}
        stripped = [d["key"] for d in domains if "stripped_copy" in d and d["stripped_copy"]]
        self.stripped_source = stripped[0] if stripped else None
        self.files: dict[Path, dict[str, Any] | None] = {}
        self.unusable: set[Path] = set()
        self.block_results: list[dict[str, Any]] = []
        self.cells: dict[Triple, dict[str, Any] | None] = {}
        self.cell_errors: dict[Triple, str] = {}

    # -- files ---------------------------------------------------------------
    def light(self, score_set: str, eval_set: str) -> dict[str, Any] | None:
        """Light fields of a main score file; None if the file does not exist."""
        path = score_path(self.scores_dir, score_set, eval_set, MAIN_RUN_TAG)
        if path not in self.files:
            if not path.exists():
                self.files[path] = None
            else:
                try:
                    result, light = check_score_file(path, self.tol)
                except CheckInputError as exc:
                    result = {"file": path.name, "failures": [f"{path.name}: {exc}"],
                              "format_warnings": [], "pass": False}
                    light = None
                self.block_results.append(result)
                self.files[path] = light
                if light is None:
                    self.unusable.add(path)
        if path in self.unusable:
            raise CheckInputError(f"{path.name} is unusable (see its block checks)")
        return self.files[path]

    def id_domain_for_role(self, role: str) -> str:
        keys = [k for k, r in self.roles.items() if r == role]
        if len(keys) != 1:
            raise CheckInputError(f"expected one domain with role {role}, found {keys}")
        return keys[0]

    def _sources(self, triple: Triple) -> list[tuple[int, str, set[str]]]:
        _, id_domain, ood_domain = triple
        if id_domain not in self.roles or self.roles[id_domain] not in ID_ROLE_EVAL_SET:
            raise CheckInputError(f"{id_domain} is not an ID domain")
        sources = [(0, ID_ROLE_EVAL_SET[self.roles[id_domain]], {id_domain})]
        if ood_domain == STRIPPED_OOD:
            if self.stripped_source is None:
                raise CheckInputError("no domain has stripped_copy: true")
            sources.append((1, STRIPPED_OOD, {self.stripped_source, STRIPPED_OOD}))
        elif ood_domain in self.roles and self.roles[ood_domain] == "ood":
            sources.append((1, "test", {ood_domain}))
        else:
            raise CheckInputError(f"{ood_domain} is not an OOD domain")
        return sources

    # -- cells ---------------------------------------------------------------
    def cell(self, triple: Triple) -> dict[str, Any] | None:
        """Bootstrap result for a cell, or None when a score file is missing."""
        if triple in self.cells:
            return self.cells[triple]
        score_set = triple[0]
        labels, docs, parts = [], [], {name: [] for name in self.score_names}
        for label, eval_set, domains in self._sources(triple):
            light = self.light(score_set, eval_set)
            if light is None:
                self.cells[triple] = None
                return None
            rows = np.flatnonzero(np.isin(light["domain"], sorted(domains)))
            if rows.size == 0:
                raise CheckInputError(f"{score_set}__{eval_set}__{MAIN_RUN_TAG}.pt has no "
                                      f"blocks of domain {sorted(domains)}")
            labels.append(np.full(rows.size, label, dtype=np.int64))
            docs.append(light["doc_id"][rows])
            for name in self.score_names:
                if name not in light:
                    raise CheckInputError(f"{score_set}__{eval_set}: missing score {name}")
                parts[name].append(light[name][rows])
        y = np.concatenate(labels)
        doc_ids = np.concatenate(docs)
        weights = np.empty(y.size, dtype=np.float64)
        for c in (0, 1):
            weights[y == c] = _doc_weights(doc_ids[y == c])
        scores = {name: np.concatenate(parts[name]) for name in self.score_names}
        seed = [self.boot_seed, self.domain_index[triple[1]], self.domain_index[triple[2]]]
        started = time.perf_counter()
        boot = doc_bootstrap(
            scores, y, doc_ids, weights, n_resamples=self.n_resamples, seed=seed,
            level=self.level, target_tpr=self.target_tpr,
            percentile_method=self.percentile_method,
        )
        boot["rng_seed"] = seed
        self.log(f"  cell {'|'.join(triple)}: {boot['n_id_docs']}+{boot['n_ood_docs']} docs, "
                 f"{time.perf_counter() - started:.1f} s")
        self.cells[triple] = boot
        return boot

    def safe_cell(self, triple: Triple) -> dict[str, Any] | None:
        try:
            return self.cell(triple)
        except CheckInputError as exc:
            self.cell_errors[triple] = str(exc)
            self.cells[triple] = None
            return None

    def cell_json(self, triple: Triple) -> dict[str, Any]:
        boot = self.cells[triple]
        return {
            "score_set": triple[0], "id_domain": triple[1], "ood_domain": triple[2],
            "n_id_docs": boot["n_id_docs"], "n_ood_docs": boot["n_ood_docs"],
            "n_id_blocks": boot["n_id_blocks"], "n_ood_blocks": boot["n_ood_blocks"],
            "rng_seed": boot["rng_seed"],
            "scores": {
                name: {
                    "auroc_w": boot["point"][name]["auroc_w"],
                    "auroc_w_ci": boot["ci"][name]["auroc_w"],
                    "fpr95_w": boot["point"][name]["fpr95_w"],
                    "fpr95_w_ci": boot["ci"][name]["fpr95_w"],
                }
                for name in self.score_names
            },
        }

    # -- plans ---------------------------------------------------------------
    def planned_table_cells(self) -> dict[str, list[Triple]]:
        sets = self.cfg["scoring"]["score_sets"]
        ood = self.cfg["analysis"]["ood_domains"]
        id_base = self.id_domain_for_role("id_base")
        id_adapter = self.id_domain_for_role("id_adapter")
        return {
            "test": [(s, id_base, o) for s in sets["test"] for o in ood],
            "test_hn": [(s, id_adapter, o) for s in sets["test_hn"] for o in ood],
            "arxiv_stripped": (
                [(s, id_base, STRIPPED_OOD) for s in sets["arxiv_stripped"]]
                + [(s, id_adapter, STRIPPED_OOD) for s in sets["arxiv_stripped"]
                   if s in sets["test_hn"]]
            ),
        }

    def families(self) -> dict[str, dict[str, Any]]:
        an = self.cfg["analysis"]
        out: dict[str, dict[str, Any]] = {}
        for fname, rows in an["families"].items():
            triples = [(s, id_dom, o) for s, id_dom in rows for o in an["ood_domains"]]
            cells = []
            for triple in triples:
                boot = self.safe_cell(triple)
                if boot is None:
                    continue
                pd = paired_delta(boot, self.primary, self.contrast, level=self.level,
                                  percentile_method=self.percentile_method)
                cells.append({
                    "score_set": triple[0], "id_domain": triple[1], "ood_domain": triple[2],
                    "auroc_w_primary": boot["point"][self.primary]["auroc_w"],
                    "auroc_w_contrast": boot["point"][self.contrast]["auroc_w"],
                    **pd,
                })
            if not cells:
                status = "not_scored"
            elif len(cells) < len(triples):
                status = "incomplete"
            else:
                status = "scored"
            if status == "scored":
                for c, p_adj in zip(cells, holm([c["p"] for c in cells])):
                    c["p_holm"] = p_adj
                    c["decision"] = _decision(c["delta"], p_adj, float(an["margin_auroc"]),
                                              float(an["alpha"]))
            else:
                for c in cells:
                    c["p_holm"] = None
            missing = [f"{'|'.join(t)}" for t in triples
                       if t not in {(c["score_set"], c["id_domain"], c["ood_domain"])
                                    for c in cells}]
            out[fname] = {"status": status, "m": len(triples), "cells": cells,
                          "missing_cells": missing}
        return out

    def descriptive(self, tables: dict[str, list[Triple]]) -> dict[str, Any]:
        an = self.cfg["analysis"]
        above = []
        for triples in tables.values():
            for t in triples:
                boot = self.cells[t]
                lo = boot["ci"][self.primary]["auroc_w"][0]
                above.append({"cell": "|".join(t), "primary_auroc_w_ci_lo": lo,
                              "above_chance": lo > 0.5})
        rep_cfg = an["descriptive"]["replication"]
        replication = {}
        for s, old_point in rep_cfg["old_point"].items():
            boot = self.safe_cell((s, rep_cfg["id"], rep_cfg["ood"]))
            if boot is None:
                replication[s] = {"status": "not_scored"}
                continue
            new_m = boot["point"][self.contrast]["auroc_w"]
            new_ci = boot["ci"][self.contrast]["auroc_w"]
            old_ci = rep_cfg["old_cluster_ci"][s]
            inside_old = old_ci[0] <= new_m <= old_ci[1]
            old_inside_new = new_ci[0] <= old_point <= new_ci[1]
            replication[s] = {"new_auroc_w_contrast": new_m, "new_ci": new_ci,
                              "old_point": old_point, "old_cluster_ci": old_ci,
                              "replicates": bool(inside_old and old_inside_new)}
        return {"above_chance": above, "replication": replication}


def _values_match(table_value: Any, mine: Any, tol: float) -> tuple[bool, float]:
    if mine is None or table_value is None:
        return (mine is None and table_value is None), math.nan
    try:
        a = np.asarray(table_value, dtype=np.float64)
        b = np.asarray(mine, dtype=np.float64)
    except (TypeError, ValueError):
        return False, math.nan
    if a.shape != b.shape:
        return False, math.nan
    return bool(_within_tol(a, b, tol).all()), _max_abs(a, b)


def _table_cells(obj: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(obj, dict) and "cells" in obj and isinstance(obj["cells"], list):
        return obj["cells"]
    raise CheckInputError(f"{name}: expected an object with a 'cells' list")


def _cell_triple(c: Any) -> Triple | None:
    try:
        return str(c["score_set"]), str(c["id_domain"]), str(c["ood_domain"])
    except (KeyError, TypeError):
        return None


def _compare_table(
    name: str, obj: Any, planned: list[Triple], rd: Rederiver
) -> tuple[list[str], int, dict[str, bool]]:
    """Compare one scorer table with the re-derivation; returns failures, count, per cell."""
    failures: list[str] = []
    per_cell: dict[str, bool] = {}
    n_compared = 0
    seen: set[Triple] = set()
    for c in _table_cells(obj, name):
        triple = _cell_triple(c)
        if triple is None:
            failures.append(f"{name}: a cell lacks score_set, id_domain or ood_domain")
            continue
        tag = "|".join(triple)
        n_before = len(failures)
        if triple in seen:
            failures.append(f"{name}: cell {tag} appears twice")
        else:
            seen.add(triple)
            n_compared += _compare_cell(name, c, triple, rd, failures)
        per_cell[tag] = (per_cell[tag] if tag in per_cell else True) and (
            len(failures) == n_before)
    for triple in planned:
        if triple not in seen and rd.safe_cell(triple) is not None:
            failures.append(f"{name}: cell {'|'.join(triple)} is missing from the table")
            per_cell["|".join(triple)] = False
    return failures, n_compared, per_cell


def _compare_cell(name: str, c: dict[str, Any], triple: Triple, rd: Rederiver,
                  failures: list[str]) -> int:
    tag = "|".join(triple)
    boot = rd.safe_cell(triple)
    if boot is None:
        why = rd.cell_errors[triple] if triple in rd.cell_errors else "missing score files"
        failures.append(f"{name}: cell {tag} cannot be rederived ({why})")
        return 0
    mine = rd.cell_json(triple)
    for key in CELL_COUNT_KEYS:
        if key in c and c[key] != mine[key]:
            failures.append(f"{name}: cell {tag} {key}: table {c[key]} vs rederived {mine[key]}")
    if "scores" not in c or not isinstance(c["scores"], dict):
        failures.append(f"{name}: cell {tag} has no 'scores' object")
        return 0
    n_compared = 0
    for score in rd.score_names:
        if score not in c["scores"] or not isinstance(c["scores"][score], dict):
            failures.append(f"{name}: cell {tag} lacks score {score}")
            continue
        theirs, ours = c["scores"][score], mine["scores"][score]
        for q in TABLE_QUANTITIES:
            if q not in theirs:
                failures.append(f"{name}: cell {tag} {score} lacks {q}")
                continue
            ok, diff = _values_match(theirs[q], ours[q], rd.tol)
            n_compared += 1
            if not ok:
                failures.append(f"{name}: cell {tag} {score} {q}: table {theirs[q]} vs "
                                f"rederived {ours[q]} (max |diff| {diff:.3g})")
    return n_compared


def _family_cells(fam: Any) -> list[Any]:
    if isinstance(fam, dict) and "cells" in fam and isinstance(fam["cells"], list):
        return fam["cells"]
    return []


def _compare_families(obj: Any, derived: dict[str, dict[str, Any]], tol: float
                      ) -> tuple[list[str], int, dict[str, bool]]:
    """Compare families.json with the re-derivation; returns failures, count, per cell."""
    name = "families.json"
    failures: list[str] = []
    per_cell: dict[str, bool] = {}
    n_compared = 0
    if not (isinstance(obj, dict) and "families" in obj and isinstance(obj["families"], dict)):
        return [f"{name}: expected an object with a 'families' object"], 0, per_cell
    theirs = obj["families"]
    for fname in theirs:
        if fname not in derived:
            failures.append(f"{name}: family {fname} is not in the YAML analysis block")
    for fname, fam in derived.items():
        their_cells = _family_cells(theirs[fname]) if fname in theirs else []
        if fam["status"] == "not_scored":
            if their_cells:
                failures.append(f"{name}: {fname} has cells but no score files to rederive")
            continue
        if fname not in theirs:
            failures.append(f"{name}: family {fname} missing (rederived status {fam['status']})")
            continue
        by_triple: dict[Triple, dict[str, Any]] = {}
        for c in their_cells:
            triple = _cell_triple(c)
            if triple is None:
                failures.append(f"{name}: {fname} has a cell without its triple")
            else:
                by_triple[triple] = c
        mine_triples = set()
        for mc in fam["cells"]:
            triple = (mc["score_set"], mc["id_domain"], mc["ood_domain"])
            mine_triples.add(triple)
            tag = "|".join(triple)
            n_before = len(failures)
            if triple not in by_triple:
                failures.append(f"{name}: {fname} cell {tag} is missing")
            else:
                n_compared += _compare_family_cell(fname, fam, mc, by_triple[triple], tol,
                                                   failures)
            per_cell[f"{fname}|{tag}"] = len(failures) == n_before
        for triple in by_triple:
            if triple not in mine_triples:
                failures.append(f"{name}: {fname} cell {'|'.join(triple)} cannot be rederived")
                per_cell[f"{fname}|{'|'.join(triple)}"] = False
    return failures, n_compared, per_cell


def _compare_family_cell(fname: str, fam: dict[str, Any], mine: dict[str, Any],
                         theirs: dict[str, Any], tol: float, failures: list[str]) -> int:
    tag = "|".join((mine["score_set"], mine["id_domain"], mine["ood_domain"]))
    where = f"families.json: {fname} cell {tag}"
    quantities = list(FAMILY_QUANTITIES)
    quantities += [q for q in FAMILY_OPTIONAL_QUANTITIES if q in theirs]
    quantities.append("p_holm")
    n_compared = 0
    for q in quantities:
        if q not in theirs:
            failures.append(f"{where} lacks {q}")
            continue
        if q == "p_holm" and fam["status"] != "scored" and theirs[q] is not None:
            failures.append(f"{where} p_holm given, but the family is {fam['status']} "
                            f"(missing {fam['missing_cells']})")
            continue
        ok, diff = _values_match(theirs[q], mine[q], tol)
        n_compared += 1
        if not ok:
            failures.append(f"{where} {q}: table {theirs[q]} vs rederived {mine[q]} "
                            f"(max |diff| {diff:.3g})")
    return n_compared


def run_rederive(cfg: dict[str, Any], scores_dir: Path, config_path: Path,
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    """S1-T5b/T5j: rebuild the tables from the score files and compare them."""
    scores_dir = Path(scores_dir)
    out_dir = scores_dir / "rederive"
    rd = Rederiver(cfg, scores_dir, log)
    failures: list[str] = []
    n_compared = 0

    planned = rd.planned_table_cells()
    for eval_set in TABLE_EVAL_SETS:
        for s in cfg["scoring"]["score_sets"][eval_set]:
            try:
                rd.light(s, eval_set)
            except CheckInputError:
                pass  # recorded in the block checks
    log(f"rederive: {sum(len(v) for v in planned.values())} planned table cells, "
        f"{rd.n_resamples} resamples")
    tables: dict[str, list[Triple]] = {}
    not_scored: list[str] = []
    for eval_set, triples in planned.items():
        tables[eval_set] = []
        for triple in triples:
            if rd.safe_cell(triple) is None:
                if triple in rd.cell_errors:
                    failures.append(f"cell {'|'.join(triple)}: {rd.cell_errors[triple]}")
                else:
                    not_scored.append("|".join(triple))
                continue
            tables[eval_set].append(triple)
    families = rd.families()

    for eval_set, triples in tables.items():
        _write_json(out_dir / f"table_{eval_set}.json",
                    {"eval_set": eval_set, "cells": [rd.cell_json(t) for t in triples]})
    _write_json(out_dir / "families.json", {"families": families})
    _write_json(out_dir / "descriptive.json", rd.descriptive(tables))

    compared_files = []
    per_cell: dict[str, dict[str, bool]] = {}
    for eval_set, triples in tables.items():
        name = f"table_{eval_set}.json"
        path = scores_dir / name
        if not path.exists():
            if triples:
                failures.append(f"{name}: missing table file")
            continue
        try:
            f, n, cells = _compare_table(name, json.loads(path.read_text(encoding="utf-8")),
                                         planned[eval_set], rd)
        except (CheckInputError, json.JSONDecodeError) as exc:
            f, n, cells = [f"{name}: {exc}"], 0, {}
        failures += f
        n_compared += n
        per_cell[name] = cells
        compared_files.append(name)
    fam_path = scores_dir / "families.json"
    if any(f["status"] != "not_scored" for f in families.values()):
        if not fam_path.exists():
            failures.append("families.json: missing table file")
        else:
            try:
                fam_obj = json.loads(fam_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                fam_obj = None
                failures.append(f"families.json: {exc}")
            if fam_obj is not None:
                f, n, cells = _compare_families(fam_obj, families, rd.tol)
                failures += f
                n_compared += n
                per_cell["families.json"] = cells
            compared_files.append("families.json")
    block_failures = [f for r in rd.block_results for f in r["failures"]]
    failures = block_failures + failures
    path_gap = max([b["auroc_path_gap"] for b in rd.cells.values() if b is not None],
                   default=0.0)
    if path_gap > AUROC_PATH_TOL:
        failures.append(f"auc(roc_curve) differs from roc_auc_score by {path_gap:.3g}")
    if not rd.block_results:
        failures.append("no score files found")

    report = {
        "check": "rederive",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "block_checks_pass": bool(rd.block_results) and not block_failures,
        "block_checks": rd.block_results,
        "not_scored_cells": not_scored,
        "families_status": {k: v["status"] for k, v in families.items()},
        "tables_compared": compared_files,
        "per_cell_pass": per_cell,
        "n_values_compared": n_compared,
        "auroc_path_max_gap": path_gap,
        "tolerance": rd.tol,
        "analysis_sha256": analysis_sha256(cfg["analysis"]),
        "config": str(config_path),
        "scores_dir": str(scores_dir),
        "rederived_dir": str(out_dir),
        "created_at": _now_utc(),
    }
    _write_json(scores_dir / "rederive_check.json", report)
    return report


# ---------------------------------------------------------------------------
# --freeze, --check prereg, --check align
# ---------------------------------------------------------------------------


def run_freeze(cfg: dict[str, Any], scores_dir: Path, config_path: Path) -> int:
    """Write prereg.json; refuse to overwrite an existing one."""
    path = Path(scores_dir) / "prereg.json"
    if path.exists():
        print(f"freeze: {path} exists; refusing to overwrite it")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"analysis_sha256": analysis_sha256(cfg["analysis"]), "frozen_at": _now_utc(),
               "config": str(config_path)}
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"freeze: wrote {path} analysis_sha256={payload['analysis_sha256']}")
    return 0


def _score_files(scores_dir: Path) -> list[tuple[Path, tuple[str, str, str]]]:
    out = []
    for path in sorted(Path(scores_dir).glob("*.pt")):
        parsed = parse_score_name(path)
        if parsed is not None:
            out.append((path, parsed))
    return out


def run_check_prereg(cfg: dict[str, Any], scores_dir: Path) -> dict[str, Any]:
    """S1-T5i: YAML hash = prereg.json = every test file's meta; frozen_at < created_at."""
    scores_dir = Path(scores_dir)
    failures: list[str] = []
    expected = analysis_sha256(cfg["analysis"])
    frozen_at = None
    prereg_path = scores_dir / "prereg.json"
    if not prereg_path.exists():
        failures.append("prereg.json: missing")
    else:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        if prereg["analysis_sha256"] != expected:
            failures.append(f"prereg.json: analysis_sha256 {prereg['analysis_sha256']} differs "
                            f"from the YAML analysis block {expected}")
        frozen_at = _parse_time(prereg["frozen_at"])
    files = [(p, n) for p, n in _score_files(scores_dir) if n[1] in PREREG_EVAL_SETS]
    for path, _ in files:
        meta = _load_pt(path, mmap=True)[0]["meta"]
        if meta["analysis_sha256"] != expected:
            failures.append(f"{path.name}: meta analysis_sha256 {meta['analysis_sha256']} "
                            f"differs from {expected}")
        created = _parse_time(meta["created_at"])
        if frozen_at is not None and not frozen_at < created:
            failures.append(f"{path.name}: created_at {meta['created_at']} is not after "
                            f"frozen_at {frozen_at.isoformat()}")
    if not files:
        failures.append(f"no score files of {list(PREREG_EVAL_SETS)} in {scores_dir}")
    report = {"check": "prereg", "status": "PASS" if not failures else "FAIL",
              "failures": failures, "analysis_sha256": expected,
              "n_test_score_files": len(files), "created_at": _now_utc()}
    _write_json(scores_dir / "prereg_check.json", report)
    return report


def _align_fields(path: Path) -> dict[str, np.ndarray]:
    sf, _ = _load_pt(path, mmap=True)
    return {
        "block_index": _np(sf["block_index"]).astype(np.int64),
        "doc_id": _np_str(sf["doc_id"]),
        "domain": _np_str(sf["domain"]),
        "offset": _np(sf["offset"]).astype(np.int64),
    }


def run_check_align(cfg: dict[str, Any], scores_dir: Path) -> dict[str, Any]:
    """S1-T5k: doc_id, domain and offset agree across the score files of each eval set.

    The reference is the first `main` file of the eval set. Other files are matched by
    block_index, so a `--block-ids` re-run is compared with the matching main rows.
    """
    scores_dir = Path(scores_dir)
    failures: list[str] = []
    groups: dict[str, list[tuple[Path, str]]] = {}
    for path, (_, eval_set, run_tag) in _score_files(scores_dir):
        groups.setdefault(eval_set, []).append((path, run_tag))
    summary = {}
    for eval_set, members in sorted(groups.items()):
        mains = [p for p, tag in members if tag == MAIN_RUN_TAG]
        ref_path = mains[0] if mains else members[0][0]
        ref = _align_fields(ref_path)
        ref_row = {int(b): i for i, b in enumerate(ref["block_index"])}
        for path, tag in members:
            if path == ref_path:
                continue
            cur = _align_fields(path)
            if tag == MAIN_RUN_TAG and not np.array_equal(cur["block_index"],
                                                          ref["block_index"]):
                failures.append(f"{path.name}: block_index differs from {ref_path.name}")
                continue
            unknown = [int(b) for b in cur["block_index"] if int(b) not in ref_row]
            if unknown:
                failures.append(f"{path.name}: {len(unknown)} block_index values are not in "
                                f"{ref_path.name}")
                continue
            rows = np.array([ref_row[int(b)] for b in cur["block_index"]], dtype=np.int64)
            for key in ALIGN_KEYS:
                if len(cur[key]) != len(rows):
                    failures.append(f"{path.name}: {key} has {len(cur[key])} rows, "
                                    f"block_index has {len(rows)}")
                    continue
                bad = int((cur[key] != ref[key][rows]).sum())
                if bad:
                    failures.append(f"{path.name}: {key} differs from {ref_path.name} on "
                                    f"{bad} blocks")
        summary[eval_set] = {"reference": ref_path.name, "n_files": len(members)}
    if not groups:
        failures.append(f"no score files in {scores_dir}")
    report = {"check": "align", "status": "PASS" if not failures else "FAIL",
              "failures": failures, "eval_sets": summary, "created_at": _now_utc()}
    _write_json(scores_dir / "align_check.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent checks for the iteration-2 eval rebuild (spec S1).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rederive", action="store_true",
                      help="rebuild the tables from the score files and compare them")
    mode.add_argument("--freeze", action="store_true",
                      help="write prereg.json with the analysis-block hash")
    mode.add_argument("--check", choices=["repro", "stream", "manifest", "prereg", "align"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="eval YAML (default: configs/i2_eval.yaml)")
    parser.add_argument("--scores-dir", type=Path, default=None,
                        help="score directory (default: scoring.out_dir from the YAML)")
    return parser.parse_args(argv)


def _print_report(report: dict[str, Any]) -> None:
    print(f"{report['check']}: {report['status']}")
    for line in report["failures"][:50]:
        print(f"  FAIL {line}")
    if len(report["failures"]) > 50:
        print(f"  ... {len(report['failures']) - 50} more failures in the report file")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the exit code (0 PASS, 1 FAIL, 2 not built or bad input)."""
    args = _parse_args(argv)
    if args.check in UNBUILT_CHECKS:
        print(f"--check {args.check} is not built yet (specs/i2-eval-rebuild.md section 5.7)")
        return 2
    try:
        cfg = load_config(args.config)
        scores_dir = _resolve_scores_dir(cfg, args.scores_dir)
        if args.freeze:
            return run_freeze(cfg, scores_dir, args.config)
        if not scores_dir.is_dir():
            print(f"error: scores directory {scores_dir} does not exist")
            return 2
        if args.rederive:
            report = run_rederive(cfg, scores_dir, args.config)
        elif args.check == "prereg":
            report = run_check_prereg(cfg, scores_dir)
        else:
            report = run_check_align(cfg, scores_dir)
    except (CheckInputError, KeyError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 2
    _print_report(report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
