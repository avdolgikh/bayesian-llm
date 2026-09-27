"""Score-file checks of check_eval_rebuild.py: block re-derivation (S1-T5b).

Part of the independent checker (S1-T5b): it imports numpy, torch, PyYAML, sklearn
and its check_eval_* siblings, never minigpt or the eval scripts.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from check_eval_common import (
    BLOCK_KEYS,
    FLOAT32_KEYS,
    G_TOK_FLOOR,
    ROW_KEYS,
    SCORE_FILE_KEYS,
    TOKEN_KEYS,
    _dtype_name,
    _load_pt,
    _np,
    _np_str,
)
from check_eval_stats import block_scores_from_logp

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
