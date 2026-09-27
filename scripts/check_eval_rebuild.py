#!/usr/bin/env python
"""Independent checks for the iteration-2 eval rebuild (specs/i2-eval-rebuild.md).

Written from spec sections 3, 5.5 and 5.6 only, without reading the scorer. It imports
numpy, torch (for torch.load), PyYAML and sklearn, never minigpt or the eval scripts
(S1-T5b). The code is split into this file and its siblings check_eval_common.py,
check_eval_stats.py, check_eval_score_file.py and check_eval_rederive.py.

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

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_eval_common import (  # noqa: F401
    ALIGN_KEYS,
    AUROC_PATH_TOL,
    BLOCK_KEYS,
    CELL_COUNT_KEYS,
    DEFAULT_CONFIG,
    FAMILY_OPTIONAL_QUANTITIES,
    FAMILY_QUANTITIES,
    FLOAT32_KEYS,
    G_TOK_FLOOR,
    ID_ROLE_EVAL_SET,
    LOGP_CHUNK_BLOCKS,
    MAIN_RUN_TAG,
    PREREG_EVAL_SETS,
    REPO_ROOT,
    ROW_KEYS,
    SCORE_FILE_KEYS,
    STRIPPED_OOD,
    TABLE_EVAL_SETS,
    TABLE_QUANTITIES,
    TOKEN_KEYS,
    UNBUILT_CHECKS,
    CheckInputError,
    Triple,
    _dtype_name,
    _load_pt,
    _now_utc,
    _np,
    _np_str,
    _parse_time,
    _resolve_scores_dir,
    _write_json,
    analysis_sha256,
    load_config,
    parse_score_name,
    score_path,
)
from check_eval_rederive import (  # noqa: F401
    Rederiver,
    _cell_triple,
    _compare_cell,
    _compare_families,
    _compare_family_cell,
    _compare_table,
    _family_cells,
    _table_cells,
    _values_match,
    run_rederive,
)
from check_eval_score_file import (  # noqa: F401
    _doc_weights,
    _max_abs,
    _within_tol,
    check_score_file,
)
from check_eval_stats import (  # noqa: F401
    _auroc_and_fpr,
    _decision,
    block_scores_from_logp,
    bootstrap_p_value,
    doc_bootstrap,
    holm,
    paired_delta,
    percentile_ci,
    weighted_auroc,
    weighted_fpr_at_tpr,
)

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
