"""Cells, tables and families of check_eval_rebuild.py --rederive (S1-T5b, S1-T5j).

Part of the independent checker (S1-T5b): it imports numpy, torch, PyYAML, sklearn
and its check_eval_* siblings, never minigpt or the eval scripts.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from check_eval_common import (
    AUROC_PATH_TOL,
    CELL_COUNT_KEYS,
    FAMILY_OPTIONAL_QUANTITIES,
    FAMILY_QUANTITIES,
    ID_ROLE_EVAL_SET,
    MAIN_RUN_TAG,
    STRIPPED_OOD,
    TABLE_EVAL_SETS,
    TABLE_QUANTITIES,
    CheckInputError,
    Triple,
    _now_utc,
    _write_json,
    analysis_sha256,
    score_path,
)
from check_eval_score_file import (
    _doc_weights,
    _max_abs,
    _within_tol,
    check_score_file,
)
from check_eval_stats import (
    _decision,
    doc_bootstrap,
    holm,
    paired_delta,
)

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
