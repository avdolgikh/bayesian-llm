"""--analyze of scripts/eval_c_checkpoints.py: document-bootstrap tables and families."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from eval_c_load import ANALYSIS_RUN_TAG, ID_ROLE_EVAL_SET, STRIPPED_OOD
from eval_c_meta import score_file_path, scores_dir

from minigpt.evalset import analysis_sha256
from minigpt.uncertainty import doc_bootstrap_auroc, holm, paired_doc_bootstrap

# ---------------------------------------------------------------------------
# --analyze: document-bootstrap tables and families (sections 3 and 5.6)
# ---------------------------------------------------------------------------

Triple = tuple[str, str, str]


def _roles(cfg: dict) -> dict[str, str]:
    return {d["key"]: d["role"] for d in cfg["eval_set"]["domains"]}


def _domain_for_role(cfg: dict, role: str) -> str:
    keys = [k for k, r in _roles(cfg).items() if r == role]
    if len(keys) != 1:
        raise ValueError(f"eval_set.domains needs exactly one {role} domain, got {keys}")
    return keys[0]


def planned_table_cells(cfg: dict) -> dict[str, list[Triple]]:
    """Cells (score set, ID domain, OOD domain) of the three result tables."""
    sets = cfg["scoring"]["score_sets"]
    ood = cfg["analysis"]["ood_domains"]
    id_base = _domain_for_role(cfg, "id_base")
    id_adapter = _domain_for_role(cfg, "id_adapter")
    return {
        "test": [(s, id_base, o) for s in sets["test"] for o in ood],
        "test_hn": [(s, id_adapter, o) for s in sets["test_hn"] for o in ood],
        "arxiv_stripped": (
            [(s, id_base, STRIPPED_OOD) for s in sets["arxiv_stripped"]]
            + [(s, id_adapter, STRIPPED_OOD) for s in sets["arxiv_stripped"]
               if s in sets["test_hn"]]
        ),
    }


def _cell_sources(cfg: dict, triple: Triple) -> list[tuple[int, str, set[str]]]:
    """[(label, eval set of the score file, block domains)] for one cell (section 5.6)."""
    _, id_domain, ood_domain = triple
    roles = _roles(cfg)
    if id_domain not in roles or roles[id_domain] not in ID_ROLE_EVAL_SET:
        raise ValueError(f"{id_domain} is not an ID domain")
    sources = [(0, ID_ROLE_EVAL_SET[roles[id_domain]], {id_domain})]
    if ood_domain == STRIPPED_OOD:
        stripped = [d["key"] for d in cfg["eval_set"]["domains"] if d["stripped_copy"]]
        sources.append((1, STRIPPED_OOD, {*stripped, STRIPPED_OOD}))
    elif ood_domain in roles and roles[ood_domain] == "ood":
        sources.append((1, "test", {ood_domain}))
    else:
        raise ValueError(f"{ood_domain} is not an OOD domain")
    return sources


def _domain_index(cfg: dict) -> dict[str, int]:
    """Bootstrap seed index: position in eval_set.domains; arxiv_stripped comes last."""
    index = {d["key"]: i for i, d in enumerate(cfg["eval_set"]["domains"])}
    index[STRIPPED_OOD] = len(index)
    return index


class _CellData:
    """Loads the block scores of the `main` score files and assembles cells."""

    def __init__(self, cfg: dict, score_names: list[str]) -> None:
        self.cfg = cfg
        self.score_names = score_names
        self._files: dict[Path, dict | None] = {}

    def _fields(self, score_set: str, eval_set: str) -> dict | None:
        path = score_file_path(self.cfg, score_set, eval_set, ANALYSIS_RUN_TAG)
        if path not in self._files:
            if not path.exists():
                self._files[path] = None
            else:
                sf = torch.load(path, weights_only=True, mmap=True)
                self._files[path] = {
                    "doc_id": np.asarray(sf["doc_id"]),
                    "domain": np.asarray(sf["domain"]),
                    **{n: sf[n].double().numpy() for n in self.score_names},
                }
        return self._files[path]

    def cell(self, triple: Triple) -> dict | None:
        """Labels, documents, weights and scores of a cell; None if a score file is missing."""
        labels, docs, parts = [], [], {n: [] for n in self.score_names}
        for label, eval_set, domains in _cell_sources(self.cfg, triple):
            fields = self._fields(triple[0], eval_set)
            if fields is None:
                return None
            rows = np.flatnonzero(np.isin(fields["domain"], sorted(domains)))
            if rows.size == 0:
                raise ValueError(f"{triple[0]}__{eval_set}: no blocks of {sorted(domains)}")
            labels.append(np.full(rows.size, label, dtype=np.int64))
            docs.append(fields["doc_id"][rows])
            for n in self.score_names:
                parts[n].append(fields[n][rows])
        y = np.concatenate(labels)
        doc_ids = np.concatenate(docs)
        weights = np.empty(y.size, dtype=np.float64)
        for c in (0, 1):  # w_b = 1 / k_d, k_d = blocks of document d in the cell
            counts = Counter(doc_ids[y == c].tolist())
            weights[y == c] = [1.0 / counts[d] for d in doc_ids[y == c].tolist()]
        return {"labels": y, "doc_ids": doc_ids, "weights": weights,
                "scores": {n: np.concatenate(parts[n]) for n in self.score_names},
                "n_id_blocks": int((y == 0).sum()), "n_ood_blocks": int((y == 1).sum())}


def _decision(delta: float, p_holm: float, margin: float, alpha: float) -> str:
    """Section 3 decision rule for one family cell."""
    if delta >= margin and p_holm < alpha:
        return "primary_beats_contrast"
    if delta <= -margin and p_holm < alpha:
        return "contrast_beats_primary"
    return "no_difference"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def run_analyze(cfg: dict) -> dict[str, Path]:
    """Write table_{test,test_hn,arxiv_stripped}.json, families.json and descriptive.json.

    Every cell uses the `main` score files, the document weights 1/k_d and the paired
    document bootstrap of minigpt.uncertainty with the generator seed
    [analysis.bootstrap.seed, i_id, i_ood]. Cells whose score files do not exist yet are
    left out; a family with a missing cell is "incomplete" and gets no Holm adjustment.
    """
    an = cfg["analysis"]
    boot_cfg = an["bootstrap"]
    if (boot_cfg["unit"] != "document" or boot_cfg["stratify_by_class"] is not True
            or boot_cfg["percentile_method"] != "linear"):
        raise ValueError("only the document-unit, class-stratified, linear-percentile "
                         "bootstrap of section 5.6 is implemented")
    n_resamples, seed, level = boot_cfg["resamples"], boot_cfg["seed"], boot_cfg["level"]
    primary, contrast = an["primary_score"], an["contrast_score"]
    score_names = [primary, contrast, *an["secondary_scores"]]
    index = _domain_index(cfg)
    data = _CellData(cfg, score_names)
    cells: dict[Triple, dict | None] = {}

    def cell_result(triple: Triple) -> dict | None:
        if triple not in cells:
            c = data.cell(triple)
            if c is None:
                cells[triple] = None
            else:
                rng_seed = [seed, index[triple[1]], index[triple[2]]]
                boot = doc_bootstrap_auroc(
                    c["scores"], c["labels"], c["doc_ids"], c["weights"], n_resamples,
                    rng_seed, level, target_tpr=an["fpr_target_tpr"],
                )
                cells[triple] = {"data": c, "boot": boot, "rng_seed": rng_seed}
        return cells[triple]

    stamp = {"analysis_sha256": analysis_sha256(cfg), "run_tag": ANALYSIS_RUN_TAG,
             "n_resamples": n_resamples, "created_at": datetime.now(timezone.utc).isoformat()}
    out = scores_dir(cfg)
    written: dict[str, Path] = {}
    tables: dict[str, list[Triple]] = {}
    for eval_set, triples in planned_table_cells(cfg).items():
        rows = []
        tables[eval_set] = []
        for triple in triples:
            res = cell_result(triple)
            if res is None:
                print(f"  not scored yet: {'|'.join(triple)}")
                continue
            tables[eval_set].append(triple)
            boot = res["boot"]
            rows.append({
                "score_set": triple[0], "id_domain": triple[1], "ood_domain": triple[2],
                "n_id_docs": boot[primary]["n_id_docs"],
                "n_ood_docs": boot[primary]["n_ood_docs"],
                "n_id_blocks": res["data"]["n_id_blocks"],
                "n_ood_blocks": res["data"]["n_ood_blocks"],
                "rng_seed": res["rng_seed"],
                "scores": {
                    n: {"auroc_w": boot[n]["auroc"], "auroc_w_ci": list(boot[n]["auroc_ci"]),
                        "fpr95_w": boot[n]["fpr95"], "fpr95_w_ci": list(boot[n]["fpr95_ci"])}
                    for n in score_names
                },
            })
        path = out / f"table_{eval_set}.json"
        _write_json(path, {"eval_set": eval_set, **stamp, "cells": rows})
        written[f"table_{eval_set}"] = path

    families = {}
    for fname, fam_rows in an["families"].items():
        triples = [(s, id_dom, o) for s, id_dom in fam_rows for o in an["ood_domains"]]
        fcells = []
        for triple in triples:
            res = cell_result(triple)
            if res is None:
                continue
            c = res["data"]
            pair = paired_doc_bootstrap(
                {n: c["scores"][n] for n in (primary, contrast)}, c["labels"], c["doc_ids"],
                c["weights"], [(primary, contrast)], n_resamples, res["rng_seed"], level,
            )[(primary, contrast)]
            fcells.append({
                "score_set": triple[0], "id_domain": triple[1], "ood_domain": triple[2],
                "auroc_w_primary": res["boot"][primary]["auroc"],
                "auroc_w_contrast": res["boot"][contrast]["auroc"],
                "delta": pair["delta"], "delta_ci": list(pair["ci"]), "p": pair["p"],
            })
        status = ("not_scored" if not fcells
                  else "incomplete" if len(fcells) < len(triples) else "scored")
        if status == "scored":
            for fc, p_adj in zip(fcells, holm([fc["p"] for fc in fcells])):
                fc["p_holm"] = float(p_adj)
                fc["decision"] = _decision(fc["delta"], fc["p_holm"], an["margin_auroc"],
                                           an["alpha"])
        else:
            for fc in fcells:
                fc["p_holm"] = None
        done = {(fc["score_set"], fc["id_domain"], fc["ood_domain"]) for fc in fcells}
        families[fname] = {"status": status, "m": len(triples), "cells": fcells,
                           "missing_cells": ["|".join(t) for t in triples if t not in done]}
    written["families"] = out / "families.json"
    _write_json(written["families"], {**stamp, "families": families})

    rep = an["descriptive"]["replication"]
    above = [
        {"cell": "|".join(t), "primary_auroc_w_ci_lo": cells[t]["boot"][primary]["auroc_ci"][0],
         "above_chance": cells[t]["boot"][primary]["auroc_ci"][0] > 0.5}
        for triples in tables.values() for t in triples
    ]
    replication = {}
    for s, old_point in rep["old_point"].items():
        res = cell_result((s, rep["id"], rep["ood"]))
        if res is None:
            replication[s] = {"status": "not_scored"}
            continue
        new_m, new_ci = res["boot"][contrast]["auroc"], list(res["boot"][contrast]["auroc_ci"])
        old_ci = rep["old_cluster_ci"][s]
        replication[s] = {
            "new_auroc_w_contrast": new_m, "new_ci": new_ci, "old_point": old_point,
            "old_cluster_ci": old_ci,
            "replicates": bool(old_ci[0] <= new_m <= old_ci[1]
                               and new_ci[0] <= old_point <= new_ci[1]),
        }
    written["descriptive"] = out / "descriptive.json"
    _write_json(written["descriptive"],
                {**stamp, "above_chance": above, "replication": replication})

    for fname, fam in families.items():
        print(f"{fname}: {fam['status']} ({len(fam['cells'])}/{fam['m']} cells)")
        for fc in fam["cells"]:
            p_holm = "n/a" if fc["p_holm"] is None else f"{fc['p_holm']:.4g}"
            print(f"  {fc['score_set']:<14} {fc['id_domain']:<14} {fc['ood_domain']:<16} "
                  f"delta {fc['delta']:+.4f} [{fc['delta_ci'][0]:+.4f}, "
                  f"{fc['delta_ci'][1]:+.4f}]  p {fc['p']:.4g}  p_holm {p_holm}")
    for path in written.values():
        print(f"Wrote {path}")
    return written
