"""Fit data of the S2 post-hoc refits: the cached Pile split, the manifest, the fit blocks.

Split out of ``minigpt.posthoc_refit``, which re-exports every name. Also holds the checks
that no fit block and no curvature token comes from an eval document.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import minigpt.data as data_mod
from minigpt.data import load_pile_data


class RefitCheckError(RuntimeError):
    """A refit safety check failed; ``scripts/refit_posthoc.py`` exits non-zero."""


# --------------------------------------------------------------------------------------------
# Data, manifest and fit blocks
# --------------------------------------------------------------------------------------------

@dataclass
class RefitData:
    """Token data of a refit, in the local positions of each ID domain's cached tensor.

    ``fit_tokens`` holds the fit domain's ``val`` slice; local position p is
    ``fit_tokens[p - fit_range[0]]``. ``train`` is ``data["train"]`` (the curvature set).
    """
    train: torch.Tensor
    fit_domain: str
    fit_tokens: torch.Tensor
    fit_range: tuple[int, int]
    train_ranges: dict[str, tuple[int, int]]
    val_ranges: dict[str, tuple[int, int]]


class _NoStreamTokenizer:
    """Stands in for the tokenizer: a refit reads the cached tensors and never re-streams."""

    def encode_ordinary(self, text: str) -> list[int]:
        raise RuntimeError("the refit tried to re-stream the Pile; a cached tensor is missing")


def load_refit_data(cfg: dict) -> RefitData:
    """Load the cached Pile split exactly as training did (``load_pile_data``).

    Reads only ``cfg["data"]`` and ``cfg["fit"]["fit_domain"]``. Every key is passed to
    ``load_pile_data`` explicitly (``data_seed`` as ``train.seed``). A missing cached tensor
    raises FileNotFoundError: the refit must use the tensors the models were trained on.
    """
    d = cfg["data"]
    if d["dataset"] != "pile":
        raise ValueError(f"data.dataset must be 'pile', got {d['dataset']!r}")
    id_domains = list(d["pile_id_domains"])
    per_domain = int(d["pile_id_tokens"])
    pile_dir = Path(data_mod.DATA_DIR) / "pile"
    needed = [f"{k}_{d['pile_id_tokens']}.pt" for k in id_domains]
    needed += [f"{k}_{d['pile_ood_tokens']}.pt" for k in d["pile_ood_domains"]]
    missing = [name for name in needed if not (pile_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"cached Pile tensors missing in {pile_dir}: {missing}")

    loader_cfg = {
        "data": {
            "dataset": "pile",
            "pile_id_domains": id_domains,
            "pile_ood_domains": list(d["pile_ood_domains"]),
            "pile_id_tokens": d["pile_id_tokens"],
            "pile_ood_tokens": d["pile_ood_tokens"],
            "val_fraction": d["val_fraction"],
            "test_fraction": d["test_fraction"],
        },
        "train": {"seed": d["data_seed"]},
    }
    splits = load_pile_data(loader_cfg, _NoStreamTokenizer())
    train, val, test_id = splits["train"], splits["val"], splits["test_id"]
    total = len(train) + len(val) + len(test_id)
    if total != len(id_domains) * per_domain:
        raise ValueError(f"the ID tensors hold {total} tokens, not "
                         f"{len(id_domains)} x {per_domain}; local positions are undefined")
    train_end, val_end = len(train), len(train) + len(val)

    train_ranges: dict[str, tuple[int, int]] = {}
    val_ranges: dict[str, tuple[int, int]] = {}
    for i, key in enumerate(id_domains):
        off = i * per_domain

        def local(pos: int, _off: int = off) -> int:
            return min(max(pos - _off, 0), per_domain)

        train_ranges[key] = (0, local(train_end))
        val_ranges[key] = (local(train_end), local(val_end))

    fit_domain = cfg["fit"]["fit_domain"]
    if fit_domain not in id_domains:
        raise ValueError(f"fit_domain {fit_domain!r} is not an ID domain {id_domains}")
    a, b = val_ranges[fit_domain]
    if b <= a:
        raise ValueError(f"the val split holds no {fit_domain} tokens")
    off = id_domains.index(fit_domain) * per_domain
    fit_tokens = val[off + a - train_end: off + b - train_end]
    return RefitData(train=train, fit_domain=fit_domain, fit_tokens=fit_tokens,
                     fit_range=(a, b), train_ranges=train_ranges, val_ranges=val_ranges)


def read_manifest(path: str | Path) -> tuple[list[dict], str]:
    """Read S1's ``manifest.jsonl``. Returns (rows, sha256 of the file bytes)."""
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, hashlib.sha256(raw).hexdigest()


def doc_ids_sha256(doc_ids: Iterable[str | None]) -> str:
    """SHA-256 of the sorted, unique document IDs, one per line."""
    ids = sorted({d for d in doc_ids if d is not None})
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _fit_rows(rows: list[dict], domain: str, split: str) -> list[dict]:
    return [r for r in rows if r["split"] == split and r["domain"] == domain
            and r.get("variant", "raw") == "raw"]


def _test_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["split"] == "test"]


def _test_intervals(test_rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray, list]]:
    by_domain: dict[str, list[dict]] = {}
    for r in test_rows:
        by_domain.setdefault(r["domain"], []).append(r)
    return {
        dom: (np.array([r["c_i"] for r in rs], dtype=np.int64),
              np.array([r["c_i"] + r["L_d"] for r in rs], dtype=np.int64),
              [r["doc_id"] for r in rs])
        for dom, rs in by_domain.items()
    }


def _first_overlap(intervals, domain: str, start: int, end: int) -> str | None:
    if domain not in intervals:
        return None
    ts, te, ids = intervals[domain]
    hit = np.nonzero((ts < end) & (te > start))[0]
    return ids[int(hit[0])] if hit.size else None


def check_fit_isolation(fit_rows: list[dict], test_rows: list[dict]) -> None:
    """Raise RefitCheckError("eval document in fit set ...") if a fit document is an eval one.

    A fit row fails when its ID is a test ID, or when its token range overlaps the range of
    a test document of the same domain.
    """
    test_ids = {r["doc_id"] for r in test_rows}
    shared = sorted({r["doc_id"] for r in fit_rows if r["doc_id"] is not None} & test_ids)
    if shared:
        raise RefitCheckError(f"eval document in fit set: {len(shared)} fit document ID(s) "
                              f"are also test IDs, e.g. {shared[0]}")
    intervals = _test_intervals(test_rows)
    for r in fit_rows:
        start = r["c_i"]
        hit = _first_overlap(intervals, r["domain"], start, start + r["L_d"])
        if hit is not None:
            name = r["doc_id"] if r["doc_id"] is not None else "the val token range"
            raise RefitCheckError(f"eval document in fit set: {name} "
                                  f"[{start}, {start + r['L_d']}) overlaps test document "
                                  f"{hit} in {r['domain']}")


def check_curvature_isolation(rows: list[dict], train_ranges: dict[str, tuple[int, int]]) -> None:
    """Raise RefitCheckError if a test document's tokens lie in the curvature (train) slice."""
    for r in _test_rows(rows):
        if r["domain"] not in train_ranges:
            continue
        a, b = train_ranges[r["domain"]]
        start, end = r["c_i"], r["c_i"] + r["L_d"]
        if a < b and start < b and end > a:
            raise RefitCheckError(f"eval document in curvature set: {r['doc_id']} "
                                  f"[{start}, {end}) overlaps the {r['domain']} train slice "
                                  f"[{a}, {b})")


@dataclass
class FitSet:
    """The fit blocks: ``blocks[i]`` = (doc_id or None, local start); x, y are (n, T)."""
    units: str
    domain: str
    blocks: list[tuple[str | None, int]]
    x: torch.Tensor
    y: torch.Tensor

    @property
    def doc_ids(self) -> list[str]:
        return sorted({d for d, _ in self.blocks if d is not None})

    def summary(self, cfg: dict) -> dict:
        f = cfg["fit"]
        blocks = [[d, int(s)] for d, s in self.blocks]
        return {
            "units": self.units,
            "domain": self.domain,
            "split": f["fit_split"],
            "n_blocks": len(blocks),
            "n_docs": len(self.doc_ids),
            "block_size": cfg["model"]["block_size"],
            "batch_size": f["batch_size"],
            "fit_seed": f["fit_seed"],
            "max_blocks_per_doc": f["max_blocks_per_doc"],
            "doc_ids_sha256": doc_ids_sha256(self.doc_ids) if self.doc_ids else None,
            "blocks_sha256": hashlib.sha256(json.dumps(blocks).encode("utf-8")).hexdigest(),
            "blocks": blocks,
        }


def build_fit_blocks(rows: list[dict], data: RefitData, cfg: dict) -> FitSet:
    """Draw ``n_fit_blocks`` non-overlapping T-token blocks from the fit domain's val documents.

    The documents are visited in a ``fit_seed`` permutation. Each gives
    k = min(max_blocks_per_doc, floor((L - 1) / T)) consecutive blocks from a uniform offset
    o in {0, ..., L - 1 - kT}, as in S1's block rule. If the manifest has no val rows for the
    domain (S1 withheld the IDs, P19), blocks are drawn uniformly from the val token range
    instead (``units: token_range``). Raises RefitCheckError when a fit document is an eval
    document, lies outside the val range, or when too few blocks exist.
    """
    f = cfg["fit"]
    T = cfg["model"]["block_size"]
    n = f["n_fit_blocks"]
    rng = np.random.default_rng(f["fit_seed"])
    a, b = data.fit_range
    fit_rows = _fit_rows(rows, f["fit_domain"], f["fit_split"])
    test_rows = _test_rows(rows)
    blocks: list[tuple[str | None, int]] = []
    if fit_rows:
        units = "document"
        check_fit_isolation(fit_rows, test_rows)
        outside = [r for r in fit_rows
                   if not (a <= r["c_i"] and r["c_i"] + r["L_d"] <= b)]
        if outside:
            raise RefitCheckError(f"val document {outside[0]['doc_id']} lies outside the "
                                  f"{data.fit_domain} val range [{a}, {b})")
        eligible = [r for r in fit_rows if r["L_d"] >= T + 1]
        for idx in rng.permutation(len(eligible)):
            r = eligible[int(idx)]
            k = min(f["max_blocks_per_doc"], (r["L_d"] - 1) // T, n - len(blocks))
            o = int(rng.integers(0, r["L_d"] - k * T))
            blocks += [(r["doc_id"], r["c_i"] + o + j * T) for j in range(k)]
            if len(blocks) == n:
                break
    else:
        units = "token_range"
        print(f"[refit] no val document IDs for {data.fit_domain}: drawing blocks from the "
              f"val token range [{a}, {b}) (fit_units: token_range)")
        pseudo = {"doc_id": None, "domain": data.fit_domain, "c_i": a, "L_d": b - a}
        check_fit_isolation([pseudo], test_rows)
        n_slots = (b - a - 1) // T
        if n_slots >= n:
            slots = np.sort(rng.choice(n_slots, size=n, replace=False))
            blocks = [(None, a + int(s) * T) for s in slots]
    if len(blocks) < n:
        raise RefitCheckError(f"only {len(blocks)} fit blocks available, n_fit_blocks={n}")
    x = torch.stack([data.fit_tokens[s - a: s - a + T] for _, s in blocks])
    y = torch.stack([data.fit_tokens[s - a + 1: s - a + T + 1] for _, s in blocks])
    return FitSet(units=units, domain=data.fit_domain, blocks=blocks, x=x, y=y)


def fit_batches(fit: FitSet, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split the fit blocks into equal batches, in draw order."""
    n = fit.x.shape[0]
    if n % batch_size:
        raise ValueError(f"{n} fit blocks do not split into batches of {batch_size}")
    return [(fit.x[i:i + batch_size], fit.y[i:i + batch_size]) for i in range(0, n, batch_size)]


def check_fit_blocks(fit: FitSet, rows: list[dict], data: RefitData, cfg: dict) -> bool:
    """End-of-run check (S2-T2c): every block read lies inside a val document of the fit
    domain (or the val range), overlaps no test document, and no fit ID is a test ID."""
    T = cfg["model"]["block_size"]
    a, b = data.fit_range
    val_by_id = {r["doc_id"]: r for r in _fit_rows(rows, fit.domain, cfg["fit"]["fit_split"])}
    test_rows = _test_rows(rows)
    test_ids = {r["doc_id"] for r in test_rows}
    intervals = _test_intervals(test_rows)
    per_doc: dict[str, int] = {}
    for doc_id, start in fit.blocks:
        end = start + T + 1
        if not (a <= start and end <= b):
            return False
        if _first_overlap(intervals, fit.domain, start, end) is not None:
            return False
        if fit.units == "document":
            r = val_by_id.get(doc_id)
            if r is None or doc_id in test_ids:
                return False
            if not (r["c_i"] <= start and end <= r["c_i"] + r["L_d"]):
                return False
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
    if per_doc and max(per_doc.values()) > cfg["fit"]["max_blocks_per_doc"]:
        return False
    starts = sorted(s for _, s in fit.blocks)
    return all(nxt - cur >= T for cur, nxt in zip(starts, starts[1:]))
