"""S2 post-hoc refits on held-out ID data (specs/i2-posthoc-fixes.md, Sections 3 and 5.5).

Two methods, one YAML each (``scripts/refit_posthoc.py --config ...``):

- ``tfb``: the TFB sigma_q search (``minigpt.tfb.fit_tfb``) with the relative tolerance
  |l_bar - l_0| <= epsilon_rel * l_0 on ``n_fit_blocks`` blocks drawn from S1's ``split: val``
  documents, then Delta-NLL(sigma_q*) re-measured at ``n_delta_samples`` seeds. The TFB
  budget is rho_TFB = Delta-NLL_TFB / l_0.
- ``laplace``: the diagonal empirical Fisher F_hat on ``data["train"]`` (fp32, no autocast),
  scaled to tau = N_seq T^2 F_hat + lambda (``minigpt.laplace.scale_laplace_state``), and
  lambda swept over a grid (one value per decade) and log-bisected until
  Delta-NLL(lambda) = rho_TFB * l_0 within ``match_tol`` (``match_prior_precision``).

Rules shared by both: every key is read as ``cfg[...]`` and checked first (``check_config``);
one weight draw per (sigma_q or lambda, seed), reused for every fit block; the same seeds
0..S-1 at every value; no fit block may come from an eval (``split: test``) document; each
fit records the SHA-256 of the base checkpoint bytes it loaded. Outputs go to ``out_dir``
(``data/checkpoints/i2_posthoc/<cell>/``): the state file and ``fit_record.json``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import minigpt.data as data_mod
from minigpt.data import load_pile_data
from minigpt.laplace import (
    LaplaceState,
    apply_sampled_params,
    fit_laplace,
    load_laplace_state,
    sample_laplace_params,
    save_laplace_state,
    scale_laplace_state,
    select_params,
)
from minigpt.layers import BayesConfig
from minigpt.lora import LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT
from minigpt.tfb import (
    SAMPLER_VERSIONS,
    TFBState,
    fit_tfb,
    load_tfb_state,
    sample_tfb_params,
    save_tfb_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_VERSION = 1
METHODS = ("tfb", "laplace")
BASE_KINDS = ("full", "blob_mean", "det_lora")
AUTOCAST_MODES = ("none", "fp16")

EXIT_OK = 0
EXIT_CHECK_FAILED = 2
EXIT_BISECTION_FAILED = 3
EXIT_NO_BUDGET = 4
MATCH_EXIT_CODES = {
    "matched": EXIT_OK,
    "matched_noisy": EXIT_OK,
    "tighter_than_budget": EXIT_OK,
    "not_matched": EXIT_OK,
    "bisection_failed": EXIT_BISECTION_FAILED,
}
SCORED_STATUSES = ("matched", "matched_noisy", "tighter_than_budget")
SCORE_SET_CELLS = {"c2_refit": "c2", "c4_lap_refit": "c4_lap", "c4_tfb_fixed": "c4_tfb"}
PROTECTED_CKPT_DIRS = ("c0", "c1", "c2", "c3", "c4_tfb", "c4_lap", "b3_lora")
CURVATURE_MEDIAN_RATIO = (0.5, 2.0)

REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "": ("method", "cell", "out_dir", "device", "autocast", "base", "model", "data", "fit"),
    "base": ("base_checkpoint", "base_kind"),
    "model": ("block_size", "n_layer", "n_head", "n_embd", "dropout", "bias"),
    "lora": ("rank", "alpha", "target"),
    "data": ("dataset", "pile_id_domains", "pile_ood_domains", "pile_id_tokens",
             "pile_ood_tokens", "val_fraction", "test_fraction", "data_seed"),
    "fit": ("manifest_path", "fit_split", "fit_domain", "n_fit_blocks", "max_blocks_per_doc",
            "fit_seed", "batch_size"),
    "tfb": ("sampler_version", "epsilon_rel", "n_search_samples", "search_min", "search_max",
            "search_precision", "n_delta_samples", "se_threshold", "se_max_doublings"),
    "laplace": ("selection_mode", "n_curvature_batches", "curvature_batch_size",
                "curvature_seed", "n_data_seqs", "tokens_per_seq", "prior_prec_grid",
                "grid_extension", "bisect_max_steps", "match_tol", "tfb_record_path",
                "n_delta_samples", "se_threshold", "se_max_doublings", "reference_state_path"),
}
_INT_KEYS = {
    "model": ("block_size", "n_layer", "n_head", "n_embd"),
    "lora": ("rank",),
    "data": ("pile_id_tokens", "pile_ood_tokens", "data_seed"),
    "fit": ("n_fit_blocks", "max_blocks_per_doc", "fit_seed", "batch_size"),
    "tfb": ("n_search_samples", "n_delta_samples", "se_max_doublings"),
    "laplace": ("n_curvature_batches", "curvature_batch_size", "curvature_seed", "n_data_seqs",
                "tokens_per_seq", "bisect_max_steps", "n_delta_samples", "se_max_doublings"),
}
_NUMBER_KEYS = {
    "model": ("dropout",),
    "lora": ("alpha",),
    "data": ("val_fraction", "test_fraction"),
    "tfb": ("epsilon_rel", "search_min", "search_max", "search_precision", "se_threshold"),
    "laplace": ("match_tol", "se_threshold"),
}


class RefitCheckError(RuntimeError):
    """A refit safety check failed; ``scripts/refit_posthoc.py`` exits non-zero."""


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------

def _require(section: dict, name: str, keys: Iterable[str]) -> None:
    if not isinstance(section, dict):
        raise KeyError(f"config section {name!r} must be a mapping")
    missing = [k for k in keys if k not in section]
    if missing:
        prefix = f"{name}." if name else ""
        raise KeyError("missing config key(s): " + ", ".join(prefix + k for k in missing))


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _number_error(path: str, v) -> ValueError:
    return ValueError(
        f"config key {path} must be a number, got {type(v).__name__} {v!r} "
        "(PyYAML reads 1e-4 without a dot as a string; write 1.0e-4)"
    )


def check_config(cfg: dict) -> None:
    """Check every key the refit reads, before any helper runs.

    Raises KeyError for a missing key and ValueError for a wrong type or value.
    """
    _require(cfg, "", REQUIRED_KEYS[""])
    if cfg["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {cfg['method']!r}")
    _require(cfg["base"], "base", REQUIRED_KEYS["base"])
    kind = cfg["base"]["base_kind"]
    if kind not in BASE_KINDS:
        raise ValueError(f"base.base_kind must be one of {BASE_KINDS}, got {kind!r}")
    sections = ["model", "data", "fit"] + (["lora"] if kind != "full" else []) + [cfg["method"]]
    _require(cfg, "", sections)
    for name in sections:
        _require(cfg[name], name, REQUIRED_KEYS[name])

    for name in sections:
        for key in _INT_KEYS.get(name, ()):
            if not _is_int(cfg[name][key]):
                raise ValueError(f"config key {name}.{key} must be an integer, "
                                 f"got {cfg[name][key]!r}")
        for key in _NUMBER_KEYS.get(name, ()):
            if not _is_number(cfg[name][key]):
                raise _number_error(f"{name}.{key}", cfg[name][key])
    if not isinstance(cfg["model"]["bias"], bool):
        raise ValueError("config key model.bias must be true or false")
    if cfg["autocast"] not in AUTOCAST_MODES:
        raise ValueError(f"autocast must be one of {AUTOCAST_MODES}, got {cfg['autocast']!r}")
    if cfg["device"] not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {cfg['device']!r}")

    d, f = cfg["data"], cfg["fit"]
    if d["dataset"] != "pile":
        raise ValueError(f"data.dataset must be 'pile', got {d['dataset']!r}")
    if f["fit_split"] != "val":
        raise ValueError(f"fit.fit_split must be 'val', got {f['fit_split']!r}")
    if f["fit_domain"] not in d["pile_id_domains"]:
        raise ValueError(f"fit.fit_domain {f['fit_domain']!r} is not in data.pile_id_domains")
    if f["n_fit_blocks"] <= 0 or f["n_fit_blocks"] % f["batch_size"] != 0:
        raise ValueError("fit.n_fit_blocks must be a positive multiple of fit.batch_size")
    if f["max_blocks_per_doc"] < 1:
        raise ValueError("fit.max_blocks_per_doc must be at least 1")

    if cfg["method"] == "tfb":
        t = cfg["tfb"]
        if t["sampler_version"] not in SAMPLER_VERSIONS:
            raise ValueError(f"tfb.sampler_version must be one of {SAMPLER_VERSIONS}")
        if not 0 < t["search_min"] < t["search_max"]:
            raise ValueError("tfb needs 0 < search_min < search_max")
        if not 0 < t["search_precision"] or not 0 < t["epsilon_rel"]:
            raise ValueError("tfb.search_precision and tfb.epsilon_rel must be positive")
    else:
        lap = cfg["laplace"]
        for key in ("prior_prec_grid", "grid_extension"):
            values = lap[key]
            if not isinstance(values, list) or not all(_is_number(v) for v in values):
                raise _number_error(f"laplace.{key}", values)
        if not lap["prior_prec_grid"]:
            raise ValueError("laplace.prior_prec_grid is empty")
        if lap["tokens_per_seq"] != cfg["model"]["block_size"]:
            raise ValueError("laplace.tokens_per_seq must equal model.block_size (the curvature "
                             "windows are block_size tokens long)")


# --------------------------------------------------------------------------------------------
# Base checkpoint
# --------------------------------------------------------------------------------------------

def load_posthoc_base(cfg: dict, device: str | torch.device = "cpu") -> tuple[MiniGPT, dict]:
    """Load the base model named by ``cfg["base"]`` and hash the bytes that were loaded.

    ``base_kind``: ``full`` (a MiniGPT, keys one to one), ``blob_mean`` (a BLoB checkpoint
    loaded into a deterministic LoRA: ``.lora_A`` <- ``.lora_A_mu``; the ``.lora_A_g`` keys
    are dropped and listed) or ``det_lora`` (a deterministic LoRA, keys one to one). Every
    target key must be in the checkpoint and every checkpoint key must be used (or dropped,
    for ``blob_mean``); otherwise KeyError. The vocabulary size is read from the checkpoint.

    Returns:
        (model in eval mode on ``device``, info dict with path, kind, sha256, n_bytes,
        dropped_keys, vocab_size and step).
    """
    path = Path(cfg["base"]["base_checkpoint"])
    kind = cfg["base"]["base_kind"]
    if kind not in BASE_KINDS:
        raise ValueError(f"base_kind must be one of {BASE_KINDS}, got {kind!r}")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    n_bytes = len(raw)
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    del raw
    sd = ckpt["model_state_dict"]
    if "token_emb.weight" not in sd:
        raise KeyError(f"base checkpoint {path} has no token_emb.weight")

    m = cfg["model"]
    vocab_size = int(sd["token_emb.weight"].shape[0])
    model = MiniGPT(GPTConfig(
        vocab_size=vocab_size,
        block_size=m["block_size"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        n_embd=m["n_embd"],
        dropout=m["dropout"],
        bias=m["bias"],
        bayes_head=BayesConfig(enabled=False),
        bayes_ffn=BayesConfig(enabled=False),
        bayes_attn_v=BayesConfig(enabled=False),
    ))
    if kind != "full":
        lora = cfg["lora"]
        inject_lora(model, LoRAConfig(rank=lora["rank"], alpha=float(lora["alpha"]),
                                      target=lora["target"]), bayesian=False)

    mapped: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    used: set[str] = set()
    for key in model.state_dict():
        src = key
        if kind == "blob_mean" and key.endswith(".lora_A"):
            src = key[: -len(".lora_A")] + ".lora_A_mu"
        if src not in sd:
            missing.append(src)
            continue
        mapped[key] = sd[src]
        used.add(src)
    if missing:
        raise KeyError(f"base checkpoint {path} ({kind}) lacks {len(missing)} key(s): "
                       f"{missing[:5]}")
    leftover = sorted(set(sd) - used)
    dropped = [k for k in leftover if kind == "blob_mean" and k.endswith(".lora_A_g")]
    unexpected = [k for k in leftover if k not in dropped]
    if unexpected:
        raise KeyError(f"base checkpoint {path} ({kind}) has {len(unexpected)} unexpected "
                       f"key(s): {unexpected[:5]}")
    model.load_state_dict(mapped, strict=True)
    model.to(device)
    model.eval()
    step = ckpt.get("step")
    info = {
        "path": str(path),
        "kind": kind,
        "sha256": sha,
        "n_bytes": n_bytes,
        "dropped_keys": dropped,
        "vocab_size": vocab_size,
        "step": int(step) if _is_int(step) else None,
    }
    return model, info


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


# --------------------------------------------------------------------------------------------
# Delta-NLL
# --------------------------------------------------------------------------------------------

def autocast_context(mode: str, device: torch.device) -> contextlib.AbstractContextManager:
    """``none`` or ``fp16`` (CUDA only), as in the scorer."""
    if mode == "none":
        return contextlib.nullcontext()
    if mode == "fp16":
        if device.type != "cuda":
            raise ValueError("autocast fp16 needs a CUDA device")
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"autocast must be one of {AUTOCAST_MODES}, got {mode!r}")


def _token_logprobs(model: nn.Module, batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
                    autocast: str) -> torch.Tensor:
    """log p(y_t | x_<t) of every fit token, flat, float32."""
    device = next(model.parameters()).device
    out = []
    with autocast_context(autocast, device):
        for x, y in batches:
            logits, _ = model(x.to(device))
            nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                  y.to(device).reshape(-1), reduction="none")
            out.append(-nll)
    return torch.cat(out)


@torch.no_grad()
def map_nll(model: nn.Module, batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
            autocast: str) -> float:
    """l_0: token-mean NLL of the unperturbed model on the fit blocks."""
    was_training = model.training
    model.eval()
    try:
        return -_token_logprobs(model, batches, autocast).double().mean().item()
    finally:
        model.train(was_training)


@torch.no_grad()
def measure_delta_nll(
    model: nn.Module,
    sample_fn: Callable[[int], dict[str, torch.Tensor]],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    seeds: Iterable[int],
    *,
    anchor_loss: float | None = None,
    autocast: str = "none",
) -> dict:
    """Delta-NLL = mean_s l(D | theta_s) - l_0, looping sample-outer (one draw per seed).

    For each seed, ``sample_fn(seed)`` is called once and its parameters are applied to every
    batch. SE = sd_s(l_s) / sqrt(S). ``ell_bma`` is the NLL of the sample-averaged
    probability of the realized tokens (logged only). ``anchor_loss`` defaults to
    ``map_nll`` on the same batches.
    """
    was_training = model.training
    model.eval()
    try:
        ell0 = map_nll(model, batches, autocast) if anchor_loss is None else float(anchor_loss)
        seeds = [int(s) for s in seeds]
        per_sample: list[float] = []
        lse = None
        for s in seeds:
            params = sample_fn(s)
            with apply_sampled_params(model, params):
                lp = _token_logprobs(model, batches, autocast)
            per_sample.append(-lp.double().mean().item())
            lse = lp.clone() if lse is None else torch.logaddexp(lse, lp)
    finally:
        model.train(was_training)
    n = len(per_sample)
    if n == 0:
        raise ValueError("measure_delta_nll needs at least one seed")
    mean_nll = sum(per_sample) / n
    se = statistics.stdev(per_sample) / math.sqrt(n) if n > 1 else float("nan")
    ell_bma = -(lse.double() - math.log(n)).mean().item()
    return {
        "n_samples": n,
        "seeds": seeds,
        "ell0": ell0,
        "mean_nll": mean_nll,
        "delta_nll": mean_nll - ell0,
        "se": se,
        "ell_bma": ell_bma,
        "per_sample_nll": per_sample,
    }


# --------------------------------------------------------------------------------------------
# Prior-precision match (pure)
# --------------------------------------------------------------------------------------------

def match_prior_precision(
    delta_fn: Callable[[float, int], tuple[float, float]],
    target: float,
    grid: Sequence[float],
    extension: Sequence[float],
    bisect_max_steps: int,
    match_tol: float,
    se_threshold: float,
    *,
    n_samples: int,
    se_max_doublings: int,
) -> dict:
    """Find lambda with |Delta-NLL(lambda) - target| <= match_tol * target.

    ``delta_fn(lam, n_samples)`` returns (Delta-NLL, SE). The grid is evaluated in full, then
    ``extension`` (only while Delta-NLL stays above the band). The first grid value inside
    the band, or else the first bracket (above the band, then below it) from the small-lambda
    side, is used; a bracket is log-bisected (lambda_mid = sqrt(lo * hi)) for at most
    ``bisect_max_steps`` steps, stopping at the first value inside the band. At the chosen
    value, SE > se_threshold * target triggers a re-measure at twice the samples, at most
    ``se_max_doublings`` times.

    Status: ``matched``; ``matched_noisy`` (matched at ``n_samples``, but after the doublings
    the SE is still above the threshold or Delta-NLL left the band); ``tighter_than_budget``
    (Delta-NLL(grid[0]) below the band; lambda* = grid[0]); ``not_matched`` (still above the
    band at the last extension value); ``bisection_failed`` (a bracket, but no step inside the
    band). ``exit_code`` is ``MATCH_EXIT_CODES[status]``.
    """
    if not target > 0:
        raise ValueError(f"target must be positive, got {target!r}")
    lams = [float(v) for v in grid]
    ext = [float(v) for v in extension]
    if not lams or any(v <= 0 for v in lams + ext):
        raise ValueError("grid and extension values must be positive")
    if any(b <= a for a, b in zip(lams + ext, (lams + ext)[1:])):
        raise ValueError("grid followed by extension must be strictly increasing")
    lo_band, hi_band = (1.0 - match_tol) * target, (1.0 + match_tol) * target
    evaluations: list[dict] = []

    def evaluate(lam: float, n: int, stage: str) -> tuple[float, float]:
        d, se = delta_fn(lam, n)
        d, se = float(d), float(se)
        evaluations.append({"lam": lam, "n_samples": n, "delta_nll": d, "se": se, "stage": stage})
        return d, se

    def where(d: float) -> str:
        return "in" if lo_band <= d <= hi_band else ("above" if d > hi_band else "below")

    points = [(lam, *evaluate(lam, n_samples, "grid")) for lam in lams]
    result = {
        "target": target, "match_tol": match_tol, "se_threshold": se_threshold,
        "status": None, "exit_code": None, "lambda_star": None, "delta_nll": None, "se": None,
        "n_samples": None, "delta_nll_first": None, "se_first": None, "n_doublings": 0,
        "n_bisect_steps": 0, "bracket": None, "final_bracket": None, "monotone": None,
        "evaluations": evaluations,
    }

    def finish(status: str) -> dict:
        deltas = [d for _, d, _ in points]
        result["monotone"] = all(b <= a for a, b in zip(deltas, deltas[1:]))
        result["status"] = status
        result["exit_code"] = MATCH_EXIT_CODES[status]
        return result

    if where(points[0][1]) == "below":
        lam, d, se = points[0]
        result.update(lambda_star=lam, delta_nll=d, se=se, n_samples=n_samples)
        return finish("tighter_than_budget")

    def scan(start: int):
        for i in range(start, len(points)):
            if where(points[i][1]) == "in":
                return ("in", i)
            if (i + 1 < len(points) and where(points[i][1]) == "above"
                    and where(points[i + 1][1]) == "below"):
                return ("bracket", i)
        return None

    found = scan(0)
    for lam in ext:
        if found is not None or where(points[-1][1]) != "above":
            break
        points.append((lam, *evaluate(lam, n_samples, "extension")))
        found = scan(len(points) - 2)
    if found is None:
        lam, d, se = points[-1]
        result.update(delta_nll=d, se=se, n_samples=n_samples)
        return finish("not_matched")

    kind, i = found
    if kind == "in":
        chosen = points[i]
    else:
        lo, hi = points[i][0], points[i + 1][0]
        result["bracket"] = [lo, hi]
        chosen = None
        for _ in range(bisect_max_steps):
            mid = math.sqrt(lo * hi)
            d, se = evaluate(mid, n_samples, "bisect")
            result["n_bisect_steps"] += 1
            if where(d) == "in":
                chosen = (mid, d, se)
                break
            if where(d) == "above":
                lo = mid
            else:
                hi = mid
        result["final_bracket"] = [lo, hi]
        if chosen is None:
            return finish("bisection_failed")

    lam, d, se = chosen
    result.update(lambda_star=lam, delta_nll_first=d, se_first=se)
    n = n_samples
    while se > se_threshold * target and result["n_doublings"] < se_max_doublings:
        n *= 2
        d, se = evaluate(lam, n, "se_doubling")
        result["n_doublings"] += 1
    result.update(delta_nll=d, se=se, n_samples=n)
    ok = se <= se_threshold * target and where(d) == "in"
    return finish("matched" if ok else "matched_noisy")


# --------------------------------------------------------------------------------------------
# Records and checks
# --------------------------------------------------------------------------------------------

def _git_state() -> tuple[str | None, bool | None]:
    """HEAD commit and dirty flag; read-only (no index refresh)."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True, timeout=60, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                                timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):
        return None, None
    return head.stdout.strip(), bool(status.stdout.strip())


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fit_record(path: str | Path, record: dict) -> Path:
    """Write ``record`` as JSON (atomic replace). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def check_search_log(log: list[dict], sigma_star: float, sampler_version: str) -> bool:
    """S2-T2a: every step is logged; sigma_q* is the largest accepted step and every rejected
    step lies above it; every step ran the configured sampler."""
    fields = ("sigma_q", "avg_loss", "anchor_loss", "accepted", "sampler_version")
    if not log or not all(all(k in step for k in fields) for step in log):
        return False
    accepted = [s["sigma_q"] for s in log if s["accepted"]]
    rejected = [s["sigma_q"] for s in log if not s["accepted"]]
    return (bool(accepted) and sigma_star == max(accepted)
            and all(s > sigma_star for s in rejected)
            and all(s["sampler_version"] == sampler_version for s in log))


def check_precheck(log: list[dict], search_max: float) -> bool:
    """S2-T2b: the first step is the pre-check at search_max, and it was rejected."""
    return (bool(log) and log[0].get("stage") == "precheck"
            and log[0]["sigma_q"] == search_max and log[0]["accepted"] is False)


def check_scores(score_paths: Sequence[str | Path], record_paths: Sequence[str | Path]
                 ) -> list[str]:
    """S2-T4d: each S1 score file records the base checkpoint SHA-256 of its fit record.

    The score set picks the record through ``SCORE_SET_CELLS``; ``meta.checkpoint_sha256``
    may be a dict (per file), a list or one string. Returns the failures (empty = pass).
    """
    records = {}
    for p in record_paths:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
        records[rec["cell"]] = rec
    failures = []
    for p in score_paths:
        meta = torch.load(p, map_location="cpu", weights_only=False)["meta"]
        score_set = meta.get("score_set")
        cell = SCORE_SET_CELLS.get(score_set)
        if cell is None or cell not in records:
            failures.append(f"{p}: no fit record for score set {score_set!r}")
            continue
        shas = meta.get("checkpoint_sha256")
        if isinstance(shas, dict):
            values = set(shas.values())
        elif isinstance(shas, (list, tuple)):
            values = set(shas)
        else:
            values = {shas}
        base_sha = records[cell]["base"]["sha256"]
        if base_sha not in values:
            failures.append(f"{p} ({score_set}): the fit record's base sha256 {base_sha[:12]} "
                            "is not in meta.checkpoint_sha256")
    return failures


def _check_out_dir(cfg: dict) -> Path:
    out_dir = Path(cfg["out_dir"]).resolve()
    protected = {Path(cfg["base"]["base_checkpoint"]).resolve().parent}
    protected |= {(REPO_ROOT / "data" / "checkpoints" / name).resolve()
                  for name in PROTECTED_CKPT_DIRS}
    if cfg["method"] == "laplace":
        protected.add(Path(cfg["laplace"]["reference_state_path"]).resolve().parent)
    if out_dir in protected:
        raise RefitCheckError(f"out_dir {out_dir} is a read-only input folder")
    return Path(cfg["out_dir"])


def _read_tfb_budget(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    rec = json.loads(raw.decode("utf-8"))
    if rec.get("method") != "tfb":
        raise RefitCheckError(f"{path} is not a TFB fit record")
    if rec.get("exit_code") != EXIT_OK or rec.get("failures"):
        raise RefitCheckError(f"the TFB fit record {path} did not pass (exit_code "
                              f"{rec.get('exit_code')}, failures {rec.get('failures')})")
    rho = rec["rho_tfb"]
    if not rho > 0:
        raise RefitCheckError(f"the TFB fit record {path} has no budget (rho_tfb={rho})")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "cell": rec["cell"],
        "rho_tfb": rho,
        "sigma_q_star": rec["tfb"]["sigma_q_star"],
        "delta_nll_tfb": rec["delta_nll_tfb"],
        "ell0": rec["ell0"],
        "base_sha256": rec["base"]["sha256"],
        "blocks_sha256": rec["fit_set"]["blocks_sha256"],
    }


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device: cuda, but CUDA is not available")
    return torch.device(name)


def _measure_with_se_rule(measure: Callable[[int], dict], n_samples: int, se_threshold: float,
                          se_max_doublings: int) -> list[dict]:
    """Measure at n_samples; while SE > se_threshold * Delta-NLL, re-measure at twice the
    samples, at most se_max_doublings times. Returns every measurement, in order."""
    out = [measure(n_samples)]
    while (out[-1]["se"] > se_threshold * out[-1]["delta_nll"]
           and len(out) - 1 < se_max_doublings):
        out.append(measure(2 * out[-1]["n_samples"]))
    return out


def _tfb_state_cpu(state: TFBState) -> TFBState:
    return TFBState(
        sigma_q=state.sigma_q,
        svd_cache={k: tuple(t.detach().cpu() for t in v) for k, v in state.svd_cache.items()},
        a_map={k: v.detach().cpu() for k, v in state.a_map.items()},
        param_names=list(state.param_names),
        epsilon=state.epsilon,
        anchor_loss=state.anchor_loss,
        epsilon_rel=state.epsilon_rel,
        search_log=list(state.search_log),
        sampler_version=state.sampler_version,
    )


def _laplace_state_cpu(state: LaplaceState) -> LaplaceState:
    return LaplaceState(
        param_names=list(state.param_names),
        phi_hat={k: v.detach().cpu() for k, v in state.phi_hat.items()},
        curvature={k: v.detach().cpu() for k, v in state.curvature.items()},
        damping=state.damping,
        sample_scale=state.sample_scale,
        sampler_version=state.sampler_version,
        n_data_seqs=state.n_data_seqs,
        tokens_per_seq=state.tokens_per_seq,
        prior_prec=state.prior_prec,
        base_checkpoint=state.base_checkpoint,
        base_sha256=state.base_sha256,
    )


def _flat_median(tensors: Iterable[torch.Tensor]) -> tuple[float, int, int]:
    """(median, number of entries, number of entries <= 0); torch.median, not quantile."""
    flat = torch.cat([t.detach().flatten().float() for t in tensors])
    return torch.median(flat).item(), flat.numel(), int((flat <= 0).sum().item())


# --------------------------------------------------------------------------------------------
# The two refits
# --------------------------------------------------------------------------------------------

def _run_tfb(cfg: dict, model: MiniGPT, batches, record: dict, out_dir: Path,
             device: torch.device, fit_ok: bool) -> int:
    t = cfg["tfb"]
    T = cfg["model"]["block_size"]
    steps = math.ceil(math.log2((t["search_max"] - t["search_min"]) / t["search_precision"]))
    with autocast_context(cfg["autocast"], device):
        state = fit_tfb(
            model, None,
            block_size=T,
            batch_size=cfg["fit"]["batch_size"],
            n_batches=len(batches),
            n_search_samples=t["n_search_samples"],
            search_range=(float(t["search_min"]), float(t["search_max"])),
            search_precision=float(t["search_precision"]),
            max_iterations=max(steps, 0) + 1,  # the precision, not this cap, ends the search
            sampler_version=t["sampler_version"],
            epsilon_rel=float(t["epsilon_rel"]),
            anchor_batches=batches,
        )
    ell0 = map_nll(model, batches, cfg["autocast"])

    def measure(n: int) -> dict:
        return measure_delta_nll(model, lambda s: sample_tfb_params(state, seed=s), batches,
                                 range(n), anchor_loss=ell0, autocast=cfg["autocast"])

    measurements = _measure_with_se_rule(measure, t["n_delta_samples"], t["se_threshold"],
                                         t["se_max_doublings"])
    final = measurements[-1]
    delta = final["delta_nll"]
    rho = delta / ell0

    state_path = out_dir / "tfb_state.pt"
    save_tfb_state(_tfb_state_cpu(state), state_path)
    readback = load_tfb_state(state_path, map_location="cpu").sampler_version

    checks = {
        "t2a_search_log": check_search_log(state.search_log, state.sigma_q,
                                           t["sampler_version"]),
        "t2b_precheck": check_precheck(state.search_log, float(t["search_max"])),
        "t2c_fit_isolation": fit_ok,
    }
    failures = [name for name, ok in checks.items() if not ok]
    if readback != t["sampler_version"]:
        failures.append(f"state reads back as {readback!r}")
    if failures:
        exit_code = EXIT_CHECK_FAILED
    elif not rho > 0:
        exit_code = EXIT_NO_BUDGET
    else:
        exit_code = EXIT_OK
    record.update({
        "sampler_version": t["sampler_version"],
        "tfb": {
            "epsilon_rel": t["epsilon_rel"],
            "tolerance": t["epsilon_rel"] * state.anchor_loss,
            "anchor_loss_search": state.anchor_loss,
            "n_search_samples": t["n_search_samples"],
            "search_min": t["search_min"],
            "search_max": t["search_max"],
            "search_precision": t["search_precision"],
            "sigma_q_star": state.sigma_q,
            "search_log": state.search_log,
        },
        "ell0": ell0,
        "delta_measurements": measurements,
        "delta_nll_tfb": delta,
        "se_tfb": final["se"],
        "se_ok": final["se"] <= t["se_threshold"] * delta,
        "rho_tfb": rho,
        "seeds": {
            "fit_seed": cfg["fit"]["fit_seed"],
            "search_seeds": list(range(t["n_search_samples"])),
            "delta_seeds": measurements[0]["seeds"],
            "final_seeds": final["seeds"],
        },
        "state_path": str(state_path),
        "state_sha256": _file_sha256(state_path),
        "state_readback_version": readback,
        "checks": checks,
        "failures": failures,
        "exit_code": exit_code,
    })
    print(f"[refit tfb] sigma_q*={state.sigma_q:.6g} l0={ell0:.5f} "
          f"dNLL={delta:.5g} (SE {final['se']:.3g}, S={final['n_samples']}) rho={rho:.5g}")
    return exit_code


def _laplace_prechecks(cfg: dict, rows: list[dict], data: RefitData) -> dict:
    lap = cfg["laplace"]
    T = lap["tokens_per_seq"]
    expected = len(data.train) // T
    if lap["n_data_seqs"] != expected:
        raise RefitCheckError(f"laplace.n_data_seqs={lap['n_data_seqs']} but "
                              f"len(data['train']) // tokens_per_seq = {expected}")
    check_curvature_isolation(rows, data.train_ranges)
    return _read_tfb_budget(lap["tfb_record_path"])


def _run_laplace(cfg: dict, model: MiniGPT, batches, data: RefitData, record: dict,
                 out_dir: Path, tfb_budget: dict, fit_ok: bool) -> int:
    lap = cfg["laplace"]
    T = lap["tokens_per_seq"]
    base = record["base"]
    selection = select_params(model, lap["selection_mode"])
    if not selection:
        raise ValueError(f"selection_mode {lap['selection_mode']!r} selects no parameter")

    torch.manual_seed(lap["curvature_seed"])  # get_batch draws from the global RNG
    v1 = fit_laplace(model, data.train, block_size=T,
                     batch_size=lap["curvature_batch_size"], selection=selection,
                     n_batches=lap["n_curvature_batches"],
                     damping=0.0,  # unused: the v2 precision is N T^2 F_hat + lambda
                     sample_scale=1.0)
    median, n_entries, n_nonpos = _flat_median(v1.curvature.values())
    ref = load_laplace_state(lap["reference_state_path"], map_location="cpu")
    ref_median, _, _ = _flat_median(ref.curvature.values())
    del ref
    ratio = median / ref_median if ref_median > 0 else float("inf")

    ell0 = map_nll(model, batches, cfg["autocast"])
    rho = tfb_budget["rho_tfb"]
    target = rho * ell0
    measurements: list[dict] = []

    def scaled(lam: float) -> LaplaceState:
        return scale_laplace_state(v1, n_data_seqs=lap["n_data_seqs"], tokens_per_seq=T,
                                   prior_prec=lam, base_checkpoint=base["path"],
                                   base_sha256=base["sha256"])

    def delta_fn(lam: float, n: int) -> tuple[float, float]:
        state = scaled(lam)
        m = measure_delta_nll(model, lambda s: sample_laplace_params(state, seed=s), batches,
                              range(n), anchor_loss=ell0, autocast=cfg["autocast"])
        measurements.append(m)
        print(f"  [lambda {lam:.4g}, S={n}] dNLL={m['delta_nll']:.5g} (SE {m['se']:.3g}; "
              f"target {target:.5g})")
        return m["delta_nll"], m["se"]

    match = match_prior_precision(
        delta_fn, target, lap["prior_prec_grid"], lap["grid_extension"],
        lap["bisect_max_steps"], lap["match_tol"], lap["se_threshold"],
        n_samples=lap["n_delta_samples"], se_max_doublings=lap["se_max_doublings"],
    )
    sweep = [{**ev, "seeds": m["seeds"], "ell0": m["ell0"], "mean_nll": m["mean_nll"],
              "ell_bma": m["ell_bma"], "per_sample_nll": m["per_sample_nll"]}
             for ev, m in zip(match.pop("evaluations"), measurements)]

    state_path = None
    readback = None
    if match["status"] in SCORED_STATUSES:
        state_path = out_dir / "laplace_state.pt"
        save_laplace_state(_laplace_state_cpu(scaled(match["lambda_star"])), state_path)
        readback = load_laplace_state(state_path, map_location="cpu").sampler_version

    lo, hi = CURVATURE_MEDIAN_RATIO
    checks = {
        "t2c_fit_isolation": fit_ok,
        "curvature_median_vs_reference": lo <= ratio <= hi,
    }
    failures = [name for name, ok in checks.items() if not ok]
    if state_path is not None and readback != "v2":
        failures.append(f"state reads back as {readback!r}")
    exit_code = EXIT_CHECK_FAILED if failures else MATCH_EXIT_CODES[match["status"]]
    same_fit = (tfb_budget["base_sha256"] == base["sha256"]
                and tfb_budget["blocks_sha256"] == record["fit_set"]["blocks_sha256"])
    doubled = [e["seeds"] for e in sweep if e["stage"] == "se_doubling"]
    record.update({
        "sampler_version": "v2",
        "n_data_seqs": lap["n_data_seqs"],
        "tokens_per_seq": T,
        "curvature": {
            "selection_mode": lap["selection_mode"],
            "n_curvature_batches": lap["n_curvature_batches"],
            "curvature_batch_size": lap["curvature_batch_size"],
            "n_windows": lap["n_curvature_batches"] * lap["curvature_batch_size"],
            "n_entries": n_entries,
            "n_nonpositive_entries": n_nonpos,
            "median_fhat": median,
            "median_scaled_precision": lap["n_data_seqs"] * T**2 * median,
            "reference_state_path": lap["reference_state_path"],
            "reference_median_fhat": ref_median,
            "median_ratio": ratio,
            "median_ratio_bounds": [lo, hi],
        },
        "tfb_record": tfb_budget,
        "rho_tfb": rho,
        "ell0": ell0,
        "target_delta_nll": target,
        "same_fit_as_tfb": same_fit,
        "sweep": sweep,
        "lambdas_tried": [e["lam"] for e in sweep],
        "match": match,
        "match_status": match["status"],
        "lambda_star": match["lambda_star"],
        "delta_nll_star": match["delta_nll"],
        "se_star": match["se"],
        "seeds": {
            "fit_seed": cfg["fit"]["fit_seed"],
            "curvature_seed": lap["curvature_seed"],
            "delta_seeds": list(range(lap["n_delta_samples"])),
            "doubled_seeds": doubled[-1] if doubled else None,
        },
        "state_path": str(state_path) if state_path is not None else None,
        "state_sha256": _file_sha256(state_path) if state_path is not None else None,
        "state_readback_version": readback,
        "checks": checks,
        "failures": failures,
        "exit_code": exit_code,
    })
    print(f"[refit laplace] median F_hat={median:.4g} (ref ratio {ratio:.3f}); l0={ell0:.5f}; "
          f"target dNLL={target:.5g}; status={match['status']} lambda*={match['lambda_star']}")
    return exit_code


@dataclass
class RefitResult:
    exit_code: int
    record: dict
    record_path: Path


def run_refit(cfg: dict, *, config_path: str | Path | None = None) -> RefitResult:
    """Run one refit from a checked config and write ``out_dir/fit_record.json``.

    Raises RefitCheckError before any output is written when a fit document is an eval
    document, a test document lies in the curvature slice, ``n_data_seqs`` disagrees with
    the data, or the TFB record has no budget. The TFB search errors ("search range too
    small", "no sigma_q accepted") propagate as ValueError.
    """
    check_config(cfg)
    device = _resolve_device(cfg["device"])
    autocast_context(cfg["autocast"], device)
    out_dir = _check_out_dir(cfg)
    f = cfg["fit"]
    rows, manifest_sha = read_manifest(f["manifest_path"])
    check_fit_isolation(_fit_rows(rows, f["fit_domain"], f["fit_split"]), _test_rows(rows))
    data = load_refit_data(cfg)
    fit = build_fit_blocks(rows, data, cfg)
    tfb_budget = _laplace_prechecks(cfg, rows, data) if cfg["method"] == "laplace" else None
    model, base_info = load_posthoc_base(cfg, device=device)
    batches = [(x.to(device), y.to(device)) for x, y in fit_batches(fit, f["batch_size"])]

    commit, dirty = _git_state()
    fit_summary = fit.summary(cfg)
    record = {
        "record_version": RECORD_VERSION,
        "method": cfg["method"],
        "cell": cfg["cell"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "config": cfg,
        "torch_version": torch.__version__,
        "device": str(device),
        "autocast": cfg["autocast"],
        "base": base_info,
        "manifest": {"path": str(f["manifest_path"]), "sha256": manifest_sha},
        "fit_units": fit.units,
        "fit_set": fit_summary,
    }
    fit_ok = check_fit_blocks(fit, rows, data, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg["method"] == "tfb":
        exit_code = _run_tfb(cfg, model, batches, record, out_dir, device, fit_ok)
    else:
        exit_code = _run_laplace(cfg, model, batches, data, record, out_dir, tfb_budget, fit_ok)
    record_path = write_fit_record(out_dir / "fit_record.json", record)
    return RefitResult(exit_code=exit_code, record=record, record_path=record_path)
