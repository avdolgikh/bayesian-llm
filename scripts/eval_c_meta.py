"""Eval config, freeze check, determinism, meta and CLI helpers for eval_c_checkpoints.py."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
import yaml
from eval_c_load import EVAL_SETS, PREREG_FILE, REPO_ROOT, repo_path

from minigpt.evalset import verify_prereg
from minigpt.uncertainty import auroc, bootstrap_ci

# ---------------------------------------------------------------------------
# Eval config, freeze check, determinism, meta
# ---------------------------------------------------------------------------

def load_eval_config(path: str | Path) -> tuple[dict, str]:
    """Return (config, sha256 of the file bytes)."""
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def scores_dir(cfg: dict) -> Path:
    return repo_path(cfg["scoring"]["out_dir"])


def check_prereg(cfg: dict) -> str:
    """Raise PreregError unless ``prereg.json`` holds the sha256 of ``cfg["analysis"]``.

    The hash rule and the check are minigpt.evalset's (section 5.5, Freeze), so the scorer
    and ``check_eval_rebuild.py --freeze`` agree on one definition.
    """
    return verify_prereg(cfg, scores_dir(cfg) / PREREG_FILE)


def resolve_seed_base(cfg: dict, requested: int | None) -> int:
    """``scoring.seed_base``, or a value from ``checks.repro.spread_seed_bases``."""
    base = cfg["scoring"]["seed_base"]
    if requested is None:
        return base
    allowed = [base, *cfg["checks"]["repro"]["spread_seed_bases"]]
    if requested not in allowed:
        raise ValueError(f"--seed-base {requested} is not one of {allowed}")
    return requested


def apply_determinism(det: dict) -> None:
    """Apply ``scoring.determinism`` (call before any model is built)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = det["cublas_workspace_config"]
    torch.use_deterministic_algorithms(
        det["use_deterministic_algorithms"], warn_only=det["warn_only"],
    )
    torch.backends.cudnn.deterministic = det["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = det["cudnn_benchmark"]


class WarningLog:
    """Collect each distinct warning raised inside the ``with`` block."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._ctx = warnings.catch_warnings()

    def __enter__(self) -> WarningLog:
        self._ctx.__enter__()
        warnings.simplefilter("always")
        warnings.showwarning = self._record
        return self

    def _record(self, message, category, filename, lineno, file=None, line=None) -> None:
        text = f"{category.__name__}: {message}"
        if text not in self.messages:
            self.messages.append(text)
            print(f"WARNING recorded in meta: {text}", file=sys.stderr)

    def __exit__(self, *exc) -> None:
        self._ctx.__exit__(*exc)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(script_paths: list[Path]) -> str:
    """sha256 over minigpt/*.py and the scripts that ran (path and bytes of each file)."""
    files = sorted((REPO_ROOT / "minigpt").glob("*.py"))
    for p in script_paths:
        p = Path(p).resolve()
        if p not in files:
            files.append(p)
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT) else p.name
        h.update(rel.encode("utf-8") + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def git_state() -> tuple[str | None, bool | None]:
    """(HEAD sha, dirty flag), read-only; (None, None) outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, bool(status.strip())


def score_file_path(cfg: dict, score_set: str, eval_set: str, run_tag: str) -> Path:
    return scores_dir(cfg) / f"{score_set}__{eval_set}__{run_tag}.pt"


def _id_base_domain(cfg: dict) -> str:
    keys = [d["key"] for d in cfg["eval_set"]["domains"] if d["role"] == "id_base"]
    if len(keys) != 1:
        raise ValueError(f"eval_set.domains needs exactly one id_base domain, got {keys}")
    return keys[0]


def mi_ratio_lines(payload: dict, file_name: str, cfg: dict) -> list[str]:
    """Descriptive MI ratios, mean M(OOD) / mean M(ID) over the blocks of one score file.

    legacy_d1 pools all OOD blocks (the old estimator); other sets give one ratio per
    ``analysis.ood_domains`` entry present in the file.
    """
    meta = payload["meta"]
    id_key = _id_base_domain(cfg)
    domains = payload["domain"]
    mi = payload["blk_mi"].double()
    id_mask = torch.tensor([d == id_key for d in domains], dtype=torch.bool)
    if not bool(id_mask.any()):
        return []
    if meta["eval_set"] == "legacy_d1":
        pairs = [("ood", ~id_mask)]
    else:
        pairs = [
            (d, torch.tensor([x == d for x in domains], dtype=torch.bool))
            for d in cfg["analysis"]["ood_domains"] if d in domains
        ]
    id_mean = float(mi[id_mask].mean())
    lines = []
    for name, mask in pairs:
        if not bool(mask.any()):
            continue
        prefix = (f"MI ratio (descriptive, block means from {file_name}) "
                  f"{meta['score_set']} {meta['eval_set']} {name}/{id_key}:")
        if id_mean == 0.0:
            lines.append(f"{prefix} n/a (mean M(ID) is 0)")
        else:
            lines.append(f"{prefix} {float(mi[mask].mean()) / id_mean:.9f}")
    return lines


def summary_lines(path: Path, cfg: dict, n_bootstrap: int | None) -> list[str]:
    """Per-domain block means; MI ratios; on legacy_d1 also block AUROCs (i.i.d. CIs)."""
    payload = torch.load(path, weights_only=True)
    domains = payload["domain"]
    lines = []
    for d in sorted(set(domains), key=domains.index):
        mask = torch.tensor([x == d for x in domains], dtype=torch.bool)
        lines.append(
            f"  {d:<18} blocks {int(mask.sum()):>5}  "
            f"mean G {float(payload['blk_g'][mask].double().mean()):.6f}  "
            f"mean M {float(payload['blk_mi'][mask].double().mean()):.6f}"
        )
    lines += mi_ratio_lines(payload, path.name, cfg)
    if payload["meta"]["eval_set"] != "legacy_d1":
        return lines
    id_key = _id_base_domain(cfg)
    labels = torch.tensor([float(d != id_key) for d in domains])
    if not 0 < float(labels.sum()) < len(labels):
        return lines
    for key in ("blk_mi", "blk_g", "blk_tu", "blk_maxprob_unc"):
        text = f"  AUROC (legacy_d1, unweighted blocks) {key}: {auroc(payload[key], labels):.4f}"
        if n_bootstrap:
            _, lo, hi = bootstrap_ci(payload[key], labels, auroc, n_bootstrap=n_bootstrap,
                                     seed=42)
            text += f" [i.i.d. 95% CI {lo:.4f}, {hi:.4f}]"
        lines.append(text)
    return lines


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def add_i2_arguments(p: argparse.ArgumentParser) -> None:
    """CLI flags of the I2 path, shared with scripts/eval_mc_dropout.py."""
    p.add_argument("--eval-config", type=str, default=None,
                   help="Eval config YAML (configs/i2_eval.yaml for real runs)")
    p.add_argument("--eval-set", choices=EVAL_SETS, default=None,
                   help="Score one eval set and write score files (I2 path)")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Score-file tag, e.g. main, rerun, batched, spread1")
    p.add_argument("--block-ids", type=str, default=None,
                   help="Subset of block indices, e.g. 0-99,500-599 (b keeps its value)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override scoring.batch_size[eval_set] (recorded in meta)")
    p.add_argument("--seed-base", type=int, default=None,
                   help="scoring.seed_base or one of checks.repro.spread_seed_bases")
    p.add_argument("--score-set", type=str, default=None,
                   help="Comma list of score sets (default: all listed for the eval set)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Device (default: cuda if available)")


def select_score_sets(
    listed: list[str],
    requested: list[str] | None,
    owned: set[str],
    scored_elsewhere: Callable[[str], str | None],
) -> tuple[list[str], list[str]]:
    """Return (score sets this script runs, score sets another script runs)."""
    names = list(listed) if requested is None else requested
    not_listed = [n for n in names if n not in listed]
    if not_listed:
        raise ValueError(f"score sets {not_listed} are not listed for this eval set: {listed}")
    unknown = [n for n in names if n not in owned and scored_elsewhere(n) is None]
    if unknown:
        raise ValueError(f"score sets {unknown} are not registered (see register_score_set)")
    to_run = [n for n in names if n in owned]
    skipped = [n for n in names if n not in owned]
    if not to_run:
        raise ValueError(f"nothing to score here among {names}")
    return to_run, skipped
