"""Constants and config, hashing and file helpers of check_eval_rebuild.py.

Part of the independent checker (S1-T5b): it imports numpy, torch, PyYAML, sklearn
and its check_eval_* siblings, never minigpt or the eval scripts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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
