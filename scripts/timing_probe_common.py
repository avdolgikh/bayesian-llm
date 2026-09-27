"""Shared constants, config, schema, machine state and device helpers of the S0 timing
probe (scripts/timing_probe.py)."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

SCHEMA = "i2-timing-probe/1"
CONFIG_SECTIONS = ("probe", "rescore", "finetune", "shape355m", "estimate", "dry_run")
OPERATOR_STATEMENTS = ("yes", "no", "unknown")

HEADER_FIELDS = (
    "schema", "created_utc", "git_head", "git_dirty", "gpu_name", "driver_version",
    "torch_version", "cuda_version", "memory_total_mib", "power_limit_w",
    "power_default_limit_w", "other_compute_procs", "sleep_disabled",
    "sysmem_fallback_confirmed", "windows_update_paused", "config_sha256",
)
ANCHOR_FIELDS = ("attempts_ms", "anchor_pass", "other_compute_procs")
CELL_FIELDS = ("method", "batch_size", "over_budget", "sampler_ms_per_sample")
CELL_MEASURED_FIELDS = ("ms_per_block", "n_blocks_timed", "peak_reserved_mib",
                        "peak_allocated_mib")
NLL_FIELDS = ("max_abs_diff_nats", "pass")
FINETUNE_FIELDS = (
    "wall_min", "train_call_min", "train_step_ms", "eval_s", "n_evals", "steps_completed",
    "trainable_params", "best_val_loss", "first_val_loss", "peak_reserved_mib",
    "config_sha256", "seed_applied", "checkpoint", "ca4_ratio", "ca4_exceeded",
)
SHAPE_FIELDS = ("base_params", "trainable_params", "peak_reserved_mib", "peak_allocated_mib",
                "median_step_ms", "max_step_ms", "oom")
SHAPE355M_FIELDS = ("shapes", "fits", "spill_suspected", "ca5_ratio", "ca5_out_of_range",
                    "fallback")
M4_FIELDS = ("gpu_h", "batch_size", "cap_applied", "blocks_per_doc", "escalate_g0",
             "low_end_holds", "m4_nights_needed", "speedup_vs_ca10", "speedup_measured")
REPRO_FIELDS = ("gpu_h_b1", "gpu_h_bstar")
ESTIMATES_FIELDS = ("m4", "repro", "s5_finetune_gpu_h")



class SchemaError(ValueError):
    """The probe JSON does not follow schema i2-timing-probe/1."""


class NoFeasibleBatchError(RuntimeError):
    """A score set has no batch size within the VRAM budget."""


# ---------------------------------------------------------------------------
# Config, paths, hashing
# ---------------------------------------------------------------------------

def repo_path(path: str | Path) -> Path:
    """Resolve a config path: absolute paths stay, relative ones hang off the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _operator_statement(value: Any, key: str) -> str:
    # YAML 1.1 reads a bare yes/no as a boolean.
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value not in OPERATOR_STATEMENTS:
        raise ValueError(f"probe.{key} must be one of {OPERATOR_STATEMENTS}, got {value!r}")
    return value


def load_config(path: str | Path, dry_run: bool = False) -> dict:
    """Load the probe YAML. With ``dry_run``, apply ``dry_run.overrides`` (dotted keys)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [s for s in CONFIG_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"{path}: missing config sections {missing}")
    if dry_run:
        from minigpt.config import apply_dict_overrides

        cfg = copy.deepcopy(cfg)
        apply_dict_overrides(cfg, cfg["dry_run"]["overrides"])
    for key in ("sysmem_fallback_confirmed", "windows_update_paused"):
        cfg["probe"][key] = _operator_statement(cfg["probe"][key], key)
    if cfg["probe"]["q1_backbone"] not in ("a", "b"):
        raise ValueError(f"probe.q1_backbone must be a or b, got {cfg['probe']['q1_backbone']!r}")
    return cfg


def config_sha256(path: str | Path) -> str:
    """sha256 of a YAML file's content in canonical JSON form.

    Key order, comments, whitespace and line endings do not change the digest, so a
    checkout with other line endings hashes the same config to the same value.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_import_paths() -> None:
    for p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _pile_cache_paths(train_cfg: dict) -> list[Path]:
    """The cached Pile tensors ``load_pile_data`` reads for this config."""
    d = train_cfg["data"]
    pile = REPO_ROOT / "data" / "pile"
    return ([pile / f"{dom}_{d['pile_id_tokens']}.pt" for dom in d["pile_id_domains"]]
            + [pile / f"{dom}_{d['pile_ood_tokens']}.pt" for dom in d["pile_ood_domains"]])


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("missing inputs (a missing Pile cache would stream from the "
                                f"network): {', '.join(missing)}")


# ---------------------------------------------------------------------------
# JSON: schema and merge
# ---------------------------------------------------------------------------

def _require(obj: Any, fields: Iterable[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise SchemaError(f"{where} must be an object")
    missing = [f for f in fields if f not in obj]
    if missing:
        raise SchemaError(f"{where} is missing {', '.join(missing)}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_rescore(sec: dict) -> None:
    _require(sec, ("anchor", "cells", "nll_check"), "rescore")
    _require(sec["anchor"], ANCHOR_FIELDS, "rescore.anchor")
    _require(sec["nll_check"], NLL_FIELDS, "rescore.nll_check")
    cells = sec["cells"]
    if not isinstance(cells, list) or not cells:
        raise SchemaError("rescore.cells must be a non-empty list")
    for i, cell in enumerate(cells):
        where = f"rescore.cells[{i}]"
        _require(cell, CELL_FIELDS, where)
        if not cell["over_budget"]:
            for field in CELL_MEASURED_FIELDS:
                if not _is_number(cell.get(field)):
                    raise SchemaError(f"{where} ({cell['method']}, batch {cell['batch_size']}) "
                                      f"has no {field} and is not over_budget")
        if cell.get("mode") == "posthoc" and not _is_number(cell["sampler_ms_per_sample"]):
            raise SchemaError(f"{where} ({cell['method']}) has no sampler_ms_per_sample")


def _validate_shape355m(sec: dict) -> None:
    if isinstance(sec, dict) and "skipped" in sec:
        return
    _require(sec, SHAPE355M_FIELDS, "shape355m")
    if not isinstance(sec["shapes"], dict) or not sec["shapes"]:
        raise SchemaError("shape355m.shapes must be a non-empty object")
    for name, shape in sec["shapes"].items():
        _require(shape, SHAPE_FIELDS, f"shape355m.shapes.{name}")


def _validate_estimates(sec: dict) -> None:
    _require(sec, ESTIMATES_FIELDS, "estimates")
    _require(sec["m4"], M4_FIELDS, "estimates.m4")
    _require(sec["repro"], REPRO_FIELDS, "estimates.repro")


_SECTION_VALIDATORS: dict[str, Callable[[dict], None]] = {
    "rescore": _validate_rescore,
    "finetune": lambda sec: _require(sec, FINETUNE_FIELDS, "finetune"),
    "shape355m": _validate_shape355m,
    "estimates": _validate_estimates,
}


def validate_schema(doc: Any, require: Iterable[str] = ()) -> None:
    """Raise ``SchemaError`` unless ``doc`` follows schema i2-timing-probe/1.

    Every section that is present is checked. ``require`` names sections that must exist.
    """
    if not isinstance(doc, dict):
        raise SchemaError("probe JSON must be an object")
    _require(doc.get("header"), HEADER_FIELDS, "header")
    if doc["header"]["schema"] != SCHEMA:
        raise SchemaError(f"header.schema is {doc['header']['schema']!r}, expected {SCHEMA!r}")
    for name in require:
        if name not in doc:
            raise SchemaError(f"section {name!r} is missing")
    for name, validator in _SECTION_VALIDATORS.items():
        if name in doc:
            validator(doc[name])


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_section(path: str | Path, name: str, section: dict, header: dict) -> dict:
    """Merge one section and the header into the probe JSON, atomically.

    The header keeps the first ``created_utc``; its other fields take the latest values.
    The merged document is validated before anything is written.
    """
    path = Path(path)
    doc = load_json(path) if path.exists() else {}
    old_header = doc.get("header") or {}
    new_header = {**old_header, **header, "updated_utc": _utc_now()}
    if "created_utc" in old_header:
        new_header["created_utc"] = old_header["created_utc"]
    doc["header"] = new_header
    doc[name] = section
    validate_schema(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return doc


# ---------------------------------------------------------------------------
# Machine state (read-only subprocess calls)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=REPO_ROOT, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _gpu_query(timeout: float) -> dict:
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,"
                "power.default_limit", "--format=csv,noheader,nounits"], timeout)
    if not out:
        return {}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) != 5:
        return {}
    return {
        "gpu_name": parts[0],
        "driver_version": parts[1],
        "memory_total_mib": _num(parts[2]),
        "power_limit_w": _num(parts[3]),
        "power_default_limit_w": _num(parts[4]),
    }


def _compute_apps(timeout: float) -> list[tuple[int, str]] | None:
    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
               timeout)
    if out is None:
        return None
    apps = []
    for line in out.splitlines():
        pid, _, name = line.partition(",")
        if pid.strip().isdigit() and int(pid) != os.getpid():
            apps.append((int(pid), name.strip()))
    return apps


def other_compute_procs(timeout: float) -> dict[str, Any]:
    """Other GPU processes from ``nvidia-smi``: all PIDs, the python ones, and GPU load.

    Under Windows WDDM the desktop apps are listed too, so ``other_python_compute_procs``
    and ``gpu_utilization_pct`` are the readable signal of a competing compute job.
    """
    apps = _compute_apps(timeout)
    load = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"], timeout)
    util = used = None
    if load:
        fields = [f.strip() for f in load.splitlines()[0].split(",")]
        if len(fields) == 2:
            util, used = _num(fields[0]), _num(fields[1])
    return {
        "other_compute_procs": None if apps is None else [pid for pid, _ in apps],
        "other_python_compute_procs": (None if apps is None else
                                       [pid for pid, name in apps if "python" in name.lower()]),
        "gpu_utilization_pct": util,
        "gpu_memory_used_mib": used,
    }


def _sleep_disabled(timeout: float) -> bool | None:
    if os.name != "nt":
        return None
    out = _run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"], timeout)
    for line in (out or "").splitlines():
        if "AC Power Setting Index" in line:
            value = line.rsplit(":", 1)[1].strip()
            try:
                return int(value, 16) == 0
            except ValueError:
                return None
    return None


def collect_header(cfg: dict, config_path: Path, dry_run: bool) -> dict:
    timeout = cfg["probe"]["subprocess_timeout_s"]
    gpu = _gpu_query(timeout)
    status = _run(["git", "status", "--porcelain"], timeout)
    procs = other_compute_procs(timeout)
    return {
        "schema": SCHEMA,
        "created_utc": _utc_now(),
        "git_head": _run(["git", "rev-parse", "HEAD"], timeout),
        "git_dirty": None if status is None else bool(status),
        "gpu_name": gpu.get("gpu_name"),
        "driver_version": gpu.get("driver_version"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "memory_total_mib": gpu.get("memory_total_mib"),
        "power_limit_w": gpu.get("power_limit_w"),
        "power_default_limit_w": gpu.get("power_default_limit_w"),
        **procs,
        "sleep_disabled": _sleep_disabled(timeout),
        "sysmem_fallback_confirmed": cfg["probe"]["sysmem_fallback_confirmed"],
        "windows_update_paused": cfg["probe"]["windows_update_paused"],
        "config_sha256": {
            "probe": config_sha256(config_path),
            "det_lora": config_sha256(repo_path(cfg["finetune"]["train_config"])),
        },
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _reset_peak(device: torch.device) -> None:
    _empty_cache(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (torch.cuda.max_memory_reserved(device) / 2**20,
            torch.cuda.max_memory_allocated(device) / 2**20)


def _autocast(device: torch.device, use_amp: bool):
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)


def _n_params(model: torch.nn.Module, trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


@dataclass
class _Context:
    device: torch.device
    dry_run: bool
    output: Path
