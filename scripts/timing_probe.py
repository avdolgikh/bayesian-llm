"""S0 timing probe: re-score rates, a deterministic LoRA fine-tune and the 355M shape.

Spec: specs/i2-timing-probe.md. Each part merges its own section into one JSON file
(``probe.output``, by default data/i2/timing_probe.json) through a temp file and
``os.replace``. ``--part all`` runs shape355m, rescore, finetune and estimate, each in its
own subprocess, so VRAM is freed between parts and a crash loses only the part in flight.
``check`` is not part of ``all``: it runs before the m4 re-score.

Usage:
    python scripts/timing_probe.py --config configs/i2_timing_probe.yaml --part all
    python scripts/timing_probe.py --part rescore
    python scripts/timing_probe.py --part estimate
    python scripts/timing_probe.py --part check --m4-config <S1 m4 YAML> [--g0-ack]
    python scripts/timing_probe.py --part all --dry-run     # tiny CPU models, no data/ reads

Exit codes of ``--part check``: 0 the m4 plan fits the probe; 2 the m4 YAML asks for more
blocks per document than the probe allows; 3 the probe escalates to G0 and ``--g0-ack`` is
missing; 4 the probe JSON (or its ``estimates.m4``) is missing; 5 bad input (m4 YAML missing,
unreadable or without ``blocks_per_doc``; bad command line). Other parts exit 0 on success
and 1 on failure.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "i2_timing_probe.yaml"

SCHEMA = "i2-timing-probe/1"
CONFIG_SECTIONS = ("probe", "rescore", "finetune", "shape355m", "estimate", "dry_run")
ALL_PARTS = ("shape355m", "rescore", "finetune", "estimate")
PARTS = (*ALL_PARTS, "check", "all")
SECTION_OF_PART = {
    "shape355m": "shape355m",
    "rescore": "rescore",
    "finetune": "finetune",
    "estimate": "estimates",
}
OPERATOR_STATEMENTS = ("yes", "no", "unknown")
BLOCKS_KEYS = ("blocks_per_doc", "max_blocks_per_doc")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKS_MISMATCH = 2
EXIT_G0_UNACKED = 3
EXIT_NO_PLAN = 4
EXIT_BAD_INPUT = 5

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


# ---------------------------------------------------------------------------
# Batched MC statistics (probe-local; S1 owns the production scorer)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_batch(
    logits_fn: Callable[[int], torch.Tensor], y: torch.Tensor, n_samples: int,
) -> dict[str, torch.Tensor]:
    """MC statistics for one batch.

    ``logits_fn(s)`` returns the logits (B, T, V) of draw s; ``y`` holds the targets (B, T).
    Running sums of p and H[p] stay in fp32 and are updated in place. Returns per-token
    ``mi``, ``pred_entropy`` (H[p-bar]), ``max_prob`` (max p-bar), ``g`` (the log-mean-exp
    gap), ``nll`` (-log p-bar(y)) and ``ll``, the N realized-token log-probs (N, B, T).
    """
    p_sum: torch.Tensor | None = None
    ent_sum: torch.Tensor | None = None
    lls = []
    for s in range(n_samples):
        logp = F.log_softmax(logits_fn(s).float(), dim=-1)
        lls.append(logp.gather(-1, y.unsqueeze(-1)).squeeze(-1))
        p = logp.exp()
        if p_sum is None:
            p_sum = torch.zeros_like(p)
            ent_sum = torch.zeros(p.shape[:-1], dtype=p.dtype, device=p.device)
        p_sum.add_(p)
        ent_sum.sub_(p.mul_(logp).sum(dim=-1))
        del logp, p
    p_bar = p_sum.div_(n_samples)
    pred_entropy = -torch.special.xlogy(p_bar, p_bar).sum(dim=-1)
    mi = pred_entropy - ent_sum / n_samples
    max_prob = p_bar.max(dim=-1).values
    ll = torch.stack(lls)
    ll64 = ll.double()
    log_p_bar_y = torch.logsumexp(ll64, dim=0) - math.log(n_samples)
    g = log_p_bar_y - ll64.mean(dim=0)
    return {"mi": mi, "pred_entropy": pred_entropy, "max_prob": max_prob, "g": g,
            "nll": -log_p_bar_y, "ll": ll}


def _logits_fn(model, x, mode, draw_fn, batch_index, n_samples, device, use_amp):
    """Logits of draw s for one batch. Post-hoc draws use seed batch_index * N + s."""
    if mode == "posthoc":
        from minigpt.laplace import apply_sampled_params

        def posthoc(s: int) -> torch.Tensor:
            sampled = draw_fn(batch_index * n_samples + s)
            with apply_sampled_params(model, sampled), _autocast(device, use_amp):
                return model(x)[0]

        return posthoc

    def plain(s: int) -> torch.Tensor:
        # One call is one weight draw shared by the batch (C1, BLoB); MC dropout masks
        # are drawn per element.
        with _autocast(device, use_amp):
            return model(x)[0]

    return plain


def _mode_context(model, mode):
    if mode == "dropout":
        from minigpt.layers import enable_dropout

        return enable_dropout(model)
    return nullcontext()


def _score_one_batch(model, xs, ys, batch_index, batch_size, n_samples, mode, draw_fn,
                     device, use_amp) -> dict[str, torch.Tensor]:
    lo = batch_index * batch_size
    x = xs[lo : lo + batch_size].to(device)
    y = ys[lo : lo + batch_size].to(device)
    fn = _logits_fn(model, x, mode, draw_fn, batch_index, n_samples, device, use_amp)
    stats = score_batch(fn, y, n_samples)
    return {k: v.cpu() for k, v in stats.items()}


@torch.no_grad()
def score_blocks(
    model: torch.nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    batch_size: int,
    n_samples: int,
    mode: str,
    draw_fn: Callable[[int], dict] | None,
    device: torch.device,
    use_amp: bool,
) -> dict[str, torch.Tensor]:
    """Score all blocks (n, T) at one batch size; returns the concatenated CPU statistics.

    ``mode`` is deterministic, variational, posthoc (``draw_fn(seed)`` gives the sampled
    parameters) or dropout.
    """
    if xs.size(0) % batch_size:
        raise ValueError(f"{xs.size(0)} blocks do not split into batches of {batch_size}")
    parts: dict[str, list[torch.Tensor]] = {}
    with _mode_context(model, mode):
        for bi in range(xs.size(0) // batch_size):
            stats = _score_one_batch(model, xs, ys, bi, batch_size, n_samples, mode, draw_fn,
                                     device, use_amp)
            for key, value in stats.items():
                parts.setdefault(key, []).append(value)
    return {k: torch.cat(v, dim=1 if k == "ll" else 0) for k, v in parts.items()}


def per_block_nll(
    model: torch.nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """Per-block mean NLL (nats) of a deterministic model at N=1 (the S0-T1(c) check)."""
    stats = score_blocks(model, xs, ys, batch_size=batch_size, n_samples=1,
                         mode="deterministic", draw_fn=None, device=device, use_amp=use_amp)
    return stats["nll"].mean(dim=-1)


# ---------------------------------------------------------------------------
# Rescore timing
# ---------------------------------------------------------------------------

def _empty_cell(batch_size: int) -> dict:
    return {"batch_size": batch_size, "ms_per_block": None, "n_blocks_timed": None,
            "peak_reserved_mib": None, "peak_allocated_mib": None, "over_budget": False,
            "oom": False}


def run_cell(
    model: torch.nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    mode: str,
    draw_fn: Callable[[int], dict] | None,
    batch_size: int,
    n_timed: int,
    warmup_batches: int,
    n_samples: int,
    seed: int,
    device: torch.device,
    use_amp: bool,
    vram_budget_mib: float,
) -> dict:
    """Time one (method, batch size) cell: warm-up batches, then ``n_timed`` blocks.

    The clock runs from a synchronize before the first timed batch to the end of the last
    host copy; it includes sampling and copies and excludes loading and warm-up.
    """
    if n_timed % batch_size:
        raise ValueError(f"n_timed {n_timed} is not a multiple of batch {batch_size}")
    n_batches = n_timed // batch_size
    if (warmup_batches + n_batches) * batch_size > xs.size(0):
        raise ValueError(f"batch {batch_size} needs more than the {xs.size(0)} loaded blocks")
    cell = _empty_cell(batch_size)
    torch.manual_seed(seed)
    _reset_peak(device)
    elapsed = None
    try:
        with torch.no_grad(), _mode_context(model, mode):
            for bi in range(warmup_batches):
                _score_one_batch(model, xs, ys, bi, batch_size, n_samples, mode, draw_fn,
                                 device, use_amp)
            _sync(device)
            t0 = time.perf_counter()
            for bi in range(warmup_batches, warmup_batches + n_batches):
                _score_one_batch(model, xs, ys, bi, batch_size, n_samples, mode, draw_fn,
                                 device, use_amp)
            elapsed = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        pass
    if elapsed is None:
        cell.update(over_budget=True, oom=True)
        _empty_cache(device)
        return cell
    reserved, allocated = _peak_mib(device)
    cell.update(ms_per_block=elapsed * 1000.0 / n_timed, n_blocks_timed=n_timed,
                peak_reserved_mib=reserved, peak_allocated_mib=allocated,
                over_budget=reserved > vram_budget_mib)
    if cell["over_budget"]:
        _empty_cache(device)
    return cell


@torch.no_grad()
def time_sampler(model, draw_fn, *, warmup: int, draws: int, device: torch.device) -> float:
    """ms per posterior draw plus the parameter swap in and out, synchronized."""
    from minigpt.laplace import apply_sampled_params

    def once(seed: int) -> None:
        with apply_sampled_params(model, draw_fn(seed)):
            _sync(device)
        _sync(device)

    for i in range(warmup):
        once(i)
    t0 = time.perf_counter()
    for i in range(draws):
        once(warmup + i)
    return (time.perf_counter() - t0) * 1000.0 / draws


def run_anchor(model, acfg: dict, *, block_size: int, device: torch.device,
               timeout: float, sleep: Callable[[float], None] = time.sleep) -> dict:
    """C0 forward-only anchor (``benchmark_latency``, the code behind report.md's 8.1 ms)."""
    _ensure_import_paths()
    from benchmark_inference import benchmark_latency

    lo = acfg["ref_ms"] * (1.0 - acfg["tolerance"])
    hi = acfg["ref_ms"] * (1.0 + acfg["tolerance"])
    attempts: list[float] = []
    procs: list[dict[str, Any]] = []
    for attempt in range(acfg["max_attempts"]):
        if attempt:
            sleep(acfg["retry_wait_s"])
        procs.append(other_compute_procs(timeout))
        ms, _ = benchmark_latency(model, device, "deterministic", None, n_samples=1,
                                  seq_len=block_size, n_warmup=acfg["warmup"],
                                  n_measure=acfg["repeats"])
        attempts.append(float(ms))
        print(f"[rescore] anchor attempt {attempt + 1}: {ms:.2f} ms "
              f"(window {lo:.3f}-{hi:.3f})", flush=True)
        if lo <= ms <= hi:
            break
    return {"attempts_ms": attempts, "anchor_pass": lo <= attempts[-1] <= hi,
            "other_compute_procs": [p["other_compute_procs"] for p in procs],
            "gpu_state_before_attempt": procs, "ref_ms": acfg["ref_ms"],
            "window_ms": [lo, hi]}


def _gaussian_draw_fn(model, names: list[str], std: float) -> Callable[[int], dict]:
    """Dry-run stand-in for a post-hoc sampler: a CPU-generator Gaussian around the MAP."""
    params = dict(model.named_parameters())
    base = {n: params[n].detach().clone() for n in names}

    def draw(seed: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return {n: t + std * torch.randn(t.shape, generator=gen, dtype=t.dtype).to(t.device)
                for n, t in base.items()}

    return draw


class _RealProvider:
    """Checkpoints and eval blocks from data/ through the existing eval scripts."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        _ensure_import_paths()
        import benchmark_inference
        import eval_c_checkpoints
        import eval_mc_dropout

        self.bench = benchmark_inference
        self.ecc = eval_c_checkpoints
        self.mcd = eval_mc_dropout
        self.device = device
        if cfg["rescore"]["block_size"] != eval_c_checkpoints.BLOCK_SIZE:
            raise ValueError("rescore.block_size must match eval_c_checkpoints.BLOCK_SIZE")

    def check_inputs(self, methods: list[str]) -> None:
        from experiments.c_milestones import build_milestone_config

        paths = [p for m in methods
                 for p in self.ecc._checkpoint_paths("c0" if m == "mc_dropout" else m)]
        _require_files([REPO_ROOT / p for p in paths]
                       + _pile_cache_paths(build_milestone_config("c0")))

    def anchor_model(self):
        model, _ = self.bench._load_model("c0", self.device)
        return model

    def blocks(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        id_seqs, _ = self.ecc.load_eval_data(n)
        if len(id_seqs) != n:
            raise ValueError(f"expected {n} ID blocks, got {len(id_seqs)}")
        return torch.stack([x for x, _ in id_seqs]), torch.stack([y for _, y in id_seqs])

    def load(self, method: str):
        from minigpt.laplace import sample_laplace_params
        from minigpt.tfb import sample_tfb_params

        if method == "mc_dropout":
            return self.mcd.load_c0_model(self.device), "dropout", None
        model, kind, state = self.ecc.load_model(method, self.device)
        if kind == "laplace":
            return model, "posthoc", lambda seed: sample_laplace_params(state, seed=seed)
        if kind == "tfb":
            return model, "posthoc", lambda seed: sample_tfb_params(state, seed=seed)
        return model, kind, None


class _DryProvider:
    """Tiny random models and random blocks on the CPU; reads nothing from data/."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        self.dry = cfg["dry_run"]
        self.block_size = cfg["rescore"]["block_size"]
        self.seed = cfg["rescore"]["seed"]
        self.device = device

    def check_inputs(self, methods: list[str]) -> None:
        return None

    def _build(self, bayes_ffn: bool = False):
        from minigpt.layers import BayesConfig
        from minigpt.model import GPTConfig, MiniGPT

        m = self.dry["model"]
        bayes = BayesConfig(**self.dry["bayes_ffn"]) if bayes_ffn else BayesConfig(enabled=False)
        return MiniGPT(GPTConfig(
            vocab_size=self.dry["vocab_size"], block_size=m["block_size"], n_layer=m["n_layer"],
            n_head=m["n_head"], n_embd=m["n_embd"], dropout=m["dropout"], bias=m["bias"],
            bayes_ffn=bayes,
        ))

    def anchor_model(self):
        return self._build().to(self.device).eval()

    def blocks(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        tokens = torch.randint(0, self.dry["vocab_size"], (n, self.block_size + 1), generator=gen)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    def load(self, method: str):
        from minigpt.lora import LoRAConfig, inject_lora

        lora = LoRAConfig(**self.dry["lora"])
        names = None
        if method == "c0":
            model, mode = self._build(), "deterministic"
        elif method == "c1":
            model, mode = self._build(bayes_ffn=True), "variational"
        elif method == "c3":
            model, mode = inject_lora(self._build(), lora, bayesian=True), "variational"
        elif method == "c2":
            model, mode = self._build(), "posthoc"
            names = [n for n, p in model.named_parameters() if ".mlp." in n and p.dim() == 2]
        elif method in ("c4_tfb", "c4_lap"):
            model, mode = inject_lora(self._build(), lora, bayesian=False), "posthoc"
            names = [n for n, _ in model.named_parameters() if n.endswith("lora_A")]
        elif method == "mc_dropout":
            model, mode = self._build(), "dropout"
        else:
            raise ValueError(f"unknown score set {method!r}")
        model = model.to(self.device).eval()
        draw = _gaussian_draw_fn(model, names, self.dry["posthoc_std"]) if names else None
        return model, mode, draw


def _nll_check(provider, xs, ys, rc: dict, device: torch.device, use_amp: bool) -> dict:
    model, _, _ = provider.load("c0")
    model.requires_grad_(False)
    n = rc["nll_check"]["blocks"]
    tol = rc["nll_check"]["tol_nats"]
    base_b, *other_b = rc["batch_sizes"]
    per_batch: dict[int, torch.Tensor | None] = {}
    for b in rc["batch_sizes"]:
        try:
            per_batch[b] = per_block_nll(model, xs[:n], ys[:n], batch_size=b, device=device,
                                         use_amp=use_amp)
        except torch.cuda.OutOfMemoryError:
            per_batch[b] = None
        _empty_cache(device)
    ref = per_batch[base_b]
    diffs = {str(b): (None if ref is None or per_batch[b] is None
                      else float((per_batch[b] - ref).abs().max()))
             for b in other_b}
    passed = all(d is not None and d <= tol for d in diffs.values())
    del model
    _empty_cache(device)
    return {"blocks": n, "tol_nats": tol, "reference_batch_size": base_b,
            "max_abs_diff_nats": diffs, "pass": passed,
            "mean_nll_reference": None if ref is None else float(ref.mean())}


def _rescore_checks(cells: list[dict], rc: dict, anchor: dict, nll: dict) -> dict:
    expected = len(rc["methods"]) * len(rc["batch_sizes"])
    feasible = {m: any(c["method"] == m and not c["over_budget"] for c in cells)
                for m in rc["methods"]}
    complete = all(
        c["over_budget"] or all(_is_number(c[f]) for f in CELL_MEASURED_FIELDS) for c in cells
    ) and all(_is_number(c["sampler_ms_per_sample"]) for c in cells if c["mode"] == "posthoc")
    return {
        "n_cells_expected": expected,
        "n_cells": len(cells),
        "feasible_by_method": feasible,
        "t1a_pass": len(cells) == expected and complete and all(feasible.values()),
        "t1b_anchor_pass": anchor["anchor_pass"],
        "t1c_nll_pass": nll["pass"],
    }


def part_rescore(cfg: dict, ctx: _Context) -> dict:
    """S0-T1: anchor, 7 score sets x batch sizes, then the untimed NLL check."""
    rc = cfg["rescore"]
    device = ctx.device
    budget = cfg["probe"]["vram_budget_mib"]
    use_amp = device.type == "cuda" and rc["amp_fp16"]
    provider = _DryProvider(cfg, device) if ctx.dry_run else _RealProvider(cfg, device)
    provider.check_inputs(rc["methods"])

    anchor_model = provider.anchor_model()
    anchor = run_anchor(anchor_model, rc["anchor"], block_size=rc["block_size"], device=device,
                        timeout=cfg["probe"]["subprocess_timeout_s"])
    del anchor_model
    _empty_cache(device)

    xs, ys = provider.blocks(rc["n_blocks_loaded"])
    cells = []
    for method in rc["methods"]:
        model, mode, draw_fn = provider.load(method)
        model.requires_grad_(False)  # autocast caches fp16 copies of weights that need grad
        n_samples = rc["n_samples_c0"] if method == "c0" else rc["n_samples"]
        sampler_ms = None
        if mode == "posthoc":
            sampler_ms = time_sampler(model, draw_fn, warmup=rc["sampler_timing_warmup"],
                                      draws=rc["sampler_timing_draws"], device=device)
        blocked = False
        for b in rc["batch_sizes"]:
            if blocked:
                cell = _empty_cell(b)
                cell.update(over_budget=True, skipped_after_over_budget=True)
            else:
                cell = run_cell(model, xs, ys, mode=mode, draw_fn=draw_fn, batch_size=b,
                                n_timed=rc["blocks_timed"][b],
                                warmup_batches=rc["warmup_batches"], n_samples=n_samples,
                                seed=rc["seed"], device=device, use_amp=use_amp,
                                vram_budget_mib=budget)
                blocked = cell["over_budget"] and rc["skip_larger_after_over_budget"]
            cells.append({"method": method, "mode": mode, "n_samples": n_samples, **cell,
                          "sampler_ms_per_sample": sampler_ms})
            ms = cell["ms_per_block"]
            print(f"[rescore] {method:<10} batch {b:>2}: "
                  f"{'-' if ms is None else f'{ms:.2f}'} ms/block, "
                  f"peak {cell['peak_reserved_mib']} MiB, over_budget {cell['over_budget']}",
                  flush=True)
        del model, draw_fn
        _empty_cache(device)

    nll = _nll_check(provider, xs, ys, rc, device, use_amp)
    return {
        "measured_utc": _utc_now(),
        "dry_run": ctx.dry_run,
        "settings": {
            "methods": rc["methods"], "n_samples": rc["n_samples"],
            "n_samples_c0": rc["n_samples_c0"], "block_size": rc["block_size"],
            "batch_sizes": rc["batch_sizes"],
            "blocks_timed": {str(k): v for k, v in rc["blocks_timed"].items()},
            "warmup_batches": rc["warmup_batches"], "seed": rc["seed"], "amp_fp16": use_amp,
            "vram_budget_mib": budget,
        },
        "anchor": anchor,
        "cells": cells,
        "nll_check": nll,
        "checks": _rescore_checks(cells, rc, anchor, nll),
    }


# ---------------------------------------------------------------------------
# 355M shape
# ---------------------------------------------------------------------------

def run_shape(sc: dict, shape: dict, batch_size: int, device: torch.device, use_amp: bool,
              vram_budget_mib: float) -> dict:
    """Warm-up and timed BLoB training steps at one shape, with a synchronize per step."""
    from minigpt.lora import LoRAConfig, inject_lora
    from minigpt.model import GPTConfig, MiniGPT
    from minigpt.train import TrainConfig, _configure_optimizer, get_batch

    res = {"n_layer": shape["n_layer"], "n_head": shape["n_head"], "n_embd": shape["n_embd"],
           "block_size": sc["block_size"], "batch_size": batch_size, "base_params": None,
           "trainable_params": None, "params_match": None, "peak_reserved_mib": None,
           "peak_allocated_mib": None, "median_step_ms": None, "max_step_ms": None,
           "step_ms": None, "oom": False, "spill_suspected": None, "fits": False}
    torch.manual_seed(sc["seed"])
    _reset_peak(device)
    times: list[float] = []
    try:
        model = MiniGPT(GPTConfig(
            vocab_size=sc["vocab_size"], block_size=sc["block_size"], n_layer=shape["n_layer"],
            n_head=shape["n_head"], n_embd=shape["n_embd"], dropout=sc["dropout"],
            bias=sc["bias"],
        ))
        res["base_params"] = _n_params(model)
        inject_lora(model, LoRAConfig(**sc["lora"]), bayesian=True)
        res["trainable_params"] = _n_params(model, trainable=True)
        expected = shape["expected_params"]
        if expected:
            res["params_match"] = (res["base_params"] == expected["base"]
                                   and res["trainable_params"] == expected["trainable"])
        data = torch.randint(0, sc["vocab_size"], (sc["n_random_tokens"],))
        model.to(device).train()
        optimizer = _configure_optimizer(model, TrainConfig(
            lr=sc["lr"], weight_decay=sc["weight_decay"], adam_beta1=sc["adam_beta1"],
            adam_beta2=sc["adam_beta2"],
        ))
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        def step() -> None:
            x, y = get_batch(data, sc["block_size"], batch_size, device)
            with _autocast(device, use_amp):
                _, loss = model(x, y)
                total = loss + model.kl_loss() * sc["kl_scale"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), sc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

        for _ in range(sc["warmup_steps"]):
            step()
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(sc["timed_steps"]):
            t0 = time.perf_counter()
            step()
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    except torch.cuda.OutOfMemoryError:
        res["oom"] = True
    if not res["oom"]:
        reserved, allocated = _peak_mib(device)
        median, worst = statistics.median(times), max(times)
        spill = worst > sc["max_step_ratio"] * median
        res.update(peak_reserved_mib=reserved, peak_allocated_mib=allocated,
                   median_step_ms=median, max_step_ms=worst, step_ms=times,
                   spill_suspected=spill, fits=reserved <= vram_budget_mib and not spill)
    model = optimizer = step = None
    _empty_cache(device)
    print(f"[shape355m] {shape['n_layer']}L/{shape['n_embd']}d batch {batch_size}: "
          f"median {res['median_step_ms']} ms, peak {res['peak_reserved_mib']} MiB, "
          f"oom {res['oom']}, fits {res['fits']}", flush=True)
    return res


def part_shape355m(cfg: dict, ctx: _Context) -> dict:
    """S0-T3: the 355M shape at batch 4 (batches 2 and 1 if it does not fit), then 76M."""
    if cfg["probe"]["q1_backbone"] == "a":
        return {"skipped": "Q1=a"}
    sc = cfg["shape355m"]
    device = ctx.device
    budget = cfg["probe"]["vram_budget_mib"]
    use_amp = device.type == "cuda" and sc["amp_fp16"]
    big = {k: sc[k] for k in ("n_layer", "n_head", "n_embd", "expected_params")}
    small = {**big, **sc["compare"]}
    r_big = run_shape(sc, big, sc["batch_size"], device, use_amp, budget)
    fallback = None
    if not r_big["fits"]:
        fallback = [run_shape(sc, big, b, device, use_amp, budget)
                    for b in sc["fallback_batch_sizes"]]
    r_small = run_shape(sc, small, sc["batch_size"], device, use_amp, budget)
    ratio = None
    if r_big["median_step_ms"] and r_small["median_step_ms"]:
        ratio = r_big["median_step_ms"] / r_small["median_step_ms"]
    lo, hi = sc["ca5_range"]
    return {
        "measured_utc": _utc_now(),
        "dry_run": ctx.dry_run,
        "vram_budget_mib": budget,
        "amp_fp16": use_amp,
        "shapes": {"355m": r_big, "76m": r_small},
        "fits": r_big["fits"],
        "spill_suspected": r_big["spill_suspected"],
        "ca5_ratio": ratio,
        "ca5_out_of_range": None if ratio is None else not lo <= ratio <= hi,
        "fallback": fallback,
    }


# ---------------------------------------------------------------------------
# Deterministic LoRA fine-tune
# ---------------------------------------------------------------------------

def t_det_min(n_steps: int, eval_interval: int, t_step_s: float, t_eval_s: float) -> float:
    """Minutes of a deterministic LoRA fine-tune at another step count (spec Section 5)."""
    return (n_steps * t_step_s + (n_steps // eval_interval + 1) * t_eval_s) / 60.0


def part_finetune(cfg: dict, ctx: _Context) -> dict:
    """S0-T2: one deterministic LoRA fine-tune on C0, fresh optimizer, never resumed."""
    from minigpt.config import (
        apply_dict_overrides,
        build_gpt_config,
        build_lora_config,
        build_train_config,
        load_yaml,
        validate_config,
    )
    from minigpt.lora import inject_lora
    from minigpt.model import MiniGPT
    from minigpt.train import estimate_loss, load_checkpoint, train

    fc = cfg["finetune"]
    train_yaml = repo_path(fc["train_config"])
    tcfg = load_yaml(train_yaml)
    if ctx.dry_run:
        apply_dict_overrides(tcfg, cfg["dry_run"]["finetune_overrides"])
        tcfg["train"]["checkpoint_dir"] = str(repo_path(cfg["dry_run"]["checkpoint_dir"]))
    validate_config(tcfg)
    t = tcfg["train"]
    if fc["lora_type"] != "deterministic":
        raise ValueError(f"finetune.lora_type must be deterministic, got {fc['lora_type']!r}")
    for key in ("patience_evals", "patience_min_delta", "checkpoint_interval"):
        if key not in t:
            raise ValueError(f"{train_yaml}: train.{key} must be set explicitly")
    if t["kl_weight"] != 0.0 or t["kl_annealing_steps"] != 0 or t["checkpoint_interval"] != 0:
        raise ValueError("deterministic probe fine-tune needs kl_weight 0, kl_annealing_steps "
                         "0 and checkpoint_interval 0")
    base_ckpt = repo_path(fc["base_checkpoint"])
    if not ctx.dry_run:
        _require_files([base_ckpt, *_pile_cache_paths(tcfg)])

    seed = int(t["seed"])
    torch.manual_seed(seed)  # before the build and the LoRA init, as experiment_setup does
    if ctx.dry_run:
        vocab_size = cfg["dry_run"]["vocab_size"]
    else:
        from minigpt.data import get_tokenizer

        vocab_size = get_tokenizer().n_vocab
    model = MiniGPT(build_gpt_config(tcfg, vocab_size=vocab_size))
    if not ctx.dry_run:
        load_checkpoint(base_ckpt, model)
    model = inject_lora(model, build_lora_config(tcfg), bayesian=False)
    trainable = _n_params(model, trainable=True)

    if ctx.dry_run:
        gen = torch.Generator()
        gen.manual_seed(seed)
        tokens = torch.randint(0, vocab_size, (cfg["dry_run"]["n_train_tokens"],), generator=gen)
        cut = int(tokens.numel() * (1.0 - cfg["dry_run"]["val_fraction"]))
        train_data, val_data = tokens[:cut], tokens[cut:]
    else:
        from minigpt.data import get_tokenizer, load_pile_data

        data = load_pile_data(tcfg, get_tokenizer())
        train_data, val_data = data["train"], data["val"]

    device = ctx.device
    _reset_peak(device)
    t0 = time.time()
    model, meta = train(model, train_data, val_data, build_train_config(tcfg), mlflow_run=None,
                        config_dict=tcfg, kl_weight=0.0, num_train_tokens=0)
    train_call_min = (time.time() - t0) / 60.0
    peak_reserved, _ = _peak_mib(device)

    model_device = next(model.parameters()).device
    pair_s = []
    for _ in range(fc["eval_timing_repeats"]):
        _sync(model_device)
        t0 = time.perf_counter()
        for split in (train_data, val_data):
            estimate_loss(model, split, t["block_size"], t["batch_size"], model_device,
                          t["eval_iters"])
        _sync(model_device)
        pair_s.append(time.perf_counter() - t0)
    eval_s = statistics.median(pair_s)

    history = meta["eval_history"]
    n_evals = len(history)
    train_s = meta["train_time_sec"]
    steps = meta["steps_completed"]
    t_step_s = (train_s - n_evals * eval_s) / steps
    wall_min = train_s / 60.0
    ref_min = fc["blob_reference_min"]
    first_val = history[0]["val_loss"] if history else None
    checkpoint = Path(t["checkpoint_dir"]) / "ckpt_best.pt"
    section = {
        "measured_utc": _utc_now(),
        "dry_run": ctx.dry_run,
        "wall_min": wall_min,
        "train_call_min": train_call_min,
        "train_step_ms": t_step_s * 1000.0,
        "eval_s": eval_s,
        "eval_pair_s": pair_s,
        "n_evals": n_evals,
        "steps_completed": steps,
        "early_stop_reason": meta["early_stop_reason"],
        "trainable_params": trainable,
        "best_val_loss": meta["best_val_loss"],
        "best_val_step": meta["best_val_step"],
        "first_val_loss": first_val,
        "peak_reserved_mib": peak_reserved,
        "config_sha256": config_sha256(train_yaml),
        "seed_applied": seed,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_exists": (checkpoint if checkpoint.is_absolute()
                              else REPO_ROOT / checkpoint).exists(),
        "ca4_ratio": wall_min / ref_min,
        "ca4_exceeded": wall_min > fc["ca4_max_ratio"] * ref_min,
    }
    section["checks"] = {
        "steps_completed": steps == t["steps"],
        "n_evals": n_evals == t["steps"] // t["eval_interval"] + 1,
        "trainable_params": trainable == fc["expected_trainable_params"],
        "best_below_first": first_val is not None and meta["best_val_loss"] < first_val,
        "seed_applied": seed == t["seed"],
    }
    print(f"[finetune] wall {wall_min:.2f} min, step {t_step_s * 1000:.1f} ms, "
          f"eval {eval_s:.2f} s, checks {section['checks']}", flush=True)
    return section


# ---------------------------------------------------------------------------
# Estimates (CPU)
# ---------------------------------------------------------------------------

def choose_batch(cells: list[dict], method: str, vram_budget_mib: float) -> int:
    """b*: the fastest batch size of ``method`` that is not over budget."""
    feasible = [
        c for c in cells
        if c["method"] == method and not c["over_budget"] and _is_number(c.get("ms_per_block"))
        and (c.get("peak_reserved_mib") is None or c["peak_reserved_mib"] <= vram_budget_mib)
    ]
    if not feasible:
        raise NoFeasibleBatchError(
            f"score set {method!r} has no batch size within {vram_budget_mib} MiB"
        )
    return min(feasible, key=lambda c: (c["ms_per_block"], c["batch_size"]))["batch_size"]


def _rate_s(cells: list[dict], method: str, batch_size: int) -> float | None:
    for c in cells:
        if c["method"] == method and c["batch_size"] == batch_size:
            ms = c.get("ms_per_block")
            return ms / 1000.0 if _is_number(ms) else None
    return None


def estimate(cfg: dict, rescore: dict, finetune: dict | None = None) -> dict:
    """S0-T4: b*, G_m4(k), the G0 flags, G_repro and G_S5 from the measured sections."""
    ec = cfg["estimate"]
    methods = cfg["rescore"]["methods"]
    budget = cfg["probe"]["vram_budget_mib"]
    cells = rescore["cells"]
    for group in (ec["lora_methods"], ec["repro"]["methods"]):
        unknown = sorted(set(group) - set(methods))
        if unknown:
            raise ValueError(f"estimate names score sets outside rescore.methods: {unknown}")
    if 1 not in ec["blocks_per_doc"]:
        raise ValueError("estimate.blocks_per_doc must include 1")

    bstar = {m: choose_batch(cells, m, budget) for m in methods}
    rate = {m: _rate_s(cells, m, bstar[m]) for m in methods}
    rate_b1 = {m: _rate_s(cells, m, 1) for m in methods}
    sum_all = sum(rate.values())
    sum_lora = sum(rate[m] for m in ec["lora_methods"])
    per_doc_block_s = ec["sets_all_methods"] * sum_all + ec["sets_lora_only"] * sum_lora
    gpu_h = {str(k): ec["docs_per_domain"] * k * per_doc_block_s / 3600.0
             for k in ec["blocks_per_doc"]}
    k_max = max(ec["blocks_per_doc"])
    cap_applied = gpu_h[str(k_max)] > ec["cap_gpu_h"]
    k_star = 1 if cap_applied else k_max
    have_b1 = all(r is not None for r in rate_b1.values())
    sum_b1 = sum(rate_b1.values()) if have_b1 else None

    rep = ec["repro"]
    rep_scale = rep["runs"] * rep["blocks"] / 3600.0
    s5 = None
    if finetune and _is_number(finetune.get("wall_min")):
        fc = cfg["finetune"]
        s5 = {f"n_blob_{n}": (fc["blob_reference_min"] * n
                              + fc["n_det_seeds"] * finetune["wall_min"]) / 60.0
              for n in fc["n_blob_seeds"]}
    return {
        "m4": {
            "gpu_h": gpu_h,
            "batch_size": bstar,
            "rate_s_per_block": rate,
            "sum_rate_s": sum_all,
            "sum_rate_lora_s": sum_lora,
            "cap_gpu_h": ec["cap_gpu_h"],
            "cap_applied": cap_applied,
            "blocks_per_doc": k_star,
            "escalate_g0": gpu_h["1"] > ec["cap_gpu_h"],
            "low_end_holds": gpu_h["1"] <= ec["low_end_gpu_h"],
            "m4_nights_needed": math.ceil(gpu_h[str(k_star)] / ec["night_gpu_h"]),
            "speedup_vs_ca10": ec["ca10_sum_s"] / sum_all,
            "speedup_measured": None if sum_b1 is None else sum_b1 / sum_all,
        },
        "repro": {
            "gpu_h_b1": (rep_scale * sum(rate_b1[m] for m in rep["methods"])
                         if all(rate_b1[m] is not None for m in rep["methods"]) else None),
            "gpu_h_bstar": rep_scale * sum(rate[m] for m in rep["methods"]),
        },
        "s5_finetune_gpu_h": s5,
    }


def part_estimate(cfg: dict, ctx: _Context) -> dict:
    doc = load_json(ctx.output)
    validate_schema(doc, require=("rescore",))
    section = estimate(cfg, doc["rescore"], doc.get("finetune"))
    section["measured_utc"] = _utc_now()
    section["dry_run"] = ctx.dry_run
    m4 = section["m4"]
    print(f"[estimate] G_m4 {m4['gpu_h']} GPU h, blocks_per_doc {m4['blocks_per_doc']}, "
          f"cap {m4['cap_applied']}, escalate {m4['escalate_g0']}, "
          f"nights {m4['m4_nights_needed']}", flush=True)
    return section


# ---------------------------------------------------------------------------
# Hand-off check (S0-T5)
# ---------------------------------------------------------------------------

def _find_blocks_values(obj: Any) -> list[Any]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in BLOCKS_KEYS:
                found.append(value)
            else:
                found.extend(_find_blocks_values(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_blocks_values(item))
    return found


def check_plan(json_path: str | Path, m4_config_path: str | Path, g0_ack: bool) -> tuple[int, str]:
    """Compare S1's m4 YAML with the probe estimates. Returns (exit code, message)."""
    json_path = Path(json_path)
    if not json_path.exists():
        return EXIT_NO_PLAN, f"timing probe JSON not found: {json_path} (run --part estimate)"
    try:
        doc = load_json(json_path)
    except (OSError, json.JSONDecodeError) as exc:
        return EXIT_NO_PLAN, f"cannot read {json_path}: {exc}"
    m4 = (doc.get("estimates") or {}).get("m4") if isinstance(doc, dict) else None
    if not isinstance(m4, dict) or not {"blocks_per_doc", "escalate_g0"} <= set(m4):
        return EXIT_NO_PLAN, f"{json_path} has no estimates.m4 (run --part estimate)"

    m4_path = Path(m4_config_path)
    try:
        plan = yaml.safe_load(m4_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return EXIT_BAD_INPUT, f"cannot read the m4 YAML {m4_path}: {exc}"
    values = _find_blocks_values(plan)
    if not values or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return EXIT_BAD_INPUT, (f"the m4 YAML {m4_path} must set blocks_per_doc (or "
                                f"max_blocks_per_doc) to an integer; found {values}")
    yaml_blocks = max(values)
    json_blocks = int(m4["blocks_per_doc"])
    if yaml_blocks > json_blocks:
        return EXIT_BLOCKS_MISMATCH, (
            f"blocks_per_doc mismatch: the m4 YAML {m4_path} sets {yaml_blocks}, "
            f"the timing probe allows at most {json_blocks}"
        )
    if m4["escalate_g0"] and not g0_ack:
        g1 = (m4.get("gpu_h") or {}).get("1")
        return EXIT_G0_UNACKED, (
            f"escalate_g0 is true (G_m4(1) = {g1} GPU h is above the cap); pass --g0-ack "
            "once the G0 decision is recorded"
        )
    return EXIT_OK, (f"m4 plan ok: blocks_per_doc {yaml_blocks} <= {json_blocks}, "
                     f"escalate_g0 {m4['escalate_g0']}, g0_ack {g0_ack}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class _Context:
    device: torch.device
    dry_run: bool
    output: Path


PART_FUNCS: dict[str, Callable[[dict, _Context], dict]] = {
    "shape355m": part_shape355m,
    "rescore": part_rescore,
    "finetune": part_finetune,
    "estimate": part_estimate,
}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # keep exit 2 for the blocks mismatch of --part check
        self.print_usage(sys.stderr)
        self.exit(EXIT_BAD_INPUT, f"{self.prog}: error: {message}\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = _Parser(description="S0 timing probe (specs/i2-timing-probe.md)")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--part", choices=PARTS, required=True)
    p.add_argument("--m4-config", type=str, default=None,
                   help="S1's m4 YAML (--part check)")
    p.add_argument("--g0-ack", action="store_true",
                   help="G0 escalation acknowledged (--part check)")
    p.add_argument("--dry-run", action="store_true",
                   help="tiny random CPU models from the dry_run section; no data/ reads")
    return p.parse_args(argv)


def run_all(config_path: Path, dry_run: bool) -> int:
    """Run shape355m, rescore, finetune and estimate, one subprocess each."""
    failed = []
    for part in ALL_PARTS:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
               "--part", part]
        if dry_run:
            cmd.append("--dry-run")
        print(f"[all] {' '.join(cmd)}", flush=True)
        if subprocess.call(cmd) != 0:
            failed.append(part)
    if failed:
        print(f"[all] failed parts: {failed}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_import_paths()  # minigpt and the eval scripts, without an installed package
    config_path = Path(args.config).resolve()
    if args.part == "all":
        return run_all(config_path, args.dry_run)
    try:
        cfg = load_config(config_path, dry_run=args.dry_run)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot load {config_path}: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    output = repo_path(cfg["probe"]["output"])

    if args.part == "check":
        if args.m4_config is None:
            print("--part check needs --m4-config", file=sys.stderr)
            return EXIT_BAD_INPUT
        code, message = check_plan(output, Path(args.m4_config).resolve(), args.g0_ack)
        print(message, file=sys.stdout if code == EXIT_OK else sys.stderr)
        return code

    if args.part == "estimate":
        device = torch.device("cpu")
    else:
        device = torch.device(cfg["dry_run"]["device"] if args.dry_run else cfg["probe"]["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"--part {args.part} needs CUDA (or --dry-run)", file=sys.stderr)
            return EXIT_FAILED
    if not args.dry_run and args.part in ("rescore", "finetune"):
        os.chdir(REPO_ROOT)  # minigpt.data and the eval scripts use paths relative to the repo
    header = collect_header(cfg, config_path, args.dry_run)
    ctx = _Context(device=device, dry_run=args.dry_run, output=output)
    section = PART_FUNCS[args.part](cfg, ctx)
    merge_section(output, SECTION_OF_PART[args.part], section, header)
    print(f"[{args.part}] wrote section {SECTION_OF_PART[args.part]!r} to {output}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
