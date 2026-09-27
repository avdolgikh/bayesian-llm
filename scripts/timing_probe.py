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
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from timing_probe_common import (  # noqa: F401
    _SECTION_VALIDATORS,
    ANCHOR_FIELDS,
    CELL_FIELDS,
    CELL_MEASURED_FIELDS,
    CONFIG_SECTIONS,
    ESTIMATES_FIELDS,
    FINETUNE_FIELDS,
    HEADER_FIELDS,
    M4_FIELDS,
    NLL_FIELDS,
    OPERATOR_STATEMENTS,
    REPO_ROOT,
    REPRO_FIELDS,
    SCHEMA,
    SCRIPTS_DIR,
    SHAPE355M_FIELDS,
    SHAPE_FIELDS,
    NoFeasibleBatchError,
    SchemaError,
    _autocast,
    _compute_apps,
    _Context,
    _empty_cache,
    _ensure_import_paths,
    _gpu_query,
    _is_number,
    _n_params,
    _num,
    _operator_statement,
    _peak_mib,
    _pile_cache_paths,
    _require,
    _require_files,
    _reset_peak,
    _run,
    _sleep_disabled,
    _sync,
    _utc_now,
    _validate_estimates,
    _validate_rescore,
    _validate_shape355m,
    collect_header,
    config_sha256,
    load_config,
    load_json,
    merge_section,
    other_compute_procs,
    repo_path,
    validate_schema,
)
from timing_probe_rescore import (  # noqa: F401
    _DryProvider,
    _empty_cell,
    _gaussian_draw_fn,
    _logits_fn,
    _mode_context,
    _nll_check,
    _RealProvider,
    _rescore_checks,
    _score_one_batch,
    part_rescore,
    per_block_nll,
    run_anchor,
    run_cell,
    score_batch,
    score_blocks,
    time_sampler,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "i2_timing_probe.yaml"

ALL_PARTS = ("shape355m", "rescore", "finetune", "estimate")
PARTS = (*ALL_PARTS, "check", "all")
SECTION_OF_PART = {
    "shape355m": "shape355m",
    "rescore": "rescore",
    "finetune": "finetune",
    "estimate": "estimates",
}
BLOCKS_KEYS = ("blocks_per_doc", "max_blocks_per_doc")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKS_MISMATCH = 2
EXIT_G0_UNACKED = 3
EXIT_NO_PLAN = 4
EXIT_BAD_INPUT = 5


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
