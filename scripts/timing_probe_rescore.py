"""Batched MC statistics and re-score timing (S0-T1) of the S0 timing probe
(scripts/timing_probe.py)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from timing_probe_common import (
    CELL_MEASURED_FIELDS,
    REPO_ROOT,
    _autocast,
    _Context,
    _empty_cache,
    _ensure_import_paths,
    _is_number,
    _peak_mib,
    _pile_cache_paths,
    _require_files,
    _reset_peak,
    _sync,
    _utc_now,
    other_compute_procs,
)
from torch.nn import functional as F

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
