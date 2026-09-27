"""Delta-NLL measurement and the prior-precision match of the S2 post-hoc refits.

Split out of ``minigpt.posthoc_refit``, which re-exports every name.
"""

from __future__ import annotations

import contextlib
import math
import statistics
from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from minigpt.laplace import apply_sampled_params

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
