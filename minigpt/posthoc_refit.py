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

import hashlib
import io
import json
import math
import os
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from minigpt.laplace import (
    LaplaceState,
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
from minigpt.refit_data import (  # noqa: F401  (re-exported for tests and scripts)
    FitSet,
    RefitCheckError,
    RefitData,
    _first_overlap,
    _fit_rows,
    _NoStreamTokenizer,
    _test_intervals,
    _test_rows,
    build_fit_blocks,
    check_curvature_isolation,
    check_fit_blocks,
    check_fit_isolation,
    doc_ids_sha256,
    fit_batches,
    load_refit_data,
    read_manifest,
)
from minigpt.refit_nll import (  # noqa: F401  (re-exported for tests and scripts)
    AUTOCAST_MODES,
    EXIT_BISECTION_FAILED,
    EXIT_CHECK_FAILED,
    EXIT_NO_BUDGET,
    EXIT_OK,
    MATCH_EXIT_CODES,
    _token_logprobs,
    autocast_context,
    map_nll,
    match_prior_precision,
    measure_delta_nll,
)
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
