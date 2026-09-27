"""Evaluate C checkpoints: the I2 scorer (score files) and the D1 tables.

I2 path (specs/i2-eval-rebuild.md, sections 5.5 and 5.7). ``--eval-set`` scores the blocks
of one eval set and writes one score file per score set, at
``{scoring.out_dir}/{score_set}__{eval_set}__{run_tag}.pt``. Every parameter comes from the
eval config named by ``--eval-config`` (``configs/i2_eval.yaml`` for real runs). The sets
``test``, ``test_hn`` and ``arxiv_stripped`` are refused unless ``prereg.json`` in
``scoring.out_dir`` holds the sha256 of the config's ``analysis`` block.

    python scripts/eval_c_checkpoints.py --eval-config configs/i2_eval.yaml \
        --eval-set legacy_d1 --run-tag main
    python scripts/eval_c_checkpoints.py --eval-config configs/i2_eval.yaml \
        --eval-set legacy_d1 --run-tag rerun --block-ids 0-99,500-599
    python scripts/eval_c_checkpoints.py --eval-config configs/i2_eval.yaml \
        --eval-set legacy_d1 --run-tag spread1 --score-set c1 --seed-base 1000000
    python scripts/eval_c_checkpoints.py --eval-config configs/i2_eval.yaml --analyze

``mc_dropout`` is scored by scripts/eval_mc_dropout.py, which imports this engine.
S2's refits (specs/i2-posthoc-fixes.md, sections 2 and 5.5) are the score sets ``c2_refit``,
``c4_lap_refit`` and ``c4_tfb_fixed``: the base through ``minigpt.posthoc_refit``, the v2
state from ``data/checkpoints/i2_posthoc/<cell>/`` and the v2 samplers. The legacy sets
``c2``, ``c4_lap`` and ``c4_tfb`` keep their pre-fix (v1_legacy) states and samplers.
``--analyze`` reads the ``main`` score files of ``test``, ``test_hn`` and ``arxiv_stripped``
and writes ``table_<eval_set>.json``, ``families.json`` and ``descriptive.json`` to
``scoring.out_dir`` (the schema that ``scripts/check_eval_rebuild.py --rederive`` checks).

Block files (written by minigpt/evalset.py): ``{eval_set.out_dir}/blocks_{eval_set}.pt``
holds at least
    tokens  int [B, T+1]    x = tokens[:, :-1], y = tokens[:, 1:]
    doc_id  list[str] [B]   each id is a ``split: test`` row of ``manifest.jsonl``
    domain  list[str] [B]
    offset  int64 [B]       token offset of the block start inside its document
    weight  float [B]       1 / k_d, k_d = number of blocks of that document in the file
The row position in the file is the block index b.

D1 path (no ``--eval-set``): the old tables, for example
    python scripts/eval_c_checkpoints.py --from-scores data/d1_scores.pt --bootstrap
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eval_c_analyze import (  # noqa: F401
    Triple,
    _cell_sources,
    _CellData,
    _decision,
    _domain_for_role,
    _domain_index,
    _roles,
    _write_json,
    planned_table_cells,
    run_analyze,
)
from eval_c_legacy import (  # noqa: F401
    _fmt_ci,
    compute_bootstrap_cis,
    legacy_mi_ratio,
    load_scores,
    print_primary_table,
    print_secondary_table,
    save_scores,
)
from eval_c_load import (  # noqa: F401
    ALL_MILESTONES,
    ANALYSIS_EVAL_SETS,
    ANALYSIS_RUN_TAG,
    BLOCK_SIZE,
    CKPT_DIR,
    EVAL_SETS,
    EXTERNAL_SCORE_SETS,
    FROZEN_EVAL_SETS,
    GLOBAL_RNG_METHODS,
    ID_ROLE_EVAL_SET,
    LABELS,
    LOG_EVERY,
    MANIFEST_FILE,
    PREREG_FILE,
    PROB_EPS,
    RUN_TAG_RE,
    STRIPPED_OOD,
    BlockSet,
    _blob_to_deterministic_lora,
    _checkpoint_paths,
    _extract_sequences,
    _legacy_split,
    build_legacy_blocks,
    load_eval_data,
    load_i2_blocks,
    load_legacy_blocks,
    load_model,
    parse_block_ids,
    repo_path,
)
from eval_c_meta import (  # noqa: F401
    WarningLog,
    _fail,
    _id_base_domain,
    _resolve_device,
    add_i2_arguments,
    apply_determinism,
    check_prereg,
    code_sha256,
    file_sha256,
    git_state,
    load_eval_config,
    mi_ratio_lines,
    resolve_seed_base,
    score_file_path,
    scores_dir,
    select_score_sets,
    summary_lines,
)

from experiments.c_milestones import build_milestone_config  # noqa: F401
from minigpt.evalset import PreregError, analysis_sha256
from minigpt.laplace import apply_sampled_params, load_laplace_state, sample_laplace_params
from minigpt.layers import enable_dropout
from minigpt.model import MiniGPT
from minigpt.posthoc_refit import (
    EXIT_OK,
    SCORE_SET_CELLS,
    SCORED_STATUSES,
    check_config,
    load_posthoc_base,
)
from minigpt.tfb import load_tfb_state, sample_tfb_params
from minigpt.uncertainty import aggregate_sequence_scores, auprc, aurc, auroc, ece, fpr_at_tpr

# ---------------------------------------------------------------------------
# Score-set registry (the hook S2 uses for its v2 samplers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreSetEntry:
    """How the scorer builds one score set.

    ``checkpoint_paths()`` lists the files that ``load`` reads (their sha256 goes into meta).
    ``load(device)`` returns ``(model, method, posthoc_state)``. ``method`` picks the draw:
    ``deterministic``, ``variational`` and ``dropout`` use the global RNG, reseeded with
    ``seed_b`` before the N passes; any key of ``PARAM_SAMPLERS`` samples parameters with
    ``PARAM_SAMPLERS[method](posthoc_state, seed=seed_b * N + s)``.
    """

    checkpoint_paths: Callable[[], list[Path]]
    load: Callable[[torch.device], tuple[nn.Module, str, Any]]


SCORE_SET_REGISTRY: dict[str, ScoreSetEntry] = {}
PARAM_SAMPLERS: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
    "laplace": sample_laplace_params,
    "tfb": sample_tfb_params,
}


def register_score_set(
    name: str,
    *,
    checkpoint_paths: Callable[[], list[Path]],
    load: Callable[[torch.device], tuple[nn.Module, str, Any]],
) -> None:
    """Make ``name`` scorable. Its YAML entries (labels, n_samples, score_sets) must exist."""
    if name in SCORE_SET_REGISTRY or name in EXTERNAL_SCORE_SETS:
        raise ValueError(f"score set {name!r} is already registered")
    SCORE_SET_REGISTRY[name] = ScoreSetEntry(checkpoint_paths=checkpoint_paths, load=load)


def register_param_sampler(
    method: str, sampler: Callable[..., dict[str, torch.Tensor]],
) -> None:
    """Add a parameter sampler ``sampler(state, seed=...) -> {param_name: tensor}``."""
    if method in PARAM_SAMPLERS or method in GLOBAL_RNG_METHODS:
        raise ValueError(f"sampler method {method!r} is already registered")
    PARAM_SAMPLERS[method] = sampler


def _register_builtin_score_sets() -> None:
    # Late binding: the lambdas read CKPT_DIR and the loaders when they run.
    for name in ALL_MILESTONES:
        register_score_set(
            name,
            checkpoint_paths=lambda m=name: _checkpoint_paths(m),
            load=lambda device, m=name: load_model(m, device),
        )


_register_builtin_score_sets()


# ---------------------------------------------------------------------------
# S2 score sets: the refit states and the v2 samplers (specs/i2-posthoc-fixes.md 2, 5.5)
# ---------------------------------------------------------------------------
# The legacy score sets above keep the "tfb" and "laplace" samplers and their pre-fix states
# (v1_legacy). The three S2 score sets draw with "tfb_v2" and "laplace_v2", which refuse any
# state that is not v2. Each S2 score set reads its refit YAML (scripts/refit_posthoc.py
# --config); the YAML names the base checkpoint and its kind, the model and LoRA shape, and
# out_dir, which holds the state file and fit_record.json. The paths inside it are read as
# the refit reads them (relative to the working directory, like CKPT_DIR).

SAMPLER_V2 = "v2"
POSTHOC_REFIT_CONFIGS: dict[str, Path] = {
    "c2_refit": Path("configs/i2_posthoc_c2.yaml"),
    "c4_lap_refit": Path("configs/i2_posthoc_c4_lap.yaml"),
    "c4_tfb_fixed": Path("configs/i2_posthoc_c4_tfb.yaml"),
}
POSTHOC_REFIT_METHODS = {"c2_refit": "laplace", "c4_lap_refit": "laplace", "c4_tfb_fixed": "tfb"}
POSTHOC_STATE_FILES = {"tfb": "tfb_state.pt", "laplace": "laplace_state.pt"}
FIT_RECORD_FILE = "fit_record.json"


def _v2_only(sampler: Callable[..., dict[str, torch.Tensor]], family: str):
    """``sampler`` restricted to v2 states (the sampler itself dispatches on the version)."""
    def sample_v2(state: Any, seed: int | None = None) -> dict[str, torch.Tensor]:
        if state.sampler_version != SAMPLER_V2:
            raise ValueError(f"the {family}_{SAMPLER_V2} sampler needs a {SAMPLER_V2} state, "
                             f"got {state.sampler_version!r}")
        return sampler(state, seed=seed)

    sample_v2.__name__ = f"sample_{family}_{SAMPLER_V2}"
    return sample_v2


register_param_sampler(f"tfb_{SAMPLER_V2}", _v2_only(sample_tfb_params, "tfb"))
register_param_sampler(f"laplace_{SAMPLER_V2}", _v2_only(sample_laplace_params, "laplace"))


def posthoc_refit_config(name: str) -> dict:
    """The checked refit YAML of S2 score set ``name`` (KeyError or ValueError if invalid)."""
    path = repo_path(POSTHOC_REFIT_CONFIGS[name])
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    check_config(cfg)
    expected = (POSTHOC_REFIT_METHODS[name], SCORE_SET_CELLS[name])
    if (cfg["method"], cfg["cell"]) != expected:
        raise ValueError(f"{path}: score set {name} needs method {expected[0]!r} and cell "
                         f"{expected[1]!r}, got {cfg['method']!r} and {cfg['cell']!r}")
    return cfg


def _posthoc_files(cfg: dict) -> tuple[Path, Path, Path]:
    """(base checkpoint, state file, fit record) of a refit YAML."""
    out_dir = Path(cfg["out_dir"])
    return (Path(cfg["base"]["base_checkpoint"]), out_dir / POSTHOC_STATE_FILES[cfg["method"]],
            out_dir / FIT_RECORD_FILE)


def posthoc_checkpoint_paths(name: str) -> list[Path]:
    """The base checkpoint and the state file of S2 score set ``name`` (hashed into meta)."""
    base, state, _ = _posthoc_files(posthoc_refit_config(name))
    return [base, state]


def _passed_fit_record(name: str, cfg: dict, record_path: Path, state_path: Path) -> dict:
    """The fit record, if the fit passed and wrote this state file; ValueError otherwise."""
    if not record_path.exists():
        raise ValueError(f"{name}: no fit record at {record_path}; run "
                         f"scripts/refit_posthoc.py --config {POSTHOC_REFIT_CONFIGS[name]}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    problems = []
    if (record.get("method"), record.get("cell")) != (cfg["method"], cfg["cell"]):
        problems.append(f"it is for method {record.get('method')!r}, cell {record.get('cell')!r}")
    if record.get("exit_code") != EXIT_OK or record.get("failures"):
        problems.append(f"the fit did not pass (exit_code {record.get('exit_code')!r}, "
                        f"failures {record.get('failures')!r})")
    if cfg["method"] == "laplace" and record.get("match_status") not in SCORED_STATUSES:
        problems.append(f"match_status {record.get('match_status')!r} is not scored "
                        f"(scored: {', '.join(SCORED_STATUSES)})")
    if record.get("state_sha256") != file_sha256(state_path):
        problems.append(f"{state_path} is not the state file the fit wrote")
    if problems:
        raise ValueError(f"{name} ({record_path}): " + "; ".join(problems))
    return record


def load_posthoc_refit(name: str, device: torch.device) -> tuple[nn.Module, str, Any]:
    """Load S2 score set ``name``: the base through ``load_posthoc_base``, the v2 state.

    Refuses (ValueError) a fit that did not pass, a state file or base checkpoint other than
    the ones in the fit record, and a state that is not v2. Returns (model, method, state)
    with method ``tfb_v2`` or ``laplace_v2``.
    """
    cfg = posthoc_refit_config(name)
    base_path, state_path, record_path = _posthoc_files(cfg)
    record = _passed_fit_record(name, cfg, record_path, state_path)
    model, info = load_posthoc_base(cfg, device=device)
    if record["base"]["sha256"] != info["sha256"]:
        raise ValueError(f"{name}: {base_path} has sha256 {info['sha256'][:12]}, but the fit "
                         f"used {record['base']['sha256'][:12]} ({record_path})")
    load_state = load_tfb_state if cfg["method"] == "tfb" else load_laplace_state
    state = load_state(state_path, map_location=device)
    if state.sampler_version != SAMPLER_V2:
        raise ValueError(f"{name}: {state_path} is a {state.sampler_version!r} state; the S2 "
                         f"score sets need {SAMPLER_V2!r}")
    method = f"{cfg['method']}_{SAMPLER_V2}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} params, method={method}, base {info['kind']} "
          f"{base_path}, state {state_path}")
    return model, method, state


def _register_posthoc_refit_score_sets() -> None:
    # Late binding, as for the built-in sets: the lambdas read POSTHOC_REFIT_CONFIGS on call.
    for name in POSTHOC_REFIT_METHODS:
        register_score_set(
            name,
            checkpoint_paths=lambda n=name: posthoc_checkpoint_paths(n),
            load=lambda device, n=name: load_posthoc_refit(n, device),
        )


_register_posthoc_refit_score_sets()


# ---------------------------------------------------------------------------
# Scorer (section 5.5)
# ---------------------------------------------------------------------------

def realized_token_scores(logp_real: torch.Tensor) -> dict[str, torch.Tensor]:
    """Realized-token Jensen gap from per-sample log-probs (spec section 3).

    ``logp_real``: [..., N, T] with ell[s, t] = log p_s(y_t | x_<=t). Returns float64
    ``log_pbar`` [..., T] = logsumexp_s ell - log N, ``g`` [..., T] = log_pbar - mean_s ell,
    and ``G`` [...] = mean_t g.
    """
    ell = logp_real.double()
    log_pbar = torch.logsumexp(ell, dim=-2) - math.log(ell.shape[-2])
    g = log_pbar - ell.mean(dim=-2)
    return {"log_pbar": log_pbar, "g": g, "G": g.mean(dim=-1)}


def _forward(model: nn.Module, x: torch.Tensor, device: torch.device, amp: bool):
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits, _ = model(x)
    return logits


@torch.no_grad()
def score_block_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    n_samples: int,
    device: torch.device,
    method: str,
    state: Any,
    seed_b: int | None,
    amp: bool,
) -> dict[str, torch.Tensor]:
    """N passes over k blocks that share one weight draw. Returns CPU tensors.

    ``x``, ``y``: [k, T]. With ``seed_b`` set, the global RNG is reseeded with ``seed_b``
    before the passes and parameter samplers get seed ``seed_b * N + s``; ``None`` keeps
    the RNG as it is (the unseeded D1 path of eval_mc_dropout.py).

    Keys: ``logp_real`` [k, N, T]; ``tok_mi``, ``tok_tu``, ``tok_au``, ``tok_maxprob``,
    ``tok_sum_p_sq``, ``tok_pbar_true`` [k, T] float32; ``tok_correct`` [k, T] bool.
    """
    if method not in GLOBAL_RNG_METHODS and method not in PARAM_SAMPLERS:
        raise ValueError(f"unknown sampler method {method!r}")
    x_dev = x.to(device)
    y_dev = y.to(device)
    k, seq_len = x.shape
    vocab_size = model.config.vocab_size

    p_sum = torch.zeros(k, seq_len, vocab_size, device=device)
    entropy_sum = torch.zeros(k, seq_len, device=device)
    ell = torch.empty(k, n_samples, seq_len, device=device)

    if seed_b is not None:
        torch.manual_seed(seed_b)
    ctx = enable_dropout(model) if method == "dropout" else nullcontext()
    with ctx:
        for s in range(n_samples):
            if method in PARAM_SAMPLERS:
                seed = None if seed_b is None else seed_b * n_samples + s
                sampled = PARAM_SAMPLERS[method](state, seed=seed)
                with apply_sampled_params(model, sampled):
                    logits = _forward(model, x_dev, device, amp)
            else:
                logits = _forward(model, x_dev, device, amp)
            lp = torch.log_softmax(logits.float(), dim=-1)
            probs = lp.exp()
            p_sum.add_(probs)
            entropy_sum.add_(-(probs * torch.log(probs + PROB_EPS)).sum(dim=-1))
            ell[:, s, :] = lp.gather(-1, y_dev.unsqueeze(-1)).squeeze(-1)
            del logits, lp, probs      # free vocab-sized tensors before the next forward pass

    p_bar = p_sum.div_(n_samples)     # in place: one vocab-sized buffer instead of two
    pred_entropy = -(p_bar * torch.log(p_bar + PROB_EPS)).sum(dim=-1)
    exp_entropy = entropy_sum / n_samples
    mi = pred_entropy - exp_entropy
    max_prob = p_bar.max(dim=-1).values

    correct = p_bar.argmax(dim=-1) == y_dev
    p_true = p_bar.gather(-1, y_dev.unsqueeze(-1)).squeeze(-1)
    sum_p_sq = (p_bar ** 2).sum(dim=-1)

    return {
        "logp_real": ell.cpu(),
        "tok_mi": mi.cpu(),
        "tok_tu": pred_entropy.cpu(),
        "tok_au": exp_entropy.cpu(),
        "tok_maxprob": max_prob.cpu(),
        "tok_sum_p_sq": sum_p_sq.cpu(),
        "tok_correct": correct.cpu(),
        "tok_pbar_true": p_true.cpu(),
    }


def _batches(block_index: list[int], batch_size: int) -> list[list[int]]:
    """Row positions in file order, split where b is not consecutive, then by batch size."""
    if batch_size < 1:
        raise ValueError(f"batch size must be >= 1, got {batch_size}")
    batches: list[list[int]] = []
    run: list[int] = []
    for pos, b in enumerate(block_index):
        if run and (b != block_index[run[-1]] + 1 or len(run) == batch_size):
            batches.append(run)
            run = []
        run.append(pos)
    if run:
        batches.append(run)
    return batches


def _block_mean(t: torch.Tensor) -> torch.Tensor:
    return t.double().mean(dim=-1).float()


_SAVED_TOKEN_KEYS = (
    "logp_real", "tok_mi", "tok_tu", "tok_au", "tok_maxprob", "tok_sum_p_sq", "tok_correct",
)


def score_blocks(
    model: nn.Module,
    method: str,
    state: Any,
    blocks: BlockSet,
    *,
    n_samples: int,
    seed_start: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
    name: str = "",
) -> dict:
    """Score every block and return the score-file tensors (section 5.5, without meta).

    Block b is seeded with ``seed_start + b`` (seed_start = seed_base + seed_offset). With
    batch size above 1, the blocks of a batch share the draw seeded by its first block.
    """
    b_list = blocks.block_index.tolist()
    parts: dict[str, list[torch.Tensor]] = {k: [] for k in _SAVED_TOKEN_KEYS}
    t0 = time.time()
    done = 0
    for rows in _batches(b_list, batch_size):
        tok = blocks.tokens[torch.tensor(rows)].long()
        out = score_block_batch(
            model, tok[:, :-1], tok[:, 1:], n_samples, device, method, state,
            seed_b=seed_start + b_list[rows[0]], amp=amp,
        )
        for key in _SAVED_TOKEN_KEYS:
            parts[key].append(out[key])
        done += len(rows)
        if done == len(rows) or done // LOG_EVERY != (done - len(rows)) // LOG_EVERY:
            print(f"  {name} {done}/{len(b_list)} blocks ({time.time() - t0:.0f}s)")
    tok_scores = {k: torch.cat(v) for k, v in parts.items()}
    rs = realized_token_scores(tok_scores["logp_real"])
    payload = dict(tok_scores)
    payload.update({
        "blk_g": rs["G"].float(),
        "blk_mi": _block_mean(tok_scores["tok_mi"]),
        "blk_tu": _block_mean(tok_scores["tok_tu"]),
        "blk_au": _block_mean(tok_scores["tok_au"]),
        "blk_nll": _block_mean(-rs["log_pbar"]),
        "blk_maxprob_unc": _block_mean(1.0 - tok_scores["tok_maxprob"].double()),
        "block_index": blocks.block_index.clone(),
        "doc_id": list(blocks.doc_id),
        "domain": list(blocks.domain),
        "offset": blocks.offset.clone(),
        "weight": blocks.weight.clone(),
    })
    return payload


# ---------------------------------------------------------------------------
# Eval config, freeze check, determinism, meta
# ---------------------------------------------------------------------------

def scoring_code_paths(script_path: Path) -> list[Path]:
    """Scripts hashed by code_sha256: the caller, this script and its eval_c_* modules."""
    here = Path(__file__).resolve()
    return [Path(script_path), here, *sorted(here.parent.glob("eval_c_*.py"))]


def run_i2(
    args: argparse.Namespace,
    *,
    script_path: Path,
    owned: set[str],
    scored_elsewhere: Callable[[str], str | None],
    load_score_set: Callable[[str, torch.device], tuple[nn.Module, str, Any]],
    checkpoint_paths: Callable[[str], list[Path]],
    load_legacy: Callable[[int, int], BlockSet],
) -> list[Path]:
    """Score ``args.eval_set`` and write one score file per score set. Returns the paths."""
    try:
        if args.eval_config is None or args.run_tag is None:
            raise ValueError("--eval-set needs --eval-config and --run-tag")
        if not RUN_TAG_RE.fullmatch(args.run_tag) or "__" in args.run_tag:
            raise ValueError(f"--run-tag {args.run_tag!r} must match {RUN_TAG_RE.pattern}")
        cfg, yaml_sha = load_eval_config(args.eval_config)
        scoring = cfg["scoring"]
        eval_set = args.eval_set
        if eval_set not in scoring["score_sets"]:
            raise ValueError(f"eval set {eval_set!r} is not in scoring.score_sets")
        if eval_set in FROZEN_EVAL_SETS:
            check_prereg(cfg)
        seed_base = resolve_seed_base(cfg, args.seed_base)
        requested = (None if args.score_set is None
                     else [s.strip() for s in args.score_set.split(",")])
        to_run, skipped = select_score_sets(
            scoring["score_sets"][eval_set], requested, owned, scored_elsewhere,
        )
        settings = {s: (scoring["n_samples"][s], scoring["labels"][s]) for s in to_run}
        # Every input file must exist before the first score set is scored (an S2 score set
        # has no state until scripts/refit_posthoc.py has run).
        missing = [(s, p) for s in to_run for p in checkpoint_paths(s) if not p.exists()]
        if missing:
            raise ValueError(
                "missing input files, so nothing was scored: "
                + ", ".join(f"{s}: {p.as_posix()}" for s, p in missing)
                + "; pass --score-set with the score sets whose files exist"
            )
        paths = {s: score_file_path(cfg, s, eval_set, args.run_tag) for s in to_run}
        existing = [p for p in paths.values() if p.exists()]
        if existing:
            raise ValueError(
                f"refusing to overwrite {existing[0]}; pass --score-set with the sets not "
                f"scored yet, or use a new --run-tag"
            )
        batch_size = (scoring["batch_size"][eval_set] if args.batch_size is None
                      else args.batch_size)
        if batch_size < 1:
            raise ValueError(f"batch size must be >= 1, got {batch_size}")
    except (PreregError, ValueError) as exc:
        _fail(str(exc))

    for s in skipped:
        print(f"Skipping {s}: scored by {scored_elsewhere(s)}")
    apply_determinism(scoring["determinism"])
    device = _resolve_device(args.device)
    block_size = cfg["eval_set"]["block_size"]
    if eval_set == "legacy_d1":
        blocks = load_legacy(args.n_sequences, block_size)
    else:
        blocks = load_i2_blocks(cfg, eval_set)
    if args.block_ids is not None:
        try:
            blocks = blocks.select(parse_block_ids(args.block_ids, len(blocks)))
        except ValueError as exc:
            _fail(str(exc))

    amp = bool(scoring["amp_fp16"]) and device.type == "cuda"
    seed_offset = scoring["seed_offsets"][eval_set]
    git_sha, git_dirty = git_state()
    code_sha = code_sha256(scoring_code_paths(script_path))
    print(f"Device: {device}  eval set: {eval_set}  blocks: {len(blocks)}  "
          f"batch size: {batch_size}  seed base: {seed_base}  offset: {seed_offset}")
    written = []
    for s in to_run:
        n_samples, label = settings[s]
        print(f"\n{'=' * 60}\n{s}: {label['display']} (N={n_samples})\n{'=' * 60}")
        with WarningLog() as wlog:
            model, method, state = load_score_set(s, device)
            payload = score_blocks(
                model, method, state, blocks, n_samples=n_samples,
                seed_start=seed_base + seed_offset, batch_size=batch_size, device=device,
                amp=amp, name=s,
            )
        payload["meta"] = {
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "code_sha256": code_sha,
            "checkpoint_sha256": {p.as_posix(): file_sha256(p) for p in checkpoint_paths(s)},
            "score_set": s,
            "display_label": label["display"],
            "sampler": label["sampler"],
            "adapter_source": label["adapter_source"],
            "eval_set": eval_set,
            "run_tag": args.run_tag,
            "block_ids": args.block_ids,
            "n_samples": n_samples,
            "seed_base": seed_base,
            "seed_offset": seed_offset,
            "batch_size": batch_size,
            "amp_fp16": amp,
            "device": str(device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "determinism_warnings": list(wlog.messages),
            "manifest_sha256": blocks.manifest_sha256,
            "yaml_sha256": yaml_sha,
            "analysis_sha256": analysis_sha256(cfg),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        paths[s].parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, paths[s])
        written.append(paths[s])
        print(f"Saved {paths[s]}")
        for line in summary_lines(paths[s], cfg, args.n_bootstrap if args.bootstrap else None):
            print(line)
        del model, state, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return written


# ---------------------------------------------------------------------------
# D1 per-sequence scoring (old tables; routed through the I2 scorer)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_sequence_full(
    model: MiniGPT,
    x: torch.Tensor,
    targets: torch.Tensor,
    n_samples: int,
    device: torch.device,
    method: str,
    state,
    seq_idx: int,
) -> dict[str, torch.Tensor]:
    """MC-score one sequence → per-token uncertainty + calibration data.

    Uses ``score_block_batch`` with seed_b = seq_idx (the legacy_d1 seeding rule).
    Returns dict with CPU tensors of shape (seq_len,):
        mi, predictive_entropy, max_prob, correct, p_true, sum_p_sq
    """
    out = score_block_batch(
        model, x.unsqueeze(0), targets.unsqueeze(0), n_samples, device, method, state,
        seed_b=seq_idx, amp=device.type == "cuda",
    )
    return {
        "mi": out["tok_mi"][0],
        "predictive_entropy": out["tok_tu"][0],
        "max_prob": out["tok_maxprob"][0],
        "correct": out["tok_correct"][0].float(),
        "p_true": out["tok_pbar_true"][0],
        "sum_p_sq": out["tok_sum_p_sq"][0],
    }


# ---------------------------------------------------------------------------
# Milestone evaluation
# ---------------------------------------------------------------------------

def evaluate_milestone(
    milestone: str,
    device: torch.device,
    n_samples: int,
    id_seqs: list,
    ood_seqs: list,
) -> dict:
    """Run full D0 metrics suite on one milestone. Returns results dict."""
    label, method_name = LABELS[milestone]
    print(f"\n{'=' * 60}")
    print(f"{label}: {method_name}")
    print(f"{'=' * 60}")
    t0 = time.time()

    model, method, state = load_model(milestone, device)
    actual_n = 1 if method == "deterministic" else n_samples

    # Accumulators — sequence-level for OOD detection
    id_scores = {"mi": [], "pred_ent": [], "max_prob_unc": []}
    ood_scores = {"mi": [], "pred_ent": [], "max_prob_unc": []}

    # Per-token for calibration (ID only)
    cal_max_probs, cal_correct, cal_p_true, cal_sum_p_sq = [], [], [], []

    # Per-token for selective prediction (combined ID+OOD)
    sp_mi, sp_correct = [], []

    def _score_split(seqs, scores_dict, offset, is_id):
        split_name = "ID" if is_id else "OOD"
        for i, (x, y) in enumerate(seqs):
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                print(f"  {split_name} {i + 1}/{len(seqs)} ({elapsed:.0f}s)")
            m = score_sequence_full(
                model, x, y, actual_n, device, method, state, seq_idx=offset + i,
            )
            scores_dict["mi"].append(aggregate_sequence_scores(m["mi"]))
            scores_dict["pred_ent"].append(
                aggregate_sequence_scores(m["predictive_entropy"]),
            )
            scores_dict["max_prob_unc"].append(
                aggregate_sequence_scores(1.0 - m["max_prob"]),
            )
            if is_id:
                cal_max_probs.append(m["max_prob"])
                cal_correct.append(m["correct"])
                cal_p_true.append(m["p_true"])
                cal_sum_p_sq.append(m["sum_p_sq"])
            sp_mi.append(m["mi"])
            sp_correct.append(m["correct"])

    _score_split(id_seqs, id_scores, 0, is_id=True)
    _score_split(ood_seqs, ood_scores, len(id_seqs), is_id=False)

    # --- OOD Detection ---
    n_id, n_ood = len(id_scores["mi"]), len(ood_scores["mi"])
    labels = torch.cat([torch.zeros(n_id), torch.ones(n_ood)])
    results = {}

    # Save raw per-sequence scores for bootstrap
    raw_scores = {}
    for name in ("mi", "pred_ent", "max_prob_unc"):
        scores = torch.tensor(id_scores[name] + ood_scores[name])
        raw_scores[name] = scores
        results[f"auroc_{name}"] = auroc(scores, labels)
        results[f"fpr95_{name}"] = fpr_at_tpr(scores, labels)
        results[f"auprc_{name}"] = auprc(scores, labels)
    raw_scores["labels"] = labels
    results["mi_ratio"] = legacy_mi_ratio(raw_scores)

    # --- Calibration (ID only, per-token) ---
    all_max_prob = torch.cat(cal_max_probs)
    all_correct = torch.cat(cal_correct)
    all_p_true = torch.cat(cal_p_true)
    all_sum_p_sq = torch.cat(cal_sum_p_sq)

    results["ece"] = ece(all_max_prob, all_correct)
    results["nll"] = float(-(all_p_true + 1e-10).log().mean())
    results["brier"] = float((1.0 - 2.0 * all_p_true + all_sum_p_sq).mean())

    # --- Selective Prediction (combined ID+OOD, per-token) ---
    results["aurc"] = aurc(torch.cat(sp_mi), torch.cat(sp_correct))

    elapsed = time.time() - t0
    results["time_s"] = elapsed
    results["_raw_scores"] = raw_scores
    print(f"  Completed in {elapsed:.0f}s")

    # Print summary
    ood_key = "max_prob_unc" if method == "deterministic" else "mi"
    print(f"  AUROC ({ood_key}): {results[f'auroc_{ood_key}']:.3f}  "
          f"FPR@95: {results[f'fpr95_{ood_key}']:.3f}  "
          f"ECE: {results['ece']:.4f}  NLL: {results['nll']:.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate C checkpoints (I2 score files, D1 tables)")
    p.add_argument(
        "--milestone", type=str, default=None,
        help="D1 path: eval single milestone (c0/c1/c2/c3/c4_tfb/c4_lap)",
    )
    p.add_argument(
        "--n-samples", type=int, default=20,
        help="D1 path: MC forward passes per sequence (default: 20)",
    )
    p.add_argument(
        "--n-sequences", type=int, default=500,
        help="Sequences per split for D1 and legacy_d1 (default: 500)",
    )
    p.add_argument(
        "--bootstrap", action="store_true",
        help="i.i.d. 95%% bootstrap CIs (D1 tables and legacy_d1 summaries)",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=10_000,
        help="Bootstrap resamples (default: 10000)",
    )
    p.add_argument(
        "--save-scores", type=str, default=None,
        help="D1 path: save per-sequence scores to .pt file",
    )
    p.add_argument(
        "--from-scores", type=str, default=None,
        help="D1 path: load saved scores and compute tables/CIs (no GPU needed)",
    )
    add_i2_arguments(p)
    p.add_argument(
        "--analyze", action="store_true",
        help="I2 path: document-bootstrap tables and families from the main score files",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # --- I2 analysis: tables and families from the score files ---
    if args.analyze:
        if args.eval_config is None or args.eval_set is not None:
            _fail("--analyze needs --eval-config and no --eval-set")
        cfg, _ = load_eval_config(args.eval_config)
        try:
            check_prereg(cfg)
        except PreregError as exc:
            _fail(str(exc))
        run_analyze(cfg)
        return

    # --- I2 path: score files for one eval set ---
    if args.eval_set is not None:
        registry = SCORE_SET_REGISTRY
        run_i2(
            args,
            script_path=Path(__file__),
            owned=set(registry),
            scored_elsewhere=lambda name: EXTERNAL_SCORE_SETS.get(name),
            load_score_set=lambda name, device: registry[name].load(device),
            checkpoint_paths=lambda name: registry[name].checkpoint_paths(),
            load_legacy=load_legacy_blocks,
        )
        return

    # --- Fast path: load saved scores (no GPU, no checkpoints) ---
    if args.from_scores:
        all_results = load_scores(Path(args.from_scores))
        cis = None
        if args.bootstrap:
            print(f"\nBootstrap CIs ({args.n_bootstrap} resamples)...")
            cis = compute_bootstrap_cis(all_results, args.n_bootstrap)
        print_primary_table(all_results, cis)
        print_secondary_table(all_results)
        return

    # --- Full eval path ---
    device = _resolve_device(args.device)
    print(f"Device: {device}")
    print(f"MC samples: {args.n_samples}, sequences per split: {args.n_sequences}")

    milestones = [args.milestone] if args.milestone else ALL_MILESTONES

    # Validate checkpoints
    valid = []
    for m in milestones:
        missing = [cp for cp in _checkpoint_paths(m) if not cp.exists()]
        if missing:
            print(f"WARNING: {m} — missing: {', '.join(str(cp) for cp in missing)}. Skipping.")
        else:
            valid.append(m)
    milestones = valid

    if not milestones:
        print("No valid milestones to evaluate.")
        return

    # Load data once (C0's ID/OOD split for all milestones)
    id_seqs, ood_seqs = load_eval_data(args.n_sequences)

    # Evaluate each milestone
    all_results = {}
    total_t0 = time.time()
    for m in milestones:
        all_results[m] = evaluate_milestone(m, device, args.n_samples, id_seqs, ood_seqs)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # Save scores if requested
    if args.save_scores:
        save_scores(all_results, Path(args.save_scores))

    # Bootstrap CIs
    cis = None
    if args.bootstrap:
        print(f"\nBootstrap CIs ({args.n_bootstrap} resamples)...")
        cis = compute_bootstrap_cis(all_results, args.n_bootstrap)

    # Print publication-ready tables
    print_primary_table(all_results, cis)
    print_secondary_table(all_results)


if __name__ == "__main__":
    main()
