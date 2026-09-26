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
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c_milestones import OOD_DOMAINS, build_milestone_config
from minigpt.config import build_gpt_config, build_lora_config
from minigpt.data import get_tokenizer, load_pile_data
from minigpt.evalset import PreregError, analysis_sha256, verify_prereg
from minigpt.laplace import (
    apply_sampled_params,
    load_laplace_state,
    sample_laplace_params,
)
from minigpt.layers import enable_dropout
from minigpt.lora import inject_lora
from minigpt.model import MiniGPT
from minigpt.posthoc_refit import (
    EXIT_OK,
    SCORE_SET_CELLS,
    SCORED_STATUSES,
    check_config,
    load_posthoc_base,
)
from minigpt.tfb import load_tfb_state, sample_tfb_params
from minigpt.train import load_checkpoint
from minigpt.uncertainty import (
    aggregate_sequence_scores,
    auprc,
    aurc,
    auroc,
    bootstrap_ci,
    doc_bootstrap_auroc,
    ece,
    fpr_at_tpr,
    holm,
    paired_doc_bootstrap,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256
ALL_MILESTONES = ["c0", "c1", "c2", "c3", "c4_tfb", "c4_lap"]
CKPT_DIR = Path("data/checkpoints")

LABELS = {
    "c0": ("C0", "Deterministic"),
    "c1": ("C1", "Variational FFN"),
    "c2": ("C2", "Laplace FFN"),
    "c3": ("C3", "BLoB LoRA"),
    "c4_tfb": ("C4-TFB", "TFB LoRA"),
    "c4_lap": ("C4-LAP", "Laplace LoRA"),
}

EVAL_SETS = ("legacy_d1", "test", "test_hn", "arxiv_stripped")
FROZEN_EVAL_SETS = ("test", "test_hn", "arxiv_stripped")
EXTERNAL_SCORE_SETS = {"mc_dropout": "scripts/eval_mc_dropout.py"}
GLOBAL_RNG_METHODS = ("deterministic", "variational", "dropout")
MANIFEST_FILE = "manifest.jsonl"
PREREG_FILE = "prereg.json"
PROB_EPS = 1e-10
LOG_EVERY = 100
RUN_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# --analyze (section 5.6): cells read the `main` score files
ANALYSIS_RUN_TAG = "main"
ANALYSIS_EVAL_SETS = ("test", "test_hn", "arxiv_stripped")
ID_ROLE_EVAL_SET = {"id_base": "test", "id_adapter": "test_hn"}
STRIPPED_OOD = "arxiv_stripped"


def repo_path(path: str | Path) -> Path:
    """A config path: absolute as given, relative to the repo root otherwise."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# ---------------------------------------------------------------------------
# Data utilities (legacy_d1 = the old D1 blocks)
# ---------------------------------------------------------------------------

def _extract_sequences(data: torch.Tensor, block_size: int, n: int):
    """Non-overlapping (x, y) sequences from flat token tensor."""
    max_seqs = (len(data) - 1) // block_size
    n = min(n, max_seqs)
    seqs = []
    for i in range(n):
        s = i * block_size
        seqs.append((data[s : s + block_size], data[s + 1 : s + block_size + 1]))
    return seqs


def _legacy_split() -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]], str]:
    """C0's D1 split: (test_id, [(ood_domain, tensor), ...], id_domain)."""
    cfg = build_milestone_config("c0")
    data = load_pile_data(cfg, get_tokenizer())
    ood_parts = [(d, data[f"test_ood_{d}"]) for d in OOD_DOMAINS if f"test_ood_{d}" in data]
    if not ood_parts:
        ood_parts = [("ood", data["test_ood"])]
    return data["test_id"], ood_parts, cfg["data"]["pile_id_domains"][-1]


def load_eval_data(n_sequences: int, block_size: int = BLOCK_SIZE):
    """Load Pile test data using C0's ID/OOD split. Returns (id_seqs, ood_seqs)."""
    test_id, ood_parts, _ = _legacy_split()
    test_ood = torch.cat([t for _, t in ood_parts])
    id_seqs = _extract_sequences(test_id, block_size, n_sequences)
    ood_seqs = _extract_sequences(test_ood, block_size, n_sequences)
    print(f"Loaded {len(id_seqs)} ID seqs, {len(ood_seqs)} OOD seqs (block_size={block_size})")
    return id_seqs, ood_seqs


@dataclass
class BlockSet:
    """Blocks of one eval set, in file order. ``tokens[:, :-1]`` is x, ``tokens[:, 1:]`` y."""

    tokens: torch.Tensor
    block_index: torch.Tensor
    doc_id: list[str]
    domain: list[str]
    offset: torch.Tensor
    weight: torch.Tensor
    manifest_sha256: str | None

    def __len__(self) -> int:
        return len(self.doc_id)

    def select(self, rows: list[int]) -> BlockSet:
        idx = torch.tensor(rows, dtype=torch.long)
        return BlockSet(
            tokens=self.tokens[idx],
            block_index=self.block_index[idx],
            doc_id=[self.doc_id[r] for r in rows],
            domain=[self.domain[r] for r in rows],
            offset=self.offset[idx],
            weight=self.weight[idx],
            manifest_sha256=self.manifest_sha256,
        )


def build_legacy_blocks(
    test_id: torch.Tensor,
    ood_parts: list[tuple[str, torch.Tensor]],
    id_domain: str,
    n_sequences: int,
    block_size: int,
) -> BlockSet:
    """legacy_d1: first n windows of test_id (b = 0..n-1), then of cat(OOD parts).

    b equals the old ``seq_idx``. ``doc_id`` is ``legacy/<domain>/<b>``, weight 1, and
    ``offset`` is the window start inside test_id or inside cat(OOD parts).
    """
    test_ood = torch.cat([t for _, t in ood_parts])
    ends = torch.cumsum(torch.tensor([len(t) for _, t in ood_parts]), 0).tolist()
    rows, doc_ids, domains, offsets = [], [], [], []
    for source, is_id in ((test_id, True), (test_ood, False)):
        n = min(n_sequences, (len(source) - 1) // block_size)
        for i in range(n):
            start = i * block_size
            if is_id:
                domain = id_domain
            else:
                domain = next(d for (d, _), end in zip(ood_parts, ends) if start < end)
            rows.append(source[start : start + block_size + 1])
            doc_ids.append(f"legacy/{domain}/{len(doc_ids)}")
            domains.append(domain)
            offsets.append(start)
    n_blocks = len(rows)
    print(f"legacy_d1: {n_blocks} blocks (block_size={block_size})")
    return BlockSet(
        tokens=torch.stack(rows),
        block_index=torch.arange(n_blocks, dtype=torch.int64),
        doc_id=doc_ids,
        domain=domains,
        offset=torch.tensor(offsets, dtype=torch.int64),
        weight=torch.ones(n_blocks, dtype=torch.float32),
        manifest_sha256=None,
    )


def load_legacy_blocks(n_sequences: int, block_size: int) -> BlockSet:
    """The legacy_d1 blocks from C0's split (same windows as ``load_eval_data``)."""
    test_id, ood_parts, id_domain = _legacy_split()
    return build_legacy_blocks(test_id, ood_parts, id_domain, n_sequences, block_size)


def load_i2_blocks(cfg: dict, eval_set: str) -> BlockSet:
    """Read ``blocks_{eval_set}.pt`` and check it against ``manifest.jsonl``.

    Raises ValueError if a block's document is not a test document of the manifest, if a
    weight is not 1 / (blocks of its document), or if the token width is not T + 1.
    """
    out_dir = repo_path(cfg["eval_set"]["out_dir"])
    block_size = cfg["eval_set"]["block_size"]
    manifest_raw = (out_dir / MANIFEST_FILE).read_bytes()
    test_docs = set()
    for line in manifest_raw.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["split"] == "test":
                test_docs.add(row["doc_id"])
    blocks_path = out_dir / f"blocks_{eval_set}.pt"
    data = torch.load(blocks_path, weights_only=True)
    tokens = data["tokens"]
    doc_id = [str(d) for d in data["doc_id"]]
    domain = [str(d) for d in data["domain"]]
    offset = torch.as_tensor(data["offset"], dtype=torch.int64)
    weight = torch.as_tensor(data["weight"], dtype=torch.float32)
    if tokens.dim() != 2 or tokens.shape[1] != block_size + 1 or tokens.is_floating_point():
        raise ValueError(
            f"{blocks_path}: tokens must be integer [B, {block_size + 1}], "
            f"got {tuple(tokens.shape)} {tokens.dtype}"
        )
    n_blocks = tokens.shape[0]
    if not (len(doc_id) == len(domain) == offset.numel() == weight.numel() == n_blocks):
        raise ValueError(f"{blocks_path}: doc_id, domain, offset and weight need {n_blocks} rows")
    not_test = sorted(set(doc_id) - test_docs)
    if not_test:
        raise ValueError(f"{blocks_path}: not test documents of the manifest: {not_test[:3]}")
    counts = Counter(doc_id)
    expected = torch.tensor([1.0 / counts[d] for d in doc_id], dtype=torch.float32)
    if not torch.allclose(weight, expected, rtol=0.0, atol=1e-6):
        raise ValueError(f"{blocks_path}: weight must be 1 / (blocks of the document)")
    print(f"{eval_set}: {n_blocks} blocks from {len(counts)} documents ({blocks_path})")
    return BlockSet(
        tokens=tokens,
        block_index=torch.arange(n_blocks, dtype=torch.int64),
        doc_id=doc_id,
        domain=domain,
        offset=offset,
        weight=weight,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )


def parse_block_ids(spec: str, n_blocks: int) -> list[int]:
    """Parse ``--block-ids`` such as ``0-99,500-599`` into ascending block indices."""
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty item in --block-ids {spec!r}")
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ValueError(f"range {part!r} in --block-ids runs backwards")
            ids.extend(range(lo, hi + 1))
        else:
            ids.append(int(part))
    if any(b < 0 or b >= n_blocks for b in ids):
        raise ValueError(f"--block-ids {spec!r} is outside [0, {n_blocks})")
    if ids != sorted(set(ids)):
        raise ValueError(f"--block-ids {spec!r} must be ascending without repeats")
    return ids


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _checkpoint_paths(milestone: str) -> list[Path]:
    """All checkpoint files needed for a milestone."""
    if milestone == "c0":
        return [CKPT_DIR / "c0/ckpt_best.pt"]
    if milestone == "c1":
        return [CKPT_DIR / "c1/ckpt_best.pt"]
    if milestone == "c2":
        return [CKPT_DIR / "c0/ckpt_best.pt", CKPT_DIR / "c2/laplace_state.pt"]
    if milestone == "c3":
        return [CKPT_DIR / "c3/ckpt_best.pt"]
    if milestone == "c4_tfb":
        return [CKPT_DIR / "c3/ckpt_best.pt", CKPT_DIR / "c4_tfb/tfb_state.pt"]
    if milestone == "c4_lap":
        return [CKPT_DIR / "c3/ckpt_best.pt", CKPT_DIR / "c4_lap/laplace_state.pt"]
    return []


def _blob_to_deterministic_lora(model: MiniGPT, cfg: dict) -> MiniGPT:
    """Convert C3 BLoB checkpoint → DeterministicLoRA model (for C4)."""
    model = inject_lora(model, build_lora_config(cfg), bayesian=False)
    blob_sd = torch.load(
        CKPT_DIR / "c3/ckpt_best.pt", weights_only=False,
    )["model_state_dict"]
    target_sd = model.state_dict()
    mapped = {}
    for key in target_sd:
        if key in blob_sd:
            mapped[key] = blob_sd[key]
        elif "lora_A" in key:
            blob_key = key.replace(".lora_A", ".lora_A_mu")
            mapped[key] = blob_sd.get(blob_key, target_sd[key])
        else:
            mapped[key] = target_sd[key]
    model.load_state_dict(mapped)
    return model


def load_model(milestone: str, device: torch.device):
    """Build and load model. Returns (model, method, posthoc_state)."""
    tokenizer = get_tokenizer()
    posthoc_state = None

    if milestone == "c0":
        cfg = build_milestone_config("c0")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        load_checkpoint(CKPT_DIR / "c0/ckpt_best.pt", model)
        method = "deterministic"

    elif milestone == "c1":
        cfg = build_milestone_config("c1")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        load_checkpoint(CKPT_DIR / "c1/ckpt_best.pt", model)
        method = "variational"

    elif milestone == "c2":
        cfg = build_milestone_config("c0")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        load_checkpoint(CKPT_DIR / "c0/ckpt_best.pt", model)
        posthoc_state = load_laplace_state(
            CKPT_DIR / "c2/laplace_state.pt", map_location=device,
        )
        method = "laplace"

    elif milestone == "c3":
        cfg = build_milestone_config("c3_phase2")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        model = inject_lora(model, build_lora_config(cfg), bayesian=True)
        load_checkpoint(CKPT_DIR / "c3/ckpt_best.pt", model)
        method = "variational"

    elif milestone in ("c4_tfb", "c4_lap"):
        cfg = build_milestone_config(milestone)
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        model = _blob_to_deterministic_lora(model, cfg)
        if milestone == "c4_tfb":
            posthoc_state = load_tfb_state(
                CKPT_DIR / "c4_tfb/tfb_state.pt", map_location=device,
            )
            method = "tfb"
        else:
            posthoc_state = load_laplace_state(
                CKPT_DIR / "c4_lap/laplace_state.pt", map_location=device,
            )
            method = "laplace"

    else:
        raise ValueError(f"Unknown milestone: {milestone}")

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} params, method={method}")
    return model, method, posthoc_state


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

    p_bar = p_sum / n_samples
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
    code_sha = code_sha256([script_path, Path(__file__)])
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
# --analyze: document-bootstrap tables and families (sections 3 and 5.6)
# ---------------------------------------------------------------------------

Triple = tuple[str, str, str]


def _roles(cfg: dict) -> dict[str, str]:
    return {d["key"]: d["role"] for d in cfg["eval_set"]["domains"]}


def _domain_for_role(cfg: dict, role: str) -> str:
    keys = [k for k, r in _roles(cfg).items() if r == role]
    if len(keys) != 1:
        raise ValueError(f"eval_set.domains needs exactly one {role} domain, got {keys}")
    return keys[0]


def planned_table_cells(cfg: dict) -> dict[str, list[Triple]]:
    """Cells (score set, ID domain, OOD domain) of the three result tables."""
    sets = cfg["scoring"]["score_sets"]
    ood = cfg["analysis"]["ood_domains"]
    id_base = _domain_for_role(cfg, "id_base")
    id_adapter = _domain_for_role(cfg, "id_adapter")
    return {
        "test": [(s, id_base, o) for s in sets["test"] for o in ood],
        "test_hn": [(s, id_adapter, o) for s in sets["test_hn"] for o in ood],
        "arxiv_stripped": (
            [(s, id_base, STRIPPED_OOD) for s in sets["arxiv_stripped"]]
            + [(s, id_adapter, STRIPPED_OOD) for s in sets["arxiv_stripped"]
               if s in sets["test_hn"]]
        ),
    }


def _cell_sources(cfg: dict, triple: Triple) -> list[tuple[int, str, set[str]]]:
    """[(label, eval set of the score file, block domains)] for one cell (section 5.6)."""
    _, id_domain, ood_domain = triple
    roles = _roles(cfg)
    if id_domain not in roles or roles[id_domain] not in ID_ROLE_EVAL_SET:
        raise ValueError(f"{id_domain} is not an ID domain")
    sources = [(0, ID_ROLE_EVAL_SET[roles[id_domain]], {id_domain})]
    if ood_domain == STRIPPED_OOD:
        stripped = [d["key"] for d in cfg["eval_set"]["domains"] if d["stripped_copy"]]
        sources.append((1, STRIPPED_OOD, {*stripped, STRIPPED_OOD}))
    elif ood_domain in roles and roles[ood_domain] == "ood":
        sources.append((1, "test", {ood_domain}))
    else:
        raise ValueError(f"{ood_domain} is not an OOD domain")
    return sources


def _domain_index(cfg: dict) -> dict[str, int]:
    """Bootstrap seed index: position in eval_set.domains; arxiv_stripped comes last."""
    index = {d["key"]: i for i, d in enumerate(cfg["eval_set"]["domains"])}
    index[STRIPPED_OOD] = len(index)
    return index


class _CellData:
    """Loads the block scores of the `main` score files and assembles cells."""

    def __init__(self, cfg: dict, score_names: list[str]) -> None:
        self.cfg = cfg
        self.score_names = score_names
        self._files: dict[Path, dict | None] = {}

    def _fields(self, score_set: str, eval_set: str) -> dict | None:
        path = score_file_path(self.cfg, score_set, eval_set, ANALYSIS_RUN_TAG)
        if path not in self._files:
            if not path.exists():
                self._files[path] = None
            else:
                sf = torch.load(path, weights_only=True, mmap=True)
                self._files[path] = {
                    "doc_id": np.asarray(sf["doc_id"]),
                    "domain": np.asarray(sf["domain"]),
                    **{n: sf[n].double().numpy() for n in self.score_names},
                }
        return self._files[path]

    def cell(self, triple: Triple) -> dict | None:
        """Labels, documents, weights and scores of a cell; None if a score file is missing."""
        labels, docs, parts = [], [], {n: [] for n in self.score_names}
        for label, eval_set, domains in _cell_sources(self.cfg, triple):
            fields = self._fields(triple[0], eval_set)
            if fields is None:
                return None
            rows = np.flatnonzero(np.isin(fields["domain"], sorted(domains)))
            if rows.size == 0:
                raise ValueError(f"{triple[0]}__{eval_set}: no blocks of {sorted(domains)}")
            labels.append(np.full(rows.size, label, dtype=np.int64))
            docs.append(fields["doc_id"][rows])
            for n in self.score_names:
                parts[n].append(fields[n][rows])
        y = np.concatenate(labels)
        doc_ids = np.concatenate(docs)
        weights = np.empty(y.size, dtype=np.float64)
        for c in (0, 1):  # w_b = 1 / k_d, k_d = blocks of document d in the cell
            counts = Counter(doc_ids[y == c].tolist())
            weights[y == c] = [1.0 / counts[d] for d in doc_ids[y == c].tolist()]
        return {"labels": y, "doc_ids": doc_ids, "weights": weights,
                "scores": {n: np.concatenate(parts[n]) for n in self.score_names},
                "n_id_blocks": int((y == 0).sum()), "n_ood_blocks": int((y == 1).sum())}


def _decision(delta: float, p_holm: float, margin: float, alpha: float) -> str:
    """Section 3 decision rule for one family cell."""
    if delta >= margin and p_holm < alpha:
        return "primary_beats_contrast"
    if delta <= -margin and p_holm < alpha:
        return "contrast_beats_primary"
    return "no_difference"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def run_analyze(cfg: dict) -> dict[str, Path]:
    """Write table_{test,test_hn,arxiv_stripped}.json, families.json and descriptive.json.

    Every cell uses the `main` score files, the document weights 1/k_d and the paired
    document bootstrap of minigpt.uncertainty with the generator seed
    [analysis.bootstrap.seed, i_id, i_ood]. Cells whose score files do not exist yet are
    left out; a family with a missing cell is "incomplete" and gets no Holm adjustment.
    """
    an = cfg["analysis"]
    boot_cfg = an["bootstrap"]
    if (boot_cfg["unit"] != "document" or boot_cfg["stratify_by_class"] is not True
            or boot_cfg["percentile_method"] != "linear"):
        raise ValueError("only the document-unit, class-stratified, linear-percentile "
                         "bootstrap of section 5.6 is implemented")
    n_resamples, seed, level = boot_cfg["resamples"], boot_cfg["seed"], boot_cfg["level"]
    primary, contrast = an["primary_score"], an["contrast_score"]
    score_names = [primary, contrast, *an["secondary_scores"]]
    index = _domain_index(cfg)
    data = _CellData(cfg, score_names)
    cells: dict[Triple, dict | None] = {}

    def cell_result(triple: Triple) -> dict | None:
        if triple not in cells:
            c = data.cell(triple)
            if c is None:
                cells[triple] = None
            else:
                rng_seed = [seed, index[triple[1]], index[triple[2]]]
                boot = doc_bootstrap_auroc(
                    c["scores"], c["labels"], c["doc_ids"], c["weights"], n_resamples,
                    rng_seed, level, target_tpr=an["fpr_target_tpr"],
                )
                cells[triple] = {"data": c, "boot": boot, "rng_seed": rng_seed}
        return cells[triple]

    stamp = {"analysis_sha256": analysis_sha256(cfg), "run_tag": ANALYSIS_RUN_TAG,
             "n_resamples": n_resamples, "created_at": datetime.now(timezone.utc).isoformat()}
    out = scores_dir(cfg)
    written: dict[str, Path] = {}
    tables: dict[str, list[Triple]] = {}
    for eval_set, triples in planned_table_cells(cfg).items():
        rows = []
        tables[eval_set] = []
        for triple in triples:
            res = cell_result(triple)
            if res is None:
                print(f"  not scored yet: {'|'.join(triple)}")
                continue
            tables[eval_set].append(triple)
            boot = res["boot"]
            rows.append({
                "score_set": triple[0], "id_domain": triple[1], "ood_domain": triple[2],
                "n_id_docs": boot[primary]["n_id_docs"],
                "n_ood_docs": boot[primary]["n_ood_docs"],
                "n_id_blocks": res["data"]["n_id_blocks"],
                "n_ood_blocks": res["data"]["n_ood_blocks"],
                "rng_seed": res["rng_seed"],
                "scores": {
                    n: {"auroc_w": boot[n]["auroc"], "auroc_w_ci": list(boot[n]["auroc_ci"]),
                        "fpr95_w": boot[n]["fpr95"], "fpr95_w_ci": list(boot[n]["fpr95_ci"])}
                    for n in score_names
                },
            })
        path = out / f"table_{eval_set}.json"
        _write_json(path, {"eval_set": eval_set, **stamp, "cells": rows})
        written[f"table_{eval_set}"] = path

    families = {}
    for fname, fam_rows in an["families"].items():
        triples = [(s, id_dom, o) for s, id_dom in fam_rows for o in an["ood_domains"]]
        fcells = []
        for triple in triples:
            res = cell_result(triple)
            if res is None:
                continue
            c = res["data"]
            pair = paired_doc_bootstrap(
                {n: c["scores"][n] for n in (primary, contrast)}, c["labels"], c["doc_ids"],
                c["weights"], [(primary, contrast)], n_resamples, res["rng_seed"], level,
            )[(primary, contrast)]
            fcells.append({
                "score_set": triple[0], "id_domain": triple[1], "ood_domain": triple[2],
                "auroc_w_primary": res["boot"][primary]["auroc"],
                "auroc_w_contrast": res["boot"][contrast]["auroc"],
                "delta": pair["delta"], "delta_ci": list(pair["ci"]), "p": pair["p"],
            })
        status = ("not_scored" if not fcells
                  else "incomplete" if len(fcells) < len(triples) else "scored")
        if status == "scored":
            for fc, p_adj in zip(fcells, holm([fc["p"] for fc in fcells])):
                fc["p_holm"] = float(p_adj)
                fc["decision"] = _decision(fc["delta"], fc["p_holm"], an["margin_auroc"],
                                           an["alpha"])
        else:
            for fc in fcells:
                fc["p_holm"] = None
        done = {(fc["score_set"], fc["id_domain"], fc["ood_domain"]) for fc in fcells}
        families[fname] = {"status": status, "m": len(triples), "cells": fcells,
                           "missing_cells": ["|".join(t) for t in triples if t not in done]}
    written["families"] = out / "families.json"
    _write_json(written["families"], {**stamp, "families": families})

    rep = an["descriptive"]["replication"]
    above = [
        {"cell": "|".join(t), "primary_auroc_w_ci_lo": cells[t]["boot"][primary]["auroc_ci"][0],
         "above_chance": cells[t]["boot"][primary]["auroc_ci"][0] > 0.5}
        for triples in tables.values() for t in triples
    ]
    replication = {}
    for s, old_point in rep["old_point"].items():
        res = cell_result((s, rep["id"], rep["ood"]))
        if res is None:
            replication[s] = {"status": "not_scored"}
            continue
        new_m, new_ci = res["boot"][contrast]["auroc"], list(res["boot"][contrast]["auroc_ci"])
        old_ci = rep["old_cluster_ci"][s]
        replication[s] = {
            "new_auroc_w_contrast": new_m, "new_ci": new_ci, "old_point": old_point,
            "old_cluster_ci": old_ci,
            "replicates": bool(old_ci[0] <= new_m <= old_ci[1]
                               and new_ci[0] <= old_point <= new_ci[1]),
        }
    written["descriptive"] = out / "descriptive.json"
    _write_json(written["descriptive"],
                {**stamp, "above_chance": above, "replication": replication})

    for fname, fam in families.items():
        print(f"{fname}: {fam['status']} ({len(fam['cells'])}/{fam['m']} cells)")
        for fc in fam["cells"]:
            p_holm = "n/a" if fc["p_holm"] is None else f"{fc['p_holm']:.4g}"
            print(f"  {fc['score_set']:<14} {fc['id_domain']:<14} {fc['ood_domain']:<16} "
                  f"delta {fc['delta']:+.4f} [{fc['delta_ci'][0]:+.4f}, "
                  f"{fc['delta_ci'][1]:+.4f}]  p {fc['p']:.4g}  p_holm {p_holm}")
    for path in written.values():
        print(f"Wrote {path}")
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


def legacy_mi_ratio(raw: dict) -> float | None:
    """Descriptive mean MI(OOD) / mean MI(ID) from a D1 score dict (None if MI(ID) is 0)."""
    mi = torch.as_tensor(raw["mi"]).double()
    labels = torch.as_tensor(raw["labels"])
    id_mean = float(mi[labels == 0].mean())
    if id_mean == 0.0:
        return None
    return float(mi[labels == 1].mean()) / id_mean


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
# Score persistence
# ---------------------------------------------------------------------------

def save_scores(all_results: dict, path: Path):
    """Save per-sequence scores for all milestones to a .pt file."""
    payload = {}
    for m, r in all_results.items():
        if "_raw_scores" in r:
            payload[m] = r["_raw_scores"]
    torch.save(payload, path)
    print(f"Saved per-sequence scores to {path}")


def load_scores(path: Path) -> dict:
    """Load per-sequence scores and recompute all metrics."""
    payload = torch.load(path, weights_only=True)
    all_results = {}
    for m, raw in payload.items():
        labels = raw["labels"]
        r = {"mi_ratio": legacy_mi_ratio(raw), "_raw_scores": raw}
        for name in ("mi", "pred_ent", "max_prob_unc"):
            scores = raw[name]
            r[f"auroc_{name}"] = auroc(scores, labels)
            r[f"fpr95_{name}"] = fpr_at_tpr(scores, labels)
            r[f"auprc_{name}"] = auprc(scores, labels)
        all_results[m] = r
    print(f"Loaded scores for {list(all_results.keys())} from {path}")
    return all_results


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def compute_bootstrap_cis(
    all_results: dict,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Compute bootstrap 95% CIs for AUROC, FPR@95, AUPRC."""
    cis = {}
    for m, r in all_results.items():
        raw = r.get("_raw_scores")
        if raw is None:
            continue
        labels = raw["labels"]
        ood_key = "max_prob_unc" if m == "c0" else "mi"
        scores = raw[ood_key]
        print(f"  {LABELS[m][0]}: bootstrapping ({n_bootstrap} resamples)...")
        _, lo, hi = bootstrap_ci(scores, labels, auroc,
                                 n_bootstrap=n_bootstrap, seed=seed)
        _, fpr_lo, fpr_hi = bootstrap_ci(scores, labels, fpr_at_tpr,
                                         n_bootstrap=n_bootstrap, seed=seed)
        _, auprc_lo, auprc_hi = bootstrap_ci(scores, labels, auprc,
                                             n_bootstrap=n_bootstrap, seed=seed)
        cis[m] = {
            "auroc_ci": (lo, hi),
            "fpr95_ci": (fpr_lo, fpr_hi),
            "auprc_ci": (auprc_lo, auprc_hi),
        }
    return cis


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def _fmt_ci(value: float, ci: tuple[float, float] | None) -> str:
    """Format value with optional CI: '0.916 [0.89, 0.94]'."""
    if ci is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def print_primary_table(results: dict, cis: dict | None = None):
    """Primary table: MI ratio (descriptive, computed from the scores) + all metrics."""
    has_ci = cis is not None
    print("\n## Primary Results Table\n")
    print("MI Ratio = mean MI(OOD) / mean MI(ID) over the scored sequences (descriptive).\n")
    if has_ci:
        print("| Milestone | Method          | MI Ratio | AUROC [95% CI]              "
              "| FPR@95 [95% CI]             | AUPRC [95% CI]              "
              "| ECE    | Brier | NLL  | AURC  |")
        print("|-----------|-----------------|----------|-----------------------------"
              "|-----------------------------|-----------------------------"
              "|--------|-------|------|-------|")
    else:
        print("| Milestone | Method          | MI Ratio | AUROC | FPR@95 | AUPRC "
              "| ECE    | Brier | NLL  | AURC  |")
        print("|-----------|-----------------|----------|-------|--------|-------"
              "|--------|-------|------|-------|")

    for m in ALL_MILESTONES:
        if m not in results:
            continue
        r = results[m]
        label, method_name = LABELS[m]
        mi_ratio = f"{r['mi_ratio']:.4f}x" if r["mi_ratio"] else "--"
        ood_key = "max_prob_unc" if m == "c0" else "mi"

        ci = cis.get(m) if cis else None
        auroc_str = _fmt_ci(r[f"auroc_{ood_key}"], ci["auroc_ci"] if ci else None)
        fpr_str = _fmt_ci(r[f"fpr95_{ood_key}"], ci["fpr95_ci"] if ci else None)
        auprc_str = _fmt_ci(r[f"auprc_{ood_key}"], ci["auprc_ci"] if ci else None)

        ece_str = f"{r['ece']:.4f}" if "ece" in r else "--"
        brier_str = f"{r['brier']:.3f}" if "brier" in r else "--"
        nll_str = f"{r['nll']:.2f}" if "nll" in r else "--"
        aurc_str = f"{r['aurc']:.4f}" if "aurc" in r else "--"

        if has_ci:
            print(
                f"| {label:<9} | {method_name:<15} | {mi_ratio:>8} "
                f"| {auroc_str:<27} | {fpr_str:<27} | {auprc_str:<27} "
                f"| {ece_str:>6} | {brier_str:>5} | {nll_str:>4} "
                f"| {aurc_str:>5} |"
            )
        else:
            print(
                f"| {label:<9} | {method_name:<15} | {mi_ratio:>8} "
                f"| {r[f'auroc_{ood_key}']:.3f} | {r[f'fpr95_{ood_key}']:.3f}  "
                f"| {r[f'auprc_{ood_key}']:.3f} "
                f"| {ece_str:>6} | {brier_str:>5} | {nll_str:>4} "
                f"| {aurc_str:>5} |"
            )


def print_secondary_table(results: dict):
    """Uncertainty score comparison (all AUROC)."""
    print("\n## Uncertainty Score Comparison (AUROC)\n")
    print("| Milestone | MI AUROC | Pred. Entropy AUROC | Max-Prob AUROC |")
    print("|-----------|----------|---------------------|----------------|")

    for m in ALL_MILESTONES:
        if m not in results:
            continue
        r = results[m]
        label = LABELS[m][0]
        mi_auroc = f"{r['auroc_mi']:.3f}" if m != "c0" and "auroc_mi" in r else "--"
        pred_ent = f"{r['auroc_pred_ent']:.3f}" if "auroc_pred_ent" in r else "--"
        max_prob = f"{r['auroc_max_prob_unc']:.3f}" if "auroc_max_prob_unc" in r else "--"
        print(
            f"| {label:<9} | {mi_auroc:>8} "
            f"| {pred_ent:>19} "
            f"| {max_prob:>14} |"
        )


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
