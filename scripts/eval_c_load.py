"""Constants, eval blocks and C model loading for scripts/eval_c_checkpoints.py."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.c_milestones import OOD_DOMAINS, build_milestone_config
from minigpt.config import build_gpt_config, build_lora_config
from minigpt.data import get_tokenizer, load_pile_data
from minigpt.laplace import load_laplace_state
from minigpt.lora import inject_lora
from minigpt.model import MiniGPT
from minigpt.tfb import load_tfb_state
from minigpt.train import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]

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
