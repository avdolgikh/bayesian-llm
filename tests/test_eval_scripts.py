"""S1-T5 and the S1-T1 legacy_d1 path: run both eval scripts end to end on a fixture.

Spec: specs/i2-eval-rebuild.md (sections 4, 5.5 and 5.7).

Fixture (spec S1-T5 "Given"): a 2-layer MiniGPT (n_embd 32, block size 32, dropout 0.1);
checkpoints and post-hoc states for c0, c1, c2, c3, c4_tfb and c4_lap in a temp dir; an
eval set with 3 domains (1 ID, 2 OOD), 20 blocks from 10 documents each; N = 3.
Both scripts are imported with importlib. `CKPT_DIR`, `build_milestone_config`,
`get_tokenizer` and `load_pile_data` are monkeypatched; the model loaders are unchanged.
The eval config comes from a fixture YAML written here, never from configs/i2_eval.yaml.

Then rows covered: S1-T5a, T5c, T5d, T5e, T5f, T5g, T5h, and the fixture form of the
S1-T1 legacy_d1 path (block layout, old seeding rule, T1d rerun equality, T1f batch check,
seed-base guard).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

from minigpt.config import build_gpt_config, build_lora_config
from minigpt.laplace import apply_sampled_params, select_params
from minigpt.lora import DeterministicLoRALinear, inject_lora
from minigpt.model import MiniGPT
from minigpt.tfb import load_tfb_state, sample_tfb_params

_MODULE_T0 = time.perf_counter()

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Fixture constants (spec section 2: test fixtures keep their constants here)
# ---------------------------------------------------------------------------

VOCAB = 96
T = 32
N_SAMPLES = 3
FIXTURE_SEED = 1234
DOCS_PER_DOMAIN = 10
BLOCKS_PER_DOC = [1, 2, 3, 2, 2, 3, 1, 2, 2, 2]  # 20 blocks from 10 documents
TEST_DOMAINS = ["stackexchange", "arxiv", "freelaw"]  # 1 ID, 2 OOD
HN_BLOCKS_PER_DOC = [2, 1, 2, 1, 2]
LEGACY_N = 6  # legacy_d1 blocks per split: b 0-5 ID, 6-11 OOD
LEGACY_ARXIV_TOKENS = 3 * T + 10  # OOD blocks 6-9 start in arxiv, 10-11 in freelaw
TFB_SIGMA_Q = 0.05
LAPLACE_SAMPLE_SCALE = 0.05
LORA_B_SCALE = 0.5
TIME_BUDGET_S = 60.0
PREREG_FROZEN_AT = "2026-09-26T00:00:00+00:00"

MODEL_OVERRIDES = {
    "model.n_layer": 2,
    "model.n_head": 2,
    "model.n_embd": 32,
    "model.block_size": T,
    "model.dropout": 0.1,
    "train.block_size": T,
}
LORA_OVERRIDES = {"lora.rank": 4, "lora.alpha": 8.0}
FIXTURE_MILESTONES = ("c0", "c1", "c2", "c3_phase2", "c4_tfb", "c4_lap")

SCORE_SETS_ALL = ["c0", "mc_dropout", "c1", "c3", "c4_tfb"]
ECC_SETS = ["c0", "c1", "c3", "c4_tfb"]
LABELS = {
    "c0": {"display": "C0 deterministic", "sampler": "none", "adapter_source": "none"},
    "mc_dropout": {"display": "MC dropout (C0)", "sampler": "dropout", "adapter_source": "none"},
    "c1": {"display": "C1 variational FFN", "sampler": "variational", "adapter_source": "none"},
    "c3": {"display": "C3 BLoB LoRA", "sampler": "variational", "adapter_source": "blob"},
    "c4_tfb": {"display": "BLoB-mean + TFB (pre-fix)", "sampler": "pre-fix",
               "adapter_source": "blob_mean"},
}
SEED_OFFSETS = {"legacy_d1": 0, "test": 100000, "test_hn": 200000, "arxiv_stripped": 300000}
SPREAD_SEED_BASES = [1000000, 2000000]

TENSOR_KEYS_BNT = ("logp_real",)
TENSOR_KEYS_BT_F32 = ("tok_mi", "tok_tu", "tok_au", "tok_maxprob", "tok_sum_p_sq")
BLOCK_KEYS_F32 = ("blk_g", "blk_mi", "blk_tu", "blk_au", "blk_nll", "blk_maxprob_unc")
META_KEYS = (
    "git_sha", "git_dirty", "code_sha256", "checkpoint_sha256", "score_set", "display_label",
    "sampler", "adapter_source", "eval_set", "run_tag", "block_ids", "n_samples", "seed_base",
    "seed_offset", "batch_size", "amp_fp16", "device", "torch_version", "cuda_version",
    "determinism_warnings", "manifest_sha256", "yaml_sha256", "analysis_sha256", "created_at",
)
RATIO_RE = re.compile(
    r"^MI ratio \(descriptive, block means from (\S+)\) (\S+) (\S+) (\S+)/(\S+): "
    r"([0-9.eE+-]+)$"
)


def _analysis_block() -> dict:
    return {
        "primary_score": "blk_g",
        "contrast_score": "blk_mi",
        "secondary_scores": ["blk_tu", "blk_au", "blk_nll", "blk_maxprob_unc"],
        "aggregation": "mean_all_positions",
        "ood_domains": ["arxiv", "freelaw"],
        "fpr_target_tpr": 0.95,
        "bootstrap": {"resamples": 200, "seed": 0, "level": 0.95, "unit": "document",
                      "stratify_by_class": True, "percentile_method": "linear"},
        "margin_auroc": 0.02,
        "alpha": 0.05,
        "families": {
            "F1_s1_primary": [["mc_dropout", "stackexchange"], ["c1", "stackexchange"],
                              ["c3", "hackernews"]],
            "F2_s1_lora_on_se": [["c3", "stackexchange"]],
        },
        "descriptive": {
            "rows": [["c4_tfb", "hackernews"], ["c4_tfb", "stackexchange"]],
            "stripped_rows": [["mc_dropout", "stackexchange"], ["c1", "stackexchange"],
                              ["c3", "hackernews"], ["c4_tfb", "hackernews"]],
            "replication": {
                "id": "stackexchange",
                "ood": "arxiv",
                "old_point": {"mc_dropout": 0.8976, "c1": 0.8739, "c3": 0.9085,
                              "c4_tfb": 0.9173},
                "old_cluster_ci": {"mc_dropout": [0.858, 0.931], "c1": [0.835, 0.910],
                                   "c3": [0.881, 0.934], "c4_tfb": [0.888, 0.942]},
            },
        },
    }


def _fixture_yaml(root: Path) -> dict:
    return {
        "eval_set": {
            "name": "i2_fixture",
            "source": {"dataset": "fixture", "split": "train", "shuffle_seed": 1337,
                       "shuffle_buffer": 10},
            "train_configs": {"base": "configs/c0.yaml", "adapter": "configs/c3_phase2.yaml"},
            "domains": [
                {"key": "stackexchange", "role": "id_base", "cache_tokens": 1000,
                 "max_stream_tokens": 5000, "stripped_copy": False},
                {"key": "hackernews", "role": "id_adapter", "cache_tokens": 1000,
                 "max_stream_tokens": 5000, "stripped_copy": False},
                {"key": "arxiv", "role": "ood", "cache_tokens": 1000,
                 "max_stream_tokens": 5000, "stripped_copy": True},
                {"key": "freelaw", "role": "ood", "cache_tokens": 1000,
                 "max_stream_tokens": 5000, "stripped_copy": False},
            ],
            "seen_tensors": [],
            "splits": ["test", "val"],
            "block_size": T,
            "min_doc_tokens": T + 1,
            "max_blocks_per_doc": 3,
            "n_test_docs": DOCS_PER_DOMAIN,
            "candidate_pool_factor": 1.2,
            "offset_seed": 0,
            "unseen_check": {"window_tokens": 8, "hash_base": 1000003,
                             "checker_hash_base": 1000000007, "chunk_tokens": 1000,
                             "near_dup_window_frac": 0.5, "report_drop_frac": 0.05,
                             "report_top_prefixes": 10},
            "strip": {"math_envs": ["equation"]},
            "out_dir": (root / "eval").as_posix(),
        },
        "scoring": {
            "seed_base": 0,
            "seed_offsets": dict(SEED_OFFSETS),
            "n_samples": {"c0": 1, "mc_dropout": N_SAMPLES, "c1": N_SAMPLES, "c3": N_SAMPLES,
                          "c4_tfb": N_SAMPLES},
            "score_sets": {
                "legacy_d1": list(SCORE_SETS_ALL),
                "test": list(SCORE_SETS_ALL),
                "arxiv_stripped": list(SCORE_SETS_ALL),
                "test_hn": ["c3", "c4_tfb"],
            },
            "labels": copy.deepcopy(LABELS),
            "batch_size": {"legacy_d1": 1, "test": 1, "test_hn": 1, "arxiv_stripped": 1},
            "amp_fp16": True,
            "determinism": {"use_deterministic_algorithms": True, "warn_only": True,
                            "cudnn_deterministic": True, "cudnn_benchmark": False,
                            "cublas_workspace_config": ":4096:8"},
            "out_dir": (root / "scores").as_posix(),
        },
        "analysis": _analysis_block(),
        "checks": {
            "repro": {"spread_seed_bases": list(SPREAD_SEED_BASES), "spread_max_miss": 0.03},
            "rederive_tol": 1.0e-6,
        },
    }


def _analysis_sha(cfg: dict) -> str:
    text = json.dumps(cfg["analysis"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Script import and global-state helpers
# ---------------------------------------------------------------------------

def _load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _determinism_state() -> tuple:
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def _restore_determinism(state: tuple) -> None:
    enabled, warn_only, cudnn_det, cudnn_bench, cublas = state
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_bench
    if cublas is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas


def _run_main(main_fn, argv: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main_fn(argv)
    return buf.getvalue()


def _argv(yaml_path: Path, eval_set: str, run_tag: str, *extra: str) -> list[str]:
    argv = ["--eval-config", str(yaml_path), "--eval-set", eval_set, "--run-tag", run_tag,
            "--device", "cpu"]
    if eval_set == "legacy_d1":
        argv += ["--n-sequences", str(LEGACY_N)]
    return argv + list(extra)


def _load_scores(path: Path) -> dict:
    return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _gpt(cfg: dict) -> MiniGPT:
    return MiniGPT(build_gpt_config(cfg, vocab_size=VOCAB))


def _save_ckpt(model: torch.nn.Module, cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, path)


def _tfb_payload(det_model: torch.nn.Module, sigma_q: float) -> dict:
    """Raw TFB state dict in the saved-file format (minigpt/tfb.py save_tfb_state)."""
    svd_cache, a_map, names = {}, {}, []
    for name, module in det_model.named_modules():
        if isinstance(module, DeterministicLoRALinear):
            u, s, v = torch.linalg.svd(module.lora_B.data, full_matrices=False)
            svd_cache[name] = (u, s, v)
            a_map[f"{name}.lora_A"] = module.lora_A.data.detach().clone()
            names.append(f"{name}.lora_A")
    return {"sigma_q": sigma_q, "svd_cache": svd_cache, "a_map": a_map, "param_names": names,
            "epsilon": 0.1, "anchor_loss": 0.0}


def _laplace_payload(selection: dict[str, torch.Tensor], gen: torch.Generator) -> dict:
    """Raw Laplace state dict in the saved-file format (minigpt/laplace.py)."""
    return {
        "param_names": list(selection),
        "phi_hat": {k: v.detach().clone() for k, v in selection.items()},
        "curvature": {k: torch.rand(v.shape, generator=gen) * 1e-2 for k, v in selection.items()},
        "damping": 1.0,
        "sample_scale": LAPLACE_SAMPLE_SCALE,
    }


def _write_checkpoints(ecc, cfgs: dict, ckpt: Path, ckpt_sigma0: Path) -> None:
    torch.manual_seed(FIXTURE_SEED)
    gen = torch.Generator().manual_seed(FIXTURE_SEED)
    c0 = _gpt(cfgs["c0"])
    _save_ckpt(c0, cfgs["c0"], ckpt / "c0/ckpt_best.pt")
    c1 = _gpt(cfgs["c1"])
    _save_ckpt(c1, cfgs["c1"], ckpt / "c1/ckpt_best.pt")
    c3 = inject_lora(_gpt(cfgs["c3_phase2"]), build_lora_config(cfgs["c3_phase2"]),
                     bayesian=True)
    with torch.no_grad():
        for name, param in c3.named_parameters():
            if name.endswith("lora_B"):
                param.copy_(torch.randn(param.shape, generator=gen) * LORA_B_SCALE)
    _save_ckpt(c3, cfgs["c3_phase2"], ckpt / "c3/ckpt_best.pt")
    (ckpt / "c2").mkdir(parents=True, exist_ok=True)
    torch.save(_laplace_payload(select_params(c0, "ffn"), gen), ckpt / "c2/laplace_state.pt")
    # C4 states are built on the script's own BLoB-mean mapping (reads CKPT_DIR/c3).
    det = ecc._blob_to_deterministic_lora(_gpt(cfgs["c4_tfb"]), cfgs["c4_tfb"])
    (ckpt / "c4_tfb").mkdir(parents=True, exist_ok=True)
    torch.save(_tfb_payload(det, TFB_SIGMA_Q), ckpt / "c4_tfb/tfb_state.pt")
    (ckpt / "c4_lap").mkdir(parents=True, exist_ok=True)
    torch.save(_laplace_payload(select_params(det, "lora"), gen),
               ckpt / "c4_lap/laplace_state.pt")
    # sigma_q = 0 copy for T5c (identical samples).
    (ckpt_sigma0 / "c3").mkdir(parents=True, exist_ok=True)
    (ckpt_sigma0 / "c4_tfb").mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt / "c3/ckpt_best.pt", ckpt_sigma0 / "c3/ckpt_best.pt")
    torch.save(_tfb_payload(det, 0.0), ckpt_sigma0 / "c4_tfb/tfb_state.pt")


def _make_docs(domain: str, blocks_per_doc: list[int], gen: torch.Generator, start: int):
    docs = []
    for j, k in enumerate(blocks_per_doc):
        extra = int(torch.randint(0, 12, (1,), generator=gen))
        length = k * T + 1 + extra
        tokens = torch.randint(0, VOCAB, (length,), generator=gen, dtype=torch.int32)
        sha1 = hashlib.sha1(tokens.numpy().astype("<i4").tobytes()).hexdigest()
        offset = int(torch.randint(0, length - k * T, (1,), generator=gen))
        docs.append({"doc_id": f"{domain}/{start + j:09d}/{sha1[:12]}", "domain": domain,
                     "tokens": tokens, "sha1": sha1, "o_d": offset, "k_d": k})
    return docs


def _block_file(docs: list[dict]) -> dict:
    rows, doc_ids, domains, offsets, weights = [], [], [], [], []
    for d in docs:
        for j in range(d["k_d"]):
            start = d["o_d"] + j * T
            rows.append(d["tokens"][start : start + T + 1])
            doc_ids.append(d["doc_id"])
            domains.append(d["domain"])
            offsets.append(start)
            weights.append(1.0 / d["k_d"])
    return {
        "tokens": torch.stack(rows).to(torch.int32),
        "doc_id": doc_ids,
        "domain": domains,
        "offset": torch.tensor(offsets, dtype=torch.int64),
        "weight": torch.tensor(weights, dtype=torch.float32),
    }


def _manifest_rows(docs: list[dict], split: str, variant: str) -> list[dict]:
    return [
        {"doc_id": d["doc_id"], "domain": d["domain"], "split": split, "variant": variant,
         "stream_index": i, "c_i": 0, "L_d": int(d["tokens"].numel()),
         "token_sha1": d["sha1"], "o_d": d["o_d"], "k_d": d["k_d"]}
        for i, d in enumerate(docs)
    ]


def _write_eval_set(eval_dir: Path) -> dict:
    gen = torch.Generator().manual_seed(FIXTURE_SEED + 1)
    eval_dir.mkdir(parents=True, exist_ok=True)
    test_docs = []
    for i, domain in enumerate(TEST_DOMAINS):
        test_docs += _make_docs(domain, BLOCKS_PER_DOC, gen, start=1000 * i)
    hn_docs = _make_docs("hackernews", HN_BLOCKS_PER_DOC, gen, start=5000)
    arxiv_raw = [d for d in test_docs if d["domain"] == "arxiv"]
    stripped = []
    for d in arxiv_raw:
        s = _make_docs("arxiv", [d["k_d"]], gen, start=0)[0]
        s["doc_id"] = d["doc_id"]
        stripped.append(s)
    val_docs = _make_docs("stackexchange", [1, 1], gen, start=9000)
    rows = (_manifest_rows(test_docs + hn_docs, "test", "raw")
            + _manifest_rows(stripped, "test", "stripped")
            + _manifest_rows(val_docs, "val", "raw"))
    with open(eval_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    files = {"test": _block_file(test_docs), "test_hn": _block_file(hn_docs),
             "arxiv_stripped": _block_file(stripped)}
    for name, payload in files.items():
        torch.save(payload, eval_dir / f"blocks_{name}.pt")
    return {"files": files, "val_docs": val_docs}


def _legacy_tensors() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(FIXTURE_SEED + 2)
    return {
        "train": torch.randint(0, VOCAB, (4 * T,), generator=gen),
        "val": torch.randint(0, VOCAB, (2 * T,), generator=gen),
        "test_id": torch.randint(0, VOCAB, (LEGACY_N * T + 7,), generator=gen),
        "test_ood_arxiv": torch.randint(0, VOCAB, (LEGACY_ARXIV_TOKENS,), generator=gen),
        "test_ood_freelaw": torch.randint(0, VOCAB, (4 * T,), generator=gen),
        "test_ood_pubmed_abstracts": torch.randint(0, VOCAB, (2 * T,), generator=gen),
    }


def _write_yaml(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _write_prereg(cfg: dict, sha: str | None = None) -> Path:
    out = Path(cfg["scoring"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prereg.json"
    payload = {"analysis_sha256": sha or _analysis_sha(cfg), "frozen_at": PREREG_FROZEN_AT}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scripts():
    ecc = _load_script("eval_c_checkpoints")
    mcd = _load_script("eval_mc_dropout")
    return SimpleNamespace(ecc=ecc, mcd=mcd)


@pytest.fixture(scope="module")
def world(tmp_path_factory, scripts):
    ecc, mcd = scripts.ecc, scripts.mcd
    root = tmp_path_factory.mktemp("s1_t5")
    real_build = ecc.build_milestone_config
    cfgs = {}
    for m in FIXTURE_MILESTONES:
        overrides = dict(MODEL_OVERRIDES)
        if m in ("c3_phase2", "c4_tfb", "c4_lap"):
            overrides.update(LORA_OVERRIDES)
        cfgs[m] = real_build(m, overrides)

    def fake_build(milestone: str, overrides: dict | None = None) -> dict:
        return copy.deepcopy(cfgs[milestone])

    legacy = _legacy_tensors()
    ckpt, ckpt_sigma0 = root / "ckpt", root / "ckpt_sigma0"
    saved_det = _determinism_state()
    mp = pytest.MonkeyPatch()
    for mod in (ecc, mcd):
        mp.setattr(mod, "CKPT_DIR", ckpt)
        mp.setattr(mod, "build_milestone_config", fake_build)
        mp.setattr(mod, "get_tokenizer", lambda: SimpleNamespace(n_vocab=VOCAB))
        mp.setattr(mod, "load_pile_data", lambda cfg, tok: dict(legacy))
    try:
        _write_checkpoints(ecc, cfgs, ckpt, ckpt_sigma0)
        eval_info = _write_eval_set(root / "eval")
        cfg = _fixture_yaml(root)
        yaml_path = _write_yaml(cfg, root / "i2_eval_fixture.yaml")
        _write_prereg(cfg)
        out = {}

        def run(script: str, eval_set: str, tag: str, *extra: str) -> str:
            fn = ecc.main if script == "ecc" else mcd.main
            text = _run_main(fn, _argv(yaml_path, eval_set, tag, *extra))
            out[(script, eval_set, tag)] = text
            return text

        t0 = time.perf_counter()
        for tag in ("main", "run2"):
            run("ecc", "test", tag)
            run("mcd", "test", tag)
        run("ecc", "legacy_d1", "main")
        run("mcd", "legacy_d1", "main")
        run("ecc", "legacy_d1", "rerun", "--block-ids", "0-1,6-7")
        run("mcd", "legacy_d1", "rerun", "--block-ids", "0-1,6-7")
        run("ecc", "legacy_d1", "batched", "--batch-size", "2")
        run("mcd", "legacy_d1", "batched", "--batch-size", "2")
        run("ecc", "test_hn", "main")
        run("ecc", "arxiv_stripped", "main")
        run("mcd", "arxiv_stripped", "main")
        mp.setattr(ecc, "CKPT_DIR", ckpt_sigma0)
        run("ecc", "test", "sigma0", "--score-set", "c4_tfb")
        mp.setattr(ecc, "CKPT_DIR", ckpt)
        analyze_stdout = _run_main(ecc.main, ["--eval-config", str(yaml_path), "--analyze"])
        scoring_seconds = time.perf_counter() - t0
        yield SimpleNamespace(
            root=root, cfg=cfg, cfgs=cfgs, yaml_path=yaml_path, ckpt=ckpt,
            ckpt_sigma0=ckpt_sigma0, scores=Path(cfg["scoring"]["out_dir"]),
            eval_dir=root / "eval", eval_info=eval_info, legacy=legacy, stdout=out,
            analyze_stdout=analyze_stdout, scoring_seconds=scoring_seconds,
        )
    finally:
        mp.undo()
        _restore_determinism(saved_det)


def _path(world, score_set: str, eval_set: str, tag: str) -> Path:
    return world.scores / f"{score_set}__{eval_set}__{tag}.pt"


# ---------------------------------------------------------------------------
# S1-T5a: keys, shapes, dtypes, meta; alignment across score files
# ---------------------------------------------------------------------------

def _check_score_file(payload: dict, n_blocks: int, n_samples: int) -> None:
    assert payload["logp_real"].shape == (n_blocks, n_samples, T)
    assert payload["logp_real"].dtype == torch.float32
    for key in TENSOR_KEYS_BT_F32:
        assert payload[key].shape == (n_blocks, T), key
        assert payload[key].dtype == torch.float32, key
    assert payload["tok_correct"].shape == (n_blocks, T)
    assert payload["tok_correct"].dtype == torch.bool
    for key in BLOCK_KEYS_F32:
        assert payload[key].shape == (n_blocks,), key
        assert payload[key].dtype == torch.float32, key
    assert payload["block_index"].shape == (n_blocks,)
    assert payload["block_index"].dtype == torch.int64
    assert payload["offset"].shape == (n_blocks,)
    assert payload["offset"].dtype == torch.int64
    assert payload["weight"].shape == (n_blocks,)
    assert payload["weight"].dtype == torch.float32
    assert isinstance(payload["doc_id"], list) and len(payload["doc_id"]) == n_blocks
    assert isinstance(payload["domain"], list) and len(payload["domain"]) == n_blocks
    assert all(isinstance(x, str) for x in payload["doc_id"] + payload["domain"])
    missing = [k for k in META_KEYS if k not in payload["meta"]]
    assert not missing, missing


def _expected_ckpts(world, score_set: str) -> list[Path]:
    ck = world.ckpt
    return {
        "c0": [ck / "c0/ckpt_best.pt"],
        "mc_dropout": [ck / "c0/ckpt_best.pt"],
        "c1": [ck / "c1/ckpt_best.pt"],
        "c3": [ck / "c3/ckpt_best.pt"],
        "c4_tfb": [ck / "c3/ckpt_best.pt", ck / "c4_tfb/tfb_state.pt"],
    }[score_set]


@pytest.mark.parametrize("eval_set,tag,n_blocks", [
    ("test", "main", 3 * sum(BLOCKS_PER_DOC)),
    ("legacy_d1", "main", 2 * LEGACY_N),
    ("arxiv_stripped", "main", sum(BLOCKS_PER_DOC)),
])
def test_t5a_score_files_keys_shapes_dtypes_and_meta(world, eval_set, tag, n_blocks):
    manifest_sha = _sha_file(world.eval_dir / "manifest.jsonl")
    for score_set in SCORE_SETS_ALL:
        payload = _load_scores(_path(world, score_set, eval_set, tag))
        n = world.cfg["scoring"]["n_samples"][score_set]
        _check_score_file(payload, n_blocks, n)
        meta = payload["meta"]
        label = LABELS[score_set]
        assert meta["score_set"] == score_set
        assert meta["display_label"] == label["display"]
        assert meta["sampler"] == label["sampler"]
        assert meta["adapter_source"] == label["adapter_source"]
        assert meta["eval_set"] == eval_set
        assert meta["run_tag"] == tag
        assert meta["block_ids"] is None
        assert meta["n_samples"] == n
        assert meta["seed_base"] == 0
        assert meta["seed_offset"] == SEED_OFFSETS[eval_set]
        assert meta["batch_size"] == 1
        assert meta["amp_fp16"] is False  # CPU: autocast is not applied
        assert meta["device"] == "cpu"
        assert meta["torch_version"] == torch.__version__
        assert isinstance(meta["determinism_warnings"], list)
        assert meta["yaml_sha256"] == _sha_file(world.yaml_path)
        assert meta["analysis_sha256"] == _analysis_sha(world.cfg)
        expected_manifest = None if eval_set == "legacy_d1" else manifest_sha
        assert meta["manifest_sha256"] == expected_manifest
        assert re.fullmatch(r"[0-9a-f]{64}", meta["code_sha256"])
        assert meta["checkpoint_sha256"] == {
            p.as_posix(): _sha_file(p) for p in _expected_ckpts(world, score_set)
        }
        assert datetime.fromisoformat(meta["created_at"]).utcoffset().total_seconds() == 0
        assert meta["git_dirty"] in (True, False, None)


def test_t5a_new_set_rows_follow_block_file(world):
    blocks = world.eval_info["files"]["test"]
    payload = _load_scores(_path(world, "c1", "test", "main"))
    assert payload["doc_id"] == blocks["doc_id"]
    assert payload["domain"] == blocks["domain"]
    assert torch.equal(payload["offset"], blocks["offset"])
    assert torch.equal(payload["weight"], blocks["weight"])
    assert torch.equal(payload["block_index"], torch.arange(len(blocks["doc_id"])))


@pytest.mark.parametrize("eval_set,tag", [
    ("test", "main"), ("legacy_d1", "main"), ("arxiv_stripped", "main"),
])
def test_t5a_doc_domain_offset_identical_across_score_files(world, eval_set, tag):
    payloads = [_load_scores(_path(world, s, eval_set, tag)) for s in SCORE_SETS_ALL]
    ref = payloads[0]
    for p in payloads[1:]:
        assert p["doc_id"] == ref["doc_id"]
        assert p["domain"] == ref["domain"]
        assert torch.equal(p["offset"], ref["offset"])
        assert torch.equal(p["block_index"], ref["block_index"])


def test_t5a_ecc_skips_mc_dropout_and_mcd_writes_it(world):
    written = sorted(p.name for p in world.scores.glob("*__test__main.pt"))
    assert written == sorted(f"{s}__test__main.pt" for s in SCORE_SETS_ALL)
    assert "mc_dropout" in world.stdout[("ecc", "test", "main")]


def test_test_hn_scores_only_its_listed_sets(world):
    written = sorted(p.name for p in world.scores.glob("*__test_hn__main.pt"))
    assert written == ["c3__test_hn__main.pt", "c4_tfb__test_hn__main.pt"]
    payload = _load_scores(_path(world, "c3", "test_hn", "main"))
    assert set(payload["domain"]) == {"hackernews"}
    assert payload["meta"]["seed_offset"] == SEED_OFFSETS["test_hn"]


# ---------------------------------------------------------------------------
# S1-T5c: Jensen gap is non-negative; zero cases
# ---------------------------------------------------------------------------

def _gap(logp_real: torch.Tensor) -> torch.Tensor:
    ell = logp_real.double()
    n = ell.shape[1]
    log_pbar = torch.logsumexp(ell, dim=1) - math.log(n)
    return log_pbar - ell.mean(dim=1)


def test_t5c_gap_nonnegative_and_block_g_matches_logp_real(world):
    files = sorted(world.scores.glob("*.pt"))
    assert len(files) >= 20
    for path in files:
        payload = _load_scores(path)
        g = _gap(payload["logp_real"])
        assert float(g.min()) >= -1e-6, path.name
        assert torch.allclose(payload["blk_g"].double(), g.mean(dim=1), atol=1e-6), path.name
        ell = payload["logp_real"].double()
        log_pbar = torch.logsumexp(ell, dim=1) - math.log(ell.shape[1])
        assert torch.allclose(payload["blk_nll"].double(), (-log_pbar).mean(dim=1), atol=1e-5)
        assert torch.allclose(payload["blk_mi"].double(),
                              payload["tok_mi"].double().mean(dim=1), atol=1e-6)
        assert torch.allclose(payload["blk_maxprob_unc"].double(),
                              (1.0 - payload["tok_maxprob"].double()).mean(dim=1), atol=1e-6)


def test_t5c_c0_has_one_sample_and_zero_g_and_mi(world):
    for eval_set, tag in (("test", "main"), ("legacy_d1", "main")):
        payload = _load_scores(_path(world, "c0", eval_set, tag))
        assert payload["logp_real"].shape[1] == 1
        assert payload["meta"]["n_samples"] == 1
        assert float(payload["blk_g"].abs().max()) <= 1e-6
        assert float(payload["blk_mi"].abs().max()) <= 1e-6


def test_t5c_tfb_sigma_zero_gives_zero_g_and_mi(world):
    payload = _load_scores(_path(world, "c4_tfb", "test", "sigma0"))
    assert payload["logp_real"].shape[1] == N_SAMPLES
    assert float(payload["blk_g"].abs().max()) <= 1e-6
    assert float(payload["blk_mi"].abs().max()) <= 1e-5
    noisy = _load_scores(_path(world, "c4_tfb", "test", "main"))
    assert float(noisy["blk_mi"].max()) > 1e-5  # the fixture posterior is not degenerate


# ---------------------------------------------------------------------------
# S1-T5d: log p_bar(y_t) from logp_real equals the full-vocabulary path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score_set", ["c1", "c3", "c4_tfb", "mc_dropout"])
def test_t5d_realized_token_logpbar_matches_full_vocab_path(world, scripts, score_set):
    ecc = scripts.ecc
    device = torch.device("cpu")
    if score_set == "mc_dropout":
        model, method, state = scripts.mcd.load_c0_model(device), "dropout", None
    else:
        model, method, state = ecc.load_model(score_set, device)
    tokens = world.eval_info["files"]["test"]["tokens"][:4].long()
    out = ecc.score_block_batch(model, tokens[:, :-1], tokens[:, 1:], N_SAMPLES, device,
                                method, state, seed_b=7, amp=False)
    rs = ecc.realized_token_scores(out["logp_real"])
    p_full = out["tok_pbar_true"].double()
    mask = p_full >= 1e-30
    assert bool(mask.all())
    diff = (rs["log_pbar"] - torch.log(p_full)).abs()[mask]
    assert float(diff.max()) <= 1e-4


def test_realized_token_scores_formula():
    ell = torch.tensor([[[-1.0, -2.0], [-3.0, -2.0]]])  # [B=1, N=2, T=2]
    ecc = _load_script("eval_c_checkpoints")
    rs = ecc.realized_token_scores(ell)
    log_pbar = torch.log(torch.tensor([(math.exp(-1) + math.exp(-3)) / 2, math.exp(-2)]))
    assert torch.allclose(rs["log_pbar"][0], log_pbar.double(), atol=1e-12)
    g = log_pbar.double() - torch.tensor([-2.0, -2.0]).double()
    assert torch.allclose(rs["g"][0], g, atol=1e-12)
    assert torch.allclose(rs["G"], g.mean().reshape(1), atol=1e-12)


# ---------------------------------------------------------------------------
# S1-T5e: two runs give identical tensors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score_set", SCORE_SETS_ALL)
def test_t5e_two_runs_identical(world, score_set):
    a = _load_scores(_path(world, score_set, "test", "main"))
    b = _load_scores(_path(world, score_set, "test", "run2"))
    for key, value in a.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, b[key]), key
    assert a["doc_id"] == b["doc_id"]
    assert a["domain"] == b["domain"]


# ---------------------------------------------------------------------------
# S1-T5f: MI_RATIOS is gone; printed ratios come from the file
# ---------------------------------------------------------------------------

def test_t5f_mi_ratios_constant_deleted():
    for name in ("eval_c_checkpoints.py", "eval_mc_dropout.py"):
        text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert "MI_RATIOS" not in text, name


def test_t5f_printed_ratios_equal_file_ratios(world):
    n_checked = 0
    for (script, eval_set, tag), text in world.stdout.items():
        for line in text.splitlines():
            m = RATIO_RE.match(line.strip())
            if not m:
                continue
            fname, score_set, es, ood, idd, value = m.groups()
            assert es == eval_set
            payload = _load_scores(world.scores / fname)
            assert payload["meta"]["score_set"] == score_set
            mi = payload["blk_mi"].double()
            dom = payload["domain"]
            id_mask = torch.tensor([d == idd for d in dom])
            ood_mask = (torch.tensor([d == ood for d in dom]) if ood != "ood"
                        else ~id_mask)
            ratio = float(mi[ood_mask].mean() / mi[id_mask].mean())
            assert abs(float(value) - ratio) <= 1e-6, line
            n_checked += 1
    # c1, c3, c4_tfb (ecc) and mc_dropout: 2 OOD domains on test, pooled on legacy_d1
    assert n_checked >= 4 * 2 * 2


# ---------------------------------------------------------------------------
# S1-T5g: the freeze guard
# ---------------------------------------------------------------------------

def _variant(world, tmp_path: Path, mutate=None) -> tuple[dict, Path]:
    cfg = copy.deepcopy(world.cfg)
    cfg["scoring"]["out_dir"] = (tmp_path / "scores").as_posix()
    if mutate is not None:
        mutate(cfg)
    return cfg, _write_yaml(cfg, tmp_path / "variant.yaml")


@pytest.mark.parametrize("script", ["ecc", "mcd"])
@pytest.mark.parametrize("prereg", ["wrong_hash", "missing"])
@pytest.mark.parametrize("eval_set", ["test", "arxiv_stripped"])
def test_t5g_frozen_sets_refused_without_matching_prereg(world, scripts, tmp_path, script,
                                                         prereg, eval_set):
    cfg, yaml_path = _variant(world, tmp_path)
    if prereg == "wrong_hash":
        _write_prereg(cfg, sha="0" * 64)
    main = scripts.ecc.main if script == "ecc" else scripts.mcd.main
    with pytest.raises(SystemExit) as exc:
        _run_main(main, _argv(yaml_path, eval_set, "main"))
    assert exc.value.code not in (0, None)
    assert not list(Path(cfg["scoring"]["out_dir"]).glob("*.pt"))


def test_t5g_test_hn_refused_and_legacy_still_runs(world, scripts, tmp_path):
    cfg, yaml_path = _variant(world, tmp_path)
    _write_prereg(cfg, sha="f" * 64)
    with pytest.raises(SystemExit) as exc:
        _run_main(scripts.ecc.main, _argv(yaml_path, "test_hn", "main"))
    assert exc.value.code not in (0, None)
    out = Path(cfg["scoring"]["out_dir"])
    assert not list(out.glob("*.pt"))
    _run_main(scripts.ecc.main, _argv(yaml_path, "legacy_d1", "main", "--score-set", "c0",
                                      "--block-ids", "0-1"))
    _run_main(scripts.mcd.main, _argv(yaml_path, "legacy_d1", "main", "--block-ids", "0-1"))
    assert sorted(p.name for p in out.glob("*.pt")) == [
        "c0__legacy_d1__main.pt", "mc_dropout__legacy_d1__main.pt",
    ]


def test_t5g_changed_analysis_block_is_refused(world, scripts, tmp_path):
    """Editing the analysis block after the freeze changes its hash, so scoring stops."""
    def mutate(cfg):
        cfg["analysis"]["margin_auroc"] = 0.03
    cfg, yaml_path = _variant(world, tmp_path, mutate)
    _write_prereg(cfg, sha=_analysis_sha(world.cfg))
    with pytest.raises(SystemExit) as exc:
        _run_main(scripts.ecc.main, _argv(yaml_path, "test", "main", "--score-set", "c0"))
    assert exc.value.code not in (0, None)
    assert not list(Path(cfg["scoring"]["out_dir"]).glob("*.pt"))


# ---------------------------------------------------------------------------
# S1-T1 legacy_d1 path on the fixture
# ---------------------------------------------------------------------------

def test_t1_legacy_block_layout_and_ids(world):
    payload = _load_scores(_path(world, "c0", "legacy_d1", "main"))
    n = LEGACY_N
    assert payload["block_index"].tolist() == list(range(2 * n))
    domains = ["stackexchange"] * n + ["arxiv"] * 4 + ["freelaw"] * 2
    assert payload["domain"] == domains
    assert payload["doc_id"] == [f"legacy/{d}/{b}" for b, d in enumerate(domains)]
    assert torch.equal(payload["weight"], torch.ones(2 * n))
    assert payload["offset"].tolist() == [i * T for i in range(n)] * 2
    assert payload["meta"]["manifest_sha256"] is None
    assert payload["meta"]["seed_offset"] == 0


def test_t1_legacy_blocks_are_first_windows_of_test_id_then_ood(world, scripts):
    """C0 is deterministic, so its realized log-probs pin down which tokens were scored."""
    model, _, _ = scripts.ecc.load_model("c0", torch.device("cpu"))
    payload = _load_scores(_path(world, "c0", "legacy_d1", "main"))
    ood = torch.cat([world.legacy[f"test_ood_{d}"]
                     for d in ("arxiv", "freelaw", "pubmed_abstracts")])
    for b in range(2 * LEGACY_N):
        src, i = (world.legacy["test_id"], b) if b < LEGACY_N else (ood, b - LEGACY_N)
        x, y = src[i * T : i * T + T], src[i * T + 1 : i * T + T + 1]
        with torch.no_grad():
            logits, _ = model(x[None])
        lp = torch.log_softmax(logits.float(), dim=-1)[0]
        ell = lp.gather(-1, y[:, None]).squeeze(-1)
        assert torch.allclose(payload["logp_real"][b, 0], ell, atol=1e-6), b


def _old_d1_loop_mi(model, state, x: torch.Tensor, first_seed: int) -> torch.Tensor:
    """Verbatim copy of the old per-sample loop (eval_c_checkpoints.py:241-263, a7a73b6)."""
    eps = 1e-10
    p_sum = torch.zeros(T, VOCAB)
    entropy_sum = torch.zeros(T)
    with torch.no_grad():
        for s in range(N_SAMPLES):
            sampled = sample_tfb_params(state, seed=first_seed + s)
            with apply_sampled_params(model, sampled):
                logits, _ = model(x[None])
            probs = F.softmax(logits[0].float(), dim=-1)
            p_sum.add_(probs)
            entropy_sum.add_(-(probs * torch.log(probs + eps)).sum(dim=-1))
    p_bar = p_sum / N_SAMPLES
    return -(p_bar * torch.log(p_bar + eps)).sum(dim=-1) - entropy_sum / N_SAMPLES


def test_t1_legacy_tfb_matches_old_seeding_rule(world, scripts):
    """On legacy_d1 the TFB seed is b * N + s, the old rule (eval_c_checkpoints.py:243).

    The new path takes probs as exp(log_softmax) instead of softmax, so token MI agrees to
    float32 rounding of entropies near log(VOCAB) (tolerance 1e-5). A wrong seed moves MI
    by far more than that.
    """
    model, method, _ = scripts.ecc.load_model("c4_tfb", torch.device("cpu"))
    assert method == "tfb"
    state = load_tfb_state(world.ckpt / "c4_tfb/tfb_state.pt")
    payload = _load_scores(_path(world, "c4_tfb", "legacy_d1", "main"))
    for b in (0, LEGACY_N + 1):
        src = world.legacy["test_id"] if b < LEGACY_N else world.legacy["test_ood_arxiv"]
        i = b if b < LEGACY_N else b - LEGACY_N
        x = src[i * T : i * T + T]
        mi = _old_d1_loop_mi(model, state, x, first_seed=b * N_SAMPLES)
        assert float((payload["tok_mi"][b] - mi).abs().max()) <= 1e-5, b
        wrong = _old_d1_loop_mi(model, state, x, first_seed=b * N_SAMPLES + 1000)
        assert float((payload["tok_mi"][b] - wrong).abs().max()) > 1e-4, b


def test_seeding_rule_vi_block_seed_is_base_plus_offset_plus_index(world, scripts):
    """VI draws use torch.manual_seed(seed_base + seed_offset[eval_set] + b) (section 5.5)."""
    model, method, _ = scripts.ecc.load_model("c1", torch.device("cpu"))
    assert method == "variational"
    payload = _load_scores(_path(world, "c1", "test", "main"))
    tokens = world.eval_info["files"]["test"]["tokens"].long()
    for b in (0, 17, 45):
        torch.manual_seed(0 + SEED_OFFSETS["test"] + b)
        ell = torch.zeros(N_SAMPLES, T)
        with torch.no_grad():
            for s in range(N_SAMPLES):
                logits, _ = model(tokens[b, :-1][None])
                lp = torch.log_softmax(logits.float(), dim=-1)[0]
                ell[s] = lp.gather(-1, tokens[b, 1:][:, None]).squeeze(-1)
        assert torch.equal(payload["logp_real"][b], ell), b


@pytest.mark.parametrize("score_set", SCORE_SETS_ALL)
def test_t1d_rerun_rows_equal_main_rows(world, score_set):
    main = _load_scores(_path(world, score_set, "legacy_d1", "main"))
    rerun = _load_scores(_path(world, score_set, "legacy_d1", "rerun"))
    rows = [0, 1, LEGACY_N, LEGACY_N + 1]
    assert rerun["block_index"].tolist() == rows
    assert rerun["meta"]["block_ids"] == "0-1,6-7"
    for key, value in rerun.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, main[key][rows]), key
    assert rerun["doc_id"] == [main["doc_id"][r] for r in rows]


def test_t1f_batched_run_records_batch_size_and_c0_matches(world):
    for score_set in SCORE_SETS_ALL:
        batched = _load_scores(_path(world, score_set, "legacy_d1", "batched"))
        assert batched["meta"]["batch_size"] == 2
        assert batched["block_index"].tolist() == list(range(2 * LEGACY_N))
    main = _load_scores(_path(world, "c0", "legacy_d1", "main"))
    batched = _load_scores(_path(world, "c0", "legacy_d1", "batched"))
    for key in BLOCK_KEYS_F32:
        assert float((main[key] - batched[key]).abs().max()) <= 1e-3, key


def test_t1f_tfb_batch_shares_one_draw_seeded_by_first_block(world):
    """With batch size 2, blocks 2k and 2k+1 share the draw seeded by block 2k."""
    main = _load_scores(_path(world, "c4_tfb", "legacy_d1", "main"))
    batched = _load_scores(_path(world, "c4_tfb", "legacy_d1", "batched"))
    assert torch.allclose(batched["logp_real"][0], main["logp_real"][0], atol=1e-5)
    assert not torch.allclose(batched["logp_real"][1], main["logp_real"][1], atol=1e-5)


@pytest.mark.parametrize("bad", ["12345", "1"])
def test_seed_base_accepts_only_config_values(world, scripts, tmp_path, bad):
    cfg, yaml_path = _variant(world, tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run_main(scripts.ecc.main, _argv(yaml_path, "legacy_d1", "spread", "--seed-base", bad,
                                          "--score-set", "c1", "--block-ids", "0"))
    assert exc.value.code not in (0, None)
    assert not list(Path(cfg["scoring"]["out_dir"]).glob("*.pt"))


def test_seed_base_spread_value_is_used(world, scripts, tmp_path):
    cfg, yaml_path = _variant(world, tmp_path)
    base = SPREAD_SEED_BASES[0]
    _run_main(scripts.ecc.main, _argv(yaml_path, "legacy_d1", "spread1", "--seed-base",
                                      str(base), "--score-set", "c1", "--block-ids", "1"))
    payload = _load_scores(Path(cfg["scoring"]["out_dir"]) / "c1__legacy_d1__spread1.pt")
    assert payload["meta"]["seed_base"] == base
    main = _load_scores(_path(world, "c1", "legacy_d1", "main"))
    assert not torch.equal(payload["logp_real"][0], main["logp_real"][1])


def test_existing_score_file_is_not_overwritten(world, scripts):
    path = _path(world, "c0", "legacy_d1", "main")
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        _run_main(scripts.ecc.main, _argv(world.yaml_path, "legacy_d1", "main",
                                          "--score-set", "c0"))
    assert exc.value.code not in (0, None)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# Registry hook (S2 v2 samplers register by name) and loaders
# ---------------------------------------------------------------------------

def test_registry_hook_scores_a_new_score_set_by_name(world, scripts, tmp_path, monkeypatch):
    ecc = scripts.ecc
    monkeypatch.setattr(ecc, "SCORE_SET_REGISTRY", dict(ecc.SCORE_SET_REGISTRY))
    ecc.register_score_set(
        "c4_tfb_copy",
        checkpoint_paths=lambda: ecc._checkpoint_paths("c4_tfb"),
        load=lambda device: ecc.load_model("c4_tfb", device),
    )
    with pytest.raises(ValueError):
        ecc.register_score_set("c4_tfb_copy", checkpoint_paths=list, load=lambda d: None)

    def mutate(cfg):
        cfg["scoring"]["labels"]["c4_tfb_copy"] = {
            "display": "C4-TFB copy", "sampler": "pre-fix", "adapter_source": "blob_mean",
        }
        cfg["scoring"]["n_samples"]["c4_tfb_copy"] = N_SAMPLES
        cfg["scoring"]["score_sets"]["legacy_d1"].append("c4_tfb_copy")

    cfg, yaml_path = _variant(world, tmp_path, mutate)
    _run_main(ecc.main, _argv(yaml_path, "legacy_d1", "reg", "--score-set", "c4_tfb_copy",
                              "--block-ids", "0-1"))
    payload = _load_scores(Path(cfg["scoring"]["out_dir"]) / "c4_tfb_copy__legacy_d1__reg.pt")
    assert payload["meta"]["display_label"] == "C4-TFB copy"
    assert payload["meta"]["sampler"] == "pre-fix"
    main = _load_scores(_path(world, "c4_tfb", "legacy_d1", "main"))
    assert torch.equal(payload["logp_real"], main["logp_real"][:2])


def test_param_sampler_registry_dispatches_by_method(scripts, monkeypatch):
    ecc = scripts.ecc
    assert {"laplace", "tfb"} <= set(ecc.PARAM_SAMPLERS)
    monkeypatch.setattr(ecc, "PARAM_SAMPLERS", dict(ecc.PARAM_SAMPLERS))
    ecc.register_param_sampler("tfb_v2_stub", ecc.PARAM_SAMPLERS["tfb"])
    assert ecc.PARAM_SAMPLERS["tfb_v2_stub"] is ecc.PARAM_SAMPLERS["tfb"]
    with pytest.raises(ValueError):
        ecc.register_param_sampler("tfb", ecc.PARAM_SAMPLERS["tfb"])


UNREGISTERED = "c9_unregistered"


def _add_unregistered(cfg: dict, eval_set: str) -> None:
    cfg["scoring"]["labels"][UNREGISTERED] = {
        "display": "not registered", "sampler": "v2", "adapter_source": "none",
    }
    cfg["scoring"]["n_samples"][UNREGISTERED] = N_SAMPLES
    cfg["scoring"]["score_sets"][eval_set].append(UNREGISTERED)


def test_unregistered_score_set_fails_before_scoring(world, scripts, tmp_path):
    cfg, yaml_path = _variant(world, tmp_path, lambda c: _add_unregistered(c, "legacy_d1"))
    with pytest.raises(SystemExit) as exc:
        _run_main(scripts.ecc.main, _argv(yaml_path, "legacy_d1", "main", "--block-ids", "0"))
    assert exc.value.code not in (0, None)
    assert not list(Path(cfg["scoring"]["out_dir"]).glob("*.pt"))


def test_mc_dropout_script_rejects_an_unregistered_score_set(world, scripts, tmp_path):
    """eval_mc_dropout.py defers only the score sets the shared registry knows."""
    cfg, yaml_path = _variant(world, tmp_path, lambda c: _add_unregistered(c, "legacy_d1"))
    with pytest.raises(SystemExit) as exc:
        _run_main(scripts.mcd.main, _argv(yaml_path, "legacy_d1", "main", "--block-ids", "0",
                                          "--score-set", f"mc_dropout,{UNREGISTERED}"))
    assert exc.value.code not in (0, None)
    assert not list(Path(cfg["scoring"]["out_dir"]).glob("*.pt"))


def test_posthoc_laplace_loaders_score_through_registry(world, scripts, tmp_path):
    """C2 and C4-LAP are not S1 score sets, but their loaders feed S2's refit entries."""
    def mutate(cfg):
        for name in ("c2", "c4_lap"):
            cfg["scoring"]["labels"][name] = {"display": name, "sampler": "pre-fix",
                                              "adapter_source": "none"}
            cfg["scoring"]["n_samples"][name] = N_SAMPLES
            cfg["scoring"]["score_sets"]["legacy_d1"].append(name)

    cfg, yaml_path = _variant(world, tmp_path, mutate)
    _run_main(scripts.ecc.main, _argv(yaml_path, "legacy_d1", "lap", "--score-set", "c2,c4_lap",
                                      "--block-ids", "0,7"))
    for name in ("c2", "c4_lap"):
        payload = _load_scores(Path(cfg["scoring"]["out_dir"]) / f"{name}__legacy_d1__lap.pt")
        assert payload["block_index"].tolist() == [0, 7]
        assert float(_gap(payload["logp_real"]).min()) >= -1e-6
        assert float(payload["blk_mi"].min()) > 0.0


# ---------------------------------------------------------------------------
# S2 score sets through the registry (specs/i2-posthoc-fixes.md sections 2 and 5.5)
# ---------------------------------------------------------------------------

S2_SCORE_SETS = ("c2_refit", "c4_lap_refit", "c4_tfb_fixed")
S2_REAL_REFIT_YAMLS = {
    "c2_refit": "configs/i2_posthoc_c2.yaml",
    "c4_lap_refit": "configs/i2_posthoc_c4_lap.yaml",
    "c4_tfb_fixed": "configs/i2_posthoc_c4_tfb.yaml",
}
S2_N_DATA_SEQS = 10
S2_PRIOR_PREC = 100.0
S2_MODEL_KEYS = ("block_size", "n_layer", "n_head", "n_embd", "dropout", "bias")


def _s2_refit_cfg(world, score_set: str, out_dir: Path) -> dict:
    """The real refit YAML, with the fixture's base, model, LoRA and output folder."""
    cfg = yaml.safe_load((REPO_ROOT / S2_REAL_REFIT_YAMLS[score_set]).read_text(encoding="utf-8"))
    full = cfg["base"]["base_kind"] == "full"
    source = world.cfgs["c0" if full else "c3_phase2"]
    cfg["model"] = {k: source["model"][k] for k in S2_MODEL_KEYS}
    if not full:
        lora = source["lora"]
        cfg["lora"] = {"rank": lora["rank"], "alpha": float(lora["alpha"]),
                       "target": lora["target"]}
    if cfg["method"] == "laplace":
        cfg["laplace"]["tokens_per_seq"] = T
    cfg["base"]["base_checkpoint"] = (world.ckpt / ("c0" if full else "c3") /
                                      "ckpt_best.pt").as_posix()
    cfg["out_dir"] = out_dir.as_posix()
    cfg["device"] = "cpu"
    return cfg


def _write_s2_cell(world, score_set: str, root: Path, *, state_version: str = "v2",
                   edit_cfg=None, edit_record=None) -> SimpleNamespace:
    """Refit YAML, state file and fit record of one S2 cell, as scripts/refit_posthoc.py
    lays them out (``out_dir``/{tfb_state.pt, laplace_state.pt, fit_record.json})."""
    from minigpt import posthoc_refit
    from minigpt.laplace import LaplaceState, save_laplace_state, scale_laplace_state
    from minigpt.tfb import TFBState, save_tfb_state

    out_dir = root / score_set
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _s2_refit_cfg(world, score_set, out_dir)
    if edit_cfg is not None:
        edit_cfg(cfg)
    model, info = posthoc_refit.load_posthoc_base(cfg)
    if cfg["method"] == "tfb":
        state_path = out_dir / "tfb_state.pt"
        payload = _tfb_payload(model, TFB_SIGMA_Q)
        save_tfb_state(TFBState(**payload, sampler_version=state_version), state_path)
    else:
        state_path = out_dir / "laplace_state.pt"
        gen = torch.Generator().manual_seed(FIXTURE_SEED + 7)
        raw = _laplace_payload(select_params(model, cfg["laplace"]["selection_mode"]), gen)
        state = LaplaceState(**raw, sampler_version="v1_legacy")
        if state_version == "v2":
            state = scale_laplace_state(state, n_data_seqs=S2_N_DATA_SEQS, tokens_per_seq=T,
                                        prior_prec=S2_PRIOR_PREC, base_checkpoint=info["path"],
                                        base_sha256=info["sha256"])
        save_laplace_state(state, state_path)
    record = {"method": cfg["method"], "cell": cfg["cell"], "base": info,
              "sampler_version": state_version, "state_path": str(state_path),
              "state_sha256": _sha_file(state_path), "failures": [], "exit_code": 0}
    if cfg["method"] == "laplace":
        record["match_status"] = "matched"
    if edit_record is not None:
        edit_record(record)
    record_path = out_dir / "fit_record.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    yaml_path = _write_yaml(cfg, root / f"refit_{score_set}.yaml")
    return SimpleNamespace(cfg=cfg, yaml=yaml_path, state_path=state_path,
                           record_path=record_path, base=Path(cfg["base"]["base_checkpoint"]))


@pytest.fixture(scope="module")
def s2cells(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("s2_cells")
    return {s: _write_s2_cell(world, s, root) for s in S2_SCORE_SETS}


def _s2_variant(world, tmp_path: Path) -> tuple[dict, Path]:
    """The fixture eval YAML plus the S2 labels and listings of configs/i2_eval.yaml."""
    real = yaml.safe_load((REPO_ROOT / "configs/i2_eval.yaml").read_text(encoding="utf-8"))
    real = real["scoring"]

    def mutate(cfg):
        sc = cfg["scoring"]
        for s in S2_SCORE_SETS:
            sc["labels"][s] = copy.deepcopy(real["labels"][s])
            sc["n_samples"][s] = N_SAMPLES
        for eval_set in ("test", "test_hn"):
            sc["score_sets"][eval_set] += [s for s in real["score_sets"][eval_set]
                                           if s in S2_SCORE_SETS]

    cfg, yaml_path = _variant(world, tmp_path, mutate)
    _write_prereg(cfg)
    return cfg, yaml_path


def _reference_logp(cell, x: torch.Tensor, y: torch.Tensor, seed_b: int, n: int,
                    sampler_version: str = "v2") -> torch.Tensor:
    """[N, T] realized-token log-probs with the seed rule seed_b * N + s, by hand."""
    import dataclasses
    import warnings

    from minigpt import posthoc_refit
    from minigpt.laplace import load_laplace_state, sample_laplace_params

    model, _ = posthoc_refit.load_posthoc_base(cell.cfg)
    if cell.cfg["method"] == "tfb":
        state, sample = load_tfb_state(cell.state_path), sample_tfb_params
    else:
        state, sample = load_laplace_state(cell.state_path), sample_laplace_params
    if sampler_version != state.sampler_version:
        state = dataclasses.replace(state, sampler_version=sampler_version)
    rows = []
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for s in range(n):
            with apply_sampled_params(model, sample(state, seed=seed_b * n + s)):
                logits, _ = model(x[None])
            lp = torch.log_softmax(logits.float(), dim=-1)[0]
            rows.append(lp.gather(-1, y[:, None]).squeeze(-1))
    return torch.stack(rows)


def test_s2_score_sets_and_v2_samplers_are_registered(scripts):
    from minigpt.laplace import sample_laplace_params

    ecc = scripts.ecc
    assert set(S2_SCORE_SETS) <= set(ecc.SCORE_SET_REGISTRY)
    assert ecc.PARAM_SAMPLERS["tfb"] is sample_tfb_params  # the legacy entries are unchanged
    assert ecc.PARAM_SAMPLERS["laplace"] is sample_laplace_params
    assert {"tfb_v2", "laplace_v2"} <= set(ecc.PARAM_SAMPLERS)
    assert {s: p.as_posix() for s, p in ecc.POSTHOC_REFIT_CONFIGS.items()} == S2_REAL_REFIT_YAMLS
    expected = {
        "c2_refit": ["data/checkpoints/c0/ckpt_best.pt",
                     "data/checkpoints/i2_posthoc/c2/laplace_state.pt"],
        "c4_lap_refit": ["data/checkpoints/c3/ckpt_best.pt",
                         "data/checkpoints/i2_posthoc/c4_lap/laplace_state.pt"],
        "c4_tfb_fixed": ["data/checkpoints/c3/ckpt_best.pt",
                         "data/checkpoints/i2_posthoc/c4_tfb/tfb_state.pt"],
    }
    for name, paths in expected.items():
        got = ecc.SCORE_SET_REGISTRY[name].checkpoint_paths()
        assert [p.as_posix() for p in got] == paths, name


def test_v2_param_samplers_refuse_a_v1_legacy_state(scripts):
    from minigpt.laplace import LaplaceState, sample_laplace_params, scale_laplace_state
    from minigpt.tfb import TFBState

    ecc = scripts.ecc
    gen = torch.Generator().manual_seed(0)
    b, a = torch.randn(6, 2, generator=gen), torch.randn(2, 5, generator=gen)
    u, s, vh = torch.linalg.svd(b, full_matrices=False)
    kw = {"sigma_q": 0.1, "svd_cache": {"l": (u, s, vh)}, "a_map": {"l.lora_A": a},
          "param_names": ["l.lora_A"], "epsilon": None, "anchor_loss": 1.0}
    tfb_v2 = TFBState(**kw, sampler_version="v2")
    assert torch.equal(ecc.PARAM_SAMPLERS["tfb_v2"](tfb_v2, seed=3)["l.lora_A"],
                       sample_tfb_params(tfb_v2, seed=3)["l.lora_A"])
    with pytest.raises(ValueError, match="v2"):
        ecc.PARAM_SAMPLERS["tfb_v2"](TFBState(**kw, sampler_version="v1_legacy"), seed=3)

    lap_v1 = LaplaceState(param_names=["w"], phi_hat={"w": torch.zeros(4)},
                          curvature={"w": torch.full((4,), 1e-3)}, damping=1.0,
                          sampler_version="v1_legacy")
    lap_v2 = scale_laplace_state(lap_v1, n_data_seqs=10, tokens_per_seq=4, prior_prec=1.0)
    assert torch.equal(ecc.PARAM_SAMPLERS["laplace_v2"](lap_v2, seed=3)["w"],
                       sample_laplace_params(lap_v2, seed=3)["w"])
    with pytest.raises(ValueError, match="v2"):
        ecc.PARAM_SAMPLERS["laplace_v2"](lap_v1, seed=3)


def test_s2_loader_returns_the_base_the_v2_state_and_the_v2_sampler(world, scripts, s2cells,
                                                                   monkeypatch):
    from minigpt import posthoc_refit

    ecc = scripts.ecc
    monkeypatch.setattr(ecc, "POSTHOC_REFIT_CONFIGS", {s: c.yaml for s, c in s2cells.items()})
    for name, cell in s2cells.items():
        model, method, state = ecc.SCORE_SET_REGISTRY[name].load(torch.device("cpu"))
        assert method == f"{cell.cfg['method']}_v2"
        assert state.sampler_version == "v2"
        assert not model.training
        ref, _ = posthoc_refit.load_posthoc_base(cell.cfg)
        ref_sd = ref.state_dict()
        assert all(torch.equal(v, ref_sd[k]) for k, v in model.state_dict().items())
        assert [p.as_posix() for p in ecc.SCORE_SET_REGISTRY[name].checkpoint_paths()] == [
            cell.base.as_posix(), cell.state_path.as_posix()]


def _set_status(record: dict, status: str) -> None:
    record["match_status"] = status


S2_BAD_INPUTS = {
    # case: (score set, write options)
    "v1_legacy_state": ("c4_tfb_fixed", {"state_version": "v1_legacy"}),
    "v1_legacy_laplace": ("c2_refit", {"state_version": "v1_legacy"}),
    "base_sha": ("c4_tfb_fixed", {"edit_record": lambda r: r["base"].update(sha256="0" * 64)}),
    "state_sha": ("c2_refit", {"edit_record": lambda r: r.update(state_sha256="0" * 64)}),
    "exit_code": ("c4_tfb_fixed", {"edit_record": lambda r: r.update(exit_code=2)}),
    "failures": ("c4_lap_refit", {"edit_record": lambda r: r.update(failures=["t2c"])}),
    "not_matched": ("c4_lap_refit", {"edit_record": lambda r: _set_status(r, "not_matched")}),
    "cell": ("c4_tfb_fixed", {"edit_cfg": lambda c: c.update(cell="c4_tfb_other")}),
    "method": ("c2_refit", {}),  # relabelled as tfb in the test body
}


@pytest.mark.parametrize("case", sorted(S2_BAD_INPUTS) + ["no_record"])
def test_s2_loader_refuses_a_state_that_did_not_pass_its_fit(world, scripts, tmp_path,
                                                            monkeypatch, case):
    ecc = scripts.ecc
    name, options = S2_BAD_INPUTS.get(case, ("c4_lap_refit", {}))
    if case == "method":
        # A laplace cell relabelled as tfb: the YAML then needs a tfb section to be valid.
        real_tfb = yaml.safe_load((REPO_ROOT / S2_REAL_REFIT_YAMLS["c4_tfb_fixed"]).read_text(
            encoding="utf-8"))["tfb"]
        cell = _write_s2_cell(world, name, tmp_path)
        cell.cfg.update(method="tfb", tfb=real_tfb)
        _write_yaml(cell.cfg, cell.yaml)
    else:
        cell = _write_s2_cell(world, name, tmp_path, **options)
    if case == "no_record":
        cell.record_path.unlink()
    monkeypatch.setitem(ecc.POSTHOC_REFIT_CONFIGS, name, cell.yaml)
    with pytest.raises(ValueError):
        ecc.SCORE_SET_REGISTRY[name].load(torch.device("cpu"))


def test_s2_missing_state_fails_before_any_score_set_is_scored(world, scripts, tmp_path,
                                                              monkeypatch):
    ecc = scripts.ecc
    cell = _write_s2_cell(world, "c2_refit", tmp_path / "cells")
    cell.state_path.unlink()
    monkeypatch.setitem(ecc.POSTHOC_REFIT_CONFIGS, "c2_refit", cell.yaml)
    cfg, yaml_path = _s2_variant(world, tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run_main(ecc.main, _argv(yaml_path, "test", "main", "--score-set", "c0,c2_refit",
                                  "--block-ids", "0"))
    assert exc.value.code not in (0, None)
    assert not list(Path(cfg["scoring"]["out_dir"]).glob("*.pt"))  # c0 was not scored either


def test_s2_score_sets_score_with_the_v2_samplers(world, scripts, s2cells, tmp_path,
                                                 monkeypatch):
    """Both scripts know the S2 names; eval_c_checkpoints.py scores them with v2 draws."""
    from minigpt import posthoc_refit

    ecc, mcd = scripts.ecc, scripts.mcd
    monkeypatch.setattr(ecc, "POSTHOC_REFIT_CONFIGS", {s: c.yaml for s, c in s2cells.items()})
    cfg, yaml_path = _s2_variant(world, tmp_path)
    out = Path(cfg["scoring"]["out_dir"])
    block_ids = [0, 1, 20, 21]
    ids = "0-1,20-21"
    _run_main(ecc.main, _argv(yaml_path, "test", "main", "--score-set", ",".join(S2_SCORE_SETS),
                              "--block-ids", ids))
    _run_main(ecc.main, _argv(yaml_path, "test_hn", "main", "--block-ids", "0-1",
                              "--score-set", "c4_lap_refit,c4_tfb_fixed"))
    text = _run_main(mcd.main, _argv(yaml_path, "test", "mcd", "--block-ids", ids))
    for s in S2_SCORE_SETS:
        assert f"Skipping {s}: scored by scripts/eval_c_checkpoints.py" in text
    assert sorted(p.name for p in out.glob("*__mcd.pt")) == ["mc_dropout__test__mcd.pt"]

    tokens = world.eval_info["files"]["test"]["tokens"]
    files = []
    for eval_set, names in (("test", S2_SCORE_SETS), ("test_hn", S2_SCORE_SETS[1:])):
        for name in names:
            path = out / f"{name}__{eval_set}__main.pt"
            files.append(path)
            payload = _load_scores(path)
            meta = payload["meta"]
            cell = s2cells[name]
            assert meta["score_set"] == name
            assert meta["sampler"] == "v2"
            assert meta["display_label"] == cfg["scoring"]["labels"][name]["display"]
            assert meta["checkpoint_sha256"] == {
                cell.base.as_posix(): _sha_file(cell.base),
                cell.state_path.as_posix(): _sha_file(cell.state_path),
            }
            assert float(_gap(payload["logp_real"]).min()) >= -1e-6
            if eval_set != "test":
                continue
            assert payload["block_index"].tolist() == block_ids
            for row in (0, 2):
                b = block_ids[row]
                seed_b = SEED_OFFSETS["test"] + b
                x, y = tokens[b, :-1].long(), tokens[b, 1:].long()
                ref = _reference_logp(cell, x, y, seed_b, N_SAMPLES)
                assert torch.allclose(payload["logp_real"][row], ref, atol=1e-6), (name, b)
                if name == "c4_tfb_fixed":  # the legacy draw of the same state is far off
                    legacy = _reference_logp(cell, x, y, seed_b, N_SAMPLES, "v1_legacy")
                    assert float((payload["logp_real"][row] - legacy).abs().max()) > 1e-3
    assert posthoc_refit.check_scores(files, [c.record_path for c in s2cells.values()]) == []


# ---------------------------------------------------------------------------
# Block ids and the block-file reader
# ---------------------------------------------------------------------------

def test_parse_block_ids(scripts):
    parse = scripts.ecc.parse_block_ids
    assert parse("0-2,5,7-8", 10) == [0, 1, 2, 5, 7, 8]
    assert parse("0-99,500-599", 1000) == list(range(100)) + list(range(500, 600))
    for bad in ("5-2", "0-10", "-1", "a", "", "3,3"):
        with pytest.raises(ValueError):
            parse(bad, 10)


def _reader_cfg(world, eval_dir: Path) -> dict:
    cfg = copy.deepcopy(world.cfg)
    cfg["eval_set"]["out_dir"] = eval_dir.as_posix()
    return cfg


def test_reader_rejects_val_docs_bad_weights_and_bad_width(world, scripts, tmp_path):
    ecc = scripts.ecc
    good = world.eval_info["files"]["test"]
    cases = {}
    val = copy.deepcopy(good)
    val["doc_id"][0] = world.eval_info["val_docs"][0]["doc_id"]
    cases["val_doc"] = val
    weight = copy.deepcopy(good)
    weight["weight"][0] = 0.25
    cases["weight"] = weight
    width = copy.deepcopy(good)
    width["tokens"] = width["tokens"][:, :-1]
    cases["width"] = width
    for name, payload in cases.items():
        eval_dir = tmp_path / name
        eval_dir.mkdir()
        shutil.copy(world.eval_dir / "manifest.jsonl", eval_dir / "manifest.jsonl")
        torch.save(payload, eval_dir / "blocks_test.pt")
        with pytest.raises(ValueError):
            ecc.load_i2_blocks(_reader_cfg(world, eval_dir), "test")
    blocks = ecc.load_i2_blocks(_reader_cfg(world, world.eval_dir), "test")
    assert blocks.manifest_sha256 == _sha_file(world.eval_dir / "manifest.jsonl")
    assert len(blocks.doc_id) == len(good["doc_id"])


# ---------------------------------------------------------------------------
# D1 path (no --eval-set): wrappers over the shared scorer; ratios from the scores
# ---------------------------------------------------------------------------

D1_KEYS = ("mi", "predictive_entropy", "max_prob", "correct", "p_true", "sum_p_sq")


def test_d1_wrappers_route_through_the_shared_scorer(world, scripts):
    ecc, mcd = scripts.ecc, scripts.mcd
    device = torch.device("cpu")
    tokens = world.eval_info["files"]["test"]["tokens"][0].long()
    x, y = tokens[:-1], tokens[1:]
    model, method, state = ecc.load_model("c4_tfb", device)
    old = ecc.score_sequence_full(model, x, y, N_SAMPLES, device, method, state, seq_idx=4)
    new = ecc.score_block_batch(model, x[None], y[None], N_SAMPLES, device, method, state,
                                seed_b=4, amp=False)
    assert set(old) == set(D1_KEYS)
    assert torch.equal(old["mi"], new["tok_mi"][0])
    assert torch.equal(old["p_true"], new["tok_pbar_true"][0])
    c0 = mcd.load_c0_model(device)
    torch.manual_seed(5)
    old = mcd.score_sequence_mc_dropout(c0, x, y, N_SAMPLES, device)
    torch.manual_seed(5)
    new = ecc.score_block_batch(c0, x[None], y[None], N_SAMPLES, device, "dropout", None,
                                seed_b=None, amp=False)
    assert set(old) == set(D1_KEYS)
    assert old["mi"].shape == (T,) and old["correct"].dtype == torch.float32
    assert torch.equal(old["mi"], new["tok_mi"][0])
    assert not c0.training  # enable_dropout restores eval mode


def test_d1_from_scores_ratio_is_computed_from_the_file(scripts, tmp_path):
    """Old-format files: the ratio is mean MI(OOD) / mean MI(ID), none hardcoded (T1e)."""
    gen = torch.Generator().manual_seed(3)
    labels = torch.cat([torch.zeros(8), torch.ones(8)])
    mi = torch.rand(16, generator=gen) * 0.1 + labels * 0.05
    payload = {
        "c1": {"mi": mi, "pred_ent": torch.rand(16, generator=gen),
               "max_prob_unc": torch.rand(16, generator=gen), "labels": labels},
        "c0": {"mi": torch.zeros(16), "pred_ent": torch.rand(16, generator=gen),
               "max_prob_unc": torch.rand(16, generator=gen), "labels": labels},
    }
    path = tmp_path / "old_scores.pt"
    torch.save(payload, path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = scripts.ecc.load_scores(path)
    expected = float(mi[8:].double().mean() / mi[:8].double().mean())
    assert abs(results["c1"]["mi_ratio"] - expected) <= 1e-12
    assert results["c0"]["mi_ratio"] is None


# ---------------------------------------------------------------------------
# --analyze (section 5.6) and S1-T5b (re-derivation by scripts/check_eval_rebuild.py)
# ---------------------------------------------------------------------------

def _analysis_json(world, name: str) -> dict:
    return json.loads((world.scores / name).read_text(encoding="utf-8"))


def _triples(table: dict) -> set[tuple[str, str, str]]:
    return {(c["score_set"], c["id_domain"], c["ood_domain"]) for c in table["cells"]}


def _holm_reference(p: list[float]) -> list[float]:
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    out, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def test_analyze_writes_the_planned_cells(world):
    ood = ["arxiv", "freelaw"]
    table = _analysis_json(world, "table_test.json")
    assert _triples(table) == {(s, "stackexchange", o) for s in SCORE_SETS_ALL for o in ood}
    table_hn = _analysis_json(world, "table_test_hn.json")
    assert _triples(table_hn) == {(s, "hackernews", o) for s in ("c3", "c4_tfb") for o in ood}
    table_st = _analysis_json(world, "table_arxiv_stripped.json")
    assert _triples(table_st) == (
        {(s, "stackexchange", "arxiv_stripped") for s in SCORE_SETS_ALL}
        | {(s, "hackernews", "arxiv_stripped") for s in ("c3", "c4_tfb")}
    )
    index = {"stackexchange": 0, "hackernews": 1, "arxiv": 2, "freelaw": 3, "arxiv_stripped": 4}
    scores = ["blk_g", "blk_mi", "blk_tu", "blk_au", "blk_nll", "blk_maxprob_unc"]
    for t in (table, table_hn, table_st):
        assert t["analysis_sha256"] == _analysis_sha(world.cfg)
        for c in t["cells"]:
            hn = c["id_domain"] == "hackernews"
            assert c["n_id_docs"] == (len(HN_BLOCKS_PER_DOC) if hn else DOCS_PER_DOMAIN)
            assert c["n_ood_docs"] == DOCS_PER_DOMAIN
            assert c["n_id_blocks"] == sum(HN_BLOCKS_PER_DOC if hn else BLOCKS_PER_DOC)
            assert c["n_ood_blocks"] == sum(BLOCKS_PER_DOC)
            assert c["rng_seed"] == [0, index[c["id_domain"]], index[c["ood_domain"]]]
            assert sorted(c["scores"]) == sorted(scores)
            for q in c["scores"].values():
                assert 0.0 <= q["auroc_w_ci"][0] <= q["auroc_w_ci"][1] <= 1.0
                assert 0.0 <= q["fpr95_w_ci"][0] <= q["fpr95_w_ci"][1] <= 1.0


@pytest.mark.parametrize("score_set,id_domain,id_set,ood_domain,ood_set,table", [
    ("c1", "stackexchange", "test", "arxiv", "test", "table_test.json"),
    ("c3", "hackernews", "test_hn", "freelaw", "test", "table_test_hn.json"),
    ("mc_dropout", "stackexchange", "test", "arxiv_stripped", "arxiv_stripped",
     "table_arxiv_stripped.json"),
])
def test_analyze_point_estimates_match_weighted_sklearn(world, score_set, id_domain, id_set,
                                                        ood_domain, ood_set, table):
    from sklearn.metrics import roc_auc_score, roc_curve

    id_p = _load_scores(_path(world, score_set, id_set, "main"))
    ood_p = _load_scores(_path(world, score_set, ood_set, "main"))
    ood_key = "arxiv" if ood_domain == "arxiv_stripped" else ood_domain
    id_rows = [i for i, d in enumerate(id_p["domain"]) if d == id_domain]
    ood_rows = [i for i, d in enumerate(ood_p["domain"]) if d == ood_key]
    y = np.array([0] * len(id_rows) + [1] * len(ood_rows))
    docs = [id_p["doc_id"][i] for i in id_rows] + [ood_p["doc_id"][i] for i in ood_rows]
    counts = {c: {} for c in (0, 1)}
    for label, d in zip(y, docs):
        counts[label][d] = counts[label].get(d, 0) + 1
    w = np.array([1.0 / counts[label][d] for label, d in zip(y, docs)])
    cell = next(c for c in _analysis_json(world, table)["cells"]
                if (c["score_set"], c["id_domain"], c["ood_domain"])
                == (score_set, id_domain, ood_domain))
    for score in ("blk_g", "blk_mi", "blk_maxprob_unc"):
        s = np.r_[id_p[score].double().numpy()[id_rows], ood_p[score].double().numpy()[ood_rows]]
        fpr, tpr, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
        assert abs(cell["scores"][score]["auroc_w"] - roc_auc_score(y, s, sample_weight=w)) \
            <= 1e-12
        assert abs(cell["scores"][score]["fpr95_w"] - fpr[np.argmax(tpr >= 0.95)]) <= 1e-12


def test_analyze_families_delta_p_and_holm(world):
    fams = _analysis_json(world, "families.json")["families"]
    assert set(fams) == set(world.cfg["analysis"]["families"])
    table = {(c["score_set"], c["id_domain"], c["ood_domain"]): c
             for name in ("table_test.json", "table_test_hn.json")
             for c in _analysis_json(world, name)["cells"]}
    resamples = world.cfg["analysis"]["bootstrap"]["resamples"]
    for name, m in (("F1_s1_primary", 6), ("F2_s1_lora_on_se", 2)):
        fam = fams[name]
        assert fam["status"] == "scored" and fam["m"] == m and len(fam["cells"]) == m
        assert fam["missing_cells"] == []
        p = [c["p"] for c in fam["cells"]]
        for c, p_ref in zip(fam["cells"], _holm_reference(p)):
            assert abs(c["p_holm"] - p_ref) <= 1e-12
            assert abs(c["delta"] - (c["auroc_w_primary"] - c["auroc_w_contrast"])) <= 1e-12
            t = table[(c["score_set"], c["id_domain"], c["ood_domain"])]
            assert abs(c["auroc_w_primary"] - t["scores"]["blk_g"]["auroc_w"]) <= 1e-12
            assert abs(c["auroc_w_contrast"] - t["scores"]["blk_mi"]["auroc_w"]) <= 1e-12
            assert 2.0 / (resamples + 1) - 1e-12 <= c["p"] <= 1.0
            assert c["delta_ci"][0] <= c["delta_ci"][1]
            assert c["decision"] in ("primary_beats_contrast", "contrast_beats_primary",
                                     "no_difference")
    assert "F1_s1_primary: scored (6/6 cells)" in world.analyze_stdout


def test_analyze_refuses_without_matching_prereg(world, scripts, tmp_path):
    cfg, yaml_path = _variant(world, tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run_main(scripts.ecc.main, ["--eval-config", str(yaml_path), "--analyze"])
    assert exc.value.code not in (0, None)
    assert not list(Path(cfg["scoring"]["out_dir"]).glob("*.json"))


def test_t5b_check_eval_rebuild_rederives_the_tables(world):
    checker = _load_script("check_eval_rebuild")
    report = checker.run_rederive(checker.load_config(world.yaml_path), world.scores,
                                  world.yaml_path, log=lambda *args: None)
    assert report["status"] == "PASS", report["failures"][:5]
    assert report["n_values_compared"] > 0
    assert sorted(report["tables_compared"]) == sorted([
        "families.json", "table_arxiv_stripped.json", "table_test.json", "table_test_hn.json",
    ])


def test_t5b_checker_imports_no_scorer_code():
    import ast

    tree = ast.parse((SCRIPTS_DIR / "check_eval_rebuild.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    forbidden = ("minigpt", "eval_c_checkpoints", "eval_mc_dropout", "scripts")
    assert not [n for n in names for f in forbidden if n == f or n.startswith(f + ".")]


# ---------------------------------------------------------------------------
# S1-T5h: runtime budget (keep last in this module)
# ---------------------------------------------------------------------------

def test_t5h_module_runs_within_budget(world):
    elapsed = time.perf_counter() - _MODULE_T0
    assert elapsed <= TIME_BUDGET_S, f"{elapsed:.1f}s (scoring {world.scoring_seconds:.1f}s)"
