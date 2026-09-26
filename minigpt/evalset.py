"""Iteration-2 eval set: document manifest, blocks, unseen check and pre-registration freeze.

Spec: specs/i2-eval-rebuild.md, sections 2 and 5.2-5.5. Config: configs/i2_eval.yaml.

Outputs under `eval_set.out_dir` (default data/eval_i2/):

- `manifest.jsonl`: one JSON row per document. Keys: `doc_id`
  (`{key}/{i:09d}/{sha1(text)[:12]}`), `domain`, `split` (test or val), `variant`
  (raw or stripped), `eval_set` (test rows only), `stream_index` (i, 0-based in the shuffled
  stream), `c_i` (the cumulative token start of the raw document), `L_d` (the length of this
  variant), `token_sha1` (sha1 of the int32 little-endian tokens), `text_sha1` (sha1 of the
  raw UTF-8 text), `o_d` and `k_d`. Val rows carry the range only: `o_d` and `k_d` are null.
- `docs_{key}.pt` and `docs_{key}_stripped.pt`: {"doc_id": [...], "tokens": [int32 tensors]}
  for the test documents, in manifest order.
- `blocks_{eval_set}.pt` for `test`, `test_hn` and `{key}_stripped`: `tokens` [B, T+1] int32
  (x = [:, :-1], y = [:, 1:]), `block_index` [B] int64, `doc_id`, `domain`, `variant` (lists),
  `offset` [B] int64 (the block's start inside its document, o_d + j T), `block_in_doc` [B]
  int64 (j), `weight` [B] float32 (1 / k_d) and `meta`.
- `build_report.json`: the S1-T2a/T2c report per domain and the build status.

Eval-set names come from the domain roles: `id_adapter` domains go to `test_hn`, the other
domains to `test`, and a domain with `stripped_copy: true` also gets `{key}_stripped`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from minigpt.data import PILE_DATASET_PATH, PILE_DOMAIN_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLES = ("id_base", "id_adapter", "ood")
ROLE_EVAL_SET = {"id_base": "test", "id_adapter": "test_hn", "ood": "test"}
PROGRESS_EVERY_TOKENS = 5_000_000
_SCAN_BATCH = 262_144


class PreregError(RuntimeError):
    """The pre-registration file is missing or its hash differs from the YAML."""


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------

def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root or REPO_ROOT) / p


def load_eval_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_configs(cfg: dict, root: Path | None = None) -> dict[str, dict]:
    paths = cfg["eval_set"]["train_configs"]
    return {name: load_eval_config(resolve_path(paths[name], root)) for name in ("base", "adapter")}


def _role_config(role: str, train_cfgs: dict[str, dict]) -> dict:
    return train_cfgs["adapter"] if role == "id_adapter" else train_cfgs["base"]


def validate_eval_config(cfg: dict, root: Path | None = None) -> None:
    """Check the `eval_set` block against itself and against the two training configs."""
    es = cfg["eval_set"]
    if es["min_doc_tokens"] != es["block_size"] + 1:
        raise ValueError("eval_set.min_doc_tokens must equal block_size + 1")
    if es["splits"] != ["test", "val"]:
        raise ValueError(f"eval_set.splits must be [test, val], got {es['splits']}")
    if es["source"]["dataset"] != PILE_DATASET_PATH:
        raise ValueError(f"eval_set.source.dataset must be {PILE_DATASET_PATH}")
    if es["max_blocks_per_doc"] < 1 or es["n_test_docs"] < 1:
        raise ValueError("max_blocks_per_doc and n_test_docs must be >= 1")
    if es["unseen_check"]["window_tokens"] > es["min_doc_tokens"]:
        raise ValueError("unseen_check.window_tokens must not exceed min_doc_tokens")
    train_cfgs = load_train_configs(cfg, root)
    for name, tcfg in train_cfgs.items():
        if tcfg["train"]["seed"] != es["source"]["shuffle_seed"]:
            raise ValueError(f"shuffle_seed differs from the {name} training config seed")
    keys = [d["key"] for d in es["domains"]]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate domain keys")
    roles = [d["role"] for d in es["domains"]]
    for role in ("id_base", "id_adapter"):
        if roles.count(role) > 1:
            raise ValueError(f"at most one domain may have role {role}")
    for d in es["domains"]:
        key, role = d["key"], d["role"]
        if key not in PILE_DOMAIN_NAMES:
            raise ValueError(f"unknown Pile domain {key!r}")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r} for {key}")
        if not isinstance(d["stripped_copy"], bool):
            raise ValueError(f"stripped_copy must be true or false for {key}")
        if f"{key}_{d['cache_tokens']}" not in es["seen_tensors"]:
            raise ValueError(f"the cache of {key} is not listed in seen_tensors")
        if d["max_stream_tokens"] <= d["cache_tokens"]:
            raise ValueError(f"max_stream_tokens must exceed cache_tokens for {key}")
        data = _role_config(role, train_cfgs)["data"]
        if role == "ood":
            ok = key in data["pile_ood_domains"] and data["pile_ood_tokens"] == d["cache_tokens"]
        else:
            ok = key in data["pile_id_domains"] and data["pile_id_tokens"] == d["cache_tokens"]
        if not ok:
            raise ValueError(f"{key}: cache_tokens or role disagrees with the training config")


def split_ranges(cfg: dict, root: Path | None = None) -> dict[str, dict[str, tuple[int, int]]]:
    """Domain-local train, val and old-test token ranges of the ID domains (S1-T3g).

    Same expressions as minigpt/data.py:198-213: the ID stream is the concatenation of the
    `pile_id_domains` caches, each `pile_id_tokens` long.
    """
    train_cfgs = load_train_configs(cfg, root)
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for d in cfg["eval_set"]["domains"]:
        if d["role"] == "ood":
            continue
        data = _role_config(d["role"], train_cfgs)["data"]
        n = data["pile_id_tokens"]
        total = n * len(data["pile_id_domains"])
        val_fraction, test_fraction = data["val_fraction"], data["test_fraction"]
        train_end = int(total * (1 - val_fraction - test_fraction))
        val_end = int(total * (1 - test_fraction))
        off = data["pile_id_domains"].index(d["key"]) * n

        def clip(x: int, off: int = off, n: int = n) -> int:
            return min(max(x - off, 0), n)

        out[d["key"]] = {
            "train": (clip(0), clip(train_end)),
            "val": (clip(train_end), clip(val_end)),
            "old_test": (clip(val_end), clip(total)),
        }
    return out


def eval_set_domains(cfg: dict) -> dict[str, list[tuple[str, str]]]:
    """Eval-set name -> [(domain key, variant)] in YAML domain order."""
    out: dict[str, list[tuple[str, str]]] = {}
    for d in cfg["eval_set"]["domains"]:
        out.setdefault(ROLE_EVAL_SET[d["role"]], []).append((d["key"], "raw"))
    for d in cfg["eval_set"]["domains"]:
        if d["stripped_copy"]:
            out[f"{d['key']}_stripped"] = [(d["key"], "stripped")]
    return out


def out_dir(cfg: dict, root: Path | None = None) -> Path:
    return resolve_path(cfg["eval_set"]["out_dir"], root)


def manifest_path(cfg: dict, root: Path | None = None) -> Path:
    return out_dir(cfg, root) / "manifest.jsonl"


def block_file_path(cfg: dict, eval_set: str, root: Path | None = None) -> Path:
    return out_dir(cfg, root) / f"blocks_{eval_set}.pt"


def doc_file_path(cfg: dict, key: str, variant: str, root: Path | None = None) -> Path:
    suffix = "" if variant == "raw" else "_stripped"
    return out_dir(cfg, root) / f"docs_{key}{suffix}.pt"


def load_block_file(cfg: dict, eval_set: str, root: Path | None = None) -> dict:
    return torch.load(block_file_path(cfg, eval_set, root), weights_only=True)


def load_manifest(cfg: dict, root: Path | None = None) -> list[dict]:
    text = manifest_path(cfg, root).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------
# Pre-registration freeze (section 5.5)
# --------------------------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def analysis_sha256(cfg: dict) -> str:
    blob = json.dumps(cfg["analysis"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prereg_path(cfg: dict, root: Path | None = None) -> Path:
    return resolve_path(cfg["scoring"]["out_dir"], root) / "prereg.json"


def freeze_prereg(cfg: dict, path: str | Path) -> dict:
    """Write `analysis_sha256` and `frozen_at` to `path`. Never overwrites an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; the pre-registration is frozen")
    record = {
        "analysis_sha256": analysis_sha256(cfg),
        "frozen_at": utc_now_iso(),
        "eval_set_name": cfg["eval_set"]["name"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return record


def verify_prereg(cfg: dict, path: str | Path) -> str:
    """Return the frozen hash; raise PreregError if the file is missing or the hash differs."""
    path = Path(path)
    if not path.exists():
        raise PreregError(f"{path} is missing: run the freeze before scoring test blocks")
    record = json.loads(path.read_text(encoding="utf-8"))
    current = analysis_sha256(cfg)
    if record["analysis_sha256"] != current:
        raise PreregError(
            f"analysis block hash {current} differs from the frozen {record['analysis_sha256']}")
    return current


# --------------------------------------------------------------------------------------------
# Documents, blocks and the stripped copy (section 5.4)
# --------------------------------------------------------------------------------------------

def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_doc_id(key: str, index: int, sha1_hex: str) -> str:
    return f"{key}/{index:09d}/{sha1_hex[:12]}"


def token_sha1(tokens: np.ndarray | torch.Tensor) -> str:
    arr = tokens.numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
    return hashlib.sha1(arr.astype("<i4").tobytes()).hexdigest()


def n_blocks_for(length: int, block_size: int, max_blocks: int) -> int:
    """k_d = min(k_max, floor((L_d - 1) / T))."""
    return min(max_blocks, (length - 1) // block_size)


def draw_offset(length: int, k: int, block_size: int, offset_seed: int, sha1_12: str,
                stripped: bool) -> int:
    """o_d ~ U{0, ..., L_d - 1 - k_d T}, from a per-document generator."""
    seq = [offset_seed, int(sha1_12, 16)] + ([1] if stripped else [])
    return int(np.random.default_rng(seq).integers(0, length - k * block_size))


_CMD_RE = re.compile(r"\\[A-Za-z@]+\*?")


def strip_latex(text: str, math_envs: list[str]) -> str:
    """Remove LaTeX math and markup (section 5.4). Escaped dollars are removed first."""
    out = text.replace("\\$", "")
    for env in math_envs:
        e = re.escape(env)
        out = re.sub(rf"\\begin\{{{e}\}}.*?\\end\{{{e}\}}", " ", out, flags=re.S)
    out = re.sub(r"\$\$.*?\$\$", " ", out, flags=re.S)
    out = re.sub(r"\\\[.*?\\\]", " ", out, flags=re.S)
    out = re.sub(r"\\\(.*?\\\)", " ", out, flags=re.S)
    out = re.sub(r"\$[^$]*\$", " ", out)
    out = _CMD_RE.sub(" ", out)
    out = re.sub(r"[{}\\$]", "", out)
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------------------------
# Unseen check (section 5.3)
# --------------------------------------------------------------------------------------------

def window_hashes(tokens: np.ndarray, window: int, base: int) -> np.ndarray:
    """h_i = sum_j t_{i+j} P^{window-1-j} mod 2^64, by Horner's rule in uint64."""
    t = np.asarray(tokens).astype(np.uint64)
    n = len(t) - window + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    b = np.uint64(base)
    for j in range(window):
        h *= b
        h += t[j:j + n]
    return h


class UnseenIndex:
    """Query windows (positions in `flat`) looked up in seen tensors by hash.

    Every hash hit is confirmed by an exact token comparison, so a collision costs time and
    never marks a query as seen. Queries with the same hash but different tokens are kept as
    separate contents inside one hash group.
    """

    def __init__(self, flat: np.ndarray, qpos: np.ndarray, window: int, base: int) -> None:
        self.window = window
        self.base = base
        self.flat = np.ascontiguousarray(np.asarray(flat, dtype=np.int64))
        self.qpos = np.asarray(qpos, dtype=np.int64)
        self._fviews = np.lib.stride_tricks.sliding_window_view(self.flat, window)
        qhash = window_hashes(self.flat, window, base)[self.qpos]
        rows = np.ascontiguousarray(self._fviews[self.qpos].astype(np.int32))
        void = rows.view(np.dtype((np.void, rows.dtype.itemsize * window))).ravel()
        _, first_q, q_content = np.unique(void, return_index=True, return_inverse=True)
        c_hash = qhash[first_q]
        order = np.argsort(c_hash, kind="stable")
        remap = np.empty_like(order)
        remap[order] = np.arange(len(order))
        self.q_content = remap[np.asarray(q_content).ravel()]
        self.c_pos = self.qpos[first_q][order]
        c_hash = c_hash[order]
        self.u_hash, self.g_start, self.g_count = np.unique(
            c_hash, return_index=True, return_counts=True)
        self.c_hit = np.zeros(len(self.c_pos), dtype=bool)
        self.c_first_tensor = np.full(len(self.c_pos), -1, dtype=np.int64)

    @property
    def query_hit(self) -> np.ndarray:
        return self.c_hit[self.q_content]

    @property
    def query_first_tensor(self) -> np.ndarray:
        return self.c_first_tensor[self.q_content]

    def scan(self, tokens: np.ndarray, chunk_tokens: int, tensor_id: int = 0) -> int:
        """Mark every query window that occurs in `tokens`. Returns confirmed hit positions."""
        tokens = np.asarray(tokens, dtype=np.int64)
        w = self.window
        n_groups = len(self.u_hash)
        if n_groups == 0:
            return 0
        max_count = int(self.g_count.max())
        n_hits = 0
        for start in range(0, max(len(tokens) - w + 1, 0), chunk_tokens):
            chunk = tokens[start:min(start + chunk_tokens + w - 1, len(tokens))]
            h = window_hashes(chunk, w, self.base)
            g = np.minimum(np.searchsorted(self.u_hash, h), n_groups - 1)
            pos = np.nonzero(self.u_hash[g] == h)[0]
            if pos.size == 0:
                continue
            gg = g[pos]
            views = np.lib.stride_tricks.sliding_window_view(chunk, w)
            for k in range(max_count):
                sel = self.g_count[gg] > k
                if not sel.any():
                    break
                p_all = pos[sel]
                c_all = self.g_start[gg[sel]] + k
                for s in range(0, len(p_all), _SCAN_BATCH):
                    p = p_all[s:s + _SCAN_BATCH]
                    c = c_all[s:s + _SCAN_BATCH]
                    eq = np.all(views[p] == self._fviews[self.c_pos[c]], axis=1)
                    hit = c[eq]
                    n_hits += int(eq.sum())
                    self.c_hit[hit] = True
                    fresh = hit[self.c_first_tensor[hit] < 0]
                    self.c_first_tensor[fresh] = tensor_id
        return n_hits


@dataclass
class Candidate:
    key: str
    index: int
    start: int
    doc_id: str
    text_sha1: str
    tokens: np.ndarray            # int32
    n_blocks: int
    offset: int
    stripped_tokens: np.ndarray | None = None
    stripped_n_blocks: int | None = None
    stripped_offset: int | None = None


def _load_seen(seen_dir: Path, name: str) -> np.ndarray:
    return torch.load(seen_dir / f"{name}.pt", weights_only=True).numpy()


def unseen_check(cands: list[Candidate], cfg: dict, root: Path | None = None,
                 log: Callable[..., None] = print) -> dict:
    """Prefix and block-window lookups of `cands` in every tensor of `seen_tensors`."""
    es = cfg["eval_set"]
    uc = es["unseen_check"]
    w, T = uc["window_tokens"], es["block_size"]
    n_win = T + 1 - w + 1
    segs: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    q_block: list[np.ndarray] = []
    prefix_q = np.zeros(len(cands), dtype=np.int64)
    block_doc: list[int] = []
    pos = 0
    n_q = 0
    for ci, cand in enumerate(cands):
        segs.append(cand.tokens[:w])
        qpos.append(np.array([pos]))
        q_block.append(np.array([-1]))
        prefix_q[ci] = n_q
        pos += w
        n_q += 1
        for j in range(cand.n_blocks):
            s = cand.offset + j * T
            segs.append(cand.tokens[s:s + T + 1])
            qpos.append(pos + np.arange(n_win))
            q_block.append(np.full(n_win, len(block_doc)))
            block_doc.append(ci)
            pos += T + 1
            n_q += n_win
    flat = np.concatenate(segs).astype(np.int64)
    index = UnseenIndex(flat, np.concatenate(qpos), w, uc["hash_base"])
    seen_dir = resolve_path(es["seen_dir"], root)
    per_tensor = {}
    for t_id, name in enumerate(es["seen_tensors"]):
        tokens = _load_seen(seen_dir, name)
        per_tensor[name] = {"tokens": int(len(tokens)),
                            "hit_positions": index.scan(tokens, uc["chunk_tokens"], t_id)}
        log(f"    scanned {name}: {per_tensor[name]['hit_positions']} confirmed hit positions")
    q_hit = index.query_hit
    qb = np.concatenate(q_block)
    in_block = qb >= 0
    block_hits = np.bincount(qb[in_block], weights=q_hit[in_block].astype(np.float64),
                             minlength=len(block_doc)).astype(np.int64)
    near_dup_block = block_hits >= uc["near_dup_window_frac"] * n_win
    block_doc_arr = np.asarray(block_doc, dtype=np.int64)
    near_dup = np.zeros(len(cands), dtype=bool)
    near_dup[block_doc_arr[near_dup_block]] = True
    doc_hits = np.bincount(block_doc_arr, weights=block_hits, minlength=len(cands))
    return {
        "prefix_hit": q_hit[prefix_q],
        "near_dup": near_dup,
        "window_hits": doc_hits.astype(np.int64),
        "windows": np.array([c.n_blocks * n_win for c in cands], dtype=np.int64),
        "per_tensor": per_tensor,
    }


# --------------------------------------------------------------------------------------------
# Streaming and the build (section 5.3)
# --------------------------------------------------------------------------------------------

def hf_text_stream(key: str, source: dict) -> Iterator[str]:
    """The Hugging Face stream, exactly as minigpt/data.py:156-161."""
    import datasets as hf_datasets  # lazy: only for the real build

    stream = hf_datasets.load_dataset(source["dataset"], name=PILE_DOMAIN_NAMES[key],
                                      split=source["split"], streaming=True)
    stream = stream.shuffle(seed=source["shuffle_seed"], buffer_size=source["shuffle_buffer"])
    for item in stream:
        yield item["text"]


def _iter_docs(texts: Iterable[str], tokenizer, max_stream_tokens: int
               ) -> Iterator[tuple[int, int, str, np.ndarray]]:
    c = 0
    for i, text in enumerate(texts):
        if c >= max_stream_tokens:
            return
        toks = np.asarray(tokenizer.encode_ordinary(text), dtype=np.int64)
        yield i, c, text, toks
        c += len(toks)


class _DomainWalk:
    """One pass over a domain's stream: match check and val rows before the cut, then
    eligible test candidates after it."""

    def __init__(self, dcfg: dict, cfg: dict, texts: Iterable[str], tokenizer,
                 val_range: tuple[int, int] | None, root: Path | None,
                 log: Callable[..., None]) -> None:
        self.d = dcfg
        self.cfg = cfg
        self.es = cfg["eval_set"]
        self.tokenizer = tokenizer
        self.log = log
        self.docs = _iter_docs(texts, tokenizer, dcfg["max_stream_tokens"])
        self.val_range = val_range
        self.root = root
        self.stats = Counter()
        self.lengths: list[int] = []
        self.stopped_by = "max_stream_tokens"
        self.tokens_read = 0
        self.docs_read = 0

    def walk_to_cut(self) -> dict:
        key, cache_tokens = self.d["key"], self.d["cache_tokens"]
        cache = _load_seen(resolve_path(self.es["seen_dir"], self.root), f"{key}_{cache_tokens}")
        first_mismatch: int | None = None if len(cache) == cache_tokens else min(
            len(cache), cache_tokens)
        compared = 0
        old_d1_tokens = self.cfg["checks"]["arxiv_old_d1_tokens"]
        old_d1_docs = 0
        val_rows: list[dict] = []
        next_report = PROGRESS_EVERY_TOKENS
        self.first_post = None
        for i, c, text, toks in self.docs:
            self._account(c, toks)
            if c >= cache_tokens:
                self.first_post = (i, c, text, toks)
                break
            if c >= next_report:
                self.log(f"  {key}: {c:,} tokens streamed")
                next_report += PROGRESS_EVERY_TOKENS
            if c < old_d1_tokens:
                old_d1_docs += 1
            end = min(c + len(toks), cache_tokens)
            if first_mismatch is None:
                ref = cache[c:end]
                got = toks[:end - c]
                if len(ref) != len(got) or not np.array_equal(ref, got):
                    n = min(len(ref), len(got))
                    diff = np.nonzero(ref[:n] != got[:n])[0]
                    first_mismatch = c + (int(diff[0]) if diff.size else n)
            compared = end
            if self.val_range is not None:
                a, b = self.val_range
                if a <= c and c + len(toks) <= b:
                    sha = text_sha1(text)
                    val_rows.append({
                        "doc_id": make_doc_id(key, i, sha), "domain": key, "split": "val",
                        "variant": "raw", "eval_set": None, "stream_index": i,
                        "c_i": c, "L_d": int(len(toks)),
                        "token_sha1": token_sha1(toks), "text_sha1": sha,
                        "o_d": None, "k_d": None})
        if self.first_post is None:
            self.stopped_by = "exhausted_before_cut"
        if first_mismatch is None and compared < cache_tokens:
            first_mismatch = compared
        return {"cache": f"{key}_{cache_tokens}", "cache_len": int(len(cache)),
                "matches": first_mismatch is None, "first_mismatch": first_mismatch,
                "old_d1_docs": old_d1_docs, "val_rows": val_rows}

    def _account(self, c: int, toks: np.ndarray) -> None:
        self.docs_read += 1
        self.tokens_read = c + len(toks)

    def _post_cut_docs(self) -> Iterator[tuple[int, int, str, np.ndarray]]:
        if self.first_post is None:
            return
        yield self.first_post
        for item in self.docs:
            self._account(item[1], item[3])
            yield item
        if self.tokens_read < self.d["max_stream_tokens"]:
            self.stopped_by = "exhausted"

    def candidates(self) -> Iterator[Candidate]:
        es = self.es
        T, kmax, min_len = es["block_size"], es["max_blocks_per_doc"], es["min_doc_tokens"]
        seed = es["offset_seed"]
        for i, c, text, toks in self._post_cut_docs():
            self.stats["post_cut_docs"] += 1
            if len(toks) < min_len:
                self.stats["ineligible_short"] += 1
                continue
            sha = text_sha1(text)
            k = n_blocks_for(len(toks), T, kmax)
            cand = Candidate(key=self.d["key"], index=i, start=c,
                             doc_id=make_doc_id(self.d["key"], i, sha), text_sha1=sha,
                             tokens=toks.astype(np.int32), n_blocks=k,
                             offset=draw_offset(len(toks), k, T, seed, sha[:12], False))
            if self.d["stripped_copy"]:
                stripped = strip_latex(text, es["strip"]["math_envs"])
                s_toks = np.asarray(self.tokenizer.encode_ordinary(stripped), dtype=np.int32)
                if len(s_toks) < min_len:
                    self.stats["ineligible_stripped_short"] += 1
                    continue
                ks = n_blocks_for(len(s_toks), T, kmax)
                cand.stripped_tokens = s_toks
                cand.stripped_n_blocks = ks
                cand.stripped_offset = draw_offset(len(s_toks), ks, T, seed, sha[:12], True)
            self.stats["eligible"] += 1
            self.lengths.append(len(toks))
            yield cand


def _take(it: Iterator[Candidate], n: int) -> list[Candidate]:
    out = []
    for cand in it:
        out.append(cand)
        if len(out) >= n:
            break
    return out


def _top_prefixes(dropped: list[Candidate], cfg: dict, tokenizer) -> list[dict]:
    w = cfg["eval_set"]["unseen_check"]["window_tokens"]
    counts = Counter(tuple(int(x) for x in c.tokens[:w]) for c in dropped)
    top = counts.most_common(cfg["eval_set"]["unseen_check"]["report_top_prefixes"])
    return [{"count": n, "tokens": list(p), "text": tokenizer.decode(list(p))} for p, n in top]


def _build_domain(dcfg: dict, cfg: dict, stream_fn: Callable[[str], Iterable[str]], tokenizer,
                  ranges: dict, kept_text: set[str], root: Path | None,
                  log: Callable[..., None]) -> tuple[dict, list[Candidate], list[dict]]:
    es = cfg["eval_set"]
    key = dcfg["key"]
    n_test = es["n_test_docs"]
    factor = es["candidate_pool_factor"]
    val_range = ranges[key]["val"] if key in ranges else None
    walk = _DomainWalk(dcfg, cfg, stream_fn(key), tokenizer, val_range, root, log)
    log(f"[{key}] streaming to the cut at {dcfg['cache_tokens']:,} tokens")
    match = walk.walk_to_cut()
    val_rows = match.pop("val_rows")
    ids_assigned = val_range is not None and match["matches"]
    if not match["matches"]:
        val_rows = []
    cand_iter = walk.candidates()
    survivors: list[Candidate] = []
    dropped: list[Candidate] = []
    local_text: set[str] = set()
    counts = Counter()
    kept_hits = 0
    kept_windows = 0
    hit_by_doc: dict[str, tuple[int, int]] = {}
    rounds = 0
    need = math.ceil(round(factor * n_test, 9))
    while True:
        new = _take(cand_iter, need)
        if not new:
            break
        rounds += 1
        log(f"[{key}] round {rounds}: unseen check on {len(new)} candidates")
        res = unseen_check(new, cfg, root, log)
        for ci, cand in enumerate(new):
            counts["candidates_checked"] += 1
            prefix, near = bool(res["prefix_hit"][ci]), bool(res["near_dup"][ci])
            counts["prefix_hit"] += prefix
            counts["near_dup_block"] += near
            if prefix or near:
                counts["unseen_total"] += 1
                dropped.append(cand)
                continue
            if cand.text_sha1 in local_text or cand.text_sha1 in kept_text:
                counts["within_test_duplicate"] += 1
                continue
            local_text.add(cand.text_sha1)
            survivors.append(cand)
            hit_by_doc[cand.doc_id] = (int(res["window_hits"][ci]), int(res["windows"][ci]))
        if len(survivors) >= n_test:
            break
        need = math.ceil(round(factor * (n_test - len(survivors)), 9))
    kept = survivors[:n_test]
    for cand in kept:
        h, n = hit_by_doc[cand.doc_id]
        kept_hits += h
        kept_windows += n
        kept_text.add(cand.text_sha1)
    checked = counts["candidates_checked"]
    drop_frac = counts["unseen_total"] / checked if checked else 0.0
    listed = drop_frac > es["unseen_check"]["report_drop_frac"]
    stats = walk.stats
    report = {
        "key": key,
        "role": dcfg["role"],
        "stream": {"docs_read": walk.docs_read, "tokens_read": walk.tokens_read,
                   "stopped_by": walk.stopped_by if len(kept) < n_test else "enough",
                   "cut_index": walk.first_post[0] if walk.first_post else None,
                   "cut_token_start": walk.first_post[1] if walk.first_post else None},
        "cache_match": {k: match[k] for k in ("cache", "cache_len", "matches", "first_mismatch")},
        "old_d1_docs": match["old_d1_docs"] if key == "arxiv" else None,
        "val": {"range": list(val_range) if val_range else None, "docs": len(val_rows),
                "ids_assigned": ids_assigned},
        "post_cut_docs_examined": stats["post_cut_docs"],
        "ineligible_short": stats["ineligible_short"],
        "ineligible_stripped_short": stats["ineligible_stripped_short"],
        "eligible_fraction": stats["eligible"] / stats["post_cut_docs"]
        if stats["post_cut_docs"] else 0.0,
        "candidates_checked": checked,
        "dropped": {k: counts[k] for k in ("prefix_hit", "near_dup_block", "unseen_total",
                                           "within_test_duplicate")},
        "drop_frac": drop_frac,
        "top_prefixes_listed": listed,
        "top_matched_prefixes": _top_prefixes(dropped, cfg, tokenizer) if listed else None,
        "rounds": rounds,
        "kept": len(kept),
        "survivors_unused": len(survivors) - len(kept),
        "median_doc_tokens": statistics.median([len(c.tokens) for c in kept]) if kept else None,
        "mean_blocks_per_doc": float(np.mean([c.n_blocks for c in kept])) if kept else None,
        "n_blocks": int(sum(c.n_blocks for c in kept)),
        "kept_block_windows": kept_windows,
        "kept_block_window_hits": kept_hits,
        "kept_block_window_hit_frac": kept_hits / kept_windows if kept_windows else 0.0,
        "median_candidate_tokens": statistics.median(walk.lengths) if walk.lengths else None,
    }
    if dcfg["stripped_copy"]:
        report["stripped"] = {
            "n_blocks": int(sum(c.stripped_n_blocks for c in kept)),
            "mean_blocks_per_doc": float(np.mean([c.stripped_n_blocks for c in kept]))
            if kept else None,
            "median_doc_tokens": statistics.median([len(c.stripped_tokens) for c in kept])
            if kept else None,
            "unseen_check": "inherited from the raw document",
        }
    return report, kept, val_rows


def _test_rows(cand: Candidate, eval_set: str, variant: str) -> dict:
    stripped = variant == "stripped"
    toks = cand.stripped_tokens if stripped else cand.tokens
    return {"doc_id": cand.doc_id, "domain": cand.key, "split": "test", "variant": variant,
            "eval_set": eval_set, "stream_index": cand.index, "c_i": cand.start,
            "L_d": int(len(toks)), "token_sha1": token_sha1(toks),
            "text_sha1": cand.text_sha1,
            "o_d": cand.stripped_offset if stripped else cand.offset,
            "k_d": cand.stripped_n_blocks if stripped else cand.n_blocks}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_outputs(cfg: dict, kept: dict[str, list[Candidate]], val_rows: dict[str, list[dict]],
                   yaml_sha: str, root: Path | None) -> str:
    es = cfg["eval_set"]
    T = es["block_size"]
    odir = out_dir(cfg, root)
    odir.mkdir(parents=True, exist_ok=True)
    sets = eval_set_domains(cfg)
    set_of = {(k, v): name for name, members in sets.items() for k, v in members}
    rows: list[dict] = []
    for d in es["domains"]:
        key = d["key"]
        variants = ["raw"] + (["stripped"] if d["stripped_copy"] else [])
        for variant in variants:
            rows += [_test_rows(c, set_of[(key, variant)], variant) for c in kept[key]]
            docs = {"domain": key, "variant": variant,
                    "doc_id": [c.doc_id for c in kept[key]],
                    "tokens": [torch.from_numpy(np.ascontiguousarray(
                        c.stripped_tokens if variant == "stripped" else c.tokens))
                        for c in kept[key]]}
            torch.save(docs, doc_file_path(cfg, key, variant, root))
        rows += val_rows[key]
    mpath = manifest_path(cfg, root)
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    manifest_sha = _sha256_file(mpath)
    for name, members in sets.items():
        toks, doc_ids, domains, variants, offsets, js, weights = [], [], [], [], [], [], []
        for key, variant in members:
            for c in kept[key]:
                stripped = variant == "stripped"
                t = c.stripped_tokens if stripped else c.tokens
                k = c.stripped_n_blocks if stripped else c.n_blocks
                o = c.stripped_offset if stripped else c.offset
                for j in range(k):
                    s = o + j * T
                    toks.append(t[s:s + T + 1].astype(np.int32))
                    doc_ids.append(c.doc_id)
                    domains.append(key)
                    variants.append(variant)
                    offsets.append(s)
                    js.append(j)
                    weights.append(1.0 / k)
        n = len(toks)
        blocks = {
            "eval_set": name,
            "tokens": torch.from_numpy(np.stack(toks)) if n else torch.zeros(
                (0, T + 1), dtype=torch.int32),
            "block_index": torch.arange(n, dtype=torch.long),
            "doc_id": doc_ids,
            "domain": domains,
            "variant": variants,
            "offset": torch.tensor(offsets, dtype=torch.long),
            "block_in_doc": torch.tensor(js, dtype=torch.long),
            "weight": torch.tensor(weights, dtype=torch.float32),
            "meta": {"format": "i2_blocks_v1", "eval_set_name": es["name"], "block_size": T,
                     "yaml_sha256": yaml_sha, "manifest_sha256": manifest_sha,
                     "created_at": utc_now_iso()},
        }
        torch.save(blocks, block_file_path(cfg, name, root))
    return manifest_sha


def build_eval_set(cfg: dict, stream_fn: Callable[[str], Iterable[str]], tokenizer,
                   stream_info: dict, root: Path | None = None,
                   log: Callable[..., None] = print, yaml_sha256: str | None = None) -> dict:
    """Stream every domain, build the test and val documents, run the unseen check and write
    the manifest, document, block and report files. Returns the build report.

    `stream_fn(key)` yields the domain's texts in shuffled stream order (`hf_text_stream` for
    the real build). `yaml_sha256` is the sha256 of the YAML file bytes; without it, the
    sha256 of the canonical JSON of `cfg` is recorded instead.
    """
    validate_eval_config(cfg, root)
    es = cfg["eval_set"]
    ranges = split_ranges(cfg, root)
    yaml_sha = yaml_sha256 or hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
    kept_text: set[str] = set()
    kept: dict[str, list[Candidate]] = {}
    val_rows: dict[str, list[dict]] = {}
    domains: dict[str, dict] = {}
    failures: list[str] = []
    for dcfg in es["domains"]:
        key = dcfg["key"]
        rep, kept[key], val_rows[key] = _build_domain(dcfg, cfg, stream_fn, tokenizer, ranges,
                                                      kept_text, root, log)
        domains[key] = rep
        if rep["kept"] != es["n_test_docs"]:
            failures.append(f"{key}: {rep['kept']} test documents, need {es['n_test_docs']}")
        if key == "arxiv" and rep["cache_match"]["matches"]:
            want = cfg["checks"]["arxiv_old_d1_docs"]
            if rep["old_d1_docs"] != want:
                failures.append(f"arxiv: old_d1_docs = {rep['old_d1_docs']}, expected {want}")
    manifest_sha = _write_outputs(cfg, kept, val_rows, yaml_sha, root)
    report = {
        "eval_set_name": es["name"],
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "created_at": utc_now_iso(),
        "yaml_sha256": yaml_sha,
        "analysis_sha256": analysis_sha256(cfg),
        "manifest_sha256": manifest_sha,
        "stream_info": stream_info,
        "split_ranges": {k: {s: list(r) for s, r in v.items()} for k, v in ranges.items()},
        "eval_sets": {name: [list(m) for m in members]
                      for name, members in eval_set_domains(cfg).items()},
        "domains": domains,
        "notes": [
            f"Documents under min_doc_tokens = {es['min_doc_tokens']} tokens are not eligible. "
            "This removes short PubMed abstracts and short StackExchange threads; S3's "
            "matched bins handle length.",
            "The stripped copy inherits the unseen-check result of its raw document.",
        ],
    }
    with open(out_dir(cfg, root) / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


# --------------------------------------------------------------------------------------------
# S1-T3 manifest checks (the builder's view; check_eval_rebuild.py --check manifest may call it)
# --------------------------------------------------------------------------------------------

def check_manifest(cfg: dict, tokenizer, root: Path | None = None) -> dict:
    """Run checks T3a-T3f on a built eval set. Values are True, False or None (n/a)."""
    es = cfg["eval_set"]
    T, kmax, n_test = es["block_size"], es["max_blocks_per_doc"], es["n_test_docs"]
    rows = load_manifest(cfg, root)
    odir = out_dir(cfg, root)
    report = json.loads((odir / "build_report.json").read_text(encoding="utf-8"))
    keys = [d["key"] for d in es["domains"]]
    cache = {d["key"]: d["cache_tokens"] for d in es["domains"]}
    test = [r for r in rows if r["split"] == "test"]
    res: dict = {}

    order: list[str] = []
    for r in rows:
        if r["domain"] not in order:
            order.append(r["domain"])
    res["T3a"] = order == keys

    ok = all(sum(1 for r in test if r["domain"] == k and r["variant"] == "raw") == n_test
             for k in keys)
    for d in es["domains"]:
        if d["stripped_copy"]:
            raw_ids = [r["doc_id"] for r in test if r["domain"] == d["key"]
                       and r["variant"] == "raw"]
            s_ids = [r["doc_id"] for r in test if r["domain"] == d["key"]
                     and r["variant"] == "stripped"]
            ok = ok and len(s_ids) == n_test and s_ids == raw_ids
    res["T3b"] = ok

    by_key = {(r["doc_id"], r["variant"]): r for r in test}
    ok = len(by_key) == len(test)
    docs: dict[tuple[str, str], np.ndarray] = {}
    for d in es["domains"]:
        for variant in ["raw"] + (["stripped"] if d["stripped_copy"] else []):
            f = torch.load(doc_file_path(cfg, d["key"], variant, root), weights_only=True)
            for doc_id, t in zip(f["doc_id"], f["tokens"]):
                docs[(doc_id, variant)] = t.numpy()
    ok = ok and set(docs) == set(by_key)
    for dk, r in by_key.items():
        t = docs.get(dk)
        k, o, n = r["k_d"], r["o_d"], r["L_d"]
        ok = ok and t is not None and len(t) == n and token_sha1(t) == r["token_sha1"]
        ok = ok and 1 <= k <= kmax and o + k * T + 1 <= n
    seen_blocks: dict[tuple[str, str], list[int]] = {}
    for name in eval_set_domains(cfg):
        b = load_block_file(cfg, name, root)
        for j in range(b["tokens"].shape[0]):
            dk = (b["doc_id"][j], b["variant"][j])
            r = by_key.get(dk)
            if r is None:
                ok = False
                continue
            off = int(b["offset"][j])
            jj = int(b["block_in_doc"][j])
            ok = ok and off == r["o_d"] + jj * T and b["domain"][j] == r["domain"]
            ok = ok and np.array_equal(b["tokens"][j].numpy(), docs[dk][off:off + T + 1])
            ok = ok and float(b["weight"][j]) == float(np.float32(1.0 / r["k_d"]))
            seen_blocks.setdefault(dk, []).append(jj)
    ok = ok and all(seen_blocks.get(dk) == list(range(r["k_d"]))
                    for dk, r in by_key.items())
    res["T3c"] = bool(ok)

    ranges = split_ranges(cfg, root)
    split_of: dict[str, set[str]] = {}
    for r in rows:
        split_of.setdefault(r["doc_id"], set()).add(r["split"])
    ok = all(len(s) == 1 for s in split_of.values())
    for r in rows:
        c, n = r["c_i"], r["L_d"]
        if r["split"] == "val":
            a, b = ranges[r["domain"]]["val"]
            ok = ok and a <= c and c + n <= b
    shas = [r["text_sha1"] for r in test if r["variant"] == "raw"]
    res["T3d"] = bool(ok and len(shas) == len(set(shas)))

    res["T3e_cut"] = all(r["c_i"] >= cache[r["domain"]] for r in test)
    if "arxiv" in report["domains"] and report["domains"]["arxiv"]["cache_match"]["matches"]:
        res["T3e_old_d1"] = (report["domains"]["arxiv"]["old_d1_docs"]
                             == cfg["checks"]["arxiv_old_d1_docs"])
    else:
        res["T3e_old_d1"] = None

    ok = True
    for name, members in eval_set_domains(cfg).items():
        if members[0][1] != "stripped":
            continue
        b = load_block_file(cfg, name, root)
        for row in b["tokens"]:
            text = tokenizer.decode(row.tolist())
            ok = ok and "\\" not in text and "$" not in text
    res["T3f"] = ok
    return res
