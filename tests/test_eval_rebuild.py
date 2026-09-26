"""S1-T2d and S1-T3: the iteration-2 eval-set builder on a synthetic stream of 5 fake domains.

Spec: specs/i2-eval-rebuild.md (sections 4, 5.2-5.4). The fixture constants live in this module.
The fixture writes its own cached tensors the way minigpt/data.py:162-178 does, so the
builder sees the same layout as the real `data/pile/` directory, only much smaller.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from minigpt import evalset

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = REPO_ROOT / "configs" / "i2_eval.yaml"
# Specs are private (agents/ is gitignored); the spec-parsing tests skip when they are absent, e.g. in CI.
SPEC_DIR = REPO_ROOT / "agents" / "specs" / "2026-10"
SPEC_PATH = SPEC_DIR / "i2-eval-rebuild.md"
S2_SPEC_PATH = SPEC_DIR / "i2-posthoc-fixes.md"
_SPECS_PRESENT = SPEC_PATH.exists() and S2_SPEC_PATH.exists()
_needs_specs = pytest.mark.skipif(not _SPECS_PRESENT, reason="private spec files not available")
S2_N_SAMPLES = 20  # S2 spec section 2: "For the three score sets: `n_samples` 20"

# --------------------------------------------------------------------------------------------
# Fixture constants
# --------------------------------------------------------------------------------------------
BLOCK = 256
WINDOW = 32
MAX_BLOCKS = 3
ID_CACHE = 20_000
OOD_CACHE = 6_000
SMALL_CACHE = 500
N_TEST = 6
POOL_FACTOR = 1.5
CHUNK = 4_096
OLD_D1_TOKENS = 1_500
ID_MAX_STREAM = 60_000
OOD_MAX_STREAM = 40_000
POST_CUT_DOCS = 24
DOMAIN_KEYS = ["stackexchange", "hackernews", "arxiv", "freelaw", "pubmed_abstracts"]
DOMAIN_SEEDS = {"wikipedia_en": 11, "stackexchange": 12, "hackernews": 13, "arxiv": 14,
                "freelaw": 15, "pubmed_abstracts": 16}
MATH_ENVS = ["equation", "equation*", "align", "align*", "eqnarray", "eqnarray*", "gather",
             "gather*", "multline", "multline*", "displaymath", "math"]
ALPHABET = np.array(list("abcdefghijklmnopqrstuvwxyz    "))
LATEX_SNIPPETS = [
    "$x^2 + y_1$",
    "$$\\int_0^1 f(x)\\,dx$$",
    "\\begin{equation} a = b + c \\end{equation}",
    "\\begin{align*} z &= w \\\\ q &= r \\end{align*}",
    "\\textbf{bold words}",
    "\\cite{ref2020}",
    "\\[ e^{i\\pi} + 1 = 0 \\]",
    "\\(\\alpha_k\\)",
    "\\section{Intro}",
    "\\emph{note}",
]
# Split arithmetic of the fixture train configs (0.8 / 0.1 / 0.1, as in configs/c0.yaml):
# base ID stream = cat(wikipedia 20K, stackexchange 20K) -> train end 32K, val end 36K.
SE_RANGES = {"train": (0, 12_000), "val": (12_000, 16_000), "old_test": (16_000, 20_000)}
HN_RANGES = {"train": (0, 16_000), "val": (16_000, 18_000), "old_test": (18_000, 20_000)}


class ByteTokenizer:
    """Invertible fake tokenizer: one token per UTF-8 byte."""

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens) -> str:
        return bytes(int(t) for t in tokens).decode("utf-8", errors="replace")


TOK = ByteTokenizer()


def _plain(rng: np.random.Generator, n: int) -> str:
    return "".join(ALPHABET[rng.integers(0, len(ALPHABET), size=n)])


def _plain_doc(rng: np.random.Generator) -> str:
    # Mostly eligible documents (>= 257 tokens), some short ones, some long enough for k = 3.
    if rng.random() < 0.2:
        return _plain(rng, int(rng.integers(60, 250)))
    return _plain(rng, int(rng.integers(300, 1_000)))


def _latex_doc(rng: np.random.Generator) -> str:
    n_target = int(rng.integers(500, 1_100))
    parts = [_plain(rng, 40)]
    total = 40
    while total < n_target:
        snippet = LATEX_SNIPPETS[int(rng.integers(0, len(LATEX_SNIPPETS)))]
        run = _plain(rng, int(rng.integers(25, 70)))
        parts += [" ", snippet, " ", run]
        total += len(snippet) + len(run) + 2
    return "".join(parts)


def _n_tok(text: str) -> int:
    return len(TOK.encode_ordinary(text))


def _domain_stream(key: str, cache_tokens: int) -> list[str]:
    """Documents until the cache is full, then POST_CUT_DOCS more (the first straddles or
    starts at the cut)."""
    rng = np.random.default_rng(DOMAIN_SEEDS[key])
    make = _latex_doc if key == "arxiv" else _plain_doc
    docs: list[str] = []
    total = 0
    while total < cache_tokens:
        text = make(rng)
        docs.append(text)
        total += _n_tok(text)
    docs += [make(rng) for _ in range(POST_CUT_DOCS)]
    # Guarantee enough eligible documents after the cut.
    docs += [_plain(rng, 600) if key != "arxiv" else _latex_doc(rng) for _ in range(12)]
    return docs


def _cut_index(docs: list[str], cache_tokens: int) -> int:
    c = 0
    for i, text in enumerate(docs):
        if c >= cache_tokens:
            return i
        c += _n_tok(text)
    raise AssertionError("stream shorter than the cache")


def _starts(docs: list[str]) -> list[int]:
    out, c = [], 0
    for text in docs:
        out.append(c)
        c += _n_tok(text)
    return out


def _cache_tensor(docs: list[str], limit: int) -> torch.Tensor:
    # Same loop as minigpt/data.py:162-177.
    tokens: list[int] = []
    for text in docs:
        tokens.extend(TOK.encode_ordinary(text))
        if len(tokens) >= limit:
            break
    return torch.tensor(tokens[:limit], dtype=torch.long)


def _train_cfg(id_domains: list[str], id_tokens: int) -> dict:
    return {
        "data": {
            "dataset": "pile",
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "pile_id_domains": id_domains,
            "pile_ood_domains": ["arxiv", "freelaw", "pubmed_abstracts"],
            "pile_id_tokens": id_tokens,
            "pile_ood_tokens": OOD_CACHE,
        },
        "train": {"seed": 1337},
    }


def _fixture_cfg(tmp_path: Path, domains: list[str], pool_factor: float) -> dict:
    cfg = evalset.load_eval_config(REAL_CONFIG)
    cfg = copy.deepcopy(cfg)
    es = cfg["eval_set"]
    base_path = tmp_path / "c0_fixture.yaml"
    adapter_path = tmp_path / "c3_fixture.yaml"
    base_path.write_text(yaml.safe_dump(_train_cfg(["wikipedia_en", "stackexchange"], ID_CACHE)))
    adapter_path.write_text(yaml.safe_dump(_train_cfg(["hackernews"], ID_CACHE)))
    es["name"] = "fixture"
    es["train_configs"] = {"base": str(base_path), "adapter": str(adapter_path)}
    roles = {"stackexchange": "id_base", "hackernews": "id_adapter"}
    es["domains"] = [
        {
            "key": key,
            "role": roles.get(key, "ood"),
            "cache_tokens": ID_CACHE if key in roles else OOD_CACHE,
            "max_stream_tokens": ID_MAX_STREAM if key in roles else OOD_MAX_STREAM,
            "stripped_copy": key == "arxiv",
        }
        for key in DOMAIN_KEYS
        if key in domains
    ]
    es["seen_dir"] = str(tmp_path / "pile")
    es["seen_tensors"] = [
        f"wikipedia_en_{ID_CACHE}", f"stackexchange_{ID_CACHE}", f"hackernews_{ID_CACHE}",
        f"arxiv_{OOD_CACHE}", f"freelaw_{OOD_CACHE}", f"pubmed_abstracts_{OOD_CACHE}",
        f"stackexchange_{SMALL_CACHE}",
    ]
    es["block_size"] = BLOCK
    es["min_doc_tokens"] = BLOCK + 1
    es["max_blocks_per_doc"] = MAX_BLOCKS
    es["n_test_docs"] = N_TEST
    es["candidate_pool_factor"] = pool_factor
    es["unseen_check"]["window_tokens"] = WINDOW
    es["unseen_check"]["chunk_tokens"] = CHUNK
    es["strip"]["math_envs"] = list(MATH_ENVS)
    es["out_dir"] = str(tmp_path / "eval_i2")
    cfg["scoring"]["out_dir"] = str(tmp_path / "scores_i2")
    cfg["checks"]["arxiv_old_d1_tokens"] = OLD_D1_TOKENS
    return cfg


class Fixture:
    def __init__(self, tmp_path: Path, *, inject: tuple[str, ...] = (),
                 domains: list[str] | None = None, pool_factor: float = POOL_FACTOR,
                 corrupt: tuple[str, int] | None = None) -> None:
        domains = list(DOMAIN_KEYS) if domains is None else domains
        self.cfg = _fixture_cfg(tmp_path, domains, pool_factor)
        self.cfg_path = tmp_path / "i2_eval_fixture.yaml"
        self.streams: dict[str, list[str]] = {}
        caches = {"wikipedia_en": ID_CACHE, "stackexchange": ID_CACHE, "hackernews": ID_CACHE,
                  "arxiv": OOD_CACHE, "freelaw": OOD_CACHE, "pubmed_abstracts": OOD_CACHE}
        for key, limit in caches.items():
            self.streams[key] = _domain_stream(key, limit)
        self.injected: dict[str, str] = {}
        rng = np.random.default_rng(99)
        if "exact_dup" in inject:
            # A cached StackExchange document of >= 257 tokens, fully inside the cache,
            # re-appears in the FreeLaw stream after its cut.
            se = self.streams["stackexchange"]
            starts = _starts(se)
            src = next(t for t, c in zip(se, starts)
                       if _n_tok(t) >= BLOCK + 1 and c + _n_tok(t) <= ID_CACHE and c > 0)
            self._insert_after_cut("freelaw", OOD_CACHE, src, 1)
            self.injected["exact_dup"] = src
        if "near_dup" in inject:
            # 280 tokens: 10 new tokens, then tokens 10-279 copied from the Wikipedia cache.
            # The copied span straddles the scan chunk edge at CHUNK tokens.
            wiki_cache = _cache_tensor(self.streams["wikipedia_en"], ID_CACHE)
            p = CHUNK - 100
            copied = TOK.decode(wiki_cache[p:p + 270].tolist())
            text = _plain(rng, 10).replace(" ", "q") + copied
            assert _n_tok(text) == 280
            self._insert_after_cut("pubmed_abstracts", OOD_CACHE, text, 2)
            self.injected["near_dup"] = text
        if "within_dup" in inject:
            text = _plain(rng, 700)
            self._insert_after_cut("hackernews", ID_CACHE, text, 1)
            self._insert_after_cut("hackernews", ID_CACHE, text, 3)
            self.injected["within_dup"] = text
        if "stripped_short" in inject:
            # Raw copy is eligible, the stripped copy is not.
            text = _plain(rng, 40) + " " + " ".join(["$a_{ij} = b^2$"] * 40)
            assert _n_tok(text) >= BLOCK + 1
            self._insert_after_cut("arxiv", OOD_CACHE, text, 1)
            self.injected["stripped_short"] = text
        arxiv = self.streams["arxiv"]
        self.expected_old_d1 = sum(1 for c in _starts(arxiv) if c < OLD_D1_TOKENS)
        self.cfg["checks"]["arxiv_old_d1_docs"] = self.expected_old_d1
        pile = tmp_path / "pile"
        pile.mkdir(parents=True, exist_ok=True)
        for key, limit in caches.items():
            t = _cache_tensor(self.streams[key], limit)
            if corrupt is not None and corrupt[0] == key:
                t[corrupt[1]] = (int(t[corrupt[1]]) + 1) % 256
            torch.save(t, pile / f"{key}_{limit}.pt")
        torch.save(_cache_tensor(self.streams["stackexchange"], SMALL_CACHE),
                   pile / f"stackexchange_{SMALL_CACHE}.pt")
        self.cfg_path.write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        self.cache_tokens = {d["key"]: d["cache_tokens"] for d in self.cfg["eval_set"]["domains"]}

    def _insert_after_cut(self, key: str, cache_tokens: int, text: str, pos: int) -> None:
        docs = self.streams[key]
        k = _cut_index(docs, cache_tokens)
        docs.insert(k + pos, text)

    def stream_fn(self, key: str) -> Iterator[str]:
        return iter(list(self.streams[key]))

    def build(self) -> dict:
        return evalset.build_eval_set(self.cfg, stream_fn=self.stream_fn, tokenizer=TOK,
                                      stream_info={"source": "synthetic"}, log=lambda *a: None)


def _manifest(cfg: dict) -> list[dict]:
    path = Path(cfg["eval_set"]["out_dir"]) / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _detections(report: dict) -> int:
    return sum(d["dropped"]["unseen_total"] for d in report["domains"].values())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s1_build")
    fx = Fixture(tmp, inject=("exact_dup", "near_dup", "within_dup", "stripped_short"))
    report = fx.build()
    return fx, report


# --------------------------------------------------------------------------------------------
# The real YAML
# --------------------------------------------------------------------------------------------

def _spec_yaml_block() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    start = text.index("### 5.2")
    m = re.search(r"```yaml\n(.*?)```", text[start:], flags=re.S)
    assert m is not None
    return yaml.safe_load(m.group(1))


def _s2_scoring_rows() -> dict[str, dict]:
    """S2 spec section 5.5, the `scoring` entries table: display, sampler, source, blocks."""
    text = S2_SPEC_PATH.read_text(encoding="utf-8")
    lines = text[text.index("Entries for the `scoring` section"):].splitlines()[1:]
    row_re = re.compile(r'^\| `(\w+)` \| "([^"]+)" \| `([\w-]+)` \| `(\w+)` \| (.+) \|$')
    rows: dict[str, dict] = {}
    for line in lines:
        m = row_re.match(line.strip())
        if m:
            rows[m.group(1)] = {"display": m.group(2), "sampler": m.group(3),
                                "adapter_source": m.group(4),
                                "blocks": re.findall(r"`(\w+)`", m.group(5))}
        elif rows and not line.startswith("|"):
            break
    return rows


@_needs_specs
def test_s2_scoring_table_parses():
    rows = _s2_scoring_rows()
    assert list(rows) == ["c2_refit", "c4_lap_refit", "c4_tfb_fixed"]
    assert rows["c2_refit"]["blocks"] == ["test"]
    assert rows["c4_tfb_fixed"]["blocks"] == ["test", "test_hn"]
    assert {r["sampler"] for r in rows.values()} == {"v2"}


@_needs_specs
def test_real_yaml_matches_spec_section_5_2():
    cfg = evalset.load_eval_config(REAL_CONFIG)
    spec = _spec_yaml_block()
    # The frozen block and the checks block are the spec's, value for value. The scoring
    # block is the spec's plus the three S2 score sets (S2 spec sections 2 and 5.5), which
    # S2 adds to `scoring` only; they follow the S1 sets in each listing.
    assert cfg["analysis"] == spec["analysis"]
    expected_scoring = copy.deepcopy(spec["scoring"])
    for name, row in _s2_scoring_rows().items():
        expected_scoring["labels"][name] = {k: row[k]
                                            for k in ("display", "sampler", "adapter_source")}
        expected_scoring["n_samples"][name] = S2_N_SAMPLES
        for eval_set in row["blocks"]:
            expected_scoring["score_sets"][eval_set].append(name)
    assert cfg["scoring"] == expected_scoring
    extra_checks = {k: v for k, v in cfg["checks"].items() if k != "arxiv_old_d1_tokens"}
    assert extra_checks == spec["checks"]
    assert cfg["checks"]["arxiv_old_d1_tokens"] == 500 * 256 + 1
    es, spec_es = cfg["eval_set"], spec["eval_set"]
    for d in es["domains"]:
        d_spec = next(x for x in spec_es["domains"] if x["key"] == d["key"])
        assert d["stripped_copy"] == d_spec.get("stripped_copy", False)
        assert {k: v for k, v in d.items() if k != "stripped_copy"} == {
            k: v for k, v in d_spec.items() if k != "stripped_copy"}
    assert [d["key"] for d in es["domains"]] == [d["key"] for d in spec_es["domains"]]
    for k, v in spec_es.items():
        if k != "domains":
            assert es[k] == v, k
    assert es["seen_dir"] == "data/pile"
    evalset.validate_eval_config(cfg)


def test_t3g_split_ranges_real_configs():
    cfg = evalset.load_eval_config(REAL_CONFIG)
    ranges = evalset.split_ranges(cfg)
    assert ranges["stackexchange"] == {"train": (0, 60_000_000), "val": (60_000_000, 80_000_000),
                                       "old_test": (80_000_000, 100_000_000)}
    assert ranges["hackernews"] == {"train": (0, 80_000_000), "val": (80_000_000, 90_000_000),
                                    "old_test": (90_000_000, 100_000_000)}
    assert set(ranges) == {"stackexchange", "hackernews"}


def test_t3g_split_ranges_fixture(tmp_path):
    fx = Fixture(tmp_path)
    ranges = evalset.split_ranges(fx.cfg)
    assert ranges["stackexchange"] == SE_RANGES
    assert ranges["hackernews"] == HN_RANGES


def test_config_validation_rejects_bad_min_doc_tokens(tmp_path):
    fx = Fixture(tmp_path)
    cfg = copy.deepcopy(fx.cfg)
    cfg["eval_set"]["min_doc_tokens"] = BLOCK
    with pytest.raises(ValueError, match="min_doc_tokens"):
        evalset.validate_eval_config(cfg)


# --------------------------------------------------------------------------------------------
# Unit pieces: hash, strip, offsets, freeze
# --------------------------------------------------------------------------------------------

def test_window_hashes_match_polynomial_formula():
    rng = np.random.default_rng(5)
    tokens = rng.integers(0, 50_257, size=300)
    base = 1_000_003
    h = evalset.window_hashes(tokens, WINDOW, base)
    assert h.dtype == np.uint64 and len(h) == 300 - WINDOW + 1
    mod = 2 ** 64
    for i in (0, 1, 137, 300 - WINDOW):
        ref = sum(int(tokens[i + j]) * pow(base, WINDOW - 1 - j, mod) for j in range(WINDOW)) % mod
        assert int(h[i]) == ref
    assert len(evalset.window_hashes(tokens[:WINDOW - 1], WINDOW, base)) == 0


def test_unseen_index_confirms_hits_exactly_under_hash_collisions():
    # With base 1 the hash is the token sum, so every permutation collides. Exact comparison
    # must keep collisions from counting as hits; true copies still hit.
    rng = np.random.default_rng(7)
    q1 = rng.integers(0, 1_000, size=WINDOW)
    q2 = q1[::-1].copy()                      # same sum, different content: a collision
    q3 = rng.integers(0, 1_000, size=WINDOW)
    flat = np.concatenate([q1, q2, q3])
    qpos = np.array([0, WINDOW, 2 * WINDOW])
    index = evalset.UnseenIndex(flat, qpos, WINDOW, base=1)
    seen_perm = np.concatenate([rng.integers(0, 1_000, size=50), np.roll(q3, 3),
                                rng.integers(0, 1_000, size=50)])
    index.scan(seen_perm, chunk_tokens=40)
    assert not index.query_hit.any()
    seen_true = np.concatenate([rng.integers(0, 1_000, size=30), q2,
                                rng.integers(0, 1_000, size=45)])
    index.scan(seen_true, chunk_tokens=40)      # q2 straddles the chunk edge at 40
    assert index.query_hit.tolist() == [False, True, False]


def test_strip_latex_removes_math_and_commands():
    text = ("Intro \\textbf{bold words} and $x^2$ plus $$\\int f$$ then \\[ a \\] and \\(b\\) "
            "\\begin{equation} E = mc^2 \\end{equation} \\begin{align*} z \\\\ q \\end{align*} "
            "\\cite{ref} end \\\\ done \\$ {x}")
    out = evalset.strip_latex(text, MATH_ENVS)
    assert "\\" not in out and "$" not in out and "{" not in out and "}" not in out
    assert "bold words" in out and "ref" in out and out.startswith("Intro")
    assert "mc" not in out and "int" not in out and "E =" not in out
    assert "  " not in out


def test_offsets_follow_per_document_rng():
    sha1_12 = "0123456789ab"
    o_raw = evalset.draw_offset(900, 3, BLOCK, 0, sha1_12, stripped=False)
    o_str = evalset.draw_offset(900, 3, BLOCK, 0, sha1_12, stripped=True)
    assert o_raw == int(np.random.default_rng([0, int(sha1_12, 16)]).integers(0, 900 - 3 * BLOCK))
    assert o_str == int(np.random.default_rng([0, int(sha1_12, 16), 1]).integers(0, 900 - 768))
    assert evalset.n_blocks_for(257, BLOCK, MAX_BLOCKS) == 1
    assert evalset.n_blocks_for(512, BLOCK, MAX_BLOCKS) == 1
    assert evalset.n_blocks_for(513, BLOCK, MAX_BLOCKS) == 2
    assert evalset.n_blocks_for(5_000, BLOCK, MAX_BLOCKS) == 3
    assert evalset.n_blocks_for(5_000, BLOCK, 1) == 1


def test_freeze_writes_hash_and_refuses_overwrite(tmp_path):
    cfg = evalset.load_eval_config(REAL_CONFIG)
    expected = hashlib.sha256(json.dumps(cfg["analysis"], sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    assert evalset.analysis_sha256(cfg) == expected
    path = tmp_path / "scores" / "prereg.json"
    rec = evalset.freeze_prereg(cfg, path)
    assert rec["analysis_sha256"] == expected
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["analysis_sha256"] == expected and "frozen_at" in on_disk
    assert evalset.verify_prereg(cfg, path) == expected
    with pytest.raises(FileExistsError):
        evalset.freeze_prereg(cfg, path)
    changed = copy.deepcopy(cfg)
    changed["analysis"]["margin_auroc"] = 0.03
    with pytest.raises(evalset.PreregError):
        evalset.verify_prereg(changed, path)
    with pytest.raises(evalset.PreregError):
        evalset.verify_prereg(cfg, tmp_path / "missing.json")


# --------------------------------------------------------------------------------------------
# S1-T2d: unseen check on the synthetic stream
# --------------------------------------------------------------------------------------------

def test_t2d_clean_stream_gives_zero_detections(tmp_path):
    report = Fixture(tmp_path).build()
    assert _detections(report) == 0
    assert report["status"] == "PASS", report["failures"]


def test_t2d_exact_duplicate_gives_one_detection(tmp_path):
    fx = Fixture(tmp_path, inject=("exact_dup",))
    report = fx.build()
    assert _detections(report) == 1
    fl = report["domains"]["freelaw"]["dropped"]
    assert fl["unseen_total"] == 1 and fl["prefix_hit"] == 1
    sha = hashlib.sha1(fx.injected["exact_dup"].encode("utf-8")).hexdigest()
    assert all(r["text_sha1"] != sha for r in _manifest(fx.cfg))


def test_t2d_near_duplicate_gives_one_detection(tmp_path):
    fx = Fixture(tmp_path, inject=("near_dup",))
    report = fx.build()
    assert _detections(report) == 1
    pm = report["domains"]["pubmed_abstracts"]["dropped"]
    assert pm["unseen_total"] == 1 and pm["near_dup_block"] == 1 and pm["prefix_hit"] == 0
    sha = hashlib.sha1(fx.injected["near_dup"].encode("utf-8")).hexdigest()
    assert all(r["text_sha1"] != sha for r in _manifest(fx.cfg))


def test_near_dup_threshold_is_half_of_the_226_windows(tmp_path):
    # One 257-token block (k = 1, offset 0): new tokens, then the last m tokens copied from a
    # seen tensor, so exactly m - 31 windows hit and the 32-token prefix is new.
    cfg = _fixture_cfg(tmp_path, DOMAIN_KEYS, POOL_FACTOR)
    rng = np.random.default_rng(3)
    seen = rng.integers(0, 50_000, size=5_000)
    seen_dir = tmp_path / "seen_only"
    seen_dir.mkdir()
    torch.save(torch.from_numpy(seen), seen_dir / "only_5000.pt")
    cfg["eval_set"]["seen_dir"] = str(seen_dir)
    cfg["eval_set"]["seen_tensors"] = ["only_5000"]
    n_windows = BLOCK + 1 - WINDOW + 1
    assert n_windows == 226
    cands = []
    for m in (113 + 31, 112 + 31):
        toks = np.concatenate([rng.integers(50_000, 50_257, size=BLOCK + 1 - m),
                               seen[1_000:1_000 + m]]).astype(np.int32)
        assert evalset.n_blocks_for(len(toks), BLOCK, MAX_BLOCKS) == 1
        cands.append(evalset.Candidate(key="freelaw", index=len(cands), start=0,
                                       doc_id=f"freelaw/{len(cands):09d}/000000000000",
                                       text_sha1="0" * 40, tokens=toks, n_blocks=1, offset=0))
    res = evalset.unseen_check(cands, cfg, log=lambda *a: None)
    assert res["window_hits"].tolist() == [113, 112]
    assert res["near_dup"].tolist() == [True, False]
    assert res["prefix_hit"].tolist() == [False, False]


def test_t3e_old_d1_is_na_when_arxiv_stream_differs(tmp_path):
    fx = Fixture(tmp_path, corrupt=("arxiv", 100))
    fx.cfg["checks"]["arxiv_old_d1_docs"] = fx.expected_old_d1 + 5   # would fail if checked
    report = fx.build()
    assert report["domains"]["arxiv"]["cache_match"]["first_mismatch"] == 100
    assert report["status"] == "PASS", report["failures"]
    assert evalset.check_manifest(fx.cfg, tokenizer=TOK)["T3e_old_d1"] is None


def test_pool_extends_when_too_few_survive(tmp_path):
    # Pool factor 1.0: the first pool holds exactly N_TEST candidates, one is dropped, so the
    # builder must extend the pool and still keep N_TEST documents.
    fx = Fixture(tmp_path, inject=("exact_dup",), pool_factor=1.0)
    report = fx.build()
    fl = report["domains"]["freelaw"]
    assert fl["rounds"] >= 2 and fl["kept"] == N_TEST
    assert report["status"] == "PASS", report["failures"]


def test_within_test_duplicates_and_short_stripped_copies_are_dropped(built):
    fx, report = built
    hn = report["domains"]["hackernews"]
    assert hn["dropped"]["within_test_duplicate"] == 1
    sha = hashlib.sha1(fx.injected["within_dup"].encode("utf-8")).hexdigest()
    rows = [r for r in _manifest(fx.cfg) if r["text_sha1"] == sha]
    assert len(rows) == 1
    ax = report["domains"]["arxiv"]
    assert ax["ineligible_stripped_short"] == 1
    sha = hashlib.sha1(fx.injected["stripped_short"].encode("utf-8")).hexdigest()
    assert all(r["text_sha1"] != sha for r in _manifest(fx.cfg))


def test_t2c_report_fields_and_window_hit_count(built):
    fx, report = built
    uc = fx.cfg["eval_set"]["unseen_check"]
    n_windows = BLOCK + 1 - WINDOW + 1
    for key in DOMAIN_KEYS:
        d = report["domains"][key]
        for field in ("candidates_checked", "dropped", "kept", "kept_block_windows",
                      "kept_block_window_hits", "kept_block_window_hit_frac", "eligible_fraction",
                      "median_doc_tokens", "mean_blocks_per_doc", "top_prefixes_listed"):
            assert field in d, (key, field)
        assert d["kept"] == N_TEST
        n_blocks = sum(r["k_d"] for r in _manifest(fx.cfg)
                       if r["domain"] == key and r["split"] == "test" and r["variant"] == "raw")
        assert d["kept_block_windows"] == n_blocks * n_windows
        drop_frac = d["dropped"]["unseen_total"] / d["candidates_checked"]
        assert d["top_prefixes_listed"] == (drop_frac > uc["report_drop_frac"])
    # Independent recount of kept-block window hits: every window of every kept block against
    # a Python set of every window of every seen tensor.
    seen_windows: set[bytes] = set()
    for name in fx.cfg["eval_set"]["seen_tensors"]:
        t = torch.load(Path(fx.cfg["eval_set"]["seen_dir"]) / f"{name}.pt").numpy()
        v = np.lib.stride_tricks.sliding_window_view(t.astype(np.int64), WINDOW)
        seen_windows.update(row.tobytes() for row in v)
    for es_name in ("test", "test_hn"):
        blocks = evalset.load_block_file(fx.cfg, es_name)
        tok = blocks["tokens"].numpy().astype(np.int64)
        for key in set(blocks["domain"]):
            rows = [i for i, d in enumerate(blocks["domain"]) if d == key]
            hits = 0
            for i in rows:
                v = np.lib.stride_tricks.sliding_window_view(tok[i], WINDOW)
                block_hits = sum(row.tobytes() in seen_windows for row in v)
                assert block_hits < n_windows / 2
                hits += block_hits
            assert hits == report["domains"][key]["kept_block_window_hits"], key
    # T2b by construction: no kept document's first 32 tokens occur in a seen tensor.
    for key in DOMAIN_KEYS:
        docs = torch.load(Path(fx.cfg["eval_set"]["out_dir"]) / f"docs_{key}.pt")
        for t in docs["tokens"]:
            assert t[:WINDOW].numpy().astype(np.int64).tobytes() not in seen_windows


# --------------------------------------------------------------------------------------------
# S1-T2a: stream vs cache match; val IDs only on a match
# --------------------------------------------------------------------------------------------

def test_t2a_match_reported_and_val_rows_present(built):
    fx, report = built
    for key in DOMAIN_KEYS:
        m = report["domains"][key]["cache_match"]
        assert m["matches"] is True and m["first_mismatch"] is None
    rows = _manifest(fx.cfg)
    for key, rng in (("stackexchange", SE_RANGES["val"]), ("hackernews", HN_RANGES["val"])):
        val = [r for r in rows if r["domain"] == key and r["split"] == "val"]
        docs = fx.streams[key]
        starts = _starts(docs)
        expected = [i for i, (t, c) in enumerate(zip(docs, starts))
                    if rng[0] <= c and c + _n_tok(t) <= rng[1]]
        assert [r["stream_index"] for r in val] == expected
        for r in val:
            text = docs[r["stream_index"]]
            assert r["c_i"] == starts[r["stream_index"]]
            assert r["L_d"] == _n_tok(text)
            assert r["o_d"] is None and r["k_d"] is None
    assert not [r for r in rows if r["split"] == "val" and r["domain"] not in
                ("stackexchange", "hackernews")]


def test_t2a_mismatch_reported_and_val_ids_withheld(tmp_path):
    fx = Fixture(tmp_path, corrupt=("stackexchange", 12_345))
    report = fx.build()
    m = report["domains"]["stackexchange"]["cache_match"]
    assert m["matches"] is False and m["first_mismatch"] == 12_345
    assert report["domains"]["hackernews"]["cache_match"]["matches"] is True
    rows = _manifest(fx.cfg)
    assert not [r for r in rows if r["domain"] == "stackexchange" and r["split"] == "val"]
    assert [r for r in rows if r["domain"] == "hackernews" and r["split"] == "val"]
    assert report["domains"]["stackexchange"]["val"]["ids_assigned"] is False
    # Test documents are still built: the unseen check is the guarantee.
    assert report["domains"]["stackexchange"]["kept"] == N_TEST


# --------------------------------------------------------------------------------------------
# S1-T3: manifest invariants
# --------------------------------------------------------------------------------------------

def test_t3_check_manifest_all_pass(built):
    fx, _ = built
    result = evalset.check_manifest(fx.cfg, tokenizer=TOK)
    for name in ("T3a", "T3b", "T3c", "T3d", "T3e_cut", "T3e_old_d1", "T3f"):
        assert result[name] is True, (name, result)


def test_t3a_domain_list_follows_yaml(built, tmp_path):
    fx, _ = built
    rows = _manifest(fx.cfg)
    seen: list[str] = []
    for r in rows:
        if r["domain"] not in seen:
            seen.append(r["domain"])
    assert seen == DOMAIN_KEYS
    fx2 = Fixture(tmp_path, domains=[k for k in DOMAIN_KEYS if k != "pubmed_abstracts"])
    report = fx2.build()
    rows2 = _manifest(fx2.cfg)
    assert {r["domain"] for r in rows2} == set(DOMAIN_KEYS) - {"pubmed_abstracts"}
    assert "pubmed_abstracts" not in report["domains"]
    assert evalset.check_manifest(fx2.cfg, tokenizer=TOK)["T3a"] is True


def test_t3b_counts_and_stripped_ids(built):
    fx, _ = built
    rows = [r for r in _manifest(fx.cfg) if r["split"] == "test"]
    for key in DOMAIN_KEYS:
        raw = [r for r in rows if r["domain"] == key and r["variant"] == "raw"]
        assert len(raw) == N_TEST, key
    raw_ax = [r["doc_id"] for r in rows if r["domain"] == "arxiv" and r["variant"] == "raw"]
    str_ax = [r["doc_id"] for r in rows if r["domain"] == "arxiv" and r["variant"] == "stripped"]
    assert len(str_ax) == N_TEST and str_ax == raw_ax
    assert not [r for r in rows if r["variant"] == "stripped" and r["domain"] != "arxiv"]


def test_t3c_documents_blocks_and_offsets(built):
    fx, _ = built
    out = Path(fx.cfg["eval_set"]["out_dir"])
    rows = {(r["doc_id"], r["variant"]): r for r in _manifest(fx.cfg) if r["split"] == "test"}
    stored: dict[tuple[str, str], np.ndarray] = {}
    for key in DOMAIN_KEYS:
        for variant, fname in (("raw", f"docs_{key}.pt"), ("stripped", f"docs_{key}_stripped.pt")):
            if variant == "stripped" and key != "arxiv":
                assert not (out / fname).exists()
                continue
            docs = torch.load(out / fname)
            for doc_id, t in zip(docs["doc_id"], docs["tokens"]):
                assert t.dtype == torch.int32
                stored[(doc_id, variant)] = t.numpy()
    assert set(stored) == set(rows)
    for (doc_id, variant), r in rows.items():
        key = r["domain"]
        i = r["stream_index"]
        text = fx.streams[key][i]
        sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
        assert doc_id == f"{key}/{i:09d}/{sha[:12]}" and r["text_sha1"] == sha
        assert r["c_i"] == _starts(fx.streams[key])[i]
        if variant == "raw":
            expected = np.array(TOK.encode_ordinary(text), dtype=np.int32)
        else:
            stripped = evalset.strip_latex(text, MATH_ENVS)
            expected = np.array(TOK.encode_ordinary(stripped), dtype=np.int32)
        t = stored[(doc_id, variant)]
        assert np.array_equal(t, expected)
        assert r["L_d"] == len(t)
        assert r["token_sha1"] == hashlib.sha1(t.astype("<i4").tobytes()).hexdigest()
        k, o, n = r["k_d"], r["o_d"], r["L_d"]
        assert 1 <= k <= MAX_BLOCKS and k == min(MAX_BLOCKS, (n - 1) // BLOCK)
        assert o + k * BLOCK + 1 <= n
        seq = [0, int(sha[:12], 16)] + ([1] if variant == "stripped" else [])
        assert o == int(np.random.default_rng(seq).integers(0, n - k * BLOCK))
    per_doc: dict[tuple[str, str], list[int]] = {}
    for es_name in ("test", "test_hn", "arxiv_stripped"):
        b = evalset.load_block_file(fx.cfg, es_name)
        n_b = b["tokens"].shape[0]
        assert b["tokens"].shape == (n_b, BLOCK + 1) and b["tokens"].dtype == torch.int32
        assert b["block_index"].tolist() == list(range(n_b))
        for j in range(n_b):
            doc_key = (b["doc_id"][j], b["variant"][j])
            r = rows[doc_key]
            assert b["domain"][j] == r["domain"]
            off = int(b["offset"][j])
            assert off == r["o_d"] + int(b["block_in_doc"][j]) * BLOCK
            doc = stored[doc_key]
            assert np.array_equal(b["tokens"][j].numpy(), doc[off:off + BLOCK + 1])
            assert b["weight"].dtype == torch.float32
            assert float(b["weight"][j]) == float(np.float32(1.0 / r["k_d"]))
            per_doc.setdefault(doc_key, []).append(int(b["block_in_doc"][j]))
    assert set(per_doc) == set(rows)
    for doc_key, js in per_doc.items():
        assert js == list(range(rows[doc_key]["k_d"]))
    test_blocks = evalset.load_block_file(fx.cfg, "test")
    doms = list(dict.fromkeys(test_blocks["domain"]))
    assert doms == ["stackexchange", "arxiv", "freelaw", "pubmed_abstracts"]
    assert set(evalset.load_block_file(fx.cfg, "test_hn")["domain"]) == {"hackernews"}
    assert set(evalset.load_block_file(fx.cfg, "arxiv_stripped")["variant"]) == {"stripped"}


def test_t3d_splits_disjoint_and_no_duplicates(built):
    fx, _ = built
    rows = _manifest(fx.cfg)
    test_ids = {r["doc_id"] for r in rows if r["split"] == "test"}
    val_ids = {r["doc_id"] for r in rows if r["split"] == "val"}
    assert not test_ids & val_ids
    raw_test = [r for r in rows if r["split"] == "test" and r["variant"] == "raw"]
    shas = [r["text_sha1"] for r in raw_test]
    assert len(shas) == len(set(shas))
    ranges = evalset.split_ranges(fx.cfg)
    for r in rows:
        c, n = r["c_i"], r["L_d"]
        if r["split"] == "val":
            a, b = ranges[r["domain"]]["val"]
            assert a <= c and c + n <= b
        else:
            assert c >= fx.cache_tokens[r["domain"]]


def test_t3e_cut_and_old_d1_count(built):
    fx, report = built
    for r in _manifest(fx.cfg):
        if r["split"] == "test":
            assert r["c_i"] >= fx.cache_tokens[r["domain"]]
            assert r["stream_index"] >= _cut_index(fx.streams[r["domain"]],
                                                   fx.cache_tokens[r["domain"]])
    ax = report["domains"]["arxiv"]
    assert ax["old_d1_docs"] == fx.expected_old_d1
    assert fx.expected_old_d1 >= 2


def test_t3e_old_d1_count_mismatch_fails_the_build(tmp_path):
    fx = Fixture(tmp_path)
    fx.cfg["checks"]["arxiv_old_d1_docs"] = fx.expected_old_d1 + 1
    report = fx.build()
    assert report["status"] == "FAIL"
    assert any("old_d1" in f for f in report["failures"])
    assert evalset.check_manifest(fx.cfg, tokenizer=TOK)["T3e_old_d1"] is False


def test_t3f_stripped_blocks_have_no_backslash_or_dollar(built):
    fx, _ = built
    b = evalset.load_block_file(fx.cfg, "arxiv_stripped")
    assert b["tokens"].shape[0] >= N_TEST
    for row in b["tokens"]:
        text = TOK.decode(row.tolist())
        assert "\\" not in text and "$" not in text
    raw = evalset.load_block_file(fx.cfg, "test")
    raw_text = "".join(TOK.decode(r.tolist()) for r, d in zip(raw["tokens"], raw["domain"])
                       if d == "arxiv")
    assert "\\" in raw_text or "$" in raw_text


# --------------------------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------------------------

def _load_script():
    path = REPO_ROOT / "scripts" / "build_eval_set.py"
    spec = importlib.util.spec_from_file_location("build_eval_set_script", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_build_script_cli(tmp_path, monkeypatch):
    fx = Fixture(tmp_path)
    mod = _load_script()
    monkeypatch.setattr(mod, "get_tokenizer", lambda: TOK)
    monkeypatch.setattr(mod.evalset, "hf_text_stream", lambda key, source: fx.stream_fn(key))
    monkeypatch.setattr(mod, "stream_info", lambda cfg: {"source": "synthetic"})
    assert mod.main(["--config", str(fx.cfg_path)]) == 0
    assert (Path(fx.cfg["eval_set"]["out_dir"]) / "manifest.jsonl").exists()
    report = json.loads((Path(fx.cfg["eval_set"]["out_dir"]) / "build_report.json")
                        .read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    # A second build refuses to overwrite the frozen eval set.
    assert mod.main(["--config", str(fx.cfg_path)]) != 0
    assert mod.main(["--config", str(fx.cfg_path), "--overwrite"]) == 0
    prereg = Path(fx.cfg["scoring"]["out_dir"]) / "prereg.json"
    assert mod.main(["--config", str(fx.cfg_path), "--freeze"]) == 0
    assert json.loads(prereg.read_text(encoding="utf-8"))["analysis_sha256"] == \
        evalset.analysis_sha256(fx.cfg)
    assert mod.main(["--config", str(fx.cfg_path), "--freeze"]) != 0
