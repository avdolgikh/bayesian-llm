"""Build the iteration-2 document-level eval set (specs/i2-eval-rebuild.md, S1 m4a).

Re-streams the 5 Pile domains of `configs/i2_eval.yaml` from Hugging Face, assigns document
IDs, keeps test documents after each cache cut that pass the unseen check against every
cached tensor, builds 1-3 blocks per document and writes `data/eval_i2/`. Network heavy
(about 250-280M tokens): a night job. It never writes into `data/pile/`.

Usage:
    python scripts/build_eval_set.py --config configs/i2_eval.yaml             # build
    python scripts/build_eval_set.py --config configs/i2_eval.yaml --overwrite # rebuild
    python scripts/build_eval_set.py --config configs/i2_eval.yaml --freeze    # prereg.json
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minigpt import evalset
from minigpt.data import get_tokenizer


def stream_info(cfg: dict) -> dict:
    import datasets as hf_datasets

    return {"datasets_version": hf_datasets.__version__, **cfg["eval_set"]["source"]}


def _log(*args) -> None:
    print(*args, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="eval-set YAML (configs/i2_eval.yaml)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing manifest in eval_set.out_dir")
    parser.add_argument("--freeze", action="store_true",
                        help="write the analysis-block sha256 to scoring.out_dir/prereg.json "
                             "(refuses to overwrite) and exit")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    raw = cfg_path.read_bytes()
    cfg = evalset.load_eval_config(cfg_path)

    if args.freeze:
        path = evalset.prereg_path(cfg)
        try:
            record = evalset.freeze_prereg(cfg, path)
        except FileExistsError as exc:
            _log(f"REFUSED: {exc}")
            return 1
        _log(f"Frozen: {path}")
        _log(json.dumps(record, indent=2))
        return 0

    evalset.validate_eval_config(cfg)
    mpath = evalset.manifest_path(cfg)
    if mpath.exists() and not args.overwrite:
        _log(f"REFUSED: {mpath} exists. Pass --overwrite to rebuild the eval set.")
        return 1

    report = evalset.build_eval_set(
        cfg,
        stream_fn=lambda key: evalset.hf_text_stream(key, cfg["eval_set"]["source"]),
        tokenizer=get_tokenizer(),
        stream_info=stream_info(cfg),
        log=_log,
        yaml_sha256=hashlib.sha256(raw).hexdigest(),
    )
    _log("")
    _log(f"{'domain':<18} {'match':<6} {'checked':>7} {'unseen':>6} {'dups':>4} {'kept':>5} "
         f"{'blocks':>6} {'win-hit':>8}")
    for key, d in report["domains"].items():
        _log(f"{key:<18} {str(d['cache_match']['matches']):<6} {d['candidates_checked']:>7} "
             f"{d['dropped']['unseen_total']:>6} {d['dropped']['within_test_duplicate']:>4} "
             f"{d['kept']:>5} {d['n_blocks']:>6} {d['kept_block_window_hit_frac']:>8.4f}")
    _log(f"Status: {report['status']}")
    for failure in report["failures"]:
        _log(f"  FAIL: {failure}")
    _log(f"Manifest sha256: {report['manifest_sha256']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
