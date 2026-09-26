"""Refit a post-hoc posterior on held-out ID data (specs/i2-posthoc-fixes.md, Section 5.5).

    python scripts/refit_posthoc.py --config configs/i2_posthoc_c4_tfb.yaml   # first
    python scripts/refit_posthoc.py --config configs/i2_posthoc_c4_lap.yaml
    python scripts/refit_posthoc.py --config configs/i2_posthoc_c2.yaml
    python scripts/refit_posthoc.py --check-scores <S1 score files> --records <fit records>

Every key is checked before any helper runs (KeyError on a missing one). Exit codes: 0 pass;
2 a safety or end-of-run check failed (for example "eval document in fit set"); 3 the lambda
bisection failed; 4 the TFB re-measure left no budget (rho_TFB <= 0).
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minigpt import posthoc_refit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="refit YAML (configs/i2_posthoc_<cell>.yaml)")
    mode.add_argument("--check-scores", nargs="+", metavar="SCORE_FILE",
                      help="S1 score files whose base checkpoint_sha256 must match the records")
    parser.add_argument("--records", nargs="+", metavar="FIT_RECORD",
                        help="fit_record.json files, used with --check-scores")
    args = parser.parse_args(argv)

    if args.check_scores:
        if not args.records:
            parser.error("--check-scores needs --records")
        failures = posthoc_refit.check_scores(args.check_scores, args.records)
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(f"check-scores: {'PASS' if not failures else f'{len(failures)} failure(s)'}")
        return posthoc_refit.EXIT_CHECK_FAILED if failures else posthoc_refit.EXIT_OK

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    posthoc_refit.check_config(cfg)
    try:
        result = posthoc_refit.run_refit(cfg, config_path=args.config)
    except posthoc_refit.RefitCheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return posthoc_refit.EXIT_CHECK_FAILED
    for failure in result.record["failures"]:
        print(f"FAIL: end-of-run check {failure}", file=sys.stderr)
    status = result.record.get("match_status", "tfb")
    print(f"[refit] {cfg['cell']}: {status}, exit {result.exit_code}; "
          f"record {result.record_path}")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
