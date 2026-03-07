"""Compare MLflow runs against uncertainty/perplexity gates.

Example:
    python scripts/compare_runs.py \
      --runs c7855477d3bf4ae09bd66fcb3949351a 1848a95f9d494a6baca5e41a5cd65829 \
      --baseline 1238951099144292844c33258721fa80
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import mlflow
from mlflow.tracking import MlflowClient


def _safe_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_run_id(run_id: str) -> str:
    return run_id[:8]


def _artifact_file_path(artifact_uri: str, artifact_name: str) -> Path | None:
    parsed = urlparse(artifact_uri)
    if parsed.scheme != "file":
        return None
    path_str = unquote(parsed.path)
    # Windows file URIs often look like /D:/path/...; strip leading slash.
    if len(path_str) >= 3 and path_str[0] == "/" and path_str[2] == ":":
        path_str = path_str[1:]
    base = Path(path_str)
    return base / artifact_name


def _read_qual_ratio(run) -> tuple[float | None, float | None, float | None]:
    """Return (id_avg, ood_avg, ratio) from qualitative_metrics.json artifact."""
    path = _artifact_file_path(run.info.artifact_uri, "qualitative_metrics.json")
    if path is None or not path.exists():
        return None, None, None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None, None

    if not isinstance(data, list):
        return None, None, None

    id_vals = [
        row.get("prompt_mi") for row in data
        if isinstance(row, dict) and row.get("split") == "ID"
    ]
    ood_vals = [
        row.get("prompt_mi") for row in data
        if isinstance(row, dict) and row.get("split") == "OOD"
    ]
    id_vals = [float(v) for v in id_vals if isinstance(v, (int, float))]
    ood_vals = [float(v) for v in ood_vals if isinstance(v, (int, float))]

    if not id_vals or not ood_vals:
        return None, None, None

    id_avg = sum(id_vals) / len(id_vals)
    ood_avg = sum(ood_vals) / len(ood_vals)
    ratio = ood_avg / max(id_avg, 1e-10)
    return id_avg, ood_avg, ratio


@dataclass
class KlSummary:
    first: float | None = None
    last: float | None = None
    delta: float | None = None
    up_steps: int = 0
    down_steps: int = 0


def _read_kl_summary(client: MlflowClient, run_id: str) -> KlSummary:
    hist = client.get_metric_history(run_id, "kl_loss")
    if not hist:
        return KlSummary()
    hist = sorted(hist, key=lambda m: m.step)
    values = [float(m.value) for m in hist]
    up_steps = sum(1 for i in range(1, len(values)) if values[i] > values[i - 1])
    down_steps = sum(1 for i in range(1, len(values)) if values[i] < values[i - 1])
    first = values[0]
    last = values[-1]
    return KlSummary(
        first=first,
        last=last,
        delta=last - first,
        up_steps=up_steps,
        down_steps=down_steps,
    )


@dataclass
class RunSummary:
    run_id: str
    run_name: str
    milestone: str
    seed: str
    kl_weight: float | None
    ffn_rho: float | None
    attn_rho: float | None
    attn_prior_std: float | None
    batch_mi_ratio: float | None
    qual_mi_ratio: float | None
    qual_mi_id: float | None
    qual_mi_ood: float | None
    test_id_ppl: float | None
    test_ood_ppl: float | None
    sigma_mean: float | None
    sigma_std: float | None
    kl_final: float | None
    kl_summary: KlSummary
    pass_batch_gate: bool
    pass_qual_gate: bool
    pass_id_ppl_gate: bool
    pass_all_gates: bool
    failed_gates: str


def _build_run_summary(
    client: MlflowClient,
    run_id: str,
    batch_mi_gate: float,
    qual_mi_gate: float,
    id_ppl_gate: float,
) -> RunSummary:
    run = client.get_run(run_id)
    params = run.data.params

    mi_ratio = _safe_float(params.get("mi_ood_id_ratio"))
    if mi_ratio is None:
        mi_ood = _safe_float(params.get("mi_ood_mean"))
        mi_id = _safe_float(params.get("mi_id_mean"))
        if mi_ood is not None and mi_id is not None:
            mi_ratio = mi_ood / max(mi_id, 1e-10)

    qual_id, qual_ood, qual_ratio = _read_qual_ratio(run)
    kl_summary = _read_kl_summary(client, run_id)

    test_id_ppl = _safe_float(params.get("test_id_perplexity"))
    failed: list[str] = []

    pass_batch = mi_ratio is not None and mi_ratio >= batch_mi_gate
    pass_qual = qual_ratio is not None and qual_ratio >= qual_mi_gate
    pass_id = test_id_ppl is not None and test_id_ppl <= id_ppl_gate

    if not pass_batch:
        failed.append(f"batch_mi<{batch_mi_gate}")
    if not pass_qual:
        failed.append(f"qual_mi<{qual_mi_gate}")
    if not pass_id:
        failed.append(f"id_ppl>{id_ppl_gate}")

    return RunSummary(
        run_id=run_id,
        run_name=run.data.tags.get("mlflow.runName", ""),
        milestone=run.data.tags.get("milestone", ""),
        seed=params.get("train.seed", ""),
        kl_weight=_safe_float(params.get("train.kl_weight")),
        ffn_rho=_safe_float(params.get("model.bayes_ffn.init_rho")),
        attn_rho=_safe_float(params.get("model.bayes_attn_v.init_rho")),
        attn_prior_std=_safe_float(params.get("model.bayes_attn_v.prior_std")),
        batch_mi_ratio=mi_ratio,
        qual_mi_ratio=qual_ratio,
        qual_mi_id=qual_id,
        qual_mi_ood=qual_ood,
        test_id_ppl=test_id_ppl,
        test_ood_ppl=_safe_float(params.get("test_ood_perplexity")),
        sigma_mean=_safe_float(params.get("sigma_mean")),
        sigma_std=_safe_float(params.get("sigma_std")),
        kl_final=_safe_float(params.get("final_kl_loss")),
        kl_summary=kl_summary,
        pass_batch_gate=pass_batch,
        pass_qual_gate=pass_qual,
        pass_id_ppl_gate=pass_id,
        pass_all_gates=pass_batch and pass_qual and pass_id,
        failed_gates=";".join(failed),
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _build_rows(summaries: list[RunSummary], baseline: RunSummary | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in summaries:
        if baseline is not None:
            d_batch = (
                None if s.batch_mi_ratio is None or baseline.batch_mi_ratio is None
                else s.batch_mi_ratio - baseline.batch_mi_ratio
            )
            d_qual = (
                None if s.qual_mi_ratio is None or baseline.qual_mi_ratio is None
                else s.qual_mi_ratio - baseline.qual_mi_ratio
            )
            d_ppl = (
                None if s.test_id_ppl is None or baseline.test_id_ppl is None
                else s.test_id_ppl - baseline.test_id_ppl
            )
        else:
            d_batch = d_qual = d_ppl = None

        rows.append(
            {
                "run_id": _short_run_id(s.run_id),
                "name": s.run_name or "-",
                "milestone": s.milestone or "-",
                "seed": s.seed or "-",
                "kl_w": _fmt(s.kl_weight, 2),
                "ffn_rho": _fmt(s.ffn_rho, 1),
                "attn_rho": _fmt(s.attn_rho, 1),
                "attn_prior": _fmt(s.attn_prior_std, 2),
                "batch_mi": _fmt(s.batch_mi_ratio, 2),
                "qual_mi": _fmt(s.qual_mi_ratio, 2),
                "id_ppl": _fmt(s.test_id_ppl, 2),
                "ood_ppl": _fmt(s.test_ood_ppl, 2),
                "sigma_mean": _fmt(s.sigma_mean, 3),
                "sigma_std": _fmt(s.sigma_std, 3),
                "kl_final_m": _fmt(None if s.kl_final is None else s.kl_final / 1e6, 3),
                "kl_delta_m": _fmt(
                    None if s.kl_summary.delta is None
                    else s.kl_summary.delta / 1e6, 3
                ),
                "kl_steps": f"{s.kl_summary.down_steps}/{s.kl_summary.up_steps}",
                "d_batch": _fmt(d_batch, 2),
                "d_qual": _fmt(d_qual, 2),
                "d_id_ppl": _fmt(d_ppl, 2),
                "pass": "PASS" if s.pass_all_gates else "FAIL",
                "failed": s.failed_gates or "-",
            }
        )
    return rows


def _print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No runs to display.")
        return

    columns = [
        "run_id", "kl_w", "ffn_rho", "attn_rho", "attn_prior",
        "batch_mi", "qual_mi", "id_ppl", "d_batch", "d_qual", "d_id_ppl",
        "kl_delta_m", "pass", "failed",
    ]
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(row[c]))

    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(row[c].ljust(widths[c]) for c in columns))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MLflow runs with fixed gates.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run IDs to compare.")
    parser.add_argument("--baseline", default=None, help="Optional baseline run ID for deltas.")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--batch-mi-gate", type=float, default=1.36)
    parser.add_argument("--qual-mi-gate", type=float, default=1.55)
    parser.add_argument("--id-ppl-gate", type=float, default=58.0)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--md", type=Path, default=None, help="Optional Markdown output path.")
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit with code 1 if any run fails the gates.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    baseline_summary: RunSummary | None = None
    if args.baseline:
        baseline_summary = _build_run_summary(
            client,
            args.baseline,
            args.batch_mi_gate,
            args.qual_mi_gate,
            args.id_ppl_gate,
        )
        print(
            f"Baseline: {_short_run_id(args.baseline)} "
            f"(batch_mi={_fmt(baseline_summary.batch_mi_ratio, 2)}, "
            f"qual_mi={_fmt(baseline_summary.qual_mi_ratio, 2)}, "
            f"id_ppl={_fmt(baseline_summary.test_id_ppl, 2)})"
        )

    summaries = [
        _build_run_summary(
            client,
            run_id,
            args.batch_mi_gate,
            args.qual_mi_gate,
            args.id_ppl_gate,
        )
        for run_id in args.runs
    ]
    rows = _build_rows(summaries, baseline_summary)
    _print_table(rows)

    if args.csv is not None:
        _write_csv(args.csv, rows)
        print(f"\nSaved CSV: {args.csv}")
    if args.md is not None:
        _write_markdown(args.md, rows)
        print(f"Saved Markdown: {args.md}")

    if args.fail_on_gate and any(not s.pass_all_gates for s in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
