"""The D1 tables of scripts/eval_c_checkpoints.py: score files, bootstrap CIs, printing."""

from __future__ import annotations

from pathlib import Path

import torch
from eval_c_load import ALL_MILESTONES, LABELS

from minigpt.uncertainty import auprc, auroc, bootstrap_ci, fpr_at_tpr


def legacy_mi_ratio(raw: dict) -> float | None:
    """Descriptive mean MI(OOD) / mean MI(ID) from a D1 score dict (None if MI(ID) is 0)."""
    mi = torch.as_tensor(raw["mi"]).double()
    labels = torch.as_tensor(raw["labels"])
    id_mean = float(mi[labels == 0].mean())
    if id_mean == 0.0:
        return None
    return float(mi[labels == 1].mean()) / id_mean


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
