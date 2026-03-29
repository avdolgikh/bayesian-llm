"""Evaluate C checkpoints with D1 metrics + bootstrap CIs.

Runs AUROC, FPR@95, AUPRC, ECE, Brier, NLL, AURC on all 6 C checkpoints.
Outputs publication-ready markdown tables. Optionally saves per-sequence
scores and computes bootstrap 95% confidence intervals.

Usage:
    python scripts/eval_c_checkpoints.py                    # eval all 6
    python scripts/eval_c_checkpoints.py --milestone c3     # eval one
    python scripts/eval_c_checkpoints.py --n-samples 10     # fewer MC passes
    python scripts/eval_c_checkpoints.py --n-sequences 200  # fewer test sequences
    python scripts/eval_c_checkpoints.py --bootstrap        # with 95% CIs
    python scripts/eval_c_checkpoints.py --from-scores data/d1_scores.pt  # CIs from saved scores
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c_milestones import OOD_DOMAINS, build_milestone_config
from minigpt.config import build_gpt_config, build_lora_config
from minigpt.data import get_tokenizer, load_pile_data
from minigpt.laplace import (
    apply_sampled_params,
    load_laplace_state,
    sample_laplace_params,
)
from minigpt.lora import inject_lora
from minigpt.model import MiniGPT
from minigpt.tfb import load_tfb_state, sample_tfb_params
from minigpt.train import load_checkpoint
from minigpt.uncertainty import (
    aggregate_sequence_scores,
    auprc,
    aurc,
    auroc,
    bootstrap_ci,
    ece,
    fpr_at_tpr,
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

MI_RATIOS = {
    "c0": None,
    "c1": 1.32,
    "c2": 1.00,
    "c3": 1.53,
    "c4_tfb": 1.35,
    "c4_lap": 1.00,
}


# ---------------------------------------------------------------------------
# Data utilities
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


def load_eval_data(n_sequences: int):
    """Load Pile test data using C0's ID/OOD split. Returns (id_seqs, ood_seqs)."""
    cfg = build_milestone_config("c0")
    tokenizer = get_tokenizer()
    data = load_pile_data(cfg, tokenizer)

    test_id = data["test_id"]
    ood_parts = [data[f"test_ood_{d}"] for d in OOD_DOMAINS if f"test_ood_{d}" in data]
    test_ood = torch.cat(ood_parts) if ood_parts else data.get("test_ood")

    id_seqs = _extract_sequences(test_id, BLOCK_SIZE, n_sequences)
    ood_seqs = _extract_sequences(test_ood, BLOCK_SIZE, n_sequences)
    print(f"Loaded {len(id_seqs)} ID seqs, {len(ood_seqs)} OOD seqs (block_size={BLOCK_SIZE})")
    return id_seqs, ood_seqs


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
# Per-sequence scoring
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

    Returns dict with CPU tensors of shape (seq_len,):
        mi, predictive_entropy, max_prob, correct, p_true, sum_p_sq
    """
    eps = 1e-10
    x_dev = x.unsqueeze(0).to(device)
    targets_dev = targets.to(device)
    seq_len = x.size(0)
    vocab_size = model.config.vocab_size
    use_amp = device.type == "cuda"

    p_sum = torch.zeros(seq_len, vocab_size, device=device)
    entropy_sum = torch.zeros(seq_len, device=device)

    for s in range(n_samples):
        if method in ("laplace", "tfb"):
            seed = seq_idx * n_samples + s
            sampled = (sample_laplace_params(state, seed=seed) if method == "laplace"
                       else sample_tfb_params(state, seed=seed))
            with apply_sampled_params(model, sampled), torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp,
            ):
                logits, _ = model(x_dev)
        else:
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp,
            ):
                logits, _ = model(x_dev)

        probs = F.softmax(logits[0].float(), dim=-1)
        p_sum.add_(probs)
        entropy_sum.add_(-(probs * torch.log(probs + eps)).sum(dim=-1))

    p_bar = p_sum / n_samples
    pred_entropy = -(p_bar * torch.log(p_bar + eps)).sum(dim=-1)
    exp_entropy = entropy_sum / n_samples
    mi = pred_entropy - exp_entropy
    max_prob = p_bar.max(dim=-1).values

    correct = (p_bar.argmax(dim=-1) == targets_dev).float()
    p_true = p_bar[torch.arange(seq_len, device=device), targets_dev]
    sum_p_sq = (p_bar ** 2).sum(dim=-1)

    return {
        "mi": mi.cpu(),
        "predictive_entropy": pred_entropy.cpu(),
        "max_prob": max_prob.cpu(),
        "correct": correct.cpu(),
        "p_true": p_true.cpu(),
        "sum_p_sq": sum_p_sq.cpu(),
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
    results = {"mi_ratio": MI_RATIOS[milestone]}

    # Save raw per-sequence scores for bootstrap
    raw_scores = {}
    for name in ("mi", "pred_ent", "max_prob_unc"):
        scores = torch.tensor(id_scores[name] + ood_scores[name])
        raw_scores[name] = scores
        results[f"auroc_{name}"] = auroc(scores, labels)
        results[f"fpr95_{name}"] = fpr_at_tpr(scores, labels)
        results[f"auprc_{name}"] = auprc(scores, labels)
    raw_scores["labels"] = labels

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
        r = {"mi_ratio": MI_RATIOS[m], "_raw_scores": raw}
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
    """Primary table: MI ratio + all metrics."""
    has_ci = cis is not None
    print("\n## Primary Results Table\n")
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
        mi_ratio = f"{r['mi_ratio']:.2f}x" if r["mi_ratio"] else "--"
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

def main():
    p = argparse.ArgumentParser(description="Evaluate C checkpoints (D1)")
    p.add_argument(
        "--milestone", type=str, default=None,
        help="Eval single milestone (c0/c1/c2/c3/c4_tfb/c4_lap)",
    )
    p.add_argument(
        "--n-samples", type=int, default=20,
        help="MC forward passes per sequence (default: 20)",
    )
    p.add_argument(
        "--n-sequences", type=int, default=500,
        help="Test sequences per split (default: 500)",
    )
    p.add_argument(
        "--bootstrap", action="store_true",
        help="Compute 95%% bootstrap CIs for AUROC, FPR@95, AUPRC",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=10_000,
        help="Bootstrap resamples (default: 10000)",
    )
    p.add_argument(
        "--save-scores", type=str, default=None,
        help="Save per-sequence scores to .pt file",
    )
    p.add_argument(
        "--from-scores", type=str, default=None,
        help="Load saved scores and compute tables/CIs (no GPU needed)",
    )
    args = p.parse_args()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
