"""P2: MC Dropout baseline evaluation.

Applies MC Dropout to the C0 deterministic checkpoint — enables dropout at
inference time and runs N forward passes to compute MI-based OOD detection.
Reports the same metrics as eval_c_checkpoints.py for direct comparison.

Usage:
    python scripts/eval_mc_dropout.py                         # default: N=20, 500 seqs
    python scripts/eval_mc_dropout.py --n-samples 5           # fewer MC passes
    python scripts/eval_mc_dropout.py --n-sequences 200       # fewer test sequences
    python scripts/eval_mc_dropout.py --bootstrap             # with 95% CIs
    python scripts/eval_mc_dropout.py --save-scores data/mc_dropout_scores.pt
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
from minigpt.config import build_gpt_config
from minigpt.data import get_tokenizer, load_pile_data
from minigpt.layers import enable_dropout
from minigpt.model import MiniGPT
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

BLOCK_SIZE = 256
CKPT_DIR = Path("data/checkpoints")
N_VALUES = [1, 3, 5, 10, 20]


# ---------------------------------------------------------------------------
# Data (shared with eval_c_checkpoints.py)
# ---------------------------------------------------------------------------

def _extract_sequences(data: torch.Tensor, block_size: int, n: int):
    max_seqs = (len(data) - 1) // block_size
    n = min(n, max_seqs)
    seqs = []
    for i in range(n):
        s = i * block_size
        seqs.append((data[s : s + block_size], data[s + 1 : s + block_size + 1]))
    return seqs


def load_eval_data(n_sequences: int):
    cfg = build_milestone_config("c0")
    tokenizer = get_tokenizer()
    data = load_pile_data(cfg, tokenizer)
    test_id = data["test_id"]
    ood_parts = [data[f"test_ood_{d}"] for d in OOD_DOMAINS if f"test_ood_{d}" in data]
    test_ood = torch.cat(ood_parts) if ood_parts else data.get("test_ood")
    id_seqs = _extract_sequences(test_id, BLOCK_SIZE, n_sequences)
    ood_seqs = _extract_sequences(test_ood, BLOCK_SIZE, n_sequences)
    print(f"Loaded {len(id_seqs)} ID + {len(ood_seqs)} OOD sequences "
          f"(block_size={BLOCK_SIZE})")
    return id_seqs, ood_seqs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_c0_model(device: torch.device):
    """Load C0 deterministic checkpoint."""
    cfg = build_milestone_config("c0")
    tokenizer = get_tokenizer()
    model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
    load_checkpoint(CKPT_DIR / "c0/ckpt_best.pt", model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded C0: {n_params:,} params, dropout={cfg['model']['dropout']}")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Per-sequence MC Dropout scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_sequence_mc_dropout(
    model: MiniGPT,
    x: torch.Tensor,
    targets: torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """MC Dropout scoring: N forward passes with dropout enabled."""
    eps = 1e-10
    x_dev = x.unsqueeze(0).to(device)
    targets_dev = targets.to(device)
    seq_len = x.size(0)
    vocab_size = model.config.vocab_size
    use_amp = device.type == "cuda"

    p_sum = torch.zeros(seq_len, vocab_size, device=device)
    entropy_sum = torch.zeros(seq_len, device=device)

    with enable_dropout(model):
        for _ in range(n_samples):
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
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_mc_dropout(
    device: torch.device,
    n_samples: int,
    id_seqs: list,
    ood_seqs: list,
) -> dict:
    """Full MC Dropout evaluation — mirrors evaluate_milestone() interface."""
    print(f"\n{'=' * 60}")
    print(f"MC Dropout (C0 checkpoint, N={n_samples})")
    print(f"{'=' * 60}")
    t0 = time.time()

    model = load_c0_model(device)

    id_scores = {"mi": [], "pred_ent": [], "max_prob_unc": []}
    ood_scores = {"mi": [], "pred_ent": [], "max_prob_unc": []}
    cal_max_probs, cal_correct, cal_p_true, cal_sum_p_sq = [], [], [], []
    sp_mi, sp_correct = [], []

    def _score_split(seqs, scores_dict, is_id):
        split_name = "ID" if is_id else "OOD"
        for i, (x, y) in enumerate(seqs):
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                print(f"  {split_name} {i + 1}/{len(seqs)} ({elapsed:.0f}s)")
            m = score_sequence_mc_dropout(model, x, y, n_samples, device)
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

    _score_split(id_seqs, id_scores, is_id=True)
    _score_split(ood_seqs, ood_scores, is_id=False)

    # OOD detection
    n_id, n_ood = len(id_scores["mi"]), len(ood_scores["mi"])
    labels = torch.cat([torch.zeros(n_id), torch.ones(n_ood)])
    results = {"mi_ratio": None}

    raw_scores = {}
    for name in ("mi", "pred_ent", "max_prob_unc"):
        scores = torch.tensor(id_scores[name] + ood_scores[name])
        raw_scores[name] = scores
        results[f"auroc_{name}"] = auroc(scores, labels)
        results[f"fpr95_{name}"] = fpr_at_tpr(scores, labels)
        results[f"auprc_{name}"] = auprc(scores, labels)
    raw_scores["labels"] = labels

    # Calibration (ID only, per-token)
    all_max_prob = torch.cat(cal_max_probs)
    all_correct = torch.cat(cal_correct)
    all_p_true = torch.cat(cal_p_true)
    all_sum_p_sq = torch.cat(cal_sum_p_sq)

    results["ece"] = ece(all_max_prob, all_correct)
    results["nll"] = float(-(all_p_true + 1e-10).log().mean())
    results["brier"] = float((1.0 - 2.0 * all_p_true + all_sum_p_sq).mean())

    # Selective prediction
    results["aurc"] = aurc(torch.cat(sp_mi), torch.cat(sp_correct))

    elapsed = time.time() - t0
    results["time_s"] = elapsed
    results["_raw_scores"] = raw_scores

    print(f"  Completed in {elapsed:.0f}s")
    print(f"  AUROC (MI): {results['auroc_mi']:.3f}  "
          f"FPR@95: {results['fpr95_mi']:.3f}  "
          f"ECE: {results['ece']:.4f}  NLL: {results['nll']:.2f}")

    return results


def evaluate_auroc_vs_n(
    device: torch.device,
    id_seqs: list,
    ood_seqs: list,
) -> list[tuple[int, float]]:
    """AUROC vs N sweep for MC Dropout."""
    print(f"\n{'=' * 60}")
    print("MC Dropout: AUROC vs N")
    print(f"{'=' * 60}")

    model = load_c0_model(device)
    results = []

    for n in N_VALUES:
        t0 = time.time()
        id_mi, ood_mi = [], []
        for i, (x, y) in enumerate(id_seqs):
            m = score_sequence_mc_dropout(model, x, y, n, device)
            id_mi.append(aggregate_sequence_scores(m["mi"]))
        for i, (x, y) in enumerate(ood_seqs):
            m = score_sequence_mc_dropout(model, x, y, n, device)
            ood_mi.append(aggregate_sequence_scores(m["mi"]))

        labels = torch.cat([torch.zeros(len(id_mi)), torch.ones(len(ood_mi))])
        scores = torch.tensor(id_mi + ood_mi)
        auc = auroc(scores, labels) if n > 1 else 0.500
        elapsed = time.time() - t0
        results.append((n, auc))
        print(f"  N={n:>2}: AUROC(MI)={auc:.3f} ({elapsed:.0f}s)")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: dict, ci: dict | None = None):
    """Print MC Dropout results in the same format as eval_c_checkpoints."""
    print("\n## MC Dropout Results\n")

    auroc_val = results["auroc_mi"]
    fpr_val = results["fpr95_mi"]
    auprc_val = results["auprc_mi"]

    if ci:
        auroc_str = f"{auroc_val:.3f} [{ci['auroc_ci'][0]:.3f}, {ci['auroc_ci'][1]:.3f}]"
        fpr_str = f"{fpr_val:.3f} [{ci['fpr95_ci'][0]:.3f}, {ci['fpr95_ci'][1]:.3f}]"
        auprc_str = f"{auprc_val:.3f} [{ci['auprc_ci'][0]:.3f}, {ci['auprc_ci'][1]:.3f}]"
    else:
        auroc_str = f"{auroc_val:.3f}"
        fpr_str = f"{fpr_val:.3f}"
        auprc_str = f"{auprc_val:.3f}"

    print("| Method       | AUROC         | FPR@95        | AUPRC         "
          "| ECE    | Brier | NLL  | AURC  |")
    print("|--------------|---------------|---------------|---------------"
          "|--------|-------|------|-------|")
    print(f"| MC Dropout   | {auroc_str:<13} | {fpr_str:<13} | {auprc_str:<13} "
          f"| {results['ece']:.4f} | {results['brier']:.3f} | {results['nll']:.2f} "
          f"| {results['aurc']:.4f} |")

    # Also print all uncertainty score AUROCs
    print("\n| Score Type     | AUROC |")
    print("|----------------|-------|")
    print(f"| MI             | {results['auroc_mi']:.3f} |")
    print(f"| Pred. Entropy  | {results['auroc_pred_ent']:.3f} |")
    print(f"| Max-Prob       | {results['auroc_max_prob_unc']:.3f} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="P2: MC Dropout baseline evaluation")
    p.add_argument("--n-samples", type=int, default=20,
                   help="MC forward passes (default: 20)")
    p.add_argument("--n-sequences", type=int, default=500,
                   help="Test sequences per split (default: 500)")
    p.add_argument("--bootstrap", action="store_true",
                   help="Compute 95%% bootstrap CIs")
    p.add_argument("--n-bootstrap", type=int, default=10_000,
                   help="Bootstrap resamples (default: 10000)")
    p.add_argument("--save-scores", type=str, default=None,
                   help="Save per-sequence scores to .pt file")
    p.add_argument("--auroc-vs-n", action="store_true",
                   help="Run AUROC vs N sweep (Table 4 equivalent)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = CKPT_DIR / "c0/ckpt_best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: C0 checkpoint not found at {ckpt_path}")
        return

    id_seqs, ood_seqs = load_eval_data(args.n_sequences)

    # Main evaluation
    results = evaluate_mc_dropout(device, args.n_samples, id_seqs, ood_seqs)

    # Save scores
    if args.save_scores:
        torch.save({"mc_dropout": results["_raw_scores"]}, Path(args.save_scores))
        print(f"Saved scores to {args.save_scores}")

    # Bootstrap CIs
    ci = None
    if args.bootstrap:
        raw = results["_raw_scores"]
        labels = raw["labels"]
        scores = raw["mi"]
        print(f"\nBootstrap CIs ({args.n_bootstrap} resamples)...")
        _, lo, hi = bootstrap_ci(scores, labels, auroc,
                                 n_bootstrap=args.n_bootstrap, seed=42)
        _, fpr_lo, fpr_hi = bootstrap_ci(scores, labels, fpr_at_tpr,
                                         n_bootstrap=args.n_bootstrap, seed=42)
        _, auprc_lo, auprc_hi = bootstrap_ci(scores, labels, auprc,
                                             n_bootstrap=args.n_bootstrap, seed=42)
        ci = {
            "auroc_ci": (lo, hi),
            "fpr95_ci": (fpr_lo, fpr_hi),
            "auprc_ci": (auprc_lo, auprc_hi),
        }

    print_results(results, ci)

    # AUROC vs N sweep
    if args.auroc_vs_n:
        n_results = evaluate_auroc_vs_n(device, id_seqs[:200], ood_seqs[:200])
        print("\n## MC Dropout: AUROC vs N\n")
        print("| N  | AUROC |")
        print("|----|-------|")
        for n, auc in n_results:
            print(f"| {n:>2} | {auc:.3f} |")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
