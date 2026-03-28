"""D2: Verify mean-weights perplexity ≈ MC-averaged perplexity (<5% gap).

Loads C1 (variational FFN) and C3 (BLoB LoRA) checkpoints on Pile test data,
computes both perplexities, verifies <5% relative difference.

Usage:
    python scripts/verify_mean_weights.py
    python scripts/verify_mean_weights.py --n-samples 30 --n-batches 50
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c_milestones import build_milestone_config
from minigpt.config import build_gpt_config, build_lora_config
from minigpt.data import get_tokenizer, load_pile_data
from minigpt.evaluate import compute_perplexity, compute_perplexity_mc
from minigpt.layers import use_mean_weights
from minigpt.lora import inject_lora
from minigpt.model import MiniGPT
from minigpt.train import load_checkpoint

CKPT_DIR = Path("data/checkpoints")


def _load_model(milestone: str, device: torch.device) -> MiniGPT:
    tokenizer = get_tokenizer()
    if milestone == "c1":
        cfg = build_milestone_config("c1")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        load_checkpoint(CKPT_DIR / "c1/ckpt_best.pt", model)
    elif milestone == "c3":
        cfg = build_milestone_config("c3_phase2")
        model = MiniGPT(build_gpt_config(cfg, vocab_size=tokenizer.n_vocab))
        model = inject_lora(model, build_lora_config(cfg), bayesian=True)
        load_checkpoint(CKPT_DIR / "c3/ckpt_best.pt", model)
    else:
        raise ValueError(f"Unsupported milestone: {milestone}")
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description="D2: verify mean-weights parity")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--n-batches", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = build_milestone_config("c0")
    tokenizer = get_tokenizer()
    data = load_pile_data(cfg, tokenizer)
    test_id = data["test_id"]
    print(f"Test ID tokens: {len(test_id):,}")

    block_size = 256
    batch_size = 8
    results = []

    for ms in ["c1", "c3"]:
        print(f"\n{'='*50}")
        print(f"Evaluating {ms.upper()}")
        print(f"{'='*50}")

        model = _load_model(ms, device)

        with use_mean_weights(model):
            ppl_mean = compute_perplexity(
                model, test_id, block_size, batch_size, device,
                n_batches=args.n_batches,
            )
        print(f"  Mean-weights PPL: {ppl_mean:.2f}")

        ppl_mc = compute_perplexity_mc(
            model, test_id, block_size, batch_size, device,
            n_samples=args.n_samples, n_batches=args.n_batches,
        )
        print(f"  MC-avg PPL (N={args.n_samples}): {ppl_mc:.2f}")

        rel_diff = abs(ppl_mean - ppl_mc) / ppl_mc * 100
        print(f"  Relative diff: {rel_diff:.2f}%")
        results.append((ms, ppl_mean, ppl_mc, rel_diff))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("D2 Verification Results")
    print(f"{'='*60}")
    print(f"{'Method':<20} | {'Mean-Wt PPL':>12} | {'MC-Avg PPL':>11} | {'Rel Diff':>8}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*11}-+-{'-'*8}")

    all_pass = True
    for ms, ppl_mean, ppl_mc, rel_diff in results:
        label = "C1 Var. FFN" if ms == "c1" else "C3 BLoB LoRA"
        status = "PASS" if rel_diff < 5.0 else "FAIL"
        print(f"{label:<20} | {ppl_mean:>12.2f} | {ppl_mc:>11.2f} | {rel_diff:>6.2f}% {status}")
        if rel_diff >= 5.0:
            all_pass = False

    print()
    if all_pass:
        print("GATE PASSED: All relative differences < 5%")
    else:
        print("GATE FAILED: Some relative differences >= 5%")
        sys.exit(1)


if __name__ == "__main__":
    main()
