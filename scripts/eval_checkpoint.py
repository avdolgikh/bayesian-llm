"""Evaluate uncertainty metrics on an arbitrary checkpoint.

Usage:
    python scripts/eval_checkpoint.py <checkpoint_path> [--config <yaml>] [--set key=value ...]

Loads model from checkpoint, runs uncertainty eval (MI on ID vs OOD), and
optionally dumps sigma statistics for the Bayesian layer.
"""

import argparse
from pathlib import Path

import torch

from minigpt.config import (
    DEFAULT_CONFIG,
    apply_overrides,
    build_gpt_config,
    deep_merge,
    load_yaml,
)
from minigpt.data import get_tokenizer, load_dataset
from minigpt.evaluate import compute_perplexity
from minigpt.layers import BayesianLinear, use_mean_weights
from minigpt.model import MiniGPT
from minigpt.train import load_checkpoint
from minigpt.uncertainty import compute_uncertainty_metrics


def _sigma_stats(model: MiniGPT) -> dict:
    """Compute sigma statistics for all BayesianLinear layers."""
    all_sigmas = []
    for name, module in model.named_modules():
        if isinstance(module, BayesianLinear):
            sigma = torch.nn.functional.softplus(module.weight_rho).detach().cpu()
            all_sigmas.append(sigma.flatten())
            print(f"\n  [{name}] weight sigma stats:")
            print(f"    shape: {list(module.weight_rho.shape)}")
            print(f"    min:    {sigma.min().item():.6f}")
            print(f"    max:    {sigma.max().item():.6f}")
            print(f"    mean:   {sigma.mean().item():.6f}")
            print(f"    median: {sigma.median().item():.6f}")
            print(f"    std:    {sigma.std().item():.6f}")

            # Percentiles
            pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            vals = torch.quantile(sigma.flatten().float(), torch.tensor([p / 100 for p in pcts]))
            print(f"    percentiles: {dict(zip(pcts, [f'{v:.6f}' for v in vals.tolist()]))}")

            if module.bias_rho is not None:
                b_sigma = torch.nn.functional.softplus(module.bias_rho).detach().cpu()
                print(
                    f"    bias sigma — min: {b_sigma.min():.6f}"
                    f"  max: {b_sigma.max():.6f}  mean: {b_sigma.mean():.6f}"
                )

    if all_sigmas:
        combined = torch.cat(all_sigmas)
        print("\n  [ALL] combined weight sigmas:")
        print(f"    total params: {combined.numel():,}")
        print(f"    mean: {combined.mean():.6f}  std: {combined.std():.6f}")
        print(f"    min: {combined.min():.6f}  max: {combined.max():.6f}")
        return {
            "mean": combined.mean().item(),
            "std": combined.std().item(),
            "min": combined.min().item(),
            "max": combined.max().item(),
            "median": combined.median().item(),
        }
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate uncertainty on a checkpoint")
    p.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    p.add_argument("--config", type=str, default=None, help="YAML config (for data/model params)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="key=value", help="Dot-notation config override")
    p.add_argument("--n-samples", type=int, default=30, help="MC samples for uncertainty")
    p.add_argument("--n-batches", type=int, default=20, help="Batches for uncertainty eval")
    p.add_argument("--no-sigma-stats", action="store_true", help="Skip sigma statistics")
    p.add_argument("--no-perplexity", action="store_true", help="Skip perplexity eval")
    args = p.parse_args()

    # --- Config ---
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        cfg = deep_merge(cfg, load_yaml(args.config))
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # --- Data ---
    tokenizer = get_tokenizer()
    data = load_dataset(cfg, tokenizer)
    test_id = data["test_id"]
    test_ood = data["test_ood"]
    print(f"Test ID tokens: {len(test_id):,}")
    if test_ood is not None:
        print(f"Test OOD tokens: {len(test_ood):,}")

    # --- Model ---
    gpt_config = build_gpt_config(cfg, vocab_size=tokenizer.n_vocab)
    model = MiniGPT(gpt_config)

    # --- Load checkpoint ---
    ckpt = load_checkpoint(ckpt_path, model)
    step = ckpt.get("step", "?")
    print(f"\nLoaded checkpoint: {ckpt_path} (step {step})")
    print(f"Bayesian head: {gpt_config.bayes.enabled}")

    # --- Device ---
    device_str = cfg["train"]["device"]
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = model.to(device)
    print(f"Device: {device_str}")

    # --- KL ---
    kl = model.kl_loss().item()
    print(f"\nKL loss: {kl:.2f}")

    # --- Sigma stats ---
    if not args.no_sigma_stats:
        print("\n=== SIGMA STATISTICS ===")
        _sigma_stats(model)

    # --- Perplexity (mean weights) ---
    if not args.no_perplexity:
        block_size = cfg["train"]["block_size"]
        batch_size = cfg["train"]["batch_size"]
        n_ppl_batches = cfg["eval"].get("n_perplexity_batches", 20)

        print("\n=== PERPLEXITY (mean weights) ===")
        with use_mean_weights(model):
            id_ppl = compute_perplexity(
                model, test_id, block_size, batch_size, device, n_batches=n_ppl_batches,
            )
            print(f"  Test ID perplexity: {id_ppl:.2f}")

            if test_ood is not None:
                ood_ppl = compute_perplexity(
                    model, test_ood, block_size, batch_size, device, n_batches=n_ppl_batches,
                )
                print(f"  Test OOD perplexity: {ood_ppl:.2f}")

    # --- Uncertainty ---
    print(f"\n=== UNCERTAINTY (N={args.n_samples} MC samples, {args.n_batches} batches) ===")
    block_size = cfg["train"]["block_size"]
    batch_size = cfg["train"]["batch_size"]

    mi_id = compute_uncertainty_metrics(
        model, test_id, block_size, batch_size, device,
        n_samples=args.n_samples, n_batches=args.n_batches,
    )
    print(f"  ID  — MI: {mi_id['mi_mean']:.4f}  "
          f"Pred H: {mi_id['predictive_entropy_mean']:.4f}  "
          f"Exp H: {mi_id['expected_entropy_mean']:.4f}  "
          f"Flip: {mi_id['flip_rate']:.4f}")

    if test_ood is not None:
        mi_ood = compute_uncertainty_metrics(
            model, test_ood, block_size, batch_size, device,
            n_samples=args.n_samples, n_batches=args.n_batches,
        )
        print(f"  OOD — MI: {mi_ood['mi_mean']:.4f}  "
              f"Pred H: {mi_ood['predictive_entropy_mean']:.4f}  "
              f"Exp H: {mi_ood['expected_entropy_mean']:.4f}  "
              f"Flip: {mi_ood['flip_rate']:.4f}")

        ratio = mi_ood["mi_mean"] / max(mi_id["mi_mean"], 1e-10)
        print(f"  MI ratio (OOD/ID): {ratio:.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
