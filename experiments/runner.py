"""Shared runner for Bayesian experiment milestones (A1, A2, …).

Handles: argparse, config, data, model, training, sigma stats,
perplexity, uncertainty (MI), qualitative eval, MLflow logging.
"""

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import mlflow
import tiktoken
import torch

from minigpt.config import (
    DEFAULT_CONFIG,
    apply_overrides,
    build_gpt_config,
    build_train_config,
    config_to_flat_params,
    deep_merge,
    load_yaml,
    validate_config,
)
from minigpt.data import CATEGORY_NAMES, get_tokenizer, load_agnews, load_dataset
from minigpt.evaluate import compute_perplexity, generate_text
from minigpt.layers import sigma_summary, use_mean_weights
from minigpt.model import MiniGPT
from minigpt.train import load_checkpoint, train
from minigpt.uncertainty import compute_uncertainty_metrics, score_sequence

# ---------------------------------------------------------------------------
# Qualitative eval helpers
# ---------------------------------------------------------------------------


def select_prompts(
    samples: list[tuple[int, str, str]],
    categories: list[int],
    n_per_category: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Pick n article openings per category for qualitative eval."""
    rng = random.Random(seed)
    prompts = []
    for cat in categories:
        cat_samples = [(title, desc) for c, title, desc in samples if c == cat]
        rng.shuffle(cat_samples)
        for title, desc in cat_samples[:n_per_category]:
            text = f"{title} {desc}"
            words = text.split()[:60]
            prompts.append({
                "category": CATEGORY_NAMES[cat],
                "category_id": cat,
                "text": " ".join(words),
            })
    return prompts


def run_qualitative_eval(
    model: MiniGPT,
    tokenizer: tiktoken.Encoding,
    prompts: list[dict],
    id_categories: list[int],
    device: torch.device,
    n_samples: int = 20,
    max_new_tokens: int = 100,
) -> tuple[str, list[dict]]:
    """Generate + score prompts, return (text report, list of result dicts)."""
    lines = ["=" * 70, "QUALITATIVE EVALUATION — Curated Prompt Panel", "=" * 70, ""]
    results = []

    for p in prompts:
        split = "ID" if p["category_id"] in id_categories else "OOD"
        tokens = tokenizer.encode_ordinary(p["text"])
        max_prompt_len = model.config.block_size - max_new_tokens
        tokens = tokens[:max_prompt_len]

        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        generated_ids = model.generate(idx, max_new_tokens=max_new_tokens, use_mean=True)
        continuation_ids = generated_ids[0, len(tokens):].tolist()
        continuation = tokenizer.decode(continuation_ids, errors="replace")

        prompt_ids = idx[0]
        if len(prompt_ids) > 1:
            metrics = score_sequence(model, prompt_ids, device, n_samples=n_samples)
            prompt_mi = metrics["mi"].mean().item()
            prompt_flip = metrics["flip_rate"].mean().item()
            prompt_pred_ent = metrics["predictive_entropy"].mean().item()
        else:
            prompt_mi = 0.0
            prompt_flip = 0.0
            prompt_pred_ent = 0.0

        prompt_text = tokenizer.decode(tokens, errors="replace")
        lines.append(f"[{split} — {p['category']}] \"{prompt_text[:100]}...\"")
        lines.append(f"  -> \"{continuation[:150]}...\"")
        lines.append(
            f"  -> MI: {prompt_mi:.4f}  |  Flip rate: {prompt_flip:.3f}"
            f"  |  Pred entropy: {prompt_pred_ent:.3f}"
        )
        lines.append("")

        results.append({
            "category": p["category"],
            "split": split,
            "prompt_mi": prompt_mi,
            "flip_rate": prompt_flip,
            "predictive_entropy": prompt_pred_ent,
        })

    id_mis = [r["prompt_mi"] for r in results if r["split"] == "ID"]
    ood_mis = [r["prompt_mi"] for r in results if r["split"] == "OOD"]
    if id_mis and ood_mis:
        avg_id = sum(id_mis) / len(id_mis)
        avg_ood = sum(ood_mis) / len(ood_mis)
        lines.append("-" * 40)
        ratio = avg_ood / max(avg_id, 1e-10)
        lines.append(
            f"Avg MI — ID: {avg_id:.4f}  OOD: {avg_ood:.4f}  Ratio: {ratio:.2f}x"
        )
        lines.append("")

    return "\n".join(lines), results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_kl_weight(cfg: dict) -> float:
    """Global KL weight is defined in train config."""
    return float(cfg["train"]["kl_weight"])


def _bayes_summary(gpt_config) -> str:
    """Build a short Bayesian-status string for the MLflow summary tag."""
    parts = []
    if gpt_config.bayes_head.enabled:
        parts.append("bayes_head=True")
    if gpt_config.bayes_ffn.enabled:
        parts.append("bayes_ffn=True")
    if gpt_config.bayes_attn_v.enabled:
        parts.append("bayes_attn_v=True")
    return ", ".join(parts) if parts else "deterministic"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_experiment(milestone: str, description: str = "Bayesian experiment") -> None:
    """Run a full Bayesian training + evaluation pipeline.

    Each milestone script (a1, a2, …) calls this with its own tag.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="key=value", help="Dot-notation config override (repeatable)")
    ap.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = ap.parse_args()

    # --- Build config: defaults → YAML → CLI overrides ---
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        yaml_cfg = load_yaml(args.config)
        cfg = deep_merge(cfg, yaml_cfg)
    if args.overrides:
        apply_overrides(cfg, args.overrides)
    validate_config(cfg)

    torch.manual_seed(cfg["train"]["seed"])

    # --- Data ---
    tokenizer = get_tokenizer()
    data = load_dataset(cfg, tokenizer)
    train_data = data["train"]
    val_data = data["val"]
    test_id = data["test_id"]
    test_ood = data["test_ood"]
    print(f"BPE vocab size: {tokenizer.n_vocab}")
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")
    print(f"Test ID tokens: {len(test_id):,}")
    if test_ood is not None:
        print(f"Test OOD tokens: {len(test_ood):,}")

    # --- Model ---
    gpt_config = build_gpt_config(cfg, vocab_size=tokenizer.n_vocab)
    model = MiniGPT(gpt_config)
    n_params = sum(p.numel() for p in model.parameters())
    n_bayes_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "weight_mu" in name or "weight_rho" in name
        or "bias_mu" in name or "bias_rho" in name
    )
    print(f"Model parameters: {n_params:,} (Bayesian: {n_bayes_params:,})")
    print(f"Bayesian head: {gpt_config.bayes_head.enabled}")
    print(f"Bayesian FFN: {gpt_config.bayes_ffn.enabled}")
    print(f"Bayesian attention V: {gpt_config.bayes_attn_v.enabled}")

    # --- Device ---
    device_str = cfg["train"]["device"]
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    # --- Resume ---
    resume_ckpt = None
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        resume_ckpt = load_checkpoint(ckpt_path, model)
        print(f"Loaded checkpoint: {ckpt_path} (step {resume_ckpt['step']})")

    # --- Train config ---
    train_cfg = build_train_config(cfg)
    kl_weight = _resolve_kl_weight(cfg)
    num_train_tokens = len(train_data)

    # --- MLflow ---
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri(cfg["experiment"].get("mlflow_uri", "sqlite:///mlflow.db"))
        mlflow.set_experiment(cfg["experiment"]["name"])

    run_ctx = (
        mlflow.start_run(run_name=cfg["experiment"]["run_name"])
        if use_mlflow
        else nullcontext()
    )
    with run_ctx as run:
        if use_mlflow:
            flat = config_to_flat_params(cfg)
            flat["vocab_size"] = str(tokenizer.n_vocab)
            flat["n_params"] = str(n_params)
            flat["n_bayes_params"] = str(n_bayes_params)
            flat["tokenizer"] = "gpt2-bpe"
            mlflow.log_params(flat)

        if use_mlflow:
            mlflow.set_tag("dataset", cfg["data"]["dataset"])
            mlflow.set_tag("milestone", milestone)
            if torch.cuda.is_available():
                mlflow.set_tag("gpu", torch.cuda.get_device_name())

        # --- Train ---
        model, train_meta = train(
            model, train_data, val_data, train_cfg,
            mlflow_run=run if use_mlflow else None,
            config_dict=cfg,
            resume_ckpt=resume_ckpt,
            kl_weight=kl_weight,
            num_train_tokens=num_train_tokens,
        )

        print(f"\nTraining time: {train_meta['train_time_sec']:.1f}s")
        print(f"Tokens/sec: {train_meta['tokens_per_sec']:.0f}")
        print(f"Best val loss: {train_meta['best_val_loss']:.4f} "
              f"(step {train_meta['best_val_step']})")

        if use_mlflow:
            mlflow.log_params({
                "best_val_loss": f"{train_meta['best_val_loss']:.4f}",
                "best_val_step": str(int(train_meta["best_val_step"])),
                "train_time_sec": f"{train_meta['train_time_sec']:.1f}",
                "tokens_per_sec": f"{train_meta['tokens_per_sec']:.0f}",
            })

        # --- Sigma statistics ---
        stats = sigma_summary(model)
        if stats:
            print(f"\nSigma stats: mean={stats['sigma_mean']:.4f} "
                  f"std={stats['sigma_std']:.4f} "
                  f"min={stats['sigma_min']:.4f} max={stats['sigma_max']:.4f}")
            if use_mlflow:
                mlflow.log_params({k: f"{v:.6f}" for k, v in stats.items()})

        # --- Perplexity (mean weights for fair comparison to A0) ---
        eval_cfg = cfg["eval"]
        n_ppl_batches = eval_cfg.get("n_perplexity_batches", 20)

        with use_mean_weights(model):
            val_ppl = compute_perplexity(
                model, val_data, cfg["train"]["block_size"],
                cfg["train"]["batch_size"], device, n_batches=n_ppl_batches,
            )
            print(f"\nVal perplexity (mean weights): {val_ppl:.2f}")

            test_id_ppl = compute_perplexity(
                model, test_id, cfg["train"]["block_size"],
                cfg["train"]["batch_size"], device, n_batches=n_ppl_batches,
            )
            print(f"Test ID perplexity (mean weights): {test_id_ppl:.2f}")

        if use_mlflow:
            mlflow.log_param("final_val_perplexity", f"{val_ppl:.2f}")
            mlflow.log_param("test_id_perplexity", f"{test_id_ppl:.2f}")

        # --- OOD perplexity ---
        ood_ppl = None
        if test_ood is not None:
            with use_mean_weights(model):
                ood_ppl = compute_perplexity(
                    model, test_ood, cfg["train"]["block_size"],
                    cfg["train"]["batch_size"], device, n_batches=n_ppl_batches,
                )
            print(f"Test OOD perplexity (mean weights): {ood_ppl:.2f}")
            print(f"ID vs OOD perplexity: {test_id_ppl:.2f} vs {ood_ppl:.2f}")
            if use_mlflow:
                mlflow.log_param("test_ood_perplexity", f"{ood_ppl:.2f}")

        # --- Generate sample (mean weights) ---
        sample = generate_text(
            model, tokenizer,
            max_new_tokens=eval_cfg.get("sample_tokens", 200),
            temperature=eval_cfg.get("temperature", 0.8),
        )
        print(f"\n=== generated sample (mean weights) ===\n{sample}\n{'=' * 40}")
        if use_mlflow:
            mlflow.log_text(sample, "generated_sample.txt")

        # --- Uncertainty evaluation ---
        n_samples = eval_cfg.get("num_samples", 20)
        print(f"\nUncertainty evaluation (N={n_samples} MC samples)...")

        mi_id = compute_uncertainty_metrics(
            model, test_id, cfg["train"]["block_size"],
            cfg["train"]["batch_size"], device, n_samples=n_samples,
        )
        print(f"  ID — MI: {mi_id['mi_mean']:.4f}  "
              f"Pred entropy: {mi_id['predictive_entropy_mean']:.4f}  "
              f"Exp entropy: {mi_id['expected_entropy_mean']:.4f}  "
              f"Flip rate: {mi_id['flip_rate']:.4f}")

        if use_mlflow:
            mlflow.log_params({
                "mi_id_mean": f"{mi_id['mi_mean']:.6f}",
                "pred_entropy_id_mean": f"{mi_id['predictive_entropy_mean']:.4f}",
                "expected_entropy_id_mean": f"{mi_id['expected_entropy_mean']:.4f}",
                "flip_rate_id": f"{mi_id['flip_rate']:.4f}",
            })

        mi_ood = None
        mi_ratio = None
        if test_ood is not None:
            mi_ood = compute_uncertainty_metrics(
                model, test_ood, cfg["train"]["block_size"],
                cfg["train"]["batch_size"], device, n_samples=n_samples,
            )
            print(f"  OOD — MI: {mi_ood['mi_mean']:.4f}  "
                  f"Pred entropy: {mi_ood['predictive_entropy_mean']:.4f}  "
                  f"Exp entropy: {mi_ood['expected_entropy_mean']:.4f}  "
                  f"Flip rate: {mi_ood['flip_rate']:.4f}")

            mi_ratio = mi_ood["mi_mean"] / max(mi_id["mi_mean"], 1e-10)
            print(f"  MI ratio (OOD/ID): {mi_ratio:.2f}x")

            if use_mlflow:
                mlflow.log_params({
                    "mi_ood_mean": f"{mi_ood['mi_mean']:.6f}",
                    "pred_entropy_ood_mean": f"{mi_ood['predictive_entropy_mean']:.4f}",
                    "expected_entropy_ood_mean": f"{mi_ood['expected_entropy_mean']:.4f}",
                    "flip_rate_ood": f"{mi_ood['flip_rate']:.4f}",
                    "mi_ood_id_ratio": f"{mi_ratio:.2f}",
                })

        # --- Qualitative evaluation ---
        print("\nQualitative evaluation...")
        agnews_samples = load_agnews()
        id_cats = cfg["data"]["id_categories"]
        ood_cats = cfg["data"]["ood_categories"]
        prompts = select_prompts(
            agnews_samples, id_cats + ood_cats,
            n_per_category=eval_cfg.get("qualitative_prompts_per_category", 5),
            seed=eval_cfg.get("qualitative_seed", 42),
        )
        report, qual_results = run_qualitative_eval(
            model, tokenizer, prompts, id_cats, device,
            n_samples=n_samples,
            max_new_tokens=eval_cfg.get("qualitative_max_new_tokens", 100),
        )
        id_mis = [r["prompt_mi"] for r in qual_results if r["split"] == "ID"]
        ood_mis = [r["prompt_mi"] for r in qual_results if r["split"] == "OOD"]
        if id_mis and ood_mis:
            avg_id = sum(id_mis) / len(id_mis)
            avg_ood = sum(ood_mis) / len(ood_mis)
            ratio = avg_ood / max(avg_id, 1e-10)
            print(f"  Qualitative MI — ID: {avg_id:.4f}  OOD: {avg_ood:.4f}  "
                  f"Ratio: {ratio:.2f}x")

        if use_mlflow:
            mlflow.log_text(report, "qualitative_eval.txt")
            mlflow.log_text(json.dumps(qual_results, indent=2), "qualitative_metrics.json")

        # --- Final KL ---
        final_kl = model.kl_loss().item()
        print(f"\nFinal KL loss: {final_kl:.2f}")
        if use_mlflow:
            mlflow.log_param("final_kl_loss", f"{final_kl:.0f}")

        # --- Run summary ---
        if use_mlflow:
            bayes_str = _bayes_summary(gpt_config)
            summary = (
                f"{cfg['experiment']['name']}, "
                f"{cfg['model']['n_layer']}L/{cfg['model']['n_head']}H/"
                f"{cfg['model']['n_embd']}d, "
                f"{cfg['data']['dataset']}, {bayes_str}, "
                f"best_val={train_meta['best_val_loss']:.4f}, "
                f"test_id_ppl={test_id_ppl:.2f}, "
                f"mi_id={mi_id['mi_mean']:.4f}"
            )
            if mi_ood is not None:
                summary += (f", mi_ood={mi_ood['mi_mean']:.4f}, "
                            f"mi_ratio={mi_ratio:.2f}x")
            mlflow.set_tag("mlflow.note.content", summary)

    print("\nDone.")
