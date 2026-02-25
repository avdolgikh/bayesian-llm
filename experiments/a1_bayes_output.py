"""A1 — Bayesian output head on AG News (BPE).

Trains miniGPT with BayesianLinear lm_head (ELBO loss), evaluates perplexity
and epistemic uncertainty (MI) on ID vs OOD splits, runs qualitative prompt panel.
"""

import argparse
import json
import random
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
from minigpt.model import MiniGPT
from minigpt.train import load_checkpoint, train
from minigpt.uncertainty import compute_uncertainty_metrics, score_sequence

# --- Curated prompts for qualitative evaluation ---

def _select_prompts(
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
            # Truncate to roughly 50 tokens worth of text
            words = text.split()[:60]
            prompts.append({
                "category": CATEGORY_NAMES[cat],
                "category_id": cat,
                "text": " ".join(words),
            })
    return prompts


def _run_qualitative_eval(
    model: MiniGPT,
    tokenizer: tiktoken.Encoding,
    prompts: list[dict],
    id_categories: list[int],
    device: torch.device,
    n_samples: int = 30,
    max_new_tokens: int = 100,
) -> str:
    """Generate + score prompts, return formatted text report."""
    lines = ["=" * 70, "QUALITATIVE EVALUATION — Curated Prompt Panel", "=" * 70, ""]
    results = []

    for p in prompts:
        split = "ID" if p["category_id"] in id_categories else "OOD"
        tokens = tokenizer.encode_ordinary(p["text"])
        # Truncate to fit block_size with room for generation
        max_prompt_len = model.config.block_size - max_new_tokens
        tokens = tokens[:max_prompt_len]

        # Generate with mean weights (deterministic)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        generated_ids = model.generate(idx, max_new_tokens=max_new_tokens, use_mean=True)
        continuation_ids = generated_ids[0, len(tokens):].tolist()
        continuation = tokenizer.decode(continuation_ids, errors="replace")

        # Score the full generated sequence for uncertainty
        full_ids = generated_ids[0]
        # Only score if sequence is long enough
        if len(full_ids) > 1:
            metrics = score_sequence(model, full_ids, device, n_samples=n_samples)
            # MI on the continuation part only
            cont_mi = metrics["mi"][len(tokens):].mean().item()
            cont_flip = metrics["flip_rate"][len(tokens):].mean().item()
            cont_pred_ent = metrics["predictive_entropy"][len(tokens):].mean().item()
        else:
            cont_mi = 0.0
            cont_flip = 0.0
            cont_pred_ent = 0.0

        prompt_text = tokenizer.decode(tokens, errors="replace")
        lines.append(f"[{split} — {p['category']}] \"{prompt_text[:100]}...\"")
        lines.append(f"  -> \"{continuation[:150]}...\"")
        lines.append(
            f"  -> MI: {cont_mi:.4f}  |  Flip rate: {cont_flip:.3f}"
            f"  |  Pred entropy: {cont_pred_ent:.3f}"
        )
        lines.append("")

        results.append({
            "category": p["category"],
            "split": split,
            "sequence_mi": cont_mi,
            "flip_rate": cont_flip,
            "predictive_entropy": cont_pred_ent,
        })

    # Summary
    id_mis = [r["sequence_mi"] for r in results if r["split"] == "ID"]
    ood_mis = [r["sequence_mi"] for r in results if r["split"] == "OOD"]
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


def main() -> None:
    p = argparse.ArgumentParser(description="A1 Bayesian output head")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="key=value", help="Dot-notation config override (repeatable)")
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    p.add_argument("--log-model", action="store_true",
                   help="Log model artifact + checkpoint to MLflow (heavy, off by default)")
    args = p.parse_args()

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
    # TODO: rename to ~ "bayes_weight_mu", etc. ??
    n_bayes_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "weight_mu" in name or "weight_rho" in name
        or "bias_mu" in name or "bias_rho" in name
    )
    print(f"Model parameters: {n_params:,} (Bayesian: {n_bayes_params:,})")
    print(f"Bayesian head: {gpt_config.bayes.enabled}")

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
    bayes_cfg = gpt_config.bayes
    num_train_tokens = len(train_data)

    # --- MLflow ---
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri(cfg["experiment"].get("mlflow_uri", "sqlite:///mlflow.db"))
        mlflow.set_experiment(cfg["experiment"]["name"])

    run_ctx = (
        mlflow.start_run(run_name=cfg["experiment"]["run_name"])
        if use_mlflow
        else _nullcontext()
    )
    with run_ctx as run:
        if use_mlflow:
            flat = config_to_flat_params(cfg)
            flat["vocab_size"] = str(tokenizer.n_vocab)
            flat["n_params"] = str(n_params)
            flat["n_bayes_params"] = str(n_bayes_params)
            flat["tokenizer"] = "gpt2-bpe"
            mlflow.log_params(flat)

        # --- Tags ---
        if use_mlflow:
            mlflow.set_tag("dataset", cfg["data"]["dataset"])
            mlflow.set_tag("milestone", "a1")
            if torch.cuda.is_available():
                mlflow.set_tag("gpu", torch.cuda.get_device_name())

        # --- Train ---
        model, train_meta = train(
            model, train_data, val_data, train_cfg,
            mlflow_run=run if use_mlflow else None,
            config_dict=cfg,
            resume_ckpt=resume_ckpt,
            kl_weight=bayes_cfg.kl_weight,
            num_train_tokens=num_train_tokens,
        )

        print(f"\nTraining time: {train_meta['train_time_sec']:.1f}s")
        print(f"Tokens/sec: {train_meta['tokens_per_sec']:.0f}")
        best_loss = train_meta['best_val_loss']
        best_step = train_meta['best_val_step']
        print(f"Best val loss: {best_loss:.4f} (step {best_step})")

        if use_mlflow:
            mlflow.log_params({
                "best_val_loss": f"{train_meta['best_val_loss']:.4f}",
                "best_val_step": str(int(train_meta["best_val_step"])),
                "train_time_sec": f"{train_meta['train_time_sec']:.1f}",
                "tokens_per_sec": f"{train_meta['tokens_per_sec']:.0f}",
            })

        # --- Perplexity evaluation (with mean weights for fair comparison to A0) ---
        from minigpt.layers import use_mean_weights

        eval_cfg = cfg["eval"]
        n_ppl_batches = eval_cfg.get("n_perplexity_batches", 20)

        with use_mean_weights(model):
            # Val perplexity
            val_ppl = compute_perplexity(
                model, val_data, cfg["train"]["block_size"],
                cfg["train"]["batch_size"], device, n_batches=n_ppl_batches,
            )
            print(f"\nVal perplexity (mean weights): {val_ppl:.2f}")

            # Test ID perplexity
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
        n_samples = cfg["eval"].get("num_samples", 30)
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
        prompts = _select_prompts(
            agnews_samples, id_cats + ood_cats,
            n_per_category=eval_cfg.get("qualitative_prompts_per_category", 5),
            seed=eval_cfg.get("qualitative_seed", 42),
        )
        report, qual_results = _run_qualitative_eval(
            model, tokenizer, prompts, id_cats, device,
            n_samples=n_samples,
            max_new_tokens=eval_cfg.get("qualitative_max_new_tokens", 100),
        )
        print(report)

        if use_mlflow:
            mlflow.log_text(report, "qualitative_eval.txt")
            mlflow.log_text(json.dumps(qual_results, indent=2), "qualitative_metrics.json")

        # --- Log model + checkpoint to MLflow (opt-in) ---
        if use_mlflow and args.log_model:
            mlflow.pytorch.log_model(model, "model")
            ckpt_dir = cfg["train"].get("checkpoint_dir", "data/checkpoints")
            best_ckpt = Path(ckpt_dir) / "ckpt_best.pt"
            if best_ckpt.exists():
                mlflow.log_artifact(str(best_ckpt))

        # --- Final KL ---
        final_kl = model.kl_loss().item()
        print(f"\nFinal KL loss: {final_kl:.2f}")
        if use_mlflow:
            mlflow.log_param("final_kl_loss", f"{final_kl:.0f}")

        # --- Run summary ---
        if use_mlflow:
            summary = (
                f"{cfg['experiment']['name']}, "
                f"{cfg['model']['n_layer']}L/{cfg['model']['n_head']}H/{cfg['model']['n_embd']}d, "
                f"{cfg['data']['dataset']}, bayes_head=True, "
                f"best_val={train_meta['best_val_loss']:.4f}, "
                f"test_id_ppl={test_id_ppl:.2f}, "
                f"mi_id={mi_id['mi_mean']:.4f}"
            )
            if test_ood is not None:
                summary += f", mi_ood={mi_ood['mi_mean']:.4f}, mi_ratio={mi_ratio:.2f}x"
            mlflow.set_tag("mlflow.note.content", summary)

    print("\nDone.")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
