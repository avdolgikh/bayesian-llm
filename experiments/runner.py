"""A-series orchestrator for Bayesian experiment milestones (A1, A2, A3)."""

from pathlib import Path

import mlflow
from eval_utils import eval_mi_suite, eval_perplexity_suite, run_qualitative_suite
from experiment_setup import parse_base_args, resolve_device, setup_data, setup_model
from mlflow_utils import (
    log_common_mlflow,
    log_mi_mlflow,
    log_perplexity_mlflow,
    log_qualitative_mlflow,
    log_train_meta_mlflow,
    mlflow_context,
)

from minigpt.config import build_train_config
from minigpt.evaluate import generate_text
from minigpt.layers import sigma_summary, use_mean_weights
from minigpt.train import load_checkpoint, train
from minigpt.uncertainty import compute_uncertainty_metrics


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


def run_experiment(milestone: str, description: str = "Bayesian experiment") -> None:
    """Run a full Bayesian training + evaluation pipeline.

    Each milestone script (a1, a2, ...) calls this with its own tag.
    """
    def add_resume_arg(ap):
        ap.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args, cfg = parse_base_args(description, add_extra_args_fn=add_resume_arg)
    tokenizer, data = setup_data(cfg)
    model, gpt_config, n_params = setup_model(cfg, tokenizer)
    device = resolve_device(cfg)

    # A-series: Bayesian param info
    n_bayes_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "weight_mu" in name or "weight_rho" in name
        or "bias_mu" in name or "bias_rho" in name
    )
    print(f"  (Bayesian: {n_bayes_params:,})")
    print(f"Bayesian head: {gpt_config.bayes_head.enabled}")
    print(f"Bayesian FFN: {gpt_config.bayes_ffn.enabled}")
    print(f"Bayesian attention V: {gpt_config.bayes_attn_v.enabled}")

    # Resume
    resume_ckpt = None
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        resume_ckpt = load_checkpoint(ckpt_path, model)
        print(f"Loaded checkpoint: {ckpt_path} (step {resume_ckpt['step']})")

    # Train config
    train_cfg = build_train_config(cfg)
    kl_weight = _resolve_kl_weight(cfg)

    # MLflow
    use_mlflow = not args.no_mlflow
    with mlflow_context(cfg, use_mlflow) as run:
        if use_mlflow:
            log_common_mlflow(
                cfg, tokenizer, n_params, milestone,
                n_bayes_params=n_bayes_params,
            )

        # --- Train ---
        model, train_meta = train(
            model, data["train"], data["val"], train_cfg,
            mlflow_run=run if use_mlflow else None,
            config_dict=cfg,
            resume_ckpt=resume_ckpt,
            kl_weight=kl_weight,
            num_train_tokens=len(data["train"]),
        )

        print(f"\nTraining time: {train_meta['train_time_sec']:.1f}s")
        print(f"Tokens/sec: {train_meta['tokens_per_sec']:.0f}")
        print(f"Best val loss: {train_meta['best_val_loss']:.4f} "
              f"(step {train_meta['best_val_step']})")

        if use_mlflow:
            log_train_meta_mlflow(train_meta)
            mlflow.log_param(
                "tokens_per_sec", f"{train_meta['tokens_per_sec']:.0f}",
            )

        # --- Sigma statistics ---
        stats = sigma_summary(model)
        if stats:
            print(f"\nSigma stats: mean={stats['sigma_mean']:.4f} "
                  f"std={stats['sigma_std']:.4f} "
                  f"min={stats['sigma_min']:.4f} max={stats['sigma_max']:.4f}")
            if use_mlflow:
                mlflow.log_params({k: f"{v:.6f}" for k, v in stats.items()})

        # --- Perplexity (mean weights) ---
        eval_cfg = cfg["eval"]
        n_ppl_batches = eval_cfg.get("n_perplexity_batches", 20)
        with use_mean_weights(model):
            ppl_results = eval_perplexity_suite(
                model, cfg, data["test_id"], data["test_ood"], device,
                n_ppl_batches, val_data=data["val"],
            )
        if use_mlflow:
            log_perplexity_mlflow(ppl_results)

        # --- Generate sample ---
        sample = generate_text(
            model, tokenizer,
            max_new_tokens=eval_cfg.get("sample_tokens", 200),
            temperature=eval_cfg.get("temperature", 0.8),
        )
        print(f"\n=== generated sample (mean weights) ===\n{sample}\n{'=' * 40}")
        if use_mlflow:
            mlflow.log_text(sample, "generated_sample.txt")

        # --- Uncertainty (MI) ---
        n_samples = eval_cfg.get("num_samples", 20)
        print(f"\nUncertainty evaluation (N={n_samples} MC samples)...")
        mi_id, mi_ood, mi_ratio = eval_mi_suite(
            compute_uncertainty_metrics, model, cfg,
            data["test_id"], data["test_ood"], device,
            n_samples, n_ppl_batches,
        )
        if use_mlflow:
            log_mi_mlflow(mi_id, mi_ood, mi_ratio)
            mlflow.log_params({
                "pred_entropy_id_mean":
                    f"{mi_id['predictive_entropy_mean']:.4f}",
                "expected_entropy_id_mean":
                    f"{mi_id['expected_entropy_mean']:.4f}",
            })
            if mi_ood is not None:
                mlflow.log_params({
                    "pred_entropy_ood_mean":
                        f"{mi_ood['predictive_entropy_mean']:.4f}",
                    "expected_entropy_ood_mean":
                        f"{mi_ood['expected_entropy_mean']:.4f}",
                })

        # --- Qualitative ---
        print("\nQualitative evaluation...")
        report, qual_results = run_qualitative_suite(
            model, tokenizer, cfg, device, n_samples,
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
            log_qualitative_mlflow(report, qual_results)

        # --- Final KL ---
        final_kl = model.kl_loss().item()
        print(f"\nFinal KL loss: {final_kl:.2f}")
        if use_mlflow:
            mlflow.log_param("final_kl_loss", f"{final_kl:.0f}")

        # --- Run summary tag ---
        if use_mlflow:
            bayes_str = _bayes_summary(gpt_config)
            summary = (
                f"{cfg['experiment']['name']}, "
                f"{cfg['model']['n_layer']}L/{cfg['model']['n_head']}H/"
                f"{cfg['model']['n_embd']}d, "
                f"{cfg['data']['dataset']}, {bayes_str}, "
                f"best_val={train_meta['best_val_loss']:.4f}, "
                f"test_id_ppl={ppl_results['test_id_ppl']:.2f}, "
                f"mi_id={mi_id['mi_mean']:.4f}"
            )
            if mi_ood is not None:
                summary += (f", mi_ood={mi_ood['mi_mean']:.4f}, "
                            f"mi_ratio={mi_ratio:.2f}x")
            mlflow.set_tag("mlflow.note.content", summary)

    print("\nDone.")
