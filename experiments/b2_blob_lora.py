"""B2 — BLoB Bayesian LoRA on MiniGPT.

Two-phase pipeline:
  Phase 1 (pretrain): Train deterministic miniGPT on AG News (single category).
  Phase 2 (finetune): Inject BLoB LoRA into FFN layers, fine-tune on
                      a different AG News category with ELBO objective.

Usage:
  # Full pipeline (Phase 1 then Phase 2):
  python experiments/b2_blob_lora.py --phase full \\
      --pretrain-config configs/b2_pretrain_agnews.yaml \\
      --config configs/b2_blob_agnews.yaml

  # Phase 1 only:
  python experiments/b2_blob_lora.py --phase pretrain \\
      --pretrain-config configs/b2_pretrain_agnews.yaml

  # Phase 2 only (base checkpoint path from config or --base-checkpoint):
  python experiments/b2_blob_lora.py --phase finetune \\
      --config configs/b2_blob_agnews.yaml
"""

from pathlib import Path

import mlflow
import torch
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

from minigpt.config import (
    DEFAULT_CONFIG,
    build_lora_config,
    build_train_config,
    deep_merge,
    load_yaml,
    validate_config,
)
from minigpt.evaluate import generate_text
from minigpt.layers import sigma_summary, use_mean_weights
from minigpt.lora import inject_lora
from minigpt.train import load_checkpoint, train
from minigpt.uncertainty import compute_uncertainty_metrics


def _run_phase1(pretrain_cfg: dict, use_mlflow: bool) -> Path:
    """Train deterministic base model. Returns best checkpoint path."""
    print("\n" + "=" * 60)
    print("Phase 1: Pretrain deterministic base model")
    print("=" * 60)

    torch.manual_seed(pretrain_cfg["train"]["seed"])
    tokenizer, data = setup_data(pretrain_cfg)
    model, _, n_params = setup_model(pretrain_cfg, tokenizer)
    device = resolve_device(pretrain_cfg)
    train_cfg = build_train_config(pretrain_cfg)

    with mlflow_context(pretrain_cfg, use_mlflow) as run:
        if use_mlflow:
            log_common_mlflow(pretrain_cfg, tokenizer, n_params, "b2-pretrain")

        model, train_meta = train(
            model, data["train"], data["val"], train_cfg,
            mlflow_run=run if use_mlflow else None,
            config_dict=pretrain_cfg,
        )

        print(f"\nTraining time: {train_meta['train_time_sec']:.1f}s")
        print(f"Best val loss: {train_meta['best_val_loss']:.4f} "
              f"(step {train_meta['best_val_step']})")

        eval_cfg = pretrain_cfg["eval"]
        n_ppl_batches = eval_cfg.get("n_perplexity_batches", 20)
        ppl_results = eval_perplexity_suite(
            model, pretrain_cfg, data["test_id"], data["test_ood"],
            device, n_ppl_batches, val_data=data["val"],
        )

        if use_mlflow:
            log_train_meta_mlflow(train_meta)
            log_perplexity_mlflow(ppl_results)

    ckpt_dir = Path(pretrain_cfg["train"].get("checkpoint_dir", "data/checkpoints"))
    best_path = ckpt_dir / "ckpt_best.pt"
    print(f"\nPhase 1 complete. Checkpoint: {best_path}")
    return best_path


def _run_phase2(cfg: dict, base_ckpt_path: str, use_mlflow: bool) -> None:
    """Inject BLoB LoRA, fine-tune on AG News, evaluate."""
    print("\n" + "=" * 60)
    print("Phase 2: BLoB LoRA fine-tune on AG News")
    print("=" * 60)

    tokenizer, data = setup_data(cfg)
    model, _, n_params = setup_model(cfg, tokenizer)

    # Load pretrained base
    ckpt_path = Path(base_ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
    print(f"Loading base checkpoint: {ckpt_path}")
    load_checkpoint(ckpt_path, model)

    # Inject BLoB LoRA adapters
    lora_cfg = build_lora_config(cfg)
    model = inject_lora(model, lora_cfg)
    print(f"LoRA injected: rank={lora_cfg.rank}, alpha={lora_cfg.alpha}, "
          f"target={lora_cfg.target}")

    n_lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_bayes_lora_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and ("lora_A_mu" in name or "lora_A_g" in name)
    )
    print(f"LoRA params: {n_lora_params:,} total, {n_bayes_lora_params:,} Bayesian")

    device = resolve_device(cfg)
    train_cfg = build_train_config(cfg)
    kl_weight = float(cfg["train"]["kl_weight"])

    with mlflow_context(cfg, use_mlflow) as run:
        if use_mlflow:
            log_common_mlflow(
                cfg, tokenizer, n_params, "b2",
                n_lora_params=n_lora_params,
                n_bayes_lora_params=n_bayes_lora_params,
            )
            # mlflow.log_params({
            #     "lora.rank": str(lora_cfg.rank),
            #     "lora.alpha": str(lora_cfg.alpha),
            #     "lora.prior_std": str(lora_cfg.prior_std),
            #     "lora.init_g": str(lora_cfg.init_g),
            #     "lora.base_checkpoint": str(ckpt_path),
            # })

        model, train_meta = train(
            model, data["train"], data["val"], train_cfg,
            mlflow_run=run if use_mlflow else None,
            config_dict=cfg,
            kl_weight=kl_weight,
            num_train_tokens=len(data["train"]),
        )

        print(f"\nTraining time: {train_meta['train_time_sec']:.1f}s")
        print(f"Best val loss: {train_meta['best_val_loss']:.4f} "
              f"(step {train_meta['best_val_step']})")
        if use_mlflow:
            log_train_meta_mlflow(train_meta)

        # LoRA sigma (G^2) statistics
        stats = sigma_summary(model)
        if stats:
            print(f"\nLoRA sigma (G^2) stats: "
                  f"mean={stats['sigma_mean']:.6f} std={stats['sigma_std']:.6f} "
                  f"min={stats['sigma_min']:.6f} max={stats['sigma_max']:.6f}")
            if use_mlflow:
                mlflow.log_params({k: f"{v:.6f}" for k, v in stats.items()})

        eval_cfg = cfg["eval"]
        n_ppl_batches = eval_cfg.get("n_perplexity_batches", 20)

        # Perplexity (mean weights — no sampling)
        with use_mean_weights(model):
            ppl_results = eval_perplexity_suite(
                model, cfg, data["test_id"], data["test_ood"],
                device, n_ppl_batches, val_data=data["val"],
            )
        if use_mlflow:
            log_perplexity_mlflow(ppl_results)

        # Generated sample (mean weights — deterministic)
        with use_mean_weights(model):
            sample = generate_text(
                model, tokenizer,
                max_new_tokens=eval_cfg.get("sample_tokens", 200),
                temperature=eval_cfg.get("temperature", 0.8),
            )
        print(f"\n=== generated sample (mean weights) ===\n{sample}\n{'=' * 40}")
        if use_mlflow:
            mlflow.log_text(sample, "generated_sample.txt")

        # Uncertainty (MI via MC sampling)
        n_samples = eval_cfg.get("num_samples", 20)
        print(f"\nUncertainty evaluation (N={n_samples} MC samples)...")
        mi_id, mi_ood, mi_ratio = eval_mi_suite(
            compute_uncertainty_metrics, model, cfg,
            data["test_id"], data["test_ood"], device,
            n_samples, n_ppl_batches,
        )
        if use_mlflow:
            log_mi_mlflow(mi_id, mi_ood, mi_ratio)

        # Qualitative
        print("\nQualitative evaluation...")
        report, qual_results = run_qualitative_suite(
            model, tokenizer, cfg, device, n_samples,
        )
        print(report)
        if use_mlflow:
            log_qualitative_mlflow(report, qual_results)

        # Final KL
        final_kl = model.kl_loss().item()
        print(f"\nFinal KL loss: {final_kl:.2f}")
        if use_mlflow:
            mlflow.log_param("final_kl_loss", f"{final_kl:.0f}")

        # Summary tag
        if use_mlflow:
            m = cfg["model"]
            summary = (
                f"b2-blob, {m['n_layer']}L/{m['n_head']}H/{m['n_embd']}d, "
                f"agnews, lora_r={lora_cfg.rank}, kl_weight={kl_weight}, "
                f"best_val={train_meta['best_val_loss']:.4f}, "
                f"test_id_ppl={ppl_results['test_id_ppl']:.2f}"
            )
            if mi_id is not None:
                summary += f", mi_id={mi_id['mi_mean']:.4f}"
            if mi_ood is not None:
                summary += f", mi_ratio={mi_ratio:.2f}x"
            mlflow.set_tag("mlflow.note.content", summary)


def main() -> None:
    def add_b2_args(ap):
        ap.add_argument(
            "--phase", choices=["pretrain", "finetune", "full"], default="full",
            help="Phase(s) to run: pretrain, finetune, or full (both)",
        )
        ap.add_argument(
            "--pretrain-config", dest="pretrain_config", type=str, default=None,
            help="Path to Phase 1 config (required for --phase pretrain/full)",
        )
        ap.add_argument(
            "--base-checkpoint", dest="base_checkpoint", type=str, default=None,
            help="Override lora.base_checkpoint for Phase 2",
        )

    args, cfg = parse_base_args("B2 — BLoB Bayesian LoRA", add_b2_args)
    phase = args.phase
    use_mlflow = not args.no_mlflow

    # --- Phase 1: Pretrain on TinyShakespeare ---
    phase1_ckpt = None
    if phase in ("pretrain", "full"):
        if args.pretrain_config is None:
            raise ValueError(
                "--pretrain-config is required for --phase pretrain/full\n"
                "  e.g.: --pretrain-config ../configs/b2_pretrain_shakespeare.yaml"
            )
        pretrain_cfg = deep_merge(DEFAULT_CONFIG.copy(), load_yaml(args.pretrain_config))
        validate_config(pretrain_cfg)
        phase1_ckpt = _run_phase1(pretrain_cfg, use_mlflow)
        if phase == "pretrain":
            print("\nDone (Phase 1 only).")
            return

    # --- Phase 2: BLoB LoRA fine-tune on AG News ---
    if phase in ("finetune", "full") and args.config is None:
        raise ValueError(
            "--config is required for --phase finetune/full\n"
            "  e.g.: --config configs/b2_blob_agnews.yaml"
        )
    if phase in ("finetune", "full"):
        if args.base_checkpoint:
            base_ckpt = args.base_checkpoint
        elif phase1_ckpt is not None:
            base_ckpt = str(phase1_ckpt)
        else:
            base_ckpt = cfg.get("lora", {}).get("base_checkpoint")
            if not base_ckpt:
                raise ValueError(
                    "No base checkpoint. Either:\n"
                    "  - Set lora.base_checkpoint in config\n"
                    "  - Pass --base-checkpoint <path>\n"
                    "  - Use --phase full to run Phase 1 first"
                )
        _run_phase2(cfg, base_ckpt, use_mlflow)

    print("\nDone.")


if __name__ == "__main__":
    main()
