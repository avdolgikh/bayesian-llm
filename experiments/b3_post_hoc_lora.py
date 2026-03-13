"""B3: Post-hoc Bayesianization of LoRA (TFB and Laplace-LoRA).

Three-phase pipeline:
1. Train a deterministic LoRA baseline.
2. Fit post-hoc posteriors (TFB and Laplace) from the same checkpoint.
3. Evaluate MI and epistemic uncertainty.

Usage:
  # Full pipeline (train + TFB + Laplace):
  python experiments/b3_post_hoc_lora.py --phase full \\
      --config configs/b3_lora_agnews.yaml

  # Phase 1 only (deterministic LoRA training):
  python experiments/b3_post_hoc_lora.py --phase train \\
      --config configs/b3_lora_agnews.yaml

  # TFB only (requires trained LoRA checkpoint):
  python experiments/b3_post_hoc_lora.py --phase tfb \\
      --config configs/b3_lora_agnews.yaml

  # Laplace only (requires trained LoRA checkpoint):
  python experiments/b3_post_hoc_lora.py --phase laplace \\
      --config configs/b3_lora_agnews.yaml
"""

import time
from functools import partial
from pathlib import Path

import torch
from eval_utils import (
    eval_perplexity_suite,
    run_qualitative_suite,
)
from experiment_setup import (
    parse_base_args,
    resolve_device,
    setup_data,
    setup_model,
)
from mlflow_utils import (
    log_common_mlflow,
    log_qualitative_mlflow,
    mlflow_context,
)

from minigpt.config import (
    build_lora_config,
    build_train_config,
)
from minigpt.laplace import (
    compute_laplace_uncertainty,
    fit_laplace,
    save_laplace_state,
    score_sequence_laplace,
    select_params,
)
from minigpt.lora import inject_lora
from minigpt.tfb import (
    compute_tfb_uncertainty,
    fit_tfb,
    save_tfb_state,
    score_sequence_tfb,
)
from minigpt.train import load_checkpoint, train


def _load_base_and_inject_lora(cfg, model, device):
    """Load base pretrain checkpoint, then inject deterministic LoRA.

    Must load base BEFORE injection — LoRA wrapping renames FFN param keys
    (fc.linear.weight -> fc.base_linear.weight), so a post-injection load
    with strict=False silently drops FFN weights.
    """
    base_ckpt_path = cfg.get("lora", {}).get("base_checkpoint")
    if base_ckpt_path and Path(base_ckpt_path).exists():
        print(f"Loading base model from {base_ckpt_path}")
        load_checkpoint(Path(base_ckpt_path), model)
    else:
        print("Warning: No base checkpoint found. Training from scratch.")

    lora_cfg = build_lora_config(cfg)
    model = inject_lora(model, lora_cfg, bayesian=False)

    n_lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora_a_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and "lora_A" in name
    )
    print(f"LoRA injected: rank={lora_cfg.rank}, alpha={lora_cfg.alpha}, "
          f"target={lora_cfg.target}")
    print(f"LoRA params: {n_lora_params:,} total, {n_lora_a_params:,} in A matrices")

    model.to(device)
    return model, n_lora_params, n_lora_a_params


def _load_lora_checkpoint(model, lora_ckpt, device):
    """Load a trained LoRA checkpoint into an already-injected model."""
    print(f"Loading LoRA checkpoint from {lora_ckpt}")
    ckpt = torch.load(lora_ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("model"))
    model.load_state_dict(sd)


def train_phase(cfg, model, tokenizer, datasets, device, use_mlflow,
                n_lora_params, n_lora_a_params):
    """Phase 1: Train deterministic LoRA."""
    train_cfg = build_train_config(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    with mlflow_context(cfg, use_mlflow) as run:
        if use_mlflow:
            log_common_mlflow(
                cfg, tokenizer, n_params, "b3-train",
                n_lora_params=n_lora_params,
                n_lora_a_params=n_lora_a_params,
            )

        model, train_meta = train(
            model, datasets["train"], datasets["val"], train_cfg,
            mlflow_run=run if use_mlflow else None, config_dict=cfg,
        )

        print(f"\nTraining time: {train_meta['train_time_sec']:.1f}s")
        print(f"Best val loss: {train_meta['best_val_loss']:.4f} "
              f"(step {train_meta['best_val_step']})")

        # Reload best checkpoint for final eval
        best_ckpt_path = Path(train_cfg.checkpoint_dir) / "ckpt_best.pt"
        if best_ckpt_path.exists():
            print(f"Loading best checkpoint for final eval: {best_ckpt_path}")
            load_checkpoint(best_ckpt_path, model)

        eval_cfg = cfg["eval"]
        n_ppl_batches = eval_cfg.get("n_perplexity_batches", 20)
        results = eval_perplexity_suite(
            model, cfg, datasets["test_id"], datasets["test_ood"],
            device, n_ppl_batches,
        )
        if use_mlflow:
            import mlflow
            mlflow.log_metrics({k: v for k, v in results.items() if v is not None})

        print(f"Phase 1 done. Test ID ppl: {results['test_id_ppl']:.2f}, "
              f"OOD ppl: {results['test_ood_ppl']:.2f}")


def tfb_phase(cfg, model, tokenizer, datasets, device, use_mlflow):
    """Phase 2a: Fit TFB and evaluate."""
    tfb_cfg = cfg["tfb"]
    train_cfg = build_train_config(cfg)
    eval_cfg = cfg["eval"]
    n_params = sum(p.numel() for p in model.parameters())

    with mlflow_context(cfg, use_mlflow):
        if use_mlflow:
            import mlflow
            log_common_mlflow(cfg, tokenizer, n_params, "b3-tfb")

        start_time = time.time()
        state = fit_tfb(
            model,
            datasets["train"],
            block_size=train_cfg.block_size,
            batch_size=train_cfg.batch_size,
            n_batches=tfb_cfg["n_anchor_batches"],
            epsilon=tfb_cfg["epsilon"],
            n_search_samples=tfb_cfg["n_search_samples"],
            search_range=(tfb_cfg["search_min"], tfb_cfg["search_max"]),
            search_precision=tfb_cfg["search_precision"],
        )
        duration = time.time() - start_time

        print(f"TFB fit done in {duration:.1f}s. sigma_q={state.sigma_q:.6f}")

        if use_mlflow:
            import mlflow
            mlflow.log_params({
                "tfb.sigma_q": f"{state.sigma_q:.6f}",
                "tfb.anchor_loss": f"{state.anchor_loss:.4f}",
                "tfb.fit_time_sec": f"{duration:.1f}",
            })

        # Save TFB state
        save_dir = Path(train_cfg.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "tfb_state.pt"
        save_tfb_state(state, save_path)
        if use_mlflow:
            import mlflow
            mlflow.log_artifact(str(save_path))

        # MC Evaluation
        n_samples = eval_cfg.get("num_samples", 20)
        n_eval_batches = eval_cfg.get("n_perplexity_batches", 20)

        print(f"Running TFB MC uncertainty evaluation (N={n_samples})...")
        id_unc = compute_tfb_uncertainty(
            model, datasets["test_id"], train_cfg.block_size,
            train_cfg.batch_size, device, state,
            n_samples=n_samples, n_batches=n_eval_batches,
        )
        ood_unc = compute_tfb_uncertainty(
            model, datasets["test_ood"], train_cfg.block_size,
            train_cfg.batch_size, device, state,
            n_samples=n_samples, n_batches=n_eval_batches,
        )
        mi_ratio = ood_unc["mi_mean"] / max(id_unc["mi_mean"], 1e-9)

        print(f"  ID  MI: {id_unc['mi_mean']:.4f}  "
              f"Flip: {id_unc['flip_rate']:.4f}")
        print(f"  OOD MI: {ood_unc['mi_mean']:.4f}  "
              f"Flip: {ood_unc['flip_rate']:.4f}")
        print(f"  MI Ratio: {mi_ratio:.2f}x")

        if use_mlflow:
            import mlflow
            mlflow.log_metrics({f"id_{k}": v for k, v in id_unc.items()})
            mlflow.log_metrics({f"ood_{k}": v for k, v in ood_unc.items()})
            mlflow.log_metric("mi_ratio", mi_ratio)

        # Qualitative panel
        print("Running TFB qualitative evaluation...")
        tfb_score_fn = partial(score_sequence_tfb, state=state)
        report, qual_results = run_qualitative_suite(
            model, tokenizer, cfg, device, n_samples=n_samples,
            score_fn=tfb_score_fn, generate=False,
        )
        print(report)
        if use_mlflow:
            log_qualitative_mlflow(report, qual_results)


def laplace_phase(cfg, model, tokenizer, datasets, device, use_mlflow):
    """Phase 2b: Fit Laplace-LoRA and evaluate."""
    lap_cfg = cfg["laplace"]
    train_cfg = build_train_config(cfg)
    eval_cfg = cfg["eval"]
    n_params = sum(p.numel() for p in model.parameters())

    with mlflow_context(cfg, use_mlflow):
        if use_mlflow:
            import mlflow
            log_common_mlflow(cfg, tokenizer, n_params, "b3-laplace")

        selection = select_params(model, lap_cfg["selection_mode"])
        print(f"Laplace: selected {sum(p.numel() for p in selection.values()):,} "
              f"params ({len(selection)} tensors)")

        start_time = time.time()
        state = fit_laplace(
            model,
            datasets["train"],
            block_size=train_cfg.block_size,
            batch_size=train_cfg.batch_size,
            selection=selection,
            n_batches=lap_cfg["n_curvature_batches"],
            damping=lap_cfg["damping"],
            sample_scale=lap_cfg["sample_scale"],
        )
        duration = time.time() - start_time

        # Log curvature stats
        curv_vals = torch.cat([v.flatten() for v in state.curvature.values()])
        print(f"Laplace fit done in {duration:.1f}s. "
              f"Curvature: mean={curv_vals.mean():.6f}, "
              f"max={curv_vals.max():.6f}")

        if use_mlflow:
            import mlflow
            mlflow.log_metrics({
                "laplace.curv_mean": curv_vals.mean().item(),
                "laplace.curv_std": curv_vals.std().item(),
                "laplace.curv_max": curv_vals.max().item(),
                "laplace.fit_time_sec": duration,
                "laplace.num_params": len(curv_vals),
            })

        # Save Laplace state
        save_dir = Path(train_cfg.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "laplace_state_lora.pt"
        save_laplace_state(state, save_path)
        if use_mlflow:
            import mlflow
            mlflow.log_artifact(str(save_path))

        # MC Evaluation
        n_samples = eval_cfg.get("num_samples", 20)
        n_eval_batches = eval_cfg.get("n_perplexity_batches", 20)

        print(f"Running Laplace MC uncertainty evaluation (N={n_samples})...")
        id_unc = compute_laplace_uncertainty(
            model, datasets["test_id"], train_cfg.block_size,
            train_cfg.batch_size, device, state,
            n_samples=n_samples, n_batches=n_eval_batches,
        )
        ood_unc = compute_laplace_uncertainty(
            model, datasets["test_ood"], train_cfg.block_size,
            train_cfg.batch_size, device, state,
            n_samples=n_samples, n_batches=n_eval_batches,
        )
        mi_ratio = ood_unc["mi_mean"] / max(id_unc["mi_mean"], 1e-9)

        print(f"  ID  MI: {id_unc['mi_mean']:.4f}  "
              f"Flip: {id_unc['flip_rate']:.4f}")
        print(f"  OOD MI: {ood_unc['mi_mean']:.4f}  "
              f"Flip: {ood_unc['flip_rate']:.4f}")
        print(f"  MI Ratio: {mi_ratio:.2f}x")

        if use_mlflow:
            import mlflow
            mlflow.log_metrics({f"id_{k}": v for k, v in id_unc.items()})
            mlflow.log_metrics({f"ood_{k}": v for k, v in ood_unc.items()})
            mlflow.log_metric("mi_ratio", mi_ratio)

        # Qualitative panel
        print("Running Laplace qualitative evaluation...")
        laplace_score_fn = partial(score_sequence_laplace, state=state)
        report, qual_results = run_qualitative_suite(
            model, tokenizer, cfg, device, n_samples=n_samples,
            score_fn=laplace_score_fn, generate=False,
        )
        print(report)
        if use_mlflow:
            log_qualitative_mlflow(report, qual_results)


def main() -> None:
    def add_b3_args(ap):
        ap.add_argument(
            "--phase", choices=["train", "tfb", "laplace", "full"],
            default="full",
            help="Phase(s) to run: train, tfb, laplace, or full (all)",
        )
        ap.add_argument(
            "--lora-checkpoint", dest="lora_checkpoint", type=str, default=None,
            help="Path to trained LoRA checkpoint (for tfb/laplace phases)",
        )
        ap.add_argument(
            "--base-checkpoint", dest="base_checkpoint", type=str, default=None,
            help="Override lora.base_checkpoint for train phase",
        )

    args, cfg = parse_base_args(
        "B3: Post-hoc Bayesianization of LoRA (TFB and Laplace-LoRA)",
        add_b3_args,
    )
    phase = args.phase
    use_mlflow = not args.no_mlflow

    # Override base checkpoint from CLI if given
    if args.base_checkpoint:
        cfg.setdefault("lora", {})["base_checkpoint"] = args.base_checkpoint

    device = resolve_device(cfg)
    tokenizer, datasets = setup_data(cfg)
    model, _, n_params = setup_model(cfg, tokenizer)

    # --- Phase 1: Train deterministic LoRA ---
    if phase in ("train", "full"):
        # Load base checkpoint BEFORE LoRA injection (C1 fix)
        model, n_lora_params, n_lora_a_params = _load_base_and_inject_lora(
            cfg, model, device,
        )
        train_phase(
            cfg, model, tokenizer, datasets, device, use_mlflow,
            n_lora_params, n_lora_a_params,
        )

    # --- Phase 2a: TFB ---
    if phase in ("tfb", "full"):
        # For standalone tfb/laplace, need to inject LoRA and load checkpoint
        if phase != "full":
            lora_cfg = build_lora_config(cfg)
            model = inject_lora(model, lora_cfg, bayesian=False)
            model.to(device)

        lora_ckpt = args.lora_checkpoint
        if not lora_ckpt:
            lora_ckpt = str(Path(cfg["train"]["checkpoint_dir"]) / "ckpt_best.pt")
        _load_lora_checkpoint(model, lora_ckpt, device)

        tfb_phase(cfg, model, tokenizer, datasets, device, use_mlflow)

    # --- Phase 2b: Laplace ---
    if phase in ("laplace", "full"):
        if phase == "laplace":
            lora_cfg = build_lora_config(cfg)
            model = inject_lora(model, lora_cfg, bayesian=False)
            model.to(device)

        # For full phase, model is already loaded from tfb_phase above.
        # For standalone laplace, load checkpoint.
        lora_ckpt = args.lora_checkpoint
        if not lora_ckpt:
            lora_ckpt = str(Path(cfg["train"]["checkpoint_dir"]) / "ckpt_best.pt")
        _load_lora_checkpoint(model, lora_ckpt, device)

        laplace_phase(cfg, model, tokenizer, datasets, device, use_mlflow)

    print("\nDone.")


if __name__ == "__main__":
    main()
