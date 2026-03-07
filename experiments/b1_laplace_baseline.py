"""B1 — Post-hoc Laplace baseline on deterministic miniGPT checkpoint.

Pipeline:
1. Train deterministic model (or load checkpoint)
2. Fit Laplace curvature on selected params
3. Evaluate perplexity and MI (ID/OOD) with posterior sampling
4. Run qualitative MI panel
5. Log all metrics to MLflow
"""

import time
from functools import partial
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

from minigpt.config import build_train_config
from minigpt.laplace import (
    compute_laplace_uncertainty,
    fit_laplace,
    load_laplace_state,
    save_laplace_state,
    score_sequence_laplace,
    select_params,
)
from minigpt.train import load_checkpoint, train


def main() -> None:
    def add_b1_args(ap):
        ap.add_argument("--laplace-state", type=str, default=None,
                        help="Path to pre-fitted Laplace state (skip fitting)")
        ap.add_argument("--skip-train", action="store_true",
                        help="Skip training (requires laplace.checkpoint)")

    args, cfg = parse_base_args("B1 — Laplace baseline", add_b1_args)
    tokenizer, data = setup_data(cfg)
    model, gpt_config, n_params = setup_model(cfg, tokenizer)
    device = resolve_device(cfg)

    # --- Laplace config ---
    lap_cfg = cfg.get("laplace", {})
    selection_mode = lap_cfg.get("selection_mode", "ffn")
    damping = lap_cfg.get("damping", 1.0)
    sample_scale = lap_cfg.get("sample_scale", 1.0)
    n_curv_batches = lap_cfg.get("n_curvature_batches", 100)

    # --- Phase 1: Train or load checkpoint ---
    ckpt_path = lap_cfg.get("checkpoint")
    train_meta = None

    if args.skip_train and ckpt_path:
        print(f"\nLoading checkpoint: {ckpt_path}")
        ckpt = load_checkpoint(Path(ckpt_path), model)
        print(f"Loaded step {ckpt['step']}")
    elif args.skip_train:
        raise ValueError("--skip-train requires laplace.checkpoint in config")
    else:
        print("\n--- Phase 1: Deterministic training ---")
        train_cfg = build_train_config(cfg)
        model, train_meta = train(
            model, data["train"], data["val"], train_cfg,
            mlflow_run=None, config_dict=cfg,
        )
        print(f"Training time: {train_meta['train_time_sec']:.1f}s")
        print(f"Best val loss: {train_meta['best_val_loss']:.4f} "
              f"(step {train_meta['best_val_step']})")

    model.to(device)
    model.eval()

    # --- Phase 2: Fit Laplace ---
    print(f"\n--- Phase 2: Laplace fit (mode={selection_mode}, "
          f"damping={damping}, batches={n_curv_batches}) ---")

    if args.laplace_state:
        print(f"Loading pre-fitted Laplace state: {args.laplace_state}")
        state = load_laplace_state(args.laplace_state)
        # Override damping/sample_scale from config (allows tuning without re-fitting)
        state.damping = damping
        state.sample_scale = sample_scale
        for name in state.param_names:
            state.phi_hat[name] = state.phi_hat[name].to(device)
            state.curvature[name] = state.curvature[name].to(device)
    else:
        selected = select_params(model, mode=selection_mode)
        n_selected = sum(p.numel() for p in selected.values())
        print(f"Selected {len(selected)} params ({n_selected:,} elements)")

        fit_start = time.time()
        state = fit_laplace(
            model, data["train"].to(device),
            block_size=cfg["train"]["block_size"],
            batch_size=cfg["train"]["batch_size"],
            selection=selected,
            n_batches=n_curv_batches,
            damping=damping,
            sample_scale=sample_scale,
        )
        print(f"Laplace fit time: {time.time() - fit_start:.1f}s")

        ckpt_dir = Path(cfg["train"].get("checkpoint_dir", "data/checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state_path = ckpt_dir / "laplace_state.pt"
        save_laplace_state(state, state_path)
        print(f"Saved Laplace state: {state_path}")

    # Curvature summary
    curv_stats = {}
    all_curv = torch.cat(
        [state.curvature[n].flatten() for n in state.param_names],
    )
    curv_stats["curv_mean"] = all_curv.mean().item()
    curv_stats["curv_std"] = all_curv.std().item()
    curv_stats["curv_min"] = all_curv.min().item()
    curv_stats["curv_max"] = all_curv.max().item()
    print(f"Curvature: mean={curv_stats['curv_mean']:.6f} "
          f"std={curv_stats['curv_std']:.6f} "
          f"min={curv_stats['curv_min']:.6f} max={curv_stats['curv_max']:.6f}")

    # --- Phase 3: Evaluate ---
    print("\n--- Phase 3: Evaluation ---")
    eval_cfg = cfg["eval"]
    n_ppl_batches = eval_cfg.get("n_perplexity_batches", 20)
    n_samples = eval_cfg.get("num_samples", 30)

    ppl_results = eval_perplexity_suite(
        model, cfg, data["test_id"], data["test_ood"], device, n_ppl_batches,
    )

    print(f"\nUncertainty evaluation (N={n_samples} Laplace MC samples)...")
    eval_start = time.time()
    mi_id, mi_ood, mi_ratio = eval_mi_suite(
        compute_laplace_uncertainty, model, cfg,
        data["test_id"], data["test_ood"], device,
        n_samples, n_ppl_batches, state=state,
    )
    eval_time = time.time() - eval_start
    print(f"Uncertainty eval time: {eval_time:.1f}s")

    print("\nQualitative evaluation...")
    laplace_score_fn = partial(score_sequence_laplace, state=state)
    report, qual_results = run_qualitative_suite(
        model, tokenizer, cfg, device, n_samples,
        score_fn=laplace_score_fn, generate=False,
    )
    print(report)

    # --- MLflow logging ---
    use_mlflow = not args.no_mlflow
    with mlflow_context(cfg, use_mlflow):
        if use_mlflow:
            n_selected = sum(
                state.phi_hat[n].numel() for n in state.param_names
            )
            log_common_mlflow(cfg, tokenizer, n_params, "b1")
            # laplace.damping/sample_scale/selection_mode/n_curvature_batches
            # are already logged by log_common_mlflow via config flattening.
            # Only log computed/CLI params here.
            mlflow.log_params({
                "laplace.num_selected_params": str(n_selected),
                "skip_train": str(args.skip_train),
                "laplace_state_path": args.laplace_state or "",
            })
            mlflow.log_params(
                {f"laplace.{k}": f"{v:.6f}" for k, v in curv_stats.items()},
            )
            if train_meta:
                log_train_meta_mlflow(train_meta)
            log_perplexity_mlflow(ppl_results)
            log_mi_mlflow(mi_id, mi_ood, mi_ratio)
            mlflow.log_param("eval_time_sec", f"{eval_time:.1f}")
            log_qualitative_mlflow(report, qual_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
