"""A0 — Deterministic miniGPT baseline (TinyShakespeare / AG News, BPE)."""

import argparse
from pathlib import Path

import mlflow
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
from minigpt.data import get_tokenizer, load_dataset
from minigpt.evaluate import compute_perplexity, evaluate
from minigpt.model import MiniGPT
from minigpt.train import load_checkpoint, train


def main() -> None:
    p = argparse.ArgumentParser(description="A0 miniGPT baseline")
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
    print(f"Model parameters: {n_params:,}")

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
        # Need a dummy optimizer to load state — train() will handle the rest
        resume_ckpt = load_checkpoint(ckpt_path, model)
        print(f"Loaded checkpoint: {ckpt_path} (step {resume_ckpt['step']})")

    # --- Train config ---
    train_cfg = build_train_config(cfg)

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
            flat["tokenizer"] = "gpt2-bpe"
            mlflow.log_params(flat)

        # --- Tags ---
        if use_mlflow:
            mlflow.set_tag("dataset", cfg["data"]["dataset"])
            mlflow.set_tag("milestone", "a0")
            if torch.cuda.is_available():
                mlflow.set_tag("gpu", torch.cuda.get_device_name())

        # --- Train ---
        model, train_meta = train(
            model, train_data, val_data, train_cfg,
            mlflow_run=run if use_mlflow else None,
            config_dict=cfg,
            resume_ckpt=resume_ckpt,
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

        # --- Evaluate ---
        eval_cfg = cfg["eval"]
        results = evaluate(
            model, val_data, tokenizer,
            cfg["train"]["block_size"], cfg["train"]["batch_size"], device,
            max_new_tokens=eval_cfg.get("sample_tokens", 200),
            temperature=eval_cfg.get("temperature", 0.8),
            n_perplexity_batches=eval_cfg.get("n_perplexity_batches", 20),
        )

        if use_mlflow:
            mlflow.log_param("final_val_perplexity", f"{results['perplexity']:.2f}")
            mlflow.log_text(results["sample"], "generated_sample.txt")

        # --- Test ID evaluation ---
        n_ppl_batches = eval_cfg.get("n_perplexity_batches", 20)
        test_id_ppl = compute_perplexity(
            model, test_id,
            cfg["train"]["block_size"], cfg["train"]["batch_size"], device,
            n_batches=n_ppl_batches,
        )
        print(f"\nTest ID perplexity: {test_id_ppl:.2f}")
        if use_mlflow:
            mlflow.log_param("test_id_perplexity", f"{test_id_ppl:.2f}")

        # --- Log model + checkpoint to MLflow (opt-in) ---
        if use_mlflow and args.log_model:
            mlflow.pytorch.log_model(model, "model")
            ckpt_dir = cfg["train"].get("checkpoint_dir", "data/checkpoints")
            best_ckpt = Path(ckpt_dir) / "ckpt_best.pt"
            if best_ckpt.exists():
                mlflow.log_artifact(str(best_ckpt))

        if use_mlflow:
            summary = (
                f"{cfg['experiment']['name']}, "
                f"{cfg['model']['n_layer']}L/{cfg['model']['n_head']}H/{cfg['model']['n_embd']}d, "
                f"{cfg['data']['dataset']}, "
                f"best_val={train_meta['best_val_loss']:.4f}, "
                f"test_id_ppl={test_id_ppl:.2f}"
            )
            mlflow.set_tag("mlflow.note.content", summary)

        # --- OOD evaluation ---
        if test_ood is not None:
            ood_ppl = compute_perplexity(
                model, test_ood,
                cfg["train"]["block_size"], cfg["train"]["batch_size"], device,
                n_batches=n_ppl_batches,
            )
            print(f"Test OOD perplexity: {ood_ppl:.2f}")
            print(f"ID vs OOD perplexity: {test_id_ppl:.2f} vs {ood_ppl:.2f}")
            if use_mlflow:
                mlflow.log_param("test_ood_perplexity", f"{ood_ppl:.2f}")

    print("\nDone.")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
