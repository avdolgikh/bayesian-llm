"""A0 — Deterministic miniGPT baseline on TinyShakespeare (BPE)."""

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
from minigpt.data import get_tokenizer, load_shakespeare, prepare_data
from minigpt.evaluate import evaluate
from minigpt.model import MiniGPT
from minigpt.train import load_checkpoint, train


def main() -> None:
    p = argparse.ArgumentParser(description="A0 miniGPT baseline (TinyShakespeare, BPE)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="key=value", help="Dot-notation config override (repeatable)")
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
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
    text = load_shakespeare()
    tokenizer = get_tokenizer()
    train_data, val_data = prepare_data(text, tokenizer)
    print(f"BPE vocab size: {tokenizer.n_vocab}")
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

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
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
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

        # --- Train ---
        model = train(
            model, train_data, val_data, train_cfg,
            mlflow_run=run if use_mlflow else None,
            config_dict=cfg,
            resume_ckpt=resume_ckpt,
        )

        # --- Evaluate ---
        results = evaluate(
            model, val_data, tokenizer,
            cfg["train"]["block_size"], cfg["train"]["batch_size"], device,
        )

        if use_mlflow:
            mlflow.log_metric("final_val_perplexity", results["perplexity"])
            mlflow.log_text(results["sample"], "generated_sample.txt")

    print("\nDone.")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
