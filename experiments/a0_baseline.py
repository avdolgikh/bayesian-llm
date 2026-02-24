"""A0 — Deterministic miniGPT baseline on TinyShakespeare (BPE)."""

import argparse

import mlflow
import torch

from minigpt.data import get_tokenizer, load_shakespeare, prepare_data
from minigpt.evaluate import evaluate
from minigpt.layers import BayesConfig
from minigpt.model import GPTConfig, MiniGPT
from minigpt.train import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser(description="A0 miniGPT baseline (TinyShakespeare, BPE)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--sample-tokens", type=int, default=200)
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # --- Data ---
    text = load_shakespeare()
    tokenizer = get_tokenizer()
    train_data, val_data = prepare_data(text, tokenizer)
    print(f"BPE vocab size: {tokenizer.n_vocab}")
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    # --- Model ---
    gpt_config = GPTConfig(
        vocab_size=tokenizer.n_vocab,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bayes=BayesConfig(enabled=False),
    )
    model = MiniGPT(gpt_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Device ---
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    # --- Train config ---
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        device=device_str,
    )

    # --- MLflow ---
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("a0-baseline")

    run_ctx = mlflow.start_run(run_name="a0-baseline") if use_mlflow else _nullcontext()
    with run_ctx as run:
        if use_mlflow:
            mlflow.log_params(
                {
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "block_size": args.block_size,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "n_embd": args.n_embd,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_steps": args.warmup_steps,
                    "min_lr": args.min_lr,
                    "grad_clip": args.grad_clip,
                    "vocab_size": tokenizer.n_vocab,
                    "n_params": n_params,
                    "tokenizer": "gpt2-bpe",
                    "dataset": "tinyshakespeare",
                }
            )

        # --- Train ---
        model = train(model, train_data, val_data, train_cfg, mlflow_run=run if use_mlflow else None)

        # --- Evaluate ---
        results = evaluate(model, val_data, tokenizer, args.block_size, args.batch_size, device)

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
