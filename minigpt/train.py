from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from minigpt.model import MiniGPT

CHECKPOINT_DIR = Path("data/checkpoints")


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    block_size: int = 128
    lr: float = 3e-4
    eval_interval: int = 200
    eval_iters: int = 20
    checkpoint_interval: int = 500
    seed: int = 1337
    device: str = "auto"


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = data.size(0) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"Corpus too small for block_size={block_size}. "
            f"Need length > {block_size + 1}, got {data.size(0)}."
        )
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    avg_loss = sum(losses) / len(losses)
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


def save_checkpoint(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    path: Path | None = None,
) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = CHECKPOINT_DIR / f"ckpt_step{step}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


def train(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
    mlflow_run=None,
) -> MiniGPT:
    device = _resolve_device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    start = time.time()
    for step in range(1, cfg.steps + 1):
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == 1:
            train_metrics = estimate_loss(
                model, train_data, cfg.block_size, cfg.batch_size, device, cfg.eval_iters,
            )
            val_metrics = estimate_loss(
                model, val_data, cfg.block_size, cfg.batch_size, device, cfg.eval_iters,
            )
            elapsed = time.time() - start
            print(
                f"step {step:5d} | "
                f"train loss {train_metrics['loss']:.4f} ppl {train_metrics['perplexity']:.1f} | "
                f"val loss {val_metrics['loss']:.4f} ppl {val_metrics['perplexity']:.1f} | "
                f"{elapsed:.1f}s"
            )
            if mlflow_run is not None:
                import mlflow
                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "train_perplexity": train_metrics["perplexity"],
                        "val_loss": val_metrics["loss"],
                        "val_perplexity": val_metrics["perplexity"],
                    },
                    step=step,
                )

        if cfg.checkpoint_interval and step % cfg.checkpoint_interval == 0:
            ckpt_path = save_checkpoint(model, optimizer, step, {"step": step})
            print(f"  -> checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = save_checkpoint(model, optimizer, cfg.steps, {"step": cfg.steps})
    print(f"  -> final checkpoint: {final_path}")
    return model


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
