import math
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
    weight_decay: float = 0.1
    warmup_steps: int = 200
    min_lr: float = 1e-5
    grad_clip: float = 1.0
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
    config_dict: dict,
    best_val_loss: float = float("inf"),
    path: Path | None = None,
) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = CHECKPOINT_DIR / f"ckpt_step{step}.pt"
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_dict,
        "best_val_loss": best_val_loss,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(ckpt, path)
    return path


def load_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load checkpoint and return the full checkpoint dict.

    Backward-compatible with old checkpoints that lack new fields.
    """
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Ensure all expected keys have defaults for old checkpoints
    ckpt.setdefault("step", 0)
    ckpt.setdefault("best_val_loss", float("inf"))
    ckpt.setdefault("config", {})
    return ckpt


def _get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    if step >= cfg.steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def _configure_optimizer(model: MiniGPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay only on 2-D weight tensors (not biases/layernorms)."""
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def train(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
    mlflow_run=None,
    config_dict: dict | None = None,
    resume_ckpt: dict | None = None,
) -> MiniGPT:
    device = _resolve_device(cfg.device)
    model = model.to(device)
    optimizer = _configure_optimizer(model, cfg)

    # Resume state
    start_step = 1
    best_val_loss = float("inf")
    if resume_ckpt is not None:
        # Optimizer state was loaded in load_checkpoint; move to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_step = resume_ckpt["step"] + 1
        best_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
        # Restore RNG states
        if "rng_state" in resume_ckpt:
            torch.set_rng_state(resume_ckpt["rng_state"])
        if "cuda_rng_state" in resume_ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state(resume_ckpt["cuda_rng_state"])
        print(f"Resuming from step {resume_ckpt['step']} (best val loss {best_val_loss:.4f})")

    cfg_dict = config_dict or {}
    best_path = CHECKPOINT_DIR / "ckpt_best.pt"

    start = time.time()
    for step in range(start_step, cfg.steps + 1):
        # Update learning rate (cosine schedule with warmup)
        lr = _get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if mlflow_run is not None:
            import mlflow
            mlflow.log_metric("lr", lr, step=step)

        if step % cfg.eval_interval == 0 or step == start_step:
            train_metrics = estimate_loss(
                model, train_data, cfg.block_size, cfg.batch_size, device, cfg.eval_iters,
            )
            val_metrics = estimate_loss(
                model, val_data, cfg.block_size, cfg.batch_size, device, cfg.eval_iters,
            )
            elapsed = time.time() - start
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    model, optimizer, step, cfg_dict,
                    best_val_loss=best_val_loss, path=best_path,
                )
            print(
                f"step {step:5d} | "
                f"train loss {train_metrics['loss']:.4f} ppl {train_metrics['perplexity']:.1f} | "
                f"val loss {val_metrics['loss']:.4f} ppl {val_metrics['perplexity']:.1f} | "
                f"{elapsed:.1f}s"
                f"{'  *best' if is_best else ''}"
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
            ckpt_path = save_checkpoint(
                model, optimizer, step, cfg_dict, best_val_loss=best_val_loss,
            )
            print(f"  -> checkpoint saved: {ckpt_path}")

    # Reload best checkpoint for downstream use
    print(f"  -> reloading best checkpoint (val loss {best_val_loss:.4f}): {best_path}")
    load_checkpoint(best_path, model)
    return model


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
