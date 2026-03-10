import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from minigpt.model import MiniGPT


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
    checkpoint_dir: str = "data/checkpoints"
    gradient_accumulation_steps: int = 1
    kl_annealing_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
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
    kl_scale: float = 0.0,
) -> dict[str, float]:
    """Estimate CE loss (and ELBO if Bayesian).

    Args:
        kl_scale: effective KL contribution = kl_scale * raw_kl.
            For Bayesian models: kl_scale = effective_kl_weight / num_train_tokens.
            For deterministic models: kl_scale = 0 (default).
    """
    model.eval()
    use_amp = next(model.parameters()).device.type == "cuda"
    ce_losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, ce_loss = model(x, y)
        ce_losses.append(ce_loss.item())
    model.train()
    avg_ce = sum(ce_losses) / len(ce_losses)
    result = {"loss": avg_ce, "perplexity": math.exp(avg_ce)}
    kl = model.kl_loss().item()
    if kl > 0:
        result["kl_loss"] = kl
        result["elbo_loss"] = avg_ce + kl_scale * kl
    return result


def save_checkpoint(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config_dict: dict,
    best_val_loss: float = float("inf"),
    path: Path | None = None,
) -> Path:
    checkpoint_dir = Path(config_dict.get("train", {}).get("checkpoint_dir", "data/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = checkpoint_dir / f"ckpt_step{step}.pt"
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
    """AdamW with weight decay only on 2-D weight tensors (not biases/layernorms/rho)."""
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Rho and g params are regularized by KL, not weight decay
        if "_rho" in name or "_g" in name:
            no_decay_params.append(p)
        elif p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2))


def train(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
    mlflow_run=None,
    config_dict: dict | None = None,
    resume_ckpt: dict | None = None,
    kl_weight: float = 0.0,
    num_train_tokens: int = 0,
) -> tuple[MiniGPT, dict]:
    """Train the model.

    Args:
        kl_weight: Bayesian KL penalty weight (0 = deterministic, no KL term).
        num_train_tokens: Total training tokens for ELBO normalization.
            Required when kl_weight > 0 (Bayesian mode).
    """
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

    # Bayesian mode: validate that num_train_tokens is provided
    is_bayesian = kl_weight > 0
    if is_bayesian and num_train_tokens <= 0:
        raise ValueError(
            "num_train_tokens must be > 0 for Bayesian training (kl_weight > 0)"
        )

    cfg_dict = config_dict or {}
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "ckpt_best.pt"
    best_val_step = 0

    # Checkpoint selection criterion
    ckpt_criterion = "ELBO" if is_bayesian else "CE"
    print(f"Best-checkpoint criterion: {ckpt_criterion}")

    # AMP: auto-enable on CUDA for mixed-precision training
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    # Gradient accumulation
    accum_steps = cfg.gradient_accumulation_steps

    start = time.time()
    total_tokens = 0
    for step in range(start_step, cfg.steps + 1):
        # Update learning rate (cosine schedule with warmup)
        lr = _get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # KL annealing (computed once per optimizer step, not per micro-batch)
        if is_bayesian:
            if cfg.kl_annealing_steps > 0:
                anneal = min(step / cfg.kl_annealing_steps, 1.0)
            else:
                anneal = 1.0
            effective_kl_weight = kl_weight * anneal
            kl_scale = effective_kl_weight / num_train_tokens
        else:
            effective_kl_weight = 0.0

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum_steps):
            x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
            total_tokens += x.numel()

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, ce_loss = model(x, y)

                # ELBO: loss = CE + kl_scale * KL.
                # KL is a property of weights, not data — add once per optimizer step.
                # We add it only on the last micro-step to avoid double-counting.
                if is_bayesian and micro_step == accum_steps - 1:
                    kl = model.kl_loss()
                    loss = ce_loss / accum_steps + kl_scale * kl
                else:
                    loss = ce_loss / accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if mlflow_run is not None:
            import mlflow
            mlflow.log_metric("lr", lr, step=step)
            if is_bayesian:
                mlflow.log_metric("effective_kl_weight", effective_kl_weight, step=step)

        if step % cfg.eval_interval == 0 or step == start_step:
            eval_kl_scale = kl_scale if is_bayesian else 0.0
            train_metrics = estimate_loss(
                model, train_data, cfg.block_size, cfg.batch_size, device, cfg.eval_iters,
                kl_scale=eval_kl_scale,
            )
            val_metrics = estimate_loss(
                model, val_data, cfg.block_size, cfg.batch_size, device, cfg.eval_iters,
                kl_scale=eval_kl_scale,
            )
            elapsed = time.time() - start
            if is_bayesian:
                val_criterion = val_metrics.get("elbo_loss", val_metrics["loss"])
            else:
                val_criterion = val_metrics["loss"]
            is_best = val_criterion < best_val_loss
            if is_best:
                best_val_loss = val_criterion
                best_val_step = step
                save_checkpoint(
                    model, optimizer, step, cfg_dict,
                    best_val_loss=best_val_loss, path=best_path,
                )

            # Print status
            status = (
                f"step {step:5d} | "
                f"train loss {train_metrics['loss']:.4f} ppl {train_metrics['perplexity']:.1f} | "
                f"val loss {val_metrics['loss']:.4f} ppl {val_metrics['perplexity']:.1f} | "
                f"{elapsed:.1f}s"
            )
            if "kl_loss" in val_metrics:
                status += f" | kl {val_metrics['kl_loss']:.1f}"
            if is_best:
                status += "  *best"
            print(status)

            if mlflow_run is not None:
                import mlflow
                metrics_to_log = {
                    "train_loss": train_metrics["loss"],
                    "train_perplexity": train_metrics["perplexity"],
                    "val_loss": val_metrics["loss"],
                    "val_perplexity": val_metrics["perplexity"],
                }
                if "kl_loss" in val_metrics:
                    metrics_to_log["kl_loss"] = val_metrics["kl_loss"]
                    metrics_to_log["val_elbo_loss"] = val_metrics["elbo_loss"]
                if "kl_loss" in train_metrics:
                    metrics_to_log["train_elbo_loss"] = train_metrics["elbo_loss"]
                mlflow.log_metrics(metrics_to_log, step=step)

        if cfg.checkpoint_interval and step % cfg.checkpoint_interval == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, step, cfg_dict, best_val_loss=best_val_loss,
            )
            print(f"  -> checkpoint saved: {ckpt_path}")

    # Reload best checkpoint for downstream use
    print(f"  -> reloading best checkpoint (val loss {best_val_loss:.4f}): {best_path}")
    load_checkpoint(best_path, model)

    train_time = time.time() - start
    tokens_per_sec = total_tokens / train_time if train_time > 0 else 0.0
    metadata = {
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "train_time_sec": train_time,
        "tokens_per_sec": tokens_per_sec,
    }
    return model, metadata


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
