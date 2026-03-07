"""Post-hoc Laplace approximation for selected model parameters.

Fits a diagonal Gaussian posterior around MAP weights using empirical Fisher
(squared gradients), then provides sampling and context-manager APIs for
uncertainty evaluation via MC forward passes.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from minigpt.train import get_batch
from minigpt.uncertainty import mc_metrics_single


@dataclass
class LaplaceState:
    """Fitted Laplace posterior state."""
    param_names: list[str]
    phi_hat: dict[str, torch.Tensor]
    curvature: dict[str, torch.Tensor]
    damping: float
    sample_scale: float = 1.0


def select_params(model: nn.Module, mode: str) -> dict[str, torch.Tensor]:
    """Select parameters for Laplace posterior by mode.

    Args:
        model: deterministic MiniGPT model.
        mode: 'ffn' | 'head' | 'all'.

    Returns:
        dict mapping param name -> param tensor (references, not copies).
    """
    if mode == "ffn":
        selected = {}
        for name, param in model.named_parameters():
            if "mlp.fc.linear.weight" in name or "mlp.proj.linear.weight" in name:
                selected[name] = param
        return selected
    elif mode == "head":
        selected = {}
        for name, param in model.lm_head.named_parameters(prefix="lm_head"):
            if "weight" in name:
                selected[name] = param
        return selected
    elif mode == "all":
        selected = {}
        for name, param in model.named_parameters():
            if "weight" in name and "ln" not in name and "emb" not in name:
                selected[name] = param
        return selected
    else:
        raise ValueError(f"Unknown selection mode: {mode!r}")


def fit_laplace(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    selection: dict[str, torch.Tensor],
    n_batches: int,
    damping: float,
    sample_scale: float = 1.0,
) -> LaplaceState:
    """Fit diagonal Laplace posterior via empirical Fisher (per-sample squared gradients).

    Processes sequences one at a time to get correct diagonal Fisher:
    F_ii = E[(dL/dθ_i)²].
    Batch-averaged gradients cancel at convergence; per-sample gradients don't.

    Args:
        model: deterministic model at MAP weights.
        data: flat token tensor for curvature accumulation.
        block_size: context window size.
        batch_size: batch size for drawing batches (each sample processed individually).
        selection: dict from select_params() -- names to target.
        n_batches: number of mini-batches for curvature accumulation.
        damping: regularization added to diagonal curvature.
        sample_scale: scaling factor for posterior samples (0 = MAP).

    Returns:
        LaplaceState with MAP weights, curvature diagonal, and config.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    param_names = list(selection.keys())
    phi_hat = {name: selection[name].detach().clone() for name in param_names}
    curvature_acc = {name: torch.zeros_like(selection[name]) for name in param_names}

    total_samples = 0
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        # Process each sequence individually for correct per-sample Fisher
        for b in range(x.size(0)):
            model.zero_grad()
            logits, loss = model(x[b : b + 1], y[b : b + 1])
            loss.backward()

            for name in param_names:
                grad = selection[name].grad
                if grad is not None:
                    curvature_acc[name].add_(grad.detach() ** 2)
            total_samples += 1

    # Average over total samples seen
    for name in param_names:
        curvature_acc[name].div_(total_samples)

    # Clean up: zero gradients so caller sees no side effects
    model.zero_grad()

    if was_training:
        model.train()

    return LaplaceState(
        param_names=param_names,
        phi_hat=phi_hat,
        curvature=curvature_acc,
        damping=damping,
        sample_scale=sample_scale,
    )


def sample_laplace_params(
    state: LaplaceState,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample parameters from the Laplace posterior.

    phi ~ N(phi_hat, diag(sample_scale^2 / (curvature + damping)))

    Args:
        state: fitted LaplaceState.
        seed: random seed for reproducibility.

    Returns:
        dict mapping param name -> sampled tensor.
    """
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    sampled = {}
    for name in state.param_names:
        phi = state.phi_hat[name]
        if state.sample_scale == 0.0:
            sampled[name] = phi.clone()
        else:
            variance = 1.0 / (state.curvature[name] + state.damping)
            std = variance.sqrt() * state.sample_scale
            # Generate on CPU (generator is CPU-only) then move to device
            eps = torch.randn(phi.shape, generator=gen, dtype=phi.dtype)
            sampled[name] = phi + std * eps.to(phi.device)

    return sampled


def _resolve_param(model: nn.Module, name: str) -> nn.Parameter:
    """Resolve a dotted param name to the actual parameter object."""
    parts = name.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return getattr(obj, parts[-1])


@contextmanager
def apply_sampled_params(
    model: nn.Module,
    sampled_params: dict[str, torch.Tensor],
):
    """Temporarily replace model parameters with sampled values.

    Restores original parameter data on exit (including on exception).
    """
    param_lookup = dict(model.named_parameters())
    originals = {}

    for name, new_data in sampled_params.items():
        if name in param_lookup:
            param = param_lookup[name]
        else:
            param = _resolve_param(model, name)
        originals[name] = (param, param.data.clone())
        param.data.copy_(new_data)

    try:
        yield
    finally:
        for name in sampled_params:
            param, original_data = originals[name]
            param.data.copy_(original_data)


def save_laplace_state(state: LaplaceState, path: str | Path) -> None:
    """Save LaplaceState to disk."""
    torch.save({
        "param_names": state.param_names,
        "phi_hat": state.phi_hat,
        "curvature": state.curvature,
        "damping": state.damping,
        "sample_scale": state.sample_scale,
    }, path)


def load_laplace_state(path: str | Path) -> LaplaceState:
    """Load LaplaceState from disk."""
    data = torch.load(path, weights_only=False)
    return LaplaceState(
        param_names=data["param_names"],
        phi_hat=data["phi_hat"],
        curvature=data["curvature"],
        damping=data["damping"],
        sample_scale=data["sample_scale"],
    )


def score_sequence_laplace(
    model: nn.Module,
    token_ids: torch.Tensor,
    device: torch.device,
    n_samples: int = 30,
    *,
    state: LaplaceState,
) -> dict[str, torch.Tensor]:
    """Score a single sequence using Laplace posterior sampling.

    Mirrors uncertainty.score_sequence but with external Laplace param patching.

    Args:
        model: deterministic MiniGPT at MAP weights.
        token_ids: (seq_len,) token indices.
        device: torch device.
        state: fitted LaplaceState.
        n_samples: number of MC posterior samples.

    Returns per-token tensors (seq_len,): mi, predictive_entropy, expected_entropy, flip_rate.
    """
    x = token_ids.unsqueeze(0).to(device)

    def get_logits(s: int) -> torch.Tensor:
        sampled = sample_laplace_params(state, seed=s)
        with apply_sampled_params(model, sampled):
            logits, _ = model(x)
        return logits

    return mc_metrics_single(
        get_logits, n_samples, x.size(1), model.config.vocab_size, device,
    )


@torch.no_grad()
def compute_laplace_uncertainty(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    state: LaplaceState,
    n_samples: int = 30,
    n_batches: int = 20,
) -> dict[str, float]:
    """Compute uncertainty metrics using Laplace posterior sampling.

    Same metric protocol as compute_uncertainty_metrics (MI, entropy, flip rate),
    but uses Laplace-sampled params instead of BayesianLinear internal sampling.

    Returns dict with scalar means:
        mi_mean, predictive_entropy_mean, expected_entropy_mean, flip_rate
    """
    model.eval()

    all_mi = []
    all_pred_ent = []
    all_exp_ent = []
    all_flip = []

    for _ in range(n_batches):
        x, _ = get_batch(data, block_size, batch_size, device)

        for b in range(x.size(0)):
            x_single = x[b:b + 1]
            seed_offset = b * n_samples

            def get_logits(s: int, _x=x_single, _off=seed_offset) -> torch.Tensor:
                sampled = sample_laplace_params(state, seed=s + _off)
                with apply_sampled_params(model, sampled):
                    logits, _ = model(_x)
                return logits

            metrics = mc_metrics_single(
                get_logits, n_samples, x_single.size(1),
                model.config.vocab_size, device,
            )
            all_mi.append(metrics["mi"].mean().item())
            all_pred_ent.append(metrics["predictive_entropy"].mean().item())
            all_exp_ent.append(metrics["expected_entropy"].mean().item())
            all_flip.append(metrics["flip_rate"].mean().item())

    return {
        "mi_mean": sum(all_mi) / len(all_mi),
        "predictive_entropy_mean": sum(all_pred_ent) / len(all_pred_ent),
        "expected_entropy_mean": sum(all_exp_ent) / len(all_exp_ent),
        "flip_rate": sum(all_flip) / len(all_flip),
    }
