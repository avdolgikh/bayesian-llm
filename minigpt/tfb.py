"""B3: Training-Free Bayesianization (TFB) for LoRA.

Finds the maximum noise (sigma_q) that can be injected into LoRA A weights
without degrading performance beyond a tolerance epsilon.
Variance is structured by SVD of the deterministic B matrix.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from minigpt.laplace import apply_sampled_params
from minigpt.train import get_batch
from minigpt.uncertainty import mc_metrics_single


@dataclass
class TFBState:
    """Fitted TFB posterior state."""
    sigma_q: float
    svd_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # U, S, V
    a_map: dict[str, torch.Tensor]
    param_names: list[str]
    epsilon: float
    anchor_loss: float


def fit_tfb(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    n_batches: int,
    epsilon: float,
    n_search_samples: int = 10,
    search_range: tuple[float, float] = (1e-4, 10.0),
    search_precision: float = 1e-4,
    max_iterations: int = 100,
) -> TFBState:
    """Find optimal sigma_q for TFB via binary search on anchor data.

    Args:
        model: deterministic model with LoRA adapters (DeterministicLoRALinear).
        data: anchor data for loss estimation.
        block_size: context window.
        batch_size: batch size for evaluation.
        n_batches: number of batches to average loss over.
        epsilon: tolerance for loss increase.
        n_search_samples: MC samples per search step.
        search_range: (min, max) sigma_q to search.
        search_precision: stop when range is smaller than this.
        max_iterations: maximum binary search iterations (safeguard).

    Returns:
        TFBState with fitted sigma_q and SVD cache.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    # 1. Identify LoRA layers and cache SVD of B and MAP of A
    svd_cache = {}
    a_map = {}
    param_names = []

    from minigpt.lora import DeterministicLoRALinear

    for name, module in model.named_modules():
        if isinstance(module, DeterministicLoRALinear):
            # Compact SVD of B (out, rank)
            U, S, V = torch.linalg.svd(module.lora_B.data, full_matrices=False)
            svd_cache[name] = (U, S, V)

            # Cache A_MAP
            a_name = f"{name}.lora_A"
            a_map[a_name] = module.lora_A.data.detach().clone()
            param_names.append(a_name)

    if not param_names:
        raise ValueError("No DeterministicLoRALinear layers found in model.")

    # 2. Draw fixed anchor batches (M1: reuse same data for all search steps)
    anchor_batches = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data, block_size, batch_size, device)
            anchor_batches.append((x, y))

    # 3. Compute anchor loss (MAP) on fixed batches
    total_loss = 0.0
    with torch.no_grad():
        for x, y in anchor_batches:
            _, loss = model(x, y)
            total_loss += loss.item()
    anchor_loss = total_loss / n_batches

    # 4. Binary search for sigma_q on fixed anchor batches
    low, high = search_range
    best_sigma = low

    print(f"Starting TFB binary search (anchor_loss={anchor_loss:.4f}, eps={epsilon})...")

    iteration = 0
    while (high - low) > search_precision and iteration < max_iterations:
        mid = (low + high) / 2

        # Estimate expected loss under mid-sigma noise
        avg_noisy_loss = 0.0
        temp_state = TFBState(
            sigma_q=mid,
            svd_cache=svd_cache,
            a_map=a_map,
            param_names=param_names,
            epsilon=epsilon,
            anchor_loss=anchor_loss,
        )

        with torch.no_grad():
            for x, y in anchor_batches:
                batch_loss = 0.0
                for s in range(n_search_samples):
                    sampled = sample_tfb_params(temp_state, seed=s)
                    with apply_sampled_params(model, sampled):
                        _, loss = model(x, y)
                        batch_loss += loss.item()
                avg_noisy_loss += (batch_loss / n_search_samples)

        avg_noisy_loss /= n_batches
        delta = abs(avg_noisy_loss - anchor_loss)

        print(f"  [{iteration}] sigma_q={mid:.4f} -> noisy_loss={avg_noisy_loss:.4f} "
              f"(delta={delta:.4f})")

        if delta <= epsilon:
            best_sigma = mid
            low = mid  # Try more noise
        else:
            high = mid  # Too much noise

        iteration += 1

    if iteration >= max_iterations:
        print(f"  Warning: binary search hit max_iterations={max_iterations}")

    if was_training:
        model.train()

    return TFBState(
        sigma_q=best_sigma,
        svd_cache=svd_cache,
        a_map=a_map,
        param_names=param_names,
        epsilon=epsilon,
        anchor_loss=anchor_loss,
    )


def sample_tfb_params(
    state: TFBState,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample A from TFB posterior: A_hat = A_MAP + Omega * eps.

    Omega_ij = sigma_q / S_i
    where S_i are singular values of B.
    """
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    sampled = {}
    for a_name in state.param_names:
        layer_name = a_name.replace(".lora_A", "")
        U, S, V = state.svd_cache[layer_name]
        a_map = state.a_map[a_name]

        if state.sigma_q == 0.0:
            sampled[a_name] = a_map.clone()
            continue

        # Omega structure: sigma_q / S_i applied to each row i of A
        # S has shape (rank,), a_map has shape (rank, in_features)
        # S can have tiny values; clamp for stability
        S_clamped = S.clamp(min=1e-6)
        std = (state.sigma_q / S_clamped).unsqueeze(1)  # (rank, 1)

        eps = torch.randn(a_map.shape, generator=gen, dtype=a_map.dtype)
        sampled[a_name] = a_map + std * eps.to(a_map.device)

    return sampled


def save_tfb_state(state: TFBState, path: str | Path) -> None:
    """Save TFBState to disk."""
    torch.save({
        "sigma_q": state.sigma_q,
        "svd_cache": state.svd_cache,
        "a_map": state.a_map,
        "param_names": state.param_names,
        "epsilon": state.epsilon,
        "anchor_loss": state.anchor_loss,
    }, path)


def load_tfb_state(path: str | Path, map_location=None) -> TFBState:
    """Load TFBState from disk."""
    data = torch.load(path, weights_only=False, map_location=map_location)
    return TFBState(
        sigma_q=data["sigma_q"],
        svd_cache=data["svd_cache"],
        a_map=data["a_map"],
        param_names=data["param_names"],
        epsilon=data["epsilon"],
        anchor_loss=data["anchor_loss"],
    )


def score_sequence_tfb(
    model: nn.Module,
    token_ids: torch.Tensor,
    device: torch.device,
    n_samples: int = 20,
    *,
    state: TFBState,
) -> dict[str, torch.Tensor]:
    """Score a single sequence using TFB posterior sampling.

    Args:
        model: deterministic MiniGPT at MAP weights.
        token_ids: (seq_len,) token indices.
        device: torch device.
        state: fitted TFBState.
        n_samples: number of MC posterior samples.

    Returns per-token tensors (seq_len,): mi, predictive_entropy, expected_entropy, flip_rate.
    """
    x = token_ids.unsqueeze(0).to(device)

    def get_logits(s: int) -> torch.Tensor:
        sampled = sample_tfb_params(state, seed=s)
        with apply_sampled_params(model, sampled):
            logits, _ = model(x)
        return logits

    return mc_metrics_single(
        get_logits, n_samples, x.size(1), model.config.vocab_size, device,
    )


@torch.no_grad()
def compute_tfb_uncertainty(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    state: TFBState,
    n_samples: int = 20,
    n_batches: int = 20,
) -> dict[str, float]:
    """Compute uncertainty metrics using TFB posterior sampling."""
    model.eval()

    all_mi = []
    all_pred_ent = []
    all_exp_ent = []
    all_flip = []

    for batch_idx in range(n_batches):
        x, _ = get_batch(data, block_size, batch_size, device)

        for b in range(x.size(0)):
            x_single = x[b:b + 1]
            # M6: unique seeds across all batches and elements
            seed_offset = (batch_idx * batch_size + b) * n_samples

            def get_logits(s: int, _x=x_single, _off=seed_offset) -> torch.Tensor:
                sampled = sample_tfb_params(state, seed=s + _off)
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
