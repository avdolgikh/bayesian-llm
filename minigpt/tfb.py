"""TFB: Training-Free Bayesianization for LoRA (Shi et al. 2024, arXiv 2412.05723).

Finds the maximum noise (sigma_q) that can be injected into LoRA A weights
without changing the anchor loss by more than a tolerance.
The variance is structured by the compact SVD of the deterministic B matrix,
B = U diag(S) Vh (torch.linalg.svd returns Vh = V^T).

Two samplers, selected by the required ``sampler_version``:

- ``"v2"``: the paper's sampler (Eqs. 4-7). The noise is isotropic in the rotated
  coordinates A' = Vh A with std sigma_q / S_i, so
  A = A_MAP + Vh^T diag(sigma_q / S) eps and E||B (A - A_MAP)||_F^2 = r n sigma_q^2.
- ``"v1_legacy"``: the pre-fix sampler, A = A_MAP + diag(sigma_q / S) eps. It skips the
  rotation, so its trace is never below the target. It is kept bit for bit so that the
  published states still reproduce, and sampling it emits a UserWarning.

The search accepts either an absolute tolerance ``epsilon`` (legacy callers) or a
relative one, ``epsilon_rel``: |mean noisy loss - anchor loss| <= epsilon_rel * anchor loss.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from minigpt.laplace import apply_sampled_params
from minigpt.train import get_batch
from minigpt.uncertainty import mc_metrics_single

SAMPLER_V1_LEGACY = "v1_legacy"
SAMPLER_V2 = "v2"
SAMPLER_VERSIONS = (SAMPLER_V1_LEGACY, SAMPLER_V2)


def _check_sampler_version(sampler_version: str) -> None:
    if sampler_version not in SAMPLER_VERSIONS:
        raise ValueError(
            f"Unknown TFB sampler_version {sampler_version!r}; "
            f"expected one of {SAMPLER_VERSIONS}"
        )


@dataclass
class TFBState:
    """Fitted TFB posterior state.

    ``svd_cache[layer]`` holds (U, S, Vh) of that layer's lora_B, as returned by
    ``torch.linalg.svd(B, full_matrices=False)``. ``search_log`` holds one dict per
    search step (see ``fit_tfb``). ``sampler_version`` is keyword-only with no default.
    """
    sigma_q: float
    svd_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # U, S, Vh
    a_map: dict[str, torch.Tensor]
    param_names: list[str]
    epsilon: float | None
    anchor_loss: float
    epsilon_rel: float | None = None
    search_log: list[dict] = field(default_factory=list)
    sampler_version: str = field(kw_only=True)

    def __post_init__(self) -> None:
        _check_sampler_version(self.sampler_version)


def fit_tfb(
    model: nn.Module,
    data: torch.Tensor | None,
    block_size: int,
    batch_size: int,
    n_batches: int,
    epsilon: float | None = None,
    n_search_samples: int = 10,
    search_range: tuple[float, float] = (1e-4, 10.0),
    search_precision: float = 1e-4,
    max_iterations: int = 100,
    *,
    sampler_version: str,
    epsilon_rel: float | None = None,
    anchor_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> TFBState:
    """Find the largest sigma_q within tolerance via bisection on fixed anchor batches.

    Exactly one of ``epsilon`` (absolute, nats) and ``epsilon_rel`` (fraction of the
    anchor loss) must be given. A step at sigma_q is accepted when
    |mean noisy loss - anchor loss| <= tolerance.

    On the ``epsilon_rel`` path only (the legacy ``epsilon`` path is unchanged):
    ``search_range[1]`` is evaluated first and the fit raises ValueError
    ("search range too small") if it is accepted; after the bisection the fit raises
    ValueError ("no sigma_q accepted") if no step was accepted.

    Every evaluated step is appended to ``search_log`` as a dict with keys
    ``stage`` ("precheck" or "bisect"), ``sigma_q``, ``avg_loss``, ``anchor_loss``,
    ``delta`` (signed, avg_loss - anchor_loss), ``tolerance`` (absolute),
    ``accepted`` and ``sampler_version``.

    Args:
        model: deterministic model with LoRA adapters (DeterministicLoRALinear).
        data: token stream for drawing anchor batches; may be None when
            ``anchor_batches`` is given.
        block_size: context window (used only when drawing batches from ``data``).
        batch_size: batch size (used only when drawing batches from ``data``).
        n_batches: number of batches to draw from ``data``.
        epsilon: absolute tolerance on the loss change (legacy callers).
        n_search_samples: MC samples per search step, at seeds 0..n_search_samples-1.
        search_range: (min, max) sigma_q to search.
        search_precision: stop when the bracket is not wider than this.
        max_iterations: maximum bisection steps (safeguard).
        sampler_version: "v2" or "v1_legacy"; the search itself runs this sampler.
        epsilon_rel: relative tolerance, as a fraction of the anchor loss.
        anchor_batches: fixed (x, y) batches to use instead of drawing from ``data``.
            All must have the same shape, so the mean of batch losses is the token mean.

    Returns:
        TFBState with the fitted sigma_q, SVD cache, search log and sampler version.
    """
    _check_sampler_version(sampler_version)
    if (epsilon is None) == (epsilon_rel is None):
        raise ValueError("Pass exactly one of epsilon (absolute) and epsilon_rel (relative).")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        # 1. Identify LoRA layers and cache SVD of B and MAP of A
        svd_cache = {}
        a_map = {}
        param_names = []

        from minigpt.lora import DeterministicLoRALinear

        for name, module in model.named_modules():
            if isinstance(module, DeterministicLoRALinear):
                # Compact SVD of B (out, rank): B = U diag(S) Vh
                U, S, Vh = torch.linalg.svd(module.lora_B.data, full_matrices=False)
                svd_cache[name] = (U, S, Vh)

                # Cache A_MAP
                a_name = f"{name}.lora_A"
                a_map[a_name] = module.lora_A.data.detach().clone()
                param_names.append(a_name)

        if not param_names:
            raise ValueError("No DeterministicLoRALinear layers found in model.")

        # 2. Fixed anchor batches (M1: reuse same data for all search steps)
        if anchor_batches is None:
            if data is None:
                raise ValueError("fit_tfb needs data or anchor_batches.")
            anchor_batches = []
            with torch.no_grad():
                for _ in range(n_batches):
                    x, y = get_batch(data, block_size, batch_size, device)
                    anchor_batches.append((x, y))
        else:
            if not anchor_batches:
                raise ValueError("anchor_batches is empty.")
            shapes = {(tuple(x.shape), tuple(y.shape)) for x, y in anchor_batches}
            if len(shapes) != 1:
                raise ValueError(f"anchor_batches must all have the same shape; got {shapes}")
            anchor_batches = [(x.to(device), y.to(device)) for x, y in anchor_batches]
        n_anchor = len(anchor_batches)

        # 3. Compute anchor loss (MAP) on fixed batches
        total_loss = 0.0
        with torch.no_grad():
            for x, y in anchor_batches:
                _, loss = model(x, y)
                total_loss += loss.item()
        anchor_loss = total_loss / n_anchor

        tolerance = epsilon if epsilon_rel is None else epsilon_rel * anchor_loss
        search_log: list[dict] = []

        def noisy_loss(sigma_q: float) -> float:
            """Mean loss over the anchor batches and seeds 0..n_search_samples-1."""
            temp_state = TFBState(
                sigma_q=sigma_q,
                svd_cache=svd_cache,
                a_map=a_map,
                param_names=param_names,
                epsilon=epsilon,
                anchor_loss=anchor_loss,
                epsilon_rel=epsilon_rel,
                sampler_version=sampler_version,
            )
            avg_noisy_loss = 0.0
            with torch.no_grad():
                for x, y in anchor_batches:
                    batch_loss = 0.0
                    for s in range(n_search_samples):
                        sampled = sample_tfb_params(temp_state, seed=s)
                        with apply_sampled_params(model, sampled):
                            _, loss = model(x, y)
                            batch_loss += loss.item()
                    avg_noisy_loss += (batch_loss / n_search_samples)
            return avg_noisy_loss / n_anchor

        def evaluate(stage: str, sigma_q: float) -> bool:
            avg_loss = noisy_loss(sigma_q)
            delta = avg_loss - anchor_loss
            accepted = abs(delta) <= tolerance
            search_log.append({
                "stage": stage,
                "sigma_q": sigma_q,
                "avg_loss": avg_loss,
                "anchor_loss": anchor_loss,
                "delta": delta,
                "tolerance": tolerance,
                "accepted": accepted,
                "sampler_version": sampler_version,
            })
            print(f"  [{stage} {len(search_log) - 1}] sigma_q={sigma_q:.6g} -> "
                  f"noisy_loss={avg_loss:.4f} (delta={delta:+.4g}, tol={tolerance:.4g}, "
                  f"{'accept' if accepted else 'reject'})")
            return accepted

        # 4. Binary search for sigma_q on fixed anchor batches
        low, high = search_range
        best_sigma = low

        print(f"Starting TFB binary search (anchor_loss={anchor_loss:.4f}, eps={epsilon}, "
              f"eps_rel={epsilon_rel}, sampler={sampler_version})...")

        if epsilon_rel is not None and evaluate("precheck", high):
            raise ValueError(
                f"search range too small: sigma_q={high} (search_max) is already within "
                f"tolerance {tolerance:.4g}"
            )

        any_accepted = False
        iteration = 0
        while (high - low) > search_precision and iteration < max_iterations:
            mid = (low + high) / 2
            if evaluate("bisect", mid):
                best_sigma = mid
                low = mid  # Try more noise
                any_accepted = True
            else:
                high = mid  # Too much noise
            iteration += 1

        if iteration >= max_iterations:
            print(f"  Warning: binary search hit max_iterations={max_iterations}")

        if epsilon_rel is not None and not any_accepted:
            raise ValueError(
                f"no sigma_q accepted in [{search_range[0]}, {search_range[1]}] "
                f"at tolerance {tolerance:.4g}"
            )

        return TFBState(
            sigma_q=best_sigma,
            svd_cache=svd_cache,
            a_map=a_map,
            param_names=param_names,
            epsilon=epsilon,
            anchor_loss=anchor_loss,
            epsilon_rel=epsilon_rel,
            search_log=search_log,
            sampler_version=sampler_version,
        )
    finally:
        if was_training:
            model.train()


def sample_tfb_params(
    state: TFBState,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample A from the TFB posterior.

    v2: A_hat = A_MAP + Vh^T (Omega * eps), with Omega_ij = sigma_q / S_i, i.e.
        the noise is drawn in A' = Vh A (TFB Eqs. 5-7) and rotated back.
    v1_legacy: A_hat = A_MAP + Omega * eps (no rotation; emits a UserWarning).

    S_i are the singular values of B. eps is drawn on the CPU from one generator
    seeded once, one tensor per name in ``param_names`` order, for both versions.
    """
    _check_sampler_version(state.sampler_version)
    legacy = state.sampler_version == SAMPLER_V1_LEGACY
    if legacy:
        warnings.warn(
            "Sampling a v1_legacy TFB state: the noise skips the SVD rotation of B. "
            "Use it only to reproduce pre-fix results.",
            UserWarning,
            stacklevel=2,
        )

    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    sampled = {}
    for a_name in state.param_names:
        layer_name = a_name.replace(".lora_A", "")
        _, S, Vh = state.svd_cache[layer_name]
        a_map = state.a_map[a_name]

        if state.sigma_q == 0.0:
            sampled[a_name] = a_map.clone()
            continue

        # Omega structure: sigma_q / S_i on row i of A' = Vh A
        # S has shape (rank,), a_map has shape (rank, in_features)
        # S can have tiny values; clamp for stability
        S_clamped = S.clamp(min=1e-6)
        std = (state.sigma_q / S_clamped).unsqueeze(1)  # (rank, 1)

        eps = torch.randn(a_map.shape, generator=gen, dtype=a_map.dtype)
        if legacy:
            sampled[a_name] = a_map + std * eps.to(a_map.device)
        else:
            sampled[a_name] = a_map + Vh.T @ (std * eps.to(a_map.device))

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
        "epsilon_rel": state.epsilon_rel,
        "search_log": state.search_log,
        "sampler_version": state.sampler_version,
    }, path)


def load_tfb_state(path: str | Path, map_location=None) -> TFBState:
    """Load TFBState from disk.

    A file without ``sampler_version`` (written before the S2 fix) loads as
    ``"v1_legacy"``; a file with an unknown version raises ValueError.
    """
    data = torch.load(path, weights_only=False, map_location=map_location)
    return TFBState(
        sigma_q=data["sigma_q"],
        svd_cache=data["svd_cache"],
        a_map=data["a_map"],
        param_names=data["param_names"],
        epsilon=data["epsilon"],
        anchor_loss=data["anchor_loss"],
        epsilon_rel=data.get("epsilon_rel"),
        search_log=data.get("search_log", []),
        sampler_version=data.get("sampler_version", SAMPLER_V1_LEGACY),
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
