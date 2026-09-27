"""Post-hoc Laplace approximation for selected model parameters.

Fits a diagonal Gaussian posterior around MAP weights using empirical Fisher
(squared gradients), then provides sampling and context-manager APIs for
uncertainty evaluation via MC forward passes.

Two sampler versions exist (specs/i2-posthoc-fixes.md, Section 5.4):

- ``v1_legacy``: std = sample_scale / sqrt(F_hat + damping). F_hat is the mean squared
  gradient of the token-mean loss, so it is never scaled to the data size. Kept only so
  that the pre-fix states reproduce bitwise.
- ``v2``: precision tau = N_seq * T^2 * F_hat + prior_prec, std = tau^(-1/2). The curvature
  tensor always holds the raw F_hat; the scaling lives in the state fields.
"""

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from minigpt.train import get_batch
from minigpt.uncertainty import mc_metrics_single

SAMPLER_VERSIONS = ("v1_legacy", "v2")
_V2_FIELDS = ("n_data_seqs", "tokens_per_seq", "prior_prec")


@dataclass
class LaplaceState:
    """Fitted Laplace posterior state.

    ``curvature`` is the raw diagonal empirical Fisher F_hat in both versions. A ``v2``
    state also holds ``n_data_seqs`` (N_seq), ``tokens_per_seq`` (T) and ``prior_prec``
    (lambda); a ``v1_legacy`` state holds none of them. ``base_checkpoint`` and
    ``base_sha256`` optionally record the base model the curvature was fitted on.
    """
    param_names: list[str]
    phi_hat: dict[str, torch.Tensor]
    curvature: dict[str, torch.Tensor]
    damping: float
    sample_scale: float = 1.0
    sampler_version: str = field(kw_only=True)
    n_data_seqs: int | None = field(default=None, kw_only=True)
    tokens_per_seq: int | None = field(default=None, kw_only=True)
    prior_prec: float | None = field(default=None, kw_only=True)
    base_checkpoint: str | None = field(default=None, kw_only=True)
    base_sha256: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.sampler_version not in SAMPLER_VERSIONS:
            raise ValueError(
                f"Unknown sampler_version {self.sampler_version!r}; "
                f"expected one of {SAMPLER_VERSIONS}"
            )
        present = [name for name in _V2_FIELDS if getattr(self, name) is not None]
        if self.sampler_version == "v1_legacy" and present:
            raise ValueError(f"A v1_legacy state must not set {present}; they are v2 fields")
        if self.sampler_version == "v2" and len(present) != len(_V2_FIELDS):
            missing = [name for name in _V2_FIELDS if name not in present]
            raise ValueError(f"A v2 state needs {missing}")


def select_params(model: nn.Module, mode: str) -> dict[str, torch.Tensor]:
    """Select parameters for Laplace posterior by mode.

    Args:
        model: deterministic MiniGPT model.
        mode: 'ffn' | 'head' | 'all' | 'lora'.

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
    elif mode == "lora":
        selected = {}
        from minigpt.lora import DeterministicLoRALinear
        for name, module in model.named_modules():
            if isinstance(module, DeterministicLoRALinear):
                # We only Bayesianize A, matching TFB and BLoB
                selected[f"{name}.lora_A"] = module.lora_A
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
        LaplaceState with MAP weights, curvature diagonal, and config. The state is
        ``v1_legacy`` because F_hat is unscaled; ``scale_laplace_state`` makes a v2 state.
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
        sampler_version="v1_legacy",
    )


def scale_laplace_state(
    state: LaplaceState,
    *,
    n_data_seqs: int,
    tokens_per_seq: int,
    prior_prec: float,
    base_checkpoint: str | None = None,
    base_sha256: str | None = None,
) -> LaplaceState:
    """Build a v2 state: precision tau = n_data_seqs * tokens_per_seq^2 * F_hat + prior_prec.

    The input may be ``v1_legacy`` (from ``fit_laplace``) or ``v2`` (a lambda sweep re-scales
    the same F_hat). The curvature stays the raw F_hat. The returned state shares the input's
    tensors (no copy) and sets ``damping`` to 0.0, which v2 ignores. The input is not changed.

    Args:
        state: fitted state whose ``curvature`` is the raw F_hat.
        n_data_seqs: N_seq, the number of training sequences (training tokens / T).
        tokens_per_seq: T, the tokens per sequence of the token-mean loss behind F_hat.
        prior_prec: lambda >= 0. lambda = 0 needs every curvature entry to be positive.
        base_checkpoint: optional path of the base checkpoint the curvature was fitted on.
        base_sha256: optional SHA-256 of that checkpoint's bytes.

    Returns:
        A new ``v2`` LaplaceState.
    """
    if int(n_data_seqs) != n_data_seqs or n_data_seqs <= 0:
        raise ValueError(f"n_data_seqs must be a positive integer, got {n_data_seqs!r}")
    if int(tokens_per_seq) != tokens_per_seq or tokens_per_seq <= 0:
        raise ValueError(f"tokens_per_seq must be a positive integer, got {tokens_per_seq!r}")
    if not prior_prec >= 0.0:
        raise ValueError(f"prior_prec must be >= 0, got {prior_prec!r}")
    if prior_prec == 0.0:
        for name in state.param_names:
            if (state.curvature[name] <= 0).any():
                raise ValueError(
                    f"prior_prec 0 with a zero curvature entry in {name} gives an infinite std"
                )
    return LaplaceState(
        param_names=list(state.param_names),
        phi_hat=dict(state.phi_hat),
        curvature=dict(state.curvature),
        damping=0.0,
        sample_scale=1.0,
        sampler_version="v2",
        n_data_seqs=int(n_data_seqs),
        tokens_per_seq=int(tokens_per_seq),
        prior_prec=float(prior_prec),
        base_checkpoint=base_checkpoint if base_checkpoint is not None else state.base_checkpoint,
        base_sha256=base_sha256 if base_sha256 is not None else state.base_sha256,
    )


def _std_fn(state: LaplaceState) -> Callable[[str], torch.Tensor]:
    """Validate the state once and return name -> std, computed one tensor at a time."""
    if state.sampler_version == "v1_legacy":
        def legacy_std(name: str) -> torch.Tensor:
            # Exact legacy expression, so the pre-fix samples stay bitwise equal
            variance = 1.0 / (state.curvature[name] + state.damping)
            return variance.sqrt() * state.sample_scale
        return legacy_std
    if state.sampler_version == "v2":
        if state.sample_scale != 1.0:
            raise ValueError(
                f"A v2 Laplace state needs sample_scale 1.0, got {state.sample_scale!r}"
            )
        data_scale = state.n_data_seqs * state.tokens_per_seq**2

        def v2_std(name: str) -> torch.Tensor:
            return (data_scale * state.curvature[name] + state.prior_prec).rsqrt()
        return v2_std
    raise ValueError(f"Unknown sampler_version {state.sampler_version!r}")


def posterior_std(state: LaplaceState) -> dict[str, torch.Tensor]:
    """Per-entry posterior std of the diagonal Laplace posterior.

    v1_legacy: sqrt(1 / (curvature + damping)) * sample_scale (the exact legacy expression).
    v2: (n_data_seqs * tokens_per_seq^2 * curvature + prior_prec)^(-1/2); v2 has no
    temperature, so ``sample_scale`` must be 1.0 (ValueError otherwise).
    """
    std_of = _std_fn(state)
    return {name: std_of(name) for name in state.param_names}


def sample_laplace_params(
    state: LaplaceState,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample parameters from the Laplace posterior: phi ~ N(phi_hat, diag(std^2)).

    std comes from ``posterior_std``. v1_legacy draws the noise on the CPU from one
    generator (bitwise equal to the pre-fix sampler) and warns. v2 draws it on the
    parameter's device from ``torch.Generator(device=phi.device)`` seeded with ``seed``,
    one ``randn`` per name in ``param_names`` order, so v2 samples depend on the device.

    Args:
        state: fitted LaplaceState.
        seed: random seed for reproducibility.

    Returns:
        dict mapping param name -> sampled tensor.
    """
    if state.sampler_version == "v1_legacy":
        warnings.warn(
            "Sampling a v1_legacy Laplace state: its std is unscaled (about 1 for real "
            "models); use scale_laplace_state for a v2 state",
            UserWarning,
            stacklevel=2,
        )
        return _sample_v1_legacy(state, seed)
    if state.sampler_version == "v2":
        return _sample_v2(state, seed)
    raise ValueError(f"Unknown sampler_version {state.sampler_version!r}")


def _sample_v1_legacy(state: LaplaceState, seed: int | None) -> dict[str, torch.Tensor]:
    """Pre-fix sampler, unchanged: CPU generator, same RNG call order."""
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    std_of = _std_fn(state)
    sampled = {}
    for name in state.param_names:
        phi = state.phi_hat[name]
        if state.sample_scale == 0.0:
            sampled[name] = phi.clone()
        else:
            std = std_of(name)
            # Generate on CPU (generator is CPU-only) then move to device
            eps = torch.randn(phi.shape, generator=gen, dtype=phi.dtype)
            sampled[name] = phi + std * eps.to(phi.device)

    return sampled


def _sample_v2(state: LaplaceState, seed: int | None) -> dict[str, torch.Tensor]:
    """v2 sampler: noise drawn on the parameter's device, one generator per device."""
    std_of = _std_fn(state)
    generators: dict[torch.device, torch.Generator] = {}
    sampled = {}
    for name in state.param_names:
        phi = state.phi_hat[name]
        gen = None
        if seed is not None:
            gen = generators.get(phi.device)
            if gen is None:
                gen = torch.Generator(device=phi.device)
                gen.manual_seed(seed)
                generators[phi.device] = gen
        eps = torch.randn(phi.shape, generator=gen, dtype=phi.dtype, device=phi.device)
        sampled[name] = phi + std_of(name) * eps

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
    """Save LaplaceState to disk, including the sampler version and the v2 fields."""
    torch.save({
        "param_names": state.param_names,
        "phi_hat": state.phi_hat,
        "curvature": state.curvature,
        "damping": state.damping,
        "sample_scale": state.sample_scale,
        "sampler_version": state.sampler_version,
        "n_data_seqs": state.n_data_seqs,
        "tokens_per_seq": state.tokens_per_seq,
        "prior_prec": state.prior_prec,
        "base_checkpoint": state.base_checkpoint,
        "base_sha256": state.base_sha256,
    }, path)


def load_laplace_state(path: str | Path, map_location=None) -> LaplaceState:
    """Load LaplaceState from disk.

    A file without ``sampler_version`` (every pre-fix file) loads as ``v1_legacy``.
    An unknown ``sampler_version`` raises ValueError.
    """
    data = torch.load(path, weights_only=False, map_location=map_location)
    return LaplaceState(
        param_names=data["param_names"],
        phi_hat=data["phi_hat"],
        curvature=data["curvature"],
        damping=data["damping"],
        sample_scale=data["sample_scale"],
        sampler_version=data.get("sampler_version", "v1_legacy"),
        n_data_seqs=data.get("n_data_seqs"),
        tokens_per_seq=data.get("tokens_per_seq"),
        prior_prec=data.get("prior_prec"),
        base_checkpoint=data.get("base_checkpoint"),
        base_sha256=data.get("base_sha256"),
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

    for batch_idx in range(n_batches):
        x, _ = get_batch(data, block_size, batch_size, device)

        for b in range(x.size(0)):
            x_single = x[b:b + 1]
            # Unique seeds across all batches and elements
            seed_offset = (batch_idx * batch_size + b) * n_samples

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
