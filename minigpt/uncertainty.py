"""Epistemic uncertainty estimation via MC weight sampling.

Core metrics:
- Predictive entropy H[p_bar]: total uncertainty
- Expected entropy E_bar: aleatoric uncertainty
- Mutual information MI = H[p_bar] - E_bar: epistemic uncertainty
- Top-1 flip rate: fraction of samples where argmax differs from mode
"""

from collections.abc import Callable

import torch
from torch.nn import functional as F

from minigpt.layers import BayesianLinear
from minigpt.model import MiniGPT
from minigpt.train import get_batch


def mc_metrics_single(
    get_logits_fn: Callable[[int], torch.Tensor],
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Core MC metrics computation for a single sequence.

    Args:
        get_logits_fn: callable(sample_idx) -> logits tensor (1, seq_len, vocab).
        n_samples: number of MC forward passes.
        seq_len: sequence length.
        vocab_size: vocabulary size.
        device: torch device for accumulators.

    Returns per-token tensors (seq_len,): mi, predictive_entropy, expected_entropy, flip_rate.
    """
    eps = 1e-10
    p_sum = torch.zeros(seq_len, vocab_size, device=device)
    entropy_sum = torch.zeros(seq_len, device=device)
    argmaxes = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)

    for s in range(n_samples):
        logits = get_logits_fn(s)
        probs = F.softmax(logits[0].float(), dim=-1)
        p_sum.add_(probs)
        entropy_sum.add_(-(probs * torch.log(probs + eps)).sum(dim=-1))
        argmaxes[s] = probs.argmax(dim=-1)

    p_bar = p_sum / n_samples
    predictive_entropy = -(p_bar * torch.log(p_bar + eps)).sum(dim=-1)
    expected_entropy = entropy_sum / n_samples
    mi = predictive_entropy - expected_entropy

    mode_tokens = argmaxes.mode(dim=0).values
    flip_rate = (argmaxes != mode_tokens.unsqueeze(0)).float().mean(dim=0)

    return {
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mi": mi,
        "flip_rate": flip_rate,
    }


def _has_bayesian_body(model: MiniGPT) -> bool:
    """Check if any BayesianLinear layer exists in the transformer blocks."""
    for block in model.blocks:
        for m in block.modules():
            if isinstance(m, BayesianLinear):
                return True
    return False


def _stream_metrics(
    model: MiniGPT,
    h: torch.Tensor,
    n_samples: int,
) -> dict[str, torch.Tensor]:
    """Streaming MC metrics for a single batch element — head-only path (A1)."""
    use_amp = h.device.type == "cuda"

    def get_logits(s: int) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=h.device.type, dtype=torch.float16, enabled=use_amp,
        ):
            return model.lm_head(h)

    return mc_metrics_single(
        get_logits, n_samples, h.size(1), model.config.vocab_size, h.device,
    )


def _stream_metrics_full(
    model: MiniGPT,
    x: torch.Tensor,
    n_samples: int,
) -> dict[str, torch.Tensor]:
    """Streaming MC metrics with full forward pass — body+head path (A2+)."""
    use_amp = x.device.type == "cuda"

    def get_logits(s: int) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=x.device.type, dtype=torch.float16, enabled=use_amp,
        ):
            logits, _ = model(x)
        return logits

    return mc_metrics_single(
        get_logits, n_samples, x.size(1), model.config.vocab_size, x.device,
    )


@torch.no_grad()
def compute_uncertainty_metrics(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    n_samples: int = 30,
    n_batches: int = 20,
) -> dict[str, float]:
    """Compute aggregate uncertainty metrics over random batches.

    Auto-selects evaluation path:
    - If Bayesian layers in body (A2+): N full forward passes per element.
    - If only head is Bayesian (A1): body once, lm_head N times (efficient).

    Memory-safe: processes batch elements one at a time during MC sampling
    to avoid allocating (N, batch, seq_len, vocab) tensors.

    Returns dict with scalar means:
        - mi_mean, predictive_entropy_mean, expected_entropy_mean, flip_rate
    """
    model.eval()
    all_mi = []
    all_pred_ent = []
    all_exp_ent = []
    all_flip = []

    bayesian_body = _has_bayesian_body(model)
    use_amp = device.type == "cuda"
    for _ in range(n_batches):
        x, _ = get_batch(data, block_size, batch_size, device)

        if bayesian_body:
            # A2+ path: full forward pass per MC sample (body is stochastic)
            for b in range(x.size(0)):
                metrics = _stream_metrics_full(model, x[b : b + 1], n_samples)
                all_mi.append(metrics["mi"].mean().item())
                all_pred_ent.append(metrics["predictive_entropy"].mean().item())
                all_exp_ent.append(metrics["expected_entropy"].mean().item())
                all_flip.append(metrics["flip_rate"].mean().item())
        else:
            # A1 path: body once, head N times (efficient)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp,
            ):
                h = model.forward_body(x)  # (batch, seq_len, n_embd)

            for b in range(x.size(0)):
                metrics = _stream_metrics(model, h[b : b + 1], n_samples)
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


@torch.no_grad()
def score_sequence(
    model: MiniGPT,
    token_ids: torch.Tensor,
    device: torch.device,
    n_samples: int = 30,
) -> dict[str, torch.Tensor]:
    """Score a single sequence and return per-token uncertainty metrics.

    Auto-selects A1 (head-only) or A2+ (full-model) MC path.

    Args:
        model: trained MiniGPT with Bayesian layers.
        token_ids: (seq_len,) token indices.
        device: torch device.
        n_samples: number of MC forward passes.

    Returns dict with per-token tensors (seq_len,):
        - mi, predictive_entropy, expected_entropy, flip_rate
    """
    model.eval()
    x = token_ids.unsqueeze(0).to(device)  # (1, seq_len)

    if _has_bayesian_body(model):
        # A2+ path: full forward pass per MC sample
        return _stream_metrics_full(model, x, n_samples)
    else:
        # A1 path: body once, head N times
        use_amp = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            h = model.forward_body(x)  # (1, seq_len, n_embd)
        return _stream_metrics(model, h, n_samples)
