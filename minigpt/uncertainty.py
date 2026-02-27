"""Epistemic uncertainty estimation via MC weight sampling.

Core metrics:
- Predictive entropy H[p_bar]: total uncertainty
- Expected entropy E_bar: aleatoric uncertainty
- Mutual information MI = H[p_bar] - E_bar: epistemic uncertainty
- Top-1 flip rate: fraction of samples where argmax differs from mode
"""

import torch
from torch.nn import functional as F

from minigpt.layers import BayesianLinear
from minigpt.model import MiniGPT
from minigpt.train import get_batch


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
    """Streaming MC metrics for a single batch element — O(seq_len * vocab) memory.

    Args:
        model: MiniGPT model (only lm_head is called).
        h: hidden states for one element, shape (1, seq_len, n_embd).
        n_samples: number of MC forward passes.

    Returns per-token tensors (seq_len,): mi, predictive_entropy, expected_entropy, flip_rate.
    """
    eps = 1e-10
    seq_len = h.size(1)
    device = h.device

    # Running accumulators — never stack N full vocab tensors
    p_sum = torch.zeros(seq_len, model.config.vocab_size, device=device)
    entropy_sum = torch.zeros(seq_len, device=device)
    argmaxes = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)

    use_amp = h.device.type == "cuda"
    for s in range(n_samples):
        with torch.amp.autocast(device_type=h.device.type, dtype=torch.float16, enabled=use_amp):
            logits = model.lm_head(h)  # (1, seq_len, vocab)
        probs = F.softmax(logits[0].float(), dim=-1)  # (seq_len, vocab) — fp32 for entropy
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


def _stream_metrics_full(
    model: MiniGPT,
    x: torch.Tensor,
    n_samples: int,
) -> dict[str, torch.Tensor]:
    """Streaming MC metrics with full forward pass per sample.

    Like _stream_metrics, but runs the entire model (not just lm_head).
    Used when the transformer body contains BayesianLinear layers (A2+).

    Args:
        model: MiniGPT model with Bayesian layers in the body.
        x: input tokens for one element, shape (1, seq_len).
        n_samples: number of MC forward passes.

    Returns per-token tensors (seq_len,): mi, predictive_entropy, expected_entropy, flip_rate.
    """
    eps = 1e-10
    seq_len = x.size(1)
    device = x.device
    vocab = model.config.vocab_size

    p_sum = torch.zeros(seq_len, vocab, device=device)
    entropy_sum = torch.zeros(seq_len, device=device)
    argmaxes = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)

    use_amp = device.type == "cuda"
    for s in range(n_samples):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x)  # full forward pass — new weight sample each time
        probs = F.softmax(logits[0].float(), dim=-1)  # (seq_len, vocab) — fp32
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
