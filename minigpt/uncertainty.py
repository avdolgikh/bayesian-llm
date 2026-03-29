"""Epistemic uncertainty estimation via MC weight sampling.

Core metrics:
- Predictive entropy H[p_bar]: total uncertainty
- Expected entropy E_bar: aleatoric uncertainty
- Mutual information MI = H[p_bar] - E_bar: epistemic uncertainty
- Top-1 flip rate: fraction of samples where argmax differs from mode

Evaluation metrics (D0):
- OOD detection: AUROC, FPR@TPR, AUPRC
- Calibration: ECE, NLL, Brier score
- Selective prediction: risk-coverage curve, AURC
- Sequence-level aggregation (mean, max, proportion)
"""

from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional as F

from minigpt.layers import BayesianLinear
from minigpt.lora import BLoBLoRALinear
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
    """Check if any stochastic Bayesian layer exists in the transformer blocks."""
    for block in model.blocks:
        for m in block.modules():
            if isinstance(m, (BayesianLinear, BLoBLoRALinear)):
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


# ---------------------------------------------------------------------------
# D0: OOD detection metrics
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def auroc(scores, labels) -> float:
    """Area Under ROC Curve for OOD detection.

    Args:
        scores: uncertainty scores (higher = more likely OOD).
        labels: binary labels (0=ID, 1=OOD).

    Returns:
        AUROC in [0, 1]. 1.0 = perfect, 0.5 = random.
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(_to_numpy(labels), _to_numpy(scores)))


def fpr_at_tpr(scores, labels, target_tpr: float = 0.95) -> float:
    """False Positive Rate at a given True Positive Rate.

    Args:
        scores: uncertainty scores (higher = more likely OOD).
        labels: binary labels (0=ID, 1=OOD).
        target_tpr: TPR threshold (default 0.95).

    Returns:
        FPR in [0, 1]. Lower is better.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(_to_numpy(labels), _to_numpy(scores))
    # Find the FPR at the first threshold where TPR >= target
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def auprc(scores, labels) -> float:
    """Area Under Precision-Recall Curve (OOD as positive class).

    Args:
        scores: uncertainty scores (higher = more likely OOD).
        labels: binary labels (0=ID, 1=OOD).

    Returns:
        AUPRC in [0, 1]. Baseline = class prior.
    """
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(_to_numpy(labels), _to_numpy(scores)))


# ---------------------------------------------------------------------------
# D0: Calibration metrics
# ---------------------------------------------------------------------------

def ece(confidences, correct, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Args:
        confidences: predicted confidence (max softmax prob) per sample.
        correct: binary (1=correct, 0=wrong) per sample.
        n_bins: number of equal-width bins (default 15 per spec).

    Returns:
        ECE in [0, 1]. Lower is better.
    """
    conf = _to_numpy(confidences).astype(np.float64)
    acc = _to_numpy(correct).astype(np.float64)
    n = len(conf)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total_ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = conf[mask].mean()
        avg_acc = acc[mask].mean()
        total_ece += (n_bin / n) * abs(avg_acc - avg_conf)

    return float(total_ece)


def nll(probs, targets) -> float:
    """Negative Log-Likelihood (mean per sample).

    Args:
        probs: (N, vocab) predicted probability distributions.
        targets: (N,) integer class labels.

    Returns:
        Mean NLL. Lower is better. Perplexity = exp(NLL).
    """
    probs_t = torch.as_tensor(probs, dtype=torch.float64)
    targets_t = torch.as_tensor(targets, dtype=torch.long)
    eps = 1e-10
    p_true = probs_t[torch.arange(len(targets_t)), targets_t]
    return float(-(p_true + eps).log().mean())


def brier_score(probs, targets) -> float:
    """Brier Score for multi-class prediction.

    Brier = mean over samples of (1 - 2*p(y_true) + sum(p_k^2)).

    Args:
        probs: (N, vocab) predicted probability distributions.
        targets: (N,) integer class labels.

    Returns:
        Mean Brier score. Lower is better. Range [0, 2].
    """
    probs_t = torch.as_tensor(probs, dtype=torch.float64)
    targets_t = torch.as_tensor(targets, dtype=torch.long)
    p_true = probs_t[torch.arange(len(targets_t)), targets_t]
    sum_p_sq = (probs_t ** 2).sum(dim=-1)
    per_sample = 1.0 - 2.0 * p_true + sum_p_sq
    return float(per_sample.mean())


# ---------------------------------------------------------------------------
# D0: Selective prediction
# ---------------------------------------------------------------------------

def risk_coverage_curve(uncertainties, correct):
    """Compute risk-coverage curve.

    Sort samples by uncertainty (descending), progressively include from
    most certain to least certain. At each coverage level, compute error rate.

    Args:
        uncertainties: scalar uncertainty per sample (higher = less certain).
        correct: binary (1=correct, 0=wrong) per sample.

    Returns:
        (coverages, risks): lists of floats, monotonically increasing coverage.
    """
    unc = _to_numpy(uncertainties)
    cor = _to_numpy(correct)
    n = len(unc)

    # Sort by uncertainty ascending (most certain first)
    order = np.argsort(unc)
    cor_sorted = cor[order]

    cumsum_correct = np.cumsum(cor_sorted)
    counts = np.arange(1, n + 1, dtype=np.float64)
    coverages = torch.from_numpy(counts / n)
    risks = torch.from_numpy(1.0 - cumsum_correct / counts)

    return coverages, risks


def aurc(uncertainties, correct) -> float:
    """Area Under Risk-Coverage Curve.

    Args:
        uncertainties: scalar uncertainty per sample.
        correct: binary (1=correct, 0=wrong) per sample.

    Returns:
        AURC in [0, 1]. Lower is better.
    """
    coverages, risks = risk_coverage_curve(uncertainties, correct)
    return float(np.trapezoid(risks.numpy(), coverages.numpy()))


# ---------------------------------------------------------------------------
# D0: Sequence-level aggregation
# ---------------------------------------------------------------------------

def aggregate_sequence_scores(
    token_scores: torch.Tensor,
    method: str = "mean",
    threshold: float = 0.0,
) -> float:
    """Aggregate per-token uncertainty scores to a single sequence-level scalar.

    Args:
        token_scores: (seq_len,) per-token uncertainty values.
        method: "mean", "max", or "proportion".
        threshold: for "proportion" method, count tokens above this value.

    Returns:
        Scalar float.
    """
    if method == "mean":
        return float(token_scores.mean())
    elif method == "max":
        return float(token_scores.max())
    elif method == "proportion":
        return float((token_scores > threshold).float().mean())
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores,
    labels,
    metric_fn,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for any metric_fn(scores, labels) -> float.

    Resamples *sequences* (paired scores+labels) with replacement.

    Args:
        scores: per-sequence uncertainty scores.
        labels: binary labels (0=ID, 1=OOD).
        metric_fn: callable(scores, labels) -> float (e.g. auroc, fpr_at_tpr).
        n_bootstrap: number of bootstrap resamples.
        ci: confidence level (default 0.95 for 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        (point_estimate, ci_low, ci_high).
    """
    scores_np = _to_numpy(scores)
    labels_np = _to_numpy(labels)
    n = len(scores_np)

    point = float(metric_fn(scores_np, labels_np))

    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_labels = labels_np[idx]
        if len(np.unique(b_labels)) < 2:
            continue  # skip degenerate resamples (single class)
        boot_values.append(metric_fn(scores_np[idx], b_labels))

    boot_values = np.asarray(boot_values)
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_values, 100 * alpha / 2))
    hi = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
    return point, lo, hi
