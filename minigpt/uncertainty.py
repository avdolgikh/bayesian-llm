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

Iteration 2 (specs/i2-eval-rebuild.md, sections 3, 5.5 and 5.6):
- Realized-token Jensen gap g_t and its block mean G
- Document-weighted AUROC / FPR@95 (optional ``sample_weight``)
- Document-clustered, class-stratified bootstrap and paired contrast
- Holm step-down correction
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

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


def realized_token_gap(logp_real: torch.Tensor) -> dict[str, torch.Tensor]:
    """Realized-token Jensen gap (specs/i2-eval-rebuild.md, section 3).

    With per-sample log-probs of the realized token ell_{s,t}:
        log p_bar(y_t) = logsumexp_s ell_{s,t} - log N
        g_t = log p_bar(y_t) - mean_s ell_{s,t}   (>= 0 by Jensen)
        G = mean_t g_t   (all positions, no skip)

    Computed in float64 and returned in the input dtype.

    Args:
        logp_real: (..., N, T) per-sample log-probs of the realized tokens
            (the score file's ``logp_real`` is [B, N, T]).

    Returns dict with ``log_pbar`` (..., T), ``g`` (..., T) and ``G`` (...).
    """
    if logp_real.dim() < 2:
        raise ValueError(f"logp_real must be (..., N, T), got shape {tuple(logp_real.shape)}")
    lp = logp_real.double()
    n = lp.shape[-2]
    log_pbar = torch.logsumexp(lp, dim=-2) - math.log(n)
    g = log_pbar - lp.mean(dim=-2)
    out_dtype = logp_real.dtype
    return {
        "log_pbar": log_pbar.to(out_dtype),
        "g": g.to(out_dtype),
        "G": g.mean(dim=-1).to(out_dtype),
    }


# ---------------------------------------------------------------------------
# D0: OOD detection metrics
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def auroc(scores, labels, sample_weight=None) -> float:
    """Area Under ROC Curve for OOD detection.

    Args:
        scores: uncertainty scores (higher = more likely OOD).
        labels: binary labels (0=ID, 1=OOD).
        sample_weight: optional per-row weights (e.g. 1/k_d per block, so each
            document counts once). None keeps the unweighted value.

    Returns:
        AUROC in [0, 1]. 1.0 = perfect, 0.5 = random.
    """
    from sklearn.metrics import roc_auc_score
    if sample_weight is not None:
        return float(roc_auc_score(
            _to_numpy(labels), _to_numpy(scores),
            sample_weight=_to_numpy(sample_weight).astype(np.float64),
        ))
    return float(roc_auc_score(_to_numpy(labels), _to_numpy(scores)))


def fpr_at_tpr(scores, labels, target_tpr: float = 0.95, sample_weight=None) -> float:
    """False Positive Rate at a given True Positive Rate.

    Args:
        scores: uncertainty scores (higher = more likely OOD).
        labels: binary labels (0=ID, 1=OOD).
        target_tpr: TPR threshold (default 0.95).
        sample_weight: optional per-row weights. When given, the weighted rule of
            specs/i2-eval-rebuild.md section 5.6 applies: the full curve
            (``drop_intermediate=False``) and the first point with TPR >= target.
            None keeps the unweighted rule and its values unchanged.

    Returns:
        FPR in [0, 1]. Lower is better.
    """
    from sklearn.metrics import roc_curve
    if sample_weight is not None:
        fpr, tpr, _ = roc_curve(
            _to_numpy(labels), _to_numpy(scores),
            sample_weight=_to_numpy(sample_weight).astype(np.float64),
            drop_intermediate=False,
        )
        return float(fpr[np.argmax(tpr >= target_tpr)])
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


# ---------------------------------------------------------------------------
# Document-clustered bootstrap (specs/i2-eval-rebuild.md, section 5.6)
# ---------------------------------------------------------------------------

_DOC_BOOT_CHUNK = 256  # resamples per vectorized chunk (memory: chunk x n_blocks float64)


def _percentile_ci(values: np.ndarray, level: float) -> tuple[float, float]:
    """Two-sided percentile CI with numpy's default linear method."""
    lo_q = round(50.0 * (1.0 - level), 10)
    hi_q = round(50.0 * (1.0 + level), 10)
    lo, hi = np.percentile(values, [lo_q, hi_q])
    return float(lo), float(hi)


def _doc_boot_inputs(
    scores: dict[str, Any], labels, doc_ids, weights,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate inputs; map each block to its document position.

    Document positions: ID documents first, then OOD documents, each group in
    numpy's lexicographic order (``np.unique``), as in section 5.6.
    """
    y = _to_numpy(labels).astype(np.int64)
    docs = np.asarray(doc_ids if isinstance(doc_ids, np.ndarray) else list(doc_ids))
    w = _to_numpy(weights).astype(np.float64)
    n = y.shape[0]
    if y.ndim != 1 or docs.shape != (n,) or w.shape != (n,):
        raise ValueError(
            f"labels, doc_ids and weights must be 1-D of equal length; got "
            f"{y.shape}, {docs.shape}, {w.shape}"
        )
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (ID) or 1 (OOD)")
    if not (np.isfinite(w).all() and (w >= 0).all()):
        raise ValueError("weights must be finite and non-negative")
    s_np: dict[str, np.ndarray] = {}
    for name, s in scores.items():
        arr = _to_numpy(s).astype(np.float64)
        if arr.shape != (n,):
            raise ValueError(f"score {name!r} has shape {arr.shape}, expected ({n},)")
        if not np.isfinite(arr).all():
            raise ValueError(f"score {name!r} has non-finite values")
        s_np[name] = arr
    id_mask = y == 0
    id_docs = np.unique(docs[id_mask])
    ood_docs = np.unique(docs[~id_mask])
    n_id, n_ood = int(id_docs.size), int(ood_docs.size)
    if n_id == 0 or n_ood == 0:
        raise ValueError("need at least one ID and one OOD document")
    if np.intersect1d(id_docs, ood_docs).size > 0:
        raise ValueError("a document ID appears in both classes")
    block_doc = np.empty(n, dtype=np.int64)
    block_doc[id_mask] = np.searchsorted(id_docs, docs[id_mask])
    block_doc[~id_mask] = n_id + np.searchsorted(ood_docs, docs[~id_mask])
    return s_np, y, w, block_doc, n_id, n_ood


def _draw_doc_counts(
    rng: np.random.Generator, n_id: int, n_ood: int, n_draws: int,
) -> np.ndarray:
    """(n_draws, n_id + n_ood) draw counts; per resample, ID drawn first, then OOD."""
    counts = np.empty((n_draws, n_id + n_ood), dtype=np.float64)
    for r in range(n_draws):
        di = rng.integers(0, n_id, size=n_id)
        do = rng.integers(0, n_ood, size=n_ood)
        counts[r, :n_id] = np.bincount(di, minlength=n_id)
        counts[r, n_id:] = np.bincount(do, minlength=n_ood)
    return counts


class _WeightedROC:
    """Weighted ROC for fixed scores under many weight vectors.

    Mirrors sklearn's ``roc_curve`` (stable descending sort, float64 cumulative
    sums, TPR = tps / tps[-1]), so FPR@TPR equals the weighted ``fpr_at_tpr``
    rule and AUROC equals ``roc_auc_score`` up to rounding.
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray):
        self.order = np.argsort(-scores, kind="stable")
        s_sorted = scores[self.order]
        self.y_pos = labels[self.order].astype(np.float64)
        self.y_neg = 1.0 - self.y_pos
        self.ends = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]

    def curves(self, w_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(fpr, tpr), each (R, n_thresholds + 1), starting at (0, 0)."""
        w_sorted = w_matrix[:, self.order]
        tps = np.cumsum(w_sorted * self.y_pos, axis=1)[:, self.ends]
        fps = np.cumsum(w_sorted * self.y_neg, axis=1)[:, self.ends]
        zeros = np.zeros((w_matrix.shape[0], 1))
        tpr = np.concatenate([zeros, tps / tps[:, -1:]], axis=1)
        fpr = np.concatenate([zeros, fps / fps[:, -1:]], axis=1)
        return fpr, tpr

    @staticmethod
    def auroc(fpr: np.ndarray, tpr: np.ndarray) -> np.ndarray:
        return np.sum(np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2.0, axis=1)

    @staticmethod
    def fpr_at(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> np.ndarray:
        idx = np.argmax(tpr >= target_tpr, axis=1)
        return fpr[np.arange(fpr.shape[0]), idx]


def _doc_boot_resamples(
    s_np: dict[str, np.ndarray],
    y: np.ndarray,
    w: np.ndarray,
    block_doc: np.ndarray,
    n_id: int,
    n_ood: int,
    n_resamples: int,
    seed: int | Sequence[int],
    target_tpr: float | None,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Per score: (AUROC_w resamples, FPR_w resamples or None), on shared draws."""
    rng = np.random.default_rng(seed)
    rocs = {name: _WeightedROC(s, y) for name, s in s_np.items()}
    auc_out = {name: np.empty(n_resamples) for name in s_np}
    fpr_out = {name: np.empty(n_resamples) for name in s_np}
    for start in range(0, n_resamples, _DOC_BOOT_CHUNK):
        stop = min(start + _DOC_BOOT_CHUNK, n_resamples)
        counts = _draw_doc_counts(rng, n_id, n_ood, stop - start)
        w_matrix = w[None, :] * counts[:, block_doc]  # w_b^(r) = w_b * c_d(b)
        for name, roc in rocs.items():
            fpr, tpr = roc.curves(w_matrix)
            auc_out[name][start:stop] = roc.auroc(fpr, tpr)
            if target_tpr is not None:
                fpr_out[name][start:stop] = roc.fpr_at(fpr, tpr, target_tpr)
    return {
        name: (auc_out[name], fpr_out[name] if target_tpr is not None else None)
        for name in s_np
    }


def doc_bootstrap_auroc(
    scores: dict[str, Any],
    labels,
    doc_ids,
    weights,
    n_resamples: int,
    seed: int | Sequence[int],
    level: float,
    *,
    target_tpr: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Document-clustered, class-stratified bootstrap of weighted AUROC and FPR@TPR.

    Rules (specs/i2-eval-rebuild.md, section 5.6):
    - ID documents = ``np.unique(doc_ids[labels == 0])``; OOD documents likewise.
    - ``rng = np.random.default_rng(seed)``; per resample, in this order:
      ``di = rng.integers(0, n_I, size=n_I)``, then ``do = rng.integers(0, n_O, size=n_O)``.
    - The resampled block weight is w_b * c_d(b), with c_d the draw count of document d.
    - Every score in ``scores`` uses the same draws, so comparisons are paired.
    - Point estimates use the original weights. CI = linear percentiles.

    Args:
        scores: name -> per-block scores (higher = more OOD), all on the same blocks.
        labels: per-block labels (0 = ID, 1 = OOD).
        doc_ids: per-block document IDs (strings).
        weights: per-block weights, 1/k_d, so each document counts once.
        n_resamples: number of bootstrap resamples B.
        seed: seed for ``np.random.default_rng`` (an int, or [seed, i_id, i_ood]).
        level: CI level (0.95 gives the 2.5 and 97.5 percentiles).
        target_tpr: TPR for FPR@TPR (the YAML ``analysis.fpr_target_tpr``).

    Returns:
        name -> {``auroc``, ``auroc_ci``, ``fpr95``, ``fpr95_ci``, ``auroc_resamples``,
        ``fpr95_resamples``, ``n_id_docs``, ``n_ood_docs``}.
    """
    s_np, y, w, block_doc, n_id, n_ood = _doc_boot_inputs(scores, labels, doc_ids, weights)
    boot = _doc_boot_resamples(
        s_np, y, w, block_doc, n_id, n_ood, n_resamples, seed, target_tpr,
    )
    out: dict[str, dict[str, Any]] = {}
    for name, s in s_np.items():
        auc_r, fpr_r = boot[name]
        out[name] = {
            "auroc": auroc(s, y, sample_weight=w),
            "auroc_ci": _percentile_ci(auc_r, level),
            "fpr95": fpr_at_tpr(s, y, target_tpr, sample_weight=w),
            "fpr95_ci": _percentile_ci(fpr_r, level),
            "auroc_resamples": auc_r,
            "fpr95_resamples": fpr_r,
            "n_id_docs": n_id,
            "n_ood_docs": n_ood,
        }
    return out


def paired_doc_bootstrap(
    scores: dict[str, Any],
    labels,
    doc_ids,
    weights,
    pairs: Sequence[tuple[str, str]],
    n_resamples: int,
    seed: int | Sequence[int],
    level: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Paired document bootstrap of Delta = AUROC_w(a) - AUROC_w(b) for listed pairs.

    Uses the same draws as ``doc_bootstrap_auroc`` with the same seed. Two-sided
    p-value with a floor of 2/(B+1):
    p = min(1, 2 (1 + min(#{Delta_r <= 0}, #{Delta_r >= 0})) / (B + 1)).

    Args:
        scores: name -> per-block scores, all on the same blocks.
        labels: per-block labels (0 = ID, 1 = OOD).
        doc_ids: per-block document IDs (strings).
        weights: per-block weights, 1/k_d.
        pairs: (a, b) score-name pairs; Delta = AUROC_w(a) - AUROC_w(b).
        n_resamples: number of bootstrap resamples B.
        seed: seed for ``np.random.default_rng`` (an int, or [seed, i_id, i_ood]).
        level: CI level.

    Returns:
        (a, b) -> {``delta``, ``ci``, ``p``, ``resamples``}.
    """
    needed: list[str] = []
    for pair in pairs:
        for name in pair:
            if name not in scores:
                raise KeyError(f"score {name!r} is not in scores")
            if name not in needed:
                needed.append(name)
    s_np, y, w, block_doc, n_id, n_ood = _doc_boot_inputs(
        {name: scores[name] for name in needed}, labels, doc_ids, weights,
    )
    boot = _doc_boot_resamples(s_np, y, w, block_doc, n_id, n_ood, n_resamples, seed, None)
    points = {name: auroc(s, y, sample_weight=w) for name, s in s_np.items()}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for a, b in pairs:
        delta_r = boot[a][0] - boot[b][0]
        n_le = int(np.count_nonzero(delta_r <= 0))
        n_ge = int(np.count_nonzero(delta_r >= 0))
        out[(a, b)] = {
            "delta": points[a] - points[b],
            "ci": _percentile_ci(delta_r, level),
            "p": min(1.0, 2.0 * (1 + min(n_le, n_ge)) / (n_resamples + 1)),
            "resamples": delta_r,
        }
    return out


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the input order.

    Sort ascending (stable); p~_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)).
    """
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ValueError("p-values must be finite and in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    out = np.empty(m, dtype=np.float64)
    out[order] = adj_sorted
    return out
