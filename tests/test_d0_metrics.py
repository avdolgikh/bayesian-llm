"""D0 unit tests — Evaluation metrics.

Tests for OOD detection (AUROC, FPR@95TPR, AUPRC), calibration (ECE, NLL, Brier),
selective prediction (risk-coverage, AURC), and sequence-level aggregation.

All tests use synthetic data — no model checkpoints required.
"""

import numpy as np
import torch

from minigpt.uncertainty import (
    auroc,
    auprc,
    brier_score,
    ece,
    fpr_at_tpr,
    nll,
    risk_coverage_curve,
    aurc,
    aggregate_sequence_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_separation():
    """ID scores all low, OOD scores all high — perfect OOD detector."""
    id_scores = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
    ood_scores = torch.tensor([0.90, 0.91, 0.92, 0.93, 0.94])
    scores = torch.cat([id_scores, ood_scores])
    labels = torch.cat([torch.zeros(5), torch.ones(5)])  # 0=ID, 1=OOD
    return scores, labels


def _no_separation():
    """ID and OOD scores drawn from same distribution — random detector."""
    rng = torch.Generator().manual_seed(42)
    scores = torch.rand(200, generator=rng)
    labels = torch.cat([torch.zeros(100), torch.ones(100)])
    return scores, labels


def _partial_separation():
    """Overlapping but separable distributions."""
    rng = torch.Generator().manual_seed(42)
    id_scores = 0.3 + 0.1 * torch.randn(100, generator=rng)
    ood_scores = 0.7 + 0.1 * torch.randn(100, generator=rng)
    scores = torch.cat([id_scores, ood_scores])
    labels = torch.cat([torch.zeros(100), torch.ones(100)])
    return scores, labels


# ---------------------------------------------------------------------------
# OOD Detection: AUROC
# ---------------------------------------------------------------------------

class TestAUROC:
    def test_perfect_separation_gives_1(self):
        scores, labels = _perfect_separation()
        assert auroc(scores, labels) == 1.0

    def test_no_separation_gives_approximately_0_5(self):
        scores, labels = _no_separation()
        result = auroc(scores, labels)
        assert 0.35 <= result <= 0.65, f"Expected ~0.5, got {result}"

    def test_inverted_scores_gives_0(self):
        """If OOD has LOWER scores than ID, AUROC should be 0."""
        scores, labels = _perfect_separation()
        result = auroc(-scores, labels)
        assert result == 0.0

    def test_partial_separation_between_0_5_and_1(self):
        scores, labels = _partial_separation()
        result = auroc(scores, labels)
        assert 0.5 < result < 1.0

    def test_returns_float(self):
        scores, labels = _perfect_separation()
        result = auroc(scores, labels)
        assert isinstance(result, float)

    def test_accepts_numpy(self):
        """Should accept numpy arrays as well as tensors."""
        scores, labels = _perfect_separation()
        result = auroc(scores.numpy(), labels.numpy())
        assert result == 1.0


# ---------------------------------------------------------------------------
# OOD Detection: FPR@95TPR
# ---------------------------------------------------------------------------

class TestFPRAtTPR:
    def test_perfect_separation_gives_0(self):
        scores, labels = _perfect_separation()
        result = fpr_at_tpr(scores, labels, target_tpr=0.95)
        assert result == 0.0

    def test_no_separation_gives_high_fpr(self):
        scores, labels = _no_separation()
        result = fpr_at_tpr(scores, labels, target_tpr=0.95)
        assert result >= 0.5, f"Expected high FPR, got {result}"

    def test_range_0_to_1(self):
        scores, labels = _partial_separation()
        result = fpr_at_tpr(scores, labels, target_tpr=0.95)
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        scores, labels = _perfect_separation()
        result = fpr_at_tpr(scores, labels, target_tpr=0.95)
        assert isinstance(result, float)

    def test_custom_target_tpr(self):
        """FPR@90TPR should also work."""
        scores, labels = _perfect_separation()
        result = fpr_at_tpr(scores, labels, target_tpr=0.90)
        assert result == 0.0


# ---------------------------------------------------------------------------
# OOD Detection: AUPRC
# ---------------------------------------------------------------------------

class TestAUPRC:
    def test_perfect_separation_gives_1(self):
        scores, labels = _perfect_separation()
        result = auprc(scores, labels)
        assert result == 1.0

    def test_no_separation_gives_approximately_base_rate(self):
        """With 50/50 class balance, random AUPRC ~ 0.5."""
        scores, labels = _no_separation()
        result = auprc(scores, labels)
        assert 0.3 <= result <= 0.7, f"Expected ~0.5 base rate, got {result}"

    def test_partial_separation_between_baseline_and_1(self):
        scores, labels = _partial_separation()
        result = auprc(scores, labels)
        assert 0.5 < result <= 1.0

    def test_returns_float(self):
        scores, labels = _perfect_separation()
        result = auprc(scores, labels)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Calibration: ECE
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfectly_calibrated_gives_zero(self):
        """If predicted confidence matches accuracy exactly, ECE=0."""
        # 100 predictions with confidence 0.8, exactly 80 correct
        n = 1000
        probs = torch.full((n,), 0.8)
        targets_correct = torch.ones(800)
        targets_wrong = torch.zeros(200)
        # Shuffle to avoid ordering bias
        rng = torch.Generator().manual_seed(42)
        idx = torch.randperm(n, generator=rng)
        correct = torch.cat([targets_correct, targets_wrong])[idx]
        result = ece(probs, correct, n_bins=10)
        assert result < 0.05, f"Expected ~0 ECE for calibrated probs, got {result}"

    def test_overconfident_gives_high_ece(self):
        """Predicts 0.99 confidence but only 50% correct — badly calibrated."""
        n = 1000
        probs = torch.full((n,), 0.99)
        rng = torch.Generator().manual_seed(42)
        correct = (torch.rand(n, generator=rng) < 0.5).float()
        result = ece(probs, correct, n_bins=15)
        assert result > 0.3, f"Expected high ECE for overconfident, got {result}"

    def test_range_0_to_1(self):
        n = 200
        rng = torch.Generator().manual_seed(42)
        probs = torch.rand(n, generator=rng)
        correct = (torch.rand(n, generator=rng) > 0.5).float()
        result = ece(probs, correct, n_bins=15)
        assert 0.0 <= result <= 1.0

    def test_default_bins_is_15(self):
        """Spec says M=15 bins. Verify default works."""
        n = 200
        rng = torch.Generator().manual_seed(42)
        probs = torch.rand(n, generator=rng)
        correct = (torch.rand(n, generator=rng) > 0.5).float()
        # Should not raise — default n_bins=15
        result = ece(probs, correct)
        assert isinstance(result, float)

    def test_returns_float(self):
        probs = torch.tensor([0.9, 0.8, 0.7])
        correct = torch.tensor([1.0, 1.0, 0.0])
        result = ece(probs, correct, n_bins=3)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Calibration: NLL
# ---------------------------------------------------------------------------

class TestNLL:
    def test_perfect_prediction_gives_zero(self):
        """If model assigns probability 1.0 to correct token, NLL=0."""
        # (batch, vocab) — model is perfectly confident
        probs = torch.zeros(4, 10)
        targets = torch.tensor([0, 3, 5, 9])
        for i, t in enumerate(targets):
            probs[i, t] = 1.0
        result = nll(probs, targets)
        assert abs(result) < 1e-6

    def test_uniform_prediction_gives_log_vocab(self):
        """Uniform over 10 classes -> NLL = log(10) ~ 2.302."""
        vocab = 10
        probs = torch.full((100, vocab), 1.0 / vocab)
        rng = torch.Generator().manual_seed(42)
        targets = torch.randint(0, vocab, (100,), generator=rng)
        result = nll(probs, targets)
        expected = np.log(vocab)
        assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"

    def test_nll_is_nonnegative(self):
        rng = torch.Generator().manual_seed(42)
        logits = torch.randn(50, 20, generator=rng)
        probs = torch.softmax(logits, dim=-1)
        targets = torch.randint(0, 20, (50,), generator=rng)
        result = nll(probs, targets)
        assert result >= 0.0

    def test_returns_float(self):
        probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        targets = torch.tensor([0, 1])
        result = nll(probs, targets)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Calibration: Brier Score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_prediction_gives_zero(self):
        """Brier = 0 when model assigns all mass to correct class."""
        probs = torch.zeros(4, 10)
        targets = torch.tensor([0, 3, 5, 9])
        for i, t in enumerate(targets):
            probs[i, t] = 1.0
        result = brier_score(probs, targets)
        assert abs(result) < 1e-6

    def test_uniform_prediction_value(self):
        """Brier for uniform over K classes: 1 - 2/K + K*(1/K)^2 = 1 - 1/K."""
        vocab = 10
        probs = torch.full((100, vocab), 1.0 / vocab)
        rng = torch.Generator().manual_seed(42)
        targets = torch.randint(0, vocab, (100,), generator=rng)
        result = brier_score(probs, targets)
        expected = 1.0 - 1.0 / vocab  # = 0.9
        assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"

    def test_range_0_to_2(self):
        """Brier score is in [0, 2] for any distribution."""
        rng = torch.Generator().manual_seed(42)
        logits = torch.randn(50, 20, generator=rng)
        probs = torch.softmax(logits, dim=-1)
        targets = torch.randint(0, 20, (50,), generator=rng)
        result = brier_score(probs, targets)
        assert 0.0 <= result <= 2.0

    def test_worse_predictions_give_higher_brier(self):
        """Confident-wrong should score worse than uncertain."""
        targets = torch.tensor([0, 0, 0, 0])
        # Good: high probability on correct class
        good_probs = torch.tensor([
            [0.9, 0.1], [0.8, 0.2], [0.85, 0.15], [0.95, 0.05],
        ])
        # Bad: high probability on wrong class
        bad_probs = torch.tensor([
            [0.1, 0.9], [0.2, 0.8], [0.15, 0.85], [0.05, 0.95],
        ])
        assert brier_score(bad_probs, targets) > brier_score(good_probs, targets)

    def test_returns_float(self):
        probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        targets = torch.tensor([0, 1])
        result = brier_score(probs, targets)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Selective Prediction: Risk-Coverage Curve & AURC
# ---------------------------------------------------------------------------

class TestRiskCoverage:
    def test_full_coverage_equals_overall_risk(self):
        """At coverage=1.0, risk should equal the overall error rate."""
        rng = torch.Generator().manual_seed(42)
        uncertainties = torch.rand(100, generator=rng)
        correct = (torch.rand(100, generator=rng) > 0.3).float()
        coverages, risks = risk_coverage_curve(uncertainties, correct)

        overall_error = 1.0 - correct.mean().item()
        # Last point should be full coverage
        assert abs(coverages[-1] - 1.0) < 1e-6
        assert abs(risks[-1] - overall_error) < 0.02

    def test_risk_decreases_with_coverage(self):
        """Removing uncertain samples should generally decrease risk.

        With a good uncertainty score, risk at low coverage <= risk at full coverage.
        """
        # Construct case where uncertainty correlates with errors
        correct = torch.cat([torch.ones(80), torch.zeros(20)])
        # High uncertainty on wrong predictions
        uncertainties = torch.cat([
            0.1 + 0.1 * torch.rand(80),   # low uncertainty on correct
            0.8 + 0.1 * torch.rand(20),   # high uncertainty on wrong
        ])
        coverages, risks = risk_coverage_curve(uncertainties, correct)

        # Risk at 80% coverage should be lower than at 100%
        idx_80 = (coverages - 0.8).abs().argmin()
        assert risks[idx_80] < risks[-1]

    def test_returns_sorted_coverages(self):
        rng = torch.Generator().manual_seed(42)
        uncertainties = torch.rand(50, generator=rng)
        correct = (torch.rand(50, generator=rng) > 0.5).float()
        coverages, risks = risk_coverage_curve(uncertainties, correct)

        # Coverages should be monotonically increasing
        for i in range(len(coverages) - 1):
            assert coverages[i] <= coverages[i + 1]

    def test_coverages_and_risks_same_length(self):
        rng = torch.Generator().manual_seed(42)
        uncertainties = torch.rand(30, generator=rng)
        correct = (torch.rand(30, generator=rng) > 0.5).float()
        coverages, risks = risk_coverage_curve(uncertainties, correct)
        assert len(coverages) == len(risks)


class TestAURC:
    def test_perfect_uncertainty_gives_low_aurc(self):
        """If uncertainty perfectly flags errors, AURC should be near 0."""
        correct = torch.cat([torch.ones(80), torch.zeros(20)])
        # Perfect: wrong predictions have highest uncertainty
        uncertainties = torch.cat([torch.zeros(80), torch.ones(20)])
        result = aurc(uncertainties, correct)
        assert result < 0.05, f"Expected near-0 AURC, got {result}"

    def test_random_uncertainty_gives_higher_aurc(self):
        """Random uncertainty should give higher AURC than perfect."""
        correct = torch.cat([torch.ones(80), torch.zeros(20)])
        perfect_unc = torch.cat([torch.zeros(80), torch.ones(20)])
        rng = torch.Generator().manual_seed(42)
        random_unc = torch.rand(100, generator=rng)

        aurc_perfect = aurc(perfect_unc, correct)
        aurc_random = aurc(random_unc, correct)
        assert aurc_random > aurc_perfect

    def test_range_0_to_1(self):
        rng = torch.Generator().manual_seed(42)
        uncertainties = torch.rand(100, generator=rng)
        correct = (torch.rand(100, generator=rng) > 0.3).float()
        result = aurc(uncertainties, correct)
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        uncertainties = torch.tensor([0.1, 0.5, 0.9])
        correct = torch.tensor([1.0, 1.0, 0.0])
        result = aurc(uncertainties, correct)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Sequence-Level Aggregation
# ---------------------------------------------------------------------------

class TestAggregateSequenceScores:
    def test_mean_aggregation(self):
        """Mean across token positions."""
        token_scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        result = aggregate_sequence_scores(token_scores, method="mean")
        expected = 0.3
        assert abs(result - expected) < 1e-6

    def test_max_aggregation(self):
        """Max across token positions — worst-case token."""
        token_scores = torch.tensor([0.1, 0.2, 0.9, 0.4, 0.5])
        result = aggregate_sequence_scores(token_scores, method="max")
        expected = 0.9
        assert abs(result - expected) < 1e-6

    def test_proportion_aggregation(self):
        """Fraction of tokens above threshold."""
        token_scores = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.05])
        result = aggregate_sequence_scores(token_scores, method="proportion", threshold=0.5)
        expected = 2.0 / 5.0  # two tokens above 0.5
        assert abs(result - expected) < 1e-6

    def test_returns_float(self):
        token_scores = torch.tensor([0.1, 0.2, 0.3])
        result = aggregate_sequence_scores(token_scores, method="mean")
        assert isinstance(result, float)

    def test_unknown_method_raises(self):
        token_scores = torch.tensor([0.1, 0.2, 0.3])
        try:
            aggregate_sequence_scores(token_scores, method="unknown")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Cross-metric consistency
# ---------------------------------------------------------------------------

class TestCrossMetricConsistency:
    def test_auroc_and_fpr95_agree_on_perfect(self):
        """Perfect separation: AUROC=1.0 and FPR@95=0.0."""
        scores, labels = _perfect_separation()
        assert auroc(scores, labels) == 1.0
        assert fpr_at_tpr(scores, labels, target_tpr=0.95) == 0.0

    def test_better_separation_improves_all_ood_metrics(self):
        """Partial separation should give better AUROC, lower FPR@95, higher AUPRC
        than no separation."""
        scores_none, labels_none = _no_separation()
        scores_partial, labels_partial = _partial_separation()

        assert auroc(scores_partial, labels_partial) > auroc(scores_none, labels_none)
        assert fpr_at_tpr(scores_partial, labels_partial) < fpr_at_tpr(scores_none, labels_none)
        assert auprc(scores_partial, labels_partial) > auprc(scores_none, labels_none)

    def test_brier_and_nll_rank_same_direction(self):
        """Better predictions should score better on both Brier and NLL."""
        targets = torch.tensor([0, 1, 2, 0, 1])
        good_probs = torch.zeros(5, 3)
        for i, t in enumerate(targets):
            good_probs[i, t] = 0.8
            good_probs[i, (t + 1) % 3] = 0.1
            good_probs[i, (t + 2) % 3] = 0.1

        bad_probs = torch.full((5, 3), 1.0 / 3)

        assert brier_score(good_probs, targets) < brier_score(bad_probs, targets)
        assert nll(good_probs, targets) < nll(bad_probs, targets)
