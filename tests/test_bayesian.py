"""A1 unit tests — Bayesian layers, uncertainty invariants."""

import torch

from minigpt.layers import (
    BayesConfig,
    BayesianLinear,
    DeterministicLinear,
    frozen_bayesian_sample,
    use_mean_weights,
)
from minigpt.model import GPTConfig, MiniGPT


def _compute_token_metrics(probs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Batch computation of MI/entropy/flip_rate from stacked probs.

    Test utility — the production code uses streaming (_stream_metrics)
    to avoid OOM on large vocab tensors.

    Args:
        probs: (N, seq_len, vocab_size) — softmax outputs from N forward passes.
    """
    eps = 1e-10
    p_bar = probs.mean(dim=0)
    predictive_entropy = -(p_bar * torch.log(p_bar + eps)).sum(dim=-1)
    per_sample_entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    expected_entropy = per_sample_entropy.mean(dim=0)
    mi = predictive_entropy - expected_entropy
    top1_tokens = probs.argmax(dim=-1)
    mode_tokens = top1_tokens.mode(dim=0).values
    flip_rate = (top1_tokens != mode_tokens.unsqueeze(0)).float().mean(dim=0)
    return {
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mi": mi,
        "flip_rate": flip_rate,
    }


def _small_deterministic_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=256, block_size=32, n_layer=2, n_head=2,
        n_embd=64, dropout=0.0, bias=True,
        bayes=BayesConfig(enabled=False),
    )


def _small_bayesian_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=256, block_size=32, n_layer=2, n_head=2,
        n_embd=64, dropout=0.0, bias=True,
        bayes=BayesConfig(enabled=True, prior_std=1.0, kl_weight=1.0),
    )


# --- BayesianLinear layer tests ---

class TestBayesianLinearKL:
    """KL divergence must be non-negative (mathematical invariant)."""

    def test_kl_nonnegative_at_init(self):
        layer = BayesianLinear(64, 256, prior_std=1.0)
        assert layer.kl_loss().item() >= 0.0

    def test_kl_nonnegative_various_prior_std(self):
        for std in [0.1, 0.5, 1.0, 2.0, 10.0]:
            layer = BayesianLinear(32, 64, prior_std=std)
            assert layer.kl_loss().item() >= 0.0, f"KL < 0 for prior_std={std}"

    def test_kl_nonnegative_no_bias(self):
        layer = BayesianLinear(64, 256, prior_std=1.0, bias=False)
        assert layer.kl_loss().item() >= 0.0

    def test_deterministic_kl_is_zero(self):
        layer = DeterministicLinear(64, 256)
        assert layer.kl_loss().item() == 0.0


class TestBayesianLinearSampling:
    """Bayesian forward produces different outputs; mean_forward is deterministic."""

    def test_bayesian_forward_produces_variance(self):
        torch.manual_seed(0)
        layer = BayesianLinear(32, 64)
        x = torch.randn(4, 32)
        out1 = layer(x)
        out2 = layer(x)
        # Different forward passes should produce different outputs
        assert not torch.allclose(out1, out2), "Two forward passes produced identical output"

    def test_mean_forward_is_deterministic(self):
        layer = BayesianLinear(32, 64)
        x = torch.randn(4, 32)
        out1 = layer.mean_forward(x)
        out2 = layer.mean_forward(x)
        assert torch.allclose(out1, out2), "mean_forward is not deterministic"

    def test_mean_forward_with_variance_shapes(self):
        layer = BayesianLinear(32, 64)
        x = torch.randn(4, 32)
        out, var = layer.mean_forward_with_variance(x)
        assert out.shape == (4, 64)
        assert var.shape == (4, 64)
        assert (var >= 0).all(), "Variance must be non-negative"

    def test_use_mean_flag(self):
        layer = BayesianLinear(32, 64)
        x = torch.randn(4, 32)
        layer._use_mean = True
        out1 = layer(x)
        out2 = layer(x)
        layer._use_mean = False
        assert torch.allclose(out1, out2), "use_mean=True should be deterministic"


class TestFrozenSampling:
    """freeze_sample makes forward deterministic; unfreeze restores stochasticity."""

    def test_frozen_is_deterministic(self):
        torch.manual_seed(0)
        layer = BayesianLinear(32, 64)
        x = torch.randn(4, 32)
        layer.freeze_sample()
        out1 = layer(x)
        out2 = layer(x)
        layer.unfreeze_sample()
        assert torch.allclose(out1, out2), "Frozen forward should be deterministic"

    def test_unfreeze_restores_variance(self):
        torch.manual_seed(0)
        layer = BayesianLinear(32, 64)
        x = torch.randn(4, 32)
        layer.freeze_sample()
        layer.unfreeze_sample()
        out1 = layer(x)
        out2 = layer(x)
        assert not torch.allclose(out1, out2), "Unfrozen should produce variance"

    def test_frozen_context_manager(self):
        torch.manual_seed(0)
        model = MiniGPT(_small_bayesian_config())
        x = torch.randint(0, 256, (2, 16))
        with frozen_bayesian_sample(model):
            out1, _ = model(x)
            out2, _ = model(x)
        assert torch.allclose(out1, out2), "Frozen context should be deterministic"

    def test_use_mean_context_manager(self):
        model = MiniGPT(_small_bayesian_config())
        x = torch.randint(0, 256, (2, 16))
        with use_mean_weights(model):
            out1, _ = model(x)
            out2, _ = model(x)
        assert torch.allclose(out1, out2), "use_mean context should be deterministic"


# --- Model-level tests ---

class TestSelectiveBayesian:
    """A1: only lm_head is Bayesian; transformer blocks are deterministic."""

    def test_weight_tying_when_deterministic(self):
        model = MiniGPT(_small_deterministic_config())
        assert model.lm_head.linear.weight is model.token_emb.weight

    def test_no_weight_tying_when_bayesian(self):
        model = MiniGPT(_small_bayesian_config())
        # BayesianLinear has weight_mu, not .linear.weight
        assert hasattr(model.lm_head, "weight_mu")
        assert not hasattr(model.lm_head, "linear")

    def test_only_lm_head_is_bayesian(self):
        model = MiniGPT(_small_bayesian_config())
        # lm_head should be BayesianLinear
        assert isinstance(model.lm_head, BayesianLinear)
        # All other linear layers should be DeterministicLinear
        for name, module in model.named_modules():
            if isinstance(module, (BayesianLinear, DeterministicLinear)):
                if name == "lm_head":
                    assert isinstance(module, BayesianLinear), f"{name} should be Bayesian"
                else:
                    assert isinstance(module, DeterministicLinear), \
                        f"{name} should be Deterministic"

    def test_kl_positive_for_bayesian_model(self):
        model = MiniGPT(_small_bayesian_config())
        kl = model.kl_loss().item()
        assert kl > 0, "Bayesian model should have KL > 0"

    def test_kl_zero_for_deterministic_model(self):
        model = MiniGPT(_small_deterministic_config())
        kl = model.kl_loss().item()
        assert kl == 0.0, "Deterministic model should have KL = 0"

    def test_forward_body_shape(self):
        model = MiniGPT(_small_bayesian_config())
        x = torch.randint(0, 256, (2, 16))
        h = model.forward_body(x)
        assert h.shape == (2, 16, 64)  # (batch, seq_len, n_embd)


# --- Uncertainty metric tests ---

class TestUncertaintyMetrics:
    """MI invariants: MI=0 for deterministic, MI>=0 always."""

    def test_mi_zero_for_identical_probs(self):
        """If all N samples produce the same distribution, MI = 0."""
        n, seq_len, vocab = 10, 20, 50
        # Same probability distribution repeated N times
        single_probs = torch.softmax(torch.randn(seq_len, vocab), dim=-1)
        probs = single_probs.unsqueeze(0).expand(n, -1, -1)
        metrics = _compute_token_metrics(probs)
        assert torch.allclose(metrics["mi"], torch.zeros(seq_len), atol=1e-5), \
            f"MI should be 0 for identical distributions, got {metrics['mi'].max():.6f}"

    def test_mi_nonnegative(self):
        """MI >= 0 always (H[p_bar] >= E_bar by Jensen's inequality)."""
        n, seq_len, vocab = 10, 20, 50
        probs = torch.softmax(torch.randn(n, seq_len, vocab), dim=-1)
        metrics = _compute_token_metrics(probs)
        assert (metrics["mi"] >= -1e-6).all(), f"MI has negative values: {metrics['mi'].min():.6f}"

    def test_predictive_entropy_ge_expected_entropy(self):
        """H[p_bar] >= E_bar (concavity of entropy)."""
        n, seq_len, vocab = 10, 20, 50
        probs = torch.softmax(torch.randn(n, seq_len, vocab), dim=-1)
        metrics = _compute_token_metrics(probs)
        diff = metrics["predictive_entropy"] - metrics["expected_entropy"]
        assert (diff >= -1e-6).all(), f"Predictive entropy < expected entropy: {diff.min():.6f}"

    def test_flip_rate_range(self):
        """Flip rate is in [0, 1]."""
        n, seq_len, vocab = 10, 20, 50
        probs = torch.softmax(torch.randn(n, seq_len, vocab), dim=-1)
        metrics = _compute_token_metrics(probs)
        assert (metrics["flip_rate"] >= 0).all()
        assert (metrics["flip_rate"] <= 1).all()

    def test_mi_zero_for_deterministic_model(self):
        """A deterministic model produces identical logits every pass → MI=0."""
        config = _small_deterministic_config()
        model = MiniGPT(config)
        model.eval()
        x = torch.randint(0, config.vocab_size, (1, 16))

        # N forward passes with deterministic model → same logits
        probs_list = []
        for _ in range(10):
            logits, _ = model(x)
            probs_list.append(torch.softmax(logits, dim=-1))
        probs = torch.stack(probs_list, dim=0)[:, 0]  # (N, seq_len, vocab)

        metrics = _compute_token_metrics(probs)
        assert metrics["mi"].max().item() < 1e-5, \
            f"MI should be ~0 for deterministic model, got {metrics['mi'].max():.6f}"

    def test_mi_positive_for_bayesian_model(self):
        """A Bayesian model should produce MI > 0 (weight samples disagree)."""
        torch.manual_seed(42)
        config = _small_bayesian_config()
        model = MiniGPT(config)
        model.eval()
        x = torch.randint(0, config.vocab_size, (1, 16))

        h = model.forward_body(x)
        probs_list = []
        for _ in range(30):
            logits = model.lm_head(h)
            probs_list.append(torch.softmax(logits, dim=-1))
        probs = torch.stack(probs_list, dim=0)[:, 0]

        metrics = _compute_token_metrics(probs)
        assert metrics["mi"].mean().item() > 0, \
            "MI should be > 0 for Bayesian model (weight samples should disagree)"
