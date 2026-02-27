"""A1/A2 unit tests — Bayesian layers, uncertainty invariants."""

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
    return {
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mi": mi,
    }


def _small_deterministic_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=256, block_size=32, n_layer=2, n_head=2,
        n_embd=64, dropout=0.0, bias=True,
        bayes_head=BayesConfig(enabled=False),
    )


def _small_bayesian_config() -> GPTConfig:
    """A1-style: only lm_head is Bayesian."""
    return GPTConfig(
        vocab_size=256, block_size=32, n_layer=2, n_head=2,
        n_embd=64, dropout=0.0, bias=True,
        bayes_head=BayesConfig(enabled=True, prior_std=1.0, kl_weight=1.0),
    )


def _small_ffn_bayesian_config() -> GPTConfig:
    """A2-style: FFN is Bayesian, head is deterministic (weight-tied)."""
    return GPTConfig(
        vocab_size=256, block_size=32, n_layer=2, n_head=2,
        n_embd=64, dropout=0.0, bias=True,
        bayes_head=BayesConfig(enabled=False),
        bayes_ffn=BayesConfig(enabled=True, prior_std=1.0, kl_weight=1.0),
    )


# --- BayesianLinear layer tests ---

class TestBayesianLinearKL:
    """KL divergence must be non-negative (mathematical invariant)."""

    def test_kl_nonnegative(self):
        """KL >= 0 across different configs (prior_std, bias)."""
        for std in [0.1, 0.5, 1.0, 2.0, 10.0]:
            layer = BayesianLinear(64, 256, prior_std=std)
            assert layer.kl_loss().item() >= 0.0, f"KL < 0 for prior_std={std}"
        layer_no_bias = BayesianLinear(64, 256, prior_std=1.0, bias=False)
        assert layer_no_bias.kl_loss().item() >= 0.0, "KL < 0 without bias"

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
        assert not torch.allclose(out1, out2), "Two forward passes produced identical output"

    def test_mean_forward_is_deterministic(self):
        layer = BayesianLinear(32, 64)
        x = torch.randn(4, 32)
        out1 = layer.mean_forward(x)
        out2 = layer.mean_forward(x)
        assert torch.allclose(out1, out2), "mean_forward is not deterministic"


class TestContextManagers:
    """Context managers for frozen sampling and mean weights."""

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


# --- A1: Model-level architecture tests ---

class TestSelectiveBayesian:
    """A1: only lm_head is Bayesian; transformer blocks are deterministic."""

    def test_no_weight_tying_when_bayesian(self):
        model = MiniGPT(_small_bayesian_config())
        assert hasattr(model.lm_head, "weight_mu")
        assert not hasattr(model.lm_head, "linear")

    def test_only_lm_head_is_bayesian(self):
        model = MiniGPT(_small_bayesian_config())
        assert isinstance(model.lm_head, BayesianLinear)
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

    def test_forward_body_shape(self):
        model = MiniGPT(_small_bayesian_config())
        x = torch.randint(0, 256, (2, 16))
        h = model.forward_body(x)
        assert h.shape == (2, 16, 64)  # (batch, seq_len, n_embd)


# --- Uncertainty metric tests ---

class TestUncertaintyMetrics:
    """MI invariants: MI=0 for deterministic, MI>=0 always, MI>0 for Bayesian."""

    def test_mi_zero_for_identical_probs(self):
        """If all N samples produce the same distribution, MI = 0."""
        n, seq_len, vocab = 10, 20, 50
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

    def test_mi_zero_for_deterministic_model(self):
        """A deterministic model produces identical logits every pass -> MI=0."""
        config = _small_deterministic_config()
        model = MiniGPT(config)
        model.eval()
        x = torch.randint(0, config.vocab_size, (1, 16))

        probs_list = []
        for _ in range(10):
            logits, _ = model(x)
            probs_list.append(torch.softmax(logits, dim=-1))
        probs = torch.stack(probs_list, dim=0)[:, 0]

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


# --- A2: FFN-Bayesian architecture tests ---

class TestFFNBayesian:
    """A2: FFN layers are Bayesian; head is deterministic with weight tying."""

    def test_weight_tying_when_ffn_bayesian(self):
        """Head is deterministic -> weight tying should be active."""
        model = MiniGPT(_small_ffn_bayesian_config())
        assert model.lm_head.linear.weight is model.token_emb.weight

    def test_ffn_layers_are_bayesian(self):
        """MLP.fc and MLP.proj in each block should be BayesianLinear."""
        model = MiniGPT(_small_ffn_bayesian_config())
        for i, block in enumerate(model.blocks):
            assert isinstance(block.mlp.fc, BayesianLinear), \
                f"block {i} MLP.fc should be BayesianLinear"
            assert isinstance(block.mlp.proj, BayesianLinear), \
                f"block {i} MLP.proj should be BayesianLinear"

    def test_attention_layers_are_deterministic(self):
        """Attention Q/K/V and proj should remain DeterministicLinear."""
        model = MiniGPT(_small_ffn_bayesian_config())
        for i, block in enumerate(model.blocks):
            assert isinstance(block.attn.qkv, DeterministicLinear), \
                f"block {i} attn.qkv should be DeterministicLinear"
            assert isinstance(block.attn.proj, DeterministicLinear), \
                f"block {i} attn.proj should be DeterministicLinear"

    def test_forward_body_stochastic(self):
        """With Bayesian FFN, forward_body should produce different outputs."""
        torch.manual_seed(0)
        model = MiniGPT(_small_ffn_bayesian_config())
        model.eval()
        x = torch.randint(0, 256, (2, 16))
        h1 = model.forward_body(x)
        h2 = model.forward_body(x)
        assert not torch.allclose(h1, h2), \
            "forward_body should be stochastic with Bayesian FFN"

    def test_mi_positive_for_ffn_bayesian_model(self):
        """FFN-Bayesian model should produce MI > 0 from full forward passes."""
        torch.manual_seed(42)
        config = _small_ffn_bayesian_config()
        model = MiniGPT(config)
        model.eval()
        x = torch.randint(0, config.vocab_size, (1, 16))

        probs_list = []
        for _ in range(30):
            logits, _ = model(x)
            probs_list.append(torch.softmax(logits, dim=-1))
        probs = torch.stack(probs_list, dim=0)[:, 0]

        metrics = _compute_token_metrics(probs)
        assert metrics["mi"].mean().item() > 0, \
            "MI should be > 0 for FFN-Bayesian model"


# --- Path detection tests ---

class TestHasBayesianBody:
    """_has_bayesian_body must route A1 vs A2 MC sampling correctly."""

    def test_head_only_bayesian(self):
        """A1 model: head Bayesian, body deterministic -> must NOT take A2 path."""
        from minigpt.uncertainty import _has_bayesian_body
        model = MiniGPT(_small_bayesian_config())
        assert not _has_bayesian_body(model)

    def test_ffn_bayesian(self):
        """A2 model: FFN Bayesian -> must take A2 path (full forward N times)."""
        from minigpt.uncertainty import _has_bayesian_body
        model = MiniGPT(_small_ffn_bayesian_config())
        assert _has_bayesian_body(model)
