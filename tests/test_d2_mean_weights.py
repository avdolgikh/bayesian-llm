"""D2 unit tests — mean-weights inference and MC-averaged perplexity."""

import statistics

import pytest
import torch
import torch.nn as nn

from minigpt.evaluate import compute_perplexity, compute_perplexity_mc
from minigpt.layers import BayesConfig, use_mean_weights
from minigpt.lora import BLoBLoRALinear, LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT
from minigpt.train import get_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _det_config() -> GPTConfig:
    """Deterministic miniGPT — no Bayesian layers."""
    return GPTConfig(
        vocab_size=256, block_size=32, n_layer=2, n_head=2,
        n_embd=64, dropout=0.0, bias=True,
    )


def _ffn_bayes_config() -> GPTConfig:
    """A2-style miniGPT — Bayesian FFN, deterministic head."""
    return GPTConfig(
        vocab_size=256, block_size=32, n_layer=2, n_head=2,
        n_embd=64, dropout=0.0, bias=True,
        bayes_ffn=BayesConfig(enabled=True, prior_std=1.0),
    )


def _lora_config() -> LoRAConfig:
    return LoRAConfig(
        rank=4, alpha=8.0, prior_std=0.2, init_g=0.05,
        target="ffn",
    )


def _det_model() -> MiniGPT:
    torch.manual_seed(42)
    return MiniGPT(_det_config())


def _ffn_bayes_model() -> MiniGPT:
    torch.manual_seed(42)
    return MiniGPT(_ffn_bayes_config())


def _lora_model() -> MiniGPT:
    """BLoB LoRA model with nonzero lora_B (actual stochasticity)."""
    torch.manual_seed(42)
    model = MiniGPT(_det_config())
    inject_lora(model, _lora_config())
    for m in model.modules():
        if isinstance(m, BLoBLoRALinear):
            with torch.no_grad():
                nn.init.normal_(m.lora_B, std=0.1)
    return model


def _data() -> torch.Tensor:
    torch.manual_seed(99)
    return torch.randint(0, 256, (2000,))


_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Tests: compute_perplexity_mc
# ---------------------------------------------------------------------------


class TestComputePerplexityMC:
    """MC-averaged perplexity: average cross-entropy over N weight samples."""

    def test_returns_positive_float(self):
        """Basic smoke test: returns a positive float."""
        model = _ffn_bayes_model()
        ppl = compute_perplexity_mc(
            model, _data(), block_size=32, batch_size=4,
            device=_DEVICE, n_samples=3, n_batches=5,
        )
        assert isinstance(ppl, float), f"Expected float, got {type(ppl)}"
        assert ppl > 0, f"Perplexity must be positive, got {ppl}"

    def test_deterministic_model_matches_single_pass(self):
        """For deterministic model, MC averaging equals single-pass perplexity."""
        model = _det_model()
        data = _data()
        torch.manual_seed(0)
        ppl_single = compute_perplexity(
            model, data, block_size=32, batch_size=4,
            device=_DEVICE, n_batches=5,
        )
        torch.manual_seed(0)
        ppl_mc = compute_perplexity_mc(
            model, data, block_size=32, batch_size=4,
            device=_DEVICE, n_samples=5, n_batches=5,
        )
        assert ppl_mc == pytest.approx(ppl_single, rel=1e-5), (
            f"Deterministic model: MC ppl ({ppl_mc:.4f}) must equal "
            f"single-pass ppl ({ppl_single:.4f})"
        )

    def test_bayesian_n1_matches_single_pass(self):
        """For Bayesian model, MC with N=1 equals single-pass (same random state)."""
        model = _ffn_bayes_model()
        data = _data()
        torch.manual_seed(0)
        ppl_single = compute_perplexity(
            model, data, block_size=32, batch_size=4,
            device=_DEVICE, n_batches=5,
        )
        torch.manual_seed(0)
        ppl_mc1 = compute_perplexity_mc(
            model, data, block_size=32, batch_size=4,
            device=_DEVICE, n_samples=1, n_batches=5,
        )
        assert ppl_mc1 == pytest.approx(ppl_single, rel=1e-5), (
            f"MC N=1 ({ppl_mc1:.4f}) must equal single-pass ({ppl_single:.4f})"
        )

    def test_bayesian_n1_differs_from_n20(self):
        """N=1 and N=20 give different perplexity (MC averaging changes result)."""
        model = _ffn_bayes_model()
        data = _data()
        torch.manual_seed(10)
        ppl_n1 = compute_perplexity_mc(
            model, data, block_size=32, batch_size=4,
            device=_DEVICE, n_samples=1, n_batches=1,
        )
        torch.manual_seed(10)
        ppl_n20 = compute_perplexity_mc(
            model, data, block_size=32, batch_size=4,
            device=_DEVICE, n_samples=20, n_batches=1,
        )
        assert ppl_n1 != pytest.approx(ppl_n20, rel=1e-3), (
            f"Bayesian model: N=1 ({ppl_n1:.4f}) and N=20 ({ppl_n20:.4f}) "
            f"should differ (MC averaging changes the result)"
        )

    def test_lora_returns_valid_perplexity(self):
        """MC-averaged perplexity works with BLoB LoRA model."""
        model = _lora_model()
        ppl = compute_perplexity_mc(
            model, _data(), block_size=32, batch_size=4,
            device=_DEVICE, n_samples=5, n_batches=5,
        )
        assert isinstance(ppl, float), f"Expected float, got {type(ppl)}"
        assert ppl > 0, f"LoRA MC perplexity must be positive, got {ppl}"


# ---------------------------------------------------------------------------
# Tests: mean-weights perplexity
# ---------------------------------------------------------------------------


class TestMeanWeightsPerplexity:
    """Mean-weights perplexity: use_mean_weights context + compute_perplexity."""

    def test_ffn_deterministic_across_calls(self):
        """Two calls with same seed and mean weights give identical ppl."""
        model = _ffn_bayes_model()
        data = _data()
        with use_mean_weights(model):
            torch.manual_seed(0)
            ppl1 = compute_perplexity(
                model, data, block_size=32, batch_size=4,
                device=_DEVICE, n_batches=5,
            )
            torch.manual_seed(0)
            ppl2 = compute_perplexity(
                model, data, block_size=32, batch_size=4,
                device=_DEVICE, n_batches=5,
            )
        assert ppl1 == ppl2, (
            f"Mean-weights ppl must be deterministic: {ppl1} vs {ppl2}"
        )

    def test_ffn_differs_from_sampled(self):
        """Mean-weights ppl != single random-sample ppl for Bayesian model.

        Uses a single batch to maximize sensitivity to weight sampling noise.
        With n_batches=10 the per-batch noise averages out and the gap can
        fall below tolerance; n_batches=1 keeps the full MC-vs-mean signal.
        """
        model = _ffn_bayes_model()
        data = _data()
        # Collect multiple single-sample ppls — at least one must differ from mean
        with use_mean_weights(model):
            torch.manual_seed(0)
            ppl_mean = compute_perplexity(
                model, data, block_size=32, batch_size=4,
                device=_DEVICE, n_batches=1,
            )
        any_differ = False
        for seed in range(5):
            torch.manual_seed(seed)
            ppl_sampled = compute_perplexity(
                model, data, block_size=32, batch_size=4,
                device=_DEVICE, n_batches=1,
            )
            if abs(ppl_mean - ppl_sampled) / ppl_mean > 1e-4:
                any_differ = True
                break
        assert any_differ, (
            f"Mean-weights ({ppl_mean:.4f}) should differ from at least one "
            f"sampled forward pass for a Bayesian model"
        )

    def test_lora_deterministic_across_calls(self):
        """Mean-weights is also deterministic for LoRA model."""
        model = _lora_model()
        data = _data()
        with use_mean_weights(model):
            torch.manual_seed(0)
            ppl1 = compute_perplexity(
                model, data, block_size=32, batch_size=4,
                device=_DEVICE, n_batches=5,
            )
            torch.manual_seed(0)
            ppl2 = compute_perplexity(
                model, data, block_size=32, batch_size=4,
                device=_DEVICE, n_batches=5,
            )
        assert ppl1 == ppl2, (
            f"LoRA mean-weights ppl must be deterministic: {ppl1} vs {ppl2}"
        )


# ---------------------------------------------------------------------------
# Tests: MC variance reduction
# ---------------------------------------------------------------------------


class TestMCVarianceReduction:
    """MC averaging reduces variance: more samples -> more stable estimate."""

    def test_averaging_reduces_loss_variance(self):
        """Averaging N=10 forward-pass losses has lower std than single passes."""
        model = _ffn_bayes_model()
        model.eval()
        torch.manual_seed(0)
        x, y = get_batch(_data(), 32, 4, _DEVICE)

        # M=20 single-pass losses (N=1 each)
        single_losses = []
        for _ in range(20):
            _, loss = model(x, y)
            single_losses.append(loss.item())

        # M=20 averaged losses (N=10 each)
        avg_losses = []
        for _ in range(20):
            batch_total = 0.0
            for _ in range(10):
                _, loss = model(x, y)
                batch_total += loss.item()
            avg_losses.append(batch_total / 10)

        std_single = statistics.stdev(single_losses)
        std_avg = statistics.stdev(avg_losses)

        assert std_avg < std_single, (
            f"Averaging N=10 should reduce loss variance: "
            f"std(single)={std_single:.6f}, std(avg)={std_avg:.6f}"
        )
