"""P0/P1 model guardrails — weight tying, perplexity bounds."""

import torch

from minigpt.layers import BayesConfig
from minigpt.model import GPTConfig, MiniGPT
from minigpt.evaluate import compute_perplexity


def _small_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=256,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
        bayes=BayesConfig(enabled=False),
    )


class TestWeightTying:
    """P0: Weight tying pointer equality.

    If broken, the model gets vocab_size * n_embd extra free parameters
    in the head. It will memorize instead of generalize.
    """

    def test_weight_tying_at_init(self):
        model = MiniGPT(_small_config())
        assert model.lm_head.linear.weight is model.token_emb.weight

    def test_weight_tying_after_forward(self):
        model = MiniGPT(_small_config())
        x = torch.randint(0, 256, (2, 16))
        model(x)
        assert model.lm_head.linear.weight is model.token_emb.weight

    def test_weight_tying_after_backward(self):
        model = MiniGPT(_small_config())
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        _, loss = model(x, y)
        loss.backward()
        assert model.lm_head.linear.weight is model.token_emb.weight


class TestPerplexityBounds:
    """P1: Perplexity bounds at init.

    At random init, perplexity should be ~vocab_size. If it's much lower,
    the loss computation is broken but training still runs and "converges."
    """

    def test_perplexity_lower_bound(self):
        """Perplexity >= 1.0 (mathematical invariant: exp(cross_entropy) >= 1)."""
        config = _small_config()
        model = MiniGPT(config)
        data = torch.randint(0, config.vocab_size, (2000,))
        ppl = compute_perplexity(model, data, block_size=32, batch_size=4,
                                 device=torch.device("cpu"), n_batches=5)
        assert ppl >= 1.0

    def test_perplexity_near_vocab_size_at_init(self):
        """At random init, perplexity should be roughly vocab_size (uniform distribution)."""
        config = _small_config()
        model = MiniGPT(config)
        data = torch.randint(0, config.vocab_size, (2000,))
        ppl = compute_perplexity(model, data, block_size=32, batch_size=4,
                                 device=torch.device("cpu"), n_batches=5)
        # Should be in the same order of magnitude as vocab_size (256)
        # Allow wide margin — not exact, but must not be suspiciously low
        assert ppl > config.vocab_size * 0.1, (
            f"Perplexity at init ({ppl:.1f}) is suspiciously low for vocab_size={config.vocab_size}"
        )
