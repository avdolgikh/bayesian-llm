"""B2 unit tests -- BLoB-style Bayesian LoRA.

Tests are written BEFORE implementation (TDD Stage #2).
They target the planned `minigpt/lora.py` module and extensions to
`minigpt/layers.py` and `minigpt/uncertainty.py`.
"""

import pytest
import torch
import torch.nn as nn

from minigpt.layers import (
    BayesConfig,
    frozen_bayesian_sample,
    sigma_summary,
    use_mean_weights,
)
from minigpt.lora import BLoBLoRALinear, LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT
from minigpt.uncertainty import _has_bayesian_body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_gpt_config() -> GPTConfig:
    """Minimal deterministic miniGPT for fast tests."""
    return GPTConfig(
        vocab_size=256, block_size=16, n_layer=2, n_head=2,
        n_embd=32, dropout=0.0, bias=True,
        bayes_head=BayesConfig(enabled=False),
        bayes_ffn=BayesConfig(enabled=False),
        bayes_attn_v=BayesConfig(enabled=False),
    )


def _small_gpt() -> MiniGPT:
    torch.manual_seed(42)
    return MiniGPT(_small_gpt_config())


def _default_lora_config() -> LoRAConfig:
    return LoRAConfig(rank=4, alpha=8.0, target="ffn", prior_std=0.2, init_g=0.05)


def _make_base_linear(in_f: int = 32, out_f: int = 64) -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(in_f, out_f, bias=True)


def _make_stochastic_layer(in_f: int = 32, out_f: int = 64) -> BLoBLoRALinear:
    """BLoBLoRALinear with nonzero lora_B.

    At init lora_B=0, so LoRA output is always zero regardless of A sampling.
    Tests that require actual stochasticity must use this helper instead.
    """
    layer = BLoBLoRALinear(_make_base_linear(in_f, out_f), rank=4, alpha=8.0,
                           prior_std=0.2, init_g=0.05)
    with torch.no_grad():
        nn.init.normal_(layer.lora_B, std=0.1)
    return layer


def _gpt_with_stochastic_lora() -> MiniGPT:
    """miniGPT with LoRA injected and lora_B nonzero for stochasticity testing.

    At init lora_B=0, so model output is deterministic despite Bayesian A.
    Tests that require MC variation must use this helper instead.
    """
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    for m in model.modules():
        if isinstance(m, BLoBLoRALinear):
            with torch.no_grad():
                nn.init.normal_(m.lora_B, std=0.1)
    return model


# ---------------------------------------------------------------------------
# Group 1 -- BLoBLoRALinear layer
# ---------------------------------------------------------------------------

def test_blob_linear_output_shape():
    """Forward output shape (batch, seq, d_out) is correct."""
    layer = BLoBLoRALinear(_make_base_linear(32, 64), rank=4, alpha=8.0, prior_std=0.2, init_g=0.05)
    x = torch.randn(2, 8, 32)
    y = layer(x)
    assert y.shape == (2, 8, 64)


def test_blob_linear_init_zero_delta():
    """At init (B=0), mean_forward(x) equals base_linear(x) exactly."""
    torch.manual_seed(1)
    base = _make_base_linear(32, 64)
    layer = BLoBLoRALinear(base, rank=4, alpha=8.0, prior_std=0.2, init_g=0.05)
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        expected = base(x)
        actual = layer.mean_forward(x)
    assert torch.allclose(expected, actual, atol=1e-6), \
        "At B=0 init, mean_forward must equal base_linear(x)"


def test_blob_linear_init_g_range():
    """lora_A_g is initialized in U(init_g/sqrt(2), init_g) and strictly positive."""
    torch.manual_seed(99)
    init_g = 0.05
    layer = BLoBLoRALinear(_make_base_linear(32, 64), rank=4, alpha=8.0,
                           prior_std=0.2, init_g=init_g)
    lo = init_g / 2 ** 0.5
    hi = init_g
    assert (layer.lora_A_g > 0).all(), "lora_A_g must be strictly positive (sigma = G^2 > 0)"
    assert (layer.lora_A_g >= lo - 1e-6).all(), \
        f"lora_A_g below lower bound {lo:.5f}"
    assert (layer.lora_A_g <= hi + 1e-6).all(), \
        f"lora_A_g above upper bound {hi:.5f}"


def test_blob_linear_numerical_forward():
    """mean_forward with known weights matches hand-computed formula."""
    in_f, out_f, rank = 4, 8, 2
    base = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    layer = BLoBLoRALinear(base, rank=rank, alpha=4.0, prior_std=0.2, init_g=0.05)
    with torch.no_grad():
        layer.lora_B.fill_(1.0)     # [out_f, rank] all ones
        layer.lora_A_mu.fill_(1.0)  # [rank, in_f] all ones
    # scaling = alpha/rank = 2.0
    # base_out = 0 (weight=0, no bias)
    # A = M = [[1,1,1,1],[1,1,1,1]]
    # x @ A.T = [[4,4]]  (sum of 4 ones per row)
    # [[4,4]] @ B.T = [[8,...,8]]  (4+4 per output)
    # * 2.0 = [[16,...,16]]
    x = torch.ones(1, 1, in_f)
    with torch.no_grad():
        y = layer.mean_forward(x)
    expected = torch.full((1, 1, out_f), 16.0)
    assert torch.allclose(y, expected), \
        f"Numerical forward mismatch: got {y}, expected {expected}"


def test_blob_linear_sampling_stochastic():
    """Two stochastic forward passes produce different outputs."""
    torch.manual_seed(2)
    layer = _make_stochastic_layer()  # lora_B nonzero so A sampling affects output
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x)
    assert not torch.allclose(y1, y2), "Stochastic forward passes must differ"


def test_blob_linear_freeze_sample_coherent():
    """After freeze_sample(), repeated forwards are identical."""
    torch.manual_seed(3)
    layer = _make_stochastic_layer()  # lora_B nonzero so freeze has observable effect
    x = torch.randn(1, 8, 32)
    layer.freeze_sample()
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x)
    layer.unfreeze_sample()
    assert torch.allclose(y1, y2), "Frozen sample must produce identical outputs"


def test_blob_linear_unfreeze_restores_stochasticity():
    """After unfreeze_sample(), outputs differ across calls."""
    torch.manual_seed(4)
    layer = _make_stochastic_layer()  # lora_B nonzero so A sampling affects output
    x = torch.randn(1, 8, 32)
    layer.freeze_sample()
    layer.unfreeze_sample()
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x)
    assert not torch.allclose(y1, y2), "Unfrozen layer must restore stochasticity"


def test_blob_linear_kl_at_prior():
    """KL ≈ 0 when M=0 and sigma=prior_std (G = sqrt(prior_std))."""
    prior_std = 0.2
    layer = BLoBLoRALinear(_make_base_linear(32, 64), rank=4, alpha=8.0, prior_std=prior_std, init_g=0.05)
    with torch.no_grad():
        layer.lora_A_mu.zero_()
        layer.lora_A_g.fill_(prior_std ** 0.5)  # sigma = G^2 = prior_std
    kl = layer.kl_loss()
    assert kl.item() == pytest.approx(0.0, abs=1e-4), \
        "KL must be ~0 when posterior equals prior"


def test_blob_linear_kl_positive_default():
    """KL > 0 with default Kaiming-initialized M."""
    layer = BLoBLoRALinear(_make_base_linear(32, 64), rank=4, alpha=8.0, prior_std=0.2, init_g=0.05)
    assert layer.kl_loss().item() > 0.0


def test_blob_linear_kl_gradients():
    """kl_loss().backward() produces grads on lora_A_mu and lora_A_g."""
    layer = BLoBLoRALinear(_make_base_linear(32, 64), rank=4, alpha=8.0, prior_std=0.2, init_g=0.05)
    layer.kl_loss().backward()
    assert layer.lora_A_mu.grad is not None, "lora_A_mu must receive KL gradient"
    assert layer.lora_A_g.grad is not None, "lora_A_g must receive KL gradient"


def test_blob_linear_forward_gradients():
    """Forward pass backpropagates gradients to lora_B, lora_A_mu, and lora_A_g."""
    torch.manual_seed(5)
    layer = _make_stochastic_layer()  # lora_B nonzero so gradients flow through LoRA path
    x = torch.randn(1, 8, 32)
    layer(x).sum().backward()
    assert layer.lora_B.grad is not None, "lora_B must receive gradient from forward pass"
    assert layer.lora_A_mu.grad is not None, "lora_A_mu must receive gradient from forward pass"
    assert layer.lora_A_g.grad is not None, "lora_A_g must receive gradient from forward pass"


def test_blob_linear_base_frozen():
    """base_linear weight and bias have requires_grad=False inside BLoBLoRALinear."""
    base = _make_base_linear(32, 64)  # bias=True
    assert base.weight.requires_grad, "Pre-condition: base weight should have grad before wrapping"
    layer = BLoBLoRALinear(base, rank=4, alpha=8.0, prior_std=0.2, init_g=0.05)
    assert not layer.base_linear.weight.requires_grad, \
        "BLoBLoRALinear must freeze base_linear.weight"
    if layer.base_linear.bias is not None:
        assert not layer.base_linear.bias.requires_grad, \
            "BLoBLoRALinear must freeze base_linear.bias"


# ---------------------------------------------------------------------------
# Group 2 -- LoRAConfig
# ---------------------------------------------------------------------------

def test_lora_config_defaults():
    """LoRAConfig has expected defaults."""
    cfg = LoRAConfig()
    assert cfg.rank == 8
    assert cfg.alpha == 16.0
    assert cfg.target == "ffn"
    assert cfg.prior_std == 0.2
    assert cfg.init_g == 0.05


def test_lora_scaling():
    """layer.scaling == alpha / rank."""
    layer = BLoBLoRALinear(_make_base_linear(32, 64), rank=4, alpha=12.0, prior_std=0.2, init_g=0.05)
    assert layer.scaling == pytest.approx(12.0 / 4)


# ---------------------------------------------------------------------------
# Group 3 -- inject_lora
# ---------------------------------------------------------------------------

def test_inject_lora_freezes_base():
    """Zero non-LoRA params have requires_grad=True after injection."""
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    non_lora_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "lora_" not in n
    ]
    assert non_lora_trainable == [], \
        f"Non-LoRA params should be frozen, found: {non_lora_trainable}"


def test_inject_lora_only_lora_trainable():
    """All trainable param names contain 'lora_'."""
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, "Must have at least some trainable LoRA params"
    assert all("lora_" in n for n in trainable), \
        f"Non-LoRA trainable params found: {[n for n in trainable if 'lora_' not in n]}"


def test_inject_lora_replaces_ffn():
    """block.mlp.fc and block.mlp.proj become BLoBLoRALinear for every block."""
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    for i, block in enumerate(model.blocks):
        assert isinstance(block.mlp.fc, BLoBLoRALinear), \
            f"block[{i}].mlp.fc must be BLoBLoRALinear"
        assert isinstance(block.mlp.proj, BLoBLoRALinear), \
            f"block[{i}].mlp.proj must be BLoBLoRALinear"


def test_inject_lora_non_ffn_unchanged():
    """Attention projections, embeddings, and lm_head are NOT BLoBLoRALinear."""
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    for i, block in enumerate(model.blocks):
        assert not isinstance(block.attn.q_proj, BLoBLoRALinear), \
            f"block[{i}].attn.q_proj must not be LoRA"
        assert not isinstance(block.attn.k_proj, BLoBLoRALinear), \
            f"block[{i}].attn.k_proj must not be LoRA"
        assert not isinstance(block.attn.v_proj, BLoBLoRALinear), \
            f"block[{i}].attn.v_proj must not be LoRA"
    assert not isinstance(model.lm_head, BLoBLoRALinear), "lm_head must not be LoRA"


def test_inject_lora_adapter_count():
    """Number of BLoBLoRALinear modules == 2 * n_layer (fc + proj per block)."""
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    adapters = [m for m in model.modules() if isinstance(m, BLoBLoRALinear)]
    expected = 2 * model.config.n_layer
    assert len(adapters) == expected, f"Expected {expected} adapters, got {len(adapters)}"


def test_inject_lora_preserves_output():
    """After injection, mean-weight logits are identical to pre-injection logits."""
    torch.manual_seed(10)
    model = _small_gpt()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        logits_before, _ = model(x)
    inject_lora(model, _default_lora_config())
    with use_mean_weights(model):
        with torch.no_grad():
            logits_after, _ = model(x)
    assert torch.allclose(logits_before, logits_after, atol=1e-5), \
        "injection with B=0 must preserve model output"


# ---------------------------------------------------------------------------
# Group 4 -- KL aggregation
# ---------------------------------------------------------------------------

def test_kl_aggregation_matches_sum():
    """model.kl_loss() equals sum of individual adapter kl_loss() values."""
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    individual_kl = sum(
        m.kl_loss().item() for m in model.modules() if isinstance(m, BLoBLoRALinear)
    )
    model_kl = model.kl_loss().item()
    assert model_kl == pytest.approx(individual_kl, rel=1e-5)


def test_kl_positive_after_injection():
    """model.kl_loss() > 0 with default init (M != 0)."""
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    assert model.kl_loss().item() > 0.0


# ---------------------------------------------------------------------------
# Group 5 -- Integration with layers.py / uncertainty.py
# ---------------------------------------------------------------------------

def test_has_bayesian_body_detects_lora():
    """_has_bayesian_body returns False before, True after inject_lora."""
    model = _small_gpt()
    assert not _has_bayesian_body(model), "Pre-injection: must be deterministic"
    inject_lora(model, _default_lora_config())
    assert _has_bayesian_body(model), "Post-injection: must detect LoRA stochasticity"


def test_use_mean_weights_disables_sampling():
    """Inside use_mean_weights, model output is deterministic across calls."""
    model = _gpt_with_stochastic_lora()  # lora_B nonzero so context manager is actually exercised
    x = torch.randint(0, 256, (1, 8))
    with use_mean_weights(model):
        with torch.no_grad():
            y1, _ = model(x)
            y2, _ = model(x)
    assert torch.allclose(y1, y2), "Mean-weight mode must be deterministic"


def test_frozen_bayesian_sample_context():
    """Inside frozen_bayesian_sample, model output is coherent across calls."""
    model = _gpt_with_stochastic_lora()  # lora_B nonzero so context manager is actually exercised
    x = torch.randint(0, 256, (1, 8))
    with frozen_bayesian_sample(model):
        with torch.no_grad():
            y1, _ = model(x)
            y2, _ = model(x)
    assert torch.allclose(y1, y2), "Frozen sample context must be coherent"


def test_sigma_summary_includes_lora():
    """sigma_summary() returns G^2 statistics in the expected range for init_g=0.05."""
    model = _small_gpt()
    inject_lora(model, _default_lora_config())
    stats = sigma_summary(model)
    assert len(stats) > 0, "sigma_summary must return stats for LoRA model"
    assert "sigma_mean" in stats
    # init_g=0.05 → G in [0.05/sqrt(2), 0.05] → sigma=G^2 in [~0.00125, ~0.0025]
    assert 5e-4 < stats["sigma_mean"] < 5e-3, \
        f"sigma_mean {stats['sigma_mean']:.6f} not in expected G^2 range for init_g=0.05"


# ---------------------------------------------------------------------------
# Group 6 -- MC variation at model level
# ---------------------------------------------------------------------------

def test_mc_sampling_produces_variation():
    """Two stochastic full-model forward passes produce different logits."""
    model = _gpt_with_stochastic_lora()  # lora_B nonzero so A sampling propagates to output
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        y1, _ = model(x)
        y2, _ = model(x)
    assert not torch.allclose(y1, y2), "MC samples must differ at model level"


# ---------------------------------------------------------------------------
# Group 7 -- Checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    """Save LoRA model state → load into fresh injected model → identical outputs."""
    torch.manual_seed(20)
    model = _small_gpt()
    lora_cfg = _default_lora_config()
    inject_lora(model, lora_cfg)
    x = torch.randint(0, 256, (1, 8))

    ckpt_path = tmp_path / "lora_test.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    model2 = _small_gpt()
    inject_lora(model2, lora_cfg)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model2.load_state_dict(ckpt["model_state_dict"])

    with use_mean_weights(model):
        with torch.no_grad():
            y1, _ = model(x)
    with use_mean_weights(model2):
        with torch.no_grad():
            y2, _ = model2(x)
    assert torch.allclose(y1, y2), "Checkpoint roundtrip must preserve model output"
