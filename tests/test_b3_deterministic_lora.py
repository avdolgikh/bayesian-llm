"""B3: Unit tests for DeterministicLoRALinear and its injection.

Ensures the deterministic LoRA baseline behaves as a standard (non-Bayesian)
adapter, matches base model output at initialization, and integrates with
inject_lora() correctly.
"""
import torch
import torch.nn as nn

from minigpt.lora import DeterministicLoRALinear, LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT


def test_deterministic_lora_forward_matches_base_at_init():
    """At initialization (B=0), DeterministicLoRALinear output equals base_linear output."""
    in_features, out_features, rank = 10, 20, 4
    base = nn.Linear(in_features, out_features, bias=False)
    lora = DeterministicLoRALinear(base, rank=rank, alpha=8.0)

    # Ensure B is initialized to zero
    torch.nn.init.zeros_(lora.lora_B)

    x = torch.randn(2, 5, in_features)
    with torch.no_grad():
        base_out = base(x)
        lora_out = lora(x)

    assert torch.allclose(base_out, lora_out, atol=1e-6)


def test_deterministic_lora_no_kl():
    """DeterministicLoRALinear has no kl_loss method (it's not a BayesianModule)."""
    base = nn.Linear(10, 20)
    lora = DeterministicLoRALinear(base, rank=4, alpha=8.0)
    assert not hasattr(lora, "kl_loss")


def test_deterministic_lora_all_params_trainable():
    """lora_A and lora_B require grad; base_linear params don't."""
    base = nn.Linear(10, 20)
    lora = DeterministicLoRALinear(base, rank=4, alpha=8.0)

    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True
    assert lora.base_linear.weight.requires_grad is False


def test_deterministic_lora_forward_deterministic():
    """Same input always produces same output (no sampling)."""
    base = nn.Linear(10, 20)
    lora = DeterministicLoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(2, 5, 10)

    out1 = lora(x)
    out2 = lora(x)
    assert torch.all(out1 == out2)


def test_inject_lora_bayesian_false():
    """inject_lora(model, cfg, bayesian=False) replaces FFN layers with DeterministicLoRALinear."""
    config = GPTConfig(
        n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=100,
    )
    model = MiniGPT(config)
    lora_cfg = LoRAConfig(rank=4, target="ffn")

    inject_lora(model, lora_cfg, bayesian=False)

    assert isinstance(model.blocks[0].mlp.fc, DeterministicLoRALinear)
    assert isinstance(model.blocks[0].mlp.proj, DeterministicLoRALinear)
    assert model.blocks[0].mlp.fc.rank == 4


def test_inject_lora_bayesian_false_freezes_base():
    """After injection, only LoRA params have requires_grad=True."""
    config = GPTConfig(
        n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=100,
    )
    model = MiniGPT(config)
    lora_cfg = LoRAConfig(rank=4, target="ffn")

    inject_lora(model, lora_cfg, bayesian=False)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 4  # fc.A, fc.B, proj.A, proj.B
    for name in trainable:
        assert "lora_" in name

    # Check some frozen ones
    assert model.token_emb.weight.requires_grad is False
    assert model.blocks[0].attn.q_proj.linear.weight.requires_grad is False


def test_deterministic_lora_param_count():
    """Correct number of LoRA params for known dimensions."""
    in_features, out_features, rank = 32, 64, 8
    base = nn.Linear(in_features, out_features)
    lora = DeterministicLoRALinear(base, rank=rank, alpha=16.0)

    # A is (rank, in), B is (out, rank)
    expected = (rank * in_features) + (out_features * rank)
    actual = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    assert actual == expected
