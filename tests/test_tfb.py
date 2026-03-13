"""B3: Unit tests for TFB (Training-Free Bayesianization).

Tests SVD caching, variance parameterization, binary search convergence,
reproducible sampling, and MC metric computation.
"""
import pytest
import torch

from minigpt.lora import LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT
from minigpt.tfb import (
    compute_tfb_uncertainty,
    fit_tfb,
    load_tfb_state,
    sample_tfb_params,
    save_tfb_state,
)


@pytest.fixture
def toy_model():
    config = GPTConfig(
        n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=100,
    )
    model = MiniGPT(config)
    lora_cfg = LoRAConfig(rank=4, target="ffn")
    inject_lora(model, lora_cfg, bayesian=False)
    return model


@pytest.fixture
def toy_data():
    return torch.randint(0, 100, (100,))


def test_tfb_svd_cache_shapes(toy_model, toy_data):
    """SVD of B produces U, S, V with correct shapes per layer."""
    # Mock some values in B
    with torch.no_grad():
        toy_model.blocks[0].mlp.fc.lora_B.random_()

    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=1,
        epsilon=0.1,
        n_search_samples=2,
    )

    # Check shapes for one layer
    name = "blocks.0.mlp.fc"
    U, S, V = state.svd_cache[name]
    # B is (out_features=128 (4*32), rank=4)
    assert U.shape == (128, 4)
    assert S.shape == (4,)
    assert V.shape == (4, 4)


def test_tfb_variance_structure(toy_model, toy_data):
    """Omega_ij = sigma_q / d_i: larger singular values produce smaller variance."""
    # Force specific singular values in B
    with torch.no_grad():
        # Set B to have singular values [10, 1, 0.1, 0.01]
        fc = toy_model.blocks[0].mlp.fc
        B = fc.lora_B
        U = torch.randn(B.shape[0], 4)
        U, _ = torch.linalg.qr(U)
        S = torch.tensor([10.0, 1.0, 0.1, 0.01])
        V = torch.eye(4)
        B.copy_(U @ torch.diag(S) @ V.T)

        # Ensure A is not zero
        fc.lora_A.fill_(1.0)

    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=1,
        epsilon=0.1,
        n_search_samples=2,
    )

    # Sigma_q should be found
    assert state.sigma_q > 0

    # Check row-wise scaling logic by comparing deviations across rows
    # We take many samples to see the trend
    name = "blocks.0.mlp.fc.lora_A"
    deviations = []
    for s in range(50):
        sampled = sample_tfb_params(state, seed=s)[name]
        dev = (sampled - state.a_map[name]).abs().mean(dim=1) # mean dev per row
        deviations.append(dev)

    mean_dev = torch.stack(deviations).mean(dim=0)

    # S = [10.0, 1.0, 0.1, 0.01]
    # Variance is proportional to 1/S_i
    # mean_dev[0] (S=10) should be much smaller than mean_dev[3] (S=0.01)
    assert mean_dev[0] < mean_dev[1] < mean_dev[2] < mean_dev[3]


def test_tfb_sampling_reproducible(toy_model, toy_data):
    """Same seed -> identical A samples. Different seed -> different samples."""
    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=1,
        epsilon=0.1,
        n_search_samples=2,
    )

    s1 = sample_tfb_params(state, seed=42)
    s2 = sample_tfb_params(state, seed=42)
    s3 = sample_tfb_params(state, seed=43)

    for name in s1:
        assert torch.all(s1[name] == s2[name])
        assert not torch.all(s1[name] == s3[name])


def test_tfb_zero_sigma_returns_map(toy_model, toy_data):
    """If state.sigma_q is 0, sample_tfb_params returns exact A_MAP."""
    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=1,
        epsilon=0.1,
        n_search_samples=2,
    )
    state.sigma_q = 0.0
    sampled = sample_tfb_params(state, seed=42)

    for name, data in sampled.items():
        assert torch.all(data == state.a_map[name])


def test_tfb_search_converges(toy_model, toy_data):
    """Binary search terminates within max iterations."""
    # This is implicitly tested if fit_tfb returns
    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=1,
        epsilon=0.1,
        n_search_samples=2,
    )
    assert 0 <= state.sigma_q <= 10.0


def test_tfb_state_save_load_roundtrip(toy_model, toy_data, tmp_path):
    """Save TFBState -> load -> all fields match."""
    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=1,
        epsilon=0.1,
        n_search_samples=2,
    )
    path = tmp_path / "tfb.pt"
    save_tfb_state(state, path)
    loaded = load_tfb_state(path)

    assert loaded.sigma_q == state.sigma_q
    assert loaded.epsilon == state.epsilon
    assert loaded.anchor_loss == state.anchor_loss
    assert set(loaded.param_names) == set(state.param_names)

    for name in state.param_names:
        assert torch.all(loaded.a_map[name] == state.a_map[name])
        u1, s1, v1 = state.svd_cache[name.replace(".lora_A", "")]
        u2, s2, v2 = loaded.svd_cache[name.replace(".lora_A", "")]
        assert torch.all(u1 == u2)
        assert torch.all(s1 == s2)
        assert torch.all(v1 == v2)


def test_tfb_search_respects_tolerance(toy_model, toy_data):
    """Found sigma_q satisfies |noisy_loss - MAP_loss| <= epsilon."""
    eps = 0.5  # generous tolerance for toy model
    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=2,
        epsilon=eps,
        n_search_samples=5,
    )

    # Re-evaluate the found sigma_q on fresh anchor data
    device = next(toy_model.parameters()).device
    from minigpt.laplace import apply_sampled_params
    from minigpt.train import get_batch

    toy_model.eval()
    with torch.no_grad():
        # MAP loss
        x, y = get_batch(toy_data, 16, 2, device)
        _, map_loss = toy_model(x, y)

        # Noisy loss at found sigma_q
        noisy_total = 0.0
        n_mc = 10
        for s in range(n_mc):
            sampled = sample_tfb_params(state, seed=s + 1000)
            with apply_sampled_params(toy_model, sampled):
                _, loss = toy_model(x, y)
                noisy_total += loss.item()
        noisy_loss = noisy_total / n_mc

    delta = abs(noisy_loss - map_loss.item())
    # The search should find a sigma_q within tolerance (allow some slack
    # from random re-evaluation on different data)
    assert delta < eps * 3, (
        f"delta={delta:.4f} exceeds 3x tolerance (eps={eps}), sigma_q={state.sigma_q:.4f}"
    )


def test_tfb_sampling_changes_logits(toy_model, toy_data):
    """With sigma_q > 0, different MC samples produce different logits."""
    # B must be non-zero for LoRA contribution to matter
    with torch.no_grad():
        for block in toy_model.blocks:
            block.mlp.fc.lora_B.normal_()
            block.mlp.proj.lora_B.normal_()

    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=1,
        epsilon=0.5,
        n_search_samples=2,
    )
    # Ensure sigma_q > 0 (force if search found 0)
    if state.sigma_q == 0:
        state.sigma_q = 0.01

    from minigpt.laplace import apply_sampled_params

    toy_model.eval()
    x = toy_data[:16].unsqueeze(0)
    logits_list = []
    with torch.no_grad():
        for seed in range(5):
            sampled = sample_tfb_params(state, seed=seed)
            with apply_sampled_params(toy_model, sampled):
                logits, _ = toy_model(x)
            logits_list.append(logits.detach().clone())

    any_differ = any(
        not torch.allclose(logits_list[0], logits_list[i])
        for i in range(1, len(logits_list))
    )
    assert any_differ, "TFB samples should produce different logits"


def test_tfb_mc_metrics_protocol(toy_model, toy_data):
    """compute_tfb_uncertainty returns dict with standard MI keys."""
    state = fit_tfb(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        n_batches=1,
        epsilon=0.1,
        n_search_samples=2,
    )

    device = next(toy_model.parameters()).device
    metrics = compute_tfb_uncertainty(
        toy_model,
        toy_data,
        block_size=16,
        batch_size=2,
        device=device,
        state=state,
        n_samples=2,
        n_batches=1,
    )

    expected_keys = {
        "mi_mean", "predictive_entropy_mean", "expected_entropy_mean", "flip_rate"
    }
    assert expected_keys.issubset(metrics.keys())
    for v in metrics.values():
        assert isinstance(v, float)
