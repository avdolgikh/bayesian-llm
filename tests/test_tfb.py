"""B3: Unit tests for TFB (Training-Free Bayesianization).

Tests SVD caching, variance parameterization, binary search convergence,
reproducible sampling, and MC metric computation.

The original tests pin the pre-fix sampler (sampler_version="v1_legacy").
The S2 tests (specs/i2-posthoc-fixes.md) cover the v2 sampler, which applies
the SVD rotation of B, and the relative-tolerance search.
"""
import pytest
import torch

from minigpt.lora import DeterministicLoRALinear, LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT
from minigpt.tfb import (
    TFBState,
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
        sampler_version="v1_legacy",
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
        sampler_version="v1_legacy",
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
        sampler_version="v1_legacy",
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
        sampler_version="v1_legacy",
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
        sampler_version="v1_legacy",
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
        sampler_version="v1_legacy",
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
        sampler_version="v1_legacy",
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
        sampler_version="v1_legacy",
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
        sampler_version="v1_legacy",
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


# ---------------------------------------------------------------------------
# S2-T1: the v2 sampler applies the SVD rotation of B (specs/i2-posthoc-fixes.md)
# ---------------------------------------------------------------------------

T1_SIGMA_Q = 0.1
T1_RANK = 8
T1_IN_FEATURES = 32
T1_TARGET_TRACE = T1_RANK * T1_IN_FEATURES * T1_SIGMA_Q ** 2  # r n sigma_q^2 = 2.56
T1_LAYER = "layer"
T1_PARAM = "layer.lora_A"


def _t1_inputs(case: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (U, B, A_MAP) for S2-T1 case 1 or 2, drawn in the order the spec pins."""
    g = torch.Generator().manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(64, T1_RANK, generator=g))
    v, _ = torch.linalg.qr(torch.randn(T1_RANK, T1_RANK, generator=g))
    a_map = torch.randn(T1_RANK, T1_IN_FEATURES, generator=g)
    if case == 1:
        d = torch.logspace(0, -0.5, T1_RANK)
        b = u @ torch.diag(d) @ v.T
    elif case == 2:
        d = torch.logspace(1, -2, T1_RANK)  # 10 down to 0.01
        r = torch.eye(T1_RANK).flip(0)
        b = u @ torch.diag(d) @ r
    else:
        raise ValueError(case)
    return u, b, a_map


def _t1_state(b: torch.Tensor, a_map: torch.Tensor, sampler_version: str) -> TFBState:
    u, s, vh = torch.linalg.svd(b, full_matrices=False)
    return TFBState(
        sigma_q=T1_SIGMA_Q,
        svd_cache={T1_LAYER: (u, s, vh)},
        a_map={T1_PARAM: a_map},
        param_names=[T1_PARAM],
        epsilon=None,
        anchor_loss=0.0,
        sampler_version=sampler_version,
    )


def _t1_deltas(state: TFBState, n_samples: int) -> torch.Tensor:
    """(S, r, n) stack of A^(s) - A_MAP at seeds 0..S-1."""
    a_map = state.a_map[T1_PARAM]
    return torch.stack([
        sample_tfb_params(state, seed=s)[T1_PARAM] - a_map for s in range(n_samples)
    ])


def _trace_hat(b: torch.Tensor, deltas: torch.Tensor) -> float:
    return (b @ deltas).pow(2).sum(dim=(1, 2)).mean().item()


@pytest.mark.parametrize("case,n_samples", [(1, 10_000), (2, 2_000)])
def test_s2_t1_v2_trace_and_covariance(case, n_samples):
    """S2-T1 (a), (b): v2 trace within 2% of r n sigma_q^2; covariance within 0.05 of I_r."""
    u, b, a_map = _t1_inputs(case)
    state = _t1_state(b, a_map, "v2")
    deltas = _t1_deltas(state, n_samples)

    trace = _trace_hat(b, deltas)
    assert 0.98 * T1_TARGET_TRACE <= trace <= 1.02 * T1_TARGET_TRACE, (
        f"case {case}: trace {trace:.4f} vs target {T1_TARGET_TRACE:.4f} "
        f"({trace / T1_TARGET_TRACE:.4f}x)"
    )

    z = u.T @ b @ deltas / T1_SIGMA_Q  # (S, r, n)
    cols = z.transpose(1, 2).reshape(-1, T1_RANK)  # every column of every sample
    cov = torch.cov(cols.T)
    max_dev = (cov - torch.eye(T1_RANK)).abs().max().item()
    assert max_dev <= 0.05, f"case {case}: covariance max deviation {max_dev:.4f}"


def test_s2_t1_legacy_sampler_overshoots_trace():
    """S2-T1 (c): on case 1, v1_legacy gives > 1.10x the target, empirically and analytically."""
    _, b, a_map = _t1_inputs(1)
    state = _t1_state(b, a_map, "v1_legacy")
    with pytest.warns(UserWarning, match="v1_legacy"):
        deltas = _t1_deltas(state, 10_000)

    trace = _trace_hat(b, deltas)
    assert trace > 1.10 * T1_TARGET_TRACE, f"legacy trace ratio {trace / T1_TARGET_TRACE:.4f}"

    # Section 5.1: E||B (A - A_MAP)||^2 = n sigma_q^2 sum_ij P_ij d_i^2 / d_j^2, P_ij = Vh_ij^2
    _, s, vh = state.svd_cache[T1_LAYER]
    p = vh.pow(2)
    analytic_ratio = (p * (s[:, None] ** 2 / s[None, :] ** 2)).sum().item() / T1_RANK
    assert analytic_ratio > 1.10, f"analytic legacy ratio {analytic_ratio:.4f}"


def test_s2_t1_v2_equals_legacy_when_vh_is_identity():
    """With Vh = I the rotation is a no-op, so v2 and v1_legacy draw the same samples."""
    u, _, a_map = _t1_inputs(1)
    s = torch.logspace(0, -0.5, T1_RANK)
    state_kwargs = dict(
        sigma_q=T1_SIGMA_Q,
        svd_cache={T1_LAYER: (u, s, torch.eye(T1_RANK))},
        a_map={T1_PARAM: a_map},
        param_names=[T1_PARAM],
        epsilon=None,
        anchor_loss=0.0,
    )
    v2 = sample_tfb_params(TFBState(**state_kwargs, sampler_version="v2"), seed=3)
    with pytest.warns(UserWarning):
        v1 = sample_tfb_params(TFBState(**state_kwargs, sampler_version="v1_legacy"), seed=3)
    assert torch.allclose(v2[T1_PARAM], v1[T1_PARAM], atol=1e-6)


def test_s2_v2_zero_sigma_returns_map():
    """v2 with sigma_q = 0 returns an exact copy of A_MAP."""
    _, b, a_map = _t1_inputs(1)
    state = _t1_state(b, a_map, "v2")
    state.sigma_q = 0.0
    sampled = sample_tfb_params(state, seed=0)[T1_PARAM]
    assert torch.equal(sampled, a_map)
    assert sampled is not a_map


def test_s2_unknown_sampler_version_rejected_when_sampling():
    """sample_tfb_params raises ValueError on a state whose version was changed to an unknown."""
    _, b, a_map = _t1_inputs(1)
    state = _t1_state(b, a_map, "v2")
    state.sampler_version = "v3"
    with pytest.raises(ValueError, match="sampler_version"):
        sample_tfb_params(state, seed=0)


# ---------------------------------------------------------------------------
# S2 section 5.3: the relative-tolerance search in fit_tfb (items 3, 5, 6, 7)
# ---------------------------------------------------------------------------


def _set_rotated_b(model: MiniGPT, seed: int = 1) -> None:
    """Set every lora_B to U diag(d) V^T with random orthogonal U, V (so Vh != I)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DeterministicLoRALinear):
                out_f, rank = module.lora_B.shape
                u, _ = torch.linalg.qr(torch.randn(out_f, rank, generator=g))
                v, _ = torch.linalg.qr(torch.randn(rank, rank, generator=g))
                d = torch.logspace(0, -0.5, rank)
                module.lora_B.copy_(u @ torch.diag(d) @ v.T)


SEARCH_CONFIG = dict(n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=100)
PATTERN_STREAM = torch.arange(2000) % 50  # period-50 token pattern


def _pattern_batch(starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([PATTERN_STREAM[i:i + 16] for i in starts.tolist()])
    y = torch.stack([PATTERN_STREAM[i + 1:i + 17] for i in starts.tolist()])
    return x, y


@pytest.fixture(scope="module")
def trained_base_state():
    """An untrained model is insensitive to adapter noise (near-uniform logits), so the
    search tests use a base briefly trained on a period-50 pattern (as in S2-T2)."""
    torch.manual_seed(0)
    model = MiniGPT(GPTConfig(**SEARCH_CONFIG))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        x, y = _pattern_batch(torch.randint(0, len(PATTERN_STREAM) - 17, (16,), generator=g))
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@pytest.fixture
def rotated_model(trained_base_state):
    model = MiniGPT(GPTConfig(**SEARCH_CONFIG))
    model.load_state_dict(trained_base_state)
    inject_lora(model, LoRAConfig(rank=4, alpha=8.0, target="ffn"), bayesian=False)
    _set_rotated_b(model)
    return model


@pytest.fixture
def fixed_batches():
    g = torch.Generator().manual_seed(2)
    starts = torch.randint(0, len(PATTERN_STREAM) - 17, (4, 2), generator=g)
    return [_pattern_batch(row) for row in starts]


def _fit_rel(model, batches, **overrides):
    kwargs = dict(
        block_size=16,
        batch_size=2,
        n_batches=len(batches),
        epsilon_rel=0.01,
        n_search_samples=3,
        search_range=(1e-4, 1.0),
        search_precision=1e-3,
        sampler_version="v2",
        anchor_batches=batches,
    )
    kwargs.update(overrides)
    return fit_tfb(model, None, **kwargs)


def test_s2_fit_tfb_rel_search_log(rotated_model, fixed_batches):
    """Every step is logged with its sampler version; sigma_q* is the largest accepted step."""
    state = _fit_rel(rotated_model, fixed_batches)

    assert state.sampler_version == "v2"
    assert state.epsilon is None
    assert state.epsilon_rel == 0.01
    log = state.search_log
    assert log, "search_log is empty"
    required = {"sigma_q", "avg_loss", "anchor_loss", "delta", "accepted", "sampler_version"}
    tol = 0.01 * state.anchor_loss
    for step in log:
        assert required <= set(step)
        assert step["sampler_version"] == "v2"
        assert step["anchor_loss"] == state.anchor_loss
        assert step["delta"] == pytest.approx(step["avg_loss"] - step["anchor_loss"])
        assert step["accepted"] == (abs(step["avg_loss"] - step["anchor_loss"]) <= tol)

    # The pre-check at search_max comes first and is rejected
    assert log[0]["stage"] == "precheck"
    assert log[0]["sigma_q"] == 1.0
    assert log[0]["accepted"] is False

    accepted = [s["sigma_q"] for s in log if s["accepted"]]
    rejected = [s["sigma_q"] for s in log if not s["accepted"]]
    assert accepted, "no accepted step"
    assert state.sigma_q == max(accepted)
    assert all(r > state.sigma_q for r in rejected)
    # 1 pre-check + ceil(log2((1 - 1e-4) / 1e-3)) = 10 bisection steps
    assert len(log) == 11


def test_s2_fit_tfb_anchor_batches_define_anchor_loss(rotated_model, fixed_batches):
    """With anchor_batches, fit_tfb draws nothing and ell_0 is the mean loss on those batches."""
    rotated_model.eval()
    with torch.no_grad():
        expected = sum(rotated_model(x, y)[1].item() for x, y in fixed_batches)
    expected /= len(fixed_batches)

    state = _fit_rel(rotated_model, fixed_batches)
    assert state.anchor_loss == pytest.approx(expected, rel=1e-6)


def test_s2_fit_tfb_anchor_batches_must_share_shape(rotated_model, fixed_batches):
    x, y = fixed_batches[0]
    ragged = fixed_batches + [(x[:1], y[:1])]
    with pytest.raises(ValueError, match="same shape"):
        _fit_rel(rotated_model, ragged)


def test_s2_fit_tfb_requires_data_or_anchor_batches(rotated_model):
    with pytest.raises(ValueError, match="anchor_batches"):
        _fit_rel(rotated_model, [], anchor_batches=None)


def test_s2_fit_tfb_search_range_too_small(rotated_model, fixed_batches):
    """epsilon_rel path: if search_max already passes, the range is too small."""
    with pytest.raises(ValueError, match="search range too small"):
        _fit_rel(rotated_model, fixed_batches, search_range=(1e-6, 1e-5))


def test_s2_fit_tfb_no_sigma_accepted(rotated_model, fixed_batches):
    """epsilon_rel path: if no step passes, fit_tfb raises instead of returning search_min."""
    with pytest.raises(ValueError, match="no sigma_q accepted"):
        _fit_rel(rotated_model, fixed_batches, epsilon_rel=0.0, search_precision=0.1)


def test_s2_fit_tfb_restores_train_mode_on_error(rotated_model, fixed_batches):
    rotated_model.train()
    with pytest.raises(ValueError):
        _fit_rel(rotated_model, fixed_batches, epsilon_rel=0.0, search_precision=0.1)
    assert rotated_model.training


def test_s2_fit_tfb_legacy_epsilon_keeps_old_behaviour(rotated_model, fixed_batches):
    """The absolute-epsilon path has no pre-check and returns search_min when nothing passes."""
    with pytest.warns(UserWarning):
        state = _fit_rel(
            rotated_model, fixed_batches,
            epsilon_rel=None, epsilon=0.0, search_precision=0.1,
            sampler_version="v1_legacy",
        )
    assert state.sigma_q == 1e-4
    assert state.search_log
    assert all(s["stage"] == "bisect" for s in state.search_log)  # no pre-check
    assert all(not s["accepted"] for s in state.search_log)


def test_s2_fit_tfb_search_runs_chosen_sampler(rotated_model, fixed_batches):
    """With Vh != I, the same search step gives a different loss under v2 and v1_legacy."""
    common = dict(epsilon_rel=None, epsilon=1e9, search_precision=0.4)
    v2 = _fit_rel(rotated_model, fixed_batches, **common)
    with pytest.warns(UserWarning):
        v1 = _fit_rel(rotated_model, fixed_batches, **common, sampler_version="v1_legacy")

    assert [s["sigma_q"] for s in v2.search_log] == [s["sigma_q"] for s in v1.search_log]
    assert all(s["sampler_version"] == "v2" for s in v2.search_log)
    assert all(s["sampler_version"] == "v1_legacy" for s in v1.search_log)
    for s2, s1 in zip(v2.search_log, v1.search_log):
        assert s2["avg_loss"] != pytest.approx(s1["avg_loss"], rel=1e-6)
