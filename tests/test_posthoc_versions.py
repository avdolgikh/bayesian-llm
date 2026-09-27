"""S2-T5: post-hoc sampler versioning (specs/i2-posthoc-fixes.md, Section 4).

Layout:
  - Shared helpers (used by every section)
  - TFB section (S2 lane: tfb)
  - Laplace section (S2 lane: laplace), appended below the TFB section

The saved-file checks skip when data/ is absent, and fail instead when
REQUIRE_POSTHOC_DATA=1 (the pre-merge command in the spec).
"""
import os
import warnings
from pathlib import Path

import pytest
import torch

from minigpt.lora import DeterministicLoRALinear, LoRAConfig, inject_lora
from minigpt.model import GPTConfig, MiniGPT
from minigpt.tfb import TFBState, fit_tfb, load_tfb_state, sample_tfb_params, save_tfb_state

# ===========================================================================
# Shared helpers
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "data" / "checkpoints"
REFERENCE_SEEDS = range(5)  # S2-T5: seeds 0-4


def require_posthoc_file(path: Path) -> Path:
    """Return path if it exists; otherwise skip, or fail when REQUIRE_POSTHOC_DATA=1."""
    if path.exists():
        return path
    if os.environ.get("REQUIRE_POSTHOC_DATA") == "1":
        pytest.fail(f"REQUIRE_POSTHOC_DATA=1 but {path} is missing")
    pytest.skip(f"{path} not present (set REQUIRE_POSTHOC_DATA=1 to fail instead)")


# ===========================================================================
# TFB section (S2 lane: tfb)
# ===========================================================================


# Reference: verbatim copy of sample_tfb_params (minigpt/tfb.py:160-194) at commit
# a7a73b6, renamed. Do not edit: T5(a) checks the v1_legacy sampler against it bit for bit.
def _reference_sample_tfb_params_a7a73b6(
    state: TFBState,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample A from TFB posterior: A_hat = A_MAP + Omega * eps.

    Omega_ij = sigma_q / S_i
    where S_i are singular values of B.
    """
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    sampled = {}
    for a_name in state.param_names:
        layer_name = a_name.replace(".lora_A", "")
        U, S, V = state.svd_cache[layer_name]
        a_map = state.a_map[a_name]

        if state.sigma_q == 0.0:
            sampled[a_name] = a_map.clone()
            continue

        # Omega structure: sigma_q / S_i applied to each row i of A
        # S has shape (rank,), a_map has shape (rank, in_features)
        # S can have tiny values; clamp for stability
        S_clamped = S.clamp(min=1e-6)
        std = (state.sigma_q / S_clamped).unsqueeze(1)  # (rank, 1)

        eps = torch.randn(a_map.shape, generator=gen, dtype=a_map.dtype)
        sampled[a_name] = a_map + std * eps.to(a_map.device)

    return sampled


def _synthetic_tfb_payload(seed: int = 0) -> dict:
    """A two-layer TFB payload in the pre-fix file format (no new fields), with Vh != I."""
    g = torch.Generator().manual_seed(seed)
    svd_cache = {}
    a_map = {}
    param_names = []
    for layer, (out_f, rank, in_f) in {
        "blocks.0.mlp.fc": (32, 4, 8),
        "blocks.0.mlp.proj": (8, 4, 32),
    }.items():
        u, _ = torch.linalg.qr(torch.randn(out_f, rank, generator=g))
        v, _ = torch.linalg.qr(torch.randn(rank, rank, generator=g))
        b = u @ torch.diag(torch.logspace(0, -1, rank)) @ v.T
        svd_cache[layer] = tuple(torch.linalg.svd(b, full_matrices=False))
        a_map[f"{layer}.lora_A"] = torch.randn(rank, in_f, generator=g)
        param_names.append(f"{layer}.lora_A")
    return {
        "sigma_q": 0.03,
        "svd_cache": svd_cache,
        "a_map": a_map,
        "param_names": param_names,
        "epsilon": 0.1,
        "anchor_loss": 4.17,
    }


def _v2_state() -> TFBState:
    payload = _synthetic_tfb_payload(seed=1)
    payload["epsilon"] = None
    return TFBState(
        **payload,
        epsilon_rel=0.003,
        search_log=[
            {"stage": "precheck", "sigma_q": 1.0, "avg_loss": 9.0, "anchor_loss": 4.17,
             "delta": 4.83, "tolerance": 0.01251, "accepted": False, "sampler_version": "v2"},
            {"stage": "bisect", "sigma_q": 0.50005, "avg_loss": 4.171, "anchor_loss": 4.17,
             "delta": 0.001, "tolerance": 0.01251, "accepted": True, "sampler_version": "v2"},
        ],
        sampler_version="v2",
    )


def _assert_samples_equal_reference(state: TFBState) -> None:
    for seed in REFERENCE_SEEDS:
        with pytest.warns(UserWarning, match="v1_legacy"):
            got = sample_tfb_params(state, seed=seed)
        want = _reference_sample_tfb_params_a7a73b6(state, seed=seed)
        assert set(got) == set(want)
        for name in want:
            assert torch.equal(got[name], want[name]), f"seed {seed}, {name}"


def _toy_lora_model() -> MiniGPT:
    torch.manual_seed(0)
    model = MiniGPT(GPTConfig(n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=100))
    inject_lora(model, LoRAConfig(rank=4, alpha=8.0, target="ffn"), bayesian=False)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DeterministicLoRALinear):
                module.lora_B.copy_(torch.randn(module.lora_B.shape, generator=g))
    return model


def _trained_toy_lora_model() -> tuple[MiniGPT, list[tuple[torch.Tensor, torch.Tensor]]]:
    """A briefly trained base (period-50 pattern) with random B, plus 4 fixed batches.

    An untrained base is insensitive to adapter noise, so a v2 search would not bracket.
    """
    stream = torch.arange(2000) % 50

    def batch(starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.stack([stream[i:i + 16] for i in starts.tolist()]),
                torch.stack([stream[i + 1:i + 17] for i in starts.tolist()]))

    torch.manual_seed(0)
    model = MiniGPT(GPTConfig(n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=100))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        x, y = batch(torch.randint(0, len(stream) - 17, (16,), generator=g))
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    inject_lora(model, LoRAConfig(rank=4, alpha=8.0, target="ffn"), bayesian=False)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DeterministicLoRALinear):
                module.lora_B.copy_(torch.randn(module.lora_B.shape, generator=g))
    starts = torch.randint(0, len(stream) - 17, (4, 2), generator=g)
    return model, [batch(row) for row in starts]


# --- T5(a): pre-fix files load as v1_legacy and sample bit for bit as before ---


def test_tfb_prefix_saved_state_loads_legacy_and_matches_reference():
    """S2-T5 (a), saved file: data/checkpoints/c4_tfb/tfb_state.pt."""
    path = require_posthoc_file(CKPT_DIR / "c4_tfb" / "tfb_state.pt")
    raw = torch.load(path, weights_only=False, map_location="cpu")
    assert "sampler_version" not in raw  # it is a pre-fix file

    state = load_tfb_state(path, map_location="cpu")
    assert state.sampler_version == "v1_legacy"
    _assert_samples_equal_reference(state)


def test_tfb_synthetic_legacy_file_loads_legacy_and_matches_reference(tmp_path):
    """S2-T5 (a), synthetic: a file in the pre-fix format loads as v1_legacy."""
    path = tmp_path / "tfb_state.pt"
    torch.save(_synthetic_tfb_payload(), path)

    state = load_tfb_state(path)
    assert state.sampler_version == "v1_legacy"
    assert state.epsilon == 0.1
    assert state.epsilon_rel is None
    assert state.search_log == []
    _assert_samples_equal_reference(state)


def test_tfb_version_flag_switches_sampler():
    """The same state sampled as v2 differs from v1_legacy when Vh != I."""
    payload = _synthetic_tfb_payload()
    legacy = TFBState(**payload, sampler_version="v1_legacy")
    v2 = TFBState(**payload, sampler_version="v2")
    with pytest.warns(UserWarning):
        s_legacy = sample_tfb_params(legacy, seed=0)
    s_v2 = sample_tfb_params(v2, seed=0)
    for name in payload["param_names"]:
        assert not torch.allclose(s_legacy[name], s_v2[name])


# --- T5(b): v2 states keep every new field through save and load ---


def test_tfb_v2_state_roundtrip_keeps_new_fields(tmp_path):
    state = _v2_state()
    path = tmp_path / "tfb_v2.pt"
    save_tfb_state(state, path)
    loaded = load_tfb_state(path)

    assert loaded.sampler_version == "v2"
    assert loaded.epsilon is None
    assert loaded.epsilon_rel == state.epsilon_rel
    assert loaded.search_log == state.search_log
    assert loaded.sigma_q == state.sigma_q
    assert loaded.anchor_loss == state.anchor_loss
    assert loaded.param_names == state.param_names
    for name in state.param_names:
        assert torch.equal(loaded.a_map[name], state.a_map[name])
    assert set(loaded.svd_cache) == set(state.svd_cache)
    for layer, tensors in state.svd_cache.items():
        for got, want in zip(loaded.svd_cache[layer], tensors):
            assert torch.equal(got, want)
    for seed in REFERENCE_SEEDS:
        a, b = sample_tfb_params(state, seed=seed), sample_tfb_params(loaded, seed=seed)
        for name in a:
            assert torch.equal(a[name], b[name])


def test_tfb_fit_v2_state_reads_back_v2(tmp_path):
    """A state written from a v2 fit_tfb run reads back with sampler_version == "v2"."""
    model, batches = _trained_toy_lora_model()
    state = fit_tfb(
        model, None,
        block_size=16, batch_size=2, n_batches=len(batches),
        epsilon_rel=0.01, n_search_samples=2,
        search_range=(1e-4, 1.0), search_precision=1e-2,
        sampler_version="v2",
        anchor_batches=batches,
    )
    assert state.sampler_version == "v2"
    path = tmp_path / "tfb_state.pt"
    save_tfb_state(state, path)
    loaded = load_tfb_state(path)
    assert loaded.sampler_version == "v2"
    assert loaded.search_log == state.search_log
    assert all(step["sampler_version"] == "v2" for step in loaded.search_log)


# --- T5(c): sampling a v1_legacy TFB state warns; v2 does not ---


def test_tfb_legacy_sampling_warns():
    state = TFBState(**_synthetic_tfb_payload(), sampler_version="v1_legacy")
    with pytest.warns(UserWarning, match="v1_legacy"):
        sample_tfb_params(state, seed=0)


def test_tfb_v2_sampling_does_not_warn():
    state = _v2_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sample_tfb_params(state, seed=0)


# --- T5(e): exactly one of epsilon and epsilon_rel ---


@pytest.mark.parametrize("tolerances", [
    {"epsilon": 0.1, "epsilon_rel": 0.003},
    {},
    {"epsilon": None, "epsilon_rel": None},
])
def test_fit_tfb_requires_exactly_one_tolerance(tolerances):
    model = _toy_lora_model()
    with pytest.raises(ValueError, match="epsilon"):
        fit_tfb(
            model, torch.randint(0, 100, (100,)),
            block_size=16, batch_size=2, n_batches=1, n_search_samples=1,
            sampler_version="v2", **tolerances,
        )


# --- T5(g): sampler_version is required, with no default ---


def test_tfb_state_without_sampler_version_raises_type_error():
    with pytest.raises(TypeError, match="sampler_version"):
        TFBState(**_synthetic_tfb_payload())


def test_fit_tfb_without_sampler_version_raises_type_error():
    model = _toy_lora_model()
    with pytest.raises(TypeError, match="sampler_version"):
        fit_tfb(
            model, torch.randint(0, 100, (100,)),
            block_size=16, batch_size=2, n_batches=1, epsilon_rel=0.003, n_search_samples=1,
        )


# --- Unknown versions are rejected (Section 5.3 items 2 and 4) ---


def test_tfb_state_unknown_sampler_version_raises():
    with pytest.raises(ValueError, match="sampler_version"):
        TFBState(**_synthetic_tfb_payload(), sampler_version="v3")


def test_tfb_load_unknown_sampler_version_raises(tmp_path):
    path = tmp_path / "tfb_state.pt"
    torch.save({**_synthetic_tfb_payload(), "sampler_version": "v3"}, path)
    with pytest.raises(ValueError, match="sampler_version"):
        load_tfb_state(path)


def test_fit_tfb_unknown_sampler_version_raises():
    model = _toy_lora_model()
    with pytest.raises(ValueError, match="sampler_version"):
        fit_tfb(
            model, torch.randint(0, 100, (100,)),
            block_size=16, batch_size=2, n_batches=1, epsilon_rel=0.003, n_search_samples=1,
            sampler_version="v3",
        )


# ===========================================================================
# Laplace section (S2 lane: laplace) -- appended below this line
# ===========================================================================
# S2-T5 Laplace rows: (a) legacy bitwise, (b) v2 round trip, (c) legacy warning,
# (d) v2 states read back as v2, (f) v2 sample_scale guard, (g) required sampler_version.
# Imports stay local to each function so this section adds nothing to the module header.


# Reference: verbatim copy of the body of sample_laplace_params (minigpt/laplace.py:140-173)
# at commit a7a73b6, renamed. Do not edit: T5(a) checks the v1_legacy sampler against it bit
# for bit. Only the ``state`` annotation is dropped, to keep the module header unchanged.
def _reference_sample_laplace_params_a7a73b6(
    state,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample parameters from the Laplace posterior.

    phi ~ N(phi_hat, diag(sample_scale^2 / (curvature + damping)))

    Args:
        state: fitted LaplaceState.
        seed: random seed for reproducibility.

    Returns:
        dict mapping param name -> sampled tensor.
    """
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    sampled = {}
    for name in state.param_names:
        phi = state.phi_hat[name]
        if state.sample_scale == 0.0:
            sampled[name] = phi.clone()
        else:
            variance = 1.0 / (state.curvature[name] + state.damping)
            std = variance.sqrt() * state.sample_scale
            # Generate on CPU (generator is CPU-only) then move to device
            eps = torch.randn(phi.shape, generator=gen, dtype=phi.dtype)
            sampled[name] = phi + std * eps.to(phi.device)

    return sampled


def _lap_legacy_payload(sample_scale: float = 1.0) -> dict:
    """A state dict in the pre-fix file format: exactly the five legacy keys."""
    g = torch.Generator().manual_seed(0)
    names = ["blocks.0.mlp.fc.lora_A", "blocks.0.mlp.proj.lora_A", "blocks.1.mlp.fc.lora_A"]
    shapes = [(4, 16), (4, 32), (4, 16)]
    return {
        "param_names": names,
        "phi_hat": {n: torch.randn(s, generator=g) * 0.02 for n, s in zip(names, shapes)},
        "curvature": {n: torch.rand(s, generator=g) * 1e-6 for n, s in zip(names, shapes)},
        "damping": 1.0,
        "sample_scale": sample_scale,
    }


def _lap_v1_state(sample_scale: float = 1.0):
    from minigpt.laplace import LaplaceState

    data = _lap_legacy_payload(sample_scale)
    return LaplaceState(
        param_names=data["param_names"], phi_hat=data["phi_hat"],
        curvature=data["curvature"], damping=data["damping"],
        sample_scale=data["sample_scale"], sampler_version="v1_legacy",
    )


def _lap_v2_state():
    from minigpt.laplace import scale_laplace_state

    return scale_laplace_state(
        _lap_v1_state(), n_data_seqs=312_500, tokens_per_seq=256, prior_prec=1.0e3,
        base_checkpoint="data/checkpoints/c3/ckpt_best.pt", base_sha256="0f" * 32,
    )


def _lap_assert_samples_equal(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name], b[name]), f"sample mismatch for {name}"


def _lap_assert_legacy_matches_reference(state) -> None:
    from minigpt.laplace import sample_laplace_params

    assert state.sampler_version == "v1_legacy"
    for seed in REFERENCE_SEEDS:
        with pytest.warns(UserWarning):
            new = sample_laplace_params(state, seed=seed)
        _lap_assert_samples_equal(new, _reference_sample_laplace_params_a7a73b6(state, seed))


# --- T5(a): pre-fix Laplace files load as v1_legacy and sample bit for bit as before ---


@pytest.mark.parametrize("cell", ["c4_lap", "c2"])
def test_laplace_prefix_saved_state_loads_legacy_and_matches_reference(cell):
    from minigpt.laplace import load_laplace_state

    path = require_posthoc_file(CKPT_DIR / cell / "laplace_state.pt")
    state = load_laplace_state(path, map_location="cpu")
    assert state.n_data_seqs is None and state.prior_prec is None
    _lap_assert_legacy_matches_reference(state)


@pytest.mark.parametrize("sample_scale", [1.0, 0.5, 0.0])
def test_laplace_synthetic_legacy_file_loads_legacy_and_matches_reference(
    tmp_path, sample_scale,
):
    from minigpt.laplace import load_laplace_state

    path = tmp_path / "laplace_state.pt"
    torch.save(_lap_legacy_payload(sample_scale), path)
    state = load_laplace_state(path)
    assert state.n_data_seqs is None
    assert state.tokens_per_seq is None
    assert state.prior_prec is None
    assert state.damping == 1.0
    _lap_assert_legacy_matches_reference(state)


# --- T5(b), T5(d): v2 Laplace states keep every new field and read back as v2 ---


def test_laplace_v2_state_roundtrip_keeps_new_fields(tmp_path):
    from minigpt.laplace import load_laplace_state, save_laplace_state

    state = _lap_v2_state()
    path = tmp_path / "laplace_state_v2.pt"
    save_laplace_state(state, path)
    loaded = load_laplace_state(path)

    assert loaded.sampler_version == "v2"
    for field_name in (
        "n_data_seqs", "tokens_per_seq", "prior_prec", "damping", "sample_scale",
        "base_checkpoint", "base_sha256", "param_names",
    ):
        assert getattr(loaded, field_name) == getattr(state, field_name), field_name
    for name in state.param_names:
        assert torch.equal(loaded.phi_hat[name], state.phi_hat[name])
        assert torch.equal(loaded.curvature[name], state.curvature[name])


def test_laplace_loaded_v2_state_samples_equal_original(tmp_path):
    from minigpt.laplace import load_laplace_state, sample_laplace_params, save_laplace_state

    state = _lap_v2_state()
    path = tmp_path / "laplace_state_v2.pt"
    save_laplace_state(state, path)
    loaded = load_laplace_state(path)
    for seed in REFERENCE_SEEDS:
        _lap_assert_samples_equal(
            sample_laplace_params(loaded, seed=seed), sample_laplace_params(state, seed=seed),
        )


def test_laplace_saved_v1_state_reads_back_legacy(tmp_path):
    from minigpt.laplace import load_laplace_state, save_laplace_state

    path = tmp_path / "laplace_state_v1.pt"
    save_laplace_state(_lap_v1_state(), path)
    loaded = load_laplace_state(path)
    assert loaded.sampler_version == "v1_legacy"
    assert loaded.prior_prec is None


# --- T5(c): sampling a v1_legacy Laplace state warns; v2 does not ---


@pytest.mark.parametrize("sample_scale", [1.0, 0.0])
def test_laplace_legacy_sampling_warns(sample_scale):
    from minigpt.laplace import sample_laplace_params

    with pytest.warns(UserWarning, match="v1_legacy"):
        sample_laplace_params(_lap_v1_state(sample_scale), seed=0)


def test_laplace_v2_sampling_does_not_warn():
    from minigpt.laplace import sample_laplace_params

    state = _lap_v2_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample_laplace_params(state, seed=0)


# --- T5(f): a v2 Laplace state with sample_scale != 1.0 cannot be sampled ---


@pytest.mark.parametrize("sample_scale", [0.0, 0.5, 2.0])
def test_laplace_v2_sample_scale_not_one_raises(sample_scale):
    from minigpt.laplace import sample_laplace_params

    state = _lap_v2_state()
    state.sample_scale = sample_scale
    with pytest.raises(ValueError, match="sample_scale"):
        sample_laplace_params(state, seed=0)


# --- T5(g): sampler_version is required, with no default ---


def test_laplace_state_without_sampler_version_raises_type_error():
    from minigpt.laplace import LaplaceState

    with pytest.raises(TypeError):
        LaplaceState(param_names=[], phi_hat={}, curvature={}, damping=1.0)


def test_laplace_state_sampler_version_is_keyword_only():
    from minigpt.laplace import LaplaceState

    with pytest.raises(TypeError):
        LaplaceState([], {}, {}, 1.0, 1.0, "v1_legacy")


# --- Unknown versions are rejected (Section 5.4 item 1) ---


def test_laplace_state_unknown_sampler_version_raises():
    from minigpt.laplace import LaplaceState

    with pytest.raises(ValueError, match="sampler_version"):
        LaplaceState(param_names=[], phi_hat={}, curvature={}, damping=1.0, sampler_version="v3")


def test_laplace_load_unknown_sampler_version_raises(tmp_path):
    from minigpt.laplace import load_laplace_state

    data = _lap_legacy_payload()
    data["sampler_version"] = "v9"
    path = tmp_path / "laplace_state.pt"
    torch.save(data, path)
    with pytest.raises(ValueError, match="sampler_version"):
        load_laplace_state(path)


# ===========================================================================
# End of Laplace section
# ===========================================================================
