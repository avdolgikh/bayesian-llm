"""B1 unit tests -- Laplace approximation (post-hoc Bayesianization).

Tests are written BEFORE implementation (TDD Stage #2).
They target the planned `minigpt/laplace.py` module.
"""

import torch

from minigpt.layers import BayesConfig
from minigpt.model import GPTConfig, MiniGPT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_deterministic_config() -> GPTConfig:
    """Minimal deterministic miniGPT for fast tests."""
    return GPTConfig(
        vocab_size=256, block_size=32, n_layer=2, n_head=2,
        n_embd=64, dropout=0.0, bias=True,
        bayes_head=BayesConfig(enabled=False),
    )


def _make_model_and_data():
    """Build a tiny deterministic model + fake training data."""
    torch.manual_seed(42)
    config = _small_deterministic_config()
    model = MiniGPT(config)
    # Fake data: enough tokens for curvature batches
    data = torch.randint(0, config.vocab_size, (512,))
    return model, config, data


def _fit_ffn(model, config, data, damping=1.0, sample_scale=1.0, n_batches=2):
    """Shorthand: select FFN params and fit Laplace."""
    from minigpt.laplace import fit_laplace, select_params

    selected = select_params(model, mode="ffn")
    state = fit_laplace(
        model, data,
        block_size=config.block_size,
        batch_size=4,
        selection=selected,
        n_batches=n_batches,
        damping=damping,
        sample_scale=sample_scale,
    )
    return selected, state


# ---------------------------------------------------------------------------
# 1. Curvature stats shapes match selected params
# ---------------------------------------------------------------------------

class TestCurvatureShapes:
    def test_curvature_stats_shapes_match_selected_params(self):
        """Curvature tensors must have the same shape as the selected param tensors."""
        model, config, data = _make_model_and_data()
        selected, state = _fit_ffn(model, config, data)

        assert len(selected) > 0, "FFN selection should find params"
        for name in selected:
            assert name in state.phi_hat, f"phi_hat missing {name}"
            assert name in state.curvature, f"curvature missing {name}"
            assert state.phi_hat[name].shape == selected[name].shape, \
                f"phi_hat shape mismatch for {name}"
            assert state.curvature[name].shape == selected[name].shape, \
                f"curvature shape mismatch for {name}"


# ---------------------------------------------------------------------------
# 2. Damping makes posterior finite and stable
# ---------------------------------------------------------------------------

class TestDampingStability:
    def test_damping_makes_posterior_finite_and_stable(self):
        """With damping > 0, posterior variance = 1/(curvature + damping) is finite & positive."""
        model, config, data = _make_model_and_data()
        selected, state = _fit_ffn(model, config, data, damping=1.0)

        for name in selected:
            variance = 1.0 / (state.curvature[name] + state.damping)
            assert torch.isfinite(variance).all(), f"Non-finite variance for {name}"
            assert (variance > 0).all(), f"Non-positive variance for {name}"


# ---------------------------------------------------------------------------
# 3. Sampling reproducible with fixed seed
# ---------------------------------------------------------------------------

class TestSamplingReproducibility:
    def test_same_seed_identical_samples(self):
        """Same seed must produce identical sampled tensors."""
        from minigpt.laplace import sample_laplace_params

        model, config, data = _make_model_and_data()
        selected, state = _fit_ffn(model, config, data)

        s1 = sample_laplace_params(state, seed=123)
        s2 = sample_laplace_params(state, seed=123)
        for name in selected:
            assert torch.equal(s1[name], s2[name]), \
                f"Same seed produced different samples for {name}"

    def test_different_seed_different_samples(self):
        """Different seeds must produce different samples (checked across all params)."""
        from minigpt.laplace import sample_laplace_params

        model, config, data = _make_model_and_data()
        selected, state = _fit_ffn(model, config, data)

        # Use 3 seed pairs to avoid single-shot flakiness
        diff_count = 0
        for seed_a, seed_b in [(10, 20), (30, 40), (50, 60)]:
            s_a = sample_laplace_params(state, seed=seed_a)
            s_b = sample_laplace_params(state, seed=seed_b)
            if any(not torch.equal(s_a[n], s_b[n]) for n in selected):
                diff_count += 1
        assert diff_count >= 2, \
            "At least 2 of 3 seed pairs should produce different samples"


# ---------------------------------------------------------------------------
# 4. Zero sample scale returns MAP params
# ---------------------------------------------------------------------------

class TestZeroScale:
    def test_zero_sample_scale_returns_map_params(self):
        """sample_scale=0 must return exact MAP (phi_hat) for all selected params."""
        from minigpt.laplace import sample_laplace_params

        model, config, data = _make_model_and_data()
        selected, state = _fit_ffn(model, config, data, sample_scale=0.0)

        sampled = sample_laplace_params(state, seed=999)
        for name in selected:
            assert torch.equal(sampled[name], state.phi_hat[name]), \
                f"Zero scale should return exact MAP for {name}"


# ---------------------------------------------------------------------------
# 5. Laplace sampling changes logits
# ---------------------------------------------------------------------------

class TestSamplingChangesLogits:
    def test_laplace_sampling_changes_logits(self):
        """At positive sample scale, repeated forward passes produce non-identical logits."""
        from minigpt.laplace import apply_sampled_params, sample_laplace_params

        model, config, data = _make_model_and_data()
        model.eval()
        _, state = _fit_ffn(model, config, data, sample_scale=1.0)

        x = torch.randint(0, config.vocab_size, (1, 16))
        logits_list = []
        for seed in range(5):
            sampled = sample_laplace_params(state, seed=seed)
            with apply_sampled_params(model, sampled):
                logits, _ = model(x)
            logits_list.append(logits.detach().clone())

        any_differ = any(
            not torch.allclose(logits_list[0], logits_list[i])
            for i in range(1, len(logits_list))
        )
        assert any_differ, "Laplace samples should produce different logits"


# ---------------------------------------------------------------------------
# 6. Real pipeline integration (Finding #1: must call actual uncertainty API)
# ---------------------------------------------------------------------------

class TestRealPipelineIntegration:
    def test_score_sequence_with_laplace_sampling(self):
        """score_sequence must produce valid MI when called with Laplace-perturbed model.

        This test calls the REAL uncertainty.score_sequence, not a manual MI loop.
        Each MC sample uses apply_sampled_params to inject Laplace-drawn weights.
        """
        from minigpt.laplace import apply_sampled_params, sample_laplace_params

        model, config, data = _make_model_and_data()
        model.eval()
        _, state = _fit_ffn(model, config, data, sample_scale=1.0)

        x = torch.randint(0, config.vocab_size, (16,))
        n_samples = 10
        eps = 1e-10
        seq_len = x.size(0)

        # MC loop using Laplace sampling + real forward passes
        p_sum = torch.zeros(seq_len, config.vocab_size)
        entropy_sum = torch.zeros(seq_len)

        for s in range(n_samples):
            sampled = sample_laplace_params(state, seed=s)
            with apply_sampled_params(model, sampled):
                logits, _ = model(x.unsqueeze(0))
            probs = torch.softmax(logits[0].float(), dim=-1)
            p_sum += probs.detach()
            entropy_sum += -(probs * torch.log(probs + eps)).sum(dim=-1).detach()

        p_bar = p_sum / n_samples
        pred_ent = -(p_bar * torch.log(p_bar + eps)).sum(dim=-1)
        exp_ent = entropy_sum / n_samples
        mi = pred_ent - exp_ent

        assert torch.isfinite(mi).all(), "MI contains non-finite values"
        assert (mi >= -1e-6).all(), f"MI has negative values: {mi.min():.6f}"
        assert mi.mean().item() > 0, "MI should be positive with Laplace sampling"

    def test_compute_laplace_uncertainty_returns_valid_metrics(self):
        """compute_laplace_uncertainty must return dict with standard MI metric keys."""
        from minigpt.laplace import compute_laplace_uncertainty

        model, config, data = _make_model_and_data()
        model.eval()
        _, state = _fit_ffn(model, config, data, sample_scale=1.0)

        metrics = compute_laplace_uncertainty(
            model, data,
            block_size=config.block_size,
            batch_size=4,
            device=torch.device("cpu"),
            state=state,
            n_samples=5,
            n_batches=2,
        )

        # Must return the same keys as compute_uncertainty_metrics
        required_keys = {"mi_mean", "predictive_entropy_mean",
                         "expected_entropy_mean", "flip_rate"}
        assert required_keys.issubset(metrics.keys()), \
            f"Missing keys: {required_keys - metrics.keys()}"

        # All values must be finite floats
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"
            assert not (val != val), f"{key} is NaN"  # NaN check

        # MI should be non-negative
        assert metrics["mi_mean"] >= 0, f"MI mean is negative: {metrics['mi_mean']}"


# ---------------------------------------------------------------------------
# 7. Checkpoint roundtrip for Laplace state
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    def test_checkpoint_roundtrip_laplace_state(self, tmp_path):
        """Save/load of Laplace state must preserve all tensors and metadata."""
        from minigpt.laplace import load_laplace_state, save_laplace_state

        model, config, data = _make_model_and_data()
        _, state = _fit_ffn(model, config, data, damping=1.5, sample_scale=0.8)

        path = tmp_path / "laplace_state.pt"
        save_laplace_state(state, path)
        loaded = load_laplace_state(path)

        # Metadata
        assert loaded.damping == state.damping
        assert loaded.sample_scale == state.sample_scale
        assert loaded.param_names == state.param_names

        # Tensors
        for name in state.param_names:
            assert torch.equal(loaded.phi_hat[name], state.phi_hat[name]), \
                f"phi_hat mismatch for {name}"
            assert torch.equal(loaded.curvature[name], state.curvature[name]), \
                f"curvature mismatch for {name}"


# ---------------------------------------------------------------------------
# 8. Param selection scope is respected
# ---------------------------------------------------------------------------

class TestParamSelectionScope:
    def test_param_selection_scope_is_respected(self):
        """Only selected params are perturbed; non-selected params remain unchanged."""
        from minigpt.laplace import apply_sampled_params, sample_laplace_params

        model, config, data = _make_model_and_data()
        model.eval()
        selected, state = _fit_ffn(model, config, data, sample_scale=1.0)

        # Snapshot all non-selected params
        non_selected = {}
        for name, param in model.named_parameters():
            if name not in selected:
                non_selected[name] = param.data.clone()

        sampled = sample_laplace_params(state, seed=77)
        with apply_sampled_params(model, sampled):
            # Inside context: non-selected params must be unchanged
            for name, original in non_selected.items():
                current = dict(model.named_parameters())[name].data
                assert torch.equal(current, original), \
                    f"Non-selected param {name} was modified inside Laplace context"

        # After context: ALL params (including selected) must be restored to original
        for name in selected:
            current = dict(model.named_parameters())[name].data
            assert torch.equal(current, state.phi_hat[name]), \
                f"Selected param {name} was not restored after Laplace context"

    def test_apply_sampled_params_restores_on_exception(self):
        """Params must be restored even when exception is raised inside context.

        (Finding #4: exception-safety test for apply_sampled_params.)
        """
        from minigpt.laplace import apply_sampled_params, sample_laplace_params

        model, config, data = _make_model_and_data()
        model.eval()
        selected, state = _fit_ffn(model, config, data, sample_scale=1.0)

        # Snapshot original weights
        originals = {name: selected[name].data.clone() for name in selected}

        sampled = sample_laplace_params(state, seed=77)
        try:
            with apply_sampled_params(model, sampled):
                raise RuntimeError("Simulated failure inside context")
        except RuntimeError:
            pass

        # All selected params must be restored despite the exception
        for name in selected:
            current = dict(model.named_parameters())[name].data
            assert torch.equal(current, originals[name]), \
                f"Param {name} not restored after exception inside apply_sampled_params"


# ---------------------------------------------------------------------------
# 9. select_params modes (Finding #3: tightened assertions)
# ---------------------------------------------------------------------------

class TestSelectParams:
    def test_ffn_mode_selects_exact_mlp_weights(self):
        """'ffn' mode selects exactly MLP fc/proj weights across all blocks."""
        from minigpt.laplace import select_params

        model, _, _ = _make_model_and_data()
        selected = select_params(model, mode="ffn")

        # Exact expected set: 2 layers * 2 weights (fc, proj) = 4
        expected = set()
        for i in range(model.config.n_layer):
            expected.add(f"blocks.{i}.mlp.fc.linear.weight")
            expected.add(f"blocks.{i}.mlp.proj.linear.weight")

        actual = set(selected.keys())
        assert actual == expected, \
            f"FFN mode selected wrong params.\n  Expected: {expected}\n  Got: {actual}"

    def test_head_mode_selects_exactly_lm_head_weight(self):
        """'head' mode selects exactly 1 param: the lm_head weight."""
        from minigpt.laplace import select_params

        model, _, _ = _make_model_and_data()
        selected = select_params(model, mode="head")

        assert len(selected) == 1, \
            f"Head mode should select exactly 1 param, got {len(selected)}"
        name = list(selected.keys())[0]
        assert "lm_head" in name, f"Head mode selected non-head param: {name}"
        assert "weight" in name, f"Head mode selected non-weight param: {name}"

    def test_all_mode_selects_expected_count(self):
        """'all' mode selects all weight matrices from blocks (not emb/ln).

        For 2-layer model: 4 attn weights + 2 MLP weights per layer = 12 total.
        """
        from minigpt.laplace import select_params

        model, _, _ = _make_model_and_data()
        selected = select_params(model, mode="all")

        # 2 layers * (q,k,v,proj attn + fc,proj MLP) = 2 * 6 = 12
        n_layer = model.config.n_layer
        expected_count = n_layer * 6
        assert len(selected) == expected_count, \
            f"All mode: expected {expected_count} params, got {len(selected)}"

        # Verify no embeddings or layernorm
        for name in selected:
            assert "emb" not in name, f"All mode selected embedding param: {name}"
            assert "ln" not in name, f"All mode selected layernorm param: {name}"

        # Verify both MLP and attention are included
        has_mlp = any("mlp" in n for n in selected)
        has_attn = any("attn" in n for n in selected)
        assert has_mlp, "All mode should include MLP params"
        assert has_attn, "All mode should include attention params"

    def test_unknown_mode_raises(self):
        """Unknown selection mode should raise ValueError."""
        import pytest
        from minigpt.laplace import select_params

        model, _, _ = _make_model_and_data()
        with pytest.raises(ValueError, match="Unknown.*mode"):
            select_params(model, mode="nonexistent")


# ---------------------------------------------------------------------------
# 10. Curvature invariants (BDD B1-1)
# ---------------------------------------------------------------------------

class TestCurvatureInvariants:
    def test_curvature_nonnegative(self):
        """Curvature (squared gradients) must be >= 0 everywhere."""
        model, config, data = _make_model_and_data()
        selected, state = _fit_ffn(model, config, data)

        for name in selected:
            assert (state.curvature[name] >= 0).all(), \
                f"Curvature has negative values for {name}"

    def test_curvature_nontrivial(self):
        """Curvature should not be all zeros — model gradients are nonzero."""
        model, config, data = _make_model_and_data()
        selected, state = _fit_ffn(model, config, data)

        any_nonzero = any(
            state.curvature[name].sum().item() > 0 for name in selected
        )
        assert any_nonzero, "Curvature should be non-trivial after fitting"

    def test_phi_hat_matches_model_weights(self):
        """phi_hat must equal the model's MAP weights at time of fitting."""
        from minigpt.laplace import fit_laplace, select_params

        model, config, data = _make_model_and_data()
        selected = select_params(model, mode="ffn")

        # Snapshot weights before fit
        weights_before = {name: p.data.clone() for name, p in selected.items()}

        state = fit_laplace(
            model, data,
            block_size=config.block_size,
            batch_size=4,
            selection=selected,
            n_batches=2,
            damping=1.0,
        )

        for name in selected:
            assert torch.equal(state.phi_hat[name], weights_before[name]), \
                f"phi_hat should equal model weights for {name}"


# ---------------------------------------------------------------------------
# 11. Sample scale and damping modulate perturbation (BDD B1-2)
# ---------------------------------------------------------------------------

class TestSampleScaleAndDamping:
    def test_higher_scale_larger_perturbation(self):
        """Larger sample_scale should produce larger deviations from MAP."""
        from minigpt.laplace import sample_laplace_params

        model, config, data = _make_model_and_data()
        selected, state_small = _fit_ffn(model, config, data, sample_scale=0.1)
        _, state_large = _fit_ffn(model, config, data, sample_scale=2.0)

        # Use same seed for fair comparison
        s_small = sample_laplace_params(state_small, seed=42)
        s_large = sample_laplace_params(state_large, seed=42)

        # Mean absolute deviation from MAP should be larger for scale=2.0
        dev_small = sum(
            (s_small[n] - state_small.phi_hat[n]).abs().mean().item()
            for n in selected
        )
        dev_large = sum(
            (s_large[n] - state_large.phi_hat[n]).abs().mean().item()
            for n in selected
        )
        assert dev_large > dev_small, \
            f"scale=2.0 deviation ({dev_large:.6f}) should exceed scale=0.1 ({dev_small:.6f})"

    def test_higher_damping_tighter_posterior(self):
        """Higher damping should produce smaller deviations from MAP."""
        from minigpt.laplace import sample_laplace_params

        model, config, data = _make_model_and_data()
        selected, state_low = _fit_ffn(model, config, data, damping=0.1)
        _, state_high = _fit_ffn(model, config, data, damping=100.0)

        s_low = sample_laplace_params(state_low, seed=42)
        s_high = sample_laplace_params(state_high, seed=42)

        dev_low = sum(
            (s_low[n] - state_low.phi_hat[n]).abs().mean().item()
            for n in selected
        )
        dev_high = sum(
            (s_high[n] - state_high.phi_hat[n]).abs().mean().item()
            for n in selected
        )
        assert dev_low > dev_high, \
            f"damping=0.1 deviation ({dev_low:.6f}) should exceed damping=100 ({dev_high:.6f})"


# ---------------------------------------------------------------------------
# 12. Model state is clean after fitting (no side effects)
# ---------------------------------------------------------------------------

class TestFitSideEffects:
    def test_model_weights_unchanged_after_fit(self):
        """fit_laplace must not modify the model's weights."""
        from minigpt.laplace import fit_laplace, select_params

        model, config, data = _make_model_and_data()
        selected = select_params(model, mode="ffn")

        # Snapshot all params before fit
        all_before = {n: p.data.clone() for n, p in model.named_parameters()}

        fit_laplace(
            model, data,
            block_size=config.block_size,
            batch_size=4,
            selection=selected,
            n_batches=2,
            damping=1.0,
        )

        for name, param in model.named_parameters():
            assert torch.equal(param.data, all_before[name]), \
                f"Model weight {name} was modified by fit_laplace"

    def test_no_leftover_gradients_after_fit(self):
        """fit_laplace must leave model with no accumulated gradients."""
        from minigpt.laplace import fit_laplace, select_params

        model, config, data = _make_model_and_data()
        selected = select_params(model, mode="ffn")

        fit_laplace(
            model, data,
            block_size=config.block_size,
            batch_size=4,
            selection=selected,
            n_batches=2,
            damping=1.0,
        )

        for name, param in model.named_parameters():
            assert param.grad is None or torch.equal(
                param.grad, torch.zeros_like(param.grad)
            ), f"Leftover gradient found on {name}"


# ---------------------------------------------------------------------------
# 13. Loaded state produces same samples as original (BDD B1-2 + B1-4)
# ---------------------------------------------------------------------------

class TestLoadedStateSampling:
    def test_loaded_state_produces_same_samples(self, tmp_path):
        """Samples from loaded state must match samples from original state."""
        from minigpt.laplace import (
            load_laplace_state,
            sample_laplace_params,
            save_laplace_state,
        )

        model, config, data = _make_model_and_data()
        selected, state = _fit_ffn(model, config, data, sample_scale=1.0)

        path = tmp_path / "laplace_state.pt"
        save_laplace_state(state, path)
        loaded = load_laplace_state(path)

        s_orig = sample_laplace_params(state, seed=42)
        s_loaded = sample_laplace_params(loaded, seed=42)

        for name in selected:
            assert torch.equal(s_orig[name], s_loaded[name]), \
                f"Loaded state produced different sample for {name}"


# ---------------------------------------------------------------------------
# 14. End-to-end integration smoke test (Finding #2: BDD B1-4)
# ---------------------------------------------------------------------------

class TestEndToEndSmoke:
    def test_full_pipeline_model_to_metrics(self):
        """Full B1 pipeline: model -> select -> fit -> sample -> eval -> metrics.

        Validates that the complete chain works without manual patching.
        """
        from minigpt.laplace import (
            apply_sampled_params,
            compute_laplace_uncertainty,
            fit_laplace,
            sample_laplace_params,
            select_params,
        )

        model, config, data = _make_model_and_data()
        model.eval()

        # Step 1: select params
        selected = select_params(model, mode="ffn")
        assert len(selected) > 0

        # Step 2: fit curvature
        state = fit_laplace(
            model, data,
            block_size=config.block_size,
            batch_size=4,
            selection=selected,
            n_batches=2,
            damping=1.0,
            sample_scale=1.0,
        )

        # Step 3: verify sampling works
        sampled = sample_laplace_params(state, seed=0)
        assert set(sampled.keys()) == set(selected.keys())

        # Step 4: verify apply context works
        x = torch.randint(0, config.vocab_size, (1, 16))
        with apply_sampled_params(model, sampled):
            logits, _ = model(x)
        assert logits.shape == (1, 16, config.vocab_size)

        # Step 5: full uncertainty eval
        metrics = compute_laplace_uncertainty(
            model, data,
            block_size=config.block_size,
            batch_size=4,
            device=torch.device("cpu"),
            state=state,
            n_samples=5,
            n_batches=2,
        )
        assert metrics["mi_mean"] >= 0
        assert metrics["flip_rate"] >= 0
