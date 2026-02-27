# A2 — Bayesian FFN Layers: Design Document

This document is the implementation blueprint for milestone A2: making the feed-forward network (FFN / MLP) layers in each transformer block Bayesian, while keeping the output head deterministic (weight-tied).

---

## 1. Motivation — Why FFN After the Output Head?

A1 proved the Bayesian infrastructure works: ELBO training, MC sampling, MI measurement, qualitative eval — all validated. But A1's MI ratio capped at **1.2–1.4x** across three training runs. The diagnosis was structural:

**The output head is a vocabulary projection, not a knowledge store.** It maps hidden states → logit space. Its weights learn which tokens are likely given a hidden representation, but they don't encode *what the model knows about topics*. The output head detects OOD at the **lexical level** — unfamiliar token co-occurrences — not at the semantic level.

**FFN layers store factual knowledge.** This is well-established in mechanistic interpretability research (Geva et al., 2021; Meng et al., 2022). The MLP acts as a key-value memory: the up-projection (`fc`) activates "knowledge keys," the down-projection (`proj`) retrieves "knowledge values." When the model encounters a topic it hasn't seen, the FFN activations will be in unfamiliar regions → Bayesian FFN weights should express high epistemic uncertainty → higher MI on OOD text.

**Expected outcome:** MI ratio should increase substantially beyond the 1.2–1.4x ceiling observed in A1. The model should be uncertain about *what it knows*, not just about *which word comes next*.

---

## 2. What Changes from A1

### 2.1. Which Layers Are Bayesian

| Component | A1 | A2 |
|-----------|----|----|
| Embedding (`token_emb`) | Deterministic | Deterministic |
| Positional embedding (`pos_emb`) | Deterministic | Deterministic |
| Attention (Q/K/V + proj) | Deterministic | Deterministic |
| FFN (`MLP.fc`, `MLP.proj`) | Deterministic | **Bayesian** |
| Output head (`lm_head`) | **Bayesian** | Deterministic |
| Weight tying | Disabled | **Restored** |

### 2.2. Parameter Count Impact

With `n_embd=256`, `n_layer=4`, `vocab_size=50257`:

**FFN per block:**
- `fc`: 256 → 1024 = 262,144 weight params + 1,024 bias = 263,168
- `proj`: 1024 → 256 = 262,144 weight params + 256 bias = 262,400
- Per block total: 525,568 params → × 2 (mu + rho) = **1,051,136 Bayesian params**

**Across 4 blocks:** 4 × 1,051,136 = **4,204,544 Bayesian params** (~4.2M)

**Compare to A1:** 25.7M Bayesian params (in the output head alone).

**Total model size:**
- A0 (deterministic, weight-tied): ~16M
- A1 (Bayesian head, no weight tying): ~42M
- **A2 (Bayesian FFN, weight-tied): ~20M**

A2 is **smaller** than A1 because weight tying is restored (the head shares weights with the embedding) and the FFN has far fewer params than the vocab-sized head. This is a significant advantage: the model fits easily on GPU, trains faster, and the Bayesian parameters are proportionally more meaningful (each weight has ~640 training tokens per parameter vs ~105 in A1).

---

## 3. Architecture Changes

### 3.1. GPTConfig — Separate Bayes Configs per Component

Currently, `GPTConfig.bayes` controls only the output head and is ambiguously named. For A2, we rename it to `bayes_head` for clarity and add `bayes_ffn`:

```python
@dataclass
class GPTConfig:
    ...
    bayes_head: BayesConfig = field(default_factory=BayesConfig)  # output head
    bayes_ffn: BayesConfig = field(default_factory=BayesConfig)   # FFN in transformer blocks
```

- `bayes_head` controls `lm_head` (renamed from `bayes` for clarity)
- `bayes_ffn` controls `MLP.fc` and `MLP.proj` in every `Block`
- Attention always deterministic (future work)

### 3.2. Block — Split Bayes Config for Attention vs FFN

Current signature: `Block(config, bayes)` — passes same `bayes` to both `CausalSelfAttention` and `MLP`.

New signature — explicit params (Option A chosen for flexibility with future attention-Bayesian experiments):

```python
class Block(nn.Module):
    def __init__(self, config: GPTConfig, bayes_attn: BayesConfig, bayes_ffn: BayesConfig):
        self.attn = CausalSelfAttention(config, bayes_attn)
        self.mlp = MLP(config, bayes_ffn)
```

### 3.3. MiniGPT — Wiring

```python
class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        ...
        no_bayes = BayesConfig(enabled=False)
        self.blocks = nn.ModuleList([
            Block(config, bayes_attn=no_bayes, bayes_ffn=config.bayes_ffn)
            for _ in range(config.n_layer)
        ])
        self.lm_head = make_linear(config.n_embd, config.vocab_size, config.bayes_head, bias=False)

        # Weight tying: only when head is deterministic
        if not config.bayes_head.enabled:
            self.lm_head.linear.weight = self.token_emb.weight
```

For A2: `config.bayes_head.enabled = False` (head deterministic, weight-tied), `config.bayes_ffn.enabled = True` (FFN Bayesian).

### 3.4. `forward_body()` — No Longer a Shortcut

In A1, `forward_body()` was an optimization: run the deterministic transformer body once, then call `lm_head` N times. In A2, the body itself is stochastic (FFN samples different weights each call), so each MC sample requires a full forward pass.

`forward_body()` still works — it returns different hidden states each time due to Bayesian FFN. But it's no longer an efficiency shortcut. It remains useful as a clean separation of body vs head for code clarity.

---

## 4. Config Changes

### 4.1. YAML Config — `configs/a2_agnews.yaml`

```yaml
experiment:
  name: a2-agnews
  run_name: a2-agnews
  mlflow_uri: "sqlite:///mlflow.db"

data:
  dataset: agnews
  id_categories: [1, 2]      # World, Sports
  ood_categories: [3, 4]     # Business, Sci/Tech
  val_fraction: 0.1
  test_fraction: 0.1

model:
  block_size: 256
  n_layer: 4
  n_head: 4
  n_embd: 256
  dropout: 0.2
  bias: true
  bayes_head:
    enabled: false            # head deterministic — weight tying restored
  bayes_ffn:
    enabled: true
    prior_std: 1.0
    kl_weight: 0.2
    init_rho: -3.0            # σ₀ ≈ 0.049 — see §6 for rationale

train:
  steps: 50000
  batch_size: 32
  block_size: 256
  lr: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 1000
  min_lr: 1.0e-5
  grad_clip: 1.0
  eval_interval: 2000
  eval_iters: 20
  checkpoint_interval: 5000
  checkpoint_dir: "data/checkpoints"
  gradient_accumulation_steps: 1
  kl_annealing_steps: 10000
  adam_beta1: 0.9
  adam_beta2: 0.95
  seed: 1337
  device: auto

eval:
  sample_tokens: 200
  temperature: 0.8
  num_samples: 20            # reduced from 30 — full passes are more expensive
  n_perplexity_batches: 20
  qualitative_prompts_per_category: 5
  qualitative_max_new_tokens: 100
  qualitative_seed: 42
```

### 4.2. DEFAULT_CONFIG Update

Add `bayes_ffn` block (default disabled), rename `bayes` → `bayes_head`:

```python
"model": {
    ...
    "bayes_head": { "enabled": False, ... },
    "bayes_ffn": { "enabled": False, "prior_std": 1.0, "kl_weight": 1.0, "init_rho": -1.0 },
}
```

### 4.3. Config Builder — `build_gpt_config()`

Extend to read `bayes_head` and `bayes_ffn` from config dict via shared helper:

```python
def _build_bayes_config(d: dict) -> BayesConfig:
    return BayesConfig(
        enabled=d.get("enabled", False),
        prior_std=d.get("prior_std", 1.0),
        kl_weight=d.get("kl_weight", 1.0),
        init_rho=d.get("init_rho", -5.0),
    )

def build_gpt_config(cfg: dict, vocab_size: int) -> GPTConfig:
    m = cfg["model"]
    return GPTConfig(
        ...,
        bayes_head=_build_bayes_config(m.get("bayes_head", {})),
        bayes_ffn=_build_bayes_config(m.get("bayes_ffn", {})),
    )
```

### 4.4. Backward Compatibility

All existing A0/A1 configs updated: `bayes` renamed to `bayes_head`, `bayes_ffn` added with `enabled: false`. `DEFAULT_CONFIG` and `build_gpt_config` use `.get()` with fallbacks, so configs missing either key still work.

---

## 5. Uncertainty Evaluation Changes

### 5.1. The A1 Efficiency Shortcut No Longer Applies

A1 `compute_uncertainty_metrics` ran the transformer body once and called `lm_head` N times:
```
body → h          # 1 pass (deterministic body)
lm_head(h) × N   # N cheap matmuls (Bayesian head)
```

A2 requires N **full** forward passes (body is stochastic):
```
(body → h → lm_head) × N   # N full passes (Bayesian FFN → different h each time)
```

### 5.2. Auto-Detection — `_has_bayesian_body()`

```python
def _has_bayesian_body(model: MiniGPT) -> bool:
    """Check if any BayesianLinear exists in transformer blocks."""
    for block in model.blocks:
        for m in block.modules():
            if isinstance(m, BayesianLinear):
                return True
    return False
```

### 5.3. Dual-Path Streaming

Modify `compute_uncertainty_metrics` and `score_sequence` to choose the right path:

```python
if _has_bayesian_body(model):
    # A2 path: N full forward passes
    metrics = _stream_metrics_full(model, x[b:b+1], n_samples)
else:
    # A1 path: body once, head N times (efficient)
    h = model.forward_body(x)
    metrics = _stream_metrics(model, h[b:b+1], n_samples)
```

### 5.4. `_stream_metrics_full()` — New Function

```python
def _stream_metrics_full(
    model: MiniGPT,
    x: torch.Tensor,        # (1, seq_len) input tokens
    n_samples: int,
) -> dict[str, torch.Tensor]:
    """Streaming MC metrics with full forward pass per sample.

    Like _stream_metrics, but runs the entire model (not just lm_head).
    Used when the transformer body contains BayesianLinear layers (A2+).
    """
    seq_len = x.size(1)
    device = x.device
    vocab = model.config.vocab_size

    p_sum = torch.zeros(seq_len, vocab, device=device)
    entropy_sum = torch.zeros(seq_len, device=device)
    argmaxes = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)

    use_amp = device.type == "cuda"
    for s in range(n_samples):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x)  # full forward pass — new weight sample each time
        probs = F.softmax(logits[0].float(), dim=-1)
        p_sum.add_(probs)
        entropy_sum.add_(-(probs * torch.log(probs + 1e-10)).sum(dim=-1))
        argmaxes[s] = probs.argmax(dim=-1)

    p_bar = p_sum / n_samples
    predictive_entropy = -(p_bar * torch.log(p_bar + 1e-10)).sum(dim=-1)
    expected_entropy = entropy_sum / n_samples
    mi = predictive_entropy - expected_entropy
    mode_tokens = argmaxes.mode(dim=0).values
    flip_rate = (argmaxes != mode_tokens.unsqueeze(0)).float().mean(dim=0)

    return {
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mi": mi,
        "flip_rate": flip_rate,
    }
```

### 5.5. Performance Considerations

**Cost comparison (per eval batch of B elements, N MC samples):**
- A1: 1 body pass + B × N head calls. With B=32, N=30: 32 body + 960 matmuls ≈ **~35 forward-equiv.**
- A2: B × N full passes. With B=32, N=20: **640 full forward passes.**

A2 eval is ~18x more expensive. Mitigations:
1. Reduce `eval.num_samples` from 30 → 20 (MI estimates stabilize well by N=20).
2. The A2 model is ~2x smaller than A1 (20M vs 42M params), so each pass is cheaper.
3. Eval runs after training — wall time impact is bounded (estimated: 5–10 min on RTX 4070).

**VRAM is not a bottleneck.** Each forward pass processes one batch element at a time during MC sampling. Peak memory is O(seq_len × vocab) for the accumulator — same as A1. The full-pass path doesn't increase memory, only compute.

---

## 6. Hyperparameter Rationale

### 6.1. init_rho = -3.0 (σ₀ ≈ 0.049)

A1 taught us the optimal sigma window is **σ ≈ 0.1–0.3**. Starting below that and letting training push sigma up is safer than starting in the middle and overshooting.

Why `-3` for A2 vs `-2` for A1:
- **Data-per-parameter ratio:** A2 has 4.2M Bayesian params vs 2.7M training tokens (ratio ≈ 0.64). A1 had 25.7M Bayesian params vs 2.7M tokens (ratio ≈ 0.10). With more data per param, the posterior should concentrate faster and more tightly — starting with a smaller sigma is appropriate.
- **Softplus gradient at rho=-3:** `sigmoid(-3) = 0.047` — 21x attenuated vs the unsaturated regime, but not the 143x attenuation of rho=-5 that caused A1 R1's slow sigma growth. sigma should reach the 0.1–0.3 sweet spot within ~10K steps.
- **Conservative start:** If init_rho=-3 is too cold, we can always bump to -2. Going the other way (starting too warm and getting noisy training) is harder to recover from.

### 6.2. kl_weight = 0.2

Same as A1's best run. The effective KL contribution at init:

```
KL per param (init) ≈ log(1/0.049) + 0.049²/2 - 0.5 ≈ 2.52 nats
Total KL ≈ 4.2M × 2.52 ≈ 10.6M nats
Effective per-token = 0.2 × 10.6M / 2.7M ≈ 0.79 nats
CE at init ≈ 10.8
KL fraction ≈ 7% of ELBO
```

This is a reasonable starting point. The KL fraction was ~17% for A1 at init — A2 starts milder because fewer params.

### 6.3. prior_std = 1.0

Same as A1. Standard N(0, 1) prior per weight. The prior_std could be reduced to 0.1–0.3 to penalize wide posteriors less aggressively (cold posterior effect), but we'll start with the standard value and adjust if needed.

### 6.4. kl_annealing_steps = 10000

Same as A1 R2. Linear warmup of KL weight from 0 → full over 10K steps. Gives the model time to learn the CE task before the KL penalty fully activates.

### 6.5. eval.num_samples = 20

Reduced from 30 to partially offset the cost of full-model MC passes. MI estimates converge well with N=20 (variance of MI estimate ∝ 1/N; going from 30 → 20 increases variance by ~50% but saves 33% eval time). If MI ratio is clear at N=20, the result is robust.

---

## 7. Training — No Changes Needed

The training loop is already generic:

1. `model.kl_loss()` calls `sum_kl_loss(model)`, which iterates all `BayesianModule` submodules. Works automatically whether Bayesian layers are in the head, FFN, or both.
2. ELBO = CE + kl_scale × KL. The `kl_weight` and `num_train_tokens` normalization are independent of where the Bayesian layers live.
3. Best-ELBO checkpoint selection (from A1 R2) applies unchanged.
4. KL annealing works the same way.

The only training consideration: **KL is much smaller** in A2 (4.2M Bayesian params vs 25.7M). The KL term will be a smaller fraction of the ELBO, so the CE loss dominates more. This should lead to **better ID perplexity** than A1 (closer to A0's 49.1 reference).

---

## 8. Experiment Script — `experiments/a2_bayes_ffn.py`

The A2 experiment script is structurally identical to `a1_bayes_output.py`. Changes:

1. **Milestone tag**: `"a2"` instead of `"a1"`.
2. **Param counting**: count Bayesian params in FFN layers (not head).
3. **Print model info**: show which components are Bayesian (FFN vs head).
4. **Everything else**: identical — same data loading, training call, perplexity eval, uncertainty eval, qualitative eval, MLflow logging.

The script should be a copy-and-adapt of `a1_bayes_output.py`, not a shared function, to keep experiments self-contained and reproducible.

---

## 9. Test Changes

### 9.1. Existing Tests — Must Continue Passing

All 38 existing tests must pass. The key risk is `test_bayesian.py`, which tests selective Bayesian architecture (6 tests). These tests construct models with `bayes_head.enabled=True/False` and check which layers are `BayesianLinear`. They need to be updated to account for the new `bayes_ffn` field.

### 9.2. New Tests

**Architecture tests (in `test_bayesian.py`):**
- FFN-only Bayesian: confirm `MLP.fc` and `MLP.proj` are `BayesianLinear` in each block, attention linears are `DeterministicLinear`, head is `DeterministicLinear`.
- Weight tying: when `bayes_head.enabled=False` and `bayes_ffn.enabled=True`, confirm `lm_head.linear.weight is model.token_emb.weight`.
- Combined (FFN + head): both FFN linears and head are `BayesianLinear`, no weight tying.
- KL non-negativity: same tests as A1 but with FFN-only Bayesian config.
- Sampling variance: FFN outputs differ across forward passes (body is stochastic).

**Uncertainty path tests:**
- `_has_bayesian_body` returns True for A2 config, False for A1 config, False for A0 config.
- `_stream_metrics_full` produces valid MI/entropy/flip_rate tensors.
- Full-model MC sampling produces different logits across samples (unlike A1 where body output was fixed).

---

## 10. Implementation Plan

### Step 1: Config + Architecture (model.py, layers.py, config.py)
- Add `bayes_ffn` field to `GPTConfig` (default disabled).
- Change `Block.__init__` to accept separate `bayes_attn` and `bayes_ffn` params.
- Update `MiniGPT.__init__` to wire `config.bayes_ffn` to blocks and `config.bayes_head` to head.
- Update `DEFAULT_CONFIG` with `bayes_ffn` section.
- Update `build_gpt_config()` to read `bayes_ffn` from YAML.
- Add `bayes_ffn: { enabled: false, ... }` to all existing YAML configs (explicit-config rule).

### Step 2: Uncertainty eval (uncertainty.py)
- Add `_has_bayesian_body()` helper.
- Add `_stream_metrics_full()` function (full forward pass per MC sample).
- Modify `compute_uncertainty_metrics()` and `score_sequence()` to auto-select A1 vs A2 path.

### Step 3: Experiment script + config
- Create `configs/a2_agnews.yaml`.
- Create `experiments/a2_bayes_ffn.py` (adapt from a1_bayes_output.py).

### Step 4: Tests
- Update existing selective-Bayesian tests for new Block signature.
- Add new architecture tests for FFN-only and combined configs.
- Add uncertainty path-detection test.

### Step 5: Training run
- Run `python experiments/a2_bayes_ffn.py --config configs/a2_agnews.yaml`.
- Compare test_id_ppl to A0 reference (49.1) — expect closer match since fewer Bayesian params.
- Compare MI ratio to A1 best (1.36x) — expect significant improvement.
- Log to MLflow, document results in AGENTS.md.

---

## 11. Success Criteria

| Metric | A0 (baseline) | A1 (best) | A2 target |
|--------|---------------|-----------|-----------|
| test_id_ppl | 49.1 | 56.3 | < 55 (closer to A0) |
| MI ratio (OOD/ID) | — | 1.36x | > 1.5x (meaningful gap) |
| MI (ID) | — | 0.293 | > 0 (model has some weight uncertainty) |
| Training time | 2.8 hrs | ~64 min | ~60 min (smaller model) |

**Primary success metric:** MI ratio > 1.5x — a clear improvement over A1's ceiling, demonstrating that FFN-layer uncertainty captures semantic OOD signal beyond vocabulary-level detection.

**Secondary:** ID perplexity regression < 15% vs A0 (i.e., test_id_ppl < 56.5).

---

## 12. Risks and Mitigations

**Risk 1: MI ratio doesn't improve over A1.**
- Possible cause: 4 layers × 256d is too small for the FFN to develop topic-specialized weights.
- Mitigation: try larger model (n_embd=384 or n_layer=6). But first check if sigma is in the right range.

**Risk 2: Posterior collapse (σ → 0).**
- Same risk as A1 R0. Mitigated by starting with kl_weight=0.2 and kl_annealing=10K (proven to work in A1).

**Risk 3: Sigma overshoots (σ too large, uniform noise).**
- Same as A1 R2. Mitigated by starting with init_rho=-3 (conservative). If sigma grows past 0.3, consider reducing kl_weight.

**Risk 4: Eval takes too long.**
- N=20 full passes × 20 batches × 32 batch_size = 12,800 forward passes. At 106K tok/s and 256 tokens/pass: ~31 sec. Total eval (ID + OOD + qualitative) ≈ 2–5 minutes. Acceptable.

**Risk 5: Weight decay interacts with Bayesian params.**
- `_configure_optimizer` applies weight_decay=0.1 to all 2-D weight tensors, including `weight_mu` and `weight_rho`. Weight decay on mu is equivalent to a Gaussian prior (redundant with KL but not harmful). Weight decay on rho pushes rho toward 0 (σ ≈ 0.69), which acts like a prior on sigma. This could interact with the KL prior.
- **Mitigation**: exclude `weight_rho` and `bias_rho` from weight decay. Add them to the no-decay group. This is a clean fix that separates the regularization roles: KL handles the Bayesian prior, weight decay handles the deterministic parameters.

---

## 13. Future: FFN + Head Combined (A2b)

After validating FFN-only, we can optionally run a combined experiment:

```yaml
model:
  bayes_head:
    enabled: true             # head Bayesian (no weight tying)
    init_rho: -4.0            # keep head sigma small — not the main signal
  bayes_ffn:
    enabled: true
    init_rho: -3.0
```

This would show whether FFN and head uncertainty are **additive** or **redundant**. If additive, the combined model should have MI ratio > A2 alone. If redundant, there's no benefit to making the head Bayesian once FFN captures the semantic uncertainty.

Expected: **mostly redundant** — FFN captures the semantic OOD signal, head adds marginal lexical signal. But worth verifying.

---

## 14. Reference Summary

| Item | Value |
|------|-------|
| Spec | `specs/a2-bayesian-ffn.md` (this file) |
| Config | `configs/a2_agnews.yaml` |
| Experiment | `experiments/a2_bayes_ffn.py` |
| Bayesian layers | MLP.fc, MLP.proj (4 blocks) |
| Bayesian params | ~4.2M (vs A1's 25.7M) |
| Total params | ~20M (vs A1's 42M) |
| Key hyperparams | init_rho=-3, kl_weight=0.2, kl_annealing=10K, steps=50K |
| MC samples for eval | N=20 |
| Success metric | MI ratio > 1.5x |
| Baseline comparisons | A0 (ppl=49.1), A1 best (MI ratio=1.36x) |
