# Technical Reference

Operational details for specific subsystems. Read when working on data, configs, metrics, experiments, or tests.

---

## Datasets

**TinyShakespeare:** ~1MB, ~304K BPE tokens. Single-domain. `load_shakespeare()` auto-downloads.

**AG News:** 127.6K articles, 4 categories (1=World, 2=Sports, 3=Business, 4=Sci/Tech). Topic split: ID (default World+Sports) → train/val/test_id (80/10/10); OOD (default Business+Sci/Tech) → test_ood. ~2.4M train tokens. Configurable via `data.id_categories`, `data.ood_categories`.

**The Pile (C milestone):** Domain-split from `ArmelR/the-pile-splitted` (HuggingFace). ID: Wikipedia + StackExchange (~200M tokens). OOD: ArXiv, FreeLaw, PubMed. LoRA fine-tune: HackerNews. Cache: `data/pile/{domain}_{token_limit}.pt`.

**Dispatcher:** `load_dataset(cfg, tokenizer)` → `{"train", "val", "test_id", "test_ood"}`.

## Tokenization

BPE via `tiktoken` (GPT-2 encoding, vocab_size=50257).

## Experiment Tracking

- **MLflow** (local, `sqlite:///mlflow.db`). `mlflow.db` and `mlruns/` are gitignored.
- Logs: hyperparams, train/val loss, perplexity, LR, test perplexity, MI metrics, sigma stats.
- Tags: `dataset`, `milestone`, `gpu`.
- `--no-mlflow` flag to disable. UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Epistemic Uncertainty Measurement

- Temperature = 0. All stochasticity from Bayesian weights.
- N forward passes (10–30) with weight sampling.
- Primary metric: **MI** (predictive_entropy − expected_entropy) — pure epistemic uncertainty.
- Secondary: flip rate, per-token MI scores.
- MI ratio: MI_OOD / MI_ID. Values >1.0 indicate OOD detection.

## Configuration System

Pipeline: `DEFAULT_CONFIG → YAML file → CLI --set overrides → validate`.

Key conventions:
- `vocab_size` always from tokenizer, never config.
- `model.bayes_head` / `model.bayes_ffn` / `model.bayes_attn_v` for Bayesian component configs.
- `train.kl_weight` is global for all enabled Bayesian components.
- `posthoc_method: "laplace"|"tfb"` disambiguates post-hoc dispatch.
- **PyYAML gotcha:** write `3.0e-4` not `3e-4` (parsed as string without decimal).
- Checkpoint resume: saves full config, `best_val_loss`, RNG states. Best-checkpoint: ELBO for Bayesian, CE for deterministic.

## Tests (217 total: 134 core + 83 pipeline)

```
tests/
  test_model.py               # Model architecture, weight tying, perplexity
  test_data.py                # Data loading, split sizes
  test_reproducibility.py     # Seed determinism
  test_bayesian.py            # Bayesian layers, KL, sampling, MI, architecture paths
  test_lora.py                # LoRA injection, BLoB forward/KL, config, context managers, grads
  test_laplace.py             # Laplace fitting, sampling, selection modes, roundtrip, LoRA selection
  test_b3_deterministic_lora.py  # DeterministicLoRALinear forward, no-KL, param count
  test_tfb.py                 # TFB SVD, variance structure, search convergence, sampling
  test_pile_data.py           # Pile data loader: splits, domains, config, caching, multiprocessing
  test_c_pipeline.py          # Pipeline: CLI, config gen, helpers, runner state/loop, policies,
                              #   posthoc templates/fit/integration, BLoB→DeterministicLoRA conversion
```

## References

- **BLoB** (NeurIPS 2024) — Bayesian LoRA by backprop. [arXiv:2406.11675](https://arxiv.org/abs/2406.11675)
- **TFB** (NeurIPS 2025) — Training-Free Bayesianization. [arXiv:2412.05723](https://arxiv.org/abs/2412.05723)
- **Laplace-LoRA** (2023) — Laplace approximation on LoRA params. [arXiv:2308.13111](https://arxiv.org/abs/2308.13111)
- **Laplace Redux** (NeurIPS 2021) — Effortless Bayesian Deep Learning. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)
- **ICLA** (WACV 2025) — Identity Curvature Laplace for OOD. [arXiv:2312.10464](https://arxiv.org/abs/2312.10464)
