# B2 BLoB LoRA Implementation Review

Date: 2026-03-10

Scope: review of the implementation against `specs/b2-blob-lora-spec.md`.

## Findings

### 1. High: invalid LoRA config can silently produce a frozen no-op model

Relevant code:
- `minigpt/lora.py:119`
- `minigpt/lora.py:123`
- `minigpt/config.py:205`

`inject_lora()` freezes the entire model first:

```python
for p in model.parameters():
    p.requires_grad_(False)
```

It only injects adapters inside the `if lora_config.target == "ffn":` branch. If `lora.target`
is misspelled or unsupported, no adapters are added and the model is left with zero trainable
parameters.

There is currently no config validation for:
- allowed `lora.target` values
- `lora.rank > 0`
- positive `lora.alpha`
- positive `lora.prior_std`
- positive `lora.init_g`

Impact:
- a misconfigured B2 run can appear to execute normally while updating nothing
- this is especially risky because the failure mode is silent

Recommended fix:
- validate the `lora` section in `validate_config()`
- make `inject_lora()` raise `ValueError` for unsupported targets instead of silently returning
  a fully frozen model
- optionally assert after injection that at least one parameter remains trainable

## 2. Medium: phase-2 CLI can silently run with the wrong config

Relevant code:
- `experiments/experiment_setup.py:38`
- `minigpt/config.py:20`
- `experiments/b2_blob_lora.py:246`
- `experiments/b2_blob_lora.py:266`

`parse_base_args()` starts from `DEFAULT_CONFIG`, and `b2_blob_lora.py` does not require
`--config` for `--phase finetune` or `--phase full`.

If the user forgets:

```bash
--config configs/b2_blob_agnews.yaml
```

phase 2 will run with the default repo config instead of the B2 AG News finetune config. Since
`DEFAULT_CONFIG` is a TinyShakespeare baseline config, this is a real pipeline footgun.

Impact:
- B2 finetune can accidentally run on the wrong dataset/settings
- the command may still complete, making the mistake easy to miss

Recommended fix:
- require `--config` for B2 finetune/full unless an explicit B2 config is loaded elsewhere
- or add a B2-specific guard that enforces:
  - `data.dataset == "agnews"`
  - presence of `lora.base_checkpoint`
  - presence of a `lora` config section

## 3. Medium: generated sample is labeled "mean weights" but is actually stochastic

Relevant code:
- `experiments/b2_blob_lora.py:179`
- `minigpt/evaluate.py:43`
- `minigpt/model.py:158`

Phase 2 prints:

```python
print(f"\n=== generated sample (mean weights) ===\n{sample}\n{'=' * 40}")
```

But `generate_text()` calls:

```python
out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
```

without `use_mean=True`. For B2, `model.generate()` is therefore sampling LoRA weights during
generation, so the sample is not a mean-weight sample.

Impact:
- the label in logs is incorrect
- generated text is not deterministic under the advertised evaluation mode
- this makes the sample inconsistent with the mean-weight perplexity evaluation immediately above

Recommended fix:
- either wrap sample generation in `use_mean_weights(model)` or pass `use_mean=True` through
  `generate_text()`
- if stochastic generation is desired, relabel it explicitly

## Notes

- `uv run pytest tests/ -q` passes locally: `84 passed`.
- I did not run a full B2 pretrain plus finetune training job for this review.
- Secondary observation: `experiments/b2_blob_lora.py` logs `n_params` before LoRA injection,
  so the headline total excludes the added 122,880 LoRA parameters, although LoRA-specific counts
  are logged separately.
