# Pile Data Loader — TDD Handoff for Codex

## Task

Write **unit tests only** (TDD red phase) for the Pile domain-split data loader. The tests go in `tests/test_pile_data.py`. **Do NOT write any implementation code** — only tests. All tests must fail when run (no implementation exists yet).

---

## What You're Testing

A new function `load_pile_data()` in `minigpt/data.py` that:
1. Downloads Pile subsets from HuggingFace (streaming)
2. Tokenizes with tiktoken GPT-2 BPE
3. Caches tokenized tensors locally
4. Returns `{"train", "val", "test_id", "test_ood_*"}` dict

Plus config validation changes in `minigpt/config.py`.

---

## Files to Read First

Read these files to understand the existing patterns and conventions:

1. **`specs/pile-data-loader-spec.md`** — the BDD spec. Every behavior (B-1 through B-11) and acceptance criterion (1–10) must have corresponding tests.

2. **`tests/test_data.py`** — existing data tests. Follow the same style (pytest, fixtures, assertions). Currently 3 tests.

3. **`minigpt/data.py`** — existing data module. Understand `load_dataset()` dispatcher, `prepare_agnews_data()` return format. The new `load_pile_data()` will follow the same patterns.

4. **`minigpt/config.py`** — existing config/validation. Understand `validate_config()`, `DEFAULT_CONFIG`. New Pile validation will be added here.

5. **`AGENTS.md`** — project rules. Key rules for tests:
   - pytest only (no unittest)
   - 4 spaces, 100-char lines
   - No extra libraries beyond what's in `pyproject.toml` (+ `datasets` will be added)
   - snake_case functions, descriptive test names

---

## Test Design Constraints

### Mocking Strategy

The tests must NOT make real network calls. Mock the HuggingFace `datasets.load_dataset` function to return synthetic data. Example mock pattern:

```python
@pytest.fixture
def mock_pile_dataset(monkeypatch):
    """Mock HuggingFace datasets.load_dataset to return synthetic documents."""
    def fake_load(path, name=None, split=None, streaming=False):
        # Return an iterable of {"text": "..."} dicts
        docs = [{"text": f"Document {i} from {name} with some words."} for i in range(100)]
        return iter(docs) if streaming else docs
    monkeypatch.setattr("datasets.load_dataset", fake_load)
```

### Caching Tests

Use `tmp_path` fixture for cache directory. Monkeypatch `DATA_DIR` or the cache path to use `tmp_path`.

### Token Count Tests

Use small token counts (e.g., 1000–5000) to keep tests fast. The function should respect `pile_id_tokens` and `pile_ood_tokens`.

---

## Behaviors to Test (from spec)

### B-1: Domain loading
- Test that `load_pile_data` calls HuggingFace with correct dataset path and domain name
- Test that documents are tokenized with tiktoken

### B-2: ID domain splitting
- Test that returned dict has `train`, `val`, `test_id` keys
- Test that split ratios match `val_fraction` and `test_fraction`
- Test that splits are non-overlapping (total tokens = train + val + test_id)

### B-3: OOD domain tensors
- Test that each OOD domain produces a `test_ood_{name}` key
- Test with 1, 2, and 3 OOD domains
- Test that OOD tensors are independent of ID data

### B-4: Token count control
- Test that ID domain respects `pile_id_tokens` limit
- Test that OOD domain respects `pile_ood_tokens` limit
- Test that fewer-than-requested tokens doesn't error (uses all available)

### B-5: Local caching
- Test that `.pt` file is created after first load
- Test that second load reads from cache (mock not called again)
- Test that different token count invalidates cache

### B-6: Config surface
- Test that default config values are used when keys absent
- Test that explicit config values override defaults

### B-7: Dispatcher integration
- Test that `load_dataset(cfg, tokenizer)` with `dataset: "pile"` dispatches correctly
- Test that `"tinyshakespeare"` and `"agnews"` still work

### B-8: Domain name mapping
- Test valid domain names are accepted
- Test invalid domain name raises `ValueError` with helpful message

### B-9: Reproducibility
- Test that same seed produces identical output
- Test that different seed produces different output

### B-10: Progress reporting (light touch)
- Test that loading prints domain name and token count (capture stdout)

### B-11: Validation
- Test `validate_config` rejects empty `pile_id_domains`
- Test `validate_config` rejects empty `pile_ood_domains`
- Test `validate_config` rejects overlapping ID/OOD domains
- Test `validate_config` rejects non-positive token counts
- Test `validate_config` accepts valid Pile config

---

## Acceptance Criteria to Test

From the spec, each of these 10 criteria needs at least one test:

1. Returns dict with train/val/test_id and test_ood_* keys
2. All tensors are non-empty torch.long 1D
3. Train + val + test_id token count matches ID target (within rounding)
4. OOD tensors have at most pile_ood_tokens tokens
5. Cache files created in data/pile/
6. Second call loads from cache
7. Different pile_id_tokens invalidates cache
8. Invalid domain name raises ValueError
9. Overlapping ID/OOD raises ValueError
10. Deterministic output given fixed seed

---

## Example Config for Tests

```python
def make_pile_config(
    id_domains=None, ood_domains=None,
    id_tokens=5000, ood_tokens=2000,
    val_fraction=0.1, test_fraction=0.1,
    seed=1337,
):
    """Build a minimal config dict for Pile data loading tests."""
    return {
        "data": {
            "dataset": "pile",
            "pile_id_domains": id_domains or ["wikipedia_en"],
            "pile_ood_domains": ood_domains or ["arxiv"],
            "pile_id_tokens": id_tokens,
            "pile_ood_tokens": ood_tokens,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
        },
        "model": {
            "block_size": 256, "n_layer": 4, "n_head": 4, "n_embd": 256,
            "dropout": 0.2, "bias": True,
            "bayes_head": {"enabled": False},
            "bayes_ffn": {"enabled": False},
            "bayes_attn_v": {"enabled": False},
        },
        "train": {
            "steps": 100, "batch_size": 8, "block_size": 256,
            "lr": 3e-4, "weight_decay": 0.1, "warmup_steps": 10,
            "min_lr": 1e-5, "grad_clip": 1.0, "eval_interval": 50,
            "eval_iters": 5, "checkpoint_interval": 100,
            "gradient_accumulation_steps": 1, "kl_weight": 0.0,
            "kl_annealing_steps": 0, "adam_beta1": 0.9, "adam_beta2": 0.95,
            "seed": seed, "device": "cpu",
        },
        "eval": {
            "sample_tokens": 200, "temperature": 0.8, "num_samples": 30,
            "n_perplexity_batches": 20,
        },
    }
```

---

## Output

A single file: `tests/test_pile_data.py`

Expected test count: ~25–30 tests covering all behaviors and acceptance criteria.

All tests must fail when run (since `load_pile_data` doesn't exist yet). Tests that check `validate_config` behavior for Pile configs will also fail since that validation doesn't exist yet.

---

## Style

- Follow existing test style in `tests/test_data.py`, `tests/test_bayesian.py`
- Use `pytest` fixtures, parametrize where appropriate
- Descriptive test names: `test_pile_returns_all_required_keys`, `test_pile_id_split_ratios`, etc.
- Group related tests with comments or class
- No docstrings needed on individual tests (names should be self-documenting)
- 100-char line limit, 4-space indent
