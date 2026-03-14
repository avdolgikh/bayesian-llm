# Pile Domain-Split Data Loader — BDD Specification

## Context

The C milestone needs a larger dataset than AG News (~5M tokens) to train a 76M-param model. We use **The Pile** (uncopyrighted) with domain-based ID/OOD splits. Each Pile document has a source-domain label, giving us natural semantic boundaries for epistemic uncertainty evaluation.

Parent spec: `specs/c-milestone-spec.md`.

---

## 1. User Stories

### US-1: Load multi-domain data for full-weight experiments (C0/C1/C2)
**As** a researcher running C0/C1/C2,
**I want** to load training data from 2 ID domains and eval data from 3 OOD domains,
**so that** I can train a base model and evaluate MI separation across semantically distinct domains.

### US-2: Load multi-domain data for LoRA experiments (C3/C4)
**As** a researcher running C3/C4,
**I want** to load a pretrain domain, a fine-tune domain, and OOD eval domains,
**so that** I can pretrain → LoRA fine-tune → evaluate OOD uncertainty in a three-way split.

### US-3: Subsample to target token counts
**As** a researcher with limited GPU time,
**I want** to specify target token counts per domain,
**so that** I can control training time without downloading the entire Pile.

### US-4: Cache tokenized data locally
**As** a researcher running experiments repeatedly,
**I want** tokenized data cached to disk after first download,
**so that** subsequent runs start instantly without re-downloading or re-tokenizing.

### US-5: Integrate with existing config system
**As** a researcher,
**I want** to configure Pile loading via the same YAML config and `--set` overrides,
**so that** the workflow is consistent with AG News experiments.

### US-6: Integrate with existing training/eval pipeline
**As** a researcher,
**I want** the data loader to return the same dict structure as AG News,
**so that** training loop, eval functions, and MLflow logging work without modification.

---

## 2. Behaviors

### B-1: Domain loading from HuggingFace

**Given** a config with `data.dataset: pile` and domain lists,
**When** `load_dataset(cfg, tokenizer)` is called,
**Then** it downloads the specified Pile subsets via HuggingFace `datasets` library (streaming mode),
**And** tokenizes each document with GPT-2 BPE (tiktoken),
**And** concatenates all tokens per domain into a single tensor.

### B-2: ID domain splitting (train/val/test_id)

**Given** one or more ID domains with a target token count,
**When** data is prepared,
**Then** tokens from all ID domains are concatenated (in domain order, shuffled within each domain),
**And** split into train / val / test_id using `val_fraction` and `test_fraction`,
**And** the splits are non-overlapping contiguous slices.

### B-3: OOD domain tensors

**Given** one or more OOD domains,
**When** data is prepared,
**Then** each OOD domain produces a separate tensor keyed as `test_ood_{domain_name}`,
**And** all OOD keys are included in the returned dict.

Example: `{"train": ..., "val": ..., "test_id": ..., "test_ood_arxiv": ..., "test_ood_freelaw": ..., "test_ood_pubmed": ...}`

### B-4: Token count control

**Given** `data.pile_tokens_per_domain: N` in config,
**When** loading each domain,
**Then** at most $N$ tokens are kept (streaming stops after $N$ tokens reached),
**And** the default is 100,000,000 (100M tokens) for ID domains,
**And** the default is 10,000,000 (10M tokens) for OOD domains.

If the domain has fewer tokens than the target, all available tokens are used (no error).

### B-5: Local caching

**Given** a domain has been downloaded and tokenized,
**When** `load_dataset` is called again with the same config,
**Then** it loads from `data/pile/{domain_name}_{n_tokens}.pt` (cached tensor),
**And** does NOT re-download or re-tokenize.

Cache key includes domain name and target token count. If the target changes, the cache is invalidated (re-downloaded).

### B-6: Config surface

New config keys under `data`:

```yaml
data:
  dataset: pile
  pile_id_domains:
    - wikipedia_en
    - stackexchange
  pile_ood_domains:
    - arxiv
    - freelaw
    - pubmed_abstracts
  pile_id_tokens: 100000000     # 100M tokens per ID domain
  pile_ood_tokens: 10000000     # 10M tokens per OOD domain
  val_fraction: 0.05
  test_fraction: 0.05
```

### B-7: Dispatcher integration

**Given** `data.dataset` is `"pile"`,
**When** `load_dataset(cfg, tokenizer)` is called,
**Then** it dispatches to `load_pile_data(cfg, tokenizer)`,
**And** returns the same dict format (values are `torch.Tensor` of dtype `torch.long`).

Existing `"tinyshakespeare"` and `"agnews"` paths are unchanged.

### B-8: Domain name mapping

The following domain name strings map to HuggingFace Pile subset identifiers:

| Config name | Pile subset | HuggingFace path |
|---|---|---|
| `wikipedia_en` | Wikipedia (en) | `ArmelR/the-pile-splitted` config `"Wikipedia (en)"` |
| `stackexchange` | StackExchange | `ArmelR/the-pile-splitted` config `"StackExchange"` |
| `hackernews` | HackerNews | `ArmelR/the-pile-splitted` config `"HackerNews"` |
| `arxiv` | ArXiv | `ArmelR/the-pile-splitted` config `"ArXiv"` |
| `freelaw` | FreeLaw | `ArmelR/the-pile-splitted` config `"FreeLaw"` |
| `pubmed_abstracts` | PubMed Abstracts | `ArmelR/the-pile-splitted` config `"PubMed Abstracts"` |
| `pile_cc` | Pile-CC | `ArmelR/the-pile-splitted` config `"Pile-CC"` |
| `gutenberg` | Gutenberg (PG-19) | `ArmelR/the-pile-splitted` config `"Gutenberg (PG-19)"` |

Invalid domain names raise `ValueError` with the list of valid names.

### B-9: Reproducibility

**Given** a seed in `train.seed`,
**When** loading and shuffling documents within each domain,
**Then** the shuffle order is deterministic for that seed,
**And** re-running with the same seed produces identical train/val/test splits.

### B-10: Progress reporting

**When** downloading and tokenizing a domain,
**Then** print progress: domain name, documents processed, tokens so far, target,
**And** print final summary: domain name, token count, cache path.

### B-11: Validation

**When** config validation runs (`validate_config`),
**And** `data.dataset` is `"pile"`,
**Then** it checks that `pile_id_domains` is a non-empty list of valid domain names,
**And** `pile_ood_domains` is a non-empty list of valid domain names,
**And** ID and OOD domain lists do not overlap,
**And** `pile_id_tokens > 0` and `pile_ood_tokens > 0`.

---

## 3. Data Flow

```
HuggingFace (streaming)
       │
       ▼
 download N documents from domain
       │
       ▼
 tokenize each doc with tiktoken GPT-2 BPE
       │
       ▼
 concatenate into 1D token tensor
       │
       ▼
 cache to data/pile/{domain}_{tokens}.pt
       │
       ▼
 [ID domains] ──concat──> train | val | test_id
       │
 [OOD domains] ──────────> test_ood_{name} (one per domain)
       │
       ▼
 return {"train": ..., "val": ..., "test_id": ..., "test_ood_*": ...}
```

---

## 4. Return Format

```python
{
    "train": torch.Tensor,          # dtype=long, 1D
    "val": torch.Tensor,            # dtype=long, 1D
    "test_id": torch.Tensor,        # dtype=long, 1D
    "test_ood_arxiv": torch.Tensor,     # dtype=long, 1D
    "test_ood_freelaw": torch.Tensor,   # dtype=long, 1D
    "test_ood_pubmed_abstracts": torch.Tensor,  # dtype=long, 1D
}
```

Keys for OOD are `"test_ood_" + domain_name` (from config).

The training loop uses only `"train"` and `"val"` — unchanged.
Evaluation code iterates over `test_ood_*` keys.

---

## 5. Dependency

New dependency: `datasets` (HuggingFace). Add to `pyproject.toml`:

```toml
dependencies = [
    ...
    "datasets",
]
```

---

## 6. Acceptance Criteria

1. `load_dataset(cfg, tokenizer)` with `dataset: pile` returns a dict with train/val/test_id and one `test_ood_*` key per OOD domain.
2. All returned tensors are non-empty `torch.long` 1D tensors.
3. Train + val + test_id token counts match the ID target (within rounding).
4. Each OOD tensor has at most `pile_ood_tokens` tokens.
5. Cached `.pt` files are created in `data/pile/`.
6. Second call with same config loads from cache (no network access).
7. Different `pile_id_tokens` invalidates cache.
8. Invalid domain name raises `ValueError`.
9. Overlapping ID/OOD domains raises `ValueError`.
10. Deterministic output given fixed seed.

---

## 7. Out of Scope

- **Eval function changes** (multi-OOD iteration) — separate spec, part of pipeline BDD.
- **Qualitative eval for Pile** — separate spec (need domain-aware prompt selection, not AG-News-specific).
- **LoRA three-way split** (pretrain/finetune/OOD) — will reuse the same loader with different config. The `pile_id_domains` becomes the fine-tune domain; pretrain checkpoint comes from a separate run. No special loader logic needed.
- **Document-level boundaries** — we concatenate all tokens flat (same as AG News). No special `<|endoftext|>` handling.
