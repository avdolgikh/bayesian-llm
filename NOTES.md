# Notes

- 2026-02-01: Initialized repo skeleton, selected PyTorch for early milestones, and added A0 mini-GPT scaffold.
- 2026-02-01: Added uv-only workflow (uv add/sync/run), created uv.lock, and noted UV_CACHE_DIR workaround for permission errors.
- 2026-02-01: Ran A0 baseline successfully after adding NumPy and ensuring corpus length exceeds block size.
- 2026-02-21: Phase 1 restructuring — flattened `src/bayesian_llm/` into `minigpt/` package. Replaced char tokenizer with BPE (tiktoken/GPT-2). Replaced hardcoded corpus with TinyShakespeare auto-download. Added MLflow experiment tracking (sqlite backend). Added perplexity reporting, checkpoint saving, standalone evaluate module. Model: 4-layer, 128-dim, 13.7M params. Smoke test passed (50 steps: loss 10.66 → 7.96).
