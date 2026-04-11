---
name: check-paper-refs
description: Verify academic paper references — automated ground truth check + live arXiv verification + multi-agent cross-review protocol.
user-invocable: true
allowed-tools: Read, Grep, Bash, Agent, WebSearch, WebFetch
effort: high
argument-hint: <paper-path>
---

# Check Paper References

Verify all references in an academic paper for correctness. Three layers of defense.

## When to Use

- Before submitting or sharing a paper draft
- After editing references or adding new citations
- When inheriting a paper with unverified references
- **Automatically in CI** (layer 1 runs on every push)

## Three Layers of Defense

### Layer 1: Ground Truth Check (fast, automated, CI)

Compare `references.bib` against `paper/references_ground_truth.json` — a human-verified, git-tracked file with correct author names, titles, years, and arXiv IDs.

```bash
uv run python scripts/verify_references.py           # fast, no network
```

Runs in CI on every push. Exit code 1 on any mismatch. This catches:
- Accidental edits to verified references
- New references added without ground truth entry
- LLM hallucination changing author names during "fixes"

### Layer 2: Live arXiv Verification (slow, periodic)

Verify the ground truth itself against arXiv API. Run after adding new references or periodically.

```bash
uv run python scripts/verify_references.py --live    # ~3s per reference
```

This catches errors in the ground truth file itself.

### Layer 3: Multi-Agent Cross-Review (manual, before submission)

Spawn **two independent agents** that each verify all references from scratch. Neither sees the other's work.

```
Agent 1: "Verify all references in paper/references.bib against arXiv.
          For each entry, search arXiv by title and report exact author names."

Agent 2: "Independently verify all references in paper/references.bib.
          For each entry, fetch the arXiv page and compare authors."
```

Compare results. If agents disagree on any reference → flag for manual verification.

**Why two agents:** A single LLM can hallucinate that wrong names are correct (this happened — "Yundong Wang" was hallucinated as correct when the real author is "Yibin Wang"). Two independent verifications with different prompts make correlated hallucination unlikely.

## Adding a New Reference

1. Add the entry to `paper/references.bib`
2. Search arXiv for the paper, verify ALL author names against the actual PDF
3. Add the verified entry to `paper/references_ground_truth.json` with `arxiv_id`
4. Run `uv run python scripts/verify_references.py` — must pass
5. Run `uv run python scripts/verify_references.py --live` — must pass

## Ground Truth File Format

```json
{
  "entries": {
    "bibtex_key": {
      "authors": ["Last, First", "Last, First"],
      "title": "Full Paper Title",
      "year": "2024",
      "arxiv_id": "2406.11675"
    }
  }
}
```

## Output

Layer 1 output (automated):
```
Checking 11 references against ground truth...
OK: All references verified.
```

Layer 2 output (live):
```
Live-verifying 10 references against arXiv API (~30s)...
  [blundell2015weight] arXiv:1505.05424 ... OK
  [wang2024blob] arXiv:2406.11675 ... MISMATCH
```
