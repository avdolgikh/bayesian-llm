---
name: check-paper-refs
description: Verify academic paper references — extract citations, cross-reference, and web-verify author names, titles, years, venues against actual papers.
user-invocable: true
allowed-tools: Read, Grep, Bash, Agent, WebSearch, WebFetch
effort: high
argument-hint: <paper-path>
---

# Check Paper References

Verify all references in an academic paper (markdown or LaTeX) for correctness.

## When to Use

- Before submitting or sharing a paper draft
- After editing references or adding new citations
- When inheriting a paper with unverified references

## Contents

```
check-paper-refs/
  skill.md                    # This file
  scripts/
    extract_refs.py           # Extraction script (markdown papers)
```

## Input

Path to a paper file (markdown or LaTeX).

## Workflow

### Step 1: Mechanical extraction

Run the extraction script to get a structured report of all references and in-text citations, including cross-reference issues (orphaned refs, missing refs):

```
python <skill-dir>/scripts/extract_refs.py <paper-path>
```

Review the output. Fix any orphaned or missing references before proceeding.

### Step 2: Web verification

For **each** reference in the extraction report, search the web (arXiv, Google Scholar, Semantic Scholar, conference proceedings) and verify:

1. **Author names** — all authors, spelled correctly, in the right order
2. **Year** — matches the publication or submission year
3. **Title** — exact title (not a colloquial name or abbreviation)
4. **Venue** — correct conference/journal (ICML, NeurIPS, ICLR, etc.) and year

Recommended search queries:
- Search by title (in quotes): `"Exact Paper Title"`
- Search by arXiv ID if available: `arXiv:XXXX.XXXXX`
- Cross-check on the conference proceedings page (e.g., openreview.net for ICLR/NeurIPS)

### Step 3: Report discrepancies

For each reference, report one of:
- **OK** — all fields verified correct
- **FIX** — list the specific fields that need correction, with the correct values

Common mistakes to watch for:
- **Wrong authors** — papers from the same lab get confused; LLMs hallucinate author lists
- **Wrong title** — colloquial names used instead of official title (e.g., "LoRA meets Laplace" vs actual title)
- **Wrong venue** — preprint cited as published, or wrong conference year
- **Wrong year** — arXiv submission year vs conference publication year

### Step 4: Apply fixes

After all discrepancies are identified, update both:
- The references section (author names, title, venue)
- In-text citations if the first author changed (e.g., `Zheng et al.` -> `Shi et al.`)

## Output

A verification table:

| # | Reference | Authors | Title | Year | Venue | Status |
|---|-----------|---------|-------|------|-------|--------|
| 1 | Blundell (2015) | OK | OK | OK | OK | OK |
| 2 | Wang (2024) | FIX: should be Wang, Shi, Han, Metaxas, Wang | OK | OK | OK | FIX |
| ... | | | | | | |
