#!/usr/bin/env python3
"""Extract and cross-reference citations from a markdown academic paper.

Parses:
  1. Reference entries from the ## References section
  2. In-text citations: (Author et al., YYYY), Author (YYYY), etc.
  3. Cross-reference: orphaned refs, missing refs

Usage:
    python extract_refs.py <paper.md>
    python extract_refs.py <paper.md> --json   # machine-readable output
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_references(text: str) -> list[dict]:
    """Extract structured references from ## References section."""
    ref_match = re.search(r"^##\s+References\s*$", text, re.MULTILINE)
    if not ref_match:
        return []

    ref_text = text[ref_match.end():]
    next_section = re.search(r"^## ", ref_text, re.MULTILINE)
    if next_section:
        ref_text = ref_text[: next_section.start()]

    refs = []
    for para in ref_text.strip().split("\n\n"):
        para = para.strip()
        if not para:
            continue

        year_match = re.search(r"\((\d{4})\)", para)
        year = year_match.group(1) if year_match else None

        if year_match:
            authors_str = para[: year_match.start()].strip().rstrip(",").strip()
        else:
            authors_str = ""

        if year_match:
            after_year = para[year_match.end():].strip().lstrip(".").strip()
            title_match = re.match(r"(.+?)\s*\*(.+?)\*", after_year)
            if title_match:
                title = title_match.group(1).strip().rstrip(".")
                venue = title_match.group(2).strip()
            else:
                title = after_year.rstrip(".")
                venue = None
        else:
            title = para
            venue = None

        first_author_last = authors_str.split(",")[0].strip() if authors_str else "Unknown"

        refs.append({
            "authors": authors_str,
            "year": year,
            "title": title,
            "venue": venue,
            "key": first_author_last.lower(),
            "raw": para,
        })

    return refs


def extract_citations(text: str) -> list[dict]:
    """Extract in-text citations from the body (before References section)."""
    ref_start = text.find("## References")
    body = text[:ref_start] if ref_start >= 0 else text

    citations: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # Pattern: Author et al., YYYY  |  Author & Author, YYYY  |  Author (YYYY)
    patterns = [
        r"(\w[\w\-]+)\s+et\s+al\.\s*[,;]\s*(\d{4})",
        r"(\w[\w\-]+)\s*&\s*\w[\w\-]+\s*[,;]\s*(\d{4})",
        r"(\w[\w\-]+)\s+et\s+al\.\s*\((\d{4})\)",
        r"(\w[\w\-]+)\s*&\s*\w[\w\-]+\s*\((\d{4})\)",
        r"(\w[\w\-]+)\s+\((\d{4})\)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, body):
            last_name = match.group(1)
            year = match.group(2)
            key = (last_name.lower(), year)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "text": match.group(0),
                    "last_name": last_name,
                    "year": year,
                    "key": last_name.lower(),
                })

    return citations


def cross_reference(
    refs: list[dict], citations: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Find orphaned refs and missing refs."""
    ref_keys = {(r["key"], r["year"]) for r in refs}
    cite_keys = {(c["key"], c["year"]) for c in citations}
    ref_by_key = {(r["key"], r["year"]): r for r in refs}

    orphaned_keys = ref_keys - cite_keys
    missing_keys = cite_keys - ref_keys

    orphaned = [ref_by_key[k] for k in orphaned_keys if k in ref_by_key]
    missing = [{"last_name": k[0], "year": k[1]} for k in missing_keys]
    return orphaned, missing


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(
    paper_path: Path,
    refs: list[dict],
    citations: list[dict],
    orphaned: list[dict],
    missing: list[dict],
) -> None:
    """Print human-readable report."""
    print(f"{'=' * 60}")
    print(f"Paper Reference Check: {paper_path.name}")
    print(f"{'=' * 60}")

    print(f"\n## {len(refs)} References Found\n")
    for i, r in enumerate(refs, 1):
        print(f"  {i:2d}. [{r['key'].title()} ({r['year']})]")
        print(f"      Authors: {r['authors']}")
        print(f"      Title:   {r['title']}")
        print(f"      Venue:   {r['venue'] or '(none)'}")
        print()

    print(f"## {len(citations)} Unique In-Text Citations\n")
    for c in sorted(citations, key=lambda x: (x["year"], x["last_name"])):
        print(f"  - {c['last_name']} ({c['year']})")

    print("\n## Cross-Reference\n")
    if orphaned:
        print("  ORPHANED (in References but never cited in text):")
        for r in orphaned:
            title_short = r["title"][:55] + "..." if len(r["title"]) > 55 else r["title"]
            print(f"    - {r['key'].title()} ({r['year']}): {title_short}")
    else:
        print("  No orphaned references.")

    if missing:
        print("\n  MISSING (cited in text but not in References):")
        for m in missing:
            print(f"    - {m['last_name']} ({m['year']})")
    else:
        print("  No missing references.")

    print("\n## Verification Needed\n")
    print("  For each reference above, verify against the actual paper:")
    print("  - Author names (first + last, all authors)")
    print("  - Year of publication")
    print("  - Exact title")
    print("  - Venue (conference/journal name)")
    print("  Recommended: search arXiv, Google Scholar, or Semantic Scholar.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python extract_refs.py <paper.md> [--json]", file=sys.stderr)
        sys.exit(1)

    paper_path = Path(sys.argv[1])
    if not paper_path.exists():
        print(f"ERROR: {paper_path} not found", file=sys.stderr)
        sys.exit(1)

    text = paper_path.read_text(encoding="utf-8")
    refs = extract_references(text)
    citations = extract_citations(text)
    orphaned, missing = cross_reference(refs, citations)

    if "--json" in sys.argv:
        result = {
            "paper": str(paper_path),
            "references": refs,
            "citations": citations,
            "orphaned": [{"key": r["key"], "year": r["year"], "title": r["title"]}
                         for r in orphaned],
            "missing": missing,
        }
        print(json.dumps(result, indent=2))
    else:
        print_report(paper_path, refs, citations, orphaned, missing)


if __name__ == "__main__":
    main()
