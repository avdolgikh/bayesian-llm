#!/usr/bin/env python
"""Verify paper references against ground truth and/or arXiv API.

Two modes:
  --check (default)  Compare references.bib against ground_truth.json (fast, no network)
  --live             Verify ground_truth.json itself against arXiv API (slow, ~3s/ref)

Exit codes:  0 = all OK,  1 = mismatches found.

Usage:
    uv run python scripts/verify_references.py              # fast local check
    uv run python scripts/verify_references.py --live       # live arXiv verification
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"
BIB_PATH = PAPER_DIR / "references.bib"
GROUND_TRUTH_PATH = PAPER_DIR / "references_ground_truth.json"

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_DELAY = 3.1  # seconds between requests (arXiv rate limit)


# ---------------------------------------------------------------------------
# BibTeX parsing
# ---------------------------------------------------------------------------

def parse_bibtex(path: Path) -> dict[str, dict[str, str]]:
    """Parse BibTeX file into {key: {field: value}}."""
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"@\w+\{(\w+),\s*(.*?)\n\}", text, re.DOTALL):
        key = m.group(1)
        body = m.group(2)
        fields: dict[str, str] = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}", body, re.DOTALL):
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2).strip())
        entries[key] = fields
    return entries


def parse_bib_authors(author_str: str) -> list[str]:
    """Split BibTeX 'and'-delimited author string into list of names."""
    return [a.strip().replace("{", "").replace("}", "") for a in author_str.split(" and ")]


# ---------------------------------------------------------------------------
# Name comparison
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return name.replace(".", "").replace(",", " ").replace("-", " ").lower().split()


def names_match(a: str, b: str) -> bool:
    """Fuzzy-match two author names (handles Last,First vs First Last)."""
    pa, pb = _norm(a), _norm(b)
    if pa == pb:
        return True
    # Compare as sets of tokens (order-insensitive)
    sa, sb = set(pa), set(pb)
    if sa == sb:
        return True
    # Last-name + first-initial match
    if len(pa) >= 2 and len(pb) >= 2:
        # Try both orderings
        def last_first(parts):
            # "last first" or "first last" — extract last name (longest token? or first/last)
            return parts
        # If last names match and first initials match
        if pa[-1] == pb[-1] and pa[0][0] == pb[0][0]:
            return True
        if pa[0] == pb[0] and pa[-1][0] == pb[-1][0]:
            return True
        # Reversed order
        if pa[-1] == pb[0] and pa[0][0] == pb[-1][0]:
            return True
        if pa[0] == pb[-1] and pa[-1][0] == pb[0][0]:
            return True
    return False


# ---------------------------------------------------------------------------
# Ground truth check (fast, no network)
# ---------------------------------------------------------------------------

def check_bib_vs_ground_truth(bib: dict, gt: dict) -> list[tuple[str, str]]:
    """Compare BibTeX entries against ground truth. Return list of (key, message)."""
    issues: list[tuple[str, str]] = []
    gt_entries = gt.get("entries", {})

    for key in bib:
        if key not in gt_entries:
            issues.append((key, "NOT IN GROUND TRUTH — add entry or run --live"))

    for key, gt_data in gt_entries.items():
        if key not in bib:
            issues.append((key, "In ground truth but MISSING from references.bib"))
            continue

        bib_authors = parse_bib_authors(bib[key].get("author", ""))
        gt_authors = gt_data.get("authors", [])

        if len(bib_authors) != len(gt_authors):
            issues.append((key,
                f"Author count: bib={len(bib_authors)} vs truth={len(gt_authors)}\n"
                f"    BIB:   {bib_authors}\n"
                f"    TRUTH: {gt_authors}"))
            continue

        bad = []
        for i, (ba, ga) in enumerate(zip(bib_authors, gt_authors)):
            if not names_match(ba, ga):
                bad.append(f"    [{i+1}] bib='{ba}'  truth='{ga}'")
        if bad:
            issues.append((key, "Author name mismatch:\n" + "\n".join(bad)))

    return issues


# ---------------------------------------------------------------------------
# Live arXiv verification (slow, needs network)
# ---------------------------------------------------------------------------

def fetch_arxiv(arxiv_id: str) -> dict | None:
    """Fetch paper metadata from arXiv API. Returns {title, authors}."""
    url = f"{ARXIV_API}?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "verify-refs/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read()
    except Exception as e:
        print(f"    WARNING: fetch failed — {e}")
        return None

    root = ET.fromstring(xml_data)
    entries = root.findall("atom:entry", ARXIV_NS)
    if not entries:
        return None

    entry = entries[0]
    # Check for error (arXiv returns an entry with id but no title for bad IDs)
    title_el = entry.find("atom:title", ARXIV_NS)
    if title_el is None or not title_el.text:
        return None

    authors = []
    for author_el in entry.findall("atom:author", ARXIV_NS):
        name_el = author_el.find("atom:name", ARXIV_NS)
        if name_el is not None and name_el.text:
            # arXiv: "First Last" → convert to "Last, First"
            parts = name_el.text.strip().rsplit(" ", 1)
            if len(parts) == 2:
                authors.append(f"{parts[1]}, {parts[0]}")
            else:
                authors.append(name_el.text.strip())

    return {
        "title": re.sub(r"\s+", " ", title_el.text.strip()),
        "authors": authors,
    }


def verify_ground_truth_vs_arxiv(gt: dict) -> list[tuple[str, str]]:
    """Verify ground truth authors against arXiv API."""
    issues: list[tuple[str, str]] = []
    gt_entries = gt.get("entries", {})

    for key, gt_data in gt_entries.items():
        arxiv_id = gt_data.get("arxiv_id")
        if not arxiv_id:
            print(f"  [{key}] SKIP (no arxiv_id)")
            continue

        print(f"  [{key}] arXiv:{arxiv_id} ...", end=" ", flush=True)
        data = fetch_arxiv(arxiv_id)
        if data is None:
            issues.append((key, "arXiv fetch failed"))
            continue

        gt_authors = gt_data.get("authors", [])
        ax_authors = data.get("authors", [])

        if len(gt_authors) != len(ax_authors):
            print("MISMATCH")
            issues.append((key,
                f"Author count: truth={len(gt_authors)} vs arXiv={len(ax_authors)}\n"
                f"    TRUTH: {gt_authors}\n"
                f"    ARXIV: {ax_authors}"))
        else:
            bad = []
            for i, (ga, aa) in enumerate(zip(gt_authors, ax_authors)):
                if not names_match(ga, aa):
                    bad.append(f"    [{i+1}] truth='{ga}'  arXiv='{aa}'")
            if bad:
                print("MISMATCH")
                issues.append((key, "Author name mismatch:\n" + "\n".join(bad)))
            else:
                print("OK")

        time.sleep(ARXIV_DELAY)

    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify paper references")
    parser.add_argument("--live", action="store_true",
                        help="Verify ground truth against arXiv API (slow, needs network)")
    parser.add_argument("--bib", type=Path, default=BIB_PATH,
                        help=f"Path to .bib file (default: {BIB_PATH})")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH,
                        help=f"Path to ground truth JSON (default: {GROUND_TRUTH_PATH})")
    args = parser.parse_args()

    if args.live:
        # Mode: verify ground truth against arXiv
        if not args.ground_truth.exists():
            print(f"ERROR: {args.ground_truth} not found")
            return 1
        with open(args.ground_truth, encoding="utf-8") as f:
            gt = json.load(f)
        n = sum(1 for v in gt.get("entries", {}).values() if v.get("arxiv_id"))
        print(f"Live-verifying {n} references against arXiv API (~{n * 3}s)...\n")
        issues = verify_ground_truth_vs_arxiv(gt)
    else:
        # Mode: check bib against ground truth (fast)
        if not args.ground_truth.exists():
            print(f"ERROR: {args.ground_truth} not found")
            print("Create it manually or copy from a verified source.")
            return 1
        bib = parse_bibtex(args.bib)
        with open(args.ground_truth, encoding="utf-8") as f:
            gt = json.load(f)
        print(f"Checking {len(bib)} references against ground truth...")
        issues = check_bib_vs_ground_truth(bib, gt)

    if issues:
        print(f"\n{'=' * 60}")
        print(f"FAILED: {len(issues)} issue(s)\n")
        for key, detail in issues:
            print(f"  [{key}] {detail}\n")
        print(f"{'=' * 60}")
        return 1

    print("\nOK: All references verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
