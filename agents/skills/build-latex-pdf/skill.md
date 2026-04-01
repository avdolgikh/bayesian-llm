---
name: build-latex-pdf
description: Compile LaTeX paper to professional PDF using Docker + TeX Live (pdflatex, NeurIPS format with Times fonts, booktabs, natbib). Self-contained skill — portable across projects.
user-invocable: true
allowed-tools: Read, Bash
effort: medium
argument-hint: <input.tex> [output.pdf]
---

# Build LaTeX PDF

Compile a LaTeX paper to professional PDF using Docker + TeX Live + pdflatex. Produces NeurIPS-formatted output with Times fonts, booktabs tables, and natbib bibliography.

**Portable:** Copy this entire directory to any project. All dependencies (style files, build scripts) are self-contained. Only your content (`.tex`, `.bib`, figures) is external.

## Contents

```
build-latex-pdf/
  skill.md                    # This file
  scripts/
    build.sh                  # Bash build script
    build.ps1                 # PowerShell build script
  references/
    neurips_2024.sty          # NeurIPS 2024 style file (bundled)
```

## Prerequisites

- **Docker Desktop** installed and running
- `texlive/texlive:latest` image (auto-pulled on first run, ~5 GB)

## Usage

```bash
# Bash (Git Bash / Linux / macOS)
bash <skill-dir>/scripts/build.sh <input.tex>

# With custom output path
bash <skill-dir>/scripts/build.sh <input.tex> <output.pdf>

# PowerShell
powershell <skill-dir>/scripts/build.ps1 -TexFile <input.tex>
```

Replace `<skill-dir>` with the path to this skill directory in your project.

## How it works

1. Auto-detects the project root (git root, or parent of tex dir)
2. Mounts the project root into Docker as `/repo` (so relative paths like `../figures/` resolve)
3. Mounts this skill directory as `/skill` (read-only)
4. Sets `TEXINPUTS=/skill//:` so LaTeX finds the bundled `.sty` without copying
5. Runs the standard `pdflatex -> bibtex -> pdflatex -> pdflatex` recipe
6. Cleans intermediate files (`.aux`, `.bbl`, `.blg`, `.log`, `.out`)

## Your .tex file needs

- `\usepackage[preprint]{neurips_2024}` — the style is found automatically via TEXINPUTS
- `\bibliographystyle{plainnat}` + `\bibliography{references}` — BibTeX, with a `references.bib` in the same directory
- `\graphicspath{{../figures/}}` or wherever your figures live (relative to the tex dir)

## Tectonic fallback (no Docker)

If Docker is unavailable, [tectonic](https://github.com/tectonic-typesetting/tectonic/releases) (~20 MB binary) can compile the paper. Copy `neurips_2024.sty` to the tex directory first:

```bash
cp <skill-dir>/references/neurips_2024.sty <tex-dir>/
cd <tex-dir> && tectonic paper.tex
```

Note: tectonic uses XeTeX (Latin Modern fonts), not pdflatex (Times). Output will look slightly different.
