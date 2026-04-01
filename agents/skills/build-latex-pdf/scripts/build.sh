#!/usr/bin/env bash
# Build LaTeX paper to PDF using Docker (TeX Live).
# Self-contained — this skill directory ships the .sty and everything needed.
#
# Usage:
#   bash <skill-dir>/scripts/build.sh <input.tex> [output.pdf]
#
# The project root is auto-detected (git root, or parent of tex dir).
# This is mounted into Docker so relative paths (e.g. ../figures/) resolve.
# The bundled neurips_2024.sty is injected via TEXINPUTS.
#
# Requirements: Docker (texlive/texlive:latest, auto-pulled on first run).

set -euo pipefail

# Skill root is one level up from scripts/
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEX_FILE="${1:?Usage: build.sh <input.tex> [output.pdf]}"
OUTPUT="${2:-}"

# Resolve to absolute path
if [[ "$TEX_FILE" != /* ]]; then
    TEX_FILE="$(pwd)/$TEX_FILE"
fi
TEX_FILE="$(cd "$(dirname "$TEX_FILE")" && pwd)/$(basename "$TEX_FILE")"

TEX_DIR="$(dirname "$TEX_FILE")"
TEX_NAME="$(basename "$TEX_FILE" .tex)"

if [ ! -f "$TEX_FILE" ]; then
    echo "ERROR: $TEX_FILE not found" >&2
    exit 1
fi

# Auto-detect project root: git root if available, else parent of tex dir.
# Mounting the project root lets relative paths like ../figures/ resolve.
# Normalize via cd+pwd so both paths use the same format (POSIX on Git Bash).
GIT_ROOT="$(cd "$TEX_DIR" && git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$GIT_ROOT" ] && [ -d "$GIT_ROOT" ]; then
    PROJECT_ROOT="$(cd "$GIT_ROOT" && pwd)"
else
    PROJECT_ROOT="$(cd "$TEX_DIR/.." && pwd)"
fi
REL_TEX_DIR="${TEX_DIR#"$PROJECT_ROOT"/}"

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is required but not found." >&2
    echo "Install Docker Desktop: https://www.docker.com/products/docker-desktop" >&2
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Start Docker Desktop first." >&2
    exit 1
fi

IMAGE="texlive/texlive:latest"

if ! docker image inspect "$IMAGE" &>/dev/null 2>&1; then
    echo "Pulling $IMAGE (first time only, ~5 GB)..."
    docker pull "$IMAGE"
fi

echo "Compiling $(basename "$TEX_FILE")..."
echo "  project root: $PROJECT_ROOT"
echo "  tex dir:      $REL_TEX_DIR/"

# Mount project root as /repo, skill dir as /skill (read-only).
# TEXINPUTS="/skill//:" -> LaTeX searches /skill/ for .sty files first,
# then falls back to default TeX Live paths.
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$PROJECT_ROOT:/repo" \
    -v "$SKILL_DIR:/skill:ro" \
    -e "TEXINPUTS=/skill//:" \
    -w "/repo/$REL_TEX_DIR" \
    "$IMAGE" \
    bash -c "
        pdflatex -interaction=nonstopmode $TEX_NAME && \
        bibtex $TEX_NAME 2>&1 || true && \
        pdflatex -interaction=nonstopmode $TEX_NAME && \
        pdflatex -interaction=nonstopmode $TEX_NAME && \
        echo '---BUILD OK---'
    " 2>&1 | tail -5

# Check output
PDF_PATH="$TEX_DIR/$TEX_NAME.pdf"
if [ ! -f "$PDF_PATH" ]; then
    echo "ERROR: PDF was not generated." >&2
    exit 1
fi

# Move to custom output path if specified
if [ -n "$OUTPUT" ]; then
    if [[ "$OUTPUT" != /* ]]; then
        OUTPUT="$(pwd)/$OUTPUT"
    fi
    cp "$PDF_PATH" "$OUTPUT"
    PDF_PATH="$OUTPUT"
fi

# Clean intermediate files
rm -f "$TEX_DIR/$TEX_NAME.aux" "$TEX_DIR/$TEX_NAME.bbl" "$TEX_DIR/$TEX_NAME.blg" \
      "$TEX_DIR/$TEX_NAME.log" "$TEX_DIR/$TEX_NAME.out" 2>/dev/null

SIZE=$(du -h "$PDF_PATH" | cut -f1)
echo "Done: $PDF_PATH ($SIZE)"
