#!/usr/bin/env bash
# Build arXiv submission zip from LaTeX source.
# Self-contained — auto-detects figures, bib, and sty dependencies.
#
# Usage:
#   bash <skill-dir>/scripts/build.sh <input.tex> [output.zip]
#
# The script parses \graphicspath, \bibliography, and \usepackage from the tex
# file to discover all dependencies. Everything is staged in a temp directory,
# \graphicspath is rewritten to {figures/}, and the result is zipped.
#
# If Docker is available, the staged source is compiled to verify correctness.
#
# Requirements: PowerShell (Windows) or zip (Linux/macOS). Docker optional.

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

TEX_FILE="${1:?Usage: build.sh <input.tex> [output.zip]}"
OUTPUT="${2:-}"

# Resolve to absolute path (handle both /unix and D:/windows forms)
if [[ "$TEX_FILE" != /* && "$TEX_FILE" != ?:/* && "$TEX_FILE" != ?:\\* ]]; then
    TEX_FILE="$(pwd)/$TEX_FILE"
fi
TEX_FILE="$(cd "$(dirname "$TEX_FILE")" && pwd)/$(basename "$TEX_FILE")"

TEX_DIR="$(dirname "$TEX_FILE")"
TEX_NAME="$(basename "$TEX_FILE" .tex)"

if [ ! -f "$TEX_FILE" ]; then
    echo "ERROR: $TEX_FILE not found" >&2
    exit 1
fi

# Auto-detect project root (git root, or parent of tex dir)
GIT_ROOT="$(cd "$TEX_DIR" && git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$GIT_ROOT" ] && [ -d "$GIT_ROOT" ]; then
    PROJECT_ROOT="$(cd "$GIT_ROOT" && pwd)"
else
    PROJECT_ROOT="$(cd "$TEX_DIR/.." && pwd)"
fi

# Default output
if [ -z "$OUTPUT" ]; then
    OUTPUT="$TEX_DIR/arxiv-submission.zip"
fi
if [[ "$OUTPUT" != /* && "$OUTPUT" != ?:/* && "$OUTPUT" != ?:\\* ]]; then
    OUTPUT="$(pwd)/$OUTPUT"
fi

# ── Parse tex file ────────────────────────────────────────────────────────────

echo "Parsing $TEX_FILE..."

# Extract \graphicspath — e.g. \graphicspath{{../figures/}}
GRAPHICS_PATH=$(grep -oP '\\graphicspath\{\{([^}]+)\}\}' "$TEX_FILE" | sed 's/\\graphicspath{{//;s/}}//' || true)
if [ -n "$GRAPHICS_PATH" ]; then
    # Resolve relative to tex dir
    FIGURES_DIR="$(cd "$TEX_DIR/$GRAPHICS_PATH" 2>/dev/null && pwd || true)"
fi

if [ -z "${FIGURES_DIR:-}" ] || [ ! -d "$FIGURES_DIR" ]; then
    echo "WARNING: Could not resolve graphicspath '$GRAPHICS_PATH'. Checking for figures/ in project root..."
    if [ -d "$PROJECT_ROOT/figures" ]; then
        FIGURES_DIR="$PROJECT_ROOT/figures"
    else
        echo "WARNING: No figures directory found. Proceeding without figures."
        FIGURES_DIR=""
    fi
fi

# Extract \bibliography{name} — e.g. \bibliography{references}
BIB_NAME=$(grep -oP '\\bibliography\{([^}]+)\}' "$TEX_FILE" | sed 's/\\bibliography{//;s/}//' || true)
BIB_FILE=""
if [ -n "$BIB_NAME" ]; then
    # Try tex dir first, then project root
    if [ -f "$TEX_DIR/$BIB_NAME.bib" ]; then
        BIB_FILE="$TEX_DIR/$BIB_NAME.bib"
    elif [ -f "$PROJECT_ROOT/$BIB_NAME.bib" ]; then
        BIB_FILE="$PROJECT_ROOT/$BIB_NAME.bib"
    fi
fi

# Extract \usepackage{name} and find .sty files in tex dir or project
STY_FILES=()
while IFS= read -r pkg; do
    pkg=$(echo "$pkg" | sed 's/\\usepackage\(\[[^]]*\]\)\?{//;s/}//')
    # Search for .sty in tex dir, project root, and common locations
    for search_dir in "$TEX_DIR" "$PROJECT_ROOT"; do
        found=$(find "$search_dir" -name "${pkg}.sty" -not -path "*/.git/*" 2>/dev/null | head -1 || true)
        if [ -n "$found" ]; then
            STY_FILES+=("$found")
            break
        fi
    done
done < <(grep -oP '\\usepackage(\[[^]]*\])?\{[^}]+\}' "$TEX_FILE" || true)

# ── Stage files ───────────────────────────────────────────────────────────────

# Stage under TEX_DIR so Docker on Windows can mount it (POSIX /tmp doesn't map)
STAGING="$TEX_DIR/.arxiv-staging-$$"
mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT

echo "Staging submission..."

# Copy tex
cp "$TEX_FILE" "$STAGING/"

# Copy bib
if [ -n "$BIB_FILE" ] && [ -f "$BIB_FILE" ]; then
    cp "$BIB_FILE" "$STAGING/"
    echo "  bib: $(basename "$BIB_FILE")"
fi

# Copy sty files
for sty in "${STY_FILES[@]}"; do
    cp "$sty" "$STAGING/"
    echo "  sty: $(basename "$sty")"
done

# Copy figures
FIG_COUNT=0
if [ -n "$FIGURES_DIR" ] && [ -d "$FIGURES_DIR" ]; then
    mkdir -p "$STAGING/figures"
    for fig in "$FIGURES_DIR"/*; do
        if [ -f "$fig" ]; then
            cp "$fig" "$STAGING/figures/"
            FIG_COUNT=$((FIG_COUNT + 1))
        fi
    done
    echo "  figures: $FIG_COUNT files"
fi

# Rewrite \graphicspath to flat zip structure
if [ -n "$GRAPHICS_PATH" ]; then
    sed -i "s|\\\\graphicspath{{[^}]*}}|\\\\graphicspath{{figures/}}|" "$STAGING/$TEX_NAME.tex"
    echo "  graphicspath: rewritten to {figures/}"
fi

# ── Verify compilation (optional) ────────────────────────────────────────────

if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    echo ""
    echo "Verifying compilation via Docker..."
    IMAGE="texlive/texlive:latest"

    # Pull if needed (silently)
    if ! docker image inspect "$IMAGE" &>/dev/null 2>&1; then
        echo "  Pulling $IMAGE (first time, ~5 GB)..."
        docker pull "$IMAGE"
    fi

    COMPILE_OUTPUT=$(MSYS_NO_PATHCONV=1 docker run --rm \
        -v "$STAGING:/work" \
        -w /work \
        "$IMAGE" \
        bash -c "
            pdflatex -interaction=nonstopmode $TEX_NAME && \
            bibtex $TEX_NAME 2>&1 || true && \
            pdflatex -interaction=nonstopmode $TEX_NAME && \
            pdflatex -interaction=nonstopmode $TEX_NAME
        " 2>&1) || true

    # Check if build succeeded
    if [ -f "$STAGING/$TEX_NAME.pdf" ]; then
        PAGES=$(echo "$COMPILE_OUTPUT" | grep -oP 'Output written on .* \(\K[0-9]+' | tail -1 || echo "?")
        echo "  Compilation OK ($PAGES pages)"
    else
        echo "  WARNING: Compilation failed. Check your LaTeX source."
        echo "$COMPILE_OUTPUT" | tail -10
    fi

    # Remove build artifacts from staging (keep only source)
    rm -f "$STAGING/$TEX_NAME.pdf" "$STAGING/$TEX_NAME.aux" "$STAGING/$TEX_NAME.bbl" \
          "$STAGING/$TEX_NAME.blg" "$STAGING/$TEX_NAME.log" "$STAGING/$TEX_NAME.out" \
          "$STAGING/texput.log" 2>/dev/null
else
    echo ""
    echo "Docker not available — skipping compilation check."
    echo "  Verify manually before uploading to arXiv."
fi

# ── Build zip ─────────────────────────────────────────────────────────────────

echo ""
echo "Building zip..."

# Remove old output if it exists
rm -f "$OUTPUT" 2>/dev/null

if command -v zip &>/dev/null; then
    (cd "$STAGING" && zip -r "$OUTPUT" .)
elif command -v powershell.exe &>/dev/null; then
    # Convert POSIX paths to Windows for PowerShell (e.g. /d/foo -> D:\foo)
    WIN_STAGING=$(cygpath -w "$STAGING" 2>/dev/null || echo "$STAGING")
    WIN_OUTPUT=$(cygpath -w "$OUTPUT" 2>/dev/null || echo "$OUTPUT")
    powershell.exe -Command "Compress-Archive -Path '$WIN_STAGING\\*' -DestinationPath '$WIN_OUTPUT' -Force"
else
    echo "ERROR: Neither zip nor powershell.exe available" >&2
    exit 1
fi

if [ ! -f "$OUTPUT" ]; then
    echo "ERROR: Failed to create zip" >&2
    exit 1
fi

SIZE=$(du -h "$OUTPUT" | cut -f1)
FILE_COUNT=$(find "$STAGING" -type f | wc -l)

echo ""
echo "Done: $OUTPUT"
echo "  $FILE_COUNT files, $SIZE (arXiv limit: 50 MB)"
