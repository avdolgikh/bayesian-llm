#!/bin/bash
# Build PDF from markdown with Mermaid diagrams + MathJax formulas
# Usage: bash agents/skills/convert-md-to-pdf/scripts/build-pdf.sh docs/paper.md paper.pdf
#
# Pipeline: mmdc (Mermaid->SVG) -> md-to-pdf (MathJax + Puppeteer -> PDF)

set -e

INPUT="${1:?Usage: build-pdf.sh input.md output.pdf}"
OUTPUT="${2:?Usage: build-pdf.sh input.md output.pdf}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/md-to-pdf.config.js"
BASENAME=$(basename "$INPUT" .md)
MERMAID_OUT="${BASENAME}-mermaid.md"

echo "==> Step 1: Mermaid diagrams -> SVG"
mmdc -i "$INPUT" -o "$MERMAID_OUT"

echo "==> Step 2: Markdown + MathJax -> PDF"
md-to-pdf --config-file "$CONFIG_FILE" "$MERMAID_OUT"

MERMAID_PDF="${BASENAME}-mermaid.pdf"
if [ -f "$MERMAID_PDF" ]; then
    mv "$MERMAID_PDF" "$OUTPUT"
    echo "==> Done: $OUTPUT"
else
    echo "==> ERROR: PDF not generated"
    exit 1
fi

rm -f "$MERMAID_OUT"
