# Skill: Convert Markdown to PDF

Convert a markdown document with LaTeX formulas, tables, and images to a publication-quality PDF.

## When to Use

- Before submitting a paper to arXiv
- When generating a PDF for review or sharing
- After editing `docs/paper.md` or any markdown document with math/figures

## Prerequisites

**Runtime:**
- **Node.js** >= 18
- **npm** (comes with Node.js)

**Global npm packages** (install once):
```
npm install -g md-to-pdf @mermaid-js/mermaid-cli
```

| Package | What it does |
|---|---|
| `md-to-pdf` | Markdown -> PDF via Puppeteer (headless Chrome). Handles HTML/CSS, script injection, PDF printing |
| `@mermaid-js/mermaid-cli` | Provides `mmdc` command. Renders mermaid code blocks to SVG images |

MathJax 3 is loaded from CDN at generation time (requires internet).

## Input

Two arguments:
1. Path to input markdown file (e.g., `docs/paper.md`)
2. Path to output PDF file (e.g., `paper.pdf`)

## Workflow

### Step 1: Check prerequisites

Verify Node.js, mmdc, and md-to-pdf are installed:

```bash
node --version && mmdc --version && md-to-pdf --version
```

If any are missing, instruct the user to install them.

### Step 2: Run the build script

**Windows (PowerShell):**
```powershell
powershell agents/skills/convert-md-to-pdf/scripts/build-pdf.ps1 <input.md> <output.pdf>
```

**Linux/macOS (Bash):**
```bash
bash agents/skills/convert-md-to-pdf/scripts/build-pdf.sh <input.md> <output.pdf>
```

### Step 3: Verify output

Check the output PDF exists and report file size.

## How the Pipeline Works

**Step 1: Mermaid -> SVG** (`mmdc`)
- Scans markdown for ` ```mermaid ` code blocks
- Renders each diagram to an `.svg` file using headless Chrome
- Replaces the code block with `![diagram](./file.svg)` in the output
- Text, tables, and math pass through unchanged

**Step 2: Markdown + MathJax -> PDF** (`md-to-pdf` with `md-to-pdf.config.js`)

Three things happen internally:

1. **Markdown -> HTML** (via `marked` parser)
   - Converts markdown to HTML
   - **Problem:** `marked` treats `_` as emphasis, breaking LaTeX subscripts in `$$...$$` blocks
   - **Fix:** `md-to-pdf.config.js` adds a `marked_extensions` tokenizer that intercepts `$$...$$` blocks before `marked`'s default processing — no underscore mangling

2. **Inject scripts + styles** into the HTML
   - Injects MathJax 3 config (inline `<script>`) + CDN (`<script src="...">`)
   - Injects custom CSS for tables, blockquotes, images, MathJax spacing

3. **HTML -> PDF** (via Puppeteer / headless Chrome)
   - Chrome executes MathJax JS -> renders all `$$...$$` and `$...$` as math
   - Puppeteer waits for `networkidle0` (MathJax CDN loaded and rendered)
   - Prints to PDF (A4, margins, `printBackground: true`)

**Why Mermaid first:** if MathJax ran first, `$` signs inside Mermaid labels would be interpreted as math.

## Three Problems Solved by md-to-pdf.config.js

| Problem | Root cause | Solution |
|---|---|---|
| LaTeX `_` subscripts broken | `marked` treats `_` as emphasis | `marked_extensions` tokenizer intercepts `$$` blocks as raw HTML passthrough |
| Math formulas not rendered | `md-to-pdf` doesn't know about LaTeX | `script` injects MathJax 3 config + CDN; Puppeteer executes it before printing |
| Tables/images unstyled | `md-to-pdf` default CSS is minimal | `css` adds borders, headers, striped rows, centered images, blockquotes |

## Project Files

| File | Role |
|---|---|
| `agents/skills/convert-md-to-pdf/scripts/build-pdf.ps1` | PowerShell orchestrator (Windows) |
| `agents/skills/convert-md-to-pdf/scripts/build-pdf.sh` | Bash orchestrator (Linux/macOS) |
| `agents/skills/convert-md-to-pdf/scripts/md-to-pdf.config.js` | All-in-one config: MathJax injection, math protection, CSS, PDF options |

## Figures

The paper references figures as `![caption](../figures/filename.pdf)`. For md-to-pdf (which uses HTML rendering), PNG figures may work better than PDF. Generate PNG versions first:

```bash
uv run python scripts/generate_figures.py --png
```

Then ensure the image paths in the markdown resolve correctly relative to where the build script runs. If paths break, temporarily adjust them or copy figures alongside the markdown.

## Troubleshooting

- **`mmdc` not found:** `npm install -g @mermaid-js/mermaid-cli`
- **`md-to-pdf` not found:** `npm install -g md-to-pdf`
- **MathJax not rendering:** Check internet connection (CDN). Look for `data-mathjax-done` attribute in the intermediate HTML.
- **Images not showing:** Check relative paths. `md-to-pdf` resolves images relative to the input markdown file's directory.
- **Unicode errors on Windows:** Set `PYTHONIOENCODING=utf-8` or pipe through `chcp 65001`.
