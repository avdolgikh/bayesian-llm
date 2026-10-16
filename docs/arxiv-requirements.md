# arXiv Submission Requirements

Official guide: https://info.arxiv.org/help/submit/index.html

## Format
- **LaTeX source**, not the PDF. A PDF made by TeX is rejected at the Process step ("we insist on submission of the TeX source"). No specific template required: any LaTeX class works (article, NeurIPS, ICML, custom).
- arXiv compiles LaTeX on its servers with pdflatex. It can run BibTeX, but it uses the `.bbl` when one is uploaded (same base name as the `.tex`), so the package ships `paper.bbl` next to `paper.tex`.
- Put `\pdfoutput=1` in the first lines of the `.tex` when the figures are PNG, JPG or PDF.
- Zip entry names must use forward slashes (`figures/fig1.png`). A zip made with PowerShell `Compress-Archive` uses backslashes and breaks figure paths on arXiv.
- Every style file in the zip must be the real file (an empty `.sty` fails the compile).

## Figures
- `.pdf`, `.jpg`, or `.png` when using pdflatex.
- Vector formats (`.pdf`) preferred for diagrams and charts.
- Raster (`.png`, `.jpg`) acceptable for photos/screenshots.

## Size
- Recommended: <10 MB compressed.
- Hard limit: 50 MB.
- No page limit.

## Metadata
- Title, authors, abstract.
- Primary category: one the submitter is endorsed for (endorsement is per category). Cross-lists are optional and may need their own endorsement.

## Endorsement
- First-time submitters to a category may need endorsement from an established arXiv author in that category.
- Check account status before submission day.

## Licensing
- Must grant arXiv an irrevocable distribution license.
- Common choice: CC BY 4.0 or arXiv perpetual non-exclusive license.

## Submission Schedule
- Papers announced Sunday–Thursday (no Friday/Saturday announcements).
- Cut-off 14:00 US Eastern on weekdays. A paper submitted before the cut-off, Monday to Thursday, is announced at 20:00 ET that day. Thursday 14:00 to Friday 14:00 goes out Sunday 20:00; Friday 14:00 to Monday 14:00 goes out Monday 20:00.

## Our Paper Specifics
- Source: `paper/paper.tex` (the build reads only this file); figures from `scripts/generate_figures.py --png`.
- Package: `paper/arxiv-submission.zip` = `paper.tex`, `paper.bbl`, `references.bib`, `neurips_2024.sty`, `figures/` (only the figures the paper includes). The build script compiles the extracted zip in Docker without BibTeX, the way arXiv does, and fails if the log shows an error.
- Categories (v1, kept by the v2 replacement): primary `cs.AI`, cross-list `cs.LG`.
