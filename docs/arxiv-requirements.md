# arXiv Submission Requirements

Official guide: https://info.arxiv.org/help/submit/index.html

## Format
- **LaTeX** (preferred) or PDF. No specific template required — any LaTeX class works (article, NeurIPS, ICML, custom).
- arXiv compiles LaTeX on their servers — submit source, not just PDF.

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
- Category: `cs.LG` (Machine Learning) and/or `stat.ML` (Statistics — Machine Learning).
- Optional cross-list to `cs.CL` (Computation and Language).

## Endorsement
- First-time submitters to a category may need endorsement from an established arXiv author in that category.
- Check account status before submission day.

## Licensing
- Must grant arXiv an irrevocable distribution license.
- Common choice: CC BY 4.0 or arXiv perpetual non-exclusive license.

## Submission Schedule
- Papers announced Sunday–Thursday (no Friday/Saturday announcements).
- Submission deadline: 14:00 US Eastern Time for next-day announcement.

## Our Paper Specifics
- Source: `docs/paper.md` → convert to LaTeX.
- Figures: generate as `.pdf` (vector) for diagrams, `.png` for raster plots.
- Target categories: `cs.LG`, cross-list `stat.ML`.
