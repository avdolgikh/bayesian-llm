# arXiv Submission Spec

## Submission Package

Flat zip archive containing:

```
paper.tex              # main source (modified — see below)
references.bib         # bibliography
neurips_2024.sty       # style file (copied from agents/skills/build-latex-pdf/references/)
figures/               # directory with all 7 PNGs
  fig1_point_vs_bayesian.png
  fig2_method_matrix.png
  fig3_blob_lora.png
  fig4_tfb_vs_laplace.png
  fig5_auroc_bars.png
  fig6_n_vs_auroc.png
  fig7_scaling_inversion.png
```

### Required `paper.tex` changes

1. **`\graphicspath`**: change `{{../figures/}}` to `{{figures/}}` (figures are inside the zip, not one level up).
2. **Remove NeurIPS "Preprint" footer override** (the `\renewcommand{\@noticestring}{}` block) — let the style render its default or remove entirely for a clean look. User decision.

### Size budget

7 PNGs + `.tex` + `.bib` + `.sty` — estimated well under 10 MB (arXiv hard limit: 50 MB).

## Pre-submission Checklist

### Account & endorsement

1. Create arXiv account: https://arxiv.org/user/register
2. Check endorsement status: https://arxiv.org/auth/endorse
   - First-time submitters to `cs.LG` need endorsement from an existing arXiv author in that category.
   - If needed, request endorsement from a colleague. Details: https://info.arxiv.org/help/endorsement.html
   - Allow a few days — this is the most common blocker.

### Verify compilation locally

arXiv compiles LaTeX server-side using TeX Live + pdflatex. Test the zip contents:

```bash
cd <submission-dir>
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

All figures must resolve, all references must link. Fix any warnings about missing files or undefined references.

Alternatively, use the Docker build skill (`/build-latex-pdf`) with adjusted paths to match the zip structure.

### Metadata (for the submission form)

| Field | Value |
|-------|-------|
| Title | from `paper.tex` `\title{...}` |
| Authors | from `paper.tex` `\author{...}` |
| Abstract | from `paper.tex` `\begin{abstract}...` |
| Primary category | `cs.LG` (Machine Learning) |
| Cross-list | `stat.ML` (Statistics — Machine Learning) |
| Optional cross-list | `cs.CL` (Computation and Language) — LLM-relevant |
| License | CC BY 4.0 (recommended) or arXiv perpetual non-exclusive |
| Comments | "10 pages, 7 figures, 5 tables" |

## Submission Steps

1. Log in: https://arxiv.org/user/login
2. Start new submission: https://arxiv.org/submit
3. Fill metadata (title, authors, abstract, categories, license, comments).
4. Upload the zip file.
5. arXiv compiles LaTeX and shows a PDF preview — review it carefully.
6. If the PDF looks correct, confirm submission.

## Post-submission

- Moderator review: 1-2 business days.
- Announcement schedule: Sunday–Thursday. Deadline: **14:00 US Eastern** for next-day listing.
- Email with arXiv ID (e.g., `2604.XXXXX`) on acceptance.
- Permanent URL: `https://arxiv.org/abs/<id>`
- Revisions: submit anytime via https://arxiv.org/user

## Potential Blockers

| Blocker | Mitigation |
|---------|------------|
| Endorsement needed | Start early — request from a colleague who published in `cs.LG` |
| Compilation failure on arXiv | Test locally with TeX Live pdflatex; arXiv has a fixed package set — NeurIPS style + standard packages should all be available |
| Figures not found | Ensure `\graphicspath{{figures/}}` matches zip structure — #1 cause of failed builds |
| Package size | Keep PNGs reasonable; current estimate well under 10 MB |

## Build Script (for preparing the zip)

```bash
mkdir -p /tmp/arxiv-submission/figures
cp paper/paper.tex /tmp/arxiv-submission/
cp paper/references.bib /tmp/arxiv-submission/
cp agents/skills/build-latex-pdf/references/neurips_2024.sty /tmp/arxiv-submission/
cp figures/*.png /tmp/arxiv-submission/figures/

# Fix graphicspath in the copy
sed -i 's|{../figures/}|{figures/}|' /tmp/arxiv-submission/paper.tex

# Build zip
cd /tmp/arxiv-submission && zip -r ~/arxiv-submission.zip .
```

## References

| Topic | Link |
|-------|------|
| Submission guide | https://info.arxiv.org/help/submit/index.html |
| LaTeX requirements | https://info.arxiv.org/help/submit_tex.html |
| Endorsement | https://info.arxiv.org/help/endorsement.html |
| Licenses | https://info.arxiv.org/help/license/index.html |
| Submission schedule | https://info.arxiv.org/help/availability.html |
| Account registration | https://arxiv.org/user/register |
