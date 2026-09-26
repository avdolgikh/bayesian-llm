# S4 Result figure: AUROC per method and OOD domain, built from the score files (`specs/i2-result-figure.md`)

Status: DRAFT (not approved). Date: 2026-09-26 (revision 1, after the critic pass; Section 6.4 lists every change). Owner: Alexey approves; agents implement.

## 1. Header

| Field | Value |
|---|---|
| Items | **m10**: one result figure. It shows the document-weighted AUROC, with a document-bootstrap CI, for every method and baseline, in one panel per OOD domain. A script builds it from the score files. It writes PDF, PNG and a figure-data CSV. |
| Issues | **P12**: the paper has no result figure. Its 3 figures are decorative PNGs, and the first result table is on page 5 of 8. |
| Approve by | Sun 2026-10-04 (roadmap Section 3, row S4). Recommended: Thu 2026-10-01, together with S3, S5 and S6. The approval commit is the design freeze (Section 3). The first test scores are written on the night of Fri Oct 2 (roadmap Section 2). An approval after that sets `design_after_test_scores: true`, and the caption stub says so. |
| Build | Code and fixture tests need no data. Session 04 (Mon Oct 5) can build them if agents are free. Otherwise session 05 (Mon Oct 12), where the roadmap puts m10. S3 and S5 are built in session 04, so their file formats are provisional until then. A session-04 build of S4 may need a small fix in session 05 (S4-T3a on the real files finds it). |
| Run | The real build runs in session 05 (Mon Oct 12). By then these exist: S2's refit scores, S3's night A (Mon Oct 5) and its noise controls (night B, Tue Oct 6), and S5's fine-tunes and refits (Tue Oct 6) and re-score (Wed Oct 7). `--check-tex` runs in session 06 (Mon Oct 19) on the rewritten paper, and again before submission (m12, Fri Oct 23). |
| Depends on | S1, as built on 2026-09-26: the score-file format and names, `prereg.json`, `analysis_sha256`, `verify_prereg`, `freeze_prereg` and `ROLE_EVAL_SET` (minigpt/evalset.py), `doc_bootstrap_auroc` and `auroc` (minigpt/uncertainty.py), the cell rule of `--analyze`, and the table JSON files. S2, as built: the score sets `c2_refit`, `c4_lap_refit` and `c4_tfb_fixed` and their fit records. S3 (draft): `c0_snapshot`, the one-pass feature files, the first-block noise controls and their tables. S5 (draft, Recommended option only): the seed score sets and `table_s5.json`. m11: the main detection table with row markers (Section 5.9). Section 2 lists every interface. |
| Feeds | m11: Figure 1 of the rewrite and a caption stub whose numbers come from the data. m12: the final checks. m13: the public figure files. m15: a second path that recomputes every figure cell from the score files and must match S1's tables (S4-T3a). |

Costs. The roadmap column is roadmap Section 3, row S4 (m10 in options-and-costs.md Section 4.1). The next column is this spec's own estimate.

| Cost | Roadmap row S4 | This spec | Difference and why | Assumption |
|---|---|---|---|---|
| Alexey, approve | in m2 (12-18 min per spec) | in m2 | none | CA1 |
| Alexey, review | 0.25 h | 0.75-1.25 h: code review 0.5-1.0 h, plus 0.25 h to look at the figure and the quiz result (S4-T4h) | +0.5-1.0 h. CA1's rate (1 h per 6-9 engineering hours) on 5.0-6.75 h gives 0.56-1.13 h, rounded to 0.5-1.0 h. Revision 0 used 0.25-0.5 h, below CA1's own low end, and PG9 rules out cutting review. The review lands in W3 (4.75-6.0 h planned), or in W2 (4.0-5.75 h) after a session-04 build. Either week then goes over 5 h in most of its range. The roadmap's cut order does not list S4. C4 is the lever. | CA1, PG9 |
| Engineering | 3-4 h | 5.0-6.75 h (est.; breakdown below) | +1.75-2.75 h. The sources (options m10, "W4 3 h"; paper-critique #25, 2.5 h by hand) price the plot from S1's files only. C4 cuts 0.75-1.25 h. | CA2 |
| GPU (RTX 4070) | 0 | 0 | none. Every input is a saved file. | |
| CPU | not listed | 3-6 min per full build (probe of the built function). About 1-2 min if one call serves all score sets of a cell's blocks. Layout changes re-plot from the CSV in seconds. | new | Section 6.2 |
| Opus agents | 5-8 per implementation spec | 3-4 Opus (test writer, implementer, reviewer) and 1 Sonnet (quiz reader) | S4 is small | CA6 |
| Cloud | $0 | $0 | none | |

Engineering breakdown (est.):

| Part | Hours | Cut by C4 |
|---|---|---|
| Figure from S1's score files: cells, CSV, plot (the sources' scope) | 2.5-3.0 | no |
| Inputs beyond S1's files: S2 fit records, S3 feature files and first-block noise files, S5 seeds, subset alignment | 0.5-0.75 | no |
| Refusals (S4-T5d) | 0.25 | no |
| Design freeze from the spec commit | 0.25 | yes |
| `--check-tables` | 0.25 | no |
| `--check-tex` and the placement check | 0.25-0.5 | yes |
| Layout checks: size, fonts, text and marker overlap | 0.25-0.5 | no |
| Quiz maker and grader | 0.25-0.5 | yes |
| Fixture writer (S1's keys; S3's and S5's provisional formats) | 0.5-0.75 | no |
| **Total** | **5.0-6.75** | 4.25-5.5 left |

CPU breakdown (probe in Section 6.2):

| Term | Value |
|---|---|
| Cells with a CI, Minimal | 24 per panel × 3 panels = 72: 18 rows with one seed each, plus 6 noise controls (Section 3 row table) |
| Cells with a CI, Recommended | 30 per panel × 3 = 90: 20 rows, 3 seeds on `hn_blob`, `hn_dtfb` and `hn_dlap`, plus 4 noise controls |
| Blocks per cell | 2,500-5,000 (S1 section 5.8: 1,500-2,000 ID blocks; 1,000-3,000 OOD blocks). First-block noise cells: about 2,000. |
| Time per cell | 3.0 s with the built `doc_bootstrap_auroc` (2,040 ID + 2,021 OOD blocks, 1,000 + 1,000 documents, 10,000 resamples, one score). Each extra score in the same call adds 0.5 s. About 2.5 s is the per-resample draw loop (minigpt/uncertainty.py:593-603). |
| Full build | $72\text{-}90 \times 2.5\text{-}4$ s = 3-6 min. One call per (ID domain, OOD domain, block subset) shares the draws: about $12 \times 2.5$ s $+\ 90 \times 0.5$ s, so 1-2 min. The probe shows that a score's CI is bit-identical whatever other scores share the call. |
| Contrast markers (block MI) | point AUROC only, 1.3 ms each |

Differences from roadmap Section 3, row S4:

| # | Roadmap text | This spec | Why |
|---|---|---|---|
| R1 | Test 1: `scripts/generate_figures.py` builds the figure | Kept as the entry point (`--result`). The logic lives in `minigpt/result_figure.py`. | CI lints `minigpt/` but not `scripts/` (.github/workflows/ci.yml:12). Tests import the module directly. |
| R2 | Test 1: "every method and baseline" | Every method and baseline on its primary ID set, with the stated exclusions (Section 3) | C0's $G$ is 0 by construction. Pre-fix sets use known wrong samplers. LoRA rows on StackExchange are secondary (S1 families F2 and F4). |
| R3 | Test 3: the paper's main table matches to 3 decimals | Kept (S4-T3b). Added: the CSV equals the table JSON files to $10^{-6}$ (S4-T3a), and the paper places the figure before its first result table (S4-T3c). | The table JSON exists two weeks before the paper table. Placement is the second half of P12. |
| R4 | 3 tests | 5 tests: readability (S4-T4) and reproducibility (S4-T5) added | The task for this spec asks for acceptance on both. Roadmap Section 3 allows 3-5. |
| R5 | Approve by Sun Oct 4 | Recommended Thu Oct 1. The approval commit is the freeze. | The m4 re-score runs on the night of Fri Oct 2. A design fixed after the scores exist is not pre-registered. A later approval is recorded, not hidden. |
| R6 | 0.25 h review, 3-4 engineering h | 0.75-1.25 h review, 5.0-6.75 engineering h | See the cost tables above |

## 2. Scope and non-scope

What changes:

| File | Change |
|---|---|
| `minigpt/result_figure.py` (new, about 400 lines, est.) | Stage 1: input files to figure-data rows (`build_rows`). Stage 2: CSV to figure (`plot_rows`). Also `write_csv`, `read_csv`, `write_meta`, `check_layout`, `check_tables`, `check_tex`, `verify`, `freeze_design`, `make_quiz`, `grade_quiz` and `main(argv)`. matplotlib is imported inside functions, so `import minigpt` does not need it. It imports nothing from `scripts/`. |
| `scripts/generate_figures.py` | New flags (Section 5.7). With `--result`, it calls `minigpt.result_figure.main`. Without `--result`, it behaves exactly as today. |
| `configs/i2_result_figure.yaml` (new) | Every key explicit (Section 5.5). Its `design` block equals the spec's block at the approval commit. All style numbers live here, not in code (S4-T2b). |
| `tests/test_result_figure.py` (new) | S4-T1 to S4-T5 on a CPU fixture (Section 5.6) |
| `figures/fig_result_auroc.{pdf,png,csv,meta.json,caption.tex}` (new, tracked if Q3 allows) | The outputs. The CSV lets a reader redraw the figure without `data/`, which is gitignored. |
| `data/scores_i2/figure_prereg.json`, `data/scores_i2/figure_quiz/` (new, gitignored) | The design freeze, the quiz questions and the hidden answers |
| AGENTS.md, README.md, `agents/technical-reference.md`, `agents/design-rationale.md` | Doc updates under the rules below |

What must not change:

- S1, S2, S3 and S5 code, their score and feature files, their tables, `prereg.json` and `configs/i2_eval.yaml`. S4 only reads them.
- The existing behaviour of `scripts/generate_figures.py` (`--fig`, `--png`), its functions `fig1`-`fig7` and the 7 PNG files in `figures/`. m11 and m13 decide their fate (handoff in Section 7).
- `paper/`. S4 reads `paper/paper.tex` in `--check-tex` and never writes it. m11 owns the table, the caption and the placement.
- report.md and the README numbers (m13).
- `pyproject.toml` and `uv.lock`. matplotlib 3.10.8 and scikit-learn 1.8.0 are already in the uv environment through mlflow (uv.lock:1367, 2723; mlflow's dependency list at 1451, 1457). S4 adds no dependency.

Interfaces. "Built" rows were checked against the working tree on 2026-09-26. "Prov." rows come from the S3 and S5 drafts of the same day, which are in their own critic passes. A rename there changes Section 5.5 of this spec before approval, or needs an amendment commit (Section 3).

| S4 needs | From | What is provided or proposed |
|---|---|---|
| Score files | S1, built: path scripts/eval_c_checkpoints.py:883-884; keys :762-778; meta :1094-1117 | `data/scores_i2/{score_set}__{eval_set}__{run_tag}.pt`. S4 reads `blk_g`, `blk_mi`, `blk_tu`, `blk_au`, `blk_nll`, `blk_maxprob_unc` ([B] float32), `block_index` (int64), `doc_id` and `domain` (lists of str), `offset` (int64), `weight` (float32; it keeps the full-set $1/k_d$ even in a subset file, :193-204), and `meta` with `score_set`, `eval_set`, `run_tag`, `block_ids`, `sampler`, `display_label`, `n_samples`, `analysis_sha256`, `checkpoint_sha256` (dict path to sha256), `created_at`. There is no writer function. The scorer builds the payload inline. |
| Eval sets | S1, built: minigpt/evalset.py:46, 153-161 | `ROLE_EVAL_SET`: `test` holds the StackExchange, arXiv, FreeLaw and PubMed blocks. `test_hn` holds the HackerNews blocks only. |
| Which blocks form a cell | S1, built: `_cell_sources` and `_CellData.cell`, scripts/eval_c_checkpoints.py:1167-1236 | ID blocks: the rows of `{s}__{ROLE_EVAL_SET[role]}__{r}.pt` with the ID domain. OOD blocks: the rows of `{s}__test__{r}.pt` with the OOD domain. ID first, then OOD, each in file order. Weights are recounted in float64 per class, $1/k_d$ over the cell's blocks (:1230-1232). The stored `weight` is not used. S4 copies this rule, because the class is private to a script. S4-T3a checks the copy. |
| Bootstrap and RNG | S1, built: minigpt/uncertainty.py:672-724; seed rule scripts/eval_c_checkpoints.py:1279 | `doc_bootstrap_auroc(scores, labels, doc_ids, weights, n_resamples, [seed, i_id, i_ood], level, target_tpr=...)` with `analysis.bootstrap` (10,000 resamples, seed 0, level 0.95) and `i` = position in `eval_set.domains` (stackexchange 0, hackernews 1, arxiv 2, freelaw 3, pubmed_abstracts 4). It returns `auroc`, `auroc_ci`, `n_id_docs` and `n_ood_docs` per score. The contrast point is `auroc(s, y, sample_weight=w)`, the same call that `doc_bootstrap_auroc` makes. S4 re-implements neither. |
| Freeze check | S1, built: minigpt/evalset.py:198-238 | `verify_prereg(cfg, path)` raises `PreregError` on a missing file or a differing hash. S4 calls it first, then refuses any score file whose `meta.analysis_sha256` differs from `analysis_sha256(cfg)`. |
| Tables for S4-T3a | S1, built: `run_analyze`, scripts/eval_c_checkpoints.py:1252-1320 | `data/scores_i2/table_test.json` (ID StackExchange) and `table_test_hn.json` (ID HackerNews). Each has `cells[]` with `score_set`, `id_domain`, `ood_domain`, `n_id_docs`, `n_ood_docs`, `rng_seed` and `scores[name].auroc_w` and `.auroc_w_ci` for the 6 `blk_*` names. Every set in `scoring.score_sets.test` or `.test_hn` that has a `main` file gets cells. So S3's `c0_snapshot` and S5's sets appear here once S3 and S5 add them to that list. |
| Fixed post-hoc rows | S2, built: minigpt/posthoc_refit.py:75-83, 1034-1064, 1150-1191; scripts/eval_c_checkpoints.py:496-501, 545-564 | Score sets `c2_refit` (ID StackExchange), `c4_lap_refit` and `c4_tfb_fixed` (ID HackerNews), with `meta.sampler == "v2"`. Fit records: `data/checkpoints/i2_posthoc/{c2,c4_lap,c4_tfb}/fit_record.json`. The two Laplace records hold `match_status`. `matched`, `matched_noisy` and `tighter_than_budget` are scored. `not_matched` is not scored, and the scorer refuses to write its file. `bisection_failed` exits non-zero. The TFB record has **no** `match_status`: it holds `exit_code` (0, 2 or 4) and `failures`. Every passed record holds `state_sha256`, and the scorer puts the state file's sha256 into `meta.checkpoint_sha256`. If the TFB fit fails, neither Laplace refit has a budget, and neither writes a record (minigpt/posthoc_refit.py:896-915). |
| Snapshot ensemble | S3 draft, Sections 2 and 5.6 (prov.) | Score set `c0_snapshot` (sampler `ensemble`, N=5) on `test`, in S1's format, from S1's scorer |
| One-pass features | S3 draft, Section 5.3 (prov.) | Feature files `data/scores_i2/onepass__{center}__{eval_set}.pt` with `op_nll`, `op_ent`, `op_maxprob_unc`, `op_md`, `op_rmd` and `op_lr`, all oriented so that higher = more OOD. `op_lr` is $\mathrm{NLL}_c-\mathrm{NLL}_{\mathrm{bg}(c)}$. The surface file `surface__{eval_set}.pt` holds the 7 `sf_*` rates. Both carry `block_index`, `doc_id`, `domain`, `offset` and `weight` in S1's order. Centers: `c0` for the StackExchange rows, `c3_mean` for the HackerNews rows. They are not S1 score files: no `analysis_sha256`, no run tag. Tables: `data/scores_i2/s3/baselines_table.json`. |
| Noise controls | S3 draft, Sections 5.2 and 5.5 (prov.) | Score sets `noise_ffn_c0_mcd`, `noise_ffn_c1`, `noise_ffn_c0_c2`, `noise_lora_c3_blob` and `noise_lora_c3_tfb`. `c4_lap_refit` shares `noise_lora_c3_tfb`, unless S3's rule gives it `noise_lora_c3_lap`. Sampler `isotropic`, run tag `first`: the first block (smallest `offset`) of each document only. Fit records under `data/checkpoints/i2_posthoc/noise_*/`. A control that ends `not_matched` or `bisection_failed` has no score file. Table: `data/scores_i2/s3/noise_table.json`, with both marginal AUROCs and their CIs per cell. |
| Seed rows | S5 draft, Sections 3 and 5.4 (prov.) | `blob_1`-`blob_3` (sampler `variational`), `det_tfb_1`-`det_tfb_3` and `det_lap_1`-`det_lap_3` (`v2`), and `lora_ens3` (`ensemble`, N=3). Each is on `test` and `test_hn` with run tag `main`. C3 stays a separate reference row. Fit records: `data/checkpoints/i2_posthoc/b2/{tfb,lap}_s{201,202,203}/fit_record.json`. Under S5's plan rule the files may hold only the first block of each document (`meta.block_ids` set). Table: `data/scores_i2/table_s5.json`. |
| The paper table | m11 | A table labelled `tab:detection`. Each row that shows a figure row ends with `% fig-row: <row_id>`. Cells in the column order of `checks.tex_columns` hold `AUROC [lo, hi]` (Section 5.9). |

AGENTS.md rules that apply:

- No notebooks. No extra Bayesian library (none is needed).
- Explicit configs. The module reads every key as `cfg["key"]`. Style numbers go in the YAML, and S4-T2b checks the module for float literals.
- LaTeX for every formula in this spec.
- Keep repo docs fresh. Add `result_figure.py` to the AGENTS.md structure block, and the new test count to AGENTS.md and README.md (structure block and Quick Start). Add the Quick Start line `uv run python scripts/generate_figures.py --result --config configs/i2_result_figure.yaml`. Record the CSV format and the new figure in `agents/technical-reference.md` ("Figures & TikZ"). Record the design decisions of Section 3 in `agents/design-rationale.md`.
- 4-space indent, 100-character lines, type hints on public functions. Run `ruff check` on `minigpt/ experiments/ tests/` and on `scripts/generate_figures.py`, and `pytest`, before each commit. Conventional Commits. Never mention AI assistants.
- Use `uv run` for every build. Global Python has matplotlib 3.10.9, and the committed bytes come from the uv environment (3.10.8).

Open choices for the approver (the default applies if Alexey says nothing):

| # | Choice | Default | Alternative | Cost of the alternative |
|---|---|---|---|---|
| C1 | Value labels | None. The numbers live in the table and the CSV. Paper-critique Section 5 asks to print each number once. | Print $G$'s AUROC to 2 decimals right of each whisker | +0.25 h. With 70-90 labels the panels get crowded, and S4-T4c then covers the labels too. |
| C2 | BLoB-mean + TFB and BLoB-mean + Laplace rows under Recommended | Table only. The figure shows the deterministic-LoRA rows from S5 (S2 section 7). | Keep both rows in the figure too | +2 rows: height 4.2 in to 4.5 in, the limit. C2 and C3 together do not fit. |
| C3 | A "one-pass combination (cross-validated)" row per ID group | None. It is a fitted model, and S3 tests it with paired CIs in its own table. | Add it, if S3 writes out-of-fold block scores in a file aligned with S1's blocks | +2 rows (+0.3 in), +0.25 h |
| C4 | Roadmap-size build | Full scope (5.0-6.75 h) | Drop the quiz (S4-T4h), `--check-tex` (S4-T3b-c become checks by eye in m12) and the design freeze (the approval commit alone is the record) | Saves 0.75-1.25 engineering h and up to 0.25 review h (4.25-5.5 h left). Readability then rests on the layout checks alone. |

## 3. Pre-registered endpoint

S4 has no scientific endpoint. It fixes how the results are shown before any test score is read, so that no display choice can follow the numbers. S1 fixes the score, the ID sets, the CIs and the tests. S3 fixes the baselines and the noise controls. S5 fixes the seeds. S4 restates none of their rules.

Frozen design. The registration is the `design` block of the fenced YAML in Section 5.5 of this file, as committed at approval. At build time, `--freeze-design --spec-commit <sha>` reads the spec from that commit (`git show <sha>:specs/i2-result-figure.md`), parses its one fenced `yaml` block, and checks that its `result_figure.design` equals the `design` block of `configs/i2_result_figure.yaml`. Then it writes `data/scores_i2/figure_prereg.json` with `design_sha256` (sha256 of `json.dumps(design, sort_keys=True, separators=(",", ":"))`, S1's hash rule), `spec_commit`, `spec_commit_time`, `plan_option`, `frozen_at` and `design_after_test_scores`. That flag is true if `spec_commit_time` is later than the earliest `meta.created_at` of any S1 `test` or `test_hn` score file. S3 uses the same mechanism (S3 Section 3). Every build refuses a design hash that differs from `figure_prereg.json`, unless `--post-hoc "<reason>"` is given. The reason goes into the meta and into the caption stub. A rename by S3 or S5 after S4's approval needs an amendment commit of this spec before any score of the renamed set exists. That commit is then the `--spec-commit`. This DRAFT, dated 2026-09-26, comes before every test score: `data/scores_i2/` does not exist yet. So only open choices C1-C3, and names that S3 and S5 change, may change at approval.

| Setting | Value | Source or reason |
|---|---|---|
| Filled circle | $\mathrm{AUROC}_w(G)$ (`blk_g`) with its 95% document-bootstrap CI | S1 pre-registered $G$ as the primary score. S1: "We do not switch after seeing the test set." |
| Hollow circle | $\mathrm{AUROC}_w(M)$ (`blk_mi`), point only, on the same blocks, multi-sample rows only | S1's primary contrast. Its paired test is in S1's `families.json`, not in the figure. |
| Panels | arXiv, FreeLaw, PubMed abstracts, in S1's `analysis.ood_domains` order; shared x and y axes | roadmap S4 test 1 |
| ID set of a row | The method's own training domain: StackExchange for MC dropout, C1, C2 and the C0 snapshot ensemble; HackerNews for C3, C4 and the S5 rows | S1 decision D5 |
| Rows and order | The row table below, in that order, in every panel. Never sorted by value. | Same position in all 3 panels. No order that depends on the outcome. |
| Excluded | C0's $G$ ($N = 1$ gives $G = 0$). The pre-fix sets `c2`, `c4_lap` and `c4_tfb` (known wrong samplers; `c4_tfb` is descriptive in S1). LoRA rows on StackExchange (S1 families F2, F4; appendix table). S3's `c1_mean` one-pass center, `op_md`, the 6 other surface rates, `c3_mean_copies` and the fitted combinations (C3). | |
| Rows without scores | A planned row stays in place with the text "not scored". A planned seed of a multi-seed row without a score gives no marker, and its row label shows the scored count ("×2"). The caption stub names each one with its reason. A missing file is allowed only when its fit record gives a status in the row's `allowed` list (`not_matched` for the Laplace rows). A planned noise control without files is allowed when its S3 fit record says `not_matched` or `bisection_failed` (S3: "that row has no control"). Anything else missing stops the build. | A hidden row hides a negative result. S2 section 3: "The row is not scored. The paper says so." |
| Direction | As stored: a higher score means more OOD (S1 section 5.6; S3 Section 5.3). No score is flipped, even when its AUROC is below 0.5. | |
| CI | Marginal 95% percentile interval, 10,000 resamples, S1's RNG rule | S1 section 5.6 |
| Seeds | One marker and whisker per seed, never pooled. C3 and the 3 new BLoB seeds are separate rows (S5 C1 default). | S5-T4 reports each seed's CI |
| Noise control | Hollow diamond with a whisker, in its method's row and colour. It is scored on the first block of each document (S3 run tag `first`), so it does not sit on the same blocks as the filled circle. S3's `noise_table.json` compares method and control on the same blocks. The caption says both. | S3-T2 flags a method whose CI overlaps its noise control |
| Block subset | A row whose files hold only first blocks (S5's plan rule) is marked `block_subset = first` in the CSV and is named in the caption. Other subsets are refused. | S5 Section 5.6 |
| x range | $[x_{\min}, 1.01]$, the same in all panels (rule in Section 5.3) | No zoom chosen by eye |
| Text in the plot | Row labels, group headers, panel titles, axis label, legend, "not scored". No value labels (C1), no "best", no bold row, no arrows, no claim text. A dashed chance line at 0.5. | paper-critique #8 (bold hid MC dropout's FPR@95) and #25; the old fig5 and fig6 carried claim text (Section 6.2) |
| Caption stub | Fixed template. Numbers only from the CSV and meta. None of the `design.banned_words`. | S4-T2d |

Row table. Source "S1" is an S1-format score file, "feat." an S3 feature file. Prov. = provisional name from the S3 or S5 draft.

| # | `row_id` | ID group | Label | Kind, colour | Source: score sets, key | Hollow MI | Noise control (run tag `first`) | Planned under |
|---|---|---|---|---|---|---|---|---|
| 1 | `se_mcd` | StackExchange | MC dropout (C0) | multi-sample, dropout | S1: `mc_dropout`, `blk_g` | yes | `noise_ffn_c0_mcd` (prov.) | both |
| 2 | `se_c1` | StackExchange | Variational FFN (C1) | multi-sample, variational | S1: `c1` | yes | `noise_ffn_c1` (prov.; S3 choice 3) | both |
| 3 | `se_c2` | StackExchange | Diag. Laplace FFN (C2) | multi-sample, post-hoc | S1: `c2_refit`, sampler v2 | yes | `noise_ffn_c0_c2` (prov.) | both; `not_matched` gives "not scored" |
| 4 | `se_snap` | StackExchange | Snapshot ensemble (C0) | multi-sample, ensemble | S1: `c0_snapshot` (prov.) | yes | none | both |
| 5 | `se_nll` | StackExchange | Sequence NLL (C0) | one-pass, black | S1: `c0`, `blk_nll` | no | none | both |
| 6 | `se_ent` | StackExchange | Token entropy (C0) | one-pass | S1: `c0`, `blk_tu` | no | none | both |
| 7 | `se_msp` | StackExchange | 1 − max-prob (C0) | one-pass | S1: `c0`, `blk_maxprob_unc` | no | none | both |
| 8 | `se_char` | StackExchange | LaTeX-character rate | one-pass | feat.: `surface`, `sf_tex` (prov.) | no | none | both |
| 9 | `se_rmd` | StackExchange | Rel. Mahalanobis (C0) | one-pass | feat.: `onepass__c0`, `op_rmd` (prov.) | no | none | both |
| 10 | `hn_c3` | HackerNews | BLoB LoRA (C3) | multi-sample, variational | S1: `c3` | yes | `noise_lora_c3_blob` (prov.) | both |
| 11 | `hn_blob` | HackerNews | BLoB LoRA (new seeds) | multi-sample, variational | S1: `blob_1`, `blob_2`, `blob_3` (prov.) | yes | none | Recommended |
| 12 | `hn_tfb` | HackerNews | BLoB-mean + TFB | multi-sample, post-hoc | S1: `c4_tfb_fixed`, sampler v2 | yes | `noise_lora_c3_tfb` (prov.) | Minimal (both under C2) |
| 13 | `hn_lap` | HackerNews | BLoB-mean + diag. Laplace | multi-sample, post-hoc | S1: `c4_lap_refit`, sampler v2 | yes | `noise_lora_c3_lap` if it exists, else `noise_lora_c3_tfb` (prov.) | Minimal (both under C2); `not_matched` gives "not scored" |
| 14 | `hn_dtfb` | HackerNews | Det. LoRA + TFB | multi-sample, post-hoc | S1: `det_tfb_1`-`det_tfb_3` (prov.) | yes | none | Recommended |
| 15 | `hn_dlap` | HackerNews | Det. LoRA + diag. Laplace | multi-sample, post-hoc | S1: `det_lap_1`-`det_lap_3` (prov.) | yes | none | Recommended; a `not_matched` seed is dropped and named |
| 16 | `hn_ens` | HackerNews | LoRA ensemble (M=3) | multi-sample, ensemble | S1: `lora_ens3` (prov.) | yes | none | Recommended |
| 17 | `hn_nll` | HackerNews | Sequence NLL (C3 mean) | one-pass | feat.: `onepass__c3_mean`, `op_nll` (prov.) | no | none | both |
| 18 | `hn_ent` | HackerNews | Token entropy (C3 mean) | one-pass | feat.: `onepass__c3_mean`, `op_ent` (prov.) | no | none | both |
| 19 | `hn_msp` | HackerNews | 1 − max-prob (C3 mean) | one-pass | feat.: `onepass__c3_mean`, `op_maxprob_unc` (prov.) | no | none | both |
| 20 | `hn_char` | HackerNews | LaTeX-character rate | one-pass | feat.: `surface`, `sf_tex` (prov.) | no | none | both |
| 21 | `hn_rmd` | HackerNews | Rel. Mahalanobis (C3 mean) | one-pass | feat.: `onepass__c3_mean`, `op_rmd` (prov.) | no | none | both |
| 22 | `hn_lr` | HackerNews | NLL(C3) − NLL(C0) | one-pass | feat.: `onepass__c3_mean`, `op_lr` (prov.) | no | none | both |

Rows per option: Minimal draws rows 1-10, 12, 13 and 17-22 (18 rows, 3.9 in tall). Recommended draws rows 1-11 and 14-22 (20 rows, 4.2 in). A row with $k$ scored seeds gets the label suffix " ×$k$", for example "Det. LoRA + TFB ×3".

What the figure shows whatever the numbers. It has no pass or fail on results. It must show, unchanged: rows at or below chance, one-pass rows to the right of multi-sample rows, S2 rows that were not scored, and seeds that disagree. The caption stub ranks nothing. Rankings come only from the paired tests of S1, S3 and S5.

## 4. Acceptance tests

| ID | Given | When | Then | Runs in |
|---|---|---|---|---|
| S4-T1 Build from the score files | **Fixture** (Section 5.6): synthetic files for 7 rows: one 3-seed row, one row with a first-block noise control, one feature-file row, and one row that is not scored. 30 documents per domain, 200 resamples, a fixture eval YAML and its `prereg.json`, a fixture result-figure YAML and its `figure_prereg.json`. **Real**: the final files in `data/scores_i2/` and `configs/i2_result_figure.yaml`, with `plan_option` from G0. | `uv run python scripts/generate_figures.py --result --config <yaml>`. Pytest calls `minigpt.result_figure.main([...])` with the same arguments. | (a) The five files are written: PDF, PNG, CSV, meta JSON and caption stub (yes/no). (b) The fixture CSV has exactly 48 data lines, 16 per panel. The real CSV has $n_{\mathrm{csv}}$ lines (Section 5.2), computed from the YAML and the fit-record statuses (yes/no). (c) Every primary and noise line has $0\le\ell\le a\le u\le1$. Every contrast line has empty `ci_lo` and `ci_hi` (yes/no). (d) The panel titles read back from the Figure equal `panel_titles` in YAML order. The y tick labels read back from each Axes equal the planned labels, with seed suffixes, in YAML order (yes/no). (e) No line pairs a score set of S1's families F1 or F3 with an ID domain other than the one those families give it (yes/no). (f) No line has score set `c2`, `c4_lap` or `c4_tfb`, and no line has (`c0`, `blk_g`) (yes/no). (g) The fixture's not-scored row has one line per panel with `status = not_scored:not_matched`, and the text "not scored" sits at that row in each panel (yes/no). (h) The fixture's noise line has `run_tag = first`, `block_subset = first`, and `n_id_blocks` = `n_id_docs` and `n_ood_blocks` = `n_ood_docs` (yes/no). (i) Real: every scored line has `n_id_docs` and `n_ood_docs` equal to `eval_set.n_test_docs` (1,000), or the build report names the domain where S1-T3b found another count (yes/no). | `tests/test_result_figure.py` (CPU, CI); real: `scripts/generate_figures.py --result` in session 05 |
| S4-T2 No hardcoded number | The fixture after one S4-T1 build, and the source of `minigpt/result_figure.py` | (i) Add 5.0 to `blk_g` of `block_index` 0 (a StackExchange block) in `mc_dropout__test__main.pt`, then build again. (ii) Scan the module with `ast`. (iii) Delete the fixture input files, then run `--from-csv` on the CSV of (i). | (a) Exactly the 3 `se_mcd` primary lines change their `auroc`. The 3 `se_mcd` contrast lines change only in the two file-sha256 columns. The other 42 lines are byte-equal. The PDF and PNG sha256 both change (yes/no). (b) The module holds 0 float literals other than 0.0 and 1.0, and 0 string literals that parse as a decimal number (yes/no). (c) The `--from-csv` outputs are byte-equal to the outputs of (i). So the plot reads the CSV only (yes/no). (d) Every number in the caption stub equals a CSV or meta value, and the stub holds none of `design.banned_words` (yes/no). | `tests/test_result_figure.py` |
| S4-T3 Same numbers as the tables and the paper | **Fixture**: `table_test.json` and `table_test_hn.json` written by S1's own `run_analyze` (scripts/eval_c_checkpoints.py, imported with importlib) on the fixture score files (Section 5.6). Six fixture `.tex` files: one that matches the CSV; one with one digit changed in one cell; one in `.917` style and one in `0.917` style; one with a number-free table before the figure; one with a result table before the figure. **Real**: `table_test.json`, `table_test_hn.json`, the S3 and S5 tables in `checks.extra_table_files`, and `paper/paper.tex` after m11. | `--check-tables`; `--check-tex <file>` | (a) Every CSV line whose (score set, key, ID domain, OOD domain, block subset) is in a listed table equals it to $10^{-6}$ in `auroc`, and primary and noise lines also in `ci_lo` and `ci_hi`. Every primary line of a score set in S1's `scoring.score_sets` is compared. Fixture: exactly 39 lines compared, 0 mismatches (number). Real: 0 mismatches; the report prints the number compared and lists the lines not compared (number). (b) `--check-tex` exits 0 on the matching fixture. It exits 1 on the changed one, with a message that names the `row_id` and the column. Both number styles parse. Every table row with a `% fig-row:` marker exists in the CSV. A CSV primary row of an S1, S2 or S5 score set that is missing from the table fails the check; a missing one-pass or noise row is listed only (yes/no each). Real paper: every shared number is within $5\times10^{-4}$ of the CSV (exit 0). (c) `paper.tex` includes `fig_result_auroc.pdf` in a `figure` environment with `width=\textwidth`, and that environment comes before the first result table: the first `table` environment whose body matches the cell pattern of Section 5.9. The number-free fixture passes and the result-table-first fixture fails (yes/no each). | `tests/test_result_figure.py` (fixtures, CI); real: `scripts/generate_figures.py --result --check-tables` in session 05; `scripts/generate_figures.py --result --check-tex paper/paper.tex` in session 06 and before m12 |
| S4-T4 Readability | The fixture figure (CI) and the real figure (session 05) | `--check-layout` on the saved figure. It also runs at the end of every build and writes `layout_checks` into the meta; the build exits non-zero if a check fails. Then `--make-quiz`. A reader agent that did not build the figure answers the 5 questions from the PNG and the caption stub only. `--grade-quiz <answers.json>` grades. | (a) Width 5.5 in to 0.01 in, height at most 4.5 in, and PNG pixels = size × 300 exactly (yes/no). (b) The smallest visible text is 7 pt or more (number). (c) 0 pairs of visible text boxes overlap, and 0 pairs of markers in different glyph slots overlap (numbers). (d) Every whisker end lies inside the x limits, the 3 panels share their x limits, and each panel has the chance line at 0.5 (yes/no each). (e) The glyph kinds differ by marker and fill alone, and every data colour is in the YAML's Okabe-Ito list (yes/no). (f) The PDF holds 0 raster images, 0 Type 3 fonts and at least 1 embedded TrueType font file (numbers). (g) Every label, with its seed suffix, has 30 characters or fewer (yes/no). (h) Quiz: 4 or more of 5 answers are correct (number). Each miss is reported with its question. On the fixture, `grade_quiz` passes with the true answers and with one wrong answer, and fails with two wrong (yes/no). | `tests/test_result_figure.py` for (a)-(g) and the grader; real: `scripts/generate_figures.py --result` and `--check-layout` for (a)-(g); `--make-quiz` and `--grade-quiz` for (h), in session 05, with one Sonnet reader |
| S4-T5 Reproducible and traceable | The fixture of S4-T1; the real outputs in `figures/` | (i) Build the fixture twice, each time in a fresh process (`sys.executable scripts/generate_figures.py --result ...`), and once by a direct call from pytest. (ii) `--verify`. (iii) Change one input, then `--verify`. (iv) Try each refusal case. | (a) The CSV, PDF and PNG sha256 are identical across the three builds (yes/no). (b) The meta holds every key in Section 5.4. `analysis_sha256` equals `prereg.json`, `design_sha256` equals `figure_prereg.json`, and there is one sha256 per input file (yes/no). (c) `--verify` exits 0 after a build. It exits non-zero and names the file after 1 byte is appended to an input file. It exits non-zero and names the CSV after one CSV cell is edited. It re-plots from the CSV into a temporary folder and compares the PDF and PNG sha256 (yes/no each). (d) Each of these exits non-zero and writes no output (yes/no each): a score file whose `meta.analysis_sha256` differs from `prereg.json`; a run tag other than the row's (`main`, or `first` for a noise control); a file whose `meta.sampler` differs from the row's `sampler` (for example a `pre-fix` file on a v2 row); a post-hoc row whose score file's `meta.checkpoint_sha256` does not hold its fit record's `state_sha256`; two files of one eval set that disagree on `doc_id`, `domain` or `offset` for one `block_index`; a file that holds neither all blocks nor exactly the first block of each document; a planned row or seed with no file and no allowed status; a design hash that differs from `figure_prereg.json` without `--post-hoc`; `--freeze-design` when the YAML's `design` differs from the spec block, or when `figure_prereg.json` exists. With `--post-hoc "reason"` the build runs, and the reason appears in the meta and in the caption stub (yes/no). (e) The test module takes 60 s or less on CPU (number). | `tests/test_result_figure.py`; real: `scripts/generate_figures.py --result --verify` before every commit of `figures/` and before m12 |

Commands for the builder:

```bash
uv run pytest tests/test_result_figure.py -rs
uv run ruff check minigpt/ experiments/ tests/ scripts/generate_figures.py
uv run python scripts/generate_figures.py --result --config configs/i2_result_figure.yaml --freeze-design --spec-commit <approval sha>
uv run python scripts/generate_figures.py --result --config configs/i2_result_figure.yaml
uv run python scripts/generate_figures.py --result --config configs/i2_result_figure.yaml --check-tables
uv run python scripts/generate_figures.py --result --config configs/i2_result_figure.yaml --make-quiz
uv run python scripts/generate_figures.py --result --config configs/i2_result_figure.yaml --grade-quiz <answers.json>
uv run python scripts/generate_figures.py --result --config configs/i2_result_figure.yaml --verify
uv run python scripts/generate_figures.py --result --config configs/i2_result_figure.yaml --check-tex paper/paper.tex
```

The real build passes when the build exits 0 (so every `layout_checks` entry passed), `--check-tables` exits 0, `--verify` exits 0, and the quiz scores 4 or more of 5.

## 5. Implementation notes for the builder

### 5.1 Order of work

| Step | Work | Test | When |
|---|---|---|---|
| 1 | Fixture writer, then the S4-T1 to S4-T5 tests; they fail first | all | session 04 or 05 |
| 2 | Stage 1: cells from the input files through S1's functions; CSV; meta; refusals; design freeze | S4-T1, S4-T2a, S4-T5 | same |
| 3 | Stage 2: plot from the CSV; layout checks | S4-T2c, S4-T4a-g | same |
| 4 | `--check-tables`, `--check-tex`, `--verify`, quiz | S4-T3, S4-T4h, S4-T5c | same |
| 5 | Docs (Section 2) | none | same |
| 6 | After S3 and S5 are built: add readers for their tables to `checks.extra_table_files`; confirm that their real file names and keys equal Section 5.5 | S4-T3a | session 05, before the real build |
| 7 | Copy the Section 5.5 YAML into `configs/i2_result_figure.yaml`, set `plan_option` from G0, run `--freeze-design --spec-commit <approval sha>`, and record the hash in the session log and STATE.md | S4-T5d | before the first real build |
| 8 | Real build, `--check-tables`, quiz, `--verify`; commit `figures/` if Q3 allows | S4-T1i, S4-T3a, S4-T4 | session 05 |
| 9 | `--check-tex` on the m11 table and figure placement | S4-T3b-c | session 06; again at m12 |

### 5.2 Cells and formulas

A cell is (row, seed, glyph, OOD domain). Its blocks follow S1's built rule (Section 2, "Which blocks form a cell"), with the run tag of the glyph: `main`, or `first` for a noise control. Feature files are read at `{scores_dir}/{stem}__{eval_set}.pt` and follow the same rule. Let $\mathcal I$ be the ID blocks and $\mathcal O$ the OOD blocks of the cell, $s_b$ the score, and $w_b = 1/k_{d(b)}$, where $k_d$ is the number of blocks of document $d$ in the cell, recounted in float64:

$$\mathrm{AUROC}_w=\frac{\sum_{i\in\mathcal{I}}\sum_{j\in\mathcal{O}}w_iw_j\big(\mathbb{1}[s_j>s_i]+\tfrac12\mathbb{1}[s_j=s_i]\big)}{\big(\sum_{i\in\mathcal{I}}w_i\big)\big(\sum_{j\in\mathcal{O}}w_j\big)}$$

For resample $r$, document $d$ drawn $c^{(r)}_d$ times gives $w^{(r)}_b = w_b\,c^{(r)}_{d(b)}$. The interval is

$$\big[\,Q_{0.025}\big(\{A^{(r)}\}_{r=1}^{B}\big),\ Q_{0.975}\big(\{A^{(r)}\}_{r=1}^{B}\big)\,\big],\qquad B=10{,}000,$$

with numpy's linear percentile. The point estimate uses the original weights.

Rules:

- Call `doc_bootstrap_auroc` from `minigpt/uncertainty.py` with exactly the inputs that S1's `--analyze` uses: the generator seed `[seed, i_id, i_ood]`, ID blocks then OOD blocks in file order, and the recounted weights. The stored `weight` is not used, because a subset file keeps the full-set value. One call may carry several score sets that share the cell's blocks; the draws depend only on the document lists, so each CI is unchanged (probe, Section 6.2). S4-T3a checks all of this against S1's tables to $10^{-6}$. If S3's `noise_table.json` uses another seed for its first-block cells, S3's rule wins and S4 follows it.
- The contrast marker (block MI) is the point `auroc(s_M, y, sample_weight=w)` only.
- Seeds are separate cells. Nothing is averaged across seeds.
- Before any cell, check each eval set: every pair of its files (score and feature files) must agree on `doc_id`, `domain` and `offset` for every shared `block_index` (the S1-T5k rule, extended to subsets). A file is `all` if its `block_index` equals every block of the full file, and `first` if it holds exactly the block with the smallest `offset` of each document. Refuse anything else.
- CSV line count. Let $a_r$ be the scored seeds of row $r$, $u_r$ its planned seeds with an allowed not-scored status, $c_r\in\{0,1\}$ its contrast flag, and $q_r=1$ if the row plans a noise control and $a_r\ge1$ (a scored diamond line, or one `not_scored` line), else 0:

$$n_{\mathrm{csv}} = 3\sum_{r}\Big(a_r\,(1+c_r) + u_r + q_r\Big)$$

### 5.3 Layout rules

x range, the same for all panels. Let $\ell_c$ be the lower CI end of cell $c$:

$$x_{\min}=\min\Big(x_{\mathrm{default}},\ \rho\,\big\lfloor (\min_c \ell_c - x_{\mathrm{pad}})/\rho\big\rfloor\Big),\qquad x_{\max}=1.01,$$

with $x_{\mathrm{default}}=0.4$, $x_{\mathrm{pad}}=0.02$ and $\rho = 0.05$ (`x_round`). Ticks sit every `x_tick_step` (0.1) from $\lceil 10\,x_{\min}\rceil/10$ to 1.0.

Height, with row pitch $p = 0.15$ in, $n_r$ rows and $n_h = 2$ group headers:

$$H = h_{\mathrm{top}} + h_{\mathrm{bottom}} + p\,(n_r + n_h) \le 4.5\ \text{in}$$

Minimal: $0.9 + 0.15 \times 20 = 3.9$ in. Recommended: $0.9 + 0.15 \times 22 = 4.2$ in. If $H$ exceeds 4.5 in (plus $10^{-9}$ for rounding), the build fails. It never shrinks the fonts.

Glyph slots. Row $r$ sits at $y_r$. It has $n_s = k_r + q_r$ slots: its planned seeds in YAML order, then its noise control. Slot $j = 0,\dots,n_s-1$ sits at

$$y_{r,j} = y_r + \Big(j - \frac{n_s - 1}{2}\Big)\,\sigma,\qquad \sigma = 0.05\ \text{in} = 3.6\ \text{pt}$$

A seed's filled circle and hollow circle share its slot and differ in x. The build asserts $n_s\,\sigma \le p$ for every row (at most 3 slots) and a marker size below $72\,\sigma$ pt, so markers in different slots cannot overlap. Revision 0 put seeds at $\pm0.2$ of a row (2.16 pt apart) with 3.5-pt markers, so they overlapped.

Overlap test for S4-T4c. Two boxes $A$ and $B$ in display coordinates from the Agg renderer overlap if

$$A_{x_0}<B_{x_1}\ \wedge\ B_{x_0}<A_{x_1}\ \wedge\ A_{y_0}<B_{y_1}\ \wedge\ B_{y_0}<A_{y_1}.$$

The text test covers every visible `Text` in the figure: tick labels, titles, headers, legend entries, "not scored" and the axis label. The marker test covers every pair of markers in one panel that sit in different slots; a marker's box is its centre plus or minus half its size.

Glyphs. Filled circle with whisker: $G$ of a multi-sample row (a weight posterior, MC dropout or an ensemble). Hollow circle: $M$ of the same row and seed. Filled square with whisker: a one-pass score. Hollow diamond with whisker: a noise control. The colour gives the family (variational, post-hoc, MC dropout, ensemble; one-pass rows are black). The row label names the method too, so colour is never the only cue. Within a group, a thin light rule separates the multi-sample rows from the one-pass rows. The legend sits below the panels in two lines: glyphs, then families.

### 5.4 Two stages, CSV and meta

Stage 1 reads the input files and writes the CSV and the meta. Stage 2 reads only the CSV and the YAML and draws the figure. `--from-csv` runs stage 2 alone. This makes S4-T2c hold by construction. It also makes layout changes cheap: they skip the bootstrap.

CSV columns, in this order: `row_id, group_id, id_domain, ood_domain, label, kind, family, glyph, seed_index, score_set, score_key, source, run_tag, block_subset, status, fit_status, auroc, ci_lo, ci_hi, n_id_docs, n_ood_docs, n_id_blocks, n_ood_blocks, n_resamples, ci_level, rng_seed, id_file_sha256, ood_file_sha256`.

- `glyph` is `primary`, `contrast` or `noise`. `source` is `score` or `feature`. `block_subset` is `all` or `first`. `status` is `scored` or `not_scored:<status>`. `fit_status` is the fit record's `match_status` (Laplace rows) or `exit_code` (TFB rows), and empty otherwise. `rng_seed` is written as `seed-i_id-i_ood`.
- Floats are written with `repr(float(x))`. Empty cells are empty strings. Lines follow the YAML row order, then seed, then glyph, then panel order. Line ends are `\n`, and the encoding is UTF-8.

Meta keys (`figures/fig_result_auroc.meta.json`): `git_sha`, `git_dirty`, `code_sha256` (over `minigpt/result_figure.py`, `minigpt/uncertainty.py` and `scripts/generate_figures.py`), `python_version`, `matplotlib_version`, `numpy_version`, `sklearn_version`, `plan_option`, `analysis_sha256`, `design_sha256`, `spec_commit`, `design_after_test_scores`, `style_sha256`, `inputs` (repo-relative path and sha256 of each score file, feature file and fit record), `csv_sha256`, `pdf_sha256`, `png_sha256`, `not_scored` (row, seed and reason), `fit_status` (per post-hoc row and seed), `post_hoc_changes`, `layout_checks`, `created_at` (UTC). Paths are repo-relative.

The caption stub (`fig_result_auroc.caption.tex`) fills `design.caption_template` with values from the CSV and the meta: documents per domain, samples per row kind, resamples, the first-block rows, the not-scored rows with their reasons, and the late-freeze sentence if `design_after_test_scores` is true. m11 may reword it but must keep the numbers from the stub.

### 5.5 `configs/i2_result_figure.yaml` (every value explicit)

This block is the registration (Section 3). `--freeze-design` compares its `design` key with the config file.

```yaml
result_figure:
  eval_config: configs/i2_eval.yaml          # S1: domains, roles, analysis block, n_test_docs
  scores_dir: data/scores_i2
  prereg_path: data/scores_i2/prereg.json
  design_prereg_path: data/scores_i2/figure_prereg.json
  spec_path: specs/i2-result-figure.md       # read at --spec-commit by --freeze-design
  plan_option: recommended                   # minimal | recommended; set at G0 (Thu Oct 1)
  out_dir: figures
  out_stem: fig_result_auroc
  quiz_dir: data/scores_i2/figure_quiz
  noise_record_pattern: "data/checkpoints/i2_posthoc/{noise}/fit_record.json"   # S3 (prov.)
  checks:
    table_files: [data/scores_i2/table_test.json, data/scores_i2/table_test_hn.json]
    extra_table_files: []    # S3: data/scores_i2/s3/{baselines,noise}_table.json;
                             # S5: data/scores_i2/table_s5.json; added in step 6 (5.1)
    table_tol: 1.0e-6
    tex_table_label: "tab:detection"
    tex_row_marker: "fig-row:"
    tex_columns: [arxiv, freelaw, pubmed_abstracts]
    tex_tol: 5.0e-4
    tex_figure_file: fig_result_auroc.pdf
  quiz: {n_questions: 5, min_gap: 0.03, pass_min: 4, seed: 0}
  design:                                    # frozen; sha256 goes to figure_prereg.json
    panels: [arxiv, freelaw, pubmed_abstracts]      # asserted equal to analysis.ood_domains
    panel_titles: {arxiv: "OOD: arXiv", freelaw: "OOD: FreeLaw",
                   pubmed_abstracts: "OOD: PubMed abstracts"}
    primary_key: blk_g                       # asserted equal to analysis.primary_score
    contrast_key: blk_mi                     # asserted equal to analysis.contrast_score
    run_tag: main
    noise_run_tag: first
    block_subsets: [all, first]
    excluded_score_sets: [c2, c4_lap, c4_tfb]
    noise_not_scored: [not_matched, bisection_failed]
    x_default_min: 0.4
    x_max: 1.01
    x_pad: 0.02
    x_round: 0.05
    chance: 0.5
    max_label_chars: 30
    seed_suffix: " ×{k}"
    not_scored_text: "not scored"
    banned_words: [best, outperform, significant, definitive, only, superior, wins]
    groups:
      - {group_id: se, header: "ID: StackExchange", id_domain: stackexchange}
      - {group_id: hn, header: "ID: HackerNews (LoRA rows)", id_domain: hackernews}
    rows:                                    # plot order, top to bottom (Section 3 row table)
      - {row_id: se_mcd, group: se, label: "MC dropout (C0)", kind: sampling, family: dropout,
         source: score, score_sets: [mc_dropout], key: blk_g, contrast: true,
         sampler: dropout, noise: [noise_ffn_c0_mcd], status: null,
         plans: [minimal, recommended]}
      - {row_id: se_c1, group: se, label: "Variational FFN (C1)", kind: sampling,
         family: variational, source: score, score_sets: [c1], key: blk_g, contrast: true,
         sampler: variational, noise: [noise_ffn_c1], status: null,
         plans: [minimal, recommended]}
      - {row_id: se_c2, group: se, label: "Diag. Laplace FFN (C2)", kind: sampling,
         family: posthoc, source: score, score_sets: [c2_refit], key: blk_g, contrast: true,
         sampler: v2, noise: [noise_ffn_c0_c2],
         status: {records: [data/checkpoints/i2_posthoc/c2/fit_record.json],
                  allowed: [not_matched]},
         plans: [minimal, recommended]}
      - {row_id: se_snap, group: se, label: "Snapshot ensemble (C0)", kind: sampling,
         family: ensemble, source: score, score_sets: [c0_snapshot], key: blk_g,
         contrast: true, sampler: ensemble, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: se_nll, group: se, label: "Sequence NLL (C0)", kind: onepass, family: onepass,
         source: score, score_sets: [c0], key: blk_nll, contrast: false, sampler: none,
         noise: [], status: null, plans: [minimal, recommended]}
      - {row_id: se_ent, group: se, label: "Token entropy (C0)", kind: onepass,
         family: onepass, source: score, score_sets: [c0], key: blk_tu, contrast: false,
         sampler: none, noise: [], status: null, plans: [minimal, recommended]}
      - {row_id: se_msp, group: se, label: "1 − max-prob (C0)", kind: onepass,
         family: onepass, source: score, score_sets: [c0], key: blk_maxprob_unc,
         contrast: false, sampler: none, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: se_char, group: se, label: "LaTeX-character rate", kind: onepass,
         family: onepass, source: feature, score_sets: [surface], key: sf_tex,
         contrast: false, sampler: null, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: se_rmd, group: se, label: "Rel. Mahalanobis (C0)", kind: onepass,
         family: onepass, source: feature, score_sets: [onepass__c0], key: op_rmd,
         contrast: false, sampler: null, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: hn_c3, group: hn, label: "BLoB LoRA (C3)", kind: sampling,
         family: variational, source: score, score_sets: [c3], key: blk_g, contrast: true,
         sampler: variational, noise: [noise_lora_c3_blob], status: null,
         plans: [minimal, recommended]}
      - {row_id: hn_blob, group: hn, label: "BLoB LoRA (new seeds)", kind: sampling,
         family: variational, source: score, score_sets: [blob_1, blob_2, blob_3],
         key: blk_g, contrast: true, sampler: variational, noise: [], status: null,
         plans: [recommended]}
      - {row_id: hn_tfb, group: hn, label: "BLoB-mean + TFB", kind: sampling,
         family: posthoc, source: score, score_sets: [c4_tfb_fixed], key: blk_g,
         contrast: true, sampler: v2, noise: [noise_lora_c3_tfb],
         status: {records: [data/checkpoints/i2_posthoc/c4_tfb/fit_record.json],
                  allowed: []},
         plans: [minimal]}                   # C2 alternative: [minimal, recommended]
      - {row_id: hn_lap, group: hn, label: "BLoB-mean + diag. Laplace", kind: sampling,
         family: posthoc, source: score, score_sets: [c4_lap_refit], key: blk_g,
         contrast: true, sampler: v2, noise: [noise_lora_c3_lap, noise_lora_c3_tfb],
         status: {records: [data/checkpoints/i2_posthoc/c4_lap/fit_record.json],
                  allowed: [not_matched]},
         plans: [minimal]}                   # C2 alternative: [minimal, recommended]
      - {row_id: hn_dtfb, group: hn, label: "Det. LoRA + TFB", kind: sampling,
         family: posthoc, source: score, score_sets: [det_tfb_1, det_tfb_2, det_tfb_3],
         key: blk_g, contrast: true, sampler: v2, noise: [],
         status: {records: [data/checkpoints/i2_posthoc/b2/tfb_s201/fit_record.json,
                            data/checkpoints/i2_posthoc/b2/tfb_s202/fit_record.json,
                            data/checkpoints/i2_posthoc/b2/tfb_s203/fit_record.json],
                  allowed: []},
         plans: [recommended]}
      - {row_id: hn_dlap, group: hn, label: "Det. LoRA + diag. Laplace", kind: sampling,
         family: posthoc, source: score, score_sets: [det_lap_1, det_lap_2, det_lap_3],
         key: blk_g, contrast: true, sampler: v2, noise: [],
         status: {records: [data/checkpoints/i2_posthoc/b2/lap_s201/fit_record.json,
                            data/checkpoints/i2_posthoc/b2/lap_s202/fit_record.json,
                            data/checkpoints/i2_posthoc/b2/lap_s203/fit_record.json],
                  allowed: [not_matched]},
         plans: [recommended]}
      - {row_id: hn_ens, group: hn, label: "LoRA ensemble (M=3)", kind: sampling,
         family: ensemble, source: score, score_sets: [lora_ens3], key: blk_g, contrast: true,
         sampler: ensemble, noise: [], status: null, plans: [recommended]}
      - {row_id: hn_nll, group: hn, label: "Sequence NLL (C3 mean)", kind: onepass,
         family: onepass, source: feature, score_sets: [onepass__c3_mean], key: op_nll,
         contrast: false, sampler: null, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: hn_ent, group: hn, label: "Token entropy (C3 mean)", kind: onepass,
         family: onepass, source: feature, score_sets: [onepass__c3_mean], key: op_ent,
         contrast: false, sampler: null, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: hn_msp, group: hn, label: "1 − max-prob (C3 mean)", kind: onepass,
         family: onepass, source: feature, score_sets: [onepass__c3_mean],
         key: op_maxprob_unc, contrast: false, sampler: null, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: hn_char, group: hn, label: "LaTeX-character rate", kind: onepass,
         family: onepass, source: feature, score_sets: [surface], key: sf_tex,
         contrast: false, sampler: null, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: hn_rmd, group: hn, label: "Rel. Mahalanobis (C3 mean)", kind: onepass,
         family: onepass, source: feature, score_sets: [onepass__c3_mean], key: op_rmd,
         contrast: false, sampler: null, noise: [], status: null,
         plans: [minimal, recommended]}
      - {row_id: hn_lr, group: hn, label: "NLL(C3) − NLL(C0)", kind: onepass,
         family: onepass, source: feature, score_sets: [onepass__c3_mean], key: op_lr,
         contrast: false, sampler: null, noise: [], status: null,
         plans: [minimal, recommended]}
    caption_template: >-
      Document-weighted AUROC for separating each OOD domain (panels) from the
      row's in-distribution test set (group headers). Filled circles: the
      realized-token Jensen gap $G$ with 95\% document-bootstrap intervals
      ({n_resamples} resamples). Hollow circles: block-averaged MI on the same
      blocks. Squares: one-pass scores. Diamonds: Gaussian weight noise matched to
      the method's held-out $\Delta$NLL, scored on the first block of each
      document. {seed_sentence}{subset_sentence}{n_docs} test documents per
      domain. {samples_sentence} Intervals are marginal; paired tests are in
      Table~\ref{{tab:detection}}. {not_scored_sentence}{prereg_sentence}
  style:                                     # free to change; bounded by S4-T4
    width_in: 5.5                            # NeurIPS \textwidth
    max_height_in: 4.5
    row_pitch_in: 0.15
    slot_pitch_in: 0.05                      # Section 5.3; at most 3 slots per row
    top_in: 0.3
    bottom_in: 0.6
    font_family: STIXGeneral                 # bundled with matplotlib; Times-like
    mathtext_fontset: stix
    font_size_pt: 7
    title_size_pt: 8
    min_font_pt: 7
    x_tick_step: 0.1
    marker_size_pt: 3.0                      # below 72 x slot_pitch_in = 3.6 pt
    line_width_pt: 0.8
    chance_line: {color: "#999999", style: "--", width_pt: 0.6}
    separator: {color: "#DDDDDD", width_pt: 0.4}
    colors: {variational: "#0072B2", posthoc: "#D55E00", dropout: "#009E73",
             ensemble: "#CC79A7", onepass: "#000000"}
    okabe_ito: ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
                "#D55E00", "#CC79A7"]
    glyphs:
      primary: {marker: "o", fill: full}
      contrast: {marker: "o", fill: none}
      onepass: {marker: "s", fill: full}
      noise: {marker: "D", fill: none}
    png_dpi: 300
    pdf_fonttype: 42
```

Notes on the block:

- The freeze hashes `design` only. `style` may change after results, because it cannot move a number and S4-T4 bounds it.
- `sampler` is compared with `meta.sampler` of every S1-format file of the row. `null` means a feature file, which has no sampler. `none` is C0's label in `configs/i2_eval.yaml`.
- `noise` lists candidate control score sets in order; the first with files on both eval sets is used. `[]` means no control is planned.
- `status.records` align with `score_sets`. A file may be missing only if its record gives a status in `allowed`. The Laplace records give `match_status`. The TFB records have none, so `allowed: []` and a missing file stops the build. The build also stops on a record with `exit_code` other than 0 (for example `bisection_failed`), and on a record whose `state_sha256` is not in the score file's `meta.checkpoint_sha256`.
- Every "(prov.)" name in Section 3 comes from the S3 or S5 draft of 2026-09-26. The S4 approver updates this block if those drafts change before approval (Section 3).

### 5.6 Test fixture

- S1 has no score-file writer function: the scorer builds the payload inline (scripts/eval_c_checkpoints.py:762-778, 1094-1117). So the test module writes the files itself, with the keys S4 reads, as S1's own checker tests do (tests/test_check_eval_rebuild.py:326-377). Each S1-format file holds all six `blk_*` arrays, because `run_analyze` loads all six (scripts/eval_c_checkpoints.py:1199-1210). Per-token arrays are not needed. A drift in S1's real format is caught by S4's loader, which raises KeyError on a missing key, and by S4-T3a on the real files.
- Save with `torch.save`. `meta` holds plain types only, so `torch.load(weights_only=True, mmap=True)` works, as S1 reads it (:1203). The fixture `prereg.json` comes from S1's `freeze_prereg` (minigpt/evalset.py:207).
- 5 domains in S1's order. 30 documents per domain, $k_d\sim\mathcal U\{1,2,3\}$, numpy `default_rng(0)`. `test` holds StackExchange (first, so block 0 is a StackExchange block) and the 3 OOD domains; `test_hn` holds HackerNews. Block scores are $\mathcal N(\mu_{\mathrm{set}}\,y, 1)$, with a fixed $\mu_{\mathrm{set}}$ per score set in the test module.
- `noise_ffn_c0_c2__test__first.pt` holds the first block of each document. Its stored `weight` keeps the full-set value, as a real subset file does. The feature files `surface__test.pt` and `surface__test_hn.pt` hold `sf_tex` and the index fields (S3's provisional format).
- Fit records: `c2` (`match_status: matched`, `exit_code: 0`, and a `state_sha256` that also appears in `c2_refit`'s `meta.checkpoint_sha256`); `c4_tfb` (`exit_code: 0`, `state_sha256`); `c4_lap` (`match_status: not_matched`, `exit_code: 0`, no state).
- The fixture eval YAML holds `eval_set.domains` (keys, roles, `stripped_copy`), `eval_set.n_test_docs: 30`, `scoring.out_dir` (the temporary folder) and `scoring.score_sets` = {`test`: [c0, mc_dropout, c2_refit], `test_hn`: [blob_1, blob_2, blob_3, c4_tfb_fixed], `arxiv_stripped`: []}. Its `analysis` block has S1's keys with 200 resamples, seed 0 and level 0.95, fixture families, and an empty `descriptive.replication`.
- The fixture rows (16 CSV lines per panel):

| Row | Files | Lines per panel |
|---|---|---|
| `se_mcd` | `mc_dropout` (G, M) | 2 |
| `se_c2` | `c2_refit` (G, M); noise `noise_ffn_c0_c2` on first blocks | 3 |
| `se_nll` | `c0`, `blk_nll` | 1 |
| `hn_blob` | `blob_1`, `blob_2`, `blob_3` (G, M each) | 6 |
| `hn_tfb` | `c4_tfb_fixed` (G, M) | 2 |
| `hn_lap` | none; the `c4_lap` record says `not_matched` | 1 |
| `hn_char` | `surface`, `sf_tex` | 1 |

- The S4-T3a reference tables come from S1's own `run_analyze(cfg)`, imported with importlib from `scripts/eval_c_checkpoints.py` as tests/test_eval_scripts.py:219 does. It is S1 code, so the check is not circular. It covers the primary and contrast lines of the S1-format rows: 13 per panel, 39 in all. The noise and feature lines have no S1 table.

### 5.7 Command-line flags of `scripts/generate_figures.py`

| Flag | Does |
|---|---|
| `--result` | Selects the result-figure path. Without it, the script behaves as today. |
| `--config PATH` | Required with `--result` |
| `--from-csv PATH` | Stage 2 only |
| `--freeze-design --spec-commit SHA` | Checks the YAML's `design` against the spec at that commit, then writes `figure_prereg.json`. Refuses to overwrite an existing file. |
| `--post-hoc TEXT` | Allows a design hash that differs from the freeze. Records the text. |
| `--check-layout` | S4-T4a-g on the saved outputs |
| `--check-tables` | S4-T3a |
| `--check-tex PATH` | S4-T3b-c |
| `--verify` | S4-T5c |
| `--make-quiz`, `--grade-quiz PATH` | S4-T4h |

### 5.8 matplotlib rules (from the probes in Section 6.2)

- `scripts/generate_figures.py:39-50` changes the global `rcParams` at import, including `savefig.bbox: tight`, which changes the figure size. So the result path runs inside `with matplotlib.rc_context():`, calls `matplotlib.rcdefaults()` first, then applies the YAML style. This also makes the CLI build and the direct pytest call give the same bytes (S4-T5a).
- Call `savefig` inside that same context. Saved outside it, the PDF gets Type 3 fonts even when `pdf.fonttype` was 42 while the figure was drawn (probe).
- Use `layout="constrained"` and `bbox_inches=None`, so the file size equals `figsize`.
- PDF: `metadata={"CreationDate": None}`, so two builds give the same bytes. PNG: `dpi` from the YAML.
- Font: `STIXGeneral` ships with matplotlib, so the bytes do not depend on system fonts. Escape `$` in labels. Use U+2212 for the minus sign in "1 − max-prob" and "NLL(C3) − NLL(C0)".

### 5.9 Paper-table check (`--check-tex`)

- Find the `table` environment whose `\label` equals `checks.tex_table_label`. Read each line that ends with `% fig-row: <row_id>`.
- In that line, the cells for the columns in `checks.tex_columns` are matched by
  `(\d?\.\d{3})\s*(?:\\,)?\s*\[\s*(\d?\.\d{3})\s*,\s*(?:\\,)?\s*(\d?\.\d{3})\s*\]`, which accepts the current paper style `.917\,[.900,\,.933]` (paper.tex:199-205) and `0.917 [0.900, 0.933]`.
- A number passes if $\lvert x_{\mathrm{tex}}-x_{\mathrm{csv}}\rvert\le 5\times10^{-4}+10^{-9}$, with the CSV's primary line for that row and column. A multi-seed row is checked against the seed the marker names, as `% fig-row: hn_dtfb seed=1`. A row that shows a mean over seeds is marked `seed=mean`. It shares no number with the CSV, so the check lists it and skips it.
- Placement (S4-T3c): the first `\includegraphics` of `checks.tex_figure_file` must sit in a `figure` environment with `width=\textwidth`, before the first result table. A result table is a `table` environment whose body matches the cell pattern above. Today paper.tex:67-78 holds a `table` with no numbers (the 2×2 method matrix) before any figure. A plain "first `\begin{table}`" rule would fail on it even with the figure placed right.

### 5.10 Quiz (S4-T4h)

`--make-quiz` writes `questions.json` and a separate `answers.json` under `quiz_dir`. It picks questions with `default_rng(quiz.seed)` from these templates. It keeps a question only if its answer is clear by at least `min_gap` = 0.03 AUROC, about 1.6 mm at print size:

| Template | Question | Eligible when |
|---|---|---|
| Q-a | In panel $P$, which row has the rightmost filled marker? | Top point minus second point $\ge 0.03$ |
| Q-b | In panel $P$, does the whisker of row $R$ cross the chance line? | $\min(\lvert\ell-0.5\rvert,\lvert u-0.5\rvert)\ge0.03$ |
| Q-c | In panel $P$, is the filled marker of multi-sample row $R_1$ right of the square of one-pass row $R_2$ (same group)? | $\lvert a_1-a_2\rvert\ge0.03$ |
| Q-d | Which ID set does row $R$ use? | always |
| Q-e | In panel $P$, is the hollow circle of row $R$ left or right of its filled marker? | $\lvert a_G-a_M\rvert\ge0.03$ |

One question per template, in this order. If a template has no eligible case, the next template supplies a second question. For a multi-seed row, Q-b, Q-c and Q-e use its first scored seed and say so. Answers come from a closed set (row labels, yes/no, left/right, ID names). Grading is an exact match after case folding and stripping. The reader agent gets the PNG, the caption stub and `questions.json`, and nothing else.

## 6. Verification evidence

### 6.1 Session-02 refuter verdicts that shape the figure

No session-02 refuter examined P12. Three confirmed verdicts decide what the figure may show. Their file:line evidence is in S1 section 6.1 and S2 section 6.

| Verdict | Evidence (from S1 and S2) | What S4 does with it |
|---|---|---|
| P1/P19 confirmed: the old ID set was StackExchange only, the OOD set was 9 arXiv papers, and FreeLaw and PubMed were never scored | scripts/eval_c_checkpoints.py:96-109 (pre-S1 line numbers); minigpt/data.py:202, 207-213; 9 papers, 2 of which gave 289 of 500 windows | Three panels, one per OOD domain. Each row uses its own ID set (S1 D5). No old D1 number is drawn. |
| P4 confirmed: diagonal Laplace was unscaled, with std 1.00 on weights of median size 0.023-0.025 | minigpt/laplace.py:115-123, 167-168; configs/c2.yaml:79-81 | The Laplace rows come only from S2's refits (`sampler: v2`, checked in S4-T5d). The old 0.536 and 0.494 are never drawn. A `not_matched` row stays in place as "not scored". |
| P5 confirmed: the TFB sampler skips the SVD rotation, and C4-TFB starts from the BLoB mean | minigpt/tfb.py:73-74, 188-192; max abs(a_map − lora_A_mu) = 0 on all 32 layers | Pre-fix C4-TFB is excluded. The fixed row is labelled "BLoB-mean + TFB". |

### 6.2 Drafting and critic passes: checks and probes (CPU only, read-only)

The drafting probes sit in the session scratchpad under `s04/` (`s4_probe_plot.py`, `s4_probe_fonttype.py`, `s4_probe_boot.py`). The critic probe is `s4c/probe_boot_built.py`. The scratchpad is temporary, so every number a test needs is copied here.

| Check | Result | Evidence |
|---|---|---|
| P12 as stated | The paper has 3 figures, all PNG schematics: point vs Bayesian weights, the method matrix and the BLoB diagram. The first result table (`tab:ood`) starts at line 187. | paper/paper.tex:82-86, 98-102, 146-150, 187 |
| A number-free table comes first | `\begin{table}[h]` at line 67 holds the 2×2 method matrix, with no numbers | paper/paper.tex:67-78 |
| The old data figures hardcode numbers | fig5 types in the 7 old AUROCs and CIs. fig6 types in Table 4 and prints "N=3: the knee (97% of full signal)". fig7 types in the 4L and 16L MI ratios, which include the hardcoded ratios of P11. fig4 plots unseeded `np.random.exponential` values as a "Fisher". fig5 also prints "CIs overlap". | scripts/generate_figures.py:287, 363-371, 396, 411-415, 431, 458-459 |
| Global style leaks | `savefig.bbox: "tight"` and a serif font are set at import | scripts/generate_figures.py:39-50 (line 48) |
| No test covers the figure script | 0 hits for `generate_figures` in `tests/` | grep |
| Where figures live | The script writes `figures/` (line 20). The paper reads `../figures/` (paper.tex:21). `paper/figures/` exists and is empty. AGENTS.md:113 and agents/technical-reference.md:50 say "3 PNG figures in `paper/figures/`". | ls; grep |
| Print width | NeurIPS 2024: `textwidth=5.5in`, `textheight=9in`. `paper/neurips_2024.sty` is 0 bytes; the build skill bundles its own copy. | agents/skills/build-latex-pdf/references/neurips_2024.sty:78-79 |
| The paper's number style | `.917\,[.900,\,.933]` | paper/paper.tex:199-205 |
| The build skill deletes `.aux` | So the figure's page number cannot be read after a build. S4-T3c checks source order instead. | agents/skills/build-latex-pdf/skill.md:55 |
| Libraries | matplotlib 3.10.8 and scikit-learn 1.8.0 in the uv environment, pulled in by mlflow; not in `pyproject.toml`. Global Python has matplotlib 3.10.9. | uv.lock:1367, 2723, 1451, 1457; pyproject.toml |
| Byte-identical builds | Two builds in one process: identical PDF and PNG sha256, with `CreationDate: None` | `s4_probe_plot.py` |
| Exact size | 5.5 × 3.9 in at 300 dpi gives a 1,650 × 1,170 px PNG | `s4_probe_plot.py` |
| Room for the rows | 21 rows of 24-character labels, 3 panels, 7-pt text: 0 overlapping text pairs, smallest font 7.0 pt, 0 raster images in the PDF. No legend or group headers were drawn. | `s4_probe_plot.py` |
| Type 3 trap | `pdf.fonttype: 42` set while drawing, but `savefig` called outside the `rc_context`: Type 3 fonts. `savefig` inside: TrueType (`CIDFontType2`), same bytes on a repeat. | `s4_probe_fonttype.py` |
| Bootstrap speed, built function | `doc_bootstrap_auroc` on 2,040 ID and 2,021 OOD blocks (1,000 documents each), 10,000 resamples: 3.00 s with one score, 3.51 s with two. The CI of the first score is bit-identical in both calls. The weighted point `auroc` equals the function's own point value exactly, in 1.3 ms. | `s4c/probe_boot_built.py`; minigpt/uncertainty.py:593-669 |
| Score-file keys and meta | As in the Section 2 interface table | scripts/eval_c_checkpoints.py:762-778, 1094-1117 |
| No score-file writer function | `minigpt/evalset.py` has none (S1 section 2 planned one). S1's checker tests write files by hand. | minigpt/evalset.py (function list); tests/test_check_eval_rebuild.py:326-377 |
| Cell weights are recounted | float64 $1/k_d$ per class over the cell; a subset file keeps full-set float32 weights | scripts/eval_c_checkpoints.py:1230-1232, 193-204, 274 |
| `test_hn` holds HackerNews only | `ROLE_EVAL_SET` maps `id_adapter` to `test_hn` and `ood` to `test` | minigpt/evalset.py:46, 153-161 |
| The TFB fit record has no `match_status` | `_run_tfb` writes `exit_code` and `failures`; only `_run_laplace` writes `match_status`. The scorer checks `match_status` for Laplace only. | minigpt/posthoc_refit.py:1034-1064, 1150-1191; scripts/eval_c_checkpoints.py:555-557 |
| No test score exists yet | `data/scores_i2/`, `data/eval_i2/` and `data/checkpoints/i2_posthoc/` are absent on 2026-09-26 | ls |
| What the reviewers asked for | A dot-and-whisker teaser from the score files, per-domain AUROC with the one-pass baselines, vector PDF; "never use fig4"; overprinted labels in fig5 | research/2026-09/paper-critique.md:76 (#25) and Section 5 "Figures" |
| Marginal CIs are not a test | "Overlapping CIs means not significant is also the wrong test" | research/2026-09/paper-critique.md:69 (#18) |

### 6.3 Differences from the research documents and the sibling drafts

| # | Source | This spec | Why |
|---|---|---|---|
| 1 | paper-critique Fig. 1: rows grouped by extra cost (none / minutes / hours), MI on the filled dot, entropy or max-prob as a hollow marker | Rows grouped by ID set. $G$ on the filled dot, block MI hollow. One-pass scores as their own rows with CIs. | S1 made $G$ primary and gave each method its own ID set. A cost grouping would mix two ID sets in one group. Cost goes in S3's ms-per-block table. |
| 2 | paper-critique Figs. 1 and 2 as two figures | One figure | roadmap S4 and m10 cost one figure |
| 3 | paper-critique Fig. 3 (AUROC against k tokens), N sweep | Not in S4 | Not in m10. See Section 7. |
| 4 | options m10: "next to the baselines" | Every S3 baseline on its ID set, plus noise controls in their method's row, on first blocks | S3-T2 flags overlap with the noise control. S3 scores controls on first blocks, so the caption says the diamond is not on the filled circle's blocks. |
| 5 | S3 draft, "Feeds": "S4 (figure rows: baselines, surface model, noise controls, combined-test panel)" | No surface-model row and no combined-test panel by default. C3 adds a combined row per group. | A fitted model is not a score of the eval set (C3). A fourth panel is outside m10's "one panel per OOD domain". Handoff in Section 7. |
| 6 | S5 draft, C1 default: 3 new BLoB seeds, C3 separate | Row `hn_blob` next to row `hn_c3` | Revision 0 pooled C3 with 2 new seeds, which is S5's alternative, not its default |

### 6.4 Critic pass (revision 1): what changed

| # | Finding | Change |
|---|---|---|
| 1 | S4-T4h and the real parts of S4-T1, S4-T3 and S4-T5 named no script | Every "Runs in" names `scripts/generate_figures.py --result` with its flag |
| 2 | S4-T3a had no fixed count on the fixture | "exactly 39 lines compared, 0 mismatches" |
| 3 | The TFB fit record has no `match_status` (Section 6.2) | Interface row fixed; `status.allowed` is empty for TFB rows; `fit_status` column |
| 4 | S1 has no score-file writer function to import | Fixture writes the files by hand, as S1's checker tests do; the drift guard is the loader and S4-T3a |
| 5 | S1 recounts float64 weights per cell; the stored float32 `weight` differs, and keeps the full-set value in subset files | Section 5.2 rule |
| 6 | The CPU cost came from a separate probe, not the built function | Measured the built function: 3-6 min per build, 1-2 min batched |
| 7 | S3 draft names differ: `c0_snapshot`, feature files with `op_*` and `sf_*` keys, 5 noise controls on run tag `first` | Rows, interfaces, refusals and CSV columns follow the S3 draft; first-block alignment rule |
| 8 | S5 draft names differ: `blob_*`, `det_tfb_*`, `det_lap_*`, `lora_ens3`; C3 is a separate row; files may be first-block subsets | New row `hn_blob`; per-seed status records; `block_subset` column |
| 9 | Seed markers at $\pm0.2$ of a 0.15-in row (2.16 pt) overlapped 3.5-pt markers; S4-T4 checked text only | Slot rule in Section 5.3; marker overlap added to S4-T4c |
| 10 | S4-T3c ("before the first `table`") fails on the number-free table at paper.tex:67 | "before the first result table"; two new fixture files |
| 11 | The freeze file could only be written at build time (Oct 5 or 12), after the Oct 2 test scores | The approval commit is the freeze, read by `--freeze-design --spec-commit`; full row list in Section 5.5; `design_after_test_scores` flag (S3's mechanism) |
| 12 | Review hours used 0.25-0.5 h, below CA1's low end; engineering parts summed to 4.75-7.0 h, not 4.5-7 h; C4's saving counted the whole table-check line | Cost tables rebuilt part by part; review 0.75-1.25 h; C4 saves 0.75-1.25 h |
| 13 | Label "log p(C3) − log p(C0)" had the opposite sign of S3's `op_lr` ($\mathrm{NLL}_c-\mathrm{NLL}_{\mathrm{bg}}$) | Label "NLL(C3) − NLL(C0)" |
| 14 | Outside m10/P12: the builder was to correct agents/technical-reference.md:50 (paper figure location) | Moved to the handoff, next to AGENTS.md:113 |
| 15 | S4-T1f listed `c4_tfb` only | Also `c2` and `c4_lap` (`design.excluded_score_sets`) |
| 16 | Fixed tick list [0.5, ..., 1.0] leaves no tick below 0.5 when $x_{\min}<0.5$ | `x_tick_step` |
| 17 | S4-T5e: 30 s is tight with 2 fresh processes that import torch and matplotlib | 60 s; the processes use `sys.executable` |

## 7. What this does not cover

- The paper text: the final caption wording, the placement of the figure and the main table. m11 writes them. S4 only checks them (S4-T3b-c).
- Other figures: AUROC against the number of tokens (paper-critique #20), the N sweep (#32), MI histograms, an ID-vs-ID row, a markup-stripped arXiv panel (S3-T4 reports it in a table), a combined-test panel and a cost axis.
- A marker for the method on first blocks next to its noise diamond. S3's `noise_table.json` holds that comparison.
- FPR@95, TPR at 5% FPR and AUPRC. They go in the tables.
- Paired tests, Holm-adjusted p-values and significance marks. S1, S3 and S5 own them. The figure shows marginal intervals only, and the caption says so.
- The LoRA rows on StackExchange (S1 families F2 and F4, S5 family F6). Appendix table.
- Whether the results are good. S4 has no result endpoint.
- Byte identity across operating systems or matplotlib versions. The CSV is the portable artefact. A redraw elsewhere may differ in bytes but not in data.
- Colour-vision testing beyond the rule that shape and fill carry every glyph kind.
- The final S3 and S5 names and file formats. They are provisional until those specs are approved and built.
- The old figures fig1-fig7 and their PNG files. See the handoff below.

Handoff to m11, m12, m13 and the sibling specs. S4 does not edit these.

| Text or file | Where | Why |
|---|---|---|
| fig5 hardcodes the withdrawn AUROCs and CIs and prints "CIs overlap" | scripts/generate_figures.py:363-371, 396; figures/fig5_auroc_bars.png | P12, P21: old numbers in a public file |
| fig6 "N=3: the knee (97% of full signal)" | scripts/generate_figures.py:411-415, 431; figures/fig6_n_vs_auroc.png | paper-critique #7 |
| fig7 MI ratios and "scaling inversion" | scripts/generate_figures.py:458-459; figures/fig7_scaling_inversion.png | P8, P11 |
| fig4 plots unseeded synthetic "Fisher" values | scripts/generate_figures.py:287; figures/fig4_tfb_vs_laplace.png | paper-critique #25: never use it |
| Paper Figs. 1-3 and the intro 2×2 table are decorative | paper.tex:67-78, 84, 100, 148 | paper-critique Section 6: cut Figs. 1 and 2 and the intro table; keep Fig. 3 only if redrawn |
| "3 PNG figures in `paper/figures/`" | AGENTS.md:113; agents/technical-reference.md:50 | The paper reads `figures/` (paper.tex:21). The Paper Publishing section and the technical reference change when the paper status changes (m11, m12). Not P12. |
| The main table needs `% fig-row:` markers and `AUROC [lo, hi]` cells | m11 | S4-T3b |
| S3's "Feeds" line names a surface-model row and a combined-test panel for S4 | specs/i2-baselines-analyses.md:16 | S4 draws neither by default (Section 6.3, row 5). The S3 critic should align the line. |
| S3's and S5's table schemas | S3 Sections 5.2-5.5; S5 Section 5.5 | S4-T3a needs a reader per table. Step 6 of Section 5.1 adds them once the schemas are built. |

## 8. Confidence

| Claim | Confidence | Why |
|---|---|---|
| The figure fits 5.5 in × at most 4.5 in with 18-20 rows, 2 headers and 7-pt text, with no text or marker overlap | medium-high | The probe drew 21 rows × 3 panels with 0 text overlaps, but without the legend and headers. The real labels run to 28 characters with the seed suffix, against 24 in the probe. The slot rule removes marker overlap by construction. C2 reaches 4.5 in exactly. |
| Two builds in the same uv environment give the same bytes | high | Probe: identical PDF and PNG sha256. The Type 3 and bbox traps are known and written into Section 5.8. |
| The CSV equals S1's tables to $10^{-6}$ | high | Same function, same seed rule, and the weight and block-order rules copied from the built `_CellData.cell`. The probe shows that the CI does not depend on the other scores in a call. |
| A full build takes 3-6 CPU min | medium-high | Measured on the built function with synthetic data of the planned size. The block counts are S1's estimates until m4 runs. |
| Engineering fits 5.0-6.75 h | medium-low | Built up part by part in Section 1. S3's and S5's formats may change after their critic passes, which adds rework. |
| The S3 and S5 names hold | low-medium | Both drafts are in critic passes on 2026-09-26. The interface table, the approval step and the amendment rule handle a rename. |
| The quiz catches an unreadable figure | medium-low | One reader and 5 questions. It complements the layout checks, which catch overlap and small text but not a confusing design. |
| P12 is closed | medium | The figure exists and is data-driven once S4 passes. "First result on page 1-2" depends on m11, which S4-T3c checks only by source order. |
| The design is frozen before any test score is read | high at a Thu Oct 1 approval; recorded, not hidden, at a later one | The approval commit is the freeze, and no score file exists on 2026-09-26. `design_after_test_scores` records a late approval, and the caption states it. |
