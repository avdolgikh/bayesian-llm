# S3 Baselines and analyses: one-pass baselines, noise control, combined-score test (`specs/i2-baselines-analyses.md`)

Status: DRAFT (not approved). Date: 2026-09-26 (revision 1, after the session-03 critic pass; Section 6.5 lists what changed). Owner: Alexey approves; agents implement.
Plan of record: `agents/plans/roadmap.md` Section 3, row S3. Template: the session-02 specs (S0-S2).

## 1. Header

| Field | Value |
|---|---|
| Items | **m8**: one-pass baselines (surface character rates, sequence NLL, token entropy, max-prob, Mahalanobis and relative Mahalanobis distance, a likelihood ratio), a C0 snapshot ensemble, ΔNLL-matched isotropic noise controls, and a table of AUROC per millisecond at N=3 and N=20. **m9**: the combined-score test (cross-validated one-pass scores alone vs plus the weight-sampling score), AUROC in matched bins, MI vs expected-entropy correlation, per-method MI scale and a noise floor for BLoB, calibration with CIs. |
| Issues | **P2** (surface features solve the benchmark), **P3** (no cheap baselines), **P13** ("epistemic" undefined: no MI vs AU correlation, no length control), **P15** (calibration without CIs), **P22** (no AUROC-per-millisecond comparison), **P23** (score definition, MI scale, no BLoB noise floor). S1 hands over two items. The length control (S1 Section 7) is in scope under P13. The method-vs-method paired tests (S1 Sections 3 and 7) answer P6 and P14, not these issues, so they are out of S3 by default (Choice 4). |
| Approve by | Thu 2026-10-01 (roadmap Section 3, row S3). The approval commit is the pre-registration (Section 3). It should land before S1's m4 re-score starts; choice 1 in Section 2 covers a later approval. |
| Build | Session 04, Mon 2026-10-05 (roadmap key dates) |
| Run | Night A, Mon Oct 5, after S2's jobs: first the baselines and diagnostics, then the noise controls of the S1 rows. Night B, Wed Oct 7, after S5's night-2 re-score: the noise controls of the S2 rows. CPU analysis of F5-F7 and T1, T3-T5 on Tue Oct 6; F8 on Thu Oct 8. Alexey reviews Oct 7-10, before G1 (Sun Oct 11). |
| Depends on | S1 (`specs/i2-eval-rebuild.md`): eval set, score files, scorer, document bootstrap, Holm, freeze rule. S2 (`specs/i2-posthoc-fixes.md`): fit records, fit-set builder, base loader, λ sweep, v2 Laplace sampler, the three S2 score sets. S0 (`specs/i2-timing-probe.md`): batch sizes and batch-1 rates. Interface tables in Section 2 are checked against the code in the working tree on 2026-09-26. |
| Feeds | S4 (figure rows: baselines, surface model, noise controls, combined-test panel). m11 (which pre-written sentence in Section 3.4 applies). S5 (the `ensemble` sampler, and S3's code for S5's own rows under S5's own frozen families). m13 (public numbers). |

Costs. The roadmap column is roadmap Section 3, row S3 (m8 plus m9 from options-and-costs.md Section 4.1). The next column is this spec's own estimate.

| Cost | Roadmap row S3 | This spec | Difference and why | Assumption |
|---|---|---|---|---|
| Alexey, approve | in m2 | 12-18 min (in m2) | none | CA1 |
| Alexey, review | 1.25-1.75 h (m8 0.75-1; m9 0.5-0.75) | 1.5-3.5 h (CA1: 1 h per 6-9 engineering h on 12.5-21.5 h, in 0.25 h steps) | +0.25-1.75 h. The roadmap figure holds only while engineering stays at or below 15.75 h (1.75 × 9). Choice 4's default keeps 54 method-pair cells out of the review. | CA1 |
| Engineering (agents) | 11-22 h (m8 8-16; m9 3-6) | 12.5-21.5 h (m8 8-13; m9 4.5-8.5) (est.) | +1.5 h at the low end: an independent re-derivation of the primary families (1-1.5 h) and the freeze-from-spec check (0.5 h). −0.5 h at the top: the method-pair families, the N sweep and the prefix curve are out of scope (Section 7). | CA2 |
| GPU (RTX 4070) | 0.5-1.5 GPU h (m8) | 1.6-7.8 GPU h (est.). Low end: a 3x batching gain. High end: CA10 batch-1 rates. The baselines and diagnostics alone take 0.3-0.7. | +1.1-6.3 h, all from the noise controls (1.4-7.1 GPU h). The roadmap priced the noise control like one S2 refit. The spec has 5-6 controls. Each runs S2's λ sweep (12-19 λ-equivalents × 20 seeds × 640 blocks) and scores 4,000 first blocks at N=20. | CA3, CA10, S0 |
| CPU | not listed | about 1 h (cross-validation fits over 5 fold seeds, bootstraps) (est.) | analysis only | est. |
| Opus agents | not listed | 6-8 | none against CA6 (5-8 per implementation spec) | CA6 |
| Cloud | $0 | $0 | none | |

GPU breakdown (est.). Batch-1 rates come from CA10 and report.md: 8.1 ms per forward pass of a C0-sized model, 14.8 ms for C3 mean weights (LoRA not merged), 14.35 ms per pass with a 33.5M-entry weight draw (C1's 287 ms / 20; S2 uses the same rate for C2), and 15.8 ms per BLoB pass (316 ms / 20). The 3x column divides every row except the timing pass by 3, which is the batching gain m4 assumes at its low end. S0 measures the real gain. The λ sweeps and the native ΔNLL already run at the fit batch size 32 (S2's YAMLs), so their batch-1 figures are upper bounds. The first-block rows get the gain only because `--first-block-only` batches consecutive rows of the subset (Section 5.5). Under S1's gap split they would run near batch 1, which adds 1.2-1.4 GPU h.

| Part | Passes | GPU h at batch-1 rates | GPU h at 3x | Night |
|---|---|---|---|---|
| One-pass: C0, C1 mean and C0 step 10,000 (8.1 ms) and C3 mean (14.8 ms) on `test`, `test_hn` and `arxiv_stripped` (12,500-14,200 blocks) | 50,000-56,800 | 0.14-0.15 | 0.05 | A |
| Gaussian fits: 15,000 training windows per center, 3 centers | 45,000 | 0.13 | 0.04 | A |
| Snapshot ensemble: 5 members on `test` (8,000-9,200 blocks), 8.1 ms plus up to 3.6 ms per parameter swap | 40,000-46,000 | 0.09-0.15 | 0.03-0.05 | A |
| Identical-copies floor: 1,000 HackerNews first blocks × 20 copies of C3 mean (14.8 ms) | 20,000 | 0.08 | 0.03 | A |
| Timing pass: 64 blocks at batch 1 and 128 at $b^\star$, N=3 and N=20, about 12 score sets | small | 0.1-0.2 | 0.1-0.2 | A |
| **Baselines and diagnostics** | | **0.54-0.71** | **0.25-0.37** | A |
| Native ΔNLL of MC dropout (14.35 ms, CA10), C1 (14.35 ms) and C3 (15.8 ms) on S2's fit sets: 640 blocks × S=20; an SE doubling re-measures at S=40 | 38,400-115,200 | 0.16-0.47 | 0.05-0.16 | A |
| λ sweeps: 12-19 λ-equivalents × 20 seeds × 640 blocks per control. FFN controls (`mcd`, `c1`, `c2`) at 8.1 ms, 0.35-0.55 h each. LoRA controls (`c3`, `c4_tfb`, and `c4_lap` only if the sharing rule fails) at 14.8 ms, 0.63-1.0 h each. | 0.77-1.46 M | 2.31-4.65 | 0.77-1.55 | A: mcd, c1, c3. B: c2, c4 |
| Noise scoring: 4,000 first blocks × N=20 per control. FFN at 14.35 ms (0.32 h each); LoRA at 15.8 ms (0.35 h each). | 80,000 per control | 1.66-2.01 | 0.55-0.67 | A: mcd, c1, c3. B: c2, c4 |
| **Noise controls** | | **4.13-7.13** | **1.38-2.38** | A, B |
| **Total** | | **4.7-7.8** | **1.6-2.8** | Night A 3.0-4.3 at batch-1 rates (1.1-1.6 at 3x); night B 1.7-3.6 (0.6-1.2 at 3x) |

Nights. S2 plans 2.2-8.2 GPU h for Mon Oct 5. Night A adds 1.1-4.3 GPU h, so that night holds 3.3-12.5 GPU h. One night gives 8-10 (CA3). The night-A queue runs S2 first, then the S3 baselines and diagnostics (0.3-0.7 h), then the S1-row noise controls. A control that has not finished by morning moves to night B. Night B holds the S2-row controls (0.6-3.6 h). It runs after S5's night-2 re-score on Wed Oct 7, because S5's night 1 (Tue Oct 6) already holds 3.6-10.35 GPU h. Only F8 (secondary) waits for night B. The primary families F5-F6 need only the S1 and S2 score files and the night-A feature files. At the top of every range, the last controls run on Thu Oct 8 and F8 is analysed on Fri Oct 9. G1 on Sun Oct 11 still holds.

## 2. Scope and non-scope

What changes:

| File | Change |
|---|---|
| `minigpt/baselines.py` (new, about 330 lines, est.) | Surface character rates. Pooled hidden state. Gaussian fit, Mahalanobis distance (MD) and relative Mahalanobis distance (RMD). Document folds. Weighted standardization. Out-of-fold logistic scores. Combined-score test and verdicts. Matched bins and stratified AUROC with a document bootstrap. Spearman, weighted ECE, NLL and Brier with a document bootstrap. $G$ from the first $N$ samples. |
| `minigpt/noise_control.py` (new, about 170 lines, est.) | A zero-curvature v2 Laplace state (isotropic noise). Parameter subsets `ffn`, `ffn_mu` and `lora_a`. A `vi_mean` base loader (C1 with `_use_mean` set on every Bayesian layer). Native ΔNLL for dropout, VI and BLoB samplers, with S2's SE rule. Noise targets from S2 fit records. The noise-YAML key check. The C4 sharing rule (Section 5.5). |
| `scripts/score_onepass.py` (new, about 200 lines) | `--part provenance`, `--part gauss`, `--part onepass`, `--part timing`, and `--check align`. Writes the feature files (Section 5.3) to `data/scores_i2/s3/`. The timing pass calls S1's `score_blocks` directly and writes no score file. |
| `scripts/fit_noise_controls.py` (new, about 80 lines) | `--config configs/i2_noise_<name>.yaml`. Measures the native target if needed, checks $\ell_0$ against S2, then calls S2's `match_prior_precision` and `write_fit_record`. Exit codes in Section 5.5. |
| `scripts/analyze_baselines.py` (new, about 250 lines) | `--freeze --spec-commit <sha>`, `--baselines`, `--families`, `--noise`, `--bins`, `--diagnostics`, `--latency`, `--check all` |
| `scripts/check_baselines.py` (new, about 120 lines) | `--rederive` for families F5 and F6. Written by a separate agent from Sections 3 and 5.4 of this spec and S1 Section 5.6 only. It imports neither `minigpt.baselines`, `minigpt.uncertainty` nor `analyze_baselines`. It may import helpers from S1's `scripts/check_eval_rebuild.py`, which is already independent of `minigpt`. |
| `configs/i2_baselines.yaml` (new) | Section 5.2. Its `analysis` block is the registration. |
| `configs/i2_noise_{mcd,c1,c2,c3,c4_tfb}.yaml` (new), plus `i2_noise_c4_lap.yaml` only if the sharing rule fails | Section 5.2, noise YAMLs |
| `scripts/eval_c_checkpoints.py` (S1's scorer) | New score sets through `register_score_set`: `c0_snapshot` (sampler `ensemble`), `c3_mean_copies` (sampler `identical`, method `deterministic` at N=20) and `noise_*` (sampler `isotropic`; method `laplace_v2` on a zero-curvature v2 state). A new `ensemble` sampler through `register_param_sampler`, shared with S5's `lora_ens3`. A flag pair `--first-block-only --domains <list>` (Section 5.5). A refusal of run tag `main` for every S3 score set. The existing score sets, flags and batching do not change. |
| `configs/i2_eval.yaml` (S1's file), `scoring` section only | `n_samples` and `labels` (`display`, `sampler`, `adapter_source`) for the new sets. The sets are added to `score_sets.test` (`c0_snapshot`, `noise_*`) and `score_sets.test_hn` (`c3_mean_copies`, `noise_lora_*`). `batch_size` stays keyed by eval set; S3 passes `--batch-size` per run. The frozen `analysis` block is not touched. |
| `tests/test_baselines.py`, `tests/test_noise_control.py`, `tests/test_analyze_baselines.py` (new) | The pytest parts of T1-T5 (CPU, CI), named in Section 4 |
| AGENTS.md, README.md, `agents/technical-reference.md`, `agents/design-rationale.md` | Doc updates under the rules below |

What must not change:

- S1's frozen `analysis` block, `data/scores_i2/prereg.json`, and S1's families F1-F4.
- S1's tables. S3 lists its score sets in `scoring.score_sets`, and S1's `--analyze` and `check_eval_rebuild.py --rederive` plan their cells from that list. Both read only `main` score files, so S3 never writes a `main` file: `c0_snapshot` uses run tag `s3`, and the noise controls and the copies use `first`. No cell of S1's `table_*.json`, `families.json` or `descriptive.json` names an S3 score set (T1g).
- S1 and S2 score files, S2 fit records, and every file under `data/checkpoints/{c0,c1,c2,c3,c4_tfb,c4_lap}/`. They are read only. New files go only to `data/scores_i2/` (the scorer's own score files), `data/scores_i2/s3/` (everything else S3 writes) and `data/checkpoints/i2_posthoc/noise_*/`. S1's checker globs `data/scores_i2/*.pt` and parses any name with two `__` as a score file (scripts/check_eval_rebuild.py:131-137, 1011-1017), so S3's feature files must not sit in that folder.
- `minigpt/uncertainty.py`, `minigpt/laplace.py`, `minigpt/tfb.py` and `minigpt/posthoc_refit.py`. S3 imports their public names and edits none. It imports no private name (a leading underscore).
- `scripts/check_eval_rebuild.py` (S1's independent checker).
- The model, layer and LoRA code, `experiments/`, `paper/`, report.md and the README numbers (PG3; m3, m11, m13).

Interface with S1 (`specs/i2-eval-rebuild.md`, revision 2; code in the working tree):

| S3 needs | What S1 provides |
|---|---|
| Blocks with document IDs and weights | The manifest, the block files, and in every score file `block_index`, `doc_id`, `domain`, `offset`, `weight` ($w_b = 1/k_d$ over the blocks of that file). `BlockSet.select` keeps the full-set weights, so every S3 cell recomputes $w_b$ as S1's analysis does (1/$k_d$ within the cell). |
| The weight-sampling scores | Block keys `blk_g` (primary), `blk_mi`, `blk_tu`, `blk_au`, `blk_nll`, `blk_maxprob_unc` [B] float32. Token keys `logp_real` [B, N, T] float32, `tok_mi`, `tok_tu`, `tok_au`, `tok_maxprob`, `tok_sum_p_sq` [B, T] float32 and `tok_correct` [B, T] bool (scripts/eval_c_checkpoints.py:721-723, 760-775). `tok_pbar_true` is not saved; $\log\bar p(y_t)$ comes from `logp_real`. File name `{score_set}__{eval_set}__{run_tag}.pt`. |
| Score-file meta | `checkpoint_sha256` is a dict {path: sha256}. Also `score_set`, `sampler`, `run_tag`, `block_ids`, `n_samples`, `batch_size`, `amp_fp16`, `analysis_sha256`, `yaml_sha256`, `created_at` (scripts/eval_c_checkpoints.py:1094-1119). |
| Document length | The manifest field $L_d$ |
| Paired statistics | `auroc` and `fpr_at_tpr` with `sample_weight`, `doc_bootstrap_auroc` (returns `auroc_resamples` and `fpr95_resamples` per score), `paired_doc_bootstrap` (AUROC only) and `holm` in `minigpt/uncertainty.py` |
| One RNG rule for every comparison | `default_rng([seed, i_id, i_ood])` with S1's domain positions (`eval_set.domains` order, `arxiv_stripped` last): stackexchange 0, hackernews 1, arxiv 2, freelaw 3, pubmed_abstracts 4, arxiv_stripped 5. S3 checks its own `domain_index` against this rule at start. |
| A scorer to extend | `register_score_set`, `register_param_sampler`, `score_blocks`. `scoring.n_samples` and `scoring.labels` are keyed by score set. `scoring.score_sets` and `scoring.batch_size` are keyed by eval set. Block $b$ is seeded with `seed_base + seed_offsets[eval_set] + b`; the blocks of a batch share the draw seeded by its first block (S1 Section 5.5). |
| A freeze rule | sha256 of `json.dumps(cfg["analysis"], sort_keys=True, separators=(",", ":"))` (`minigpt.evalset.analysis_sha256`) |
| A check on the one-pass path | `c0__test__main.pt` and `c0__arxiv_stripped__main.pt` at N=1. C0 is not scored on `test_hn`. Entropy uses `PROB_EPS` = 1e-10. S1's batch tolerance for C0 is `checks.repro.batch_c0_max_abs` = 0.001. |
| Split ranges | S1's rule: the ranges come from the training configs through minigpt/data.py:207-213, and nobody types them a second time. S3 reads the train ranges through S2's `load_refit_data` (Section 5.3). |

Interface with S2 (`specs/i2-posthoc-fixes.md`, revision 1; `minigpt/posthoc_refit.py` in the working tree):

| S3 needs | What S2 provides |
|---|---|
| The ΔNLL budget of each S2 row | Laplace records (`c2`, `c4_lap`): `delta_nll_star`, `se_star`, `match.n_samples` (20, or 40 after the SE doubling), `match_status`, `lambda_star`, `ell0`. The TFB record (`c4_tfb`): `delta_nll_tfb`, `se_tfb`, `delta_measurements[-1].n_samples`, `ell0`, `tfb.sigma_q_star`; it has no `match_status`. All records: `exit_code`, `failures`, `base.sha256`, `fit_set.blocks_sha256`, and `fit_set.doc_ids_sha256`, which is None when `fit_units` is `token_range`. |
| The same held-out fit sets | `read_manifest`, `load_refit_data`, `build_fit_blocks` and `fit_batches` on the `data` and `fit` sections of `configs/i2_posthoc_c2.yaml` (SE val) and `configs/i2_posthoc_c4_tfb.yaml` (HN val) |
| Base loading with SHA-256 | `load_posthoc_base` with `base_kind` `full` and `blob_mean`. It builds a MiniGPT without Bayesian layers, so the C1 center (`vi_mean`) needs S3's own loader. |
| An isotropic noise sampler | The v2 `LaplaceState`, `scale_laplace_state`, `posterior_std`, `sample_laplace_params` (GPU draw), `apply_sampled_params`, `select_params` (modes `ffn` and `lora`). With $\hat F\equiv0$, the v2 std is $\lambda^{-1/2}$. |
| ΔNLL | `map_nll` and `measure_delta_nll` (one draw per seed, reused for every fit batch). The SE doubling helper is private, so S3 re-writes its rule (Section 5.5). |
| The λ search | `match_prior_precision(delta_fn, target, grid, extension, bisect_max_steps, match_tol, se_threshold, *, n_samples, se_max_doublings)`, a pure function with five statuses. `MATCH_EXIT_CODES` maps `not_matched` to 0, so S3 sets its own exit codes. |
| Records | `write_fit_record`, `REQUIRED_KEYS`. `check_config` accepts only methods `tfb` and `laplace`, so S3 does not call it on the noise YAMLs. |
| The S2 rows | `c2_refit`, `c4_lap_refit`, `c4_tfb_fixed` `main` score files on `test` and `test_hn` |

From S0 (`data/i2/timing_probe.json`): `rescore.cells` [{`method`, `batch_size`, `ms_per_block`, `peak_reserved_mib`, `over_budget`}] for c0, c1, c2, c3, c4_tfb, c4_lap and mc_dropout, and $b^\star_m$ per set in `estimates.m4.batch_size`. S3's sets take the $b^\star$ of the S0 set with the same forward pass and draw: `c0_snapshot` ← c0, `c3_mean_copies` ← c3, `noise_ffn_*` ← c2, `noise_lora_*` ← c4_lap. A set whose timing-pass peak memory exceeds S0's 9,728 MiB budget runs at the next smaller S0 batch size. T1(e) compares with the S0 batch-1 rates. The timing helper is `scripts/timing_probe.py`'s.

With S5 (`specs/i2-lora-seeds.md`, draft): the `ensemble` sampler serves `c0_snapshot` and S5's `lora_ens3`; whichever spec is built first adds it. S5's draft names its families F5 and F6 in `analysis_s5`, which collides with S3's names; S3 keeps F5-F8 because they continue S1's F1-F4. S5's draft also lists "the ΔNLL of each b2 fit for the noise control" as a feed to S3. S3 does not take it up: S5 runs S3's code on its own rows under its own frozen families (Section 7).

AGENTS.md rules that apply:

- No extra Bayesian libraries. Gaussians, Mahalanobis and the noise sampler use torch and numpy. scikit-learn (`LogisticRegression`, `roc_auc_score`) and scipy (`spearmanr`) are not Bayesian libraries. Both are installed today as mlflow dependencies (sklearn 1.8.0, scipy 1.17.0), and `minigpt/uncertainty.py` already imports sklearn. sklearn 1.8 deprecates the `penalty` argument, so the YAML sets `l1_ratio: 0.0`, which gives the same L2 fit (Section 6.5).
- Explicit configs. Every S3 parameter is in `configs/i2_baselines.yaml` or a noise YAML. The scripts read `cfg[...]` and check the full key list before any helper runs.
- LaTeX for every formula in this spec. No notebooks.
- Keep repo docs fresh. Add the two modules and four scripts to the AGENTS.md structure block and scripts line, and update the test count in AGENTS.md and README.md. Record in `agents/technical-reference.md`: feature files and where they live, noise controls, families F5-F8. Record in `agents/design-rationale.md`: G as the added score, own-center one-pass sets, fixed C, the ceiling rule, first-block noise controls and their batching, the zero-curvature control, the `_use_mean` rule for C1's control, the "no `main` file" rule, the snapshot provenance rule, and the freeze from the spec commit.
- 4-space indent, 100-character lines, type hints on public APIs. Run `ruff check` (also on the four new scripts, since CI lints only `minigpt/ experiments/ tests/`) and `pytest` before each commit. Conventional Commits. Never mention AI assistants.

Choices for the approver. The default applies if Alexey says nothing.

| # | Choice | Default | Alternative | Cost of the alternative |
|---|---|---|---|---|
| 1 | When S3 is approved | Approve and commit S3 before S1's m4 re-score starts. The schedule is now ahead of the roadmap, so this may be before Oct 1. | Approve later | Every S3 table prints `prereg_after_s1_scores: true`, and the paper says that the S3 endpoint was registered after S1's test scores existed (but before any S3 score) |
| 2 | Samples per point in the noise λ sweep | S=20 at every grid and bisection point, exactly as S2 | S=10 during the sweep, then S2's SE rule at $\lambda^\star$ | Saves about 45% of the sweep: 1.0-2.1 GPU h at batch-1 rates (0.3-0.7 at 3x). The sweep rule then differs from S2's. |
| 3 | Noise control for C1 | Yes | No control for C1. Its F8 cells are "no control". | Saves 0.7-1.0 GPU h at batch-1 rates (0.25-0.35 at 3x) |
| 4 | Method-vs-method paired tests (S1 Sections 3 and 7 name S3) | Not in S3: they answer P6 and P14, not m8, m9 or this spec's issues. The paper makes no method-ranking claim unless S5, which owns P14, registers one. | Add secondary families F9 (StackExchange ID, 15 pairs of the 6 rows × 3 OOD domains) and F10 (HackerNews ID, 3 pairs × 3), labelled "one training run per method", before approval | +0.5 engineering h, +0.25 Alexey h, 54 more cells to review |

## 3. Pre-registered endpoint

The registration is the `analysis` block in Section 5.2 of this file, as committed on the branch at approval. At build time, `scripts/analyze_baselines.py --freeze --spec-commit <sha>` reads the spec from that commit (`git show <sha>:specs/i2-baselines-analyses.md`), parses the one block fenced as `yaml` (Section 5.2), takes its `analysis` key, and checks that it equals the `analysis` block of `configs/i2_baselines.yaml`. Then it writes `data/scores_i2/prereg_s3.json` with `analysis_sha256` (S1's hash rule), `spec_commit`, `spec_commit_time` and `frozen_at`. Every S3 analysis step refuses to run if the hash differs. The analysis also writes `prereg_after_s1_scores`: true if `spec_commit_time` is later than the earliest `meta.created_at` of S1's `main` score files on `test`, `test_hn` and `arxiv_stripped`.

No setting below changes after any S3 score, or any S1 test table, has been seen. A later change is reported as post hoc, next to the frozen result.

### 3.1 Frozen settings

| Setting | Value | Source |
|---|---|---|
| Rows, ID sets and centers | `mc_dropout`, ID StackExchange, center C0 (dropout off). `c1`, StackExchange, C1 means. `c3`, HackerNews, C3 means (BLoB mean adapter). `c2_refit`, StackExchange, C0. `c4_lap_refit` and `c4_tfb_fixed`, HackerNews, C3 means. | S1 Section 3 (ID sets, D5). The center is the deterministic model the method samples around. |
| OOD domains | arXiv, FreeLaw, PubMed (`test` blocks) | S1 |
| Added score | $G$ (`blk_g`), primary. $M$ (`blk_mi`), secondary. | S1's pre-registered primary score. The roadmap text says MI; see Section 6.4, row 1. |
| One-pass set $B_m$ | 13 features. 7 surface rates of the block text (model free). 6 scores of the row's center: NLL, token entropy, max-prob uncertainty, MD, RMD, likelihood ratio (Section 5.3). | This spec |
| Transforms | $\log(x+10^{-4})$ for the 6 character rates. $\log(x+10^{-6})$ for MD. $\log(\max(x,0)+10^{-6})$ for $G$ and $M$. All others raw. | This spec. The audit found log vs raw MI equal within 0.001. |
| Combination | L2 logistic regression, $C=1.0$, lbfgs, `max_iter` 5,000, `l1_ratio` 0.0, block weights $w_b$. Features standardized with the weighted mean and sd of the training folds. No tuning of $C$. | Probe (Section 6.3): choosing $C$ by inner-fold AUROC gave a false gain |
| Folds | 10, by document, stratified by class, drawn with `default_rng([0, i_id, i_ood])`. Stability check: fold seeds 1-4, report only. | This spec |
| Metric | $\mathrm{AUROC}_w$ of the pooled out-of-fold scores | S1 Section 5.6 |
| Bootstrap | S1's `paired_doc_bootstrap`: 10,000 resamples, seed `[0, i_id, i_ood]`, 95% percentile CI, two-sided p with floor $2/(B+1)$ | S1 Section 5.6 |
| Margin | $\delta = 0.02$ AUROC | S1 Section 3 (twice the reproduction tolerance) |
| Level | $\alpha = 0.05$ after Holm, within each family | S1 |
| Ceiling | A cell is at ceiling if $\mathrm{AUROC}_w(B_m) > 1-\delta = 0.98$. Then no score can add $\delta$. | This spec. Probe: one-pass AUROC 0.999 on PubMed, 0.977 on FreeLaw. |
| Noise controls | Isotropic Gaussian noise on the row's parameter subset of its center (Section 5.5). $\lambda$ by S2's sweep: grid $10^0,\dots,10^7$, extension $10^8$, at most 8 log-bisection steps, ±10% band, S=20 (seeds 0-19), one SE doubling to S=40. S2's fit sets (640 blocks, `fit_seed` 0). | Roadmap S3 test 2; S2 Section 3 |
| Noise targets | The method's own ΔNLL on the same fit set: from S2's records for `c2_refit`, `c4_lap_refit` (`delta_nll_star`) and `c4_tfb_fixed` (`delta_nll_tfb`); measured by S3 (S=20, native sampler, SE rule) for `mc_dropout`, `c1`, `c3` | This spec |
| Noise blocks | The first block (smallest offset) of each document of the row's ID domain and the 3 OOD domains: 4,000 blocks per control, weight 1 | This spec (halves the GPU cost) |
| Bins | Inner edges at the weighted pooled quantiles 0.2, 0.4, 0.6, 0.8, merged when equal. A point mass stays in one bin. A bin is reported only with at least 30 documents of each class. | This spec |
| Snapshot members | C0 steps 80,000, 85,000, 88,000 (`ckpt_best`), 90,000 and 95,000 of MLflow run `6215391b` | Provenance probe (Section 6.3); PG11 |
| BLoB floor | factor 10 | Roadmap S3 test 5 |

### 3.2 Primary quantity and families

For a cell $c$ = (row $m$, its ID domain, OOD domain $d$):

$$\Delta_c=\mathrm{AUROC}_w\big(\hat s_{B_m+G_m}\big)-\mathrm{AUROC}_w\big(\hat s_{B_m}\big)$$

where $\hat s$ are pooled out-of-fold logistic scores (Section 5.4).

| Family | Cells | Holm over | Role |
|---|---|---|---|
| F5 (S3 primary, S1 rows) | $G$ added; rows `mc_dropout`, `c1`, `c3` × 3 OOD domains | 9 | primary |
| F6 (S3 primary, S2 rows) | $G$ added; rows `c2_refit`, `c4_lap_refit`, `c4_tfb_fixed` × 3 | the scored cells (9 at most) | primary |
| F7 | $M$ added; the same 18 cells | the scored cells (18 at most) | secondary |
| F8 (noise) | $\Delta^{\mathrm{noise}}_c=\mathrm{AUROC}_w(G_m)-\mathrm{AUROC}_w(G_{\mathrm{noise}(m)})$ on first blocks; 6 rows × 3 | the controlled cells (18 at most) | secondary |

A row that S2 marks `not_matched` has no score file. Its cells are "not scored", and Holm runs over the rest. A control that does not end `matched` or `matched_noisy` leaves its F8 cells "no control". Rows added by later specs (S5) get their own families in their own frozen YAML. Nobody edits F5-F8.

### 3.3 Decision rules

F5, F6 and F7. The first rule that applies wins.

| Verdict | Rule | What the paper says |
|---|---|---|
| not scored | the row has no score file | "<method> is not scored (S2: no prior precision matched the budget)." |
| ceiling | $\mathrm{AUROC}_w(B_m) > 0.98$ | NF2 below |
| adds | $\Delta_c \ge 0.02$ and Holm-adjusted $p < 0.05$ | P1 below |
| no useful gain | the upper end of the 95% CI of $\Delta_c$ is below 0.02 | NF1 below |
| inconclusive | otherwise | "The gain is <Δ> [<lo>, <hi>]. At this sample size we cannot tell a useful gain from none." |

Every cell also reports $\Delta\mathrm{FPR95}=\mathrm{FPR95}_w(B_m)-\mathrm{FPR95}_w(B_m+G_m)$ with its paired CI, and the range of $\Delta_c$ over fold seeds 1-4. A cell whose verdict differs under any of those seeds is marked "fold-unstable". Neither changes the verdict.

F8 (noise):

| Verdict | Rule |
|---|---|
| structure credited | $\Delta^{\mathrm{noise}}_c \ge 0.02$ and Holm-adjusted $p < 0.05$ |
| matched noise does as well | the upper end of the 95% CI of $\Delta^{\mathrm{noise}}_c$ is below 0.02 |
| inconclusive | otherwise |
| flag (roadmap test 2; reported next to the verdict) | the marginal 95% CIs of $\mathrm{AUROC}_w(G_m)$ and $\mathrm{AUROC}_w(G_{\mathrm{noise}(m)})$ overlap |

Descriptive only (no test): single-feature AUROCs, the surface model alone, the model-only model, $B_m$ alone, $G$ alone against $B_m$, the snapshot ensemble (alone and added to $B_{\mathrm{C0}}$), the stripped copy, the bins, Spearman, the own-TU bins, MI scale, calibration, $G$ at N=3, and the latency table.

### 3.4 Framing, written in advance

These are drafts for m11. The verdicts pick the sentences.

- P1 (adds): "Adding the realized-token score $G$ of <method> to the one-pass scores of the same network raises AUROC from <a> to <b> on <domain> (Δ <x> [<lo>, <hi>], Holm-adjusted p <p>)."
- NF1 (no useful gain; the roadmap's negative): "On <FreeLaw / PubMed>, weight sampling adds less than 0.02 AUROC beyond one-pass scores of the same network: Δ <x> [<lo>, <hi>] for <method>. On these domains the weight-sampling score adds nothing useful beyond one-pass scores."
- NF2 (ceiling): "On <domain>, one-pass scores of the same network already reach AUROC <a>. No score can add 0.02 there. Weight sampling is not needed to detect this shift."
- NF3 (surface): "A model-free model of 7 character rates reaches AUROC <a> [<lo>, <hi>] on <domain>. It matches or beats every weight-sampling score." This sentence is used whenever the surface model's CI overlaps or lies above the best $G$ AUROC of that cell.
- NF4 (noise): "At the same held-out NLL cost, isotropic Gaussian noise on the same weights gives AUROC <a>, against <b> for <method> (paired Δ <x> [<lo>, <hi>]). The detection cannot be credited to the shape of the posterior."
- P2 (noise, positive): "<method> beats isotropic noise at the same held-out NLL cost by <x> [<lo>, <hi>] AUROC (Holm-adjusted p <p>)."
- Scope sentence, always: "The combined model is fitted by cross-validation on labelled test documents. It measures how much information each score adds. It is not a detector one could deploy without OOD labels."

## 4. Acceptance tests

A test passes only if every check under it passes, except where a check says "report only". Every check gives a number or a yes/no, and names where it runs:

- `[align]`: `python scripts/score_onepass.py --config configs/i2_baselines.yaml --check align`
- `[check]`: `python scripts/analyze_baselines.py --config configs/i2_baselines.yaml --check all`. It writes `data/scores_i2/s3/check_all.json` with one entry per check ID below.
- `[rederive]`: `python scripts/check_baselines.py --config configs/i2_baselines.yaml --rederive`
- `[fit]`: the exit code of `scripts/fit_noise_controls.py` and its `fit_record.json`
- `[s1]`: S1's `python scripts/check_eval_rebuild.py --config configs/i2_eval.yaml` with `--check align`, `--check prereg` and `--rederive`
- `pytest file::name`: CPU, CI

Each script exits non-zero if any of its checks fails.

| ID | Given | When | Then | Runs in |
|---|---|---|---|---|
| S3-T1 One-pass baselines, snapshot ensemble, timing, S1 non-regression | S1's manifest, block files and `main` score files. The C0, C1 and C3 checkpoints and the C0 step checkpoints. `configs/i2_baselines.yaml`. S0's `data/i2/timing_probe.json`. | `scripts/score_onepass.py` runs `--part provenance`, `--part gauss`, `--part onepass` (centers C0, C1 mean, C3 mean and background C0 step 10,000, on `test`, `test_hn`, `arxiv_stripped`) and `--part timing`. The scorer runs `c0_snapshot` on `test` (run tag `s3`) and `c3_mean_copies` on `test_hn` (run tag `first`). `scripts/analyze_baselines.py --baselines` builds the table. | (a) `[align]` Every block of every eval set has the 7 surface rates, and for each center the 6 model scores. `block_index`, `doc_id`, `domain` and `offset` equal those of S1's first `main` file of that eval set (yes/no per file). (b) `[check]` C0's `op_nll`, `op_ent` and `op_maxprob_unc` equal `blk_nll`, `blk_tu` and `blk_maxprob_unc` of `c0__test__main.pt` and `c0__arxiv_stripped__main.pt` within 1e-3 on every block, S1's `batch_c0_max_abs` (yes/no). (c) `[check]` `snapshot_provenance.json` admits exactly `ckpt_step80000`, `ckpt_step85000`, `ckpt_best` (step 88,000), `ckpt_step90000` and `ckpt_step95000` as members, admits `ckpt_step10000` as the background, and rejects `ckpt_step97800` (yes/no each). (d) `[check]` `c0_snapshot__test__s3.pt` has `n_samples` 5 and 5 `checkpoint_sha256` entries equal to the provenance file's hashes. The mean NLL of member `ckpt_best` equals C0's mean `op_nll` on the same blocks within 1e-3. The provenance file holds each member's mean StackExchange NLL (yes/no each). (e) `[check]` `timing.json` holds ms per block at N=3 and N=20, at batch 1 and at $b^\star$, for every stochastic score set in the tables, and the one-pass rate of each center. The batch-1 rate of `c0` (one pass) and the batch-1 N=20 rates of `mc_dropout`, `c1` and `c3` (samplers unchanged since S0) are within ±25% of S0's `rescore.cells` at batch 1 (yes/no each). (f) `[check]` `baselines_table.json` holds $\mathrm{AUROC}_w$ and $\mathrm{FPR95}_w$ with document-bootstrap CIs for the 13 features, the surface model, the model-only model, $B_m$ and the snapshot ensemble ($G$ and $M$), for every center and (ID, OOD) pair (yes/no per cell). (g) `[s1]` After every S3 scorer run, S1's three checks exit 0. No cell of S1's `table_*.json`, `families.json` or `descriptive.json` names an S3 score set. No score file of an S3 score set has run tag `main` (yes/no each). (h) pytest. `tests/test_baselines.py::test_surface_features`: the 7-character string (A, backslash, b, \$, 1, newline, é) with $T=256$ gives exactly tex 2/7, digit 1/7, upper 1/7, punct 0, newline 1/7, non-ASCII 1/7, chars per token 7/256 (yes/no). `::test_mahalanobis`: on a synthetic Gaussian, MD of the mean is 0 within 1e-9, and RMD is 0 within 1e-9 when the ID and background Gaussians are equal (yes/no). `::test_provenance_rejects`: synthetic checkpoint records with an mtime outside the run window, a different parameter count, or a different config are each rejected (yes/no each). `tests/test_analyze_baselines.py::test_ensemble_sampler`: on S1-T5's 2-layer fixture with 3 distinct member checkpoints, `logp_real[:, s, :]` equals an independent forward pass of member $s$ within 1e-5 for $s=0,1,2$; `meta` holds `n_samples` 3 and `sampler` `ensemble`; `--run-tag main` exits non-zero (yes/no each). | `score_onepass.py`, `eval_c_checkpoints.py`, `analyze_baselines.py`, S1's checker; `tests/test_baselines.py`, `tests/test_analyze_baselines.py` |
| S3-T2 Noise control | S2's fit records and fit sets. The MC dropout, C1 and C3 samplers. The noise YAMLs. | `scripts/fit_noise_controls.py` measures the native ΔNLL of MC dropout, C1 and C3 (S=20, SE rule), then finds $\lambda^\star$ for each control with S2's `match_prior_precision`. The scorer scores each control with `--first-block-only --domains` at N=20 (run tag `first`). `analyze_baselines.py --noise` builds F8. | (a) `[fit]` Every control exits 0 and ends `matched` or `matched_noisy`. `match.delta_nll_first` is within ±10% of the target; for `matched`, the final ΔNLL is too. `fit_record.json` holds $\lambda^\star$, $\sigma^\star=\lambda^{\star-1/2}$, the target and its source (path, SHA-256, field, $S$), the base SHA-256, the parameter subset, and its entry count: 33,554,432 for `ffn` and `ffn_mu`, 655,360 for `lora_a` (yes/no per control). (b) `[check]` Each control's `fit_set.blocks_sha256` equals that of S2's record for the same fit set (`c2` for StackExchange, `c4_tfb` for HackerNews). Its $\ell_0$ equals that record's `ell0` within 1e-5 for the controls on C0 and on C3 mean. For the C1 control, $\ell_0$ of the native path (`use_mean_weights`) equals the control's $\ell_0$ within 1e-4. The base SHA-256 of each noise score file (in `meta.checkpoint_sha256`) equals its record's `base.sha256`, and that hash appears in the method's own S1 score-file meta or S2 record (yes/no each). (c) `[check]` `noise_table.json` holds, for every controlled cell, both marginal AUROCs with CIs, the overlap flag, the paired $\Delta^{\mathrm{noise}}$ with CI, p and Holm-adjusted p, and the verdict (yes/no per cell). (d) pytest, `tests/test_noise_control.py`. `::test_zero_curvature_std`: a zero-curvature v2 state with $\lambda=400$ gives `posterior_std` 0.05 with relative error at most 1e-6 on every entry (float32; critic probe 1.5e-8) (yes/no). `::test_match_synthetic`: with `delta_fn` returning $(250/\lambda, 0)$, target 0.01 and S2's grid and extension, S2's function returns `matched` with $\lambda^\star=10^{4.375}=23{,}713.7\pm0.1$ after 3 bisection steps in the bracket $[10^4, 10^5]$ (yes/no). `::test_no_control_exit`: `delta_fn` returning $(1.0, 0)$ gives `not_matched`, exit code 5 and no state file; $(10^{-6}, 0)$ gives `tighter_than_budget`, exit code 5 (yes/no each). `::test_targets_from_records`: a synthetic Laplace `matched_noisy` record yields its `delta_nll_star` (the S=40 value) as target; a TFB record yields `delta_nll_tfb`; a record with a non-zero `exit_code` or a non-empty `failures` raises ValueError (yes/no each). `::test_c4_sharing`: a C4-LAP target at 1.05 × the C4-TFB control's matched ΔNLL shares that control; at 1.15 × it gets `noise_lora_c3_lap` (yes/no each). `::test_flag_fixture` (fixture F-noise, Section 5.5): the near copy raises the flag and its paired CI contains 0 (probe: 0.0000 [−0.0001, 0.0002], p 0.91); the shifted copy does not raise the flag and gets "structure credited" (probe: +0.2260 [+0.2092, +0.2441], p 0.0010) (yes/no each). `::test_dropout_fixture_match`: on a 2-layer fixture with dropout 0.1 (the S2-T2 recipe, copied), a control on `ffn` matched to the fixture's own native dropout ΔNLL ends `matched` (yes/no). `::test_vi_mean_control`: on a 2-layer Bayesian-FFN fixture, the `vi_mean` loader leaves `_use_mean` true on every `BayesianLinear`; two scorer passes with the unperturbed `weight_mu` give identical logits (`torch.equal`); noise on `weight_mu` changes the logits; after `apply_sampled_params` exits, the logits equal the originals bitwise (yes/no each). `tests/test_analyze_baselines.py::test_first_block_only`: on S1-T5's fixture eval set, `--first-block-only --domains` keeps exactly the smallest-offset block of each document of the listed domains; `block_index` keeps the full-set values; at batch size 4 the batches are consecutive rows of the subset, each seeded by its first block; two runs give identical tensors (`torch.equal`) (yes/no each). | `fit_noise_controls.py`, `eval_c_checkpoints.py`, `analyze_baselines.py --noise`; `tests/test_noise_control.py`, `tests/test_analyze_baselines.py` |
| S3-T3 Combined-score test (primary endpoint) | The S1, S2 and S3 score and feature files. The frozen `analysis` block and `prereg_s3.json`. | `analyze_baselines.py --families` builds F5-F7. A separate agent's `scripts/check_baselines.py --rederive` rebuilds F5 and F6. | (a) `tests/test_analyze_baselines.py::test_freeze_refuses`: with a fixture YAML whose `analysis` block differs from the spec block by one value, `--freeze` and `--families` exit non-zero and write no table (yes/no). `[check]` `prereg_s3.json` holds the sha256 of the YAML's `analysis` block, and `prereg_after_s1_scores` is written (true or false, report only). (b) `[check]` Every document's blocks sit in one fold. Each of the 10 folds holds $\lfloor n/10\rfloor$ or $\lceil n/10\rceil$ documents of each class. Two runs give identical out-of-fold arrays (`np.array_equal`) (yes/no each). `tests/test_baselines.py::test_folds` checks the same on fixture F-cv (yes/no). (c) `tests/test_baselines.py::test_cv_fixture` (fixture F-cv, Section 5.4, seed 303, 2,000 resamples). $\mathrm{AUROC}_w(B)$ = 0.7384. Added noise: verdict "no useful gain", CI contains 0, $\Delta$ = +0.0023. Added signal: verdict "adds", $\Delta$ = +0.1352, p = 0.0010. Added near copy of $x_1$: CI contains 0, $\Delta$ = 0.0000. Fold seed `[1, 0, 2]`: $\mathrm{AUROC}_w(B)$ 0.7401, noise +0.0015, signal +0.1343, near copy −0.0001. Every $\Delta$ and AUROC within ±0.001 of these values; the critic's independent re-run matched the writer's probe to 4 decimals (yes/no each). (d) `[check]` `families_s3.json` holds, for every cell of F5-F7: $\mathrm{AUROC}_w(B_m)$, $\mathrm{AUROC}_w(B_m+\text{score})$, $\Delta_c$, CI, p, Holm-adjusted p, verdict, $\Delta\mathrm{FPR95}$ with CI, and the fold-seed range (yes/no per cell). (e) `[rederive]` reproduces $\Delta_c$, CI, p, Holm-adjusted p and the verdict of every F5 and F6 cell within 1e-6 (yes/no per cell), and the import check passes (yes/no). | `analyze_baselines.py --families`; `check_baselines.py --rederive`; `tests/test_baselines.py`, `tests/test_analyze_baselines.py` |
| S3-T4 Matched bins and the stripped copy | Per block: the LaTeX rate `sf_tex`, the center's `op_nll` and $\log L_d$. S1's `arxiv_stripped` score files (MC dropout, C1, C3, C4-TFB pre-fix). | `analyze_baselines.py --bins` | (a) `[check]` For every F5-F6 cell and each of the 3 confounds: a per-bin table with blocks and documents per class, and $\mathrm{AUROC}_w$ of $G$, $M$ and the confound itself in every reportable bin. The stratified AUROC of $G$ and $M$ over reportable bins, with a document-bootstrap CI (yes/no per cell). (b) `[check]` In every cell where 20% or more of the block weight has `sf_tex` = 0, bin 0 holds exactly the blocks with `sf_tex` = 0 (yes/no). Bins whose confound AUROC is outside [0.35, 0.65] are marked "poorly matched" (report only). (c) `[check]` Stripped copy: $\mathrm{AUROC}_w$ of $G$ and $M$ for S1's four stripped rows, of `sf_tex`, of the surface model and of $B_m$, and $\Delta_c$ of $G$ (descriptive), with CIs (yes/no per row). (d) `tests/test_baselines.py::test_bins_fixture` (fixture F-bin, Section 5.7, seed 404). Inner edges [0, 0.077, 0.673, 1.838] within 1e-3. Confound only: raw 0.7381, confound 0.7797, stratified 0.5758, each within ±0.0005, and the stratified AUROC at least 0.10 below the raw one. Signal: raw 0.9151 and stratified 0.9220, each within ±0.0005, and the stratified AUROC at most 0.02 below the raw one. Bin 0 holds exactly the $v=0$ blocks. Bin 1 (25 ID and 26 OOD documents) is not reported (yes/no each). | `analyze_baselines.py --bins`; `tests/test_baselines.py` |
| S3-T5 Epistemic diagnostics, MI scale, BLoB floor, calibration, latency | The per-block and per-token arrays of every row. `c3_mean_copies__test_hn__first.pt` (20 identical copies of C3 mean). `timing.json`. | `analyze_baselines.py --diagnostics --latency` | (a) `[check]` Spearman($M$, AU) and Spearman($G$, AU) per row and domain, on first blocks, with a document-bootstrap CI; and $\mathrm{AUROC}_w$ of $G$ and $M$ within quintile bins of the row's own TU (yes/no per cell). (b) `[check]` Floor: the mean ID $M$ of C3 is at least 10 times the mean ID $\lvert M\rvert$ of the copies, and the same for $G$ (yes/no; if no, the BLoB row is flagged "below floor" in every table). `tests/test_baselines.py::test_identical_copies_floor`: 3 identical copies of a 2-layer fixture give block $\lvert M\rvert$ and $\lvert G\rvert$ of at most 1e-6 (yes/no). (c) `[check]` An MI-scale table: ID mean and median of $M$ and $G$ per row, and their ratio to the floor (yes/no). (d) `[check]` Calibration (report only, as the roadmap says): ECE (15 bins), NLL and Brier of $\bar p$ per row and for C0, per domain, with document-bootstrap CIs. Consistency: C0's NLL here equals the weighted mean of its `blk_nll` within 1e-6 (yes/no). `tests/test_baselines.py::test_weighted_ece`: with all weights 1, the weighted ECE equals `minigpt.uncertainty.ece` within 1e-12 (yes/no). (e) `[check]` $\mathrm{AUROC}_w$ of $G$ from samples 0-2 of `logp_real` (N=3) and from all 20, per row and domain, with CIs (yes/no). `tests/test_baselines.py::test_g_first_n`: $G$ from the first 20 of 20 samples equals `blk_g` within 1e-6 (yes/no). (f) `[check]` `latency_table.json` pairs every score's $\mathrm{AUROC}_w$ per OOD domain with its passes and ms per block from T1(e) (yes/no). | `analyze_baselines.py`; `tests/test_baselines.py` |

Commands before merge (CPU, CI):

```bash
uv run ruff check minigpt/ experiments/ tests/ scripts/score_onepass.py scripts/fit_noise_controls.py \
  scripts/analyze_baselines.py scripts/check_baselines.py
uv run pytest tests/test_baselines.py tests/test_noise_control.py tests/test_analyze_baselines.py -rs
```

Checks after the runs (machine with `data/`). Each exits non-zero on failure:

```bash
python scripts/score_onepass.py --config configs/i2_baselines.yaml --check align
python scripts/analyze_baselines.py --config configs/i2_baselines.yaml --check all
python scripts/check_baselines.py --config configs/i2_baselines.yaml --rederive
python scripts/check_eval_rebuild.py --config configs/i2_eval.yaml --check align
python scripts/check_eval_rebuild.py --config configs/i2_eval.yaml --check prereg
python scripts/check_eval_rebuild.py --config configs/i2_eval.yaml --rederive
```

## 5. Implementation notes for the builder

### 5.1 Order of work

| Step | Work | Test | Machine | When |
|---|---|---|---|---|
| 0 | Commit this spec on the branch at approval. The commit is the registration. | T3a | none | at approval, before S1's m4 re-score |
| 1 | `minigpt/baselines.py` pure functions, tests first | T1h, T3b-c, T4d, T5b, T5d-e (pytest parts) | CPU | session 04 |
| 2 | `minigpt/noise_control.py`, `scripts/fit_noise_controls.py`, noise YAMLs | T2d | CPU | session 04 |
| 3 | Scorer entries: the `ensemble` sampler, `c0_snapshot`, `c3_mean_copies`, the noise loaders, `--first-block-only --domains`, the `main` refusal | T1h, T2d | CPU | session 04 |
| 4 | `scripts/score_onepass.py` | T1a-e (fixture run) | CPU | session 04 |
| 5 | `scripts/analyze_baselines.py`, including `--freeze` | T3a, T3d | CPU | session 04 |
| 5b | A separate agent writes `scripts/check_baselines.py --rederive` from Sections 3 and 5.4 and S1 Section 5.6 only | T3e | CPU | session 04 |
| 6 | Night A, part 1: provenance, Gaussians, one-pass, snapshot, copies, timing | T1 | GPU, 0.3-0.7 h | Mon Oct 5, after S2's jobs |
| 7 | Night A, part 2: native ΔNLL, then controls `mcd`, `c1`, `c3` (fit, then score) | T2 (part) | GPU, 0.8-3.6 h | Mon Oct 5, after step 6. What does not finish moves to step 9. |
| 8 | CPU analysis of F5-F7, T1, T3-T5, and S1's three checks | T1f-g, T3-T5 | CPU, about 1 h | Tue Oct 6 |
| 9 | Night B: controls `c2` and `c4_tfb` (and `c4_lap` if the sharing rule fails), plus any carry-over from step 7 | T2 | GPU, 0.6-3.6 h | Wed Oct 7, after S5's night-2 re-score; Thu Oct 8 at the top of the ranges |
| 10 | F8 analysis, then `--check all`, `--rederive` and S1's three checks again | T2c, T3e, T1g | CPU | Thu Oct 8; Fri Oct 9 at the latest |

### 5.2 `configs/i2_baselines.yaml`

Every key is explicit. The `analysis` block below is the registration (Section 3). Paths, batch sizes, timing and scorer run tags sit outside it.

```yaml
paths:
  eval_config: configs/i2_eval.yaml
  scores_dir: data/scores_i2                 # the scorer's score files (S1's folder)
  out_dir: data/scores_i2/s3                 # all other S3 files; outside S1's *.pt glob
  prereg: data/scores_i2/prereg_s3.json
  spec: specs/i2-baselines-analyses.md
  mlflow_db: mlflow.db
  timing_probe: data/i2/timing_probe.json
device: cuda
autocast_fp16: true
batch_size: {onepass: 32, gauss_fit: 32}       # forward passes only; no weight sampling
model: {block_size: 256, n_layer: 16, n_head: 8, n_embd: 512, dropout: 0.1, bias: true}
lora: {rank: 16, alpha: 32.0, target: ffn}
bayes_ffn_c1: {enabled: true, prior_std: 1.0, init_rho: -2.0}   # configs/c1.yaml
centers:
  c0: {base_kind: full, base_checkpoint: data/checkpoints/c0/ckpt_best.pt}
  c1_mean: {base_kind: vi_mean, base_checkpoint: data/checkpoints/c1/ckpt_best.pt}
  c3_mean: {base_kind: blob_mean, base_checkpoint: data/checkpoints/c3/ckpt_best.pt}
  c0_step10000: {base_kind: full, base_checkpoint: data/checkpoints/c0/ckpt_step10000.pt}
timing:
  blocks_batch1: 64
  blocks_bstar: 128
  warmup_batches: 1
  n_values: [3, 20]
  s0_rate_tol: 0.25
  s0_compare_sets: [c0, mc_dropout, c1, c3]
  bstar_source: {c0_snapshot: c0, c3_mean_copies: c3, noise_ffn: c2, noise_lora: c4_lap}
  vram_budget_mib: 9728
scorer_runs:
  run_tags: {c0_snapshot: s3, c3_mean_copies: first, noise: first}
  first_block_domains:
    stackexchange_rows: {test: [stackexchange, arxiv, freelaw, pubmed_abstracts]}
    hackernews_rows: {test_hn: [hackernews], test: [arxiv, freelaw, pubmed_abstracts]}
analysis:
  version: s3-v1
  ood_domains: [arxiv, freelaw, pubmed_abstracts]
  domain_index: {stackexchange: 0, hackernews: 1, arxiv: 2, freelaw: 3, pubmed_abstracts: 4,
                 arxiv_stripped: 5}        # checked against S1's rule at start
  rows:
    mc_dropout: {id_domain: stackexchange, center: c0, noise_subset: ffn}
    c1: {id_domain: stackexchange, center: c1_mean, noise_subset: ffn_mu}
    c3: {id_domain: hackernews, center: c3_mean, noise_subset: lora_a}
    c2_refit: {id_domain: stackexchange, center: c0, noise_subset: ffn}
    c4_lap_refit: {id_domain: hackernews, center: c3_mean, noise_subset: lora_a}
    c4_tfb_fixed: {id_domain: hackernews, center: c3_mean, noise_subset: lora_a}
  added_scores: {primary: blk_g, secondary: blk_mi}
  features:
    surface: [sf_tex, sf_digit, sf_upper, sf_punct, sf_newline, sf_nonascii, sf_chars_per_tok]
    model: [op_nll, op_ent, op_maxprob_unc, op_md, op_rmd, op_lr]
    punct_excluded: ["\\", "$"]
    log_eps: {sf_tex: 1.0e-4, sf_digit: 1.0e-4, sf_upper: 1.0e-4, sf_punct: 1.0e-4,
              sf_newline: 1.0e-4, sf_nonascii: 1.0e-4, op_md: 1.0e-6, blk_g: 1.0e-6,
              blk_mi: 1.0e-6}
    clip_min_zero: [blk_g, blk_mi]
    entropy_eps: 1.0e-10                   # S1's PROB_EPS, so op_ent matches blk_tu
    pooling: mean_all_positions_after_ln_f
    gaussian:
      windows_per_domain: 5000
      fit_seed: 0
      ridge_frac: 1.0e-3
      window_domain_index: {wikipedia_en: 0, stackexchange: 1, hackernews: 2}
      range_source: {wikipedia_en: configs/i2_posthoc_c2.yaml,
                     stackexchange: configs/i2_posthoc_c2.yaml,
                     hackernews: configs/i2_posthoc_c4_tfb.yaml}
      id_fit: {c0: stackexchange, c1_mean: stackexchange, c3_mean: hackernews}
      background_fit: [wikipedia_en, stackexchange, hackernews]
    lr_background: {c0: c0_step10000, c1_mean: c0_step10000, c3_mean: c0}
  combined:
    n_folds: 10
    cv_seed: 0
    stability_cv_seeds: [1, 2, 3, 4]
    logistic: {C: 1.0, solver: lbfgs, max_iter: 5000, l1_ratio: 0.0}
    standardize: weighted_train_folds
  bootstrap: {resamples: 10000, seed: 0, level: 0.95, descriptive_resamples: 2000}
  margin_auroc: 0.02
  alpha: 0.05
  ceiling_auroc: 0.98
  fpr_target_tpr: 0.95
  families:
    F5_s3_primary_s1_rows: {score: blk_g, rows: [mc_dropout, c1, c3]}
    F6_s3_primary_s2_rows: {score: blk_g, rows: [c2_refit, c4_lap_refit, c4_tfb_fixed]}
    F7_s3_secondary_mi: {score: blk_mi,
                         rows: [mc_dropout, c1, c3, c2_refit, c4_lap_refit, c4_tfb_fixed]}
    F8_noise: {score: blk_g, blocks: first_block,
               rows: [mc_dropout, c1, c3, c2_refit, c4_lap_refit, c4_tfb_fixed]}
  noise:
    controls: {mc_dropout: noise_ffn_c0_mcd, c1: noise_ffn_c1, c3: noise_lora_c3_blob,
               c2_refit: noise_ffn_c0_c2, c4_tfb_fixed: noise_lora_c3_tfb,
               c4_lap_refit: noise_lora_c3_tfb}
    own_control_if_outside_band: {c4_lap_refit: noise_lora_c3_lap}
    targets: {mc_dropout: native, c1: native, c3: native, c2_refit: s2_record,
              c4_tfb_fixed: s2_record, c4_lap_refit: s2_record}
    s2_target_field: {laplace: delta_nll_star, tfb: delta_nll_tfb}
    native_samples: 20
    prior_prec_grid: [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]
    grid_extension: [100000000.0]
    bisect_max_steps: 8
    match_tol: 0.10
    n_delta_samples: 20
    se_threshold: 0.05
    se_max_doublings: 1
    accepted_statuses: [matched, matched_noisy]
    ell0_tol_s2: 1.0e-5
    ell0_tol_native: 1.0e-4
    n_score_samples: 20
  bins:
    confounds: [sf_tex, op_nll, log_doc_len]
    inner_quantiles: [0.2, 0.4, 0.6, 0.8]
    min_docs_per_class: 30
    poorly_matched_band: [0.35, 0.65]
  diagnostics:
    spearman_blocks: first_block
    tu_quantiles: [0.2, 0.4, 0.6, 0.8]
    floor_factor: 10
    floor_score_set: c3_mean_copies
    latency_n: [3, 20]
  calibration: {ece_bins: 15, target: next_token_top1, unit_weight: block_weight,
                rows: [mc_dropout, c1, c3, c2_refit, c4_lap_refit, c4_tfb_fixed, c0]}
  snapshot:
    mlflow_run_prefix: 6215391b
    members: [ckpt_step80000.pt, ckpt_step85000.pt, ckpt_best.pt, ckpt_step90000.pt,
              ckpt_step95000.pt]
    background: ckpt_step10000.pt
    reference: ckpt_best.pt
    min_members: 3
```

Noise YAMLs (`configs/i2_noise_<name>.yaml`; the control names are in `analysis.noise.controls`). Top level: `method: isotropic`, `cell` (the control name), `out_dir` (`data/checkpoints/i2_posthoc/<control>/`), `device`, `autocast`. Sections `base`, `model`, `data` and `fit` with S2's key lists (`minigpt.posthoc_refit.REQUIRED_KEYS`), plus `lora` for `base_kind: blob_mean` and `bayes_ffn` for `base_kind: vi_mean`. The `data` and `fit` sections equal, key for key, those of `configs/i2_posthoc_c2.yaml` (controls for `mcd`, `c1`, `c2`) or `configs/i2_posthoc_c4_tfb.yaml` (controls for `c3`, `c4_*`); the script checks this. A `noise` section holds `param_subset` (`ffn`, `ffn_mu` or `lora_a`), `center` (a key of `centers`), `target` (`native` with `native_sampler: dropout | vi | blob`, or `s2_record` with `s2_record_path`, one of `data/checkpoints/i2_posthoc/{c2,c4_lap,c4_tfb}/fit_record.json`), and the sweep keys, whose values must equal `analysis.noise`. S2's `check_config` is not called, because it accepts only `tfb` and `laplace`. `out_dir` must match `data/checkpoints/i2_posthoc/noise_*`.

### 5.3 Surface and one-pass features

Block $b$ has input tokens $x_{1:T}$ and targets $y_{1:T}$, $T=256$ (S1 Section 5.4). Its text is `enc.decode(x.tolist())` with tiktoken `gpt2`. With $n_{\mathrm{char}}(b)$ the number of characters of that text:

$$r_{\mathrm{tex}}(b)=\frac{n_{\backslash}(b)+n_{\$}(b)}{n_{\mathrm{char}}(b)},\qquad r_{\mathrm{digit}},\ r_{\mathrm{upper}},\ r_{\mathrm{punct}},\ r_{\mathrm{newline}},\ r_{\mathrm{nonascii}}\ \text{likewise},\qquad r_{\mathrm{cpt}}(b)=\frac{n_{\mathrm{char}}(b)}{T}$$

- `sf_digit` counts `str.isdigit`, `sf_upper` counts `str.isupper`, `sf_punct` counts characters in `string.punctuation` other than `\` and `$`, `sf_newline` counts `\n`, `sf_nonascii` counts code points above 127 (the U+FFFD from a split UTF-8 sequence counts). The surface file `data/scores_i2/s3/surface_{eval_set}.pt` does not depend on any model.
- The LaTeX rate is the audit's 0.931 feature (research/2026-09/audit/audit_d1_metrics.py:170-171).

One pass of center $c$ (fp16 autocast, fp32 `log_softmax`, `requires_grad_(False)` on every parameter, as in S0 Section 5):

$$\mathrm{NLL}_c(b)=-\frac1T\sum_{t=1}^{T}\log p_c(y_t\mid x_{\le t}),\qquad \mathrm{ENT}_c(b)=\frac1T\sum_{t=1}^{T}H\big[p_c(\cdot\mid x_{\le t})\big],\qquad U_c(b)=1-\frac1T\sum_{t=1}^{T}\max_v p_c(v\mid x_{\le t})$$

$$z_c(b)=\frac1T\sum_{t=1}^{T}h_{c,t}(b),\qquad h_{c,t}=\text{`forward\_body'} \text{ output after } \mathrm{ln}_f$$

$H$ uses S1's formula $-\sum_v p_v\log(p_v+10^{-10})$, so $\mathrm{ENT}$ of C0 matches `blk_tu`. Call `forward_body` once and `lm_head` on its output, so the hidden state and the logits come from the same pass. For `c1_mean` both run inside `use_mean_weights(model)`.

Gaussians, fitted once per center on training windows only. For each domain, the train range $[a,b)$ is `load_refit_data(cfg).train_ranges[domain]` for the S2 YAML named in `range_source`. That is the minigpt/data.py arithmetic, so no range is typed a second time (S1's rule). For today's configs the ranges are wikipedia_en $[0, 10^8)$, stackexchange $[0, 6\times10^7)$ and hackernews $[0, 8\times10^7)$ in local positions of each domain's cached tensor. Window starts are drawn with `default_rng([fit_seed, j])` for domain index $j$, uniformly in $[a, b-T-1)$, and the tokens are read from `data/pile/{domain}_100000000.pt`. The windows never touch S1's test documents (after the cache cut) or S2's val ranges.

$$\hat\mu=\frac1n\sum_i z_i,\qquad \hat\Sigma=\frac{1}{n-1}\sum_i(z_i-\hat\mu)(z_i-\hat\mu)^\top,\qquad \Sigma_r=\hat\Sigma+r\,\frac{\operatorname{tr}\hat\Sigma}{d}\,I,\quad r=10^{-3},\ d=512$$

$$\mathrm{MD}(z)=(z-\hat\mu_{\mathrm{ID}})^\top\Sigma_{r,\mathrm{ID}}^{-1}(z-\hat\mu_{\mathrm{ID}}),\qquad \mathrm{RMD}(z)=\mathrm{MD}(z)-(z-\hat\mu_{\mathrm{bg}})^\top\Sigma_{r,\mathrm{bg}}^{-1}(z-\hat\mu_{\mathrm{bg}})$$

The ID Gaussian uses the 5,000 windows of the center's own ID training range. The background Gaussian uses 15,000 windows, 5,000 each from Wikipedia, StackExchange and HackerNews training ranges: the union of all text any center trained on. The ID windows are the background's windows of that domain, so each center needs 15,000 passes. This follows Lee et al. 2018 (arXiv 1807.03888) for MD and Ren et al. 2021 (arXiv 2106.09022) for RMD. $z$ is stored as float32; MD and RMD are computed in float64 from the stored float32 values, so a re-derivation recomputes them exactly.

Likelihood ratio, with background model $\mathrm{bg}(c)$ from `lr_background`:

$$\mathrm{LR}_c(b)=\mathrm{NLL}_c(b)-\mathrm{NLL}_{\mathrm{bg}(c)}(b)$$

For `c3_mean` the background is C0, so $\mathrm{LR}$ is the fine-tuned-vs-base ratio (Zhang et al. 2024, arXiv 2404.08679): the adapter helps less on OOD text. For `c0` and `c1_mean` the background is C0's step-10,000 checkpoint from C0's run, in the spirit of Ren et al. 2019 (arXiv 1906.02845). All six model scores are oriented so that a higher value means "more OOD".

Feature file `data/scores_i2/s3/onepass_{center}_{eval_set}.pt`:

| Key | Shape, type | Note |
|---|---|---|
| `block_index`, `doc_id`, `domain`, `offset`, `weight` | [B] | in S1's block-file order; equal to S1's score files (T1a) |
| `op_nll`, `op_ent`, `op_maxprob_unc`, `op_lr` | [B] float32 | |
| `op_md`, `op_rmd` | [B] float64 | |
| `z_pool` | [B, 512] float32 | kept so that the re-derivation can recompute MD and RMD |
| `meta` | dict | S1's meta keys where they apply (`git_sha`, `code_sha256`, `checkpoint_sha256`, `batch_size`, `amp_fp16`, `device`, `created_at`, ...), plus `center`, `bg_model`, `gauss_sha256` |

`data/scores_i2/s3/gauss_{center}.pt` holds $\hat\mu$ and $\Sigma_r^{-1}$ (float64) for ID and background, $r$, the train range and window starts per domain, and the center's checkpoint SHA-256.

Snapshot provenance (`--part provenance`). A C0 checkpoint is admitted if all four hold: (1) its file mtime lies inside `[start_time, end_time]` of the MLflow run whose ID starts with `6215391b` (read-only sqlite, `mode=ro`); (2) its stored `step` equals the step in its file name (`ckpt_best` is exempt); (3) its state-dict key set and floating-point parameter count equal `ckpt_best`'s; (4) its embedded `config` dict equals `ckpt_best`'s. The script writes `data/scores_i2/s3/snapshot_provenance.json` with each member's SHA-256 and mean StackExchange NLL. If fewer than 3 listed members pass, the snapshot row is "n/a" and T1(c) fails. The members are always the listed ones. Admitting a checkpoint does not add it.

### 5.4 Combined-score test

Cell $c$ = (row $m$, ID domain, OOD domain $d$). Its blocks are the ID-domain blocks (label 0) and the $d$ blocks (label 1), taken as S1 Section 5.6 says: for ID = HackerNews, ID blocks from `{m}__test_hn__main.pt` and OOD blocks from `{m}__test__main.pt`. The features of block $b$ are the transformed 13 features of $B_m$ (Section 3.1) from the center's feature file and the surface file, plus, for the "plus" model, the transformed added score. Weights are $w_b=1/k_d$ within the cell, as in S1's analysis.

Folds. `rng = np.random.default_rng([cv_seed, i_id, i_ood])`. For class 0 and then class 1: `docs = np.unique(doc_ids[y == cls])`, `perm = rng.permutation(docs.size)`, and document `docs[perm[j]]` goes to fold $j \bmod 10$. All blocks of a document follow it.

Row and column order. Rows are the ID blocks in their score-file order, then the OOD blocks in theirs. Columns are the 7 surface rates, then the 6 model scores, in the order of `analysis.features`, then the added score. The re-derivation must use the same order, so that lbfgs sees the same sums.

Fit. For each fold $f$, standardize each column with the weighted mean and sd of the training blocks, $\mu=\sum_b w_bx_b/\sum_b w_b$ and $\mathrm{sd}=\big(\sum_b w_b(x_b-\mu)^2/\sum_b w_b\big)^{1/2}$ (an sd of 0 is replaced by 1), then fit

$$\min_{\beta_0,\beta}\ \tfrac12\lVert\beta\rVert_2^2+C\sum_{b\notin f}w_b\Big[\log\big(1+e^{\beta_0+\beta^\top\tilde x_b}\big)-y_b\big(\beta_0+\beta^\top\tilde x_b\big)\Big],\qquad C=1$$

which is `LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, l1_ratio=0.0).fit(X, y, sample_weight=w)`. Do not pass `penalty`: sklearn 1.8 deprecates it, and `l1_ratio=0.0` gives the same coefficients (Section 6.5). The out-of-fold score of $b\in f$ is $\hat s(b)=\beta_0+\beta^\top\tilde x_b$ (`decision_function`). Pool the 10 folds.

Statistics. $\Delta_c$ and its CI and p come from `paired_doc_bootstrap({"plus": s_plus, "b": s_b}, y, doc_ids, w, [("plus", "b")], 10000, [0, i_id, i_ood], 0.95)`. $\Delta\mathrm{FPR95}$ comes from `doc_bootstrap_auroc` with the same seed, so its draws are the same: the CI is the percentile interval of `fpr95_resamples["b"] − fpr95_resamples["plus"]`. Holm runs within each family on the cells' p-values (`holm`).

Fixture F-cv (for T3c), built in the test module:

| Step | Construction |
|---|---|
| Documents | `rng = np.random.default_rng(303)`; 400 ID and 400 OOD documents, `y_doc = [0]*400 + [1]*400`; `k = rng.integers(1, 4, size=800)`; `u = rng.normal(0, 1, 800)`; `doc = np.repeat(np.arange(800), k)` (1,609 blocks); `y = y_doc[doc]`; `w = 1/k[doc]`; IDs `f"d{d:04d}"` |
| One-pass features | In this order: `x1 = 0.8*y + 0.6*u[doc] + rng.normal(0, 0.8, nb)`, `x2 = 0.5*y + rng.normal(0, 1, nb)`, `x3 = rng.normal(0, 1, nb)` |
| Added scores | Then: `noise = exp(rng.normal(0, 1, nb))`; `v = rng.normal(0, 1, 800)`; `signal = exp(1.0*y + 0.5*v[doc] + rng.normal(0, 0.5, nb))`; `redundant = exp(x1 + rng.normal(0, 0.05, nb))`. Each is added as $\log(\cdot+10^{-6})$. |
| Run | Folds seed `[0, 0, 2]`; bootstrap seed `[0, 0, 2]`, $B=2{,}000$; one-cell family, so Holm-adjusted p = p |
| Reference (writer's probe `s3_probe_cv_fixture.py`; the critic's `s3crit/probe_cv.py` gives the same values) | $\mathrm{AUROC}_w(B)$ 0.7384. Noise: +0.0023 [−0.0019, 0.0063], p 0.288. Signal: +0.1352 [0.1101, 0.1613], p 0.0010. Redundant: 0.0000 [−0.0008, 0.0008], p 0.997. Fold seed `[1, 0, 2]`: $\mathrm{AUROC}_w(B)$ 0.7401; noise +0.0015 [−0.0024, 0.0054]; signal +0.1343 [0.1093, 0.1605]; redundant −0.0001 [−0.0009, 0.0007]. Fold seeds 1-4: noise 0.0015-0.0025, signal 0.1343-0.1357, redundant −0.0001 to 0.0001. Two runs identical. |

### 5.5 Noise controls

For row $m$ with center $c_m$ and parameter subset $\mathcal P_m$, sample $s$ of the control is

$$\theta^{(s)}_j=\theta_{c_m,j}+\lambda^{-1/2}E^{(s)}_j\quad (j\in\mathcal P_m),\qquad E^{(s)}_j\overset{\mathrm{iid}}{\sim}\mathcal N(0,1)$$

and every other parameter keeps its center value. This is S2's v2 diagonal Laplace with zero curvature:

$$\tau_j=N_{\mathrm{seq}}\,T^2\cdot 0+\lambda=\lambda,\qquad \sigma_j=\tau_j^{-1/2}=\lambda^{-1/2}$$

So `zero_curvature_state(model, subset)` builds a `LaplaceState` with `phi_hat` = the center's values, `curvature` = zeros, `damping` 0.0 (never used, because no draw is made from this state) and `sampler_version="v1_legacy"`. Then `scale_laplace_state(state, n_data_seqs=1, tokens_per_seq=1, prior_prec=lam)` gives the v2 state. $\lambda>0$ always, so the zero curvature is valid. Sampling, the GPU draw and `apply_sampled_params` are S2's, unchanged. In float32, `posterior_std` at $\lambda=400$ is 0.05 up to a relative error of 1.5e-8, not exactly (T2d).

| Subset | Parameters | Entries |
|---|---|---|
| `ffn` (C0) | `blocks.*.mlp.fc.linear.weight`, `blocks.*.mlp.proj.linear.weight` (`select_params(mode="ffn")`, the C2 set) | 33,554,432 |
| `ffn_mu` (C1) | `blocks.*.mlp.fc.weight_mu`, `blocks.*.mlp.proj.weight_mu` | 33,554,432 |
| `lora_a` (C3 mean) | `*.lora_A` (`select_params(mode="lora")`, the C4 set) | 655,360 |

The C1 control needs one extra rule. `BayesianLinear.forward` draws a VI sample unless `_use_mean` is set (minigpt/layers.py:75-89), and S1's scorer calls the model without `use_mean_weights` (`score_block_batch`). So the `vi_mean` loader builds C1 as `configs/c1.yaml` does, loads the checkpoint, hashes its bytes, and sets `_use_mean = True` on every `BayesianLinear` of its own model instance. It never resets the flag. Without this, every scored pass would add a VI draw on top of the isotropic draw.

Targets. For row $m$ on its fit set $\mathcal D_{\mathrm{fit}}$ (S2's 640 blocks: SE val for `mc_dropout`, `c1`, `c2_refit`; HN val for `c3`, `c4_*`):

$$\Delta\mathrm{NLL}_m=\bar\ell_m(\mathcal D_{\mathrm{fit}})-\ell_0(\mathcal D_{\mathrm{fit}}\mid c_m),\qquad \bar\ell_m=\frac1S\sum_{s=1}^{S}\ell\big(\mathcal D_{\mathrm{fit}}\mid\text{sample }s\big)$$

with $\ell$ as in S2 Section 5.1 (mean token NLL). For `c2_refit` and `c4_lap_refit` the target is `delta_nll_star` of S2's record, and for `c4_tfb_fixed` it is `delta_nll_tfb`. That is the final measurement: S=20, or S=40 after an SE doubling (`matched_noisy`), or the value at $\lambda=1$ for `tighter_than_budget`. A record with a non-zero `exit_code` or any `failures` is refused. For `mc_dropout`, `c1` and `c3`, S3 measures the target with the native sampler. The model comes from the scorer's `load_model` (`c0` under `enable_dropout`, `c1`, `c3`), so the ΔNLL uses the same model code as the scores. `torch.manual_seed(s)` runs before sample $s$, then all 640 blocks run in S2's fit batches of 32. Dropout draws masks per element; VI and BLoB draw one weight sample per forward call, so one native "sample" averages 20 draws. The expected ΔNLL is the same, but the native SE is smaller than an S2 SE. $S=20$ (seeds 0-19). If $\mathrm{SE}>0.05\,\Delta\mathrm{NLL}_m$, the target is re-measured once at $S=40$ (seeds 0-39), as S2's rule does. $\ell_0$ uses the center (dropout off; C1 and C3 under `use_mean_weights`).

$\ell_0$ checks. Each control computes $\ell_0$ with `map_nll` on its own center. For the controls on C0 (`mcd`, `c2`) it must equal `ell0` of S2's `c2` record within 1e-5. For the controls on C3 mean (`c3`, `c4_*`) it must equal `ell0` of S2's `c4_tfb` record within 1e-5. Same base, blocks, batches and autocast, so this confirms that the fit set and base match. For `c1`, the native path's $\ell_0$ must equal the control's within 1e-4.

Match. `fit_noise_controls.py` calls

`match_prior_precision(delta_fn, target, grid, extension, bisect_max_steps, match_tol, se_threshold, n_samples=20, se_max_doublings=1)`

with `delta_fn(lam, n)` = the `delta_nll` and `se` of `measure_delta_nll(model, lambda s: sample_laplace_params(state(lam), seed=s), batches, range(n), anchor_loss=ell0, autocast=cfg["autocast"])`. One draw per ($\lambda$, seed) is reused for all 640 blocks (S2 Section 5.6). The status table is S2's. `tighter_than_budget` cannot occur in practice: at $\lambda=1$ the weight std is 1, about 40 times the median weight.

Records and exit codes. The script writes `fit_record.json` with `write_fit_record`: `record_version`, `method` (`isotropic`), `cell`, `created_at`, `git_commit`, `config_sha256`, `config`, `base` (the `load_posthoc_base` info, or the `vi_mean` equivalent), `manifest`, `fit_units`, `fit_set` (`FitSet.summary`), `ell0`, `ell0_check`, `target_delta_nll`, `target_source` {`kind`, `path`, `sha256`, `field`, `n_samples`}, `native` (native targets only), `match`, `sweep`, `match_status`, `lambda_star`, `sigma_star`, `delta_nll_star`, `se_star`, `param_subset`, `n_entries`, `state_path`, `state_sha256`, `checks`, `failures` and `exit_code`. Exit 0: `matched` or `matched_noisy` with every check passing; only then is `laplace_state.pt` written. Exit 2: a failed check (S2's `EXIT_CHECK_FAILED`). Exit 3: `bisection_failed` (S2's code). Exit 5: `not_matched` or `tighter_than_budget` (S3's `EXIT_NO_CONTROL`). S2 maps `not_matched` to 0, because there a row without a match is a valid "not scored" outcome. For S3 it means the row has no control.

C4 sharing rule. `c4_lap_refit` uses the `c4_tfb_fixed` control if its target lies within ±10% of that control's matched ΔNLL. Otherwise it gets `noise_lora_c3_lap` with its own target. Both S2 targets aim at the same $\Delta\mathrm{NLL}_{\mathrm{TFB}}$, but each lands anywhere in its ±10% band, so the rule can still fail. `mc_dropout` and `c2_refit` both perturb C0's FFN, but their targets differ by about 10x (probe), so they never share.

Scoring. The controls are score sets in S1's scorer with sampler label `isotropic` and method `laplace_v2`, N=20. The loader checks the fit record as S1's scorer checks S2's (exit code 0, accepted status, `state_sha256`, base SHA-256) and returns the zero-curvature v2 state. `checkpoint_paths` lists the base checkpoint and `laplace_state.pt`. The runs use `--first-block-only --domains <list>` with the lists in `scorer_runs.first_block_domains`. That is `test` with the 4 domains for StackExchange rows, and `test_hn` (hackernews) plus `test` (the 3 OOD domains) for HackerNews rows: 4,000 blocks per control. The flag keeps, for each document of the listed domains, its block with the smallest `offset`. `block_index` keeps its full-set value. Batches are consecutive rows of the subset, split by batch size only; S1's split at gaps in $b$ is off for this flag. Each batch's draw is seeded by `seed_start + b` of its first row, S1's per-batch rule. So the same command gives the same file. A first-block file is never compared with a full run. `meta.block_ids` records `first-block:<domains>`. The run tag is `first`, and every S3 score set refuses `--run-tag main`.

The F8 cell uses the same first blocks for the method (a subset of its `main` file, matched on `block_index`) and for the control. Its weights are recomputed within the cell, which gives 1 on first blocks. The files' `weight` fields hold the full-set $1/k_d$ (`BlockSet.select` keeps them) and are not used.

Fixture F-noise (for T2d `::test_flag_fixture`): `rng = np.random.default_rng(505)`; 400 ID and 400 OOD documents with one block each, `y = [0]*400 + [1]*400`, `w = 1`, IDs `f"d{d:04d}"`; control `c = 0.5*y + rng.normal(0, 1, 800)`; near copy `c + rng.normal(0, 1e-3, 800)`; shifted copy `c + 1.0*y`. Seed `[0, 0, 2]`, $B=2{,}000$, one-cell family. Critic probe (`s3crit/probe_noise_flag.py`): control 0.6085 [0.5680, 0.6474]. Near copy 0.6085 [0.5681, 0.6474], CIs overlap, paired Δ 0.0000 [−0.0001, 0.0002], p 0.9125. Shifted copy 0.8344 [0.8061, 0.8626], no overlap, Δ +0.2260 [+0.2092, +0.2441], p 0.0010.

### 5.6 Snapshot ensemble and the identical-copies floor

`c0_snapshot`: the loader reads the 5 members with S2's `load_posthoc_base(full)`. The first member is the host model. The state of the `ensemble` sampler holds the members' parameter dicts (from `named_parameters()`, on the GPU: 5 × 76,301,824 fp32 values, 1.53 GB, plus the 0.31 GB clone that `apply_sampled_params` makes). `sampler(state, seed)` returns member `seed mod N`. Under S1's rule the seed of pass $s$ is $\text{seed}_b\cdot N+s$, so pass $s$ uses member $s$, in the listed order. The rest is S1's path, so $G$ and $M$ come from the same code as every Bayesian row. The run tag is `s3`. The row is labelled "C0 snapshot ensemble (5 checkpoints of one run, cosine decay; weak stand-in, PG11)". S5's `lora_ens3` uses the same sampler with adapter-only members.

`c3_mean_copies`: C3 mean weights (`load_posthoc_base(blob_mean)`), method `deterministic`, N=20, on the HackerNews first blocks (`test_hn`, `--first-block-only --domains hackernews`, run tag `first`), with the same autocast and determinism settings as `c3`. The floor rule:

$$\bar M_{\mathrm{ID}}(\mathrm{C3})\ \ge\ 10\ \overline{\lvert M\rvert}_{\mathrm{ID}}(\text{copies}),\qquad \bar G_{\mathrm{ID}}(\mathrm{C3})\ \ge\ 10\ \overline{\lvert G\rvert}_{\mathrm{ID}}(\text{copies})$$

with $\bar{\cdot}$ the weighted mean over HackerNews test blocks (first blocks for the copies). The copies' $M$ can be slightly negative from float rounding, hence the absolute value.

### 5.7 Matched bins and the stripped copy

For a cell and confound $v$ (pooled ID and OOD blocks, weights $w$): sort $v$ stably, let $\kappa$ be the cumulative weight divided by the total, and take the inner edges $e = \operatorname{unique}\big(\operatorname{interp}(q,\kappa,v_{\mathrm{sorted}})\big)$ for $q\in\{0.2,0.4,0.6,0.8\}$. The bin of a block is `np.searchsorted(e, v, side="left")`, so bin $k$ holds $(e_{k-1}, e_k]$ and bin 0 holds $v\le e_0$. A point mass (for example $r_{\mathrm{tex}}=0$) stays in one bin. A bin is reportable if it holds at least 30 ID and 30 OOD documents. The stratified AUROC over reportable bins $k$ is

$$A_{\mathrm{strat}}=\frac{\sum_k W_k\,\mathrm{AUROC}_{w,k}}{\sum_k W_k},\qquad W_k=\Big(\sum_{i\in\mathcal I_k}w_i\Big)\Big(\sum_{j\in\mathcal O_k}w_j\Big)$$

It is the chance that a random OOD block outranks a random ID block from the same bin. Its CI resamples documents with S1's draw rule and seed `[0, i_id, i_ood]`, $B=2{,}000$, and recomputes the bins' AUROCs with the resampled weights. The bin edges and the set of reportable bins stay fixed from the original sample. $\log L_d$ comes from the manifest.

Fixture F-bin (for T4d): `rng = np.random.default_rng(404)`; 400 ID and 400 OOD documents; `k = rng.integers(1, 4, 800)`; `doc = np.repeat(np.arange(800), k)`; `y = y_doc[doc]`; then `zero = rng.random(nb) < np.where(y == 0, 0.6, 0.15)`; `v = np.where(zero, 0.0, rng.exponential(np.where(y == 0, 1.0, 2.0)))`; `w = 1/k[doc]`; then one draw `e = rng.normal(0, 0.5, nb)`, and `s_a = v + a*y + e` with $a=0$ (confound only) and $a=1$ (signal) from the same `e`. Probe (`s3_probe_bins.py`, re-run by the critic): inner edges [0, 0.077, 0.673, 1.838]; confound only: raw 0.7381, confound 0.7797, stratified 0.5758; signal: raw 0.9151, stratified 0.9220; bin 1 holds 25 ID and 26 OOD documents. A second normal draw for $a=1$ gives raw 0.9071 and stratified 0.9014 instead (Section 6.5).

Stripped copy. S1 scores `mc_dropout`, `c1`, `c3` and `c4_tfb` (pre-fix) on `arxiv_stripped` (S1 Section 2). S3 adds the surface file and the three centers' feature files for it (T1). The stripped rows are descriptive. S2's rows are not scored on the stripped copy (S2 Section 5.5), so they have no stripped row.

### 5.8 Diagnostics, calibration and latency

- Spearman. On first blocks (one per document, no weights), per row and domain: $\rho_S(M,\mathrm{AU})$ and $\rho_S(G,\mathrm{AU})$ with `blk_au` = TU − MI (the expected entropy). The CI resamples the domain's documents with `default_rng([0, i_domain])`, $B=2{,}000$.
- Own-TU bins. The Section 5.7 binning on the row's `blk_tu`, then $\mathrm{AUROC}_w$ of $G$ and $M$ per bin. This repeats the audit's quintile check (metrics-assessment.md Section 3, M3) on the rebuilt set.
- MI scale. Per row, the weighted mean and median of $M$ and $G$ on its ID blocks, and their ratio to the floor (Section 5.6).
- Calibration (P15), from S1's per-token arrays with token weight $w_b$ ($T$ tokens per block), for the 6 rows and for C0 (S1's `c0` file). The C1-mean and C3-mean centers have no per-token arrays, so they are not calibrated.

$$\mathrm{ECE}=\sum_{k=1}^{15}\frac{W_k}{W}\big\lvert\mathrm{acc}_k-\mathrm{conf}_k\big\rvert,\qquad \mathrm{NLL}=-\overline{\log\bar p(y_t)},\qquad \mathrm{Brier}=\overline{1-2\bar p_t(y_t)+\textstyle\sum_v\bar p_t(v)^2}$$

  Confidence is `tok_maxprob`, correctness `tok_correct`, $\log\bar p(y_t)$ comes from `logp_real` by logsumexp, $\bar p_t(y_t)$ is its exponential, and $\sum_v\bar p_t(v)^2$ is `tok_sum_p_sq`. Bins are 15 equal-width bins on [0, 1], with `minigpt.uncertainty.ece`'s edge rule. The CI resamples documents ($B=2{,}000$) from per-document, per-bin sums computed once. Pooled token AURC is dropped (P15; metrics-assessment.md Section 3, m1).
- $G$ at N=3. $G_3(b)$ uses samples 0, 1, 2 of `logp_real`, a valid 3-sample draw for every sampler. $M$ at $N<20$ cannot be rebuilt from the files, so the N=3 column is for $G$ only.
- Latency (P22). `latency_table.json` has one row per score: passes per block (surface model 0 GPU passes; each model score 1; LR and $B_m$ 2; $G$ and $M$ at N; snapshot 5), ms per block at batch 1 and at $b^\star$ from T1(e), and $\mathrm{AUROC}_w$ with CI per OOD domain, at N=3 and N=20 for the stochastic scores. Timing follows S0 Section 5: synchronize before the first timed batch and after the last `.cpu()` copy, sampling and host copies included, model loading and the warm-up batch excluded.

### 5.9 Numbers checked while writing this spec

Session 03, CPU only, with CUDA hidden. Scripts in the scratchpad folder `s03/s3/`: `s3_probe_onepass.py`, `s3_probe_cv_fixture.py`, `s3_probe_noise.py`, `s3_probe_provenance.py`, `s3_probe_bins.py`. The one-pass and noise probes read cache blocks that the HP gates also read, so they informed the design only. They are not results.

| Quantity | Value |
|---|---|
| One-pass probe: blocks | 150 per domain, spread evenly over StackExchange [80M, 100M) and the whole 10M arXiv, FreeLaw and PubMed caches. C0 in fp32. Gaussians from 500 StackExchange-train and 500 Wikipedia-train windows. |
| C0 mean NLL: StackExchange / arXiv / FreeLaw / PubMed | 2.748 / 3.287 / 4.368 / 4.877 |
| arXiv vs StackExchange AUROC: NLL, entropy, max-prob, MD, RMD | 0.637, 0.578, 0.603, 0.677, 0.692 |
| arXiv: LaTeX rate; 7-rate surface model (10-fold CV); 5 model scores (CV); all 12 (CV) | 0.883; 0.961; 0.830; 0.963 |
| FreeLaw: NLL, entropy, max-prob, MD, RMD; surface model; model scores; all 12 | 0.894, 0.880, 0.808, 0.974, 0.976; 0.940; 0.972; 0.977 |
| PubMed: NLL, entropy, max-prob, MD, RMD; surface model; model scores; all 12 | 0.976, 0.984, 0.923, 0.981, 0.983; 0.999; 0.990; 0.999 |
| PubMed and FreeLaw: characters per token alone | 0.935 and 0.796 |
| Share of blocks with LaTeX rate 0: StackExchange / arXiv / FreeLaw / PubMed | 0.66 / 0.11 / 0.89 / 0.99 |
| Spearman(NLL, RMD): arXiv / FreeLaw / PubMed | 0.16 / 0.57 / 0.71 |
| Noise probe: 16 StackExchange val blocks [60M, 80M), C0 fp32, $\ell_0$ | 2.612 nats |
| MC dropout (p = 0.1, 4 masks): ΔNLL | 0.086 nats (3.3% of $\ell_0$) |
| Isotropic noise on C0's FFN (33,554,432 entries, median $\lvert w\rvert$ 0.0247), 2 draws: ΔNLL at $\sigma$ = 3e-4 / 1e-3 / 2e-3 / 4e-3 / 1e-2 | 2.7e-5 / 5.6e-4 / 2.5e-3 / 0.0107 / 0.083 nats |
| Implied $\lambda^\star$ for C0 FFN | about $10^4$ at MC dropout's budget; about $8\times10^4$ at a 0.3% budget (TFB's tolerance). Both inside the grid $10^0$-$10^7$. |
| Provenance: MLflow run `6215391b` | 2026-03-21, 09:44:40 to 14:16:20 local time. 19 step files (5,000-95,000) plus `ckpt_best`, all with mtimes inside the window. |
| Provenance: steps 5,000, 10,000, 80,000, 85,000, 90,000, 95,000 and `ckpt_best` | same key set, 102,033,408 floats and the same embedded config as `ckpt_best` |
| Provenance: `ckpt_step97800` | 103,081,984 floats, mtime 2026-03-17, different config, stored best val loss 7.354: rejected |
| Stored running-best val loss: 80K, 85K, best (88K), 90K, 95K, 10K | 2.551, 2.551, 2.437, 2.437, 2.437, 3.203 |

## 6. Verification evidence

### 6.1 Session-02 refuter verdicts that apply

The three session-02 refuters worked on CPU only, read only. Their evidence is in S1 Section 6.1 and S2 Section 6. What S3 takes from each:

| Verdict | Evidence (from S1 and S2 Section 6) | What S3 does with it |
|---|---|---|
| P1/P19 (eval split): confirmed | ID = StackExchange[80,000,000:80,128,001]; OOD = arXiv[0:128,001], 9 papers, 2 of them 58% of the blocks; FreeLaw and PubMed never scored (scripts/eval_c_checkpoints.py:96-109 before S1; minigpt/data.py:202,207-213) | The 0.931 LaTeX-rate result and the MI-adds-to-surface result rest on 9 papers. S3 re-runs both per domain on 1,000 unseen documents per domain, and adds the surface model and model one-pass scores for FreeLaw and PubMed. |
| P4 (Laplace scale): confirmed | stored std median 1.000; sampled NLL 9.10 (C2) and 9.73 (C4-LAP) against 2.79 (report.md:39,42,45) | A noise control is meaningful only at a matched held-out NLL cost. S3 matches every control to its method's ΔNLL on S2's fit sets, with S2's rules. |
| P5 (TFB rotation): confirmed | C4-TFB's A equals C3's `lora_A_mu` exactly; C4-TFB is BLoB-mean + TFB | The C4 rows' center is C3 mean, so their noise controls and one-pass scores use C3 mean. The HackerNews likelihood ratio uses C3 mean against C0. |

### 6.2 Session-01 audit numbers that S3 builds on

From `research/2026-09/metrics-assessment.md` and `research/2026-09/audit/audit_d1_metrics.json` (recomputed from saved scores; old 1,000 blocks):

| Finding | Number | Used in |
|---|---|---|
| LaTeX rate vs MI | rate AUROC 0.931 (FPR@95 0.342); MI 0.874-0.917 | P2; the `sf_tex` feature |
| MI added to the rate, 8 cluster folds | rate alone 0.929 (per-fold mean) or 0.917 (pooled out-of-fold); with MI 0.966-0.975 or 0.961-0.971 | the combined test; pooled out-of-fold is S3's metric |
| sklearn `GroupKFold(8)` on 16 clusters | 2 one-class folds; pooled AUROC 0.910 (an artifact) | S3's folds are class-stratified, so every fold has both classes (T3b) |
| Low-LaTeX quarter of arXiv (473 ID, 125 OOD blocks) | MI 0.883-0.953; rate 0.821 | the matched bins |
| Pooled quintile bins of the rate | first bin empty (ties at 0: 65.8% of ID blocks, 5% of OOD blocks); other bins 371/29, 103/97, 20/180, 6/194 ID/OOD blocks | S3's bin rule keeps a point mass in one bin and reports only bins with 30+ documents per class |
| Spearman(MI, AU) | 0.40-0.52 for MC dropout, C1, BLoB, TFB; −0.89 (C2) and −0.74 (C4-LAP) | T5a |
| MI within own-TU quintiles | 0.77-0.98, lowest in the top quintile | T5a |
| ID mean MI | BLoB 0.0035 nats, TFB 0.079 nats (22x) | P23; the floor and the MI-scale table |
| C0 checkpoints | 492 of 511 step files come from other runs | the provenance rule |
| Reviewer scratch runs (paper-critique.md item 19, not reproduced) | C0 NLL AUROC 0.920 (FreeLaw), 0.982 (PubMed); MC dropout MI 0.861, 0.970 | the ceiling rule; NF1 and NF2 |

### 6.3 The writer's own probes (session 03, CPU only, read only)

| Probe | Result | What it changed in this spec |
|---|---|---|
| One-pass baselines on cache blocks (Section 5.9) | PubMed: every one-pass family reaches 0.98-0.999. FreeLaw: MD/RMD 0.974-0.976; all 12 features 0.977. arXiv: the 7-rate surface model 0.961, above the LaTeX rate alone (0.883). | The ceiling rule ($\mathrm{AUROC}_w(B_m)>0.98$) and NF2. Surface features are 7 rates, not the LaTeX rate alone, because the LaTeX rate is weak on FreeLaw and PubMed (0.37, 0.33) while characters per token is strong (0.80, 0.94). |
| Combined test, fixture F-cv, nested choice of $C$ in {0.001, ..., 100} by inner-fold AUROC | $C=0.001$ won in 6 of 10 folds. $\mathrm{AUROC}_w(B)$ fell from 0.7384 to 0.7202. A near-copy of $x_1$ then "added" +0.0165 [0.0072, 0.0263], p = 0.001. | $C$ is fixed at 1.0. With $C=1$ (or $10^6$) the near-copy adds 0.0000 [−0.0008, 0.0008] (or −0.0011 [−0.0031, 0.0007]). |
| Combined test, fixture F-cv, $C=1$ | noise +0.0023 [−0.0019, 0.0063]; signal +0.1352 [0.1101, 0.1613], p = 0.0010; reruns identical; 4.7 s for 18 runs of 1,609 blocks | T3c reference values; the CPU estimate |
| Noise scale on C0's FFN | ΔNLL grows about 4x per doubling of $\sigma$ from 1e-3 to 4e-3, and 7.8x from 4e-3 to 1e-2. MC dropout's budget (3.3%) is about 11x a 0.3% budget. | $\lambda^\star$ sits inside S2's grid for both kinds of target. MC dropout and C2 never share a control. |
| Snapshot provenance | 5 members and the step-10,000 background pass all four rules; step 97,800 fails all four | T1c; the member list in the registration |
| Matched bins, fixture F-bin | A 5-level discrete confound merged into 3 bins and left residual confounding (stratified 0.577). A point-mass confound behaves as intended (0.5758 vs raw 0.7381; signal kept, 0.9220 vs 0.9151). | The fixture uses a point mass, as the LaTeX rate has. The "poorly matched" report flags residual confounding in real bins. |

### 6.4 Differences from roadmap Section 3 and the options document

| # | Roadmap or options document | This spec | Why |
|---|---|---|---|
| 1 | Endpoint: "the AUROC gain from adding MI" | $G$ added in the primary families F5-F6; MI in F7 | S1 pre-registered $G$ as the primary score. Using MI here would be a second primary chosen later. |
| 2 | "the best cross-validated combination of one-pass scores" | An L2 logistic regression on all 13 one-pass features of the row's own center, $C=1$ fixed | "Best" by an inner search produced a false gain in the probe (Section 6.3). The own center is the network a user of that method already has. |
| 3 | No ceiling rule | Ceiling at 0.98 | PubMed one-pass 0.999 and FreeLaw 0.977 in the probe. Without the rule, "adds nothing" would be claimed where nothing could add. |
| 4 | Fixture: "that CI contains 0" | Also a pre-registered "no useful gain" rule (CI upper end below 0.02) for real cells | A CI that contains 0 does not show that the gain is small |
| 5 | "Random Gaussian weight noise scaled to each method's held-out ΔNLL (±10%)" | Per-method controls on the method's own subset and center, matched with S2's sweep, scored on first blocks; flag by CI overlap as written, plus a paired test (F8) | Overlapping CIs is not a test (paper-critique.md item 18). First blocks halve the cost. |
| 6 | Likelihood ratio "log p_C3 − log p_C0" | That ratio for the HackerNews rows; C0 against its own step-10,000 checkpoint for the StackExchange rows | C3 is not a model of StackExchange, so the first ratio has no clear direction there (metrics-assessment.md Section 3, M2) |
| 7 | Snapshot "C0 at 60K, 80K, 90K and best" (options m8) | 80K, 85K, 88K (best), 90K, 95K, each checked for provenance | 60K is a worse model (stored val loss 2.70). Roadmap Section 6, row 5 requires the provenance check. |
| 8 | Bins on character rate and NLL | Also document length | S1 hands the length control to S3 (S1 Section 7); P13 names it |
| 9 | Not listed | No method-vs-method families by default (Choice 4) | S1 Sections 3 and 7 name S3 for method ranking, but that answers P6 and P14, which are not S3's issues. The draft's F9-F10 are now the alternative of Choice 4. |
| 10 | Relative Mahalanobis | MD and RMD | MD (Lee 2018) is the classic baseline and costs nothing extra; P3 names Mahalanobis |
| 11 | "ms per block at N=3 and N=20" | Measured by a timing pass; AUROC at N=3 from the first 3 of the 20 stored samples, for $G$ only | $M$ at N=3 cannot be rebuilt from the score files |
| 12 | GPU 0.5-1.5 h | 1.6-7.8 GPU h (3x gain to batch-1 rates) | Section 1: the noise controls cost 1.4-7.1 |
| 13 | Calibration with CIs | For the 6 rows and C0 only | The C1-mean and C3-mean centers have no per-token arrays |

### 6.5 Critic pass (session 03, CPU only, read only)

The critic read the draft against `scripts/eval_c_checkpoints.py`, `scripts/check_eval_rebuild.py`, `minigpt/posthoc_refit.py`, `minigpt/laplace.py`, `minigpt/layers.py`, `minigpt/uncertainty.py`, `configs/i2_eval.yaml`, the S2 refit YAMLs, S0's JSON schema, S5's draft and roadmap Section 3. Probe scripts are in the session scratchpad folder `s3crit/`.

| Finding | Evidence | Change in this revision |
|---|---|---|
| Feature files `onepass__{center}__{eval_set}.pt` in `data/scores_i2/` would be parsed by S1's checker as score files of eval set `{center}`, and S1's `--check align` would then fail | `parse_score_name` splits any `*.pt` stem on `__` into 3 parts; `run_check_align` groups by the middle part (scripts/check_eval_rebuild.py:131-137, 1064-1110) | All S3 files except the scorer's own score files move to `data/scores_i2/s3/` |
| Adding S3 score sets to `scoring.score_sets` adds cells to S1's tables if a `main` file exists | `planned_table_cells` in both S1 scripts reads `scoring.score_sets`; cells read only `main` files and a missing file is "not scored", not a failure (scripts/eval_c_checkpoints.py:1150-1164, 1199-1211; check_eval_rebuild.py:530-549, 906-914) | S3 never writes `main`; the scorer refuses it for S3 sets; T1(g) runs S1's three checks after S3's runs |
| `scoring.batch_size` is keyed by eval set, not by score set | configs/i2_eval.yaml; `run_i2` (scripts/eval_c_checkpoints.py:1055-1056) | No `batch_size` entries for S3 sets; `--batch-size` per run, with the $b^\star$ mapping from S0 |
| Under S1's batching, first blocks are split at every gap in $b$ | `_batches` (scripts/eval_c_checkpoints.py:701-714) | `--first-block-only` batches consecutive subset rows; without it the first-block rows would cost 1.2-1.4 GPU h more |
| The C1 control would add a VI draw to every scored pass | `BayesianLinear.forward` samples unless `_use_mean`; `score_block_batch` has no `use_mean_weights` (minigpt/layers.py:75-89; scripts/eval_c_checkpoints.py:663-672) | The `vi_mean` loader sets `_use_mean` permanently; T2(d) `::test_vi_mean_control` |
| `match_prior_precision` has keyword-only `n_samples` and `se_max_doublings`; `not_matched` exits 0 in S2 | minigpt/posthoc_refit.py:75-81, 684-695 | The call and S3's own exit codes (Section 5.5); T2(d) `::test_no_control_exit` |
| S2's `check_config` rejects `method: isotropic`, and `load_posthoc_base` has no `vi_mean` kind | minigpt/posthoc_refit.py:67-68, 153-165, 222-265 | S3 checks the noise YAMLs against `REQUIRED_KEYS` itself and owns the `vi_mean` loader |
| The TFB record has no `match_status`; the target field differs by method; `doc_ids_sha256` is None for `token_range` fits | minigpt/posthoc_refit.py:495, 1034-1065, 1150-1192 | Target fields named per method; T2(b) compares `fit_set.blocks_sha256` and adds $\ell_0$ checks against S2's `ell0` |
| `matched_noisy` can end with a final ΔNLL outside the band | `match_prior_precision` docstring and lines 797-804 | T2(a) checks `match.delta_nll_first` for both statuses and the final ΔNLL for `matched` |
| T2(d) "posterior_std 0.05 to 1e-12" fails in float32 | critic probe `probe_zero_curv.py`: max abs error 7.45e-10, relative 1.5e-8 | relative tolerance 1e-6 |
| The synthetic match test gave no expected $\lambda^\star$ | critic probe: `matched`, $\lambda^\star$ = 23,713.7, ΔNLL 0.01054, 3 bisection steps, bracket $[10^4, 10^5]$ | T2(d) pins these values |
| The flag fixture had no construction | critic probe `probe_noise_flag.py` | Fixture F-noise and its reference values (Section 5.5) |
| F-cv reference values | critic re-implementation from Section 5.4 alone (`probe_cv.py`) reproduced every value to 4 decimals, including fold seed `[1, 0, 2]` | T3(c) pins them within ±0.001 |
| F-bin was ambiguous: a second normal draw for $a=1$ gives raw 0.9071 and stratified 0.9014, not 0.9151 and 0.9220 | critic probe `probe_bins.py` | Section 5.7 specifies one shared draw `e` |
| `LogisticRegression(penalty="l2")` warns in sklearn 1.8.0 | FutureWarning "'penalty' was deprecated in version 1.8 and will be removed in 1.10"; `l1_ratio=0.0` gave identical coefficients | YAML key `l1_ratio: 0.0` |
| `z_pool` in float16 would stop the re-derivation from recomputing MD to 1e-6 | fp16 rounding is about 1e-3 relative | `z_pool` float32; MD from the stored values |
| The YAML typed the train ranges a second time | S1 Section 2 rule; `load_refit_data` computes them (minigpt/posthoc_refit.py:330-389) | `range_source` names the S2 YAMLs |
| Calibration "per center" had no data for C1 mean and C3 mean | the one-pass feature files hold block values only | Calibration for the 6 rows and C0 |
| Out of m8/m9 and P2, P3, P13, P15, P22, P23: F9-F10 (P6, P14), the N sweep with $\phi_N$ (P10, owned by S6 and m3), the prefix curve (paper-critique.md item 20, generated-output latency) | options-and-costs.md Sections 2 and 4.1; roadmap Section 3 | Removed; Choice 4; Section 7 |
| Cost figures | CA10 rates with a per-pass draw for the scored controls (14.35 and 15.8 ms, not 8.1); all three native targets may double; night B cannot share S5's night 1 | Section 1 tables; night B after S5's night 2 |
| Name clash with S5's draft (families F5, F6) and S5's "Feeds S3" line | specs/i2-lora-seeds.md:15, 76 | Recorded in Section 2 ("With S5") for S5's critic |

## 7. What this does not cover

- Generated tokens, G-NLL and answer-level scores (S6, P7). Every S3 score is on 256-token blocks of reference text.
- A real aleatoric control or exposure ground truth (S7). Spearman(MI, AU) and TU bins are diagnostics, not a separation test (P13 stays partly open).
- The deterministic-LoRA post-hoc rows and the LoRA ensemble (S5). S5 reuses S3's code, including noise controls for its b2 fits, under its own frozen families. S5's draft lists these as a feed to S3; S3 does not take them up.
- Method-vs-method ranking (P6, P14). Choice 4: by default no S3 family ranks methods, although S1 Sections 3 and 7 point to S3. S5, which owns P14, may register a ranking on its seeded rows.
- The N sweep beyond N=3 and N=20 and the fraction-above-chance $\phi_N$ (P10; S6 and m3). The latency table gives AUROC of $G$ at N=3 and N=20 with CIs.
- The prefix curve, AUROC over the first $k$ tokens (paper-critique.md item 20). It concerns generated output with low latency, so it belongs to S6 or the backlog.
- A retrained deep ensemble. The snapshot ensemble is one run with a decayed learning rate, so it is weak (PG11).
- kNN distance, energy scores, other layers, token-level Mahalanobis and trained hidden-state probes. The combined model already uses the test labels by cross-validation, so it bounds what a linear read of these 13 features can do.
- A detector one could deploy. The combined model is fitted on labelled test documents (Section 3.4, scope sentence).
- Refit variance in the combined-test CI. The bootstrap resamples fixed out-of-fold scores. The fold-seed range is reported instead.
- Multiplicity across families. Holm runs within each family only.
- Training-seed variance (P14). Every row is one run.
- An ID-vs-ID row (Wikipedia vs StackExchange) and near-OOD pairs matched on markup. S1 has no Wikipedia test set, and a benchmark redesign is in the backlog.
- Latency during generation and merged-LoRA timing (S6-T5; m3). S3 times teacher-forced blocks with S1's scorer.
- Calibration decisions and fixes such as temperature scaling. Calibration is reported only. The C1-mean and C3-mean centers are not calibrated.
- S2's rows on the stripped copy.
- Any change to S1's primary score, families, tables or scorer behaviour for existing score sets and flags.

Handoff to m11 and m13. S3 edits no text. These claims depend on S3's outputs:

| Text | Where | S3 output that decides it |
|---|---|---|
| "MI is the only effective uncertainty score … weight-level disagreement is required" | paper.tex:213 | F5-F7 verdicts, NF3 and the surface-model row |
| Missing baselines; character-count row not in the main table | paper.tex Table 2 (PG5) | `baselines_table.json` |
| "N=3 captures 97% of the signal" | paper.tex:52,259,277; report.md:130 | `latency_table.json`: AUROC of $G$ at N=3 and N=20 with CIs. The "fraction of the signal" wording is P10 (m3, S6). |
| "Zero overhead" and the production section | paper.tex:95,299 | `latency_table.json` (teacher-forced blocks only) |
| Calibration columns without CIs | paper.tex Table 1 | `calibration.json` |
| Any sentence crediting the posterior's shape | paper.tex Section 6 | F8 verdicts and flags |

## 8. Confidence

| Claim | Confidence | Why |
|---|---|---|
| The combined test gives no false gain for noise or for a redundant feature at $C=1$ | high | Pinned fixtures: noise +0.0023 [−0.0019, 0.0063]; near-copy 0.0000 [−0.0008, 0.0008]; stable over 4 fold seeds. The critic's independent re-implementation gave the same values. |
| The fixture reference values in T2-T4 are right | high | F-cv, F-bin (with the shared draw), F-noise and the synthetic match were each re-run by the critic with the repo's functions |
| S3's scorer runs leave S1's tables and checks unchanged | medium-high | S1's code reads only `main` files and globs only its top folder (Section 6.5). T1(g) tests it on the real files. |
| One-pass scores are at or near ceiling on PubMed and close to it on FreeLaw | medium-high | Probe on 150 cache blocks per domain (0.977-0.999), and a reviewer's GPU run (0.920, 0.982). The rebuilt set uses whole documents with 257+ tokens, which may shift the numbers. |
| On arXiv the surface model is the strongest one-pass family | medium | Probe 0.961 vs 0.830 for model scores. The rebuilt set has 1,000 papers, not 9. |
| Every noise control reaches `matched` | medium | ΔNLL is smooth and monotone in $\sigma$ in the probe, and $\lambda^\star$ sits inside the grid. The SE at S=20 is not measured on the GPU. |
| The snapshot member list is right | high | mtime, step, key set, parameter count and config all checked on the files |
| The BLoB floor check passes | medium-high | With deterministic kernels, identical copies should give $\lvert M\rvert$ near 1e-7, far below BLoB's 0.0035. Not run. |
| Engineering fits 12.5-21.5 h | medium-low | Built up by part. The independent re-derivation and the scorer entries may push it up. |
| GPU fits 1.6-7.8 h | medium-low | CA10 rates, S1's estimated block counts and S2's λ-equivalent counts. The batching gain is unmeasured (S0), and the v2 GPU draw cost per scored pass is assumed equal to C1's VI draw. |
| The registration precedes S1's test scores | medium | It needs the approval commit before the m4 re-score. The schedule moved earlier than the roadmap. Choice 1 records a late approval. |
| Weight sampling adds a useful amount beyond one-pass scores | unknown | That is the pre-registered question. The probe suggests little room on FreeLaw and PubMed and some room on arXiv. |
