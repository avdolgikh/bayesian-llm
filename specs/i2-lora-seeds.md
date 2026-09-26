# S5 LoRA seeds: three BLoB and three deterministic LoRA seeds, refits on deterministic LoRA, LoRA ensemble (`specs/i2-lora-seeds.md`)

Status: DRAFT (not approved). Date: 2026-09-26. First draft, then one critic pass that checked it against the built S0-S2 code (Section 6.4). Owner: Alexey approves; agents implement.

## 1. Header

| Field | Value |
|---|---|
| Items | **b1**: this spec. It fixes the seed plan, the seed script (about 30 lines of CLI), the refit plan, the LoRA ensemble and the pre-registered comparison. **b2**: train 3 BLoB and 3 deterministic LoRA seeds on HackerNews. Refit TFB and diagonal Laplace-LoRA on each deterministic LoRA with S2's script. Build a LoRA ensemble (M=3) from the 3 deterministic LoRAs. Re-score all of them with S1's scorer. Report 3-seed spreads. |
| Issues | **P3**: no retrained ensemble baseline. S5 covers the LoRA-ensemble part only. S3 covers the one-pass baselines. **P14**: one training seed per method, and the C pipeline never seeds. S5 covers the LoRA rows only. **P18**: the C4 post-hoc fits start from the C3 BLoB posterior mean, not from a deterministic LoRA. |
| Approve by | Thu 2026-10-01 (roadmap Section 3, row S5) |
| Build | The first agent session after approval: session 04, Mon 2026-10-05, in the roadmap. The session-03 plan keeps the seed script out of session 03 (agents/plans/session-03-plan.md, "Out of scope"). That settles the timing clash that S0 Section 7 notes. |
| Run | Two nights by default. Night 1 (Tue Oct 6): the 6 fine-tunes, then the 6 refits. Night 2 (Wed Oct 7): the re-score. `--plan` runs the morning after night 1 and writes `data/i2/s5_plan.json` (Section 5.6): the block subset and the night-2 list. The fine-tunes depend on no other spec. If the seed script is built and reviewed before Sat Oct 3, they may run on the Sat Oct 3 or Sun Oct 4 night, which the roadmap leaves free. No agent session is planned before Mon Oct 5, so this needs an extra one. |
| Depends on | **S0** (`data/i2/timing_probe.json`): `rescore.cells` (methods `c3`, `c4_tfb`, `c4_lap`; `ms_per_block` per batch size), `estimates.m4.batch_size` ($b^*_m$), `finetune.wall_min` ($t_D$) and `estimates.s5_finetune_gpu_h.n_blob_3`. **S1**: the manifest `val` rows (for the fits), the `test` and `test_hn` block files, `build_report.json` (`domains.<key>.n_blocks` and `domains.<key>.kept`), the scorer with $G$ and per-sample log-probs, the document bootstrap, the freeze mechanism, and the checker's `--check align` and `--check prereg`. **S2**: `scripts/refit_posthoc.py` with `base_kind: det_lora`, the v2 samplers, `load_posthoc_base` and the fit record. The C4 refits must pass the fit part of S2-T4(d) (both runs exit 0, no `bisection_failed`, median check passed) before any b2 refit starts. The re-score part of S2-T4(d) is not needed first. |
| Feeds | S3: the LoRA ensemble as the retrained-ensemble baseline, and the ΔNLL of each b2 fit for the noise control. S4: `table_s5.json`. m11: the LoRA block of the main table and the seed spreads. m13: the P18 disclosure becomes a row label ("deterministic LoRA + TFB"). |

Costs. The roadmap column is roadmap Section 3, row S5 (b1 plus b2 from options-and-costs.md Section 4.2). The next column is this spec's own estimate.

| Cost | Roadmap row S5 | This spec | Difference and why | Assumption |
|---|---|---|---|---|
| Alexey, approve | 0.25 (b1) | 0.25 | none | CA1 |
| Alexey, review | 0.25-0.5 (b2) | 0.75-1.5: code 0.5-1.25, results 0.25 | +0.5-1.0 h. CA1 prices review at 1 h per 6-9 engineering hours. The new code is 4.75-7.5 engineering hours (seed script, ensemble scorer, analysis). The roadmap figure did not apply CA1 to b2. Choice C2 below keeps the roadmap figure. | CA1 |
| Engineering | 6.5-10 (b1 0.5-1; b2 6-9) | 6.5-10.75 (est.; table below) | +0-0.75 h: the frozen S5 family, the plan rule and the independent re-derivation | CA2 |
| GPU, fine-tunes | inside 2-8 | 1.7-2.7; up to 6.7 at CA4's 4x tail on the deterministic runs | +0.45 h for the third BLoB seed (options b1 had 2 more BLoB seeds) | CA4; S0-T2 |
| GPU, refits | inside 2-8 | 1.9-7.65 | S2's method replaces the old fits (7 min for TFB, 17 s for Laplace): an 18-step $\sigma_q$ search (pre-check plus 17 bisection steps) at $S=10$ and 12-19 λ values at $S=20$, over 640 blocks, per seed | CA10; S2 Section 1 |
| GPU, re-score | inside 2-8 | 3.35-11.85 on all blocks; 5.29 on the first block of each document. Under the plan rule: 3.35 (3x gain, all blocks) to 5.29 (first block), and never above 8 unless the rule escalates | 10 score sets on S1's 9,500-11,200 `test` and `test_hn` blocks. The first-block subset is priced at batch 1, because the scorer starts a new batch at every gap in the block index (scripts/eval_c_checkpoints.py:701-714), so the subset gets little batching gain. b2 priced 2 extra seeds at 1-2 blocks per document. | CA10; S1 Section 5.8 |
| GPU, total | 2-8 | 6.9-15.6 at the two ends of the range under the plan rule (Section 5.6). The rule caps the re-score at 8 GPU h, so the worst case is 18.4. 22.2 if all blocks are forced. | +4.9-7.6 h at the ends of the range; 2 nights instead of 1, 3 in the worst case | CA3 |
| Opus agents | not in the row | 5-8 | CA6 gives 5-8 per implementation spec | CA6 |
| Cloud | $0 | $0 | none | |
| Disk | not costed | about 4.5 GB: 6 checkpoints of 0.33 GB each; 9 score sets of about 0.27 GB and 1 of 0.1 GB | new | est. |

Engineering breakdown (est.):

| Part | Eng h | Source |
|---|---|---|
| Seed script, two YAMLs, S5-T1 | 1-1.5 | options b2 (W3: seed script 1 h) |
| 6 refit YAMLs, `--check provenance`, the unit parts of S5-T2 | 0.5-1 | est. |
| Scorer: 10 score-set entries, the S5 refit map, the ensemble sampler, `--block-ids-file`, the S5 freeze check, S5-T3 | 2-3 | options b2 (W3: LoRA ensembles 3 h); code-assessment.md K14 (3 h) |
| Seed-mean paired bootstrap, decision rule, `scripts/analyze_lora_seeds.py` (`--plan`, `--analyze`), S5-T4 and S5-T5 | 1.75-3 | est. |
| `--rederive-s5`, written by a separate agent | 0.5-1 | est. (the m15 pattern from S1) |
| Night lists with the night-1 log, launches, checks | 0.5-1 | options b2 (W3: seeds 3 h, mostly runs) |
| Doc updates | 0.25 | est. |
| **Total** | **6.5-10.75** | |

GPU breakdown (est.). Low end: a 3x batching gain (S0-T1) and $t_D$ = 7 min. High end: batch 1 at CA10 rates and $t_D$ = 27 min.

| Part | Count | GPU h |
|---|---|---|
| BLoB fine-tunes | 3 × 26.9 min (C3's `train_time_sec`, MLflow run 3b22bfcf) | 1.35 |
| Deterministic fine-tunes | 3 × $t_D$; 7-27 min (options b2), measured by S0-T2 | 0.35-1.35 (5.4 at CA4's 4x tail) |
| TFB refit per seed | S2's C4 figures: search 0.75, ΔNLL re-measure 0.08-0.17, MAP and curvature 0.05 or less | 0.29-0.97 |
| Laplace refit per seed | S2's C4-LAP λ sweep: 12-19 λ values × 20 samples × 640 blocks at 23.4 ms | 0.33-1.58 |
| Refits, 3 seeds | 3 × (0.63-2.55) | 1.9-7.65 |
| Re-score per block | $3\,(0.316+0.469+0.469)+\tfrac{3}{20}\,0.316=3.806$ s at batch 1 (CA10: BLoB = C3; Laplace-LoRA = TFB; ensemble = 3 deterministic-LoRA passes, priced as 3/20 of C3) | |
| Re-score, all blocks | 9,500-11,200 blocks | 10.05-11.85 at batch 1; 3.35-3.95 at 3x |
| Re-score, first block of each document | 5,000 blocks | 5.29, priced at batch 1 (see the re-score row above) |

Effect on the calendar and on G1:

| Item | Effect |
|---|---|
| Nights | 2 in the typical case, against 1 in the roadmap (Tue Oct 6). Night 1: fine-tunes, then refits (3.6-10.35 GPU h; at the top end it runs past one night, and the queue continues before the re-score). Night 2: re-score (3.35-5.29 GPU h under the plan rule; at most 8 GPU h unless the rule escalates). The worst case needs a third night. One night holds 8-10 GPU h (CA3). |
| W2 GPU | Roadmap: 3.2-11.0 GPU h over 2 nights (Mon Oct 5 1.2-3.0, plus b2's 2-8). With this spec: 8.1-18.6 GPU h over 3 nights (4 in the worst case). |
| October GPU total | Roadmap: 4.9-22.6 GPU h. With this spec: 9.8-30.2 GPU h. The 4070 has 215-270 GPU h to Oct 25 (CA3). |
| W2 hours | +0.5-1.0 h over the roadmap's 4.0-5.75 h. W2 is then 4.5-6.75 h, over the 5 h ceiling in most of the range. |
| G1(b) | The plan through Oct 11 rises from 7.5-11.0 h to 8.0-12.0 h. Stage A is then less likely to join v1. |

## 2. Scope and non-scope

What changes:

| File | Change |
|---|---|
| `scripts/train_lora_seed.py` (new; about 30 lines of CLI plus `train_one_seed`, about 90 lines in total, est.) | Trains one LoRA of kind `blob` or `det_lora` for one seed from the YAML list. It writes `ckpt_best.pt` and `seed_record.json`. It has no `--resume` flag. It never overwrites an output folder. |
| `configs/i2_lora_seed_blob.yaml`, `configs/i2_lora_seed_det.yaml` (new) | C3's phase-2 schedule with every key explicit (Section 5.2) |
| `configs/i2_posthoc_b2_{tfb,lap}_s{201,202,203}.yaml` (6 new) | Copies of S2's `configs/i2_posthoc_c4_tfb.yaml` and `configs/i2_posthoc_c4_lap.yaml` that differ only in the keys of Section 5.3 |
| `configs/i2_eval.yaml` (S1's file), `scoring` section only | The 10 new score sets: `n_samples`, `score_sets.test`, `score_sets.test_hn` and `labels` (Section 5.4). The `analysis` block and its hash do not change. |
| `configs/i2_eval_s5.yaml` (new) | The frozen `analysis_s5` block: families F5 and F6, the descriptive items and the plan rule (Section 5.7) |
| `scripts/eval_c_checkpoints.py` (S1's scorer) | 10 entries through `register_score_set`. A scorer-local map for the 6 b2 refit sets. An ensemble parameter sampler through `register_param_sampler`, so the members pass through the existing per-sample loop and `apply_sampled_params`. A `--block-ids-file` flag. A freeze check against `prereg_s5.json` for the S5 score sets only, and the meta key `analysis_s5_sha256` in their files. Details in Section 5.4. The existing score sets behave as before, and their tests pass unchanged. |
| `minigpt/uncertainty.py` | New `seed_mean_paired_doc_bootstrap(...)` next to S1's `paired_doc_bootstrap`, on the same draws. Nothing else changes. |
| `scripts/analyze_lora_seeds.py` (new) | `--plan` (Section 5.6), `--check provenance` (S5-T2), `--check prereg-s5`, `--analyze` (tables and families), `--freeze` for `prereg_s5.json` |
| `scripts/check_eval_rebuild.py` (S1's checker) | New `--rederive-s5`, written by a separate agent from Sections 3, 5.5 and 5.7 of this spec only. It imports neither `minigpt.uncertainty` nor the eval or analysis scripts. |
| `tests/test_lora_seeds.py` (new) | S5-T1 to S5-T5, CPU only, no reads of `data/` |
| `data/checkpoints/i2_seeds/{blob,det}_s{seed}/` (new, gitignored) | 6 folders with `ckpt_best.pt` and `seed_record.json` |
| `data/checkpoints/i2_posthoc/b2/{tfb,lap}_s{seed}/` (new, gitignored) | 6 refit states and `fit_record.json` files, written by S2's script |
| `data/scores_i2/` (gitignored) | 20 score files named `{score_set}__{test,test_hn}__s5.pt`, `prereg_s5.json`, `table_s5.json`, `families_s5.json`, `rederive_s5.json` |
| `data/i2/` (gitignored) | `s5_night1_log.json` (start, end and exit code of every night-1 command), `s5_plan.json`, `s5_provenance.json`, and `s5_block_ids_{test,test_hn}.json` if the plan picks the first-block subset |
| AGENTS.md, README.md, `agents/technical-reference.md`, `agents/design-rationale.md`, `agents/milestone-history.md` | The scripts line and the test count (AGENTS.md, README.md). The seed script and the S5 files (technical-reference). The decisions in Sections 6.3 and 6.4 (design-rationale). The run IDs at completion (milestone-history). |

What must not change:

- `data/checkpoints/{c0,c3,c4_tfb,c4_lap}/`, `data/checkpoints/i2_probe/` (S0's adapter) and `data/checkpoints/i2_posthoc/{c2,c4_tfb,c4_lap}/` (S2's refits). They are read only. C3 is not retrained and not overwritten.
- S1's `analysis` block, `prereg.json`, families F1-F4, and S1's result files (`table_*.json`, `families.json`, `descriptive.json`). S1's `--analyze` and `--rederive` read only `main` score files, so the S5 files (run tag `s5`) never enter them. S1 lists the 10 new score sets as not scored; nothing else in its output changes.
- S2's module (`minigpt/posthoc_refit.py`), samplers and scripts. S5 adds YAML files and scorer entries only. If S5 needs a change in S2's code, it goes back to S2 as a request (choice C4).
- `minigpt/train.py`, `minigpt/lora.py`, `minigpt/model.py`, `minigpt/laplace.py`, `minigpt/tfb.py`, `minigpt/data.py`, and `bootstrap_ci`.
- `experiments/` and `.pipeline-state/`. The C pipeline stays the frozen record of the C results.
- `mlflow.db`. The seed runs log no MLflow run.
- `paper/`, report.md and the README numbers (m3, m11, m13).

Interface with S0, S1 and S2, checked against the built code. Line numbers for `scripts/eval_c_checkpoints.py`, `scripts/check_eval_rebuild.py`, `minigpt/posthoc_refit.py`, `minigpt/evalset.py` and `minigpt/laplace.py` refer to the uncommitted session-03 working tree of 2026-09-26.

| S5 needs | Built code | Status, and what S5 does |
|---|---|---|
| Held-out HackerNews documents for the refits | S1 manifest rows with `split: val` inside [80M, 90M). S2's `fit` section (`fit_split: val`, `fit_domain: hackernews`) and `check_fit_isolation` (posthoc_refit.py:434) | Matches. The P19 fallback (token ranges) applies unchanged. |
| The eval blocks | `blocks_test.pt` and `blocks_test_hn.pt` with `tokens`, `doc_id`, `domain`, `offset` and `weight`, checked on load (eval_c_checkpoints.py:253-299) | Matches |
| A score file with per-sample realized-token log-probs | Payload (eval_c_checkpoints.py:759-775): `logp_real` [B, N, T] float32, `tok_*`, `blk_g`, `blk_mi`, `blk_tu`, `blk_au`, `blk_nll`, `blk_maxprob_unc`, `block_index`, `doc_id`, `domain`, `offset`, `weight`. Meta (:1094-1119): `checkpoint_sha256` as a {path: sha256} dict, `sampler` and `adapter_source` from `scoring.labels`, `block_ids` (the CLI string), `n_samples`, `created_at`. | Matches, with two consequences. (1) `weight` is copied from the full block file (:773), so a subset file stores the full-set $1/k_d$. (2) A subset file tagged `main` would fail S1's `--check align` (check_eval_rebuild.py:1085-1088: a `main` file must have the reference `block_index`) and S1's `--rederive` weight check (:464-466). So every S5 file uses run tag `s5`, and the S5 analysis recomputes the weights per cell, as S1 does (eval_c_checkpoints.py:1229-1232). S1's `--check align` matches non-`main` files by `block_index`, so the S5 files pass it. |
| New score sets | `register_score_set` (:449-458). `run_i2` reads `scoring.n_samples`, `scoring.labels` and `scoring.score_sets` (:1028-1038) and refuses a run whose input files are missing (:1041-1047) | S5 registers 10 entries. The night-2 list names the scoreable sets with `--score-set`. |
| A per-sample hook for the ensemble | `PARAM_SAMPLERS` and `register_param_sampler` (:443-467). `score_block_batch` calls `PARAM_SAMPLERS[method](state, seed=seed_b * N + s)` and applies the result with `apply_sampled_params` (:665-670; minigpt/laplace.py:349-373) | The ensemble is sampler method `ensemble`. Member index $s$ = `seed` mod $M$ (Section 5.4). Everything after the logits stays S1's code. |
| Refit score sets on a new base | `posthoc_refit_config` checks the YAML's cell against S2's `SCORE_SET_CELLS` (eval_c_checkpoints.py:520-529; posthoc_refit.py:83), which holds only `c2_refit`, `c4_lap_refit` and `c4_tfb_fixed`. `_passed_fit_record` (:545-564) and `load_posthoc_refit` (:567-590) | The map lives in S2's module, so S5 cannot extend it. The scorer gets its own `S5_REFIT_SETS` map (score set → YAML, method, cell), and the cell lookup reads S2's map plus this one. `_passed_fit_record` and `load_posthoc_base` are reused as they are. |
| A block subset of about 5,000 blocks | Only `--block-ids` with a range string (:976-977, 302-321). `_batches` starts a new batch at every gap in $b$ (:701-714) | S5 adds `--block-ids-file` (Section 5.4). The first-block subset is priced at batch-1 rates (Section 5.6). |
| Document-paired resampling across score sets | `default_rng([seed, i_id, i_ood])` per cell (:1279), with $i$ the position in `eval_set.domains` (:1184-1188). `paired_doc_bootstrap` (minigpt/uncertainty.py:727-780) and `_doc_boot_resamples` (:641-670) share draws for the same seed. | Matches. With the same documents in a cell, S5's contrasts use the same draws as S1's and S2's rows. |
| A freeze mechanism for new rows | S1 Section 3: "A row that a later spec adds ... gets its own family in a separate YAML file, frozen the same way before its first test score exists." `check_prereg` hashes `cfg["analysis"]` only (:792-798; evalset.py:198-200) | S5 adds a second check, for its own score sets only, with the same hash rule on `analysis_s5` |
| A post-hoc refit on a deterministic LoRA | `base_kind: det_lora` is in `BASE_KINDS` (posthoc_refit.py:68). `load_posthoc_base` injects a deterministic LoRA and loads keys one to one; a missing or extra key raises KeyError (:222-301) | Matches. `_check_out_dir` protects the base folder (:891-900), so outputs go to `data/checkpoints/i2_posthoc/b2/`. |
| A record of what each fit used | `fit_record.json` (posthoc_refit.py:1228-1246, 1034-1065, 1150-1192): `method`, `cell`, `created_at`, `config_sha256`, `base` {`path`, `kind`, `sha256`, `n_bytes`, `vocab_size`, `step`}, `sampler_version`, `tfb.sigma_q_star`, `ell0`, `delta_nll_tfb`, `se_tfb`, `rho_tfb`, `match_status`, `lambda_star`, `delta_nll_star`, `tfb_record`, `same_fit_as_tfb`, `curvature.median_ratio`, `state_sha256`, `failures`, `exit_code`. No wall-time field. | S5-T2(d) and D6 use these names. The first draft asked S2 for a `wall_time_s` field. S2 is built, so `--plan` reads fit times from S5's own night-1 log instead. |
| A score-to-record hash check | `check_scores` (posthoc_refit.py:858-888) finds the record through `SCORE_SET_CELLS` | It cannot check S5 files. S5's `--check provenance --scores` does it (S5-T2d, S5-T3g). |
| Laplace end-of-run checks | `curvature_median_vs_reference`: the median $\hat F$ must lie within 0.5-2x of the state at `reference_state_path` (posthoc_refit.py:85, 1138-1146). A failure gives exit 2, and the scorer then refuses the fit (eval_c_checkpoints.py:554-556). | The b2 YAMLs keep C4-LAP's legacy state as the reference, and that state was fitted on the BLoB-mean base. $\hat F$ on LoRA $A$ scales with $\lVert B\rVert^2$, so a new deterministic LoRA can fall outside 0.5-2x for a real reason. Risk in Section 8; choice C4. |
| S0 inputs | `rescore.cells` [{`method`, `batch_size`, `ms_per_block`, `over_budget`, ...}] (timing_probe.py:78-80), `estimates.m4.batch_size`, `finetune.wall_min` | `--plan` converts ms per block to s per block |
| S1 block counts | `build_report.json`: `domains.<key>.n_blocks` and `domains.<key>.kept` (evalset.py:690-694) | $n_{\mathrm{all}}$ = the sum of `n_blocks` over the 5 domains; $n_{\mathrm{first}}$ = the sum of `kept` |

AGENTS.md rules that apply:

- Explicit configs. The seed script reads every key as `cfg[...]` and checks the full key list (Section 5.2) before it calls any helper. These helpers have `.get` defaults that the script must not rely on: `build_lora_config` (minigpt/config.py:203-212; rank default 8, while C3 uses 16), `build_train_config` (minigpt/config.py:229-235; patience default 10) and `load_pile_data` (minigpt/data.py:186-192). `load_pile_data` also uses `train.seed` as the Pile shuffle seed on a cache miss (minigpt/data.py:186). So the script asserts that every cache exists before it loads data. A miss would silently change the data with the seed.
- No notebooks. No new Bayesian library. torch only.
- LaTeX for every formula in this spec.
- Keep repo docs fresh: the new scripts and the test count go into AGENTS.md and README.md at once. The design decisions go into `agents/design-rationale.md`.
- 4-space indent, 100-character lines, type hints on public functions. `ruff check` and `pytest` before each commit. CI lints only `minigpt/ experiments/ tests/`, so the builder also runs `uv run ruff check` on the four touched scripts. Conventional Commits. Never mention AI assistants.

Open choices for the approver (the default applies if Alexey says nothing):

| # | Choice | Default | Alternative | Cost of the alternative |
|---|---|---|---|---|
| C1 | BLoB seed count | 3 new seeds from the seed script. C3 stays as a separate reference row. | 2 new seeds, with C3 as the third (options b1) | Saves 0.45 GPU h of training and 0.8-1.0 GPU h of re-score at batch 1. The BLoB row then mixes an unseeded pipeline run with seeded script runs, and C3 fails S5-T2 ("logs its seed"). |
| C2 | Review hours | 0.75-1.5 h (CA1) | The roadmap's 0.25-0.5 h: Alexey reviews the seed script and the results only. The ensemble scorer and the analysis are checked only by the reviewer agent and by `--rederive-s5`. | Saves 0.5-1.0 h. It goes against PG9: the current defects came from unreviewed agent code. |
| C3 | Ensemble size | M=3, as in roadmap S5-T3 and BLoB Table 1 ("ENS", 3 LoRAs) | M=5 (Ambitious item a3) | +2 deterministic fine-tunes (0.25-0.9 GPU h), +0.1 GPU h of re-score, +0.25 engineering h. Not in v1. |
| C4 | S2's curvature-median check on the b2 Laplace fits | Keep S2 as built. A b2 Laplace fit that fails only `curvature_median_vs_reference` is not scored. Its `curvature.median_ratio`, λ sweep and `match_status` are reported (D6). | Before night 1, send S2 a change request: the 0.5-2x bounds become a YAML key, and the b2 Laplace YAMLs turn the check off. It is a same-base reproduction test, and a new deterministic LoRA has no same-base reference. | 0.25-0.5 engineering h in S2, 0.1-0.25 h of Alexey review, and a CPU re-run of S2-T4(a)-(c). S5-T2(e) then also allows that key. |

## 3. Pre-registered endpoint

S1 fixes the primary score (the per-block realized-token Jensen gap $G$), the ID sets (HackerNews primary for the LoRA rows, StackExchange secondary) and the bootstrap. S5 does not restate those rules. S5 adds one question that no earlier spec asks (P3): does a posterior over the LoRA adapter detect OOD documents better than 3 independently trained deterministic LoRAs? S1's rule for new rows (S1 Section 3) requires a frozen family. The families below are frozen in `configs/i2_eval_s5.yaml` before any S5 score on a test block exists (S5-T5a).

Frozen settings. None of them changes after any S5 score is seen. A later change is reported as post hoc, next to the frozen result.

| Setting | Value | Source |
|---|---|---|
| Seeds | BLoB: 101, 102, 103. Deterministic LoRA: 201, 202, 203. | This spec. Distinct values avoid a shared $A$ init between the kinds. 1337 is avoided: it is the unapplied C3 template seed and S0's probe seed. |
| Base | `data/checkpoints/c0/ckpt_best.pt`, frozen | C3's phase-1 base is a copy of this file (experiments/c_pipeline.py:81-91) |
| Training schedule | C3's phase 2: 10,000 steps, batch 32, block 256, lr $3\times10^{-4}$, warmup 500, cosine to $10^{-5}$, grad clip 1.0, AdamW (0.9, 0.95), weight decay 0, eval at step 1 and every 500 steps with 20 iterations, dropout 0.1, LoRA rank 16, alpha 32, FFN | configs/c3_phase2.yaml; minigpt/train.py:299 |
| Kind settings | BLoB: `kl_weight` 1.0, `kl_annealing_steps` 1,000, `prior_std` 0.2, `init_g` 0.1, `num_train_tokens` = len(`data["train"]`). Deterministic: `kl_weight` 0.0, `kl_annealing_steps` 0. | configs/c3_phase2.yaml; experiments/pipeline_runner.py:360-375 |
| Checkpoint | Best on validation (ELBO for BLoB, CE for deterministic; minigpt/train.py:219-220). `patience_evals` 0. `checkpoint_interval` 0. Never resumed. | C3's patience of 10 never triggered (S0 Section 6), so 0 changes nothing for C3's schedule |
| Refits | S2's frozen settings (S2 Section 3) with `base_kind: det_lora`, HackerNews val, `n_data_seqs` 312,500. The Laplace target for seed $k$ is that seed's own $\Delta\mathrm{NLL}_{\mathrm{TFB},k}$. | S2 Sections 3 and 5.2 |
| Ensemble | M=3. The members are the 3 deterministic seeds, in seed order, with equal weights. | roadmap S5-T3 |
| Scoring | S1's scorer and seeding. $N=20$ for BLoB, TFB and Laplace. $N=M=3$ for the ensemble. Eval sets `test` and `test_hn`. Run tag `s5`. | S1 Sections 3 and 5.5 |
| Block subset | All blocks, unless `s5_plan.json` says `first_block_per_document` (rule in Section 5.6). The rule is applied before any S5 score exists. | This spec |
| Margin, level | $\delta=0.02$ AUROC; 0.05 after Holm | S1 Section 3 |

Primary quantity. For row $m$ with scored seeds $K_m$, and bootstrap resample $r$ with S1's document weights $w^{(r)}$:

$$\bar A_m^{(r)}=\frac{1}{\lvert K_m\rvert}\sum_{k\in K_m}\mathrm{AUROC}_w\!\left(G_{m,k};\,w^{(r)}\right),\qquad \Delta_m^{(r)}=\bar A_m^{(r)}-\mathrm{AUROC}_w\!\left(G_{\mathrm{ens}};\,w^{(r)}\right)$$

The point estimate $\Delta_m$ uses the original weights $w_b$. So does the per-seed value $\Delta_{m,k}=\mathrm{AUROC}_w(G_{m,k})-\mathrm{AUROC}_w(G_{\mathrm{ens}})$. The CI is the 2.5 and 97.5 percentiles. The p-value is S1's two-sided rule with a floor of $2/(B+1)$:

$$p=\min\!\left(1,\ \frac{2\left(1+\min\left(\#\{\Delta_m^{(r)}\le 0\},\ \#\{\Delta_m^{(r)}\ge 0\}\right)\right)}{B+1}\right)$$

The bootstrap resamples documents only. The seeds enter through the mean and through the sign rule below. Three seeds are too few for a seed-level test.

| Family | Rows | ID | Cells | Correction |
|---|---|---|---|---|
| F5 (S5, primary) | `blob`, `det_tfb`, `det_lap`, each against `lora_ens3` | HackerNews | 3 rows × 3 OOD domains (arXiv, FreeLaw, PubMed) = 9 | Holm over 9 |
| F6 (S5, secondary) | the same | StackExchange | 9 | Holm over 9 |

A row with fewer than 2 scored seeds (for example, 2 Laplace seeds that are not scoreable) is `not_tested` in its cells. Those cells enter Holm with $p=1$, so $m$ stays 9.

Decision rules, per cell:

| Result | Rule | What the paper says |
|---|---|---|
| Method beats ensemble | $\Delta_m\ge+0.02$, Holm-adjusted $p<0.05$, and $\Delta_{m,k}>0$ for every $k\in K_m$ | "<m> scores higher than a 3-member LoRA ensemble on <domain>, on each of <n> seeds: Δ = <x> [<lo>, <hi>]." |
| Ensemble beats method | $\Delta_m\le-0.02$, Holm-adjusted $p<0.05$, and $\Delta_{m,k}<0$ for every $k\in K_m$ | "A 3-member LoRA ensemble scores higher than <m> on <domain>, on each of <n> seeds." |
| Seed-dependent | $\lvert\Delta_m\rvert\ge0.02$ and Holm-adjusted $p<0.05$, but the signs of $\Delta_{m,k}$ differ | "The difference between <m> and the ensemble on <domain> depends on the training seed." |
| No difference | anything else | "<m> and a 3-member LoRA ensemble rank documents the same at this margin." |
| Not tested | $\lvert K_m\rvert<2$ | "<m> had <n> scoreable seeds, so this cell was not tested." |

Descriptive items. They have no margin and no correction. They are frozen in the same block, so they cannot be added after the scores are seen.

| ID | Issue | Quantity | Rule |
|---|---|---|---|
| D1 | P14 | Per seed: $\mathrm{AUROC}_w$ and $\mathrm{FPR95}_w$ of $G$ and $M$ with document CIs. Per row: mean, min and max over seeds. | S1 Section 5.6 functions |
| D2 | P14 | C3 inside the BLoB seed range | `within_seed_range` = ($\min_k A_{\mathrm{blob},k}\le A_{\mathrm{C3}}\le\max_k A_{\mathrm{blob},k}$), per cell |
| D3 | P14 | Seed range against eval noise | $R_m=\max_k A_{m,k}-\min_k A_{m,k}$ and $W_m=\operatorname{median}_k(\mathrm{hi}_{m,k}-\mathrm{lo}_{m,k})$. `seed_dominated` = ($R_m>W_m$). The expected range of 3 normal draws is about $1.69\sigma$ and a 95% CI is about $3.92\,\mathrm{SE}$. So the flag means a seed SD above about 2.3 times the eval SE. |
| D4 | P18 | Base effect | $\bar A_{\mathrm{det\_tfb}}-A_{\mathrm{c4\_tfb\_fixed}}$ and $\bar A_{\mathrm{det\_lap}}-A_{\mathrm{c4\_lap\_refit}}$, with the paired seed-mean bootstrap and no Holm |
| D5 | P3 | Matched cost | $G^{(3)}$ from the first 3 of the 20 saved samples, for each Bayesian row, against the ensemble (Section 5.5). The ensemble uses 3 passes and the Bayesian rows 20, so this is the equal-pass comparison. Only $G$ can be recomputed at $N=3$. $M$ needs the full vocabulary. Latency per millisecond (P22) stays in S3. |
| D6 | P18 | Fit budgets | Per deterministic seed: `tfb.sigma_q_star` ($\sigma^\star_{q,k}$), `rho_tfb` ($\rho_{\mathrm{TFB},k}$), `lambda_star` ($\lambda^\star_k$), `match_status`, `curvature.median_ratio` and `exit_code`, from the fit records |

Framing, written in advance. These are drafts for m11; the results pick the ending.

- N1 (no difference, the likely case). "A 3-member ensemble of independently trained deterministic LoRAs matches <m> on <domain>: Δ = <x> [<lo>, <hi>]. At 76M, a posterior over the LoRA adapter adds nothing measurable over training the adapter three times."
- N2 (ensemble wins). "A 3-member LoRA ensemble scores higher than <m> on <domain>, on each of 3 seeds. It needs 3 forward passes, against 20 for <m>."
- N3 (P14). "Across 3 training seeds of the adapter, <row>'s AUROC on <domain> spans <min>-<max>. That range is wider than its document CI. Differences smaller than this range between the one-seed rows (C0, C1, MC dropout) should not be read as method effects. Their full-training seed spread is not measured and may be larger."
- N4 (P18). "Fitting TFB on a deterministic LoRA instead of the BLoB posterior mean changes AUROC by <x> [<lo>, <hi>]."
- In every case: the seed spread covers adapter training only. All LoRAs share one C0 base (Section 7).

## 4. Acceptance tests

| ID | Given | When | Then | Runs in |
|---|---|---|---|---|
| S5-T1 Seed determinism | A CPU fixture. Base: `GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=1, n_embd=32, dropout=0.1, bias=True)` after `torch.manual_seed(0)`, saved in `tmp_path` in the `save_checkpoint` format. Data: `torch.randint(0, 100, (20000,), generator=torch.Generator().manual_seed(123))`, split 16,000 train and 4,000 val. LoRA rank 4, alpha 8, FFN. 30 steps, batch 8, lr $3\times10^{-3}$, warmup 5, eval every 10 steps with 2 iterations, `patience_evals` 0, `checkpoint_interval` 0. BLoB: `kl_weight` 1.0, `kl_annealing_steps` 5. Fixture YAMLs list seeds [1001, 1002]. | `train_one_seed` runs for each kind: seed 1001, seed 1001 again in a fresh output folder, then seed 1002 | (a) For both kinds, the two seed-1001 runs give `torch.equal` on every `lora_*` tensor of `ckpt_best.pt` (8 tensors for deterministic, 12 for BLoB) (yes/no). (b) Seed 1002 differs from seed 1001 in at least one `lora_A` (deterministic) or `lora_A_mu` (BLoB) entry (yes/no). Session-03 probe: max abs difference 0.354 and 0.334. (c) Each `seed_record.json` holds every field in Section 5.2 (yes/no per field). `config_sha256_noseed` is equal across seeds of one kind. `config_sha256` differs (yes/no). `seed_applied` equals the requested seed (yes/no). The model's vocabulary size equals the base checkpoint's `token_emb.weight` rows, 100 (yes/no). (d) A spy on the script's `train` symbol sees `resume_ckpt=None` in every call (yes/no). `--resume` on the CLI exits with argparse code 2 (yes/no). (e) A seed not in `seeds` raises ValueError. An output folder that already holds `ckpt_best.pt` raises FileExistsError. A YAML without `lora.rank`, or without `train.patience_evals`, raises KeyError before any helper runs (yes/no each). (f) The test function takes 60 s or less on CPU. | `tests/test_lora_seeds.py` (CPU, CI) |
| S5-T2 Six fresh runs, refits on deterministic bases | The 6 seed YAML runs on the RTX 4070, then the 6 b2 refits with S2's script, and the saved C0 and C3 checkpoints | `python scripts/analyze_lora_seeds.py --check provenance` after night 1. Unit parts: pytest with fixture checkpoints. | (a) 6 `seed_record.json` files exist, one per (kind, seed) in the YAML lists. Each has `steps_completed` = 10,000, `n_evals` = 21, `resumed` = false, `best_val_loss` below `first_val_loss`, and `trainable_params` = 1,966,080 (BLoB) or 1,310,720 (deterministic). `checkpoint_sha256` equals the file's hash at check time (yes/no each). (b) Within each kind, `config_sha256_noseed` is equal. All 6 records carry the same `pile_cache_sha256` for the HackerNews cache (yes/no). (c) In every checkpoint, every non-LoRA tensor equals C0's `ckpt_best.pt` tensor under the key with `.base_linear` removed (`torch.equal`) (yes/no). The 3 BLoB seeds differ pairwise and from C3 in `lora_A_mu` (yes/no). (d) 6 fit records exist under `data/checkpoints/i2_posthoc/b2/`. Each has `base.kind` = `det_lora`, `sampler_version` = `v2`, and `base.sha256` equal to the `checkpoint_sha256` of its deterministic seed record. No `base.sha256` equals the hash of C3's `ckpt_best.pt` or of a BLoB seed checkpoint. Each Laplace record has `same_fit_as_tfb` = true and `tfb_record.cell` = its seed's TFB cell (yes/no each). Each TFB record has `exit_code` 0 and empty `failures` (yes/no). Each Laplace record either has `exit_code` 0, empty `failures` and a `match_status` in S2's `SCORED_STATUSES`, or its seed is listed under `not_scored` in `s5_provenance.json` with the record's `match_status`, `exit_code` and `failures` (yes/no). No record has `match_status` = `bisection_failed` (yes/no). (e) Each of the 6 refit YAMLs differs from its S2 C4 YAML only in the keys listed in Section 5.3 (yes/no). (f) Unit, CPU: `load_posthoc_base` with `base_kind: det_lora` on a BLoB fixture checkpoint raises KeyError. A missing Pile cache makes the seed script raise FileNotFoundError before any stream opens (yes/no each). | `scripts/analyze_lora_seeds.py --check provenance` (real, writes `data/i2/s5_provenance.json`, exit non-zero on any failure); `tests/test_lora_seeds.py` for (f) |
| S5-T3 LoRA ensemble through S1's scorer | Fixture: S1-T5's 2-layer fixture model and eval set (tests/test_eval_scripts.py: 3 domains, 20 blocks from 10 documents each), fixture `prereg.json` and `prereg_s5.json`, plus 3 deterministic LoRA member checkpoints on the same base, with `lora_A` and `lora_B` drawn from `torch.Generator().manual_seed(7)`. Real: the 3 deterministic seeds after night 1. | `main()` of `scripts/eval_c_checkpoints.py` scores `lora_ens3` (N=3) on CPU, twice. On real data, the night-2 re-score. | (a) With 3 identical members, $G=0$ to $10^{-6}$ and $M=0$ to $10^{-5}$ on every block (yes/no). Session-03 probe: $\max\lvert g_t\rvert=0$ and $\max\lvert\mathrm{MI}_t\rvert=9.5\times10^{-7}$. (b) With distinct members, `logp_real[:, s, :]` equals the log-probs of an independent forward pass of member $s$ to $10^{-5}$, for $s=0,1,2$ (yes/no). (c) A spy on `score_block_batch` sees `method` = `ensemble` and `n_samples` = 3 in every call. A spy on `realized_token_scores` receives a `logp_real` of shape [B, 3, T]. $G$ recomputed with numpy from `logp_real` equals `blk_g` to $10^{-6}$ (yes/no each). (d) The two runs give identical tensors (`torch.equal`) (yes/no). (e) `meta` holds `n_samples` = 3, `sampler` = `ensemble`, `adapter_source` = `det_lora`, `analysis_s5_sha256` equal to the fixture `prereg_s5.json`, and a `checkpoint_sha256` dict with exactly 3 entries, the 3 member files (member 1 is also the host), whose values equal the files' hashes (yes/no each). (f) A member whose non-LoRA tensors differ from the host's raises ValueError. The ensemble sampler called with `seed=None`, or a score set with `n_samples` ≠ 3, raises ValueError (yes/no each). (g) Real: the 20 files `{score_set}__{test,test_hn}__s5.pt` of the scoreable sets exist. The member SHA-256 values of the two `lora_ens3` files equal the 3 deterministic seed records (yes/no). `scripts/check_eval_rebuild.py --check align` exits 0 with them present, so `doc_id`, `domain` and `offset` match S1's `main` files at the same `block_index` (yes/no). (h) `--block-ids-file` with the list [0, 3, 4] writes tensors equal (`torch.equal`) to those of `--block-ids 0,3-4`. `parse_block_ids(meta["block_ids"])` returns [0, 3, 4], and `meta["block_ids_file"]` holds the file's path and sha256. A list that is not ascending exits with code 2 (yes/no each). | `tests/test_lora_seeds.py` (CPU, CI) for (a)-(f) and (h); `scripts/analyze_lora_seeds.py --check provenance --scores` and `scripts/check_eval_rebuild.py --check align` for (g) |
| S5-T4 Seed-spread table | Fixture: score files for rows `blob` and `det_tfb` with 3 seeds each and for `lora_ens3`, built from S1's fixture F-i scores (tests/test_doc_bootstrap.py) plus per-seed shifts of 0.0, 0.3 and 0.6 on the OOD class. One `det_lap` seed file is absent, and its fit record says `not_matched`. Real: `data/scores_i2/` after night 2. | `scripts/analyze_lora_seeds.py --analyze` writes `table_s5.json`. On real data, `scripts/check_eval_rebuild.py --rederive-s5` rebuilds it. | (a) Every per-seed $\mathrm{AUROC}_w$ equals `roc_auc_score(y, s, sample_weight=w)` to $10^{-12}$, with $w$ recomputed as $1/k_d$ inside the cell. The row mean, min and max equal numpy on the per-seed values to $10^{-12}$ (yes/no). (b) The table holds, for every (row, ID domain, OOD domain), $\mathrm{AUROC}_w$ and $\mathrm{FPR95}_w$ of $G$ and $M$ per seed, each with its document CI, plus the seed mean, min and max. It holds this for both ID sets (HackerNews, StackExchange) and all 3 OOD domains (yes/no per field). (c) The `det_lap` row shows `n_seeds` = 2 and names the missing seed with its `match_status` (yes/no). (d) The table has the rows `blob`, `det_tfb`, `det_lap` and `lora_ens3`, and the reference rows `c3`, `c4_tfb_fixed` and `c4_lap_refit`. Each BLoB cell has the D2 flag `within_seed_range`, and each row has the D3 flag `seed_dominated`. Both flags equal their Section 3 rules recomputed with numpy from the table's own per-seed values and CIs (yes/no each). (e) Real: `--rederive-s5` equals `table_s5.json` and `families_s5.json` to $10^{-6}$ in every cell (yes/no per cell). The checker imports neither `minigpt.uncertainty` nor `analyze_lora_seeds` (yes/no, by an import check). | `tests/test_lora_seeds.py` (CPU, CI) for (a)-(d); `scripts/check_eval_rebuild.py --rederive-s5` for (e) |
| S5-T5 Frozen families, seed-mean contrast, plan rule | Fixture YAML and `prereg_s5.json` in `tmp_path`. Fixture F-ii of S1 (tests/test_doc_bootstrap.py: 200 + 200 documents, $k=3$, $\rho=0.9$). Synthetic S0 and S1 inputs and a synthetic night-1 log for the plan. Real: `configs/i2_eval_s5.yaml` and the S5 score files. | `--freeze`, then the scorer on a fixture S5 score set. `seed_mean_paired_doc_bootstrap` with $B=2{,}000$ and seed 0. The decision function on synthetic inputs. `--plan` on synthetic inputs. On real data, `--check prereg-s5`. | (a) `prereg_s5.json` holds the sha256 of `json.dumps(cfg["analysis_s5"], sort_keys=True, separators=(",", ":"))`, and `--freeze` refuses to overwrite it. With a differing hash, the scorer exits non-zero and writes no file for a `test` or `test_hn` S5 score set, and still scores S1's score sets (yes/no each). Real: `frozen_at` is earlier than the `created_at` of every S5 test score file, and every S5 file's `meta.analysis_s5_sha256` equals `prereg_s5.json` (yes/no each). (b) If all 3 seeds equal the ensemble scores, the Δ CI is [0, 0] and $p=1$. If each seed equals the ensemble score plus $0.8y$, every $\Delta_r>0$ and $p=2/(B+1)=0.0009995$. The seed-mean point Δ equals the mean of the per-seed $\Delta_{m,k}$ to $10^{-12}$. With one seed, Δ, CI and $p$ equal S1's `paired_doc_bootstrap` on the same inputs and seed to $10^{-12}$ (yes/no each). (c) The decision function returns, for $(\Delta_m, p_{\mathrm{Holm}}, [\Delta_{m,k}])$: (0.05, 0.01, [0.04, 0.05, 0.06]) → `method_beats_ensemble`; (−0.05, 0.01, [−0.04, −0.05, −0.06]) → `ensemble_beats_method`; (0.05, 0.01, [0.10, 0.07, −0.02]) → `seed_dependent`; (0.01, 0.01, [0.01, 0.01, 0.01]) → `no_difference`; (0.05, 0.10, [0.04, 0.05, 0.06]) → `no_difference`; one scored seed → `not_tested` with $p=1$ in Holm (yes/no each). (d) Holm over the 9 cells of F5 and of F6 equals S1's `holm` to $10^{-12}$ (yes/no). (e) `--plan` with the CA10 fixture rates as both the $b^*$ and the batch-1 rates (C3 0.3156, C4-TFB and C4-LAP 0.4686 s per block), $n_{\mathrm{all}}$ = 10,000, $n_{\mathrm{first}}$ = 5,000, and a night-1 log whose 6 fine-tunes sum to 2.345 GPU h and whose fits take 0.9 (TFB) and 1.3 (Laplace) GPU h each gives $G^{\mathrm{all}}_{\mathrm{rs}}$ = 10.572 ± 0.001, `block_subset` = `first_block_per_document`, $G^{\mathrm{first}}_{\mathrm{rs}}$ = 5.286 ± 0.001, `escalate` = false, `rescore_nights` = 1 and `nights_needed` = 2. With the $b^*$ rates ÷ 3.5 (batch-1 rates unchanged): $G^{\mathrm{all}}_{\mathrm{rs}}$ = 3.020 ± 0.001 and `block_subset` = `all`. With both rate sets × 1.6: `escalate` = true (yes/no each). | `tests/test_lora_seeds.py` (CPU, CI) for (a)-(e); `scripts/analyze_lora_seeds.py --check prereg-s5` for the real part of (a) |

Pre-merge command. The builder runs it on the machine that holds `data/`, after night 2:

```bash
uv run pytest tests/test_lora_seeds.py -rs
uv run python scripts/analyze_lora_seeds.py --check provenance --scores
uv run python scripts/analyze_lora_seeds.py --check prereg-s5
uv run python scripts/check_eval_rebuild.py --rederive-s5
uv run python scripts/check_eval_rebuild.py --check align
uv run python scripts/check_eval_rebuild.py --check prereg
```

It passes when every command exits 0. The last two are S1's checks; they must still pass with the S5 files present. The pytest module reads no `data/`, so it runs in CI unchanged.

## 5. Implementation notes for the builder

### 5.1 Order of work

| Step | Work | Test | Machine | When |
|---|---|---|---|---|
| 1 | Seed script and the two YAMLs | S5-T1 | CPU | build session |
| 2 | Scorer entries, the S5 refit map, the ensemble sampler, `--block-ids-file`, the S5 freeze check | S5-T3 (a)-(f), (h); S5-T5 (a) | CPU | build session |
| 3 | `seed_mean_paired_doc_bootstrap`, the decision function, `analyze_lora_seeds.py` | S5-T4 (a)-(d), S5-T5 (b)-(e) | CPU | build session |
| 3b | A separate agent writes `--rederive-s5` from Sections 3, 5.5 and 5.7 only, not from the analysis code | S5-T4 (e) on fixtures | CPU | build session |
| 4 | After approval: write `configs/i2_eval_s5.yaml`, run `--freeze`, record the hash in the session log and STATE.md | S5-T5 (a) | CPU | before step 8 |
| 5 | 6 fine-tunes, one process per run: BLoB 101-103, then deterministic 201-203. The night-1 launcher logs each command to `data/i2/s5_night1_log.json`. | S5-T2 (a)-(c) | GPU 1.7-2.7 h | night 1 (Tue Oct 6), or Sat Oct 3 / Sun Oct 4 if steps 1 and review are done |
| 6 | 6 refits with `scripts/refit_posthoc.py`, TFB before Laplace for each seed (exit-code rules in Section 5.3). Only after the C4 refits pass the fit part of S2-T4(d). | S5-T2 (d)-(e) | GPU 1.9-7.65 h | night 1, after step 5 |
| 7 | `--plan` writes `data/i2/s5_plan.json`, the block-ID files if needed, and the night-2 list | S5-T5 (e) | CPU | morning after night 1 |
| 8 | Re-score of the scoreable S5 sets on `test` and `test_hn`, run tag `s5` | S5-T3 (g) | GPU 3.35-5.29 h under the rule; at most 8 unless it escalates | night 2 (Wed Oct 7) |
| 9 | `--analyze`, `--check provenance --scores`, `--check prereg-s5`, `--rederive-s5`, S1's `--check align` and `--check prereg` | S5-T3 (g), S5-T4 (e), S5-T5 (a) | CPU | Thu Oct 8 |

Alexey reviews the code before step 5 and the tables in the W2 review window (Thu Oct 8-Sat Oct 10). G1 stays on Sun Oct 11.

### 5.2 Seed script and YAMLs

CLI: `python scripts/train_lora_seed.py --config configs/i2_lora_seed_blob.yaml --seed 101`. One process per run. A crash then loses only the run in flight.

`train_one_seed(cfg: dict, seed: int, *, data: dict | None = None) -> dict` does this, in order:

1. Check every key in the list below with `cfg[...]`, then call `validate_config(cfg)` (minigpt/config.py:241). Assert `seed in cfg["seeds"]` (ValueError otherwise).
2. Resolve `cfg["train"]["seed"] = seed` and `cfg["train"]["checkpoint_dir"] = cfg["out_dir_template"].format(seed=seed)`. Refuse a folder that already holds `ckpt_best.pt` (FileExistsError).
3. If `data` is None: assert the 4 caches exist (`hackernews_100000000.pt`, `arxiv_10000000.pt`, `freelaw_10000000.pt`, `pubmed_abstracts_10000000.pt` in `data/pile/`; FileNotFoundError), hash the HackerNews cache file, and call `load_dataset(cfg, tokenizer)`. Tests pass `data` in.
4. `torch.manual_seed(seed)`; record `seed_applied = torch.initial_seed()`. experiments/experiment_setup.py:44 seeds before its data load. A cached load consumes no RNG, so the RNG state at model build is the same. S0-T2 seeds at the same point.
5. Read the vocabulary size from the base checkpoint's `model_state_dict["token_emb.weight"]` rows, as `load_posthoc_base` does (minigpt/posthoc_refit.py:249). When `data` is None, assert that it equals the tokenizer's `n_vocab`. Build `MiniGPT(build_gpt_config(cfg, vocab_size))`. `load_checkpoint(cfg["base_checkpoint"], model)`, which is a strict load (minigpt/train.py:124). Hash the base file.
6. `inject_lora(model, LoRAConfig(rank, alpha, target, prior_std, init_g), bayesian=(cfg["kind"] == "blob"))`, with every field from `cfg["lora"]`.
7. `train(model, data["train"], data["val"], build_train_config(cfg), mlflow_run=None, config_dict=cfg, resume_ckpt=None, kl_weight=cfg["train"]["kl_weight"], num_train_tokens=len(data["train"]) if blob else 0)`. The script defines no resume path. Resume drops the AdamW state (minigpt/train.py:185; code-assessment.md Section 2.5).
8. Hash `ckpt_best.pt` and write `seed_record.json` through a temp file and `os.replace`.

Randomness in a LoRA fine-tune, all driven by `torch.manual_seed`: the batch indices (`torch.randint` on the CPU generator, minigpt/train.py:47), the Kaiming init of $A$ (minigpt/lora.py:59, 142), the BLoB scale init and noise (:63, :75), and dropout 0.1. $B$ starts at 0 (:55, :138). So deterministic members differ only by the $A$ init, the batch order and the dropout masks. GPU training is not bitwise reproducible (atomic adds in the backward pass). So S5-T1 checks bitwise equality on CPU only.

`seed_record.json` fields: `kind`, `seed`, `seed_applied`, `config_sha256` (sha256 of `json.dumps(cfg, sort_keys=True, separators=(",", ":"))` after step 2), `config_sha256_noseed` (the same without `train.seed` and `train.checkpoint_dir`), `base_checkpoint`, `base_sha256`, `pile_cache_sha256` (null in tests), `checkpoint`, `checkpoint_sha256`, `trainable_params`, `steps_completed`, `n_evals` (= len(`eval_history`)), `first_val_loss` (= `eval_history[0]["val_loss"]`, the step-1 CE), `best_val_loss` (the checkpoint criterion; for BLoB it is the ELBO, which is at least the CE, so "best below first" is a conservative check), `best_val_step`, `train_time_sec`, `wall_min` (= `train_time_sec`/60, C3's clock), `peak_reserved_mib` (null on CPU), `resumed` (always false), `git_sha`, `git_dirty`, `code_sha256`, `created_at` (UTC).

Required keys (both YAMLs; the script checks all of them first):

| Section | Keys |
|---|---|
| Top level | `kind` (`blob` or `det_lora`), `seeds`, `base_checkpoint`, `out_dir_template` |
| `experiment` | `name`, `run_name` |
| `data` | `dataset`, `val_fraction`, `test_fraction`, `pile_id_domains`, `pile_ood_domains`, `pile_id_tokens`, `pile_ood_tokens` |
| `model` | `block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`, and `bayes_head`, `bayes_ffn`, `bayes_attn_v` with `enabled: false` (validate_config reads them, minigpt/config.py:254-256) |
| `train` | `steps`, `batch_size`, `block_size`, `lr`, `weight_decay`, `warmup_steps`, `min_lr`, `grad_clip`, `eval_interval`, `eval_iters`, `checkpoint_interval`, `gradient_accumulation_steps`, `patience_evals`, `patience_min_delta`, `kl_weight`, `kl_annealing_steps`, `adam_beta1`, `adam_beta2`, `device` |
| `lora` | `rank`, `alpha`, `target`, `prior_std`, `init_g` |

`train.seed` and `train.checkpoint_dir` are not in the YAML. The script sets them in step 2 from `seeds` and `out_dir_template`, and the resolved config is hashed and saved in the checkpoint.

YAML sketch (BLoB; the deterministic file differs only where marked):

```yaml
kind: blob                                   # det_lora
seeds: [101, 102, 103]                       # [201, 202, 203]
base_checkpoint: data/checkpoints/c0/ckpt_best.pt
out_dir_template: data/checkpoints/i2_seeds/blob_s{seed}   # .../det_s{seed}
experiment: {name: i2_lora_seed_blob, run_name: i2_lora_seed_blob}   # i2_lora_seed_det
data:
  dataset: pile
  val_fraction: 0.1
  test_fraction: 0.1
  pile_id_domains: [hackernews]
  pile_ood_domains: [arxiv, freelaw, pubmed_abstracts]
  pile_id_tokens: 100000000
  pile_ood_tokens: 10000000
model:
  block_size: 256
  n_layer: 16
  n_head: 8
  n_embd: 512
  dropout: 0.1
  bias: true
  bayes_head: {enabled: false, prior_std: 1.0, init_rho: -1.0}
  bayes_ffn: {enabled: false, prior_std: 1.0, init_rho: -1.0}
  bayes_attn_v: {enabled: false, prior_std: 1.0, init_rho: -1.0}
train:
  steps: 10000
  batch_size: 32
  block_size: 256
  lr: 0.0003
  weight_decay: 0.0
  warmup_steps: 500
  min_lr: 1.0e-05
  grad_clip: 1.0
  eval_interval: 500
  eval_iters: 20
  checkpoint_interval: 0                     # C3 used 5000; no periodic files here
  gradient_accumulation_steps: 1
  patience_evals: 0                          # C3's default 10 never triggered
  patience_min_delta: 0.001
  kl_weight: 1.0                             # 0.0
  kl_annealing_steps: 1000                   # 0
  adam_beta1: 0.9
  adam_beta2: 0.95
  device: auto
lora: {rank: 16, alpha: 32.0, target: ffn, prior_std: 0.2, init_g: 0.1}
```

S0's adapter is not reused as a member. S0 allows reuse only if the configs hash to the same `config_sha256`. S0's run has a different `experiment.name` and `train.checkpoint_dir`, and a different code path, so the hash cannot match. The saving (7-27 GPU min) is smaller than the review cost of a provenance exception. `--check provenance` prints S0's `best_val_loss` next to the three deterministic seeds (report only).

### 5.3 Refits on the deterministic seeds (b2)

For each deterministic seed $k$: run `scripts/refit_posthoc.py --config configs/i2_posthoc_b2_tfb_s{seed}.yaml`, then the same with `..._lap_s{seed}.yaml`. The YAMLs are copies of S2's C4 files. They differ only in these keys (S5-T2e):

| Key | TFB file | Laplace file |
|---|---|---|
| `cell` | `b2_tfb_s{seed}` | `b2_lap_s{seed}` |
| `out_dir` | `data/checkpoints/i2_posthoc/b2/tfb_s{seed}` | `data/checkpoints/i2_posthoc/b2/lap_s{seed}` |
| `base.base_checkpoint` | `data/checkpoints/i2_seeds/det_s{seed}/ckpt_best.pt` | the same |
| `base.base_kind` | `det_lora` | `det_lora` |
| `laplace.tfb_record_path` | not present | `data/checkpoints/i2_posthoc/b2/tfb_s{seed}/fit_record.json` |

Everything else stays as in S2: the fit set (640 HackerNews val blocks, at most 3 per document, `fit_seed` 0), $\epsilon_{\mathrm{rel}}=0.003$, the search range $[10^{-4}, 1]$, $S=10$ and $S=20$, the λ grid $10^{0},\dots,10^{7}$ with one extension to $10^{8}$, `match_tol` 0.10, the SE rule, `n_data_seqs` 312,500 and $T=256$. `laplace.reference_state_path` also stays `data/checkpoints/c4_lap/laplace_state.pt` (choice C4). So seed $k$'s Laplace target is its own TFB budget:

$$\Delta\mathrm{NLL}^\star_{\mathrm{LAP},k}=\rho_{\mathrm{TFB},k}\,\ell^{(k)}_0=\Delta\mathrm{NLL}_{\mathrm{TFB},k}\!\left(\sigma^\star_{q,k}\right),\qquad \rho_{\mathrm{TFB},k}=\frac{\Delta\mathrm{NLL}_{\mathrm{TFB},k}}{\ell^{(k)}_0}$$

The two forms are equal because the TFB and Laplace fits of one seed share the base and the fit set (`same_fit_as_tfb`). The same fit set for every seed means that the seed changes only the training run.

The LoRA update is $\Delta W=\frac{\alpha}{r}BA$ with $B_0=0$ (minigpt/lora.py:138, 146). After 10,000 steps $B$ is non-zero, so TFB's SVD of $B$ is defined. S2 clamps singular values at $10^{-6}$ (minigpt/tfb.py:311-313).

Night-list rules, from S2's exit codes (scripts/refit_posthoc.py: 0 pass, 2 a failed check, 3 `bisection_failed`, 4 no TFB budget):

- A TFB exit other than 0 skips that seed's Laplace refit. S5-T2(d) then fails, and the builder reports it.
- A Laplace fit with `match_status` `not_matched` (exit 0), or with exit 2 (for example the curvature-median check), is not scored. `--plan` leaves the seed's `det_lap` set out of the night-2 list and records the reason in `s5_plan.json`.
- A Laplace exit 3 (`bisection_failed`) is also not scored, and S5-T2(d) fails, as it does for S2.
- The list always continues with the next seed.

### 5.4 Scorer entries and the ensemble

The 10 score sets (the `scoring` section of `configs/i2_eval.yaml`; seed index $k$ maps to the seed in the YAML lists):

| Score set | Load rule | `checkpoint_paths` (hashed into meta) | `n_samples` | `sampler` | `adapter_source` | `display` |
|---|---|---|---|---|---|---|
| `blob_1`-`blob_3` | MiniGPT from `configs/i2_lora_seed_blob.yaml` model and lora keys, each read as `cfg[...]`, and the vocabulary size from the checkpoint; `inject_lora(bayesian=True)`; strict `load_checkpoint` of `data/checkpoints/i2_seeds/blob_s{seed}/ckpt_best.pt`. Method `variational`. | the checkpoint | 20 | `variational` | `blob` | "BLoB LoRA (seed <seed>)" |
| `det_tfb_1`-`det_tfb_3` | Through `S5_REFIT_SETS`: `load_posthoc_base` (`det_lora`, `det_s{seed}`), `_passed_fit_record`, then the TFB v2 state from `.../b2/tfb_s{seed}/`. Method `tfb_v2`. | base checkpoint and state file | 20 | `v2` | `det_lora` | "Det. LoRA + TFB (seed <seed>)" |
| `det_lap_1`-`det_lap_3` | The same base and checks, then the Laplace v2 state from `.../b2/lap_s{seed}/`. Method `laplace_v2`. | base checkpoint and state file | 20 | `v2` | `det_lora` | "Det. LoRA + diag. Laplace LoRA-A (scaled) (seed <seed>)" |
| `lora_ens3` | Host: `load_posthoc_base` with `base` = {`base_checkpoint`: `det_s201/ckpt_best.pt`, `base_kind`: `det_lora`} and `model` and `lora` from `configs/i2_lora_seed_det.yaml`. Members: the `lora_A` and `lora_B` tensors of the 3 deterministic checkpoints, strict key match, on the device. Method `ensemble`. | the 3 member checkpoints | 3 | `ensemble` | `det_lora` | "LoRA ensemble (M=3)" |

`S5_REFIT_SETS` maps each of the 6 refit score sets to its YAML, method and cell (for example `det_tfb_1` → (`configs/i2_posthoc_b2_tfb_s201.yaml`, `tfb`, `b2_tfb_s201`)). `posthoc_refit_config` then reads the expected cell from S2's `SCORE_SET_CELLS` plus this map. S2's module is not edited.

All 10 go into `score_sets.test` and `score_sets.test_hn`. They are not scored on `legacy_d1` or `arxiv_stripped`. Every S5 run uses `--run-tag s5` and names its sets with `--score-set`, because `run_i2` refuses a run in which any listed set lacks its input files (scripts/eval_c_checkpoints.py:1041-1047). S1's runs keep passing `--score-set` with the S1 sets, as the YAML comment already asks.

The ensemble in S1's per-sample loop. Sample $s$ is member $s$:

$$\theta_s=(W_0,\,A_s,\,B_s),\qquad \ell_{s,t}=\log p_{\theta_s}(y_t\mid x_{\le t}),\qquad s=1,\dots,M,\quad M=3$$

$$\log\bar p(y_t)=\operatorname{logsumexp}_{s=1}^{M}\ell_{s,t}-\log M,\qquad g_t=\log\bar p(y_t)-\frac{1}{M}\sum_{s=1}^{M}\ell_{s,t},\qquad G(b)=\frac{1}{T}\sum_{t=1}^{T}g_t$$

This is S1's formula with $N=M$ (`realized_token_scores`, scripts/eval_c_checkpoints.py:610-620). Here $\bar p$ is the ensemble's own predictive distribution, not a Monte Carlo estimate of one.

- The ensemble is a parameter sampler: `register_param_sampler("ensemble", sample_member)`, where the state holds the 3 member dicts {`...lora_A`: tensor, `...lora_B`: tensor} and `sample_member(state, seed)` returns `state.members[seed % M]`. `score_block_batch` passes `seed = seed_b * N + s` (scripts/eval_c_checkpoints.py:667), and $N=M$, so `seed % M` $= s$. `seed=None` raises ValueError.
- `score_block_batch` applies the returned tensors with `apply_sampled_params` (minigpt/laplace.py:349-373), which copies any named parameter and restores it on exit. The session-03 probe gave a maximum logit difference of 0 against each member's own forward pass, and an exact restore.
- Everything after the logits (log-softmax, $\ell_{s,t}$, MI, TU, AU, $G$) is S1's shared code. Only the sample source differs. S5-T3(c) checks this with spies.
- No RNG is used. `score_block_batch` still calls `torch.manual_seed(seed_b)`, which has no effect here. The model is in `eval()` mode, so dropout is off.
- Before scoring, assert that every non-LoRA tensor of each member equals the host's (ValueError otherwise). Assert `n_samples == len(members)`.

`--block-ids-file <json>` takes an ascending list of full-set block indices, because 5,000 first blocks would overflow the Windows command line (32,767 characters). The built scorer has only `--block-ids` (scripts/eval_c_checkpoints.py:976-977). The new flag excludes `--block-ids`. It validates the list as `parse_block_ids` does (:302-321). It writes the list as the canonical range string into `meta.block_ids`, so `parse_block_ids` reads it back, and it adds `meta.block_ids_file` = {path, sha256}. `BlockSet.select` keeps each block's full-set index, so block $b$ keeps its seed $\mathrm{seed\_start}+b$.

S5 freeze check. The scorer reads `configs/i2_eval_s5.yaml` (a constant path, like `POSTHOC_REFIT_CONFIGS`). For a score set named in `analysis_s5.rows`, it scores `test` or `test_hn` only if `data/scores_i2/prereg_s5.json` exists and its hash equals the block's hash. Such files also get `meta.analysis_s5_sha256`. The check for S1's own sets does not change.

### 5.5 Analysis: seed-mean contrast, table, re-derivation

- Cells follow S1 Section 5.6. For the S5 rows, the ID blocks come from `{s}__test_hn__s5.pt` (HackerNews) or from the StackExchange rows of `{s}__test__s5.pt`, and the OOD blocks from `{s}__test__s5.pt`. The reference rows (C3, `c4_tfb_fixed`, `c4_lap_refit`) come from their `__main.pt` files. The RNG is S1's `default_rng([seed, i_id, i_ood])`, with $i$ the position in `eval_set.domains`, and draws in S1's order.
- Weights are recomputed inside each cell as $w_b=1/k_d$, with $k_d$ the number of blocks of document $d$ in the cell, as S1 does. The stored `weight` field is never read (Section 2, interface table).
- Under `first_block_per_document`, each document keeps only its block with the smallest `offset`. Its weight is then 1. The reference rows are cut to the same `block_index` set from their S1 and S2 files. The analysis asserts that every row of a cell has the same (`doc_id`, `block_index`) set.
- `seed_mean_paired_doc_bootstrap(scores_a: list[np.ndarray], score_b: np.ndarray, labels, doc_ids, weights, n_resamples: int, seed: int | Sequence[int], level: float) -> dict` returns $\Delta_m$, the per-seed $\Delta_{m,k}$, the CI, the p-value and the resampled $\bar A^{(r)}$. It takes the seed, not a generator, as S1's `paired_doc_bootstrap` does, and draws with S1's `_doc_boot_resamples` on the seeds and the ensemble together. For each resample, it computes every seed's $\mathrm{AUROC}_w$ with the same $w^{(r)}$ and then averages. It never averages scores across seeds, because the scales of different seeds need not match. With one seed it equals `paired_doc_bootstrap` (S5-T5b).
- The decision function takes $(\Delta_m, p_{\mathrm{Holm}}, [\Delta_{m,k}])$ and applies the table in Section 3. Holm is S1's `holm`, within F5 and within F6.
- Matched cost (D5). For the Bayesian rows, recompute $G$ from the first 3 saved samples:

$$G^{(3)}(b)=\frac{1}{T}\sum_{t=1}^{T}\left[\operatorname{logsumexp}_{s=1}^{3}\ell_{s,t}-\log 3-\frac{1}{3}\sum_{s=1}^{3}\ell_{s,t}\right]$$

  The first 3 of the 20 draws are valid draws. BLoB reseeds with `torch.manual_seed(seed_b)` before the passes, and TFB and Laplace use seeds $\mathrm{seed}_b N+s$ (scripts/eval_c_checkpoints.py:661-668).
- `table_s5.json` holds, per (row, ID domain, OOD domain): for each seed, $\mathrm{AUROC}_w$ and $\mathrm{FPR95}_w$ of $G$ and of $M$ with CIs, $G^{(3)}$'s $\mathrm{AUROC}_w$ for the Bayesian rows, `match_status` for the post-hoc rows, and `n_seeds`; per row, the mean, min and max; the flags D2 and D3; and the D6 fit budgets per deterministic seed. `families_s5.json` holds, per F5 and F6 cell: $\Delta_m$, the per-seed $\Delta_{m,k}$, the CI, $p$, the Holm-adjusted $p$ and the decision. It also holds D4.
- `--rederive-s5` rebuilds both files with numpy, `torch.load` and sklearn only (S5-T4e). It follows S1's `--rederive` rules.

### 5.6 GPU plan (`--plan`, CPU)

`--plan` runs the morning after night 1. By then the fine-tunes and the refits have run. It reads only costs, never scores.

Inputs:

- S0's `data/i2/timing_probe.json`: per method (`c3`, `c4_tfb`, `c4_lap`), `ms_per_block` at batch 1 ($r^{(1)}_m$) and at $b^*_m$ from `estimates.m4.batch_size` ($r^*_m$), divided by 1,000.
- S1's `data/eval_i2/build_report.json`: $n_{\mathrm{all}}=\sum_d$ `domains.<d>.n_blocks` and $n_{\mathrm{first}}=\sum_d$ `domains.<d>.kept`, over the 5 domains of `test` and `test_hn` (about 5,000 = 1,000 documents × 5).
- S5's `data/i2/s5_night1_log.json`: the 6 fine-tune durations and the 6 refit durations. Fallbacks, in order: the seed records' `wall_min` for the fine-tunes; 0.97 GPU h (TFB) and 1.58 GPU h (Laplace) per fit, the batch-1 upper figures of Section 1. S2's fit records have no wall-time field.

$$G_{\mathrm{ft}}=\sum_{\text{6 fine-tunes}}\frac{\text{duration}}{3600\ \mathrm{s}},\qquad G_{\mathrm{fit}}=\sum_{k}\left(T_{\mathrm{TFB},k}+T_{\mathrm{LAP},k}\right)$$

Before night 1, the forecast is $G_{\mathrm{ft}}=(3\,t_B+3\,t_D)/60$ with $t_B=26.9$ min and $t_D$ = S0's `wall_min`. This equals S0's `estimates.s5_finetune_gpu_h.n_blob_3`.

$$G_{\mathrm{rs}}(n,\,r)=\frac{n}{3600}\left[3\left(r_{\mathrm{C3}}+r_{\mathrm{TFB}}+r_{\mathrm{LAP}}\right)+\frac{3}{20}\,r_{\mathrm{C3}}\right],\qquad G^{\mathrm{all}}_{\mathrm{rs}}=G_{\mathrm{rs}}\!\left(n_{\mathrm{all}},\,r^*\right),\qquad G^{\mathrm{first}}_{\mathrm{rs}}=G_{\mathrm{rs}}\!\left(n_{\mathrm{first}},\,r^{(1)}\right)$$

- $r_m$ is in seconds per block at $N=20$. The C4-LAP rate was timed with the legacy CPU draw. S2's v2 draw is on the GPU, so this rate is an upper bound.
- The ensemble term is 3 deterministic-LoRA passes, priced as 3/20 of C3's rate.
- The first-block subset uses the batch-1 rates. `_batches` starts a new batch at every gap in the block index (scripts/eval_c_checkpoints.py:701-714). First blocks are consecutive only after one-block documents, so the subset gets little of the batching gain.
- The rates are re-scaled for the seeds not scored (Section 5.3): each missing set removes its $r_m$ term once.

Rule (frozen in `analysis_s5.plan`):

| Condition | Result |
|---|---|
| $G^{\mathrm{all}}_{\mathrm{rs}}\le 8$ GPU h | `block_subset` = `all` |
| $G^{\mathrm{all}}_{\mathrm{rs}}>8$ | `block_subset` = `first_block_per_document`. `--plan` writes `data/i2/s5_block_ids_test.json` and `..._test_hn.json` from the block files' `doc_id` and `offset`. |
| $G^{\mathrm{first}}_{\mathrm{rs}}>8$ | `escalate` = true. Alexey decides. If he is silent, the re-score is split over two nights, and $N$ stays 20. |
| always | $\texttt{rescore\_nights}=\left\lceil G_{\mathrm{rs}}(\text{chosen})/8\right\rceil$. For the record, $\texttt{nights\_needed}=\left\lceil\left(G_{\mathrm{ft}}+G_{\mathrm{fit}}+G_{\mathrm{rs}}(\text{chosen})\right)/8\right\rceil$ for the whole run, as one sequential queue. |

8 GPU h is the low end of a night (CA3).

Fixture values for S5-T5(e): the per-block sum is $3(0.3156+0.4686+0.4686)+0.15\times0.3156=3.80574$ s. So $G_{\mathrm{rs}}(10{,}000)=10.5715$ and $G_{\mathrm{rs}}(5{,}000)=5.2857$. $G_{\mathrm{fit}}=3(0.9+1.3)=6.6$. `rescore_nights` = 1. The queue total is $2.345+6.6+5.2857=14.23$, so `nights_needed` = 2. With the $b^*$ rates ÷ 3.5, $G^{\mathrm{all}}_{\mathrm{rs}}=3.0204$. With both rate sets × 1.6, $G^{\mathrm{all}}_{\mathrm{rs}}=16.91$ and $G^{\mathrm{first}}_{\mathrm{rs}}=8.457$.

### 5.7 `configs/i2_eval_s5.yaml` (frozen at approval)

```yaml
analysis_s5:                     # sha256 -> data/scores_i2/prereg_s5.json (--freeze)
  primary_score: blk_g
  contrast_score: blk_mi
  score_run_tag: s5
  reference_run_tag: main
  rows:
    blob: [blob_1, blob_2, blob_3]
    det_tfb: [det_tfb_1, det_tfb_2, det_tfb_3]
    det_lap: [det_lap_1, det_lap_2, det_lap_3]
    lora_ens3: [lora_ens3]
  ensemble_row: lora_ens3
  reference_rows: {blob: c3, det_tfb: c4_tfb_fixed, det_lap: c4_lap_refit}   # D2, D4
  id_domains: [hackernews, stackexchange]
  ood_domains: [arxiv, freelaw, pubmed_abstracts]
  bootstrap: {resamples: 10000, seed: 0, level: 0.95, unit: document, stratify_by_class: true,
              percentile_method: linear}
  weights: recompute_per_cell    # w_b = 1/k_d inside the cell; the stored weight is not read
  margin_auroc: 0.02
  alpha: 0.05
  min_scored_seeds: 2
  seed_rule: all_scored_seeds_same_sign
  families:
    F5_s5_vs_ensemble_hn: {id_domain: hackernews, rows: [blob, det_tfb, det_lap]}
    F6_s5_vs_ensemble_se: {id_domain: stackexchange, rows: [blob, det_tfb, det_lap]}
  descriptive: [D1_per_seed, D2_c3_in_range, D3_seed_vs_ci, D4_base_effect, D5_matched_n3,
                D6_fit_budgets]
  matched_n: 3
  plan: {rescore_night_gpu_h: 8.0, night_gpu_h: 8.0,
         fallback_subset: first_block_per_document, first_block_rates: batch_1}
```

## 6. Verification evidence

### 6.1 Session-02 refuter verdicts that apply

Three refuters worked on CPU only, read only. Their verdicts are in S1 Section 6.1 and S2 Section 6. S5 uses the parts below.

| Claim | Verdict | Evidence (file:line) | What S5 takes from it |
|---|---|---|---|
| P5, item 4: C4-TFB sits on the BLoB mean | confirmed | experiments/c_pipeline.py:107-110, 123-159; scripts/eval_c_checkpoints.py:133-150 (commit 1dcb12f). The largest absolute difference between `a_map` and C3's `lora_A_mu` is 0 on all 32 layers, and $B$ equals C3's `lora_B` (S2 Section 6, P5 table). | P18 is real. The b2 fits start from deterministic LoRAs, and S5-T2(d) checks every base hash. |
| P5: the TFB sampler skips the rotation | confirmed | minigpt/tfb.py:73-74, 188-192 (commit 1dcb12f) | The b2 TFB fits use S2's v2 sampler only (`sampler_version` = `v2` in S5-T2d). |
| P4: diagonal Laplace was never scaled | confirmed | minigpt/laplace.py:112-123, 167-168 (commit 1dcb12f); stored std median 1.000 against a median weight of 0.023 | The b2 Laplace fits use S2's scaled v2 state and its λ match. |
| P1(a): the LoRA rows were scored on StackExchange | confirmed | scripts/eval_c_checkpoints.py:96-109; minigpt/data.py:207-213 (commit 1dcb12f) | HackerNews is the primary ID set for F5. StackExchange is F6. |
| D5 (code-assessment.md): the post-hoc cells are fitted on the BLoB mean | confirmed by the P5 refuter | experiments/c_pipeline.py:108-151 | The fix is a deterministic LoRA per seed (K36), as in this spec |

### 6.2 This draft's own checks (session 03, CPU only, read only)

The probe script is `scratchpad/s05/s5_seed_probe.py` in this session's scratchpad. It imports the real `train`, `inject_lora` and `apply_sampled_params`. Nothing in the repo changed. The file:line references in this table are to commit 1dcb12f, except where Section 2 names the working tree.

| Check | Result | Evidence |
|---|---|---|
| The C pipeline never seeds | 0 calls to `torch.manual_seed` in experiments/c_pipeline.py and experiments/pipeline_runner.py. C3's YAML has `train.seed: 1337` (configs/c3_phase2.yaml:60), but nothing applies it. So C3's seed is unknown. | grep; code-assessment.md:80 |
| C3 started from a fresh optimizer | The pipeline loads the base through `load_checkpoint` into the model only and calls `train()` without `resume_ckpt` | experiments/c_pipeline.py:166-170; experiments/pipeline_runner.py:366-375 |
| Seed determinism on CPU | 2-layer fixture, 30 steps. Same seed: `torch.equal` on all 8 (deterministic) and all 12 (BLoB) LoRA tensors, also across two separate processes. Different seed: max abs difference 0.354 and 0.334. | `s5_seed_probe.py`; fixture as in S5-T1 |
| Ensemble member swap | The logits under `apply_sampled_params` with a member's `lora_A` and `lora_B` equal that member's own logits (max difference 0). The host is restored exactly. The frozen base is identical across members. | `s5_seed_probe.py`; minigpt/laplace.py:185-210 (1dcb12f; 349-373 in the working tree); minigpt/lora.py:176 |
| Identical members | $\max\lvert g_t\rvert=0$, $\max\lvert\mathrm{MI}_t\rvert=9.5\times10^{-7}$ (float32 rounding). This sets S5-T3(a) at $10^{-6}$ for $G$ and $10^{-5}$ for $M$, as in S1-T5c. | `s5_seed_probe.py` |
| Distinct members | mean MI $6.5\times10^{-5}$, mean $g_t$ $8.8\times10^{-5}$, min $g_t$ 0 on the untrained fixture | `s5_seed_probe.py` |
| A cache miss changes the data with the seed | `load_pile_data` passes `cfg["train"].get("seed", 1337)` to the streaming shuffle | minigpt/data.py:186; :149-151 |
| Config helpers with defaults | `build_lora_config` (minigpt/config.py:203-212) and `build_train_config` (:229-235) use `.get` | grep |
| C3 time base and size | `train_time_sec` 1,614.1 s (26.9 min), `best_val_step` 7,000, patience not triggered (S0 Section 6). `c3/ckpt_best.pt` is 329,018,426 bytes. | mlflow.db run 3b22bfcf (read by S0); `ls` |
| Trainable parameters | deterministic LoRA 1,310,720; BLoB 1,966,080 | S0 Section 6 (meta-device count); $16\times 2\times(16\cdot512+2048\cdot16)=1{,}310{,}720$, plus $655{,}360$ for `lora_A_g` |
| The eval loader hardcodes C3's path | `_checkpoint_paths` and `_blob_to_deterministic_lora` read only `c3/ckpt_best.pt` | scripts/eval_c_checkpoints.py:114-131, 133-150 (1dcb12f; 328-362 in the working tree) |

### 6.3 Differences from roadmap Section 3 and the options document

These go into `agents/design-rationale.md` when S5 is built.

| # | Roadmap or options document | This spec | Why |
|---|---|---|---|
| 1 | Options b1: 2 more BLoB seeds. Roadmap S5-T2: 3 BLoB seeds. S0 R9 left the count to S5. | 3 new BLoB seeds; C3 is a reference row (choice C1) | C3 was never seeded and came from the pipeline. It cannot meet S5-T2's provenance checks. |
| 2 | S0: S5 may reuse S0's adapter | Not reused | The config hash cannot match (Section 5.2). The saving is 7-27 GPU min. |
| 3 | Roadmap S5-T3: the ensemble "appears in the main table" | It is a row of `table_s5.json`, which S4 and m11 read, and it is the reference of the frozen families F5 and F6 | A spec can check a file, not a paper table |
| 4 | No S5 endpoint in the roadmap | F5 and F6: Bayesian LoRA rows against the ensemble, with a seed-mean paired bootstrap and a sign rule over seeds | S1 requires a frozen family for any new row. P3 asks exactly this question (BLoB Table 1 compares with "ENS"). |
| 5 | Roadmap: one night (Tue Oct 6), 2-8 GPU h | Two nights (three in the worst case), 6.9-15.6 GPU h under the plan rule, at most 18.4 | S2's refit method and S1's rebuilt block counts (Section 1) |
| 6 | Options b2: re-score at 1-2 blocks per document | All blocks, or the first block of each document by a frozen cost rule | The rule reads only costs and is fixed before any score |
| 7 | Roadmap S5-T4: mean and min-max | Adds D3 (seed range against the CI) and D5 (matched $N=3$) | P14 asks whether seed variance dominates eval noise. The ensemble uses 3 passes and the Bayesian rows 20. |

### 6.4 Critic pass against the built code (session 03)

The critic read the built S0-S2 code in the working tree and changed the draft as follows.

| # | Finding | Evidence | Change |
|---|---|---|---|
| 1 | A first-block subset tagged `main` fails two S1 checks, because the scorer copies the full-set `weight` and S1 requires a `main` file to hold the reference block index | scripts/eval_c_checkpoints.py:773; scripts/check_eval_rebuild.py:464-466, 1085-1088 | Run tag `s5` for every S5 file; weights recomputed per cell; S1's `--check align` and `--check prereg` added to the pre-merge command |
| 2 | S2's `SCORE_SET_CELLS` gates the refit loader and `check_scores`, and it lives in S2's module | minigpt/posthoc_refit.py:83, 858-888; scripts/eval_c_checkpoints.py:520-529 | A scorer-local `S5_REFIT_SETS` map; S5's own score-to-record check |
| 3 | The draft's per-sample step bypassed the built hook, and its spy target ("S1's shared statistics helper") does not exist as one function | scripts/eval_c_checkpoints.py:443-467, 610-698 | The ensemble is a registered parameter sampler; S5-T3(c) spies on `score_block_batch` and `realized_token_scores` |
| 4 | The ensemble loads 3 files, not 4; `checkpoint_sha256` is a path-keyed dict | scripts/eval_c_checkpoints.py:1098 | S5-T3(e) expects 3 entries |
| 5 | `_batches` splits at every gap in $b$, so a first-block subset gets little batching gain | scripts/eval_c_checkpoints.py:701-714 | The first-block re-score is priced at batch-1 rates (5.29 GPU h, not 1.76); the plan uses two rate sets |
| 6 | The fit record has no wall-time field, and S2 is built | minigpt/posthoc_refit.py:1228-1246 | Fit times from S5's night-1 log; the request to S2 dropped |
| 7 | Fit-record field names differ from the draft (`base.kind`, `base.sha256`, `same_fit_as_tfb`) | minigpt/posthoc_refit.py:292-300, 1147-1175 | S5-T2(d) and D6 use the built names |
| 8 | S2's curvature-median check compares every Laplace refit with C4-LAP's BLoB-mean state and fails the fit outside 0.5-2x | minigpt/posthoc_refit.py:85, 1138-1146; scripts/eval_c_checkpoints.py:554-556 | Choice C4; night-list rules in Section 5.3; a risk row in Section 8 |
| 9 | LoRA keys carry `.base_linear`, so C0 keys do not match one to one | minigpt/lora.py:46, 130 | S5-T2(c) maps the keys |
| 10 | The seed script had no source for the vocabulary size, and the fixture uses 100 | minigpt/posthoc_refit.py:249 | Read from the base checkpoint (step 5) |
| 11 | S2-T4(d) includes a re-score check that b2's refits do not need | S2 Section 4 | S5 waits only for the fit part |
| 12 | D6 of the draft (G against M on the S5 rows) is S1's question, not P3, P14 or P18 | S1 families F1-F4 | Removed; the old D7 (fit budgets) is now D6 |
| 13 | The draft's "Opus agents" roadmap figure is not in roadmap row S5 | agents/plans/roadmap.md:90 | Labelled as CA6 |

## 7. What this does not cover

- Base-model seeds. All LoRAs share one C0 base, so the seed spread covers adapter training only. C0, C1 and MC dropout keep one training seed (options Section 7, backlog B5; Ambitious a2).
- Full-retrain deep ensembles (backlog B3) and M=5 (choice C3; Ambitious a3).
- A seed-level significance test. The bootstrap resamples documents. Three seeds support only the sign rule.
- MC dropout mask seeds. S1's spread branch covers eval-time noise for MC dropout, C1 and C3.
- The S5 rows on `arxiv_stripped` and `legacy_d1`. S3's one-pass baselines, noise control, combined-score test and matched bins for the S5 rows. S3 may reuse the S5 score files. Each member's sequence NLL can be derived from the ensemble file's `logp_real`.
- The $G$-against-$M$ contrast on the S5 rows. It is S1's question (F1-F4). The first draft had it as D6; it is outside P3, P14 and P18. Anyone who computes it later from the S5 files reports it as post hoc.
- Latency per millisecond (P22). D5 compares at equal passes only; S3 times the passes.
- Any hyperparameter change or tuning for the seeds. C3's schedule is kept as it is.
- Bitwise reproduction of the GPU training runs. S5-T1 checks CPU determinism only.
- The S2 rows `c4_tfb_fixed` and `c4_lap_refit`. They stay as one labelled secondary row, "BLoB-mean + TFB" and "BLoB-mean + diag. Laplace" (S2 Section 7). S5 compares with them only descriptively (D4).
- The LoRA versus full-weight confounds (P8). The seeds do not address them.
- Stage A and Exposure (S6, S7). Their seed runs (b11) reuse this seed script.
- Paper text, README.md and report.md (m3, m11, m13).
- The measured GPU hours. S0 and S2 give the inputs, and `--plan` computes them.

## 8. Confidence

| Claim | Confidence | Why |
|---|---|---|
| The seed script gives bitwise-identical adapters on CPU for a fixed seed | high | The session-03 probe ran both kinds twice, in two processes |
| Every b2 fit records a deterministic LoRA base, never a BLoB checkpoint | high | S2's strict `det_lora` loader rejects BLoB keys (minigpt/posthoc_refit.py:270-287), and S5-T2(d) compares every base hash |
| The ensemble MI and $G$ come from the same code as the Bayesian rows | high | The member swap reproduces each member exactly (probe). The ensemble goes through the built sampler hook, and S5-T3(c) checks it with spies. |
| The S5 files leave S1's results and checks unchanged | high | S1's `--analyze` and `--rederive` read only `main` files; `--check align` matches other tags by `block_index` (critic check, Section 6.4) |
| The 6 fine-tunes take 1.7-2.7 GPU h | medium | BLoB is one measured run (26.9 min). The deterministic time is unmeasured until S0-T2. CA4's tail gives up to 6.7 h. |
| The 6 refits take 1.9-7.65 GPU h | medium-low | These are S2's estimates for C4, not measurements |
| The re-score fits one night under the plan rule | medium | The first-block re-score is 5.29 GPU h at batch 1. Escalation needs batch-1 rates 1.5x or more above CA10. |
| The S5 run needs 2 nights | medium | The queue total is 6.9-15.6 GPU h at the ends of the range under the rule. A batching gain near the rule threshold (about 1.3-1.5x) gives up to 18.4 GPU h and a third night. |
| Engineering fits 6.5-10.75 h | medium-low | Built up from the options items plus the frozen family, the plan rule and the re-derivation |
| Review needs 0.75-1.5 h | medium | CA1's rate on 4.75-7.5 engineering hours of new code |
| All 3 b2 Laplace fits pass S2's curvature-median check | medium-low | The reference is C4-LAP's state on the BLoB-mean base. $\hat F$ on $A$ scales with $\lVert B\rVert^2$, and $B$ can differ between BLoB and deterministic training. Unmeasured. Choice C4 is the fix. |
| All 3 Laplace refits reach a scoreable status | medium-low | Same basis as S2 for the λ match (the median scaled precision lies inside the λ grid), plus the median check above. Unmeasured on deterministic bases. |
| An M=3 ensemble gives a usable disagreement signal | low-medium | $B$ starts at 0, so members differ only by the $A$ init, the batch order and dropout. code-assessment.md calls M=3 MI noisy. |
| Three seeds detect a Bayesian-vs-ensemble gap of 0.02 | low | The sign rule is conservative. With 3 seeds it detects only gaps larger than the seed spread. |
| The BLoB seeds reproduce C3 inside their range | unknown | No seed run exists yet. D2 reports it. |
| The Bayesian LoRA rows beat the ensemble | unknown; the literature leans to no | LoRA-Curve (arXiv 2605.29580) reports that ensembles add little for LoRA, and G-NLL beats multi-sample methods in 13 of 18 settings (options PG12). N1 is written for this case. |
