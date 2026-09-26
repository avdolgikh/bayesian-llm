# S6 Stage A: decode and rescore the models' own greedy answers (`specs/i2-stage-a-rescore.md`)

Status: DRAFT (not approved). Date: 2026-09-26 (revision 1, after one critic pass; Section 6.3 maps each critic item to its fix). Owner: Alexey approves; agents implement.
Plan of record: `agents/plans/roadmap.md` Section 3, row S6. Stage A runs in v1 only if G1 passes on Sun 2026-10-11. Otherwise it moves to the November v2 with this spec unchanged except the dates.

## 1. Header

| Field | Value |
|---|---|
| Items | **b3**: this spec (the decode-and-rescore protocol and the Stage A endpoints). **b4**: the decode-and-rescore module. It decodes a greedy answer with the mean weights, rescores the answer with N weight samples in one stacked pass, and computes $g_t$, full-vocabulary MI and decoder-matched EU per token, and their sum, mean and max per response. It has end-to-end tests on a fixture, and two checker modes that re-derive the results without the module. **b5**: the Stage A run. Prompts come from S1's unseen test documents (1,000 per domain). 6 score sets. Answers of at most 64 tokens. One-pass baselines: G-NLL, max-prob, entropy, prompt NLL and answer length. Latency of decoding and rescoring, with the merged mean-weight LoRA. |
| Issues | **P7 (in part)**: no generated token is scored today. Stage A scores the models' own answers, but only for domain shift. There are no correctness labels and no ground truth for epistemic uncertainty. S7 (Exposure) adds those. **P10**: overstated production claims. Stage A times decoding and rescoring on real generated answers, times the merged mean-weight LoRA against C0, and reports the N-sweep as a share above chance on generated answers. The text fixes stay in m3. |
| Approve by | Thu 2026-10-01 (roadmap Section 3, row S6) |
| Build | Session 04, Mon 2026-10-05. CPU only: S6-T1, S6-T2, S6-T4(a)-(d) and (f), and S6-T5(a). At the end of the session, `--part freeze` (Section 5.1, step 5). The roadmap key-date row says session 03 (Wed Sep 30). The session-03 plan moves S6 out of session 03, because S6 is approved only on Thu Oct 1. |
| Gate | G1, Sun 2026-10-11. (b) Alexey's logged hours through Oct 11 are 8.5 h or less. (c) `uv run pytest tests/test_stage_a.py -k "s6_t1 or s6_t2"` passes with no test skipped, on the build branch. If (b) or (c) fails, Stage A moves to v2. The sunk cost is the approval (0.25-0.5 h), and v2 reuses it. |
| Run | Tue 2026-10-13, only if G1 passes. Alexey reviews the module (1-1.5 h). That night: prompts, decode, latency, rescore, re-run (1.7-5.0 GPU h, one night). Wed Oct 14: analysis, report and the two checker calls on CPU. Stage A cutoff Tue Oct 20: the results are reviewed and the b6 subsection is drafted, or Stage A moves to v2. |
| Depends on | S1 (`specs/i2-eval-rebuild.md`, built in session 03): the manifest, the stored test-document tokens and block files, the score-set registry and meta helpers in `scripts/eval_c_checkpoints.py`, `realized_token_gap`, `doc_bootstrap_auroc`, `paired_doc_bootstrap` and `holm` in `minigpt/uncertainty.py`, the freeze rule, S1's `prereg.json`, and the `main` score files of `test` and `test_hn` (for $\bar g_{\mathrm{ref}}$ only). S2 (`specs/i2-posthoc-fixes.md`, built): the three refit states and fit records under `data/checkpoints/i2_posthoc/`, and the v2 samplers. S0 (`specs/i2-timing-probe.md`, built): `data/i2/timing_probe.json` and the header helper. Section 2 lists the exact names. |
| Feeds | b6 (paper subsection and figure panel, in m11). S4, if a Stage A panel is added there. S7 (Exposure reuses the module in November). m3 and m13 (the P10 wording, Section 7). |

Costs. The roadmap column is roadmap Section 3, row S6 (b3 + b4 + b5 from options-and-costs.md Section 4.2). Where the row has no value, the column says so and names the options assumption instead. The next column is this spec's own estimate.

| Cost | Roadmap row S6 | This spec | Difference and why | Assumption |
|---|---|---|---|---|
| Alexey, approve | 0.25-0.5 h (b3) | 0.25-0.5 h | none | CA1 |
| Alexey, review | 1.5-2.25 h (b4 1-1.5 after G1; b5 0.5-0.75) | 1.5-2.25 h if the review covers the core files only (choice C3). CA1's rate on 21.5-41 engineering h of new code gives 2.4-6.8 h. | up to +4.5 h for a full line-by-line review | CA1 |
| Engineering | 20.5-41 h (b3 0.5-1; b4 16-32; b5 4-8) | 22-42 h (est.): b3 0.5-1 (this draft); b4 17.5-33 (build-up below); b5 4-8 | +1.5 / +1 h: the stacked pass for 5 sampler kinds and the separate re-derivation. Reuse of S1 and S2 code offsets most of it. The plan keeps the roadmap range. | CA2 (W2's "about 1 week" read as 20-40 h, split 80/20 between b4 and b5) |
| GPU (RTX 4070) | 1-5 GPU h | 1.7-5.0 GPU h (est., Section 5.10) at 1,000 documents per domain; 1.0-2.6 at 500 (choice C2) | inside the row. Answers are shared by the score sets of one decoder (3 decodes per prompt, not 6). Sample passes apply the vocabulary head to the 64 answer positions only. Each score set re-runs the decoder pass for $q_j$ (+0.03 GPU h). | CA3, CA10 |
| Network, disk | not in the row | 0; under 0.3 GB under `data/` | none | est. |
| Opus agents | not in the row; CA6 gives 5-8 per implementation spec | 5-8, one of them writes the checker modes from this spec only | none | CA6 |
| Cloud | not in the row; $0 in the roadmap total | $0 | none | CA7 |

Not in S6: b6 (paper subsection and figure panel: 2-3 engineering h, 0.5-1 Alexey h) and b7 (one more agent session, 0.25-0.5 Alexey h). Cut step 1 of the roadmap (move Stage A to v2) saves b4-b7: 2.25-3.75 h.

Build-up of b4 (est.):

| Part | Engineering h |
|---|---|
| Prompt builder from S1's manifest, with alignment checks and the whole-batch rule | 1-2 |
| Greedy decode, stop rule, decoder files | 1.5-3 |
| Decoders: S1 and S2 loaders, the LoRA merge, the mean-weight paths | 1.5-3 |
| Stacked rescoring for 5 sampler kinds, MC dropout, and the per-sample loop reference | 4-8 |
| Per-token and per-response scores, score files, meta, freeze | 2-3 |
| Analysis, $\bar g_{\mathrm{ref}}$, families, verdicts, report | 2-4 |
| Checker: `--check stage-a` and `--rederive-stage-a` (separate agent) | 1.5-3 |
| Latency part | 1-2 |
| Tests S6-T1, S6-T2, S6-T4(a)-(d) and (f), S6-T5(a) | 3-5 |
| **Total** | **17.5-33** |

Differences from roadmap Section 3, for Alexey to accept at approval:

| # | Roadmap text | This spec | Why |
|---|---|---|---|
| R1 | T1: "$g_t \ge 0$"; "$g_t$ and full-vocabulary MI are both 0" | $g_t$, MI and EU are each at least $-10^{-6}$, and 0 to $10^{-6}$ for identical samples. T1 adds a greedy-identity check, a position-alignment check, the stop rule and a padding check. | Float tolerance. An off-by-one between answer tokens and logits would pass the roadmap's checks. |
| R2 | T2: batched and per-sample scores match to 1e-5 | Kept for VI, BLoB, TFB v2 and both Laplace v2 kinds. MC dropout gets a determinism check instead. | Dropout masks are drawn per element inside the layers, so a replicated batch cannot reuse the loop's draws. |
| R3 | T3: "1,000-2,000 ID and OOD prompts" | 1,000 prompts per domain, one per S1 test document: 4,000 per decoder for C0 and C1, and 5,000 for the LoRA decoder, which adds StackExchange as a secondary ID set (choice C4) | With 500 documents per class, the standard error of a paired AUROC difference is about 0.010-0.014 (est., Hanley-McNeil at AUROC 0.8, correlation 0.75-0.5). Holm over 9 then needs a difference of about 0.03-0.04, above the 0.02 margin. At 1,000 per class the standard error is about 0.007-0.010, so Holm needs about 0.02-0.03: close to the margin, not below it. The GPU cost stays inside the row (choice C2 keeps 500). |
| R4 | T3: "answers of at most 64 tokens" | Greedy to 64 tokens, then the answer is cut before its first repeated 4-gram. The uncut 64-token answer is scored too, as a descriptive variant. | In the drafting probe, all 32 greedy answers repeated a 4-gram within 64 tokens (Section 6.2). The models never emit an end token, because the caches hold none (S1 Section 6.1). |
| R5 | T3: sum, mean and max of $g_t$, MI and EU, plus G-NLL, max-prob, entropy, prompt NLL and answer length | Kept. G-NLL in mean and summed forms. Adds the repeat share of the 64-token answer, and $g_t$ on the reference continuation of the same prompt (from S1's score files, no GPU). | Degeneration and "own answer vs someone else's text" are the two confounds this protocol exposes (P7). |
| R6 | T3: "6 models" | The 6 stochastic score sets that S1 and S2 score: `mc_dropout`, `c1`, `c2_refit`, `c3`, `c4_tfb_fixed`, `c4_lap_refit`. They use 3 decoders. The pre-fix `c4_tfb` and the old Laplace states are excluded. | P4 and P5 (confirmed). C3 and the two C4 rows share one decoder (Section 5.4). |
| R7 | No margin given | $\delta = 0.02$ AUROC, as in S1 | Twice S1's ±0.01 reproduction tolerance |
| R8 | Build in session 03 (roadmap key dates) | Build in session 04 | Session-03 plan, "Out of scope" |
| R9 | T5: "reported next to the time of one greedy decode" | Adds a merged-vs-unmerged equality check and a wording rule for "zero overhead". States that decoding has no KV cache. | P10. AGENTS.md parks the KV cache under Future Work. |
| R10 | T4: "the report quotes the pre-registered endpoint" | The report reads the endpoint from the frozen YAML blocks (`stage_a`, `decode`, `score_sets`, `analysis`) and exits non-zero if their hash changed. A test also checks the frozen YAML values against Section 3. | A copied endpoint could drift from the approved one |

What Alexey needs to do:

| # | Action | By | Minutes | Default if Alexey is silent |
|---|---|---|---|---|
| A1 | Approve S6. This fixes Section 3. | Thu Oct 1 | 15-30 | Not built. Stage A goes to v2 with this spec still DRAFT. |
| A2 | Choices C1-C4 (Section 2) | Thu Oct 1 | inside A1 | The defaults in Section 2 |
| A3 | G1: log the hours; decide on Stage A | Sun Oct 11 | inside G1 | Hours above 8.5 h, or S6-T1/T2 failing, move Stage A to v2 |
| A4 | Review the module: `minigpt/stage_a.py`, the verdict code and the tests | Tue Oct 13 | 60-90 | Stage A does not run in v1 |
| A5 | Review the results and `stage_a_report.md` | by Tue Oct 20 | 30-45 | Stage A moves to v2 (cutoff) |

## 2. Scope and non-scope

Scope check. Every part maps to b3, b4 or b5, and to P7 (in part) or P10:

| Part | Item | Issue |
|---|---|---|
| Section 3: endpoint, families F1-F3, verdict rules, framing sentences | b3 | P7 |
| `minigpt/stage_a.py`, `scripts/stage_a.py`, `tests/test_stage_a.py`, the two checker modes | b4 | P7 |
| Prompts, decoding, rescoring, one-pass baselines, degeneration table, $\bar g_{\mathrm{ref}}$ | b5 | P7 (the two confounds of R5) |
| Latency, merge check, N-sweep share above chance | b5 | P10 |
| Wording rules and the m3/m13 handoff (Section 7) | b3: framing written before the run. S6 edits no text. | P10 |
| StackExchange prompts for the three LoRA rows | b5, descriptive rows only | P1 is S1's issue. Choice C4 can cut it. |
| Removed in revision 1: S3's combined-score test on the Stage A scores | m9 (S3) | P2, P3. Now in Section 7. |

What changes:

| File | Change |
|---|---|
| `minigpt/stage_a.py` (new, about 350-450 lines, est.) | `answer_length(tokens, n, max_len)`; `decode_greedy(decoder, prompts, max_new, batch_size, amp)`; `merge_lora_mean(model, a_key)`; the stacked-sample context manager and the draw collectors per sampler kind (Section 5.5); `rescore(...)` (stacked) and `rescore_loop(...)` (reference); per-token and per-response scores; the freeze function; the verdict functions. The API takes any prompt tensor, so S7 can reuse it. |
| `scripts/stage_a.py` (new, about 150 lines) | CLI: `--config`, `--part {freeze, prompts, decode, latency, rescore, analyze, report, all}`, `--run-tag`, `--response-ids`. `all` runs every part except `freeze`, in the Section 5.1 order. With `--response-ids`, only `decode` and `rescore` run, and the IDs must select whole batches (Section 5.3), or the script exits 2. Exits non-zero when a check fails. |
| `configs/i2_stage_a.yaml` (new) | Every key explicit (Section 5.9) |
| `tests/test_stage_a.py` (new) | S6-T1, S6-T2, S6-T4(a)-(d) and (f), S6-T5(a). Test names start with `test_s6_t1_`, `test_s6_t2_`, `test_s6_t4_` and `test_s6_t5_`. CPU only. No test reads `data/`. T4(c) reads `configs/i2_stage_a.yaml`. |
| `scripts/check_eval_rebuild.py` (S1's) | Two new modes: `--check stage-a` (a new choice of `--check`) and `--rederive-stage-a` (a new flag in the mutually exclusive mode group). Each runs as its own call with `--config configs/i2_stage_a.yaml`. The scores directory is `rescore.out_dir`, and S1's YAML comes from `stage_a.eval_config`. Called with S1's YAML, they exit 2. A separate agent writes them from Sections 3, 5.2, 5.3, 5.6, 5.7 and 5.8 of this spec. They keep the checker's import rule (scripts/check_eval_rebuild.py:3-5): numpy, torch for `torch.load`, PyYAML and sklearn; no `minigpt` module and no file under `scripts/`. Exit codes as S1's: 0 PASS, 1 FAIL, 2 bad input. |
| `data/stage_a/`, `data/scores_i2/stage_a/` (new, gitignored) | Prompt file, decoder files, score files, $\bar g_{\mathrm{ref}}$ files, tables, report, latency JSON, `prereg_stage_a.json`, check reports. The subdirectory keeps S1's `--check prereg` and `--check align` unaffected, because they read `data/scores_i2/*.pt` only (scripts/check_eval_rebuild.py:1011-1017). |
| AGENTS.md, README.md, `agents/technical-reference.md`, `agents/design-rationale.md`, `agents/milestone-history.md` | Structure block (`stage_a.py` in `minigpt/`, the scripts line, the test count); README test count and scripts; the Stage A protocol and file formats; the design choices (stop rule, stacked pass instead of vmap, shared decoders, G-NLL form); the milestone record when Stage A completes |

What must not change:

- `minigpt/model.py`, `layers.py`, `lora.py`, `train.py`, `data.py`, `tfb.py`, `laplace.py`, `uncertainty.py`, `evalset.py`, `posthoc_refit.py`. S6 only imports them. A helper that is missing goes into `minigpt/stage_a.py`.
- `scripts/eval_c_checkpoints.py`, `scripts/eval_mc_dropout.py` and `scripts/timing_probe.py`. S6 imports them with `importlib`, as S1's tests do (tests/test_eval_scripts.py:219-220).
- S1's checker modes (`--rederive`, `--freeze`, `--check prereg`, `--check align`) and their outputs. `tests/test_check_eval_rebuild.py` passes unchanged, including `test_checker_imports_no_scorer_code`.
- `configs/i2_eval.yaml` (including its frozen `analysis` block), `configs/i2_posthoc_*.yaml`, `configs/i2_timing_probe.yaml`, S1's `prereg.json` and `data/i2/timing_probe.json`. S6 freezes its own blocks in its own file (S1 Section 3: "A row that a later spec adds gets its own family in a separate YAML file").
- S1 and S2 score files, `data/eval_i2/`, and everything under `data/checkpoints/`.
- `experiments/`, `paper/`, report.md and the README numbers.

Interface with S1, S2 and S0, as built in session 03 (read on 2026-09-26):

| S6 needs | Built interface |
|---|---|
| Unseen test documents with IDs and offsets | `data/eval_i2/manifest.jsonl` rows with `split: test`, `variant: raw`. Keys used: `doc_id` (`{key}/{i:09d}/{sha1[:12]}`), `domain`, `eval_set` (`test` or `test_hn`), `o_d`, `k_d`, `L_d`. `docs_{key}.pt` = `{"doc_id": [...], "tokens": [one int32 tensor per document]}`, in manifest order (minigpt/evalset.py:7-18). Helpers: `load_manifest`, `doc_file_path(cfg, key, "raw")`, `block_file_path` (minigpt/evalset.py:168-187). S1-T2b guarantees the documents are unseen. |
| S1's block 0 of a document | `blocks_{test,test_hn}.pt`: the row with `block_in_doc == 0`, whose `offset` equals $o_d$ |
| Model loaders that record what was loaded | `SCORE_SET_REGISTRY[s].load(device)` returns `(model, method, state)`, and `SCORE_SET_REGISTRY[s].checkpoint_paths()` lists the files to hash (scripts/eval_c_checkpoints.py:427-480). `c0`, `c1`, `c3` go through `load_model` (method `deterministic`, `variational`, `variational`). `c2_refit`, `c4_lap_refit`, `c4_tfb_fixed` go through `load_posthoc_refit` (method `laplace_v2`, `laplace_v2`, `tfb_v2`; scripts/eval_c_checkpoints.py:567-603). `mc_dropout` is not in the registry: `load_c0_model(device)` in scripts/eval_mc_dropout.py:107-116, method `dropout`, checkpoint `data/checkpoints/c0/ckpt_best.pt`. |
| Meta and determinism helpers | `file_sha256`, `code_sha256`, `git_state`, `apply_determinism`, `WarningLog` (scripts/eval_c_checkpoints.py:812-880); S1's meta keys (scripts/eval_c_checkpoints.py:1094-1119) |
| The $g_t$ helper | `realized_token_gap(logp_real)` (minigpt/uncertainty.py:227-254). It returns `log_pbar`, `g` and `G`. `G` is the mean over all positions, so Stage A aggregates `g` with its own mask. |
| Bootstrap and Holm | `doc_bootstrap_auroc(scores, labels, doc_ids, weights, n_resamples, seed, level, *, target_tpr)`, `paired_doc_bootstrap(scores, labels, doc_ids, weights, pairs, n_resamples, seed, level)` and `holm(pvalues)` (minigpt/uncertainty.py:672-795). `seed` accepts `[seed, i_id, i_ood]`. `weights` is required; Stage A passes ones. |
| The freeze rule | `analysis_sha256`, `freeze_prereg` and `verify_prereg` (minigpt/evalset.py:198-234). S6 copies the rule in its own function (Section 5.7), because `freeze_prereg` hashes only `cfg["analysis"]` and reads `cfg["eval_set"]["name"]`. |
| $g_t$ on the reference continuation | `data/scores_i2/{s}__test__main.pt` and `{s}__test_hn__main.pt`: `logp_real` [B, N, 256] float32, `block_index` [B] int64, `doc_id` and `domain` (lists of str), `offset` [B] int64, `weight` [B] float32 (scripts/eval_c_checkpoints.py:726-775). `test` holds all six score sets; `test_hn` holds `c3`, `c4_tfb_fixed` and `c4_lap_refit` (configs/i2_eval.yaml, `scoring.score_sets`). |
| v2 samplers | `PARAM_SAMPLERS["tfb_v2"]` and `PARAM_SAMPLERS["laplace_v2"]`: `sample_tfb_params` and `sample_laplace_params` restricted to v2 states (scripts/eval_c_checkpoints.py:504-517). Laplace v2 draws on the state's device from `torch.Generator(device=...)` (minigpt/laplace.py:319-336), and `load_posthoc_refit` maps the state to the scoring device, so the run draws on the GPU. TFB v2 draws on the CPU and moves the noise (minigpt/tfb.py:293-319). `apply_sampled_params` swaps them in (minigpt/laplace.py:348-373). |
| Which rows exist | `load_posthoc_refit` raises ValueError for a fit that did not pass: `exit_code` not 0, a non-empty `failures`, a Laplace `match_status` outside `SCORED_STATUSES` = (`matched`, `matched_noisy`, `tighter_than_budget`), or a state or base hash that differs from the fit record (scripts/eval_c_checkpoints.py:545-590). A `not_matched` fit writes no state file (minigpt/posthoc_refit.py:1131-1136). Refit outputs: `data/checkpoints/i2_posthoc/{c2,c4_lap,c4_tfb}/` with `laplace_state.pt` or `tfb_state.pt` and `fit_record.json` (`out_dir` of `configs/i2_posthoc_*.yaml`). |
| The shared LoRA decoder | `configs/i2_posthoc_c4_{tfb,lap}.yaml`: `base_kind: blob_mean`, `base_checkpoint: data/checkpoints/c3/ckpt_best.pt`. `load_posthoc_base` maps `.lora_A` from `.lora_A_mu` and records `base.sha256` (minigpt/posthoc_refit.py:222-301). `c2_refit`: `base_kind: full` on `data/checkpoints/c0/ckpt_best.pt`. |
| Batch sizes and the header | S0's `data/i2/timing_probe.json` (`probe.output`). `rescore.cells[]` holds `method` (legacy names `c0`, `c1`, `c2`, `c3`, `c4_tfb`, `c4_lap`, `mc_dropout`), `batch_size` (1, 8, 32), `ms_per_block`, `peak_reserved_mib`, `over_budget` and `sampler_ms_per_sample`. The cells time one forward per sample on 256-token blocks, so S0 has no cell for the stacked pass (Section 5.5). Header: `collect_header(cfg, config_path, dry_run)` and `HEADER_FIELDS` (scripts/timing_probe.py:71-76, 375-405). |
| Checker CLI | scripts/check_eval_rebuild.py:1118-1167: one mode per call (`--rederive`, `--freeze` and `--check {...}` are mutually exclusive), `--config` (default `configs/i2_eval.yaml`), `--scores-dir`. |

AGENTS.md rules that apply:

- No notebooks. No extra Bayesian library. Only torch.
- No modern transformer tricks. There is no KV cache (AGENTS.md, Future Work). Decoding re-runs the whole context at each step, as `MiniGPT.generate` does (minigpt/model.py:176-192).
- Explicit configs. New code reads `cfg["key"]`, never `.get()` with a default. Test fixtures keep their constants in the test module.
- LaTeX for every formula in this spec.
- Keep repo docs fresh: the new module, the new script and the test count go into AGENTS.md and README.md at once. The design choices go into `agents/design-rationale.md`.
- 4-space indent, 100-character lines, type hints on public functions. `ruff check` and `pytest` before each commit, and `ruff check` on the touched scripts, because CI lints only `minigpt/ experiments/ tests/`. Conventional Commits, only if Q3 allows commits. Never mention AI assistants.

Open choices for the approver (the default applies if Alexey says nothing):

| # | Choice | Default | Alternative | Cost of the alternative |
|---|---|---|---|---|
| C1 | Decode rule | Plain greedy to 64 tokens. The primary answer is cut before its first repeated 4-gram. The uncut 64 tokens are a descriptive variant from the same pass. | (a) Greedy with a no-repeat-4-gram block, as the primary. (b) The same, added as a second pre-registered variant. | (a) 0 GPU h, but the chosen token is then not always the argmax, so G-NLL is no longer the NLL of the greedy answer. The probe text stayed weak. (b) +1 decode and rescore: +1.5-4.6 GPU h (est.), which puts the night at 3.2-9.6 GPU h, and +1 engineering h. |
| C2 | Prompts per domain | 1,000 (all S1 test documents) | 500 (the first 500 in manifest order) | Saves 0.7-2.4 GPU h. The paired CI then cannot resolve the 0.02 margin under Holm (R3). |
| C3 | Review scope | Alexey reviews `minigpt/stage_a.py`, the verdict code and the tests (1-1.5 h). S6-T2 (loop equality), S6-T3(f) (separate re-derivation) and S6-T4(f) (the checker catches a changed value) cover the plumbing. | Full review at CA1's rate | +0.9-4.5 h. G1(b) would then likely fail, so Stage A would move to v2. |
| C4 | StackExchange prompts for the three LoRA rows | Kept, as descriptive rows only. S1 reports the LoRA rows on both ID sets (its families F2 and F4), and these rows let the report set all six score sets on one ID domain. | Drop them: `id_secondary: []` and no `descriptive_rows`. The LoRA decoder then serves 4,000 prompts. | Saves 0.2-0.5 GPU h (1,000 decodes and 3,000 rescorings, est.). F1-F3 do not change. |

## 3. Pre-registered endpoint

This section is fixed at approval, before any decode. The `stage_a`, `decode`, `score_sets` and `analysis` blocks of `configs/i2_stage_a.yaml` hold it. Their sha256 (Section 5.7) goes to `data/scores_i2/stage_a/prereg_stage_a.json`. `--part freeze` writes that file at the end of session 04, before any Stage A answer exists. Every other part refuses to run if the file is missing or the hash differs. A later change is reported as post hoc, next to the frozen result.

Frozen settings:

| Setting | Value | Source |
|---|---|---|
| Prompts | One per S1 test document: the first $P = 128$ tokens of its block 0, $t_{o_d:o_d+128}$. All 1,000 documents per domain (choice C2). Domains: StackExchange, HackerNews, arXiv, FreeLaw, PubMed (raw variant only). | S1 Sections 2 and 5.4 |
| Reference continuation | The next 64 document tokens, $t_{o_d+128:o_d+192}$ | S1 block 0 |
| Decoder | The posterior mean or MAP of each score set (Section 5.4): C0 for MC dropout and C2; the C1 means $\mu$ for C1; the merged BLoB mean for C3, C4-TFB and C4-LAP | P5 (confirmed): C4's $A_{\mathrm{MAP}}$ equals C3's `lora_A_mu` |
| Decode rule | Greedy argmax at every step, ties to the lowest token id (`torch.argmax`), 64 steps, fp16 autocast, no temperature, no repetition penalty | roadmap S6 test 1; choice C1 |
| Answer | Cut before the first repeated 4-gram inside the answer (Section 5.2), so $1 \le L_r \le 64$. Repeats of prompt text do not count. | R4; choice C1 |
| Samples | $N = 20$ for every score set (`stage_a.n_samples`). For VI, BLoB, TFB and Laplace, each (batch, sample) has one weight draw, shared by the responses of the batch. MC dropout masks are drawn per element. Seed rule in Section 5.5. | S1 Section 3 ($N = 20$) |
| Responses kept | All. None is dropped for length or degeneration. | |
| Sign | Higher score means more OOD, for every score. No sign is flipped after the run. | S1 Section 5.6 |
| ID sets | StackExchange for MC dropout, C1 and C2. HackerNews for C3, C4-TFB and C4-LAP, with StackExchange as a descriptive secondary set (choice C4). | S1 Section 3, decision D5 |
| OOD sets | arXiv, FreeLaw, PubMed | S1 |
| Bootstrap | S1's rules (Section 5.6 there): document unit, class-stratified, 10,000 resamples, linear percentiles, one generator per (ID, OOD) domain pair with seed $[6, i_{\mathrm{id}}, i_{\mathrm{ood}}]$. Seed 6 keeps the draws apart from S1's seed 0. With one response per document, the weight of every response is 1. | S1 |
| Margin | $\delta = 0.02$ AUROC | S1 Section 3 |
| Level | 0.05 after Holm, within each family | S1 Section 3 |

Primary score: the per-response mean of the realized-token Jensen gap over the kept answer tokens,

$$\bar g(r)=\frac{1}{L_r}\sum_{j=1}^{L_r} g_j,\qquad g_j=\log\bar p(y_j)-\frac1N\sum_{s=1}^{N}\ell_{s,j}\;\ge\;0 .$$

Primary comparator: the length-normalized G-NLL of the same answer under the same decoder, from one pass,

$$\mathrm{GNLL}(r)=-\frac{1}{L_r}\sum_{j=1}^{L_r}\log p_{\bar\theta}(y_j\mid x,y_{<j}).$$

G-NLL comes from Aichberger et al. 2024 (arXiv 2412.15176). metrics-assessment.md (Section 6, B1) calls it length-normalized. The literature survey calls it the NLL of the greedy decoding and does not say. This spec pre-registers the length-normalized form, because the primary score is length-normalized. The summed form $L_r\,\mathrm{GNLL}(r)$ is reported next to it, with the wording rule below.

Primary contrast, per cell (score set, ID domain, OOD domain):

$$\Delta_G=\mathrm{AUROC}(\bar g)-\mathrm{AUROC}(\mathrm{GNLL}),$$

with the paired document bootstrap p-value of S1 Section 5.6.

Families:

| Family | Rows ([score set, ID domain]) × OOD domains | Contrast | Holm over |
|---|---|---|---|
| S6-F1 (primary) | [`mc_dropout`, StackExchange], [`c1`, StackExchange], [`c3`, HackerNews] × 3 | $\Delta_G$ | 9 |
| S6-F2 (primary, post-hoc rows) | [`c2_refit`, StackExchange], [`c4_lap_refit`, HackerNews], [`c4_tfb_fixed`, HackerNews] × 3 | $\Delta_G$ | 9 |
| S6-F3 (secondary) | the 6 rows above × 3 | $\Delta_P=\mathrm{AUROC}(\bar g)-\mathrm{AUROC}(\mathrm{PNLL})$, with PNLL the prompt NLL under the same decoder | 18 |
| Descriptive | The LoRA rows with ID = StackExchange (choice C4). $\Delta$ of $\bar g$ against MI, EU and the summed G-NLL. Sum and max aggregations. $\bar g$ on the uncut 64-token answer. $\bar g$ on the reference continuation ($\Delta_{\mathrm{ref}}$). Above-chance checks. The N-sweep. Answer length and repeat share. | | none |

S6-F2 mirrors S1's F3: the refit rows have their own family. A row that S1 or S2 does not score, or that fails a shared-decoder assert (Section 5.4), is dropped with its reason, and $m$ for its family shrinks by 3.

Decision rules per cell (the same for F1, F2 and F3, with the family's comparator):

| Verdict | Rule |
|---|---|
| `g_beats` | $\Delta\ge+0.02$ and Holm-adjusted $p<0.05$ |
| `comparator_beats` | $\Delta\le-0.02$ and Holm-adjusted $p<0.05$ |
| `no_difference` | anything else |

Verdict per row (F1 and F2), over its OOD cells:

| Row verdict | Rule |
|---|---|
| POSITIVE | `g_beats` in at least 2 of its 3 cells and `comparator_beats` in none. A row with fewer than 3 cells needs `g_beats` in all of them. |
| NEGATIVE | `g_beats` in no cell |
| MIXED | anything else |

Overall Stage A verdict (F1 and F2 together): SUPPORTED if at least one row is POSITIVE; NOT SUPPORTED if every row is NEGATIVE; MIXED otherwise.

Descriptive rules:

| Check | Rule |
|---|---|
| Above chance | The 95% CI of $\mathrm{AUROC}(\bar g)$ lies above 0.5 |
| Inverted comparator | A comparator whose AUROC CI lies wholly below 0.5 is flagged "inverted" in the report and in the text |
| N-sweep | $\bar g^{(n)}$ from the first $n\in\{3,5,10\}$ of the 20 samples; share above chance $(A_n-0.5)/(A_{20}-0.5)$ with a CI from the same bootstrap draws. Reported only in cells where $\bar g$ is above chance, and "n/a" elsewhere, because the ratio is unstable near chance. |
| Degeneration | Per domain and decoder: the share of answers with $L_r<64$, the median $L_r$, the count with $L_r<8$, and the mean repeat share of the 64-token answer |

Wording rules for b6 and m3. They apply whatever the numbers are.

| The text may say | Only if |
|---|---|
| "$\bar g$ beats G-NLL on <domain> for <method>" | the F1 or F2 cell is `g_beats` |
| "... but not the summed G-NLL" must be added | $\Delta$ against the summed G-NLL is below +0.02 in that cell |
| "$\bar g$ beats the one-pass scores we tested on <domain>" | both the F1/F2 cell and the F3 cell are `g_beats` |
| Generated answers in the title or abstract (PG6) | Stage A is in v1. A positive claim there also needs the overall verdict SUPPORTED. |
| "the merged posterior mean decodes at C0's speed" | S6-T5(c): the ratio lies in [0.90, 1.10] |

Framing, written in advance. S6-T4 fills the numbers.

- N1 (NOT SUPPORTED): "At 76M parameters, weight-sampling scores on the models' own greedy answers do not beat one-pass scores. The mean realized-token Jensen gap over the answer does not exceed the AUROC of the answer's greedy NLL (G-NLL) by 0.02 in any of the <m> pre-registered cells (Holm, α = 0.05)."
- N2 (per `comparator_beats` cell): "On <domain>, G-NLL from one mean-weight pass ranks OOD answers above ID answers better than $\bar g$ from N = 20 weight samples, for <method>: Δ = <Δ> [<lo>, <hi>]."
- N3 (F3): "Scoring the prompt alone with one pass (prompt NLL) does as well as or better than $\bar g$ on the generated answer in <k> of <m> cells."
- P1 (POSITIVE row): "For <method>, $\bar g$ on the model's own answer beats G-NLL on <domains>: Δ = <Δ> [<lo>, <hi>], Holm p = <p>."
- Always: "These are OOD-detection results on continuations of held-out documents, not error detection. The answers have no correctness labels. <x>% of ID and <y>% of OOD greedy answers repeat a 4-gram within 64 tokens; the median answer is <a> (ID) and <b> (OOD) tokens long."

This endpoint does not decide which posterior method is best, and it does not test the scores on reference text. S1 and S3 own those.

## 4. Acceptance tests

A test passes only if every check under it passes.

| ID | Given | When | Then | Runs in |
|---|---|---|---|---|
| S6-T1 Decode and score | A 2-layer fixture: `GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.1, bias=True)`, `torch.manual_seed(0)`. A BLoB LoRA (rank 4, alpha 8, prior 0.2, init_g 0.1). Its `lora_B` is set to 0.5 × a Gaussian (generator seed 1), and its `lora_A_g` is multiplied by 3, so the samples differ. 6 prompts of $P = 12$ tokens (generator seed 1), answers of $L = 8$ tokens, $N = 5$. A C0-type fixture with no stochastic layer ($N = 1$). Synthetic answers for the stop rule. | `decode_greedy` decodes with `use_mean_weights`. `rescore` scores prompt + answer in one stacked pass. | (a) Every answer equals a reference greedy loop written in the test (argmax of `model(idx)[0][:, -1]` under `use_mean_weights`), token for token. The batched decode (batch 6) equals batch-1 decoding (yes/no). (b) Greedy identity: $\log p_{\bar\theta}(y_j)$ equals $\max_v \log p_{\bar\theta}(v)$ at the same position to $10^{-6}$. Alignment: the log-probs recorded at decode step $j$ equal the teacher-forced decoder log-probs at 0-based position $P+j-2$ to $10^{-5}$. Probe: 0.0 and $4.8\times10^{-7}$. (c) $g_j$, $\mathrm{MI}_j$ and $\mathrm{EU}_j$ are each at least $-10^{-6}$ on every kept token (probe minima: $1.2\times10^{-3}$, $3.2\times10^{-3}$, $5.4\times10^{-3}$). (d) With $N$ identical samples, and for the C0 fixture, $\lvert g_j\rvert$, $\lvert\mathrm{MI}_j\rvert$ and $\lvert\mathrm{EU}_j\rvert$ are at most $10^{-6}$ (probe: 0.0 and $4.8\times10^{-7}$). (e) $\log\bar p(y_j)$ from `logp_real` equals $\log\bar p_j[y_j]$ from the full-vocabulary path to $10^{-4}$. (f) `answer_length` with $n=4$ and max 12: [5,6,7,8,9,5,6,7,8,1,2,3] gives 5; [3]×12 gives 1; [1,2,3,4,1,2,3,5,1,2,3,4] gives 8; 1-12 in order gives 12 (yes/no each). (g) Replacing the answer tokens after $L_r$ with random tokens leaves every `resp_*` value unchanged to $10^{-6}$. Response 0 scored alone equals response 0 scored in a batch of 3 (same seed) to $10^{-6}$ (yes/no each). | `tests/test_stage_a.py`, tests `test_s6_t1_*` (CPU, CI) |
| S6-T2 Stacked pass vs per-sample loop | The T1 fixture, plus: a VI fixture (`bayes_ffn` enabled, init_rho −2); a deterministic-LoRA fixture with a v2 TFB state and a v2 Laplace-LoRA state; the C0 fixture with a v2 Laplace FFN state (S2's `scale_laplace_state` on a synthetic curvature); MC dropout on the C0 fixture. The MC dropout seed base 3,000,000 is a constant in the test module. | `rescore` (stacked) runs with `sample_chunk` 1, 2 and 5. `rescore_loop` is the reference: one forward call per sample through the existing `frozen_bayesian_sample`, or through `sample_tfb_params` / `sample_laplace_params` and `apply_sampled_params`. | (a) VI, BLoB, TFB v2, Laplace v2 LoRA and Laplace v2 FFN: every returned tensor (`logp_real`, `tok_mi`, `tok_tu`, `tok_eu`, every `resp_*`) equals the loop to $10^{-5}$ (probe: 0.0 for VI and BLoB). (b) The three chunk sizes give `torch.equal` values of `logp_real` (yes/no). (c) MC dropout: two stacked runs with the same seeds are `torch.equal`; seed base 3,000,000 changes `resp_g_mean` of at least one response; every $g_j \ge -10^{-6}$ (yes/no each). (d) Two calls of `rescore` with the same inputs give `torch.equal` outputs for every sampler kind (yes/no). (e) The whole module runs in 60 s or less on CPU, as pytest's summary line reports (number). | `tests/test_stage_a.py`, tests `test_s6_t2_*` (CPU, CI) |
| S6-T3 Stage A run | S1's manifest, stored document tokens, block files and `main` score files; S1's `prereg.json`; S2's three refit states and fit records; `configs/i2_stage_a.yaml` with its frozen blocks and `prereg_stage_a.json` | `python scripts/stage_a.py --config configs/i2_stage_a.yaml --part all --run-tag main` (Section 5.1 order). A fresh process runs `--part all --run-tag rerun --response-ids 0-63,2000-2063` (whole batches, Section 5.3). Then two calls: `python scripts/check_eval_rebuild.py --config configs/i2_stage_a.yaml --check stage-a`, and the same with `--rederive-stage-a`. | (a) The prompt file holds exactly 1,000 prompts per domain, each from a distinct S1 `test` document of the raw variant, in manifest order. Each prompt equals the stored tokens $t_{o_d:o_d+128}$ and x[0:128] of S1's block 0. Each reference equals $t_{o_d+128:o_d+192}$ and y[127:191] of that block (yes/no each). (b) Each decoder writes one decoder file. Every answer has 64 decoded tokens and $1 \le L_r \le 64$. The three LoRA score sets read the same decoder file (equal `answer_sha256`). The fit records of `c4_tfb_fixed` and `c4_lap_refit` name a base whose sha256 equals that of `data/checkpoints/c3/ckpt_best.pt`, and `c2_refit`'s equals that of `data/checkpoints/c0/ckpt_best.pt`. The greedy identity holds on at least 99.5% of kept tokens under fp16 (`rescore.greedy_agree_min`); the share is recorded (yes/no each). (c) Every score file has every key, shape and dtype of Section 5.6 and S1's meta keys. `meta.analysis_sha256` equals `prereg_stage_a.json`, `meta.s1_analysis_sha256` equals S1's `prereg.json`, and `frozen_at` is earlier than every `created_at` (yes/no each). (d) The `rerun` files equal the matching rows of the `main` files, answers included, with `torch.equal` on every tensor. Their meta has the same `decode_batch_size`, `rescore_batch_size` and `sample_chunk` (yes/no each). (e) `table_stage_a.json` holds AUROC and FPR@95 with document-bootstrap CIs for every score in `analysis.report_scores`, in every cell of F1-F3 and every descriptive row, or a named drop reason (yes/no per cell). (f) `--rederive-stage-a` rebuilds $\bar g$ from `logp_real` and $L_r$, $\bar g_{\mathrm{ref}}$ from S1's `logp_real`, and the sum, mean and max of MI and EU from `tok_mi`, `tok_eu` and `mask`. These match every stored `resp_*` value. It then rebuilds every AUROC, CI, $\Delta$, p, Holm-adjusted p, cell verdict, row verdict and the overall verdict. All match the tables and `stage_a_verdicts.json` to $10^{-6}$ (yes/no per value). `tests/test_check_eval_rebuild.py::test_checker_imports_no_scorer_code` still passes (yes/no). | (a)-(e): `check_eval_rebuild.py --check stage-a`, report `stage_a_check.json`, exit 0 = PASS. (f): `check_eval_rebuild.py --rederive-stage-a`, report `rederive_stage_a_check.json`, exit 0 = PASS. Decoding and rescoring need the GPU; both checks run on CPU. |
| S6-T4 Pre-registered report | The frozen blocks; synthetic families in the test module; fixture Stage A files; the real tables | `scripts/stage_a.py --part report` writes `stage_a_report.md` and `stage_a_verdicts.json`. The fixture checker call in (f). | (a) The cell verdict function returns: (Δ = +0.03, Holm p = 0.01) → `g_beats`; (−0.03, 0.01) → `comparator_beats`; (+0.03, 0.20) → `no_difference`; (+0.015, 0.001) → `no_difference`. The row verdict returns POSITIVE for [g_beats, g_beats, no_difference] and for [g_beats, g_beats]; MIXED for [g_beats, no_difference, no_difference], for [g_beats, g_beats, comparator_beats] and for [g_beats, no_difference]; NEGATIVE for [no_difference, comparator_beats, no_difference]. The overall verdict returns SUPPORTED for [POSITIVE, NEGATIVE, NEGATIVE], NOT SUPPORTED for [NEGATIVE] × 6 and MIXED for [MIXED, NEGATIVE] (yes/no each). (b) When every row is NEGATIVE, the report contains sentence N1 verbatim, with the numbers filled in (yes/no). (c) The report quotes the margin 0.02, α = 0.05, the primary score `resp_g_mean`, the comparator `resp_gnll_mean`, the families and their Holm $m$, all read from the frozen YAML. If the frozen-block hash differs from `prereg_stage_a.json`, the part exits non-zero and writes nothing. `configs/i2_stage_a.yaml` holds the Section 3 values: $P = 128$, 64 answer and reference tokens, `stop_ngram` 4, $N = 20$, seed offset 400,000, bootstrap seed 6 with 10,000 resamples, margin 0.02, α 0.05, primary `resp_g_mean`, comparators `resp_gnll_mean` and `resp_prompt_nll`, and families with 3, 3 and 6 rows (yes/no each). (d) A fixture row marked "not scored" (S2 `not_matched`) is listed as such, and $m$ for its family drops from 9 to 6 (yes/no). (e) Real data: `stage_a_verdicts.json` holds Δ, CI, raw p, Holm p and verdict for every family cell, each row verdict and the overall verdict (yes/no per cell, checked by `--check stage-a`). The report's numbers equal that file, and the file equals the re-derived values to $10^{-6}$ (`--rederive-stage-a`). (f) Fixture checker: the module writes Stage A files from the T1 fixture into `tmp_path` (one row, one ID and one OOD domain, 6 responses each, plus S1-format files for $\bar g_{\mathrm{ref}}$), with a fixture Stage A YAML and S1 YAML. `check_eval_rebuild.py --config <fixture YAML> --rederive-stage-a` exits 0. After one `logp_real` value in the score file is raised by $10^{-3}$, it exits 1 (yes/no each). | `tests/test_stage_a.py`, tests `test_s6_t4_*`, for (a)-(d) and (f) (CPU, CI); `scripts/stage_a.py --part report` and the two checker calls for (e) |
| S6-T5 Latency | The merged mean-weight LoRA decoder (C3's BLoB mean), C0 and the C1 means. The first 64 StackExchange prompts of the prompt file and their uncut 64-token answers. The 6 score sets. | `scripts/stage_a.py --part latency`, before `--part rescore` | (a) Merge check: merged and unmerged mean-weight logits differ by at most $10^{-4}$ in fp32, on the T1 fixture in pytest (probe: $3.7\times10^{-7}$) and on 4 real prompts of 192 tokens in the script (probe on real C3, CPU: $5.0\times10^{-5}$, argmax agreement 1.0) (yes/no). (b) `stage_a_latency.json` holds, per decoder, the ms per 64-token greedy decode at batch 1 (merged and unmerged for the LoRA decoder). Per score set and $N\in\{3,10\}$, it holds the ms per response of the stacked rescoring pass at batch 1 and at batch 8, one teacher-forced mean-weight pass over 192 tokens, peak reserved MiB, and S0's header fields. It also holds `rescore_peak_mib`, the peak reserved MiB of the stacked pass at the configured `batch_size` and `sample_chunk` with $N = 20$, which is at most `vram_budget_mib`; otherwise `--part rescore` refuses to start (yes/no per field). (c) The ratio of merged to C0 decode time is recorded, with `zero_overhead` = (ratio in [0.90, 1.10]) (yes/no). (d) The ratios of rescoring to decoding at $N = 3$ and $N = 10$ are recorded, and the JSON states `kv_cache: false` (yes/no). | `tests/test_stage_a.py::test_s6_t5_merge_fixture` for the fixture part of (a); `scripts/stage_a.py --part latency` for the real part of (a); `check_eval_rebuild.py --check stage-a` checks the fields of (b)-(d) against the Section 5.8 list |

G1(c) is the command in Section 1: `uv run pytest tests/test_stage_a.py -k "s6_t1 or s6_t2"` passes with no test skipped. No test in this module reads `data/`, so CI runs all of them.

## 5. Implementation notes for the builder

### 5.1 Order of work

| Step | Work | Test | Machine | When |
|---|---|---|---|---|
| 1 | `answer_length`, `decode_greedy`, `merge_lora_mean`, per-token scores on the fixture | S6-T1 | CPU | session 04, Mon Oct 5 |
| 2 | Stacked pass for 5 sampler kinds, MC dropout, `rescore_loop` | S6-T2 | CPU | session 04 |
| 3 | Verdicts, report, freeze check, the frozen-values check | S6-T4(a)-(d) | CPU | session 04 |
| 4 | Checker modes, by a separate agent from Sections 3, 5.2, 5.3, 5.6, 5.7 and 5.8 only | S6-T4(f) | CPU | session 04 |
| 5 | `--part freeze`, after steps 1-4 pass. Record the hash in the session log and STATE.md, because `data/` is gitignored and Q3 may block commits (as S1 does). | S6-T4(c) | CPU | end of session 04 |
| 6 | G1: Alexey logs the hours; the G1(c) pytest command | G1(b), G1(c) | none | Sun Oct 11 |
| 7 | Alexey reviews the module (choice C3) | none | none | Tue Oct 13 |
| 8 | `--part prompts`, `decode`, `latency`, `rescore`, then the `rerun` process | S6-T3(a)-(d), S6-T5 | GPU, 1.7-5.0 h | Tue Oct 13, night |
| 9 | `--part analyze`, `report`; then `check_eval_rebuild.py --check stage-a` and `check_eval_rebuild.py --rederive-stage-a`, as two calls | S6-T3(e)-(f), S6-T4(e) | CPU | Wed Oct 14 |
| 10 | Outside S6: Alexey reviews the results (A5); b6 drafts the subsection | none | none | by Tue Oct 20 (cutoff) |

### 5.2 Formulas

Setup. The prompt is $x = x_{1:P}$ with $P = 128$. The decoder weights are $\bar\theta$. The weight samples are $\theta_1,\dots,\theta_N$. The context of answer token $j$ is $h_j = (x, y_{<j})$.

Greedy decoding:

$$y_j=\arg\max_{v}\,p_{\bar\theta}(v\mid h_j),\qquad j=1,\dots,64 .$$

Stop rule. With $y_{a:b}=(y_a,\dots,y_b)$:

$$j^\star=\min\big\{\,j\in\{5,\dots,64\} : \exists\, i\in\{4,\dots,j-1\},\ y_{i-3:i}=y_{j-3:j}\,\big\},\qquad L_r=\begin{cases}j^\star-4 & \text{if } j^\star \text{ exists}\\ 64 & \text{otherwise.}\end{cases}$$

The kept answer $y_{1:L_r}$ holds no repeated 4-gram and no start of one.

Per-token scores, for $j \le L_r$. Let $p_{s,j}=p_{\theta_s}(\cdot\mid h_j)$, $\bar p_j=\frac1N\sum_s p_{s,j}$, $q_j=p_{\bar\theta}(\cdot\mid h_j)$, and $\ell_{s,j}=\log p_{s,j}(y_j)$:

$$\log\bar p(y_j)=\operatorname{logsumexp}_{s=1}^{N}\ell_{s,j}-\log N,\qquad g_j=\log\bar p(y_j)-\frac1N\sum_{s=1}^{N}\ell_{s,j}$$

$$\mathrm{MI}_j=H[\bar p_j]-\frac1N\sum_{s=1}^{N}H[p_{s,j}],\qquad \mathrm{TU}_j=H[\bar p_j],\qquad \mathrm{AU}_j=\mathrm{TU}_j-\mathrm{MI}_j$$

$$\mathrm{EU}_j=\frac1N\sum_{s=1}^{N}\mathrm{KL}\big(q_j\,\big\|\,p_{s,j}\big)=-H[q_j]-\frac1N\sum_{s=1}^{N}\sum_{v}q_j(v)\log p_{s,j}(v)$$

EU is the decoder-matched score of metrics-assessment.md Section 4.2, which cites Schweighofer et al. 2024 (arXiv 2410.10786). The options document calls it "EU(A3)". That label was not re-checked in the paper; the formula above is the definition.

Per-response aggregation of a per-token score $S_j$:

$$\bar S(r)=\frac1{L_r}\sum_{j=1}^{L_r}S_j,\qquad S^{\Sigma}(r)=\sum_{j=1}^{L_r}S_j,\qquad S^{\max}(r)=\max_{j\le L_r}S_j .$$

One-pass scores, from the decoder pass only:

$$\mathrm{GNLL}(r)=-\frac1{L_r}\sum_{j=1}^{L_r}\log q_j(y_j),\qquad \mathrm{GNLL}^{\Sigma}(r)=L_r\,\mathrm{GNLL}(r),\qquad \mathrm{MSP}(r)=\frac1{L_r}\sum_{j=1}^{L_r}\Big(1-\max_v q_j(v)\Big)$$

$$\mathrm{ENT}(r)=\frac1{L_r}\sum_{j=1}^{L_r}H[q_j],\qquad \mathrm{PNLL}(r)=-\frac{1}{P-1}\sum_{i=2}^{P}\log p_{\bar\theta}(x_i\mid x_{<i}),\qquad \rho_4(r)=1-\frac{\lvert\{y_{i-3:i}\}_{i=4}^{64}\rvert}{61}$$

Under greedy decoding, $\log q_j(y_j)=\max_v\log q_j(v)$, so G-NLL is the mean negative log max-prob of the answer. MSP and G-NLL then differ only in how they transform each token. The answer length $L_r$ is itself a one-pass score.

Reference continuation. S1 block 0 holds $x^{\mathrm{blk}}=t_{o_d:o_d+256}$, and its 0-based position $i$ predicts $t_{o_d+i+1}$. So reference token $j$ ($t_{o_d+P+j-1}$) sits at position $P+j-2$. With $g^{\mathrm{S1}}_i$ from S1's `logp_real` through `realized_token_gap`:

$$\bar g_{\mathrm{ref}}(r)=\frac1{L_r}\sum_{j=1}^{L_r}g^{\mathrm{S1}}_{P+j-2} .$$

It uses the same $L_r$ as the model's own answer, so the lengths match. Its draws are S1's, not Stage A's.

Contrasts and the N-sweep:

$$\Delta_G=A(\bar g)-A(\mathrm{GNLL}),\qquad \Delta_P=A(\bar g)-A(\mathrm{PNLL}),\qquad \Delta_{\mathrm{ref}}=A(\bar g)-A(\bar g_{\mathrm{ref}}),\qquad \mathrm{share}_n=\frac{A(\bar g^{(n)})-0.5}{A(\bar g)-0.5}$$

Here $A$ is AUROC with ID responses as label 0 and OOD responses as label 1, and $\bar g^{(n)}$ uses the first $n$ samples. Every response has weight 1, so S1's weighted AUROC equals the plain AUROC.

### 5.3 Prompt set

- Read S1's manifest (`load_manifest` on the YAML named by `stage_a.eval_config`). Take the rows with `split: test` and `variant: raw` for each domain in `stage_a.prompt_domains` order, in manifest order. Take the first `n_prompt_docs` per domain. Assert that `stage_a.manifest_path` and `stage_a.doc_tokens_dir` agree with S1's `eval_set.out_dir`.
- Read $o_d$ and $L_d$ from the manifest row. Read the tokens from `docs_{domain}.pt` (`doc_file_path(cfg, key, "raw")`), a list of int32 tensors in manifest order. Assert that the doc IDs line up, that each tensor has $L_d$ tokens, and that $o_d+192\le L_d$ (it holds, because S1 blocks need $o_d+257\le L_d$).
- Response index $r$ runs over the file in domain-major order: StackExchange 0-999, HackerNews 1000-1999, arXiv 2000-2999, FreeLaw 3000-3999, PubMed 4000-4999.
- Batches. The decode and the rescore parts both form batches per domain: from the domain's first response, `batch_size` consecutive responses each. The last batch of a domain may be shorter. `--response-ids` must select whole batches for both `rescore.decode_batch_size` and `rescore.batch_size`, or the script exits 2. A re-run then sees the same batch shapes and seeds as the main run, which `torch.equal` in S6-T3(d) needs on the GPU. With batch sizes 32 and 8, `0-63,2000-2063` is whole; `0-99` is not.
- Write `data/stage_a/prompts.pt` (Section 5.6).

### 5.4 Decoders and score sets

Load every score set as S1's scorer does, so Stage A scores exactly the models S1 and S2 scored. Import `scripts/eval_c_checkpoints.py` with `importlib`, and call `SCORE_SET_REGISTRY[s].load(device)`. It returns `(model, method, state)`. The files in `SCORE_SET_REGISTRY[s].checkpoint_paths()` go into `checkpoint_sha256`, as in S1's meta. Load `mc_dropout` with `load_c0_model(device)` from `scripts/eval_mc_dropout.py`, method `dropout`, as that script does. The returned `method` must equal `score_sets.<s>.method` in the YAML, or the part exits 2.

A score set whose loader raises ValueError is dropped, with the error text as its reason. That covers an S2 fit that did not pass, a `match_status` that is not scored, and a hash mismatch; a `not_matched` fit has no state file at all. The reason goes into the run's meta and the report.

| Score set | Loader (method) | Decoder | Stochastic tensors per sample | Draw rule (loop and stacked pass alike) | ID prompts (secondary) |
|---|---|---|---|---|---|
| `mc_dropout` | `load_c0_model` (`dropout`) | C0, eval mode | dropout masks, per element | `torch.manual_seed(seed_b)` once per batch, then train mode (`enable_dropout`, minigpt/layers.py:230-244) | StackExchange |
| `c2_refit` | `load_posthoc_refit` (`laplace_v2`): C0 weights and the v2 Laplace FFN state (32 weight tensors `blocks.{i}.mlp.{fc,proj}.linear.weight`, no biases) | C0 | $W_s$ per FFN weight | `PARAM_SAMPLERS["laplace_v2"](state, seed=seed_b·N + s)`, drawn on the state's device | StackExchange |
| `c1` | `load_model("c1")` (`variational`): `BayesianLinear` FFN | C1 means $\mu$ (`use_mean_weights`, minigpt/layers.py:213-227) | $(W_s, b_s)$ per FFN layer | `torch.manual_seed(seed_b)`; per sample, `freeze_sample()` on each layer in `modules()` order (minigpt/layers.py:103-109) | StackExchange |
| `c3` | `load_model("c3")` (`variational`): BLoB LoRA | merged BLoB mean | $A_s$ per LoRA layer | as `c1` (minigpt/lora.py:87-90) | HackerNews (StackExchange) |
| `c4_tfb_fixed` | `load_posthoc_refit` (`tfb_v2`): `blob_mean` base and the v2 TFB state | merged BLoB mean | $A_s$ = `lora_A` per layer | `PARAM_SAMPLERS["tfb_v2"](state, seed=seed_b·N + s)`; CPU generator, noise moved to the device | HackerNews (StackExchange) |
| `c4_lap_refit` | `load_posthoc_refit` (`laplace_v2`): `blob_mean` base and the v2 Laplace-LoRA state | merged BLoB mean | $A_s$ = `lora_A` per layer | `PARAM_SAMPLERS["laplace_v2"](state, seed=seed_b·N + s)`, drawn on the state's device | HackerNews (StackExchange) |

- The decoders load through the same registry: `c0`, `c1` (then `use_mean_weights`), and `merge_lora_mean` on the `c3` model.
- `merge_lora_mean` builds a C0-shaped `MiniGPT`, loads the base weights of the C3 checkpoint (keys `.base_linear.` mapped to `.linear.`, strict load, LoRA keys excluded), and adds $\frac{\alpha}{r}BA_{\mu}$ to each FFN weight. The drafting probe built it this way and matched the unmerged mean path (Section 6.2).
- Shared-decoder asserts. (i) The `lora_A` tensors of the S2 `blob_mean` base equal C3's `lora_A_mu` (`torch.equal`). (ii) The fit records' `base.sha256` equals `file_sha256` of `data/checkpoints/c3/ckpt_best.pt` for the two C4 rows, and of `data/checkpoints/c0/ckpt_best.pt` for `c2_refit`. A row that fails an assert is dropped with that reason. So the three LoRA rows share one decoder, and `c2_refit` shares C0's.
- Decoder pass. After decoding, the decode part runs one teacher-forced pass `model(z)` over $z=(x, y_{1:64})$ minus the last token (191 tokens), at the decode batch size. It gives the prompt NLL and the 64 answer values of `logp_dec`, `tok_maxlogp_dec` and `tok_ent_dec`. It also gives `greedy_agree`: the share of kept tokens whose teacher-forced argmax equals $y_j$.
- EU needs the full $q_j$, which the decoder file cannot hold (about 80 GB for 13,000 answers in fp16). So the rescore part runs the same decoder pass again for each rescore batch. It records in meta the largest absolute difference between that pass's $\log q_j(y_j)$ and the decoder file's `logp_dec` on kept tokens (descriptive).

### 5.5 The stacked rescoring pass

- The batch axis holds `sample_chunk` × B rows in sample-major order (row $= s\cdot B + b$). Embeddings, attention, LayerNorm and the LM head run unchanged on all rows.
- A context manager in `minigpt/stage_a.py` swaps each stochastic FFN module (`block.mlp.fc`, `block.mlp.proj`) for a stacked wrapper during the pass and restores the original on exit. This is the same module replacement `inject_lora` uses (minigpt/lora.py:153-207). No model class is edited.

| Kind (`score_sets.<s>.stacked`) | Stacked wrapper |
|---|---|
| `full_ffn` (`c1`, `c2_refit`) | $y=\mathrm{bmm}(x,\,W_s^{\top})+b_s$ ($b_s = b$ for `c2_refit`, whose biases are not sampled) |
| `lora` (`c3`, `c4_tfb_fixed`, `c4_lap_refit`) | $y=W_0x+b_0+\frac{\alpha}{r}\,\mathrm{bmm}(x,\,A_s^{\top})\,B^{\top}$ |
| `replicate` (MC dropout) | none: the $B$ responses are repeated `sample_chunk` times and run in train mode in one call |

- Do not use `torch.vmap` over the model. In the drafting probe, vmap over `functional_call` gave the right numbers, but PyTorch warned that SDPA flash attention has no batching rule on CPU and falls back to a per-sample loop. That would remove the speed-up the stacked pass exists for.
- Seeds. Let $r_b$ be the first response index of rescore batch $b$ (batches per domain, Section 5.3). Then $\text{seed}_b=\text{seed\_base}+\text{seed\_offset}+r_b$, with `stage_a.seed_base` 0 and `stage_a.seed_offset` 400,000. For TFB and Laplace, the seed of sample $s$ is $\text{seed}_b\cdot N+s$, as in S1's scorer (scripts/eval_c_checkpoints.py:661-668). For VI, BLoB and MC dropout, `torch.manual_seed(seed_b)` runs once per batch, and the draws follow in sample order. These seeds do not collide with S1's (legacy 0-999; test from 100,000; test_hn from 200,000; stripped from 300,000; spread from 1,000,000) or with S2's (0-39). The MC dropout check in S6-T2(c) uses seed base 3,000,000, a test constant.
- Memory. For the sample rows, call `model.forward_body(z)` (minigpt/model.py:149-161), then apply `model.lm_head` to the 64 answer positions only (0-based $P-1,\dots,P+62$). Run under fp16 autocast, and take `log_softmax` in fp32. With `sample_chunk` 10 and $B = 8$ (80 rows), the answer log-probs take about 1.0 GB in fp32, and the peak is about 3 GB (est.: fp16 logits, fp32 log-probs and probabilities, the running sums). S0 has no cell for this pass: its cells run one forward per sample on at most 32 rows of 256 tokens. So `--part latency` measures `rescore_peak_mib` at the configured sizes with $N = 20$ (S6-T5(b)), and `--part rescore` refuses to start if it exceeds `vram_budget_mib` (9,728 MiB, S0's budget). Record both sizes in meta.
- Streaming. Keep running sums over samples of $p_{s,j}$, $H[p_{s,j}]$ and $\sum_v q_j(v)\log p_{s,j}(v)$ in fp32. Keep every $\ell_{s,j}$ in fp32 (S1 Section 5.5 explains why fp16 is too coarse). Compute $g$ with `realized_token_gap` and aggregate its `g` over the mask; do not use its `G`, which averages all 64 positions.
- Positions after $L_r$ are computed but masked out of every `resp_*` value except `resp_g64_mean`, which uses all 64 positions.
- Determinism. Apply `rescore.determinism` with S1's `apply_determinism` before any model loads, and record every warning in meta (`WarningLog`).

### 5.6 Files

Meta keys. Stage A files carry S1's meta keys (scripts/eval_c_checkpoints.py:1094-1119): `git_sha`, `git_dirty`, `code_sha256`, `checkpoint_sha256`, `score_set`, `display_label`, `sampler`, `adapter_source`, `eval_set`, `run_tag`, `block_ids`, `n_samples`, `seed_base`, `seed_offset`, `batch_size`, `amp_fp16`, `device`, `torch_version`, `cuda_version`, `determinism_warnings`, `manifest_sha256`, `yaml_sha256`, `analysis_sha256`, `created_at`. In Stage A files, `eval_set` is `stage_a`; `block_ids` holds the `--response-ids` string or null; `batch_size` is the batch size of the part that wrote the file; `yaml_sha256` hashes the bytes of `configs/i2_stage_a.yaml`; `analysis_sha256` is the Stage A freeze hash (Section 5.7), not S1's. The labels come from S1's `scoring.labels`; in a decoder file, `score_set` holds the decoder name. Every file also has `s1_analysis_sha256`, copied from S1's `prereg.json`.

`data/stage_a/prompts.pt`:

| Key | Shape, type |
|---|---|
| `resp_index`, `offset` ($o_d$) | [R] int64 |
| `doc_id`, `domain` | lists of str, length R |
| `prompt` | [R, 128] int32 |
| `reference` | [R, 64] int32 |
| `meta` | `manifest_sha256`, `yaml_sha256`, `analysis_sha256`, `s1_analysis_sha256`, `created_at` |

`data/scores_i2/stage_a/decoder__{decoder}__{run_tag}.pt`:

| Key | Shape, type | Note |
|---|---|---|
| `resp_index` | [R'] int64 | the responses this decoder serves |
| `doc_id`, `domain` | lists of str, length R' | for the join checks |
| `answer` | [R', 64] int32 | all 64 decoded tokens |
| `ans_len` | [R'] int64 | $L_r$ |
| `ans_rep4` | [R'] float32 | $\rho_4$ on the 64 tokens |
| `logp_dec`, `tok_maxlogp_dec`, `tok_ent_dec` | [R', 64] float32 | $\log q_j(y_j)$, $\max_v\log q_j(v)$, $H[q_j]$, from the teacher-forced decoder pass |
| `resp_gnll_mean`, `resp_gnll_sum`, `resp_msp`, `resp_ent`, `resp_prompt_nll` | [R'] float32 | on the kept tokens; the prompt NLL on the 127 prompt predictions |
| `greedy_agree` | [R'] float32 | Section 5.4 |
| `meta` | dict | the meta keys above, plus `decoder`, `answer_sha256` (sha256 of `answer` as int32 little-endian bytes), `decode_batch_size`, `stop_ngram`, `greedy_agree_share` |

`data/scores_i2/stage_a/{score_set}__stage_a__{run_tag}.pt`:

| Key | Shape, type | Note |
|---|---|---|
| `resp_index` | [R'] int64 | |
| `doc_id`, `domain` | lists of str, length R' | |
| `weight` | [R'] float32 | all 1 |
| `logp_real` | [R', N, 64] float32 | $\ell_{s,j}$ |
| `tok_mi`, `tok_tu`, `tok_au`, `tok_eu` | [R', 64] float32 | |
| `mask` | [R', 64] bool | $j \le L_r$ |
| `resp_g_{mean,sum,max}`, `resp_mi_{mean,sum,max}`, `resp_eu_{mean,sum,max}`, `resp_tu_mean`, `resp_g64_mean` | [R'] float32 | |
| `meta` | dict | the meta keys above, plus `decoder`, `decoder_file_sha256`, `n_samples`, `sample_chunk`, `rescore_batch_size`, `decode_batch_size`, `logp_dec_max_abs_diff` |

`data/scores_i2/stage_a/gref__{score_set}__main.pt`, written by `--part analyze`:

| Key | Shape, type | Note |
|---|---|---|
| `resp_index` | [R'] int64 | |
| `doc_id`, `s1_eval_set` | lists of str, length R' | `s1_eval_set` is `test` or `test_hn` |
| `s1_block_index` | [R'] int64 | the row of S1's block 0 |
| `resp_gref_mean` | [R'] float32 | $\bar g_{\mathrm{ref}}$ |
| `meta` | dict | `file_sha256` of each S1 score file read, `analysis_sha256`, `created_at` |

A score set without S1 `main` files has no $\bar g_{\mathrm{ref}}$ file. Its `resp_gref_mean` cells and $\Delta_{\mathrm{ref}}$ rows are dropped with that reason.

Analysis outputs in `data/scores_i2/stage_a/`: `table_stage_a.json` (AUROC and FPR@95 with CIs per score and cell), `families_stage_a.json` ($\Delta$, CI, p and Holm p per family cell), `nsweep_stage_a.json`, `degeneration_stage_a.json`, `stage_a_verdicts.json`, `stage_a_report.md`, `stage_a_latency.json`, `prereg_stage_a.json`, and the checker reports `stage_a_check.json` and `rederive_stage_a_check.json`. The checker writes its own rebuilt tables under `rederive_stage_a/`.

### 5.7 Analysis and report

- A cell is (score set, ID domain, OOD domain). Its responses are those of the two domains served by the score set's decoder. The scores come from the score file, the decoder file and, for $\bar g_{\mathrm{ref}}$, the gref file. Join by `resp_index` and check `doc_id` equality.
- Call `doc_bootstrap_auroc(scores, labels, doc_ids, weights, 10000, [6, i_id, i_ood], 0.95, target_tpr=0.95)` and `paired_doc_bootstrap(..., [(primary, comparator)], 10000, [6, i_id, i_ood], 0.95)` with `weights` all 1. $i$ is the position in S1's `eval_set.domains` (stackexchange 0, hackernews 1, arxiv 2, freelaw 3, pubmed_abstracts 4), read from `stage_a.eval_config`. Call `holm` within each family.
- $\bar g_{\mathrm{ref}}$. For each response, find S1's block 0 of its document: in `{s}__test__main.pt` for StackExchange and the OOD domains, in `{s}__test_hn__main.pt` for HackerNews. It is the row with the same `doc_id` and `offset` equal to $o_d$. Take `logp_real[b, :, 127:127+L_r]`, apply `realized_token_gap`, and average `g`.
- The freeze hash is the sha256 of `json.dumps({"stage_a": ..., "decode": ..., "score_sets": ..., "analysis": ...}, sort_keys=True, separators=(",", ":"))`. This is S1's rule (minigpt/evalset.py:198-234) over four blocks, in a function of `minigpt/stage_a.py`. `--part freeze` writes `prereg_stage_a.json` with `analysis_sha256`, `frozen_at` and `config`. It opens the file in exclusive mode and refuses to overwrite an existing one.
- The report fills the Section 3 sentences from `stage_a_verdicts.json`. It lists dropped rows with their reason. It shows the degeneration table next to the results.

### 5.8 Latency

- Reuse S0's timing rules (`scripts/timing_probe.py`, Section 5 there): `torch.cuda.synchronize()` around the timed region, warm-up runs, `reset_peak_memory_stats()` per cell, and `requires_grad_(False)` on all parameters before timing.
- Header. Call S0's `collect_header(cfg, config_path, dry_run=False)` on S0's config (`latency.timing_probe_config`). Then set `schema` to `latency.schema` and add `config_sha256.stage_a` (the sha256 of `configs/i2_stage_a.yaml`). The header then has every field of S0's `HEADER_FIELDS`.
- Decode timing: the 64 StackExchange prompts, 64 steps, batch 1, for C0, the C1 means, the merged LoRA and the unmerged mean path (`use_mean_weights` on C3). 3 warm-up prompts, then the median of 3 repeats.
- Rescoring timing: per score set, $N\in\{3,10\}$, batch 1 and batch 8, on the uncut 64-token answers, including the weight draws and the decoder pass.
- Peak check: one stacked rescoring batch per score set at `rescore.batch_size` and `rescore.sample_chunk` with $N = 20$. `rescore_peak_mib` is the largest peak reserved MiB over the six sets.
- One teacher-forced mean-weight pass over 192 tokens per decoder. This is the compute floor for a decoder with a KV cache. It is not a latency figure, and the text must not present it as one.

$$\text{zero\_overhead}\iff\frac{t_{\mathrm{dec}}(\text{merged})}{t_{\mathrm{dec}}(\mathrm{C0})}\in[0.90,\,1.10],\qquad \text{cost ratio}(N)=\frac{t_{\mathrm{rs}}(N)}{t_{\mathrm{dec}}}$$

Fields of `stage_a_latency.json` (the list `--check stage-a` checks):

| Field | Content |
|---|---|
| `header` | S0's `HEADER_FIELDS`, with `schema` = `i2-stage-a-latency/1` |
| `merge_check` | `max_abs_logit_diff`, `argmax_agree`, `n_prompts`, `tol`, `pass` |
| `decode` | per path (`c0`, `c1_mean`, `blob_mean_merged`, `blob_mean_unmerged`): `ms_per_answer_median`, `repeats_ms` |
| `zero_overhead` | `ratio`, `band`, `value` |
| `rescore` | per score set and $N$: `ms_per_response_b1`, `ms_per_response_b8`, `peak_reserved_mib` |
| `cost_ratio` | per score set and $N$: rescoring ms at batch 1 over the decode ms of its decoder |
| `teacher_forced_192_ms` | per decoder |
| `rescore_peak_mib`, `rescore_peak_config` | the peak check; `batch_size`, `sample_chunk`, `n_samples` |
| `kv_cache` | `false` |

### 5.9 `configs/i2_stage_a.yaml` (sketch; every value explicit)

```yaml
stage_a:                                   # frozen
  name: i2_stage_a_v1
  eval_config: configs/i2_eval.yaml        # S1: manifest, domain order, split rules
  manifest_path: data/eval_i2/manifest.jsonl   # must agree with S1's eval_set.out_dir
  doc_tokens_dir: data/eval_i2                 # must agree with S1's eval_set.out_dir
  s1_scores_dir: data/scores_i2                # S1's prereg.json and main score files
  prompt_domains: [stackexchange, hackernews, arxiv, freelaw, pubmed_abstracts]
  n_prompt_docs: 1000                      # choice C2: 500
  prompt_tokens: 128
  answer_max_tokens: 64
  reference_tokens: 64
  stop_ngram: 4
  n_samples: 20
  seed_base: 0
  seed_offset: 400000
  prompts_path: data/stage_a/prompts.pt
decode:                                    # frozen
  rule: greedy                             # torch.argmax; ties go to the lowest token id
  amp_fp16: true
  decoders:
    c0: {kind: full, load_as: c0}
    c1_mean: {kind: variational_mean, load_as: c1}
    blob_mean_merged: {kind: lora_merged, load_as: c3, a_key: lora_A_mu}
score_sets:                                # frozen; method = the loader's method (Section 5.4)
  mc_dropout: {decoder: c0, method: dropout, stacked: replicate, id: stackexchange,
               id_secondary: []}
  c2_refit: {decoder: c0, method: laplace_v2, stacked: full_ffn, id: stackexchange,
             id_secondary: []}
  c1: {decoder: c1_mean, method: variational, stacked: full_ffn, id: stackexchange,
       id_secondary: []}
  c3: {decoder: blob_mean_merged, method: variational, stacked: lora, id: hackernews,
       id_secondary: [stackexchange]}      # choice C4: [] on the three LoRA rows
  c4_tfb_fixed: {decoder: blob_mean_merged, method: tfb_v2, stacked: lora, id: hackernews,
                 id_secondary: [stackexchange]}
  c4_lap_refit: {decoder: blob_mean_merged, method: laplace_v2, stacked: lora,
                 id: hackernews, id_secondary: [stackexchange]}
rescore:                                   # not frozen: run settings, recorded in meta
  decode_batch_size: 32
  batch_size: 8                            # responses per rescore batch
  sample_chunk: 10                         # samples per forward call (80 rows)
  amp_fp16: true
  vram_budget_mib: 9728                    # S0's budget
  determinism: {use_deterministic_algorithms: true, warn_only: true, cudnn_deterministic: true,
                cudnn_benchmark: false, cublas_workspace_config: ":4096:8"}
  rerun_response_ids: "0-63,2000-2063"     # whole batches; "0-63,1000-1063" under C2
  greedy_agree_min: 0.995
  out_dir: data/scores_i2/stage_a
latency:
  timing_probe_config: configs/i2_timing_probe.yaml   # S0's header helper reads it
  schema: i2-stage-a-latency/1
  prompt_domain: stackexchange
  n_prompts: 64
  n_samples: [3, 10]
  batch_sizes: [1, 8]
  warmup: 3
  repeats: 3
  merge_check_prompts: 4
  merge_tol: 1.0e-4
  zero_overhead_band: [0.90, 1.10]
  out: data/scores_i2/stage_a/stage_a_latency.json
analysis:                                  # frozen
  primary_score: resp_g_mean
  primary_comparator: resp_gnll_mean
  secondary_comparator: resp_prompt_nll
  report_scores: [resp_g_mean, resp_g_sum, resp_g_max, resp_mi_mean, resp_mi_sum, resp_mi_max,
                  resp_eu_mean, resp_eu_sum, resp_eu_max, resp_tu_mean, resp_g64_mean,
                  resp_gref_mean, resp_gnll_mean, resp_gnll_sum, resp_msp, resp_ent,
                  resp_prompt_nll, ans_len, ans_rep4]
  ood_domains: [arxiv, freelaw, pubmed_abstracts]
  fpr_target_tpr: 0.95
  bootstrap: {resamples: 10000, seed: 6, level: 0.95, unit: document, stratify_by_class: true,
              percentile_method: linear}
  margin_auroc: 0.02
  alpha: 0.05
  families:
    S6_F1_primary: [[mc_dropout, stackexchange], [c1, stackexchange], [c3, hackernews]]
    S6_F2_primary_posthoc: [[c2_refit, stackexchange], [c4_lap_refit, hackernews],
                            [c4_tfb_fixed, hackernews]]
    S6_F3_vs_prompt_nll: [[mc_dropout, stackexchange], [c1, stackexchange], [c3, hackernews],
                          [c2_refit, stackexchange], [c4_lap_refit, hackernews],
                          [c4_tfb_fixed, hackernews]]
  descriptive_rows: [[c3, stackexchange], [c4_lap_refit, stackexchange],
                     [c4_tfb_fixed, stackexchange]]   # choice C4: []
  row_positive_min_cells: 2
  n_sweep: [3, 5, 10]
  short_answer_tokens: 8
  rederive_tol: 1.0e-6
```

### 5.10 GPU estimate (CA10 rates, N = 20)

Rates. Decoding: 8.1 ms per step at batch 1 (C0's 256-token forward, report.md:90). The C1 mean path and the merged LoRA have C0's shape. The unmerged BLoB mean path (14.8 ms, report.md:91) is used only in the latency part. Rescoring per response at batch 1: MC dropout 287 ms, `c2_refit` 287 ms, C1 287 ms, C3 316 ms, TFB 469 ms, Laplace-LoRA 469 ms. Each score set adds 8.1 ms for its decoder pass (Section 5.4). The pass is shorter than a 256-token block (192 tokens), but it adds vocabulary work on 64 positions per sample. So these rates are neither a clear upper nor a clear lower bound (est.).

| Part | Count | GPU h at batch 1 | At a 3x batching gain |
|---|---|---|---|
| Decode, 3 decoders | 13,000 answers × 64 steps × 8.1 ms | 1.87 | 0.62 |
| Rescore, C0 decoder (`mc_dropout`, `c2_refit`) | 4,000 × (0.287 + 0.287 + 2 × 0.0081) s | 0.66 | 0.22 |
| Rescore, C1 | 4,000 × 0.295 s | 0.33 | 0.11 |
| Rescore, LoRA decoder (`c3`, `c4_tfb_fixed`, `c4_lap_refit`) | 5,000 × (0.316 + 0.469 + 0.469 + 3 × 0.0081) s | 1.78 | 0.59 |
| Re-run (S6-T3d) | 128 responses per decoder: 384 decodes and 768 rescorings | 0.13 | 0.04 |
| Latency (S6-T5) | 64 prompts: 4 decode paths × 3 repeats; 6 sets × 2 N × 2 batch sizes; the peak check | 0.15-0.20 | 0.15-0.20 |
| **Total** | | **5.0** | **1.7** |

- With 500 prompts per domain (choice C2), the totals are about 2.6 and 1.0 GPU h.
- Dropping the StackExchange prompts of the LoRA decoder (choice C4) saves 0.50 and 0.17 GPU h.
- The built v2 Laplace sampler draws on the state's device, and `load_posthoc_refit` maps the state to the GPU. So `c2_refit` carries no CPU-draw cost (S0 Section 6: 102-149 ms per CPU draw of 33.5M normals, which would add about 2.8 GPU h at batch 1). S6-T5(b) shows `c2_refit`'s real rescoring rate, draws included.
- The night of Tue Oct 13 has no other GPU work in the roadmap (W3: 0 GPU h without Stage A). One night holds 8-10 GPU h (CA3).

## 6. Verification evidence from session 02

### 6.1 Refuter verdicts that apply

The session-02 refuters worked on CPU only and read-only. Their verdicts are recorded in S1 Section 6.1 and S2 Section 6. None was about Stage A itself, because nothing generated has ever been scored. These verdicts shape it:

| Verdict | Evidence (file:line, from S1 and S2) | What S6 does with it |
|---|---|---|
| P1(a) confirmed: every method was scored against StackExchange, while the LoRA rows were fine-tuned on HackerNews | scripts/eval_c_checkpoints.py:96-109 (pre-rebuild lines); minigpt/data.py:202,207-213 | ID prompts follow S1: StackExchange for MC dropout, C1 and C2; HackerNews for C3 and the C4 rows, with StackExchange as a secondary set |
| P1(b) and P19 confirmed: the old OOD set was 9 arXiv papers, and the caches have no document boundaries | eval_c_checkpoints.py:103-104 (pre-rebuild lines); minigpt/data.py:162-166,177 | Prompts come only from S1's unseen test documents, one per document, so the document bootstrap is exact |
| P1(d) confirmed: all 11 cache tensors hold 0 EOT tokens (50256) | S1 Section 6.1 | The models never learned to stop. S6 decodes a fixed 64 tokens and applies the repeat stop rule. The drafting probe saw 0 EOT tokens in 4,096 decoded tokens. |
| P5 confirmed: C4-TFB's A equals C3's `lora_A_mu` exactly (maximum difference 0 on all 32 layers), and B equals C3's `lora_B` | minigpt/tfb.py:73-74,188-192; experiments/c_pipeline.py:107-110 | C3, `c4_tfb_fixed` and `c4_lap_refit` share one decoder (the merged BLoB mean) and one answer set. The pre-fix `c4_tfb` is excluded. The labels stay "BLoB-mean + TFB (fixed)" and "BLoB-mean + diag. Laplace LoRA-A (scaled)". |
| P4 confirmed: the old Laplace samples are near uniform (NLL 9.10 and 9.73; predictive entropy 9.4 of 10.8) | minigpt/laplace.py:115-123,167-168 (pre-fix lines); report.md:39,42,45 | Only S2's refit states are rescored |
| S0 check: one CPU draw of C2's 33.5M normals takes 102-149 ms | S0 Section 6 | The built v2 Laplace sampler draws on the GPU (minigpt/laplace.py:319-336), so Section 5.10 carries no CPU-draw cost |

Other evidence from the research documents:

| Source | Finding | Effect on S6 |
|---|---|---|
| metrics-assessment.md Section 1.1 (minigpt/model.py:176-196; experiments/eval_utils.py:74-86) | `generate` samples at temperature 1 with mean weights. The qualitative eval scores only the prompt. `frozen_bayesian_sample` is unused by any eval. | S6 writes its own greedy decode and does not change `generate` |
| metrics-assessment.md Section 4.2 | $g_t$ is a realized-token score. Its mean equals reverse MI only when $y_t\sim\bar p$. On greedy mean-weight answers it is not an RMI estimate. | The report and b6 call $\bar g$ "the realized-token Jensen gap", not "reverse MI" |
| paper-critique.md item 36 | Reviewers ranked scoring the model's own text low: "a 76M model trained from scratch writes weak text" | The stop rule, the repeat share and the degeneration sentence make the weakness visible instead of hiding it |
| paper-critique.md item 19 | On the old windows, C0 sequence NLL gave AUROC 0.920 on FreeLaw and 0.982 on PubMed | Prompt NLL is likely a strong comparator, so F3 is pre-registered |
| options-and-costs.md PG12; literature-survey.md G-NLL card | G-NLL beats multi-sample methods in 13 of 18 settings (W1). Epistemic scores win 1 of 16 OOD pairings in Schweighofer et al. (W2). | The negative framing N1 is written first |

### 6.2 Drafting probes (2026-09-26, CPU only, read-only)

Scripts in the session scratchpad: `scratchpad/s06/s6_probe_fixture.py`, `s6_probe_stacked.py` and `s6_probe_real.py` (full path: `C:/Users/alexe/AppData/Local/Temp/claude/D--dev-avdolgikh-github-repos-bayesian-llm/a8221562-e1e9-43b4-bfe9-32c916f249a4/scratchpad/s06/`). The scratchpad is temporary. Every number a test needs is copied here. No GPU job ran. Nothing in the repo changed.

| Probe | Result | Used in |
|---|---|---|
| Fixture (2 layers, width 32, vocab 100), BLoB with a non-zero B, $P = 12$, $L = 8$, $N = 5$: decode-step log-probs against teacher-forced decoder log-probs at positions $P-1,\dots,P+L-2$ | maximum difference $4.8\times10^{-7}$ | S6-T1(b) |
| Same: $\log q_j(y_j)-\max_v\log q_j(v)$ | 0.0 | S6-T1(b) |
| Same: minimum $g_j$, $\mathrm{MI}_j$, $\mathrm{EU}_j$; means | $1.2\times10^{-3}$, $3.2\times10^{-3}$, $5.4\times10^{-3}$; means 0.0079, 0.0046, 0.0081 | S6-T1(c) |
| Same with 5 identical samples | maximum $\lvert g\rvert$ 0.0; maximum $\lvert\mathrm{MI}\rvert$ $4.8\times10^{-7}$ | S6-T1(d) |
| Batched greedy decode (batch 3) against batch 1 | identical | S6-T1(a) |
| Stacked bmm wrappers against the per-sample loop through `frozen_bayesian_sample`: VI (`BayesianLinear` FFN) and BLoB, $N = 5$ | maximum difference 0.0 on every log-prob, both kinds. `modules()` order equals the block order. | S6-T2(a); Section 5.5 |
| `torch.func.vmap` over `functional_call` with stacked `lora_A`, chunk sizes 1, 2, 5 | equal to the loop (0.0). PyTorch warned: "There is a performance drop because we have not yet implemented the batching rule for aten::_scaled_dot_product_flash_attention_for_cpu". | Section 5.5: no vmap |
| MC dropout, replicated batch, the same seed twice | `torch.equal`: yes | S6-T2(c) |
| Fixture merge of $\frac{\alpha}{r}BA_\mu$ into $W_0$ | maximum logit difference $3.7\times10^{-7}$ | S6-T5(a) |
| Real C3 checkpoint, CPU fp32, 4 inputs of 192 tokens: merged BLoB mean against `use_mean_weights` | maximum logit difference $5.0\times10^{-5}$; argmax agreement 1.0 | S6-T5(a) tolerance $10^{-4}$; Section 5.4 |
| Saved states | C2 Laplace: 32 FFN weight tensors (`blocks.i.mlp.{fc,proj}.linear.weight`), no biases. C4-TFB: 32 `lora_A` tensors. | Section 5.4 |
| Real C0 on StackExchange 80M+ and arXiv's first 128K tokens; merged C3 mean on HackerNews 90M+ and arXiv. 8 prompts of 128 tokens per set, greedy to 64 tokens. | All 32 answers repeat a 4-gram within 64 tokens. The first repeat completes at token 5-53; the median per set is 18-25. The mean repeat share over 64 tokens is 0.52-0.66, and 3-6 of 8 answers per set are above 0.5. 0 EOT tokens. | R4; Section 3 stop rule; choice C1 |
| Same prompts with a no-repeat-4-gram block | 0 repeats by construction. The text is still weak (for example, invented terms such as "F-type"). | choice C1 |

These prompts come from slices that the old D1 eval and the gates already read, and that S1's new test set excludes. No uncertainty score and no AUROC was computed on any probe answer. The stop rule was chosen from repetition statistics alone, before any Stage A score exists.

### 6.3 Critic pass (session 03) and where this revision answers it

The critic checked two questions: does every test give a number or yes/no and name its script or pytest module, and is anything outside b3, b4, b5 and P7 (in part), P10. It also checked the interfaces against the code built in session 03 and the costs against roadmap Section 3.

| Critic item | Fix | Where |
|---|---|---|
| T3 named `check_eval_rebuild.py` but no mode or report per check; the draft ran both new modes in one call, which the checker's mutually exclusive group refuses | (a)-(e) → `--check stage-a`, `stage_a_check.json`; (f) → `--rederive-stage-a`, `rederive_stage_a_check.json`; two calls | S6-T3; Section 5.1 step 9 |
| T4(e) and T5(b)-(d) had no named check | `--check stage-a` checks them, against the Section 5.8 field list | S6-T4(e), S6-T5; Section 5.8 |
| G1(c) ran the whole module, which also holds T4 and T5 | `-k "s6_t1 or s6_t2"`, with no test skipped | Section 1 Gate; Section 4 |
| The re-run ranges (`0-99`) cut the batches of 8 and 32, so `torch.equal` could fail on shape alone | Per-domain batches, a whole-batch rule, `0-63,2000-2063` | Section 5.3; S6-T3(d); Section 5.9 |
| $N = 20$, the seeds and the score-set-to-decoder mapping sat outside the hashed blocks | `score_sets` frozen; `n_samples`, `seed_base` and `seed_offset` moved into `stage_a`; the decode batch size moved out of `decode` | Section 3; Sections 5.7, 5.9 |
| Freezing on the run night left the YAML unchecked for 8 days after the build | Freeze at the end of session 04; T4(c) checks the YAML against Section 3 | Section 5.1 step 5; S6-T4(c) |
| "S1's `load_model` with S2's entries" does not exist; `mc_dropout` is not in the registry | `SCORE_SET_REGISTRY[s].load`, `load_posthoc_refit`, `load_c0_model`; the YAML `method` must equal the loader's | Section 2 interface table; Section 5.4 |
| `realized_token_gap` returns `G` over all positions | The module masks `g` itself | Section 5.5 |
| $o_d$ read from the docs file, which holds only IDs and a list of token tensors | $o_d$ and $L_d$ from the manifest | Section 5.3 |
| `doc_id` and `domain` typed as tensors | lists of str, as in S1's files | Section 5.6 |
| `analysis_sha256` in meta clashed with S1's meaning | Defined as the Stage A freeze hash; `s1_analysis_sha256` added | Section 5.6; S6-T3(c) |
| S1's `freeze_prereg` cannot hash four blocks | Own function, same rule | Section 5.7 |
| Where $q_j$ for EU comes from was open; the decoder file cannot hold it | Decoder pass per rescore batch; the GPU table counts it per score set | Section 5.4; Section 5.10 |
| "Choose from S0's cells" was not possible: S0 has no stacked-pass cell | Peak measured in the latency part; `rescore` refuses above the budget | Section 5.5; S6-T5(b) |
| The block-0 lookup and the output file for $\bar g_{\mathrm{ref}}$ were open | Lookup rule and the gref file | Sections 5.6, 5.7 |
| The checker import rule was weaker than the built checker's | numpy, torch, PyYAML and sklearn only; S1's import test must still pass | Section 2; S6-T3(f) |
| No test showed that the checker catches a wrong value | S6-T4(f) on fixtures | S6-T4(f) |
| Out of scope: S3's combined-score test (m9) | Removed | Section 3; Section 7 |
| Borderline scope: StackExchange prompts for the LoRA rows (P1 is S1's) | Choice C4, default kept as descriptive | Section 2 |
| The cost column gave agents, network and disk as roadmap-row values | Marked "not in the row", with the options assumption | Section 1 |
| GPU: the decoder pass per score set, the latency repeats and the re-run size were not counted | Recomputed: 1.7-5.0 unchanged; 1.0-2.6 under C2 | Section 5.10 |
| The hidden CPU-draw cost for `c2_refit` no longer applies to the built sampler | Removed | Sections 5.10, 6.1 |
| R3 said the margin binds at 1,000 per class | At 1,000 Holm needs about 0.02-0.03 | R3; Section 8 |
| The N-sweep share is unstable near chance | Reported in above-chance cells only | Section 3 |
| "Drawn once per (batch, sample)" was wrong for MC dropout | Masks per element | Section 3 |
| The test seed base 3,000,000 sat in the YAML | Test constant | S6-T2; Section 5.9 |

## 7. What this does not cover

- Correctness. A 76M model trained from scratch cannot answer QA (metrics-assessment.md Section 6, Protocol B). Stage A tests OOD detection on continuations. It is not error or hallucination detection, so PRR, E-AURC and answer-level ECE are out.
- Ground truth for epistemic uncertainty: exposure counts, aleatoric controls, multi-answer facts. That is S7 (Exposure), in November only.
- Other decode rules: ancestral sampling from $\bar p$ (which would make the mean of $g$ an RMI estimate), the no-repeat block (choice C1), beam search and temperature.
- Semantic entropy, a TokUR re-implementation, hidden-state probes, and Mahalanobis at the last prompt token. S3 has distance baselines on blocks only.
- S3's combined-score test on the Stage A one-pass scores (m9, S3). A later spec can add it with its own frozen block.
- A KV cache. It is parked in AGENTS.md. Decode times are without one, so they overstate the decode cost of a production decoder.
- The S5 rows (deterministic LoRA + TFB or Laplace, and the LoRA ensemble). They can be added later with their own frozen family in a separate YAML. Each needs its own decoder (the deterministic LoRA), so one added TFB or Laplace-LoRA set costs about 0.5-1.4 GPU h at 5,000 responses (est.).
- The pre-fix `c4_tfb` row and the old Laplace states.
- Monte Carlo noise at $N = 20$ beyond the determinism re-run. No second seed base is scored.
- Whether the repeat stop rule is the best cut. The uncut 64-token answer is the only other variant scored.
- Paper text (b6, m11), a figure panel (S4 or b6), README.md and report.md (m13).
- Any result at 355M or larger.

Handoff to m3 and m13. S6 gives numbers for these texts. It does not edit them.

| Text | Where (per metrics-assessment.md Section 1.3 and Section 7) | What S6 gives |
|---|---|---|
| "Mean weights serve with zero overhead" | paper.tex:95, :299 | the merged-to-C0 decode ratio and the wording rule (S6-T5c) |
| "N=3 captures 97% of the signal" | paper.tex:52, :259, :277 | the share above chance on generated answers, with CIs (descriptive) |
| Key finding 8: "N=3 MC as a post-processing step on generated text" | report.md:132 | the first measurement on generated answers |
| Latency from one 256-token block of random tokens at batch 1 | scripts/benchmark_inference.py:186-234; report.md:86-101 | decode and rescoring ms per response on real answers (S6-T5b, d) |

## 8. Confidence

| Claim | Confidence | Why |
|---|---|---|
| The stacked pass equals the per-sample loop for VI and BLoB | high | Probe: difference 0.0 on the fixture, CPU fp32 |
| It also equals the loop for TFB v2 and both Laplace v2 kinds | medium-high | Same mechanism: explicit draws swapped in. The v2 samplers are built (session 03), but the stacked pass was not probed with them. |
| `torch.vmap` would not give a real batch speed-up | medium | PyTorch's fallback warning on CPU. CUDA was not tested. |
| At 76M, most greedy answers loop within 64 tokens | high | 32 of 32 probe answers, on two decoders and four slices |
| Loop rates and answer lengths per domain | low | 8 answers per set, on burned slices |
| The merged BLoB mean equals the unmerged mean path | high | $5.0\times10^{-5}$ maximum logit difference on the real C3 checkpoint; argmax agreement 1.0 |
| No answer contains an end token | high | 0 EOT in 4,096 probe tokens; 0 EOT in every cache |
| The three LoRA rows share one decoder | high | P5 verdict: maximum difference 0 between C4's $A_{\mathrm{MAP}}$ and C3's `lora_A_mu`; both S2 configs name `data/checkpoints/c3/ckpt_best.pt` as the base |
| The interfaces in Section 2 match the built code | high | Read from the session-03 code on 2026-09-26, with file:line. A later change to S1 or S2 code would need this table re-checked. |
| `c2_refit` draws on the GPU | high | minigpt/laplace.py:319-336; `load_posthoc_refit` maps the state to the device |
| The re-run is bitwise equal on the GPU | medium | Whole batches keep shapes and seeds equal. `use_deterministic_algorithms` runs with `warn_only`, so a non-deterministic kernel only warns; the warnings go to meta. |
| GPU 1.7-5.0 h | low-medium | CA10 rates; the decode rate assumes C0's 8.1 ms per step; the batching gain for decoding is not measured. S0 does not time the stacked pass; the latency part measures it before the rescore part. |
| Engineering 22-42 h | medium-low | Built up from parts. The stacked pass and the re-derivation carry most of the risk. |
| Review 1.5-2.25 h | low | CA1's rate gives up to 6.8 h. It holds only under choice C3. |
| At 1,000 prompts per class, Holm needs a difference of about 0.02-0.03 | medium-low | Hanley-McNeil arithmetic at AUROC 0.8, with an assumed correlation of 0.5-0.75 between the two AUROC estimates. Not measured. |
| $\bar g$ beats G-NLL on generated answers | unknown | The literature leans negative (PG12). Loops may drive both scores. Prompt NLL was already strong on FreeLaw and PubMed on the old windows. |
| Stage A joins v1 | low | Roadmap: G1(b) needs 8.5 h or less through Oct 11, and the plan is 7.5-11.0 h |
