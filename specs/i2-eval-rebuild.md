# S1 Eval rebuild: document-level eval set, realized-token score, document bootstrap, eval-script tests

Status: DRAFT (not approved). Date: 2026-09-26. Owner: Alexey approves; agents implement.
Revision 2, after the session-02 critic. Section 6.4 maps each critic item to the place that answers it.
File: `specs/i2-eval-rebuild.md`. Plan of record: `agents/plans/roadmap.md` section 3, row S1.

## 1. Header

| Field | Value |
|---|---|
| Items | m4a (data rebuild), m4 (eval rebuild and re-score), m5 (document-clustered paired bootstrap), m15 (tests that run the eval scripts, plus an independent re-derivation of the table) |
| Issues | P1, P6, P11, P19, P20, P23 |
| Approve by | Tue Sep 29 (roadmap section 3). Build in session 03, Wed Sep 30. Re-stream Thu Oct 1, night. Reproduction test and m4 re-score Fri Oct 2, night. |
| Depends on | Q3 (commit policy) must be answered before session 03 writes code (session-02 plan). S0-T1 gives the batching gain and the batch size. S0-T4 sets `max_blocks_per_doc`. |
| Feeds | S2 (val documents, the scorer, families F3 and F4), S3 and S4 (score files), S5 (the scorer) |

| Cost | Range | Split by item | Assumption |
|---|---|---|---|
| Alexey, approve | inside m2 (1-1.5 h for S0-S4, so about 12-18 min for this spec) | none | CA1 |
| Alexey, review | 1.0-1.25 h. This is the plan figure. | m4a 0.25; m4 0.25-0.5; m5 0.25; m15 0.25 | CA1, roadmap row S1 |
| Engineering (agents) | 8.5-20 h | m4a 3-8; m4 2-5; m5 1.5-3; m15 2-4 | CA2 |
| GPU (RTX 4070), roadmap | 1.2-10.6 h | reproduction 0.2-0.6; m4 re-score 1-10 | CA3, CA10 |
| GPU, this spec's estimate | 2.0-5.6 h, plus up to 0.5 h if the S1-T1 spread branch runs | reproduction 0.46-0.48; m4 1.5-5.1 (section 5.8) | CA10 batch-1 rates; the low end assumes a 3x batching gain (S0-T1) |
| Network | 1-4 h wall clock, 0 GPU h. Volume about 250-280M tokens re-encoded (section 5.8). | m4a | est. (options m4a); unmeasured until the P19 probe |
| Disk | 1.5-2.5 GB under `data/` | score files with float32 per-sample log-probs, plus test-document tokens | est. |

Notes on the costs:
- Review hours. CA1 prices review at 1 h per 6-9 engineering hours. Over 8.5-20 engineering hours that gives 0.94-3.33 h. The plan uses the roadmap figure, 1.0-1.25 h. That holds while engineering stays at or below about 11 h (1.25 h × 9). Above that, the extra review is up to about 2.1 h. Whether to spend it is Alexey's call (decision D3).
- GPU. This spec's 2.0-5.6 h sits inside the roadmap range. S0-T4 replaces it with a measured number. If S0-T4 puts m4 above 10 GPU h, `max_blocks_per_doc` drops to 1 (G0 rule).
- Cost moved to S2. S1 does not score C2, C4-LAP or the fixed TFB. S2 must score them on the same blocks: 4.1-4.7 GPU h at batch 1, or 1.4-1.6 h with a 3x batching gain (section 5.8). S2's roadmap row has 0.7-1.5 GPU h. So S2's row is under-costed by this design. S1 also scores TFB twice in total (pre-fix here, fixed in S2), which roadmap m4 counted once.

Decisions only Alexey can make. He is unavailable, so each has a default.

| # | Decision | Default while Alexey is silent | What it blocks |
|---|---|---|---|
| D1 | Approve S1 | None. S1 stays DRAFT and session 03 does not build it. | All of S1 |
| D2 | Q3: branch and commit policy | Suggested by options section 6: code, configs, specs and tests on a branch, pushed after review. If Q3 is still open at session 03: code stays in the working tree, no commits, and every score file records `code_sha256` of the source it ran (section 5.5). | All code |
| D3 | Which review-hours figure is the plan | 1.0-1.25 h (roadmap). Overrun, if any, is logged and costs up to about +2.1 h. | nothing |
| D4 | What happens if S1-T1 fails | The failure-branch table under S1-T1 | m4 |
| D5 | Primary ID set for the LoRA rows (C3, C4) | HackerNews, their fine-tuning domain (refuter P1(a)). StackExchange is secondary. | the freeze |
| D6 | S2 GPU budget for C2, C4-LAP and fixed TFB | Score all blocks: 4.1-4.7 GPU h at batch 1, over one night. If S0-T1 gives less than 3x, score only the first block of each document: 2.0 h at batch 1, 0.65 h at 3x. | S2 |

## 2. Scope and non-scope

What this spec delivers:
1. A document-level eval set built from YAML. It has 5 domains, a document ID on every block, and 1-3 blocks per document. No document is in two splits. No test document is an exact or near duplicate of any cached token stream that training, validation, an HP gate or the old eval could read. "Near duplicate" means that 50% or more of a block's 32-token windows occur in a cached tensor. Shorter shared spans are measured and reported per domain.
2. A seeded scorer in both eval scripts. It saves the N per-sample log-probs of every realized token, the per-token MI, TU and AU, and the document ID of every block.
3. A document-clustered, class-stratified, paired bootstrap for AUROC and FPR@95 (m5). The RNG, the draw order and the percentile rule are fixed (section 5.6).
4. Tests that run both eval scripts end to end on a tiny fixture. Separate code, written from this spec only, re-derives the tables from the score files (m15).
5. A reproduction test on the old 1,000 blocks, before any fix, plus a determinism re-run and a batch-size check.
6. A frozen pre-registration: a hash of the `analysis` block that the scorer checks before it scores any test block.

The eval set:

| YAML key | Role | Cached tensor (read-only) | Who read the cached tokens | Test documents (new) | Val documents (for S2 fits) |
|---|---|---|---|---|---|
| `stackexchange` | Primary ID for MC dropout and C1. Secondary ID for C3 and C4. | `stackexchange_100000000.pt` | [0, 60M): C0/C1 training. [60M, 80M): validation. [80M, 100M): C0 perplexity gate, C1 MI-ratio gate, and the old D1 eval (first 128,001 tokens, about 210 threads). | 1,000 documents after the 100M cut | fully inside [60M, 80M) |
| `hackernews` | Primary ID for C3 and C4 | `hackernews_100000000.pt` | [0, 80M): LoRA training. [80M, 90M): validation. [90M, 100M): C3 gate. | 1,000 after the 100M cut | fully inside [80M, 90M) |
| `arxiv` | OOD | `arxiv_10000000.pt` | Whole tensor: C0 perplexity gate and C1/C3 MI-ratio gates (random windows). First 128,001 tokens: old D1 eval, which is 9 papers; 2 of them give 58% of its 500 blocks. | 1,000 after the 10M cut | none |
| `freelaw` | OOD | `freelaw_10000000.pt` | Whole tensor: gates | 1,000 after the 10M cut | none |
| `pubmed_abstracts` | OOD | `pubmed_abstracts_10000000.pt` | Whole tensor: gates | 1,000 after the 10M cut | none |
| `arxiv`, stripped | OOD variant, used by S3-T4 | none | none | The same 1,000 test papers with LaTeX markup removed | none |
| `wikipedia_en` | None. Checked against only. | `wikipedia_en_100000000.pt` | C0/C1 training: all 100M tokens | none | none |

Rules:
- The split ranges come from `configs/c0.yaml` and `configs/c3_phase2.yaml` through the arithmetic in minigpt/data.py:207-213. Nobody types them in a second time.
- A document that straddles a range edge (10M, 60M, 80M, 90M or 100M) goes in no split.
- There is no dev split. No later spec names a consumer for one. S3 cross-validates by document on the test set (roadmap S3 test 3).
- Val documents get IDs only if the re-stream matches the cache for that domain (S1-T2a). Otherwise S2 uses the val ranges as flat token ranges. S1 does not score val.
- Test documents are the first 1,000 eligible documents in stream order after the cut that pass the unseen check (section 5.3). A document is eligible if it has at least 257 tokens. For arXiv, the stripped copy must also have at least 257 tokens, so both arXiv variants hold the same 1,000 papers.

Scoring coverage in S1:

| Eval set | Blocks | Score sets | Seed offset |
|---|---|---|---|
| `legacy_d1` | The old 500 StackExchange + 500 arXiv blocks | c0, mc_dropout, c1, c3, c4_tfb, with the current samplers | 0 |
| `test` | StackExchange, arXiv, FreeLaw and PubMed test blocks | c0, mc_dropout, c1, c3, c4_tfb (pre-fix) | 100,000 |
| `arxiv_stripped` | Stripped arXiv test blocks | the same 5 | 300,000 |
| `test_hn` | HackerNews test blocks | c3, c4_tfb (pre-fix) | 200,000 |

- C4-TFB is BLoB-mean + TFB. Its adapter is C3's `lora_A_mu` with C3's `lora_B` (refuter P5). Every table labels it "BLoB-mean + TFB (pre-fix)", and its meta records `adapter_source: blob_mean`. It is descriptive only in S1 (section 3).
- S1 does not score C2 or C4-LAP. Their samplers are broken (P4, confirmed; section 6.1). S2 scores C2, C4-LAP and the fixed TFB with this scorer on these blocks, under the score-set names `c2_refit`, `c4_lap_refit` and `c4_tfb_fixed`.

Files:

| Change | Files |
|---|---|
| New | `configs/i2_eval.yaml`; `minigpt/evalset.py` (manifest, blocks, score-file writer, freeze check); `scripts/build_eval_set.py`; `scripts/check_eval_rebuild.py`; `tests/test_eval_rebuild.py`; `tests/test_doc_bootstrap.py`; `tests/test_eval_scripts.py` |
| Edited | `minigpt/uncertainty.py`: new functions, plus an optional `sample_weight` argument on `auroc` and `fpr_at_tpr`. With the default `None`, both return today's values. `scripts/eval_c_checkpoints.py`; `scripts/eval_mc_dropout.py` |
| New data (gitignored) | `data/eval_i2/` (manifest, document tokens, block files, build report); `data/scores_i2/` (score files, tables, check reports, `prereg.json`) |
| Docs (AGENTS.md "Keep repo docs fresh" and "Detail Documents") | AGENTS.md structure block (scripts line, test count); README.md structure block and Quick Start; `agents/technical-reference.md` (eval set, G score, score-file format); `agents/design-rationale.md` (the choices in section 6.3); `agents/milestone-history.md` when S1 completes |

Must not change:
- Model, layer, LoRA, training and sampler code: minigpt/model.py, layers.py, lora.py, train.py, laplace.py, tfb.py. The samplers change only in S2.
- The training data path in minigpt/data.py, and every cached tensor under `data/pile/`. The builder imports the constants at data.py:127-138, streams on its own code path and writes no cache file.
- `experiments/` and `.pipeline-state/`. The pipeline stays frozen as the record of the C results (code-assessment.md section 5.2).
- `data/d1_scores.pt`, `data/mc_dropout_scores.pt` and `data/checkpoints/`. They are the reproduction reference.
- The behaviour of `bootstrap_ci` (minigpt/uncertainty.py:434-476) and its tests (tests/test_d0_metrics.py:467-515). All 288 existing tests keep passing.
- `paper/`, report.md and the README numbers. Text changes wait until the recomputed tables exist (PG3; m3, m11, m13).

AGENTS.md rules that apply:
- Explicit configs. Every parameter in this spec lives in `configs/i2_eval.yaml` (section 5.2). New code reads `cfg["key"]`, never `.get()` with a default. Test fixtures keep their constants in the test module, as listed in section 5.6.
- No notebooks. No new Bayesian library.
- LaTeX formulas in specs.
- Keep repo docs fresh: new scripts and the new test count go into AGENTS.md and README.md at once. Design choices go into `agents/design-rationale.md`.
- Lint and tests pass before any commit. CI lints only `minigpt/ experiments/ tests/`, so the builder also runs `uv run ruff check` on the four touched scripts. 100-character lines. Type hints on public functions.
- Commit rules (Conventional Commits, no mention of AI assistants) apply only if D2 allows commits.

## 3. Pre-registered endpoint

This section is fixed at approval, before any score on a test block exists. The `analysis` block of `configs/i2_eval.yaml` holds all of it. Its sha256 goes into `data/scores_i2/prereg.json`, and the scorer refuses the `test`, `test_hn` and `arxiv_stripped` sets while the hash differs (S1-T5g, S1-T5i). The hash is also written in the session log and STATE.md, because `data/` is gitignored and Q3 may block commits.

Primary score: the per-block mean of the realized-token Jensen gap.

$$\ell_{s,t}=\log p_{\theta_s}(y_t\mid x_{\le t}),\qquad \log\bar p(y_t)=\operatorname{logsumexp}_{s=1}^{N}\ell_{s,t}-\log N$$

$$g_t=\log\bar p(y_t)-\frac{1}{N}\sum_{s=1}^{N}\ell_{s,t}\;\ge\;0,\qquad G(b)=\frac{1}{T}\sum_{t=1}^{T}g_t$$

Reported next to it, on the same blocks, is block-averaged MI:

$$\mathrm{MI}_t=H[\bar p_t]-\frac{1}{N}\sum_{s=1}^{N}H[p_{\theta_s,t}],\qquad M(b)=\frac{1}{T}\sum_{t=1}^{T}\mathrm{MI}_t$$

The score files also hold block TU, AU, NLL under $\bar p$, and 1 − max-prob.

Fixed settings: $T=256$. $N=20$ for every stochastic score set and $N=1$ for C0. The mean runs over all 256 positions, with no skip and no max. Seed base 0. fp16 autocast, as in the old runs.

| Quantity | Definition | Where |
|---|---|---|
| Primary metric | Document-weighted AUROC of $G$, per method and OOD domain (arXiv, FreeLaw, PubMed). The ID set is the method's own training domain: StackExchange for MC dropout, C1 and C2; HackerNews for C3 and C4. | section 5.6 |
| Primary contrast | $\Delta=\mathrm{AUROC}_w(G)-\mathrm{AUROC}_w(M)$ for the same method and blocks. Paired document bootstrap, 10,000 resamples. | section 5.6 |
| Family F1 (S1, primary) | MC dropout and C1 on ID = StackExchange; C3 on ID = HackerNews. 3 rows × 3 OOD domains = 9 cells. Holm over 9. | `analysis.families` |
| Family F2 (S1, secondary) | C3 on ID = StackExchange. 3 cells. Holm over 3. | same |
| Family F3 (S2, primary, frozen now) | `c2_refit` on StackExchange; `c4_lap_refit` and `c4_tfb_fixed` on HackerNews. 9 cells. Holm over 9. Tested after the S2 refits. | same |
| Family F4 (S2, secondary, frozen now) | `c4_lap_refit` and `c4_tfb_fixed` on StackExchange. 6 cells. Holm over 6. | same |
| Descriptive only | C4-TFB pre-fix on both ID sets. arXiv stripped for MC dropout, C1, C3 and C4-TFB pre-fix. Above-chance and replication checks. | same |
| Margin | $\delta=0.02$ AUROC. This is twice the ±0.01 reproduction tolerance. A smaller difference may be Monte Carlo or GPU noise. | |
| Level | 0.05 after Holm | |

A row that a later spec adds (for example a deterministic-LoRA + TFB row from S5) gets its own family in a separate YAML file, frozen the same way before its first test score exists. Nobody edits the families above.

Decision rules:

| Result | Rule | What the paper says |
|---|---|---|
| G beats MI in a cell | $\Delta\ge+0.02$ and Holm-adjusted $p<0.05$ | "G scores higher than MI for <method> on <domain>." |
| MI beats G in a cell | $\Delta\le-0.02$ and Holm-adjusted $p<0.05$ | "MI scores higher than G for <method> on <domain>." G stays the primary score. We do not switch after seeing the test set. |
| No difference | anything else | "G and MI rank blocks the same at this margin." |
| Above chance (descriptive) | the 95% CI of $\mathrm{AUROC}_w(G)$ lies above 0.5 | reported per method and domain |
| Old arXiv number replicates (descriptive) | ID = StackExchange for all 4 methods. Both must hold: the new arXiv $\mathrm{AUROC}_w(M)$ lies inside the old 64-block-cluster CI (MC dropout [0.858, 0.931]; C1 [0.835, 0.910]; C3 [0.881, 0.934]; C4-TFB [0.888, 0.942]), and the old point (0.8976, 0.8739, 0.9085, 0.9173) lies inside the new document CI. The old CIs rest on 8 clusters per class, which metrics-assessment.md:121 calls indicative. | "replicates" or "does not replicate on 1,000 unseen papers" |

Negative-result framing, written in advance:
- N1. "On 1,000 unseen documents per domain, weight-sampling disagreement from <method> does not separate <FreeLaw or PubMed> from <its training domain>: AUROC <x> [<lo>, <hi>], an interval that includes 0.5."
- N2. "The published arXiv AUROCs (0.874-0.917) came from 9 papers, 2 of which gave 58% of the blocks, drawn from an arXiv cache that the tuning gates also read. On 1,000 unseen papers they are <x>, outside the old intervals."
- N3. "Scoring only the realized token (G) does not rank blocks better than full-vocabulary MI in <k> of 9 cells of F1. We keep G as the primary score because it needs only N log-probs per token and applies unchanged to generated answers (S6)."

This endpoint does not decide two things. Whether MI or G beats one-pass scores is S3's endpoint. Which posterior method is best is decided by S3's paired tests with Holm correction.

## 4. Acceptance tests

A test passes only if every check listed under it passes, except where a check says "report only".

| ID | Given | When | Then |
|---|---|---|---|
| S1-T1 Reproduction and determinism | The legacy set: blocks 0-499 are StackExchange cache tokens [80,000,000, 80,128,001) (all_id[180,000,000:180,128,001]); blocks 500-999 are arXiv cache tokens [0, 128,001). The saved checkpoints, the current samplers, `data/d1_scores.pt` and `data/mc_dropout_scores.pt`. | Both eval scripts run `--eval-set legacy_d1 --run-tag main` on all 1,000 blocks at batch 1. A fresh process runs `--block-ids 0-99,500-599 --run-tag rerun`. If `scoring.batch_size` for `test` is above 1, a third run uses `--block-ids 0-99,500-599 --batch-size <that value> --run-tag batched`. | Checks T1a-T1f all PASS. Runner: `scripts/check_eval_rebuild.py --check repro`, which writes `data/scores_i2/repro_check.json` with PASS, PASS-WITH-SPREAD or FAIL. No m4 re-score starts on FAIL. |
| S1-T2 Stream and unseen check | The Hugging Face stream `ArmelR/the-pile-splitted`, shuffle seed 1337, buffer 10,000. The 11 cached tensors in `seen_tensors` (6 large, 5 small). | `scripts/build_eval_set.py` re-encodes the 5 domains with `encode_ordinary` and runs the unseen check. `scripts/check_eval_rebuild.py --check stream` rescans on its own code path. | Pass = T2b and T2c and T2d. T2a is reported only, and it decides only whether val documents get IDs. |
| S1-T3 Eval-set invariants | The manifest `data/eval_i2/manifest.jsonl`, the document-token and block files, and `configs/i2_eval.yaml` | `scripts/check_eval_rebuild.py --check manifest` runs on the real build. `tests/test_eval_rebuild.py` runs the same checks on a synthetic stream of 5 fake domains. | Checks T3a-T3g all PASS |
| S1-T4 Document bootstrap | The fixtures F-i and F-ii of section 5.6, built in the test module from fixed seeds | `tests/test_doc_bootstrap.py` runs `doc_bootstrap_auroc`, `paired_doc_bootstrap` and `holm` with 2,000 resamples and seed 0 | Checks T4a-T4f all PASS. The session-02 probe `s1_boot_fixture.py` gave the reference values. |
| S1-T5 Eval scripts end to end, re-derivation, freeze | Fixture: a 2-layer model (n_embd 32, block size 32, dropout 0.1); checkpoints and post-hoc states for c0, c1, c2, c3, c4_tfb and c4_lap in `tmp_path`; an eval set with 3 domains (1 ID, 2 OOD), 20 blocks from 10 documents each; N = 3; 200 bootstrap resamples. Real: `data/scores_i2/` after m4. | `tests/test_eval_scripts.py` runs `main()` of both scripts on CPU, twice. Then `scripts/check_eval_rebuild.py --rederive` rebuilds the tables with numpy, torch.load and sklearn only. On real data, `--check prereg`, `--check align` and `--rederive` run after m4 and again on the final table in session 05. | Checks T5a-T5k all PASS |

### S1-T1 checks

| Check | Pass rule | Runner |
|---|---|---|
| T1a | MI AUROC is within ±0.010 of the saved value: MC dropout 0.8976, C1 0.8739, C3 0.9085, C4-TFB 0.9173 | `check_eval_rebuild.py --check repro` |
| T1b | C0 max-prob AUROC is within ±0.002 of 0.5911 | same |
| T1c | Per-block Spearman correlation between new and saved scores is at least 0.99 for C0 (`blk_tu` against saved `pred_ent`) and for C4-TFB (`blk_mi` against saved `mi`). Both had deterministic draws. | same |
| T1d | The `rerun` files equal the matching rows of the `main` files: `torch.equal` on every tensor, for all 5 score sets (yes/no). Block indices keep their full-set values, so every seed is unchanged. | same |
| T1e | (1) From the old saved files, mean M(OOD) / mean M(ID) is C1 1.2637, C3 1.4051, C4-TFB 1.3443 and MC dropout 1.3049, each ±1e-4. This confirms the hardcoded 1.32 / 1.53 / 1.35 did not come from them. (2) Every ratio the new scripts print equals the ratio computed from the new file to 1e-6. (3) The new C4-TFB ratio is within ±0.01 of 1.3443. | same |
| T1f | Only if the `batched` run exists; otherwise "n/a". On the 200 blocks: C0's per-block absolute difference to batch 1 is at most 1e-3 on every `blk_*` score. For MC dropout, C1, C3 and C4-TFB, MI AUROC is within ±0.01 of the batch-1 value on the same blocks. | same |

Failure branch (the default for D4):

| Case | Status | Action |
|---|---|---|
| T1a-T1f all pass | PASS | m4 starts |
| C0 or C4-TFB fails any of T1a-T1e | FAIL | Their draws are deterministic, so this points at the scorer. Stop all of m4, report, fix, re-run T1. |
| MC dropout, C1 or C3 misses T1a by more than 0.010 and at most 0.030, and C0 and C4-TFB pass | PASS-WITH-SPREAD or FAIL | Re-run that method on all 1,000 legacy blocks with seed bases 1,000,000 and 2,000,000. If the saved value lies inside [min − 0.010, max + 0.010] of the three runs, the status is PASS-WITH-SPREAD, the spread is recorded, and m4 starts. Otherwise that method is FAIL and m4 skips it. The other methods go ahead. |
| Any method misses T1a by more than 0.030 | FAIL for that method | m4 skips that method. The others go ahead if C0 and C4-TFB pass. |
| T1d fails | FAIL | Stop all of m4. Report the op named in `determinism_warnings`, fix it, re-run T1d. |
| T1f fails | PASS (noted) | m4 runs at batch 1. The GPU cost rises to the batch-1 figure (section 5.8). |

### S1-T2 checks

| Check | Pass rule | Runner |
|---|---|---|
| T2a (report only) | Per domain: does the re-encoded stream equal the whole cached tensor (`torch.equal`, yes/no), and at which index is the first mismatch. The `datasets` version is recorded. | `build_eval_set.py`; printed by `--check stream` |
| T2b | 0 kept test documents whose first 32 tokens occur in any tensor in `seen_tensors`. 0 kept test blocks with 50% or more of their 226 windows (32 tokens each) in a seen tensor. The checker confirms both with its own rescan: a separate implementation with hash base `checker_hash_base`, followed by an exact token comparison of every hit. It does not read the builder's hit list. | `check_eval_rebuild.py --check stream` |
| T2c | `build_report.json` gives, per domain: candidates checked, documents dropped by each rule, within-test duplicates dropped, and the fraction of kept-block windows that occur in a seen tensor. The checker's rescan reproduces the kept-block window-hit count exactly (yes/no). If more than 5% of a domain's candidates are dropped, the report lists the 10 most common matched prefixes (yes/no). | same |
| T2d | In `tests/test_eval_rebuild.py`, on a synthetic stream: 1 injected exact duplicate of a cached document gives exactly 1 detection. 1 injected 280-token document whose tokens 10-279 are copied from a cached tensor gives exactly 1 detection (its first 32 tokens are new). 0 injected duplicates give 0 detections. | pytest |

### S1-T3 checks

| Check | Pass rule | Runner |
|---|---|---|
| T3a | The manifest's domain list equals the YAML list. In pytest, removing `pubmed_abstracts` from a fixture YAML removes it from the fixture manifest (yes/no). | `--check manifest`; pytest |
| T3b | StackExchange, HackerNews, arXiv, FreeLaw and PubMed each have exactly `n_test_docs` = 1,000 test documents. The stripped arXiv set has exactly 1,000, with the same document IDs as arXiv (yes/no). | same |
| T3c | Every block's document ID exists in the manifest. Every test document has $1\le k_d\le$ `max_blocks_per_doc` and $o_d+k_dT+1\le L_d$. Every block equals the stored document tokens $t_{o_d+jT\,:\,o_d+(j+1)T+1}$ exactly. The sha1 of each stored document's tokens equals the manifest (yes/no; real build and pytest). | same |
| T3d | 0 document IDs appear in two splits. 0 documents straddle a range edge. 0 repeated text sha1 values among test documents, across all domains. | same |
| T3e | Every test document of every domain has $c_i\ge$ `cache_tokens` for its domain (yes/no). If T2a is yes for arXiv: the number of arXiv documents with $c_i<128{,}001$, which lie under the old D1 blocks, equals 9 (yes/no). A different count stops the build until it is explained. If T2a is no for arXiv: "n/a". | same |
| T3f | The decoded stripped blocks contain 0 `\` characters and 0 `$` characters | same |
| T3g | The YAML-derived split function gives StackExchange train [0, 60M), val [60M, 80M), old test [80M, 100M), and HackerNews train [0, 80M), val [80M, 90M), old test [90M, 100M) (yes/no) | pytest |

### S1-T4 checks

| Check | Pass rule | Runner |
|---|---|---|
| T4a | On F-i, $\mathrm{AUROC}_w$ over blocks equals the one-row-per-document AUROC to 1e-12, and $\mathrm{FPR95}_w$ equals the one-row-per-document FPR@95 to 1e-12. Probe: 0.729278 for both AUROCs, against 0.731459 unweighted; 0.723333 for both FPR@95 values. | `tests/test_doc_bootstrap.py` |
| T4b | On F-i, the block-level CI equals the one-row-per-document CI to 1e-9 (probe: [0.6881, 0.7667], difference 1e-15). The count-weight form of section 5.6 equals the concatenation form to 1e-9 (probe: 2e-15). | same |
| T4c | Two runs with seed 0 give identical resample arrays (`np.array_equal`) | same |
| T4d | Paired, with score b = score a: the $\Delta$ CI is [0, 0] and p = 1. Paired, with a = s and b = s − 0.8y on F-ii: every $\Delta_r>0$ (probe: minimum 0.185) and $p=2/(B+1)=0.0009995$. | same |
| T4e | On F-ii, the document CI is at least 1.4 times as wide as the i.i.d. block CI from `bootstrap_ci` with seed 0. Probe: 1.59. The design effect $\sqrt{1+(k-1)\rho}=\sqrt{2.8}=1.67$. | same |
| T4f | `holm([0.01, 0.04, 0.03])` returns [0.03, 0.06, 0.06] to 1e-12 | same |

### S1-T5 checks

| Check | Pass rule | Runner |
|---|---|---|
| T5a | Both scripts write score files with every key, shape and dtype in the section 5.5 table, and every meta key listed there (yes/no per key). In each fixture eval set, `doc_id`, `domain` and `offset` are identical across all score files (yes/no). | `tests/test_eval_scripts.py` |
| T5b | `--rederive` reproduces block G and M, $\mathrm{AUROC}_w$, $\mathrm{FPR95}_w$, CIs, $\Delta$, p and Holm-adjusted p in every table file to 1e-6 (yes/no per cell). `check_eval_rebuild.py` imports neither `minigpt.uncertainty` nor the eval scripts (yes/no, by an import check). | same |
| T5c | $g_t\ge-10^{-6}$ for every token. For C0 (N = 1), G and M are 0 to 1e-6. For TFB with $\sigma_q=0$ (identical samples), G is 0 to 1e-6 and M is 0 to 1e-5. | same |
| T5d | $\log\bar p(y_t)$ from `logp_real` equals $\log\bar p_t[y_t]$ from the full-vocabulary path to 1e-4, on every token with $\bar p_t[y_t]\ge10^{-30}$ | same |
| T5e | The two runs give identical tensors (`torch.equal`) | same |
| T5f | `MI_RATIOS` no longer appears in `scripts/eval_c_checkpoints.py` (0 grep hits). Any printed MI ratio equals mean M(OOD) / mean M(ID) from the file to 1e-6. | same |
| T5g | With a `prereg.json` whose hash differs from the fixture YAML's `analysis` block, `main(--eval-set test)` exits with a non-zero code and writes no score file (yes/no). `legacy_d1` still runs (yes/no). | same |
| T5h | The whole test module takes 60 s or less on CPU | same |
| T5i | Real data: the sha256 of the YAML `analysis` block equals `prereg.json` and equals `meta.analysis_sha256` in every test score file. `prereg.json`'s `frozen_at` is earlier than every test score file's `created_at` (yes/no). | `check_eval_rebuild.py --check prereg` |
| T5j | Real data: `--rederive` on `data/scores_i2/` equals `table_test.json`, `table_test_hn.json`, `table_arxiv_stripped.json` and `families.json` to 1e-6 in every cell (yes/no). It runs after m4 and again on the final table in session 05 (m15). | `check_eval_rebuild.py --rederive` |
| T5k | Real data: `doc_id`, `domain` and `offset` are identical across all score files of each eval set (yes/no) | `check_eval_rebuild.py --check align` |

## 5. Implementation notes for the builder

### 5.1 Order of work

| Step | Work | Test | Machine | When |
|---|---|---|---|---|
| 1 | Weighted AUROC, weighted FPR@95, document bootstrap and `holm()` in `minigpt/uncertainty.py` | S1-T4 | CPU | session 03 |
| 2 | Scorer changes in both scripts; score-file format; `--analyze`; delete `MI_RATIOS` | S1-T5a, c-h | CPU | session 03 |
| 2b | A separate agent writes `check_eval_rebuild.py --rederive`. It sees sections 3, 5.5 and 5.6 of this spec and the score-file format only, not the scorer code. | S1-T5b | CPU | session 03 |
| 3 | `minigpt/evalset.py` and `scripts/build_eval_set.py`: stream, IDs, splits, unseen check, blocks, stripped copy | S1-T2d, S1-T3 (pytest) | CPU | session 03 |
| 4 | After approval: write `configs/i2_eval.yaml` and run `check_eval_rebuild.py --freeze`, which writes `prereg.json`. Record the hash in the session log and STATE.md. | S1-T5i | CPU | before step 6 |
| 5 | Re-stream and build | S1-T2, S1-T3 (real) | network, CPU | Thu Oct 1, night |
| 5b | Reproduction run on the legacy blocks | S1-T1 | GPU, 0.46-0.48 h | Fri Oct 2 |
| 6 | m4 re-score of `test`, `arxiv_stripped` and `test_hn`, then `--analyze` | S1-T5i-k | GPU, 1.5-5.1 h | Fri Oct 2, night |

### 5.2 `configs/i2_eval.yaml` (every value explicit)

```yaml
eval_set:
  name: i2_v1
  source: {dataset: ArmelR/the-pile-splitted, split: train, shuffle_seed: 1337, shuffle_buffer: 10000}
  train_configs: {base: configs/c0.yaml, adapter: configs/c3_phase2.yaml}
  domains:                       # list order = bootstrap seed index (section 5.6)
    - {key: stackexchange, role: id_base, cache_tokens: 100000000, max_stream_tokens: 110000000}
    - {key: hackernews, role: id_adapter, cache_tokens: 100000000, max_stream_tokens: 120000000}
    - {key: arxiv, role: ood, cache_tokens: 10000000, max_stream_tokens: 50000000,
       stripped_copy: true}
    - {key: freelaw, role: ood, cache_tokens: 10000000, max_stream_tokens: 30000000}
    - {key: pubmed_abstracts, role: ood, cache_tokens: 10000000, max_stream_tokens: 15000000}
  seen_tensors: [wikipedia_en_100000000, stackexchange_100000000, hackernews_100000000,
                 arxiv_10000000, freelaw_10000000, pubmed_abstracts_10000000,
                 wikipedia_en_50000, stackexchange_50000, arxiv_10000, freelaw_10000,
                 pubmed_abstracts_10000]
  splits: [test, val]            # no dev split
  block_size: 256
  min_doc_tokens: 257            # asserted equal to block_size + 1
  max_blocks_per_doc: 3          # 1 if S0-T4 puts m4 above 10 GPU h
  n_test_docs: 1000
  candidate_pool_factor: 1.2
  offset_seed: 0
  unseen_check:
    window_tokens: 32
    hash_base: 1000003
    checker_hash_base: 1000000007
    chunk_tokens: 10000000       # chunks overlap by window_tokens - 1
    near_dup_window_frac: 0.5
    report_drop_frac: 0.05
    report_top_prefixes: 10
  strip:
    math_envs: [equation, equation*, align, align*, eqnarray, eqnarray*, gather, gather*,
                multline, multline*, displaymath, math]
  out_dir: data/eval_i2
scoring:
  seed_base: 0
  seed_offsets: {legacy_d1: 0, test: 100000, test_hn: 200000, arxiv_stripped: 300000}
  n_samples: {c0: 1, mc_dropout: 20, c1: 20, c3: 20, c4_tfb: 20}
  score_sets:
    legacy_d1: [c0, mc_dropout, c1, c3, c4_tfb]
    test: [c0, mc_dropout, c1, c3, c4_tfb]
    arxiv_stripped: [c0, mc_dropout, c1, c3, c4_tfb]
    test_hn: [c3, c4_tfb]
  labels:
    c0: {display: "C0 deterministic", sampler: none, adapter_source: none}
    mc_dropout: {display: "MC dropout (C0)", sampler: dropout, adapter_source: none}
    c1: {display: "C1 variational FFN", sampler: variational, adapter_source: none}
    c3: {display: "C3 BLoB LoRA", sampler: variational, adapter_source: blob}
    c4_tfb: {display: "BLoB-mean + TFB (pre-fix)", sampler: pre-fix, adapter_source: blob_mean}
  batch_size: {legacy_d1: 1, test: 1, test_hn: 1, arxiv_stripped: 1}   # S0-T1 sets the last 3
  amp_fp16: true
  determinism: {use_deterministic_algorithms: true, warn_only: true, cudnn_deterministic: true,
                cudnn_benchmark: false, cublas_workspace_config: ":4096:8"}
  out_dir: data/scores_i2
analysis:                        # frozen at approval; sha256 goes to data/scores_i2/prereg.json
  primary_score: blk_g
  contrast_score: blk_mi
  secondary_scores: [blk_tu, blk_au, blk_nll, blk_maxprob_unc]
  aggregation: mean_all_positions
  ood_domains: [arxiv, freelaw, pubmed_abstracts]
  fpr_target_tpr: 0.95
  bootstrap: {resamples: 10000, seed: 0, level: 0.95, unit: document, stratify_by_class: true,
              percentile_method: linear}
  margin_auroc: 0.02
  alpha: 0.05
  families:                      # rows = [score_set, id_domain]; cells = rows x ood_domains
    F1_s1_primary: [[mc_dropout, stackexchange], [c1, stackexchange], [c3, hackernews]]
    F2_s1_lora_on_se: [[c3, stackexchange]]
    F3_s2_primary: [[c2_refit, stackexchange], [c4_lap_refit, hackernews],
                    [c4_tfb_fixed, hackernews]]
    F4_s2_lora_on_se: [[c4_lap_refit, stackexchange], [c4_tfb_fixed, stackexchange]]
  descriptive:
    rows: [[c4_tfb, hackernews], [c4_tfb, stackexchange]]
    stripped_rows: [[mc_dropout, stackexchange], [c1, stackexchange], [c3, hackernews],
                    [c4_tfb, hackernews]]
    replication:
      id: stackexchange
      ood: arxiv
      old_point: {mc_dropout: 0.8976, c1: 0.8739, c3: 0.9085, c4_tfb: 0.9173}
      old_cluster_ci: {mc_dropout: [0.858, 0.931], c1: [0.835, 0.910], c3: [0.881, 0.934],
                       c4_tfb: [0.888, 0.942]}
checks:
  repro:
    mi_auroc_target: {mc_dropout: 0.8976, c1: 0.8739, c3: 0.9085, c4_tfb: 0.9173}
    mi_auroc_tol: 0.010
    c0_maxprob_target: 0.5911
    c0_maxprob_tol: 0.002
    spearman_min: 0.99
    spearman_pairs: {c0: [blk_tu, pred_ent], c4_tfb: [blk_mi, mi]}
    rerun_block_ids: "0-99,500-599"
    spread_seed_bases: [1000000, 2000000]
    spread_max_miss: 0.030
    old_ratio_target: {c1: 1.2637, c3: 1.4051, c4_tfb: 1.3443, mc_dropout: 1.3049}
    old_ratio_tol: 0.0001
    new_ratio_tol_c4_tfb: 0.01
    batch_c0_max_abs: 0.001
    batch_mi_auroc_tol: 0.01
  arxiv_old_d1_docs: 9
  rederive_tol: 1.0e-6
```

### 5.3 Streaming, document IDs, splits and the unseen check

- Stream exactly as minigpt/data.py:156-161: `load_dataset(PILE_DATASET_PATH, name=PILE_DOMAIN_NAMES[key], split="train", streaming=True).shuffle(seed=1337, buffer_size=10_000)`, with the constants from data.py:127-138. Encode each item with `encode_ordinary`, as at data.py:164. Stop at `max_stream_tokens`.
- Document ID: `f"{key}/{i:09d}/{sha1(text)[:12]}"`, where $i$ is the 0-based index in the shuffled stream and `text` is UTF-8.
- Let $c_i$ be the cumulative token start of document $i$ and $L_i$ its length. Document $i$ is inside a range $[a,b)$ if $a\le c_i$ and $c_i+L_i\le b$. The cut index is $K=\min\{i : c_i\ge \texttt{cache\_tokens}\}$. This matches the loop at data.py:163-166, which stops after the document that crosses the limit. Test candidates have $i\ge K$. The rule applies whether or not the stream matches. If it does not match, positions refer to the re-stream, and the unseen check is the guarantee.
- Match check (T2a): compare the first `cache_tokens` re-encoded tokens with `torch.load(cache)` using `torch.equal`. Report the first mismatch index.
- Split arithmetic, from data.py:198-213. The base ID stream is cat(wikipedia 100M, stackexchange 100M). Training ends at $\lfloor 0.8\cdot 2\times10^8\rfloor=1.6\times10^8$ and validation at $1.8\times10^8$. In local StackExchange positions: train [0, 60M), val [60M, 80M), old test [80M, 100M). HackerNews alone: train [0, 80M), val [80M, 90M), old test [90M, 100M). The function computes these from the two YAML configs with the same expressions as data.py:207-208 (T3g).
- Manifest row per document: `doc_id`, `domain`, `split`, `variant` (raw or stripped), $i$, $c_i$, $L_d$, `token_sha1` (sha1 of the int32 little-endian token bytes), $o_d$, $k_d$. Test-document tokens are stored as int32 in `data/eval_i2/docs_{domain}.pt`, so the checker can verify every block (T3c). Val rows store the range only.
- Candidate pool. Walk the stream past the cut and collect the first $\lceil 1.2\times1000\rceil=1{,}200$ eligible documents per domain. Build their blocks (section 5.4). Run the unseen check on all of them. Drop within-test duplicates by text sha1, keeping the first. Keep the first 1,000 survivors. If fewer survive, extend the pool and repeat, up to `max_stream_tokens`. If that is not enough, T3b fails and the report gives the count.
- Unseen check. The query set holds the first 32 tokens of each candidate document, plus every 32-token window of each of its blocks: $257-31=226$ windows per block, about 3.4M queries in total. Scan every 32-token window of each tensor in `seen_tensors` with a 64-bit polynomial hash, with $P$ = `hash_base` = 1,000,003:
  $$h_i=\sum_{j=0}^{31} t_{i+j}\,P^{31-j}\bmod 2^{64}$$
  numpy uint64 arithmetic wraps mod $2^{64}$. Compute it by Horner's rule, 32 vectorized passes: $h\leftarrow hP+t_{j:j+n-31}$. Work on one tensor at a time, in 10M-token chunks that overlap by 31 tokens. Look up the hashes in the sorted query set with `np.searchsorted`. Confirm every hit by exact token comparison, so a hash collision costs time but never a wrong drop. Drop a candidate if its first 32 tokens hit, or if any of its blocks has 50% or more of its windows hit. Report the window-hit fraction of the kept blocks per domain. If more than 5% of a domain's candidates are dropped, list the 10 most common matched prefixes, because they are likely boilerplate. The stripped arXiv copy inherits the result of its raw document.
- The checker (`--check stream`) runs its own scan with base $P_2$ = `checker_hash_base` = 1,000,000,007 and its own code. It rechecks the kept test set only.
- If the stream does not match for a domain, its val documents get no IDs. S2 then fits on the val ranges as flat token ranges. The test documents stay valid, because the unseen check tests the tokens themselves.

### 5.4 Blocks per document, and the stripped arXiv copy

- A document is eligible if $L_d\ge T+1=257$. For arXiv, the stripped copy must also have $L\ge257$. Then
$$k_d=\min\!\big(k_{\max},\ \lfloor (L_d-1)/T\rfloor\big),\qquad o_d\sim\mathcal{U}\{0,\dots,L_d-1-k_dT\}$$
Block $j$ has $x=t_{o_d+jT\,:\,o_d+(j+1)T}$ and $y=t_{o_d+jT+1\,:\,o_d+(j+1)T+1}$, for $j=0,\dots,k_d-1$. So $o_d+k_dT+1\le L_d$ always holds.
- Per-document RNG: `np.random.default_rng([offset_seed, int(sha1_12, 16)])` for the raw copy and `np.random.default_rng([offset_seed, int(sha1_12, 16), 1])` for the stripped copy. The offset then does not depend on document order.
- The report gives, per domain, the eligible fraction, the median $L_d$ and the mean $k_d$. The 257-token rule removes short PubMed abstracts and short StackExchange threads. The report states this, and S3's matched bins deal with length.
- Stripped arXiv. On the text: delete `$...$`, `$$...$$`, `\[...\]`, `\(...\)` and the environments in `strip.math_envs`. Delete the remaining `\command` tokens (regex `\\[A-Za-z@]+\*?`), keep the text of their brace arguments, then delete braces and any stray `\` or `$`. Collapse whitespace, re-encode, and apply the same block rule. Keep the same document ID with `variant: "stripped"`.

### 5.5 Scorer and score file

- In the per-sample loop at scripts/eval_c_checkpoints.py:241-258 and scripts/eval_mc_dropout.py:118-126, compute `lp = torch.log_softmax(logits.float(), -1)`. Use `probs = lp.exp()` in place of the softmax at :256 and :124. Keep `ell[s] = lp.gather(-1, y[..., None]).squeeze(-1)` in float32 on the device. After the loop, one shared helper in `minigpt/uncertainty.py` computes $\log\bar p(y_t)$, $g_t$ and $G$ as in section 3. Keep the MI, TU and AU code at :260-263 and :128-131. `mc_metrics_single` (uncertainty.py:28-71) stays as it is, because the pipeline uses it.
- Seeding. Let $b$ be the block's index in the full eval-set file. With `--block-ids`, $b$ keeps its full-set value. Let
$$\text{seed}_b=\text{seed\_base}+\text{seed\_offset}[\text{eval\_set}]+b$$
For VI, BLoB and MC dropout, call `torch.manual_seed(seed_b)` before the N passes. These samplers draw from the global RNG (minigpt/layers.py:72; minigpt/lora.py:75, 89; dropout via layers.py:230-242). For Laplace and TFB, the seed of sample $s$ is $\text{seed}_b\cdot N+s$. On `legacy_d1` (seed base 0, offset 0, $b$ = old `seq_idx`: 0-499 ID, 500-999 OOD) this equals the old seeding at eval_c_checkpoints.py:243 and :318, :335-336.
- No seed collides. VI seeds: legacy 0-999, test from 100,000, test_hn from 200,000, stripped from 300,000, spread re-runs from 1,000,000 and 2,000,000. TFB seeds are those values times 20, plus $s$, so the ranges stay apart.
- With batch size above 1, one weight draw is shared by the blocks of a batch and seeded by the batch's first block. Batches are consecutive $b$ in file order. The batch size goes into meta. The main S1-T1 run is always at batch 1.
- Determinism: apply `scoring.determinism` before the model loads: `torch.use_deterministic_algorithms(True, warn_only=True)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Record every warning in meta.
- Score file `data/scores_i2/{score_set}__{eval_set}__{run_tag}.pt`:

| Key | Shape, type | Note |
|---|---|---|
| `logp_real` | [B, N, T] float32 | $\ell_{s,t}$. float32, because fp16 spacing is 0.004-0.008 for log-probs between −16 and −4, which is above BLoB's mean MI of 0.0035 nats. |
| `tok_mi`, `tok_tu`, `tok_au` | [B, T] float32 | full-vocabulary path |
| `tok_maxprob` | [B, T] float32 | $\max_v\bar p_t(v)$ |
| `tok_sum_p_sq` | [B, T] float32 | $\sum_v\bar p_t(v)^2$, as at eval_c_checkpoints.py:268. Brier: $1-2\bar p_t(y_t)+\sum_v\bar p_t(v)^2$ (:361). |
| `tok_correct` | [B, T] bool | $\arg\max_v\bar p_t(v)=y_t$ |
| `blk_g`, `blk_mi`, `blk_tu`, `blk_au`, `blk_nll`, `blk_maxprob_unc` | [B] float32 | block means over all T positions |
| `block_index`, `doc_id`, `domain`, `offset`, `weight` | [B] | `block_index` = $b$; weight $=1/k_d$. On `legacy_d1`: `doc_id` = `legacy/<domain>/<b>`, weight 1. |
| `meta` | dict | `git_sha`, `git_dirty`, `code_sha256` (sha256 over the minigpt/*.py files and the script that ran), `checkpoint_sha256` (per file), `score_set`, `display_label`, `sampler`, `adapter_source`, `eval_set`, `run_tag`, `block_ids`, `n_samples`, `seed_base`, `seed_offset`, `batch_size`, `amp_fp16`, `device`, `torch_version`, `cuda_version`, `determinism_warnings`, `manifest_sha256` (null on `legacy_d1`), `yaml_sha256`, `analysis_sha256`, `created_at` (UTC) |

- Freeze. `analysis_sha256` is the sha256 of `json.dumps(cfg["analysis"], sort_keys=True, separators=(",", ":"))`. `check_eval_rebuild.py --freeze` writes it and `frozen_at` to `data/scores_i2/prereg.json`, and refuses to overwrite an existing file. The scorer refuses `test`, `test_hn` and `arxiv_stripped` if `prereg.json` is missing or its hash differs.

### 5.6 Weighted AUROC, weighted FPR@95 and the document bootstrap

The re-derivation agent works from this section only, so every rule is spelled out.

- Cell. A cell is (score set, ID domain, OOD domain). Its blocks are the ID-domain blocks (label 0) and the OOD-domain blocks (label 1) of that score set. For ID = HackerNews, the ID blocks come from `{s}__test_hn__main.pt` and the OOD blocks from `{s}__test__main.pt`. For the stripped rows, the OOD blocks come from `{s}__arxiv_stripped__main.pt`. Higher score means more OOD.
- The block weight is $w_b=1/k_{d(b)}$, so each document counts once:
$$\mathrm{AUROC}_w=\frac{\sum_{i\in\mathcal{I}}\sum_{j\in\mathcal{O}}w_iw_j\big(\mathbb{1}[s_j>s_i]+\tfrac12\mathbb{1}[s_j=s_i]\big)}{\big(\sum_{i\in\mathcal{I}}w_i\big)\big(\sum_{j\in\mathcal{O}}w_j\big)}$$
This is `roc_auc_score(y, s, sample_weight=w)`.
- Weighted FPR@95. With $\mathrm{TPR}_w(\tau)=\sum_{j\in\mathcal O}w_j\mathbb 1[s_j\ge\tau]\big/\sum_{j\in\mathcal O}w_j$ and $\mathrm{FPR}_w(\tau)$ defined the same way over $\mathcal I$:
$$\tau^*=\max\{\tau:\mathrm{TPR}_w(\tau)\ge0.95\},\qquad \mathrm{FPR95}_w=\mathrm{FPR}_w(\tau^*)$$
In code: `fpr, tpr, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)`, then `fpr[np.argmax(tpr >= 0.95)]`. With `sample_weight=None`, `fpr_at_tpr` (uncertainty.py:244-261) keeps today's call, so old numbers do not move.
- Document order. ID documents are `np.unique(doc_id[y == 0])` and OOD documents are `np.unique(doc_id[y == 1])`, both in numpy's lexicographic order. $n_{\mathcal I}$ and $n_{\mathcal O}$ are their counts.
- RNG. One generator per (ID domain, OOD domain) pair: `rng = np.random.default_rng([seed, i_id, i_ood])`. Here `seed` = `analysis.bootstrap.seed` and `i_id`, `i_ood` are positions in `eval_set.domains`: stackexchange 0, hackernews 1, arxiv 2, freelaw 3, pubmed_abstracts 4, and arxiv_stripped 5. Every score set and every score in that pair uses the same draws, so every comparison is paired. The T4 fixtures call the functions with the integer seed 0.
- Resample $r=1,\dots,B$, in this order: `di = rng.integers(0, n_I, size=n_I)`, then `do = rng.integers(0, n_O, size=n_O)`. Let $c^{(r)}_d$ be the number of times document $d$ was drawn. The resampled block weight is
$$w_b^{(r)}=w_b\,c^{(r)}_{d(b)}$$
This equals taking all blocks of each drawn document with multiplicity (T4b checks both forms). Compute every score's $\mathrm{AUROC}_w$ and $\mathrm{FPR95}_w$ with $w^{(r)}$. The contrast is $\Delta_r=A^{(a)}_r-A^{(b)}_r$.
- Point estimates use the original weights $w_b$, not the bootstrap mean.
- CI: `np.percentile(values, [2.5, 97.5])` with numpy's default linear method.
- p-value, two-sided, with a floor of $2/(B+1)$:
$$p=\min\Big\{1,\ \frac{2\,\big(1+\min(\#\{\Delta_r\le0\},\ \#\{\Delta_r\ge0\})\big)}{B+1}\Big\}$$
- Holm, within each family. Sort the $m$ p-values in ascending order with a stable sort. Then
$$\tilde p_{(i)}=\max_{j\le i}\ \min\{1,\ (m-j+1)\,p_{(j)}\}$$
and return them in the original order.
- Speed (optional). The scores do not change across resamples, so a faster weighted Mann-Whitney with a single sort is allowed. It must equal `roc_auc_score` to 1e-12 on the T4 fixtures. `--rederive` uses sklearn.
- New functions sit next to `bootstrap_ci` (uncertainty.py:434-476) and leave it unchanged: `doc_bootstrap_auroc(scores: dict[str, np.ndarray], labels, doc_ids, weights, n_resamples, seed, level)`, `paired_doc_bootstrap(...)`, which returns $\Delta$, the CI and p for listed pairs, and `holm(pvalues)`.
- `--analyze` writes `table_test.json`, `table_test_hn.json`, `table_arxiv_stripped.json` (per cell: $\mathrm{AUROC}_w$ and $\mathrm{FPR95}_w$ for every score, with CIs) and `families.json` ($\Delta$, CI, p and Holm-adjusted p per cell).

Fixtures for S1-T4. They are copied from the session-02 probe `s1_boot_fixture.py`, so the reference values reproduce.

| Fixture | Construction |
|---|---|
| F-i | `rng = np.random.default_rng(101)`. 300 ID and 300 OOD documents, named `d0000`-`d0299` (ID) and `d0300`-`d0599` (OOD). `k = rng.integers(1, 4, size=600)`. Document score `s_doc = rng.normal(loc=1.0 * y_doc, scale=1.0)`. Each document has k blocks, and each block carries its document's score. $w=1/k$. 1,182 blocks. |
| F-ii | `rng = np.random.default_rng(202)`. 200 ID and 200 OOD documents with k = 3 (1,200 blocks). `u = rng.normal(0, sqrt(0.9), size=400)` per document, then `e = rng.normal(0, sqrt(0.1), size=1200)` per block. Block score `u[doc] + e + 0.8 * y`. Within-document correlation $\rho=0.9$. |
| Runs | B = 2,000 resamples, bootstrap seed 0. The i.i.d. reference is `bootstrap_ci(s, y, auroc, n_bootstrap=2000, seed=0)`. |

### 5.7 Script changes (smallest change)

| Place | Change |
|---|---|
| scripts/eval_c_checkpoints.py:71-78, 341, 400, 477 | Delete `MI_RATIOS`. If a ratio is printed, compute it from the score file and label it descriptive. |
| :96-109 and scripts/eval_mc_dropout.py:66-77 | Keep `load_eval_data` as the `legacy_d1` path. Add `--eval-config`, `--eval-set {legacy_d1, test, test_hn, arxiv_stripped}`, `--block-ids` (comma list of ranges, such as `0-99,500-599`), `--run-tag` (for example `main`, `rerun`, `batched`, `spread1`), `--batch-size` (overrides the YAML value; recorded in meta) and `--seed-base` (accepts only `scoring.seed_base` or a value in `checks.repro.spread_seed_bases`). The new sets read the manifest and block files. |
| :215-277 and eval_mc_dropout.py:99-145 | Log-prob capture, seeding and determinism (section 5.5). |
| :299 | Take N from `scoring.n_samples[score_set]`, not from `method == "deterministic"`. C0 stays at N = 1. |
| :384-408 and eval_mc_dropout.py:339-341 | New score-file format (section 5.5). |
| :415-441 and eval_mc_dropout.py:344-360 | On the new sets, replace the i.i.d. CIs with the document bootstrap. Keep the i.i.d. CIs on `legacy_d1` for comparison. Add `--analyze`. |
| Tests | `tests/test_eval_scripts.py` imports each script with `importlib`. It uses monkeypatch to point `CKPT_DIR` at `tmp_path` and `build_milestone_config` at 2-layer fixture configs. The scripts' model loaders stay unchanged. Rebuilding models from `ckpt["config"]` (K29) is out of scope. |
| scripts/check_eval_rebuild.py | Subcommands `--check repro`, `--check stream`, `--check manifest`, `--check prereg`, `--check align`, `--freeze` and `--rederive`. `--rederive` and `--check stream` must not import `minigpt.uncertainty`, `minigpt.evalset` or the eval scripts. Session 05 reuses `--rederive` for the final table (m15). |

### 5.8 GPU and network estimates (CA10 rates, N = 20, batch 1)

Per-block cost: C0 8.1 ms, MC dropout 287 ms (assumed equal to C1), C1 287 ms, C3 316 ms, C4-TFB 469 ms. That is 1.367 s for the 5 score sets and 0.785 s for the 2 HackerNews score sets.

| Set | Docs | Blocks (est.) | Score sets | GPU h at batch 1 |
|---|---|---|---|---|
| S1-T1: legacy 1,000 + `rerun` 200 + `batched` 200 | none | 1,400 | 5 | 0.46-0.48 (the batched run at 3x) |
| S1-T1 spread branch, only if needed | none | up to 2 × 1,000 | MC dropout, C1, C3 | up to 0.49 |
| StackExchange test | 1,000 | 1,500-2,000 (about 610 tokens per thread; metrics-assessment.md:118) | 5 | 0.57-0.76 |
| arXiv test | 1,000 | 3,000 (14-26K tokens per paper) | 5 | 1.14 |
| arXiv stripped | 1,000 | 3,000 | 5 | 1.14 |
| FreeLaw test | 1,000 | 2,500-3,000 (est.) | 5 | 0.95-1.14 |
| PubMed test | 1,000 | 1,000-1,200 (est.) | 5 | 0.38-0.46 |
| HackerNews test | 1,000 | 1,500-2,000 (est.) | 2 | 0.33-0.44 |
| m4 total | | 12,500-14,200 | | 4.5-5.1; 1.5-1.7 with a 3x batching gain; 2.1 with `max_blocks_per_doc` = 1 |
| S1 total | | | | 2.0-5.6, plus up to 0.49 for the spread branch |
| Not S1: S2 re-score of `c2_refit`, `c4_lap_refit`, `c4_tfb_fixed` | | 11,000-12,200 outside HackerNews at 1.225 s; 1,500-2,000 on HackerNews at 0.938 s | 3 | 4.1-4.7; 1.4-1.6 at 3x; 2.0 (0.65 at 3x) on the first block of each document |

The spread branch would take the reproduction line to 0.97 h, above the roadmap's 0.2-0.6 h. It still fits the S1 total.

Network volume, re-encoded from the start of each domain (est.):

| Domain | Tokens | Why |
|---|---|---|
| StackExchange | about 101M | 100M to reach the cut, plus about 1M for 1,200 candidates |
| HackerNews | 101-106M | 100M, plus 1-6M (document length unknown) |
| arXiv | 27-41M | 10M, plus 1,200 papers at 14-26K tokens. m4a assumed 15-30M, before test papers moved past the 10M cut. |
| FreeLaw | 12-17M | 10M, plus 1,200 opinions at 2-6K tokens (est.) |
| PubMed | about 11M | 10M, plus about 2,700 abstracts to find 1,200 of 257 tokens or more |
| Total | about 250-280M | about 1.0-1.1 GB of text at about 4 bytes per token |

The 1-4 h wall-clock estimate (m4a) is kept. It is unmeasured until the P19 probe runs in session 03.

## 6. Verification evidence from session 02

### 6.1 Refuter verdicts that apply

All three refuters worked on CPU only, read-only (`map_location='cpu'`, CUDA hidden). Probes are in the session-02 scratchpad `s02/`.

P1/P19 (eval split, MI-ratio hardcode, no document boundaries): CONFIRMED on all four sub-claims. Recorded in agents/logs/2026-09-26-session-02-log.md:13 and agents/memory/STATE.md:71-72.

| Sub-claim | Verdict | Evidence (file:line) | Numbers |
|---|---|---|---|
| (a) ID is StackExchange only | confirmed | scripts/eval_c_checkpoints.py:96-109 (always `build_milestone_config("c0")`; first 500 windows, :85-93); minigpt/data.py:202, 207-213 (test_id = all_id[180M:200M]); experiments/c_milestones.py:12, 19, 78; configs/c0.yaml:15-23; configs/c3_phase2.yaml:15-16; configs/c4_tfb.yaml:15-16 | ID = StackExchange[80,000,000:80,128,001], about 210 threads (`Q:` starts). C0 CPU forward, saved vs recomputed on SE[80M], ID rows 0-2: (1.4976, 0.3733) vs (1.4974, 0.3733); (2.8424, 0.5536) vs (2.8421, 0.5535); (3.6552, 0.5706) vs (3.6547, 0.5706). HackerNews[90M] gives (4.98, 0.73) and does not match. Rows 499 and 999 match to about 1e-4. |
| (b) OOD is a few arXiv papers | confirmed; exactly 9, not "5-9" | eval_c_checkpoints.py:103-104 (cat in `OOD_DOMAINS` order); c_milestones.py:12 | OOD = arxiv[0:128,001], fully inside the 10M arXiv cache. 9 documents (5 start with `abstract:`, 4 with `author:`; 9 `\ntitle:` lines). Windows per document: 26, 16, 27, 167, 12, 51, 47, 32, 122. Two documents give 289 of 500 windows (58%). FreeLaw and PubMed are never scored. OOD rows 500-502 match arxiv[0]. |
| (c) MI ratio is hardcoded | confirmed | eval_c_checkpoints.py:71-78, 341, 400, 477; experiments/pipeline_runner.py:521-537 and experiments/eval_utils.py:179 (a different estimator: mean of per-domain ratios, random batches, each milestone's own test_id); agents/milestone-history.md:38 (C3 1.53x from MLflow run 3b22bfcf on HackerNews) | Recomputed from `data/d1_scores.pt`: C1 0.059882/0.047387 = 1.2637; C3 0.004959/0.003529 = 1.4051; C4-TFB 0.105905/0.078780 = 1.3443; C2 1.0141; C4-LAP 0.9973; MC dropout 1.3049. Hardcoded: 1.32 / 1.53 / 1.35. |
| (d) No document boundaries | confirmed | minigpt/data.py:162-166, 177 (documents joined with no separator; last one truncated) | All 11 cache tensors in `data/pile/` hold 0 EOT tokens (50256). |
| Score file and AUROCs | confirmed | `data/d1_scores.pt` | 6 methods × 1,000 rows: 500 ID then 500 OOD; keys `mi`, `pred_ent`, `max_prob_unc`, `labels`; no domain or document field. AUROC (sklearn and Mann-Whitney agree): C0 max-prob 0.5911, C1 0.8739, C2 0.5358, C3 0.9085, C4-TFB 0.9173, C4-LAP 0.4941, MC dropout 0.8976. They match report.md. |
| Paper text | contradicted by code | paper/paper.tex:173, 176 ("HackerNews" ID; "ArXiv + FreeLaw + PubMed" OOD); report.md:41, 43, 44 and paper.tex:201, 203, 204 (MI ratio next to AUROC) | none |

What S1 does with it: the test set is rebuilt with document IDs and all 5 domains (T3). ID follows each method's training domain, so HackerNews is primary for C3 and C4 (D5, section 3). `MI_RATIOS` is deleted and ratios come from the files (T1e, T5f). The bootstrap resamples documents (T4). T1 targets are the recomputed AUROCs above. T3e expects 9 arXiv documents under the old blocks.

P5 (TFB rotation): CONFIRMED. It applies to how S1 labels and uses C4-TFB.

| Evidence (file:line) | What it shows | Number |
|---|---|---|
| minigpt/tfb.py:73-74, 178, 188-192 | `svd` returns (U, S, Vh); the code stores Vh as "V" and never uses it. Noise is row-scaled on A (std $\sigma_q/S_i$), not isotropic in $A'=V^\top A$. | Real C4-TFB state: per-layer trace ratio (code / target) 1.063-2.020, median 1.150 |
| scripts/eval_c_checkpoints.py:126-127, 133-150, 186-194; experiments/c_pipeline.py:107-110, 123-159; experiments/c_milestones.py:106-118, 272-273 | C4-TFB loads C3's BLoB checkpoint and maps `lora_A_mu` to `lora_A` | max abs(a_map − lora_A_mu) = 0 on all 32 layers; B equals C3's `lora_B` |
| paper/paper.tex:75, 94, 159, 326 | The paper calls TFB "zero Bayesian training" | none |

What S1 does with it: C4-TFB is scored with the current sampler, labelled "BLoB-mean + TFB (pre-fix)", with `adapter_source: blob_mean` in meta. It is descriptive only. The fixed TFB row is `c4_tfb_fixed` in families F3 and F4, scored in S2.

P4 (Laplace scale): CONFIRMED. It applies to why S1 does not score C2 or C4-LAP.

| Evidence (file:line) | What it shows | Number |
|---|---|---|
| minigpt/laplace.py:115-123, 167-168; configs/c2.yaml:79-81; configs/c4_lap.yaml:78-80 | The Fisher is a mean, not scaled by data size; damping 1.0; std $=1/\sqrt{F+1}$ | Stored std median 1.000 (C2 min 0.998834; C4-LAP min 0.999911); median abs(w) 0.0247 (C2) and 0.0233 (C4-LAP) |
| report.md:39, 42, 45; `data/d1_scores.pt` | The sampled models are near-uniform | NLL 9.10 (C2) and 9.73 (C4-LAP) against MAP C0 2.79; predictive entropy 9.40 and 9.16 of 10.82; MI ratio 1.014 and 0.997 |

What S1 does with it: C2 and C4-LAP are not scored in S1. S2 refits them and scores them with this scorer on these blocks (families F3 and F4; cost in section 5.8).

### 6.2 The writer's own checks (CPU only, read-only)

The probe scripts are in `C:/Users/alexe/AppData/Local/Temp/claude/D--dev-avdolgikh-github-repos-bayesian-llm/a8221562-e1e9-43b4-bfe9-32c916f249a4/scratchpad/s02/` (`s1_probe_scores.py`, `s1_boot_sim.py`, `s1_boot_fixture.py`). The scratchpad is temporary. Everything a test needs from them is copied into this spec.

| Check | Result | Evidence |
|---|---|---|
| Saved score files | 6 methods plus MC dropout. Each holds 4 arrays of 1,000 block means, with labels 500 / 500. There is no domain, document or offset field. | `s1_probe_scores.py`; scripts/eval_c_checkpoints.py:343-351, 384-391 |
| S1-T1 targets | MI AUROC: MC dropout 0.8976, C1 0.8739, C3 0.9085, C4-TFB 0.9173. C0 max-prob 0.5911, C0 TU 0.5449. | `s1_probe_scores.py` |
| C0 uses N = 1 | `actual_n = 1 if method == "deterministic"` | eval_c_checkpoints.py:299 |
| Brier and NLL inputs | `sum_p_sq` and `p_true` per token; Brier $1-2p+\sum p^2$ | eval_c_checkpoints.py:266-268, 360-361 |
| Current FPR@95 rule | `roc_curve` with default `drop_intermediate`, then the first index with tpr at or above the target | minigpt/uncertainty.py:244-261 |
| Eval domains come from code, not YAML | `build_milestone_config("c0")` and a hardcoded `OOD_DOMAINS` | eval_c_checkpoints.py:98, 103; c_milestones.py:12 |
| The gates read the full OOD caches and the ID test slice | Random, unseeded 256-token windows from `test_id` and from each whole `test_ood_{domain}` tensor. The C1 gate needs MI ratio > 1.2, the C3 gate > 1.05, and the C0 gate uses OOD perplexity. So FreeLaw and PubMed caches were read too. P11's "the gates never saw them" does not hold at the cache level. | experiments/pipeline_runner.py:505-537; experiments/eval_utils.py:158-185; minigpt/train.py:40-50; experiments/c_milestones.py:197-208 |
| No selection happened in practice | Every C milestone finished on run 1 | `.pipeline-state/c0.json` to `c4_tfb.json` |
| The cache key ignores the seed | `cache_path = pile_dir / f"{domain_key}_{token_limit}.pt"` | minigpt/data.py:149 |
| Split arithmetic and cache sizes | StackExchange and HackerNews caches hold 100M tokens; arXiv, FreeLaw and PubMed hold 10M. 5 small caches (10K-50K tokens) also exist. Shuffle seed 1337. | data.py:198-213; configs/c0.yaml:7-8, 15-22, 61; configs/c3_phase2.yaml:7-8, 15-16, 60; `data/pile/` listing |
| Current bootstrap | i.i.d. over pooled blocks. The class counts vary between resamples. | minigpt/uncertainty.py:434-476 (draw at :466) |
| Sampler seeding | VI and BLoB use the global RNG. MC dropout uses train mode. TFB and Laplace use a seeded CPU generator with seed = seq_idx·N + s. TFB with $\sigma_q=0$ returns the MAP copy. | layers.py:72, 230-242; lora.py:75, 89; eval_c_checkpoints.py:241-254; minigpt/tfb.py:169-183 |
| No test imports the eval scripts | A grep over `tests/` for both script names finds 0 files | tests/ |
| T4 reference values, exact fixtures | F-i: weighted block AUROC 0.729278 = document AUROC 0.729278 (difference 1e-15); unweighted block AUROC 0.731459; FPR@95 0.723333 both ways; CI [0.6881, 0.7667] both ways; count form vs concatenation form 2e-15. F-ii: width ratio 1.587. Shift test: minimum $\Delta_r$ 0.185, p = 0.0009995. Identical scores: CI [0, 0], p = 1. `holm([0.01, 0.04, 0.03])` = [0.03, 0.06, 0.06]. | `s1_boot_fixture.py` (the reviser's probe). The earlier `s1_boot_sim.py` gave 0.7350 / 0.7424 / 1.66 with a chained RNG; the spec now fixes separate seeds 101 and 202. |

### 6.3 Differences from roadmap section 3 and the options document

These choices also go into `agents/design-rationale.md` when S1 is built.

| # | Roadmap or options document | This spec | Why |
|---|---|---|---|
| 1 | T3: ID test = StackExchange tokens 80M-100M | ID test = 1,000 StackExchange documents after the 100M cut. The 80M-100M slice is not used. | The C0 perplexity gate, the C1 MI-ratio gate and the old D1 eval read that slice (eval_utils.py:164-167; train.py:47). It costs the same streaming run, plus about 1M tokens. |
| 2 | T3: "the 5-9 arXiv papers that the gates read are absent" | Every document in the cached 10M tokens of arXiv, FreeLaw and PubMed is excluded from test | The gates read windows from the whole of each cached OOD tensor, not only the 9 old papers (pipeline_runner.py:518-537). |
| 3 | T2: the hash check runs only if the stream does not match, on first-32-token prefixes | It always runs, on prefixes and on every 32-token window of every block. The checker rescans on its own code path. | A duplicate past the cut would pass a position-only rule. A self-graded report proves nothing. The cost is two CPU scans. |
| 4 | AUROC over blocks, unweighted (implicit) | Document-weighted, with weight 1/k | T4's equality with the document-level bootstrap holds only with these weights (0.7293 vs 0.7315 on F-i). |
| 5 | m4: re-score 7 score sets once | 5 score sets in S1. C2, C4-LAP and fixed TFB are scored in S2. | Their samplers are broken (P4, P5). This moves 4.1-4.7 GPU h at batch 1 to S2, above S2's 0.7-1.5 h row (D6). TFB is scored twice in total. |
| 6 | T1: two full runs, identical score files | One full run, a 200-block re-run with `torch.equal` on every tensor, plus a batch-size check | The same "identical" rule on fewer blocks keeps T1 inside 0.2-0.6 GPU h (0.46-0.48 h). The batch check covers m4 at batch above 1, which the roadmap did not test. |
| 7 | metrics-assessment: per-token arrays in float16 | Per-sample log-probs in float32 | The fp16 spacing is above BLoB's MI scale (section 5.5). |
| 8 | m4: "report LoRA rows on both ID sets" (no primary) | HackerNews is primary for C3 and C4; StackExchange is secondary | Refuter P1(a): each method's ID set should be its own training domain. |
| 9 | One pre-registered family | F1-F4, with F3 and F4 frozen now for the S2 rows. C4-TFB pre-fix is descriptive. | The rows the paper reports (fixed TFB, refit Laplace) need a family fixed before their scores exist. The pre-fix sampler is known to be wrong. |
| 10 | T3: 1,000 documents per domain | arXiv eligibility also needs the stripped copy to have 257 tokens or more | Both arXiv variants then hold the same 1,000 papers (T3b). |
| 11 | No dev split named | No dev split | No later spec names a consumer (critic Q2). |
| 12 | m4a: 15-30M arXiv tokens | 27-41M arXiv tokens; FreeLaw and PubMed also stream past 10M | Test documents start after each cut. |

### 6.4 Critic items and where this revision answers them

| Critic item | Where |
|---|---|
| T2 pass rule ambiguous; T2 self-graded | S1-T2 row and T2b-T2c; section 5.3 (checker base $P_2$) |
| T1 re-run not runnable; block indices; output overwrite | S1-T1 row; section 5.5 (seeding with full-set $b$); section 5.7 (`--block-ids`, `--run-tag`, file name) |
| T1 tolerance looser than the roadmap | T1d now uses `torch.equal`; section 6.3 row 6 |
| Bootstrap RNG, draw order, percentiles, weighted FPR@95 unspecified | Section 5.6 |
| T4 fixtures under-specified | Section 5.6 fixture table; T4a-T4e reference values |
| T3e pass value; all domains; T3c block check; T3f short stripped papers | T3c, T3e; section 5.3 (stored document tokens); section 5.4 and 6.3 row 10 |
| T1 failure-branch wording and default | Failure-branch table under S1-T1 |
| Seed bases 1 and 2 collide | Seed bases 1,000,000 and 2,000,000; per-set seed offsets (section 5.5) |
| Section 6.1 placeholder; "5-9 papers" | Section 6.1; sections 2, 3 (N2), T3e now say 9 |
| LoRA primary ID; C4-TFB adapter source; family for S2 rows | Section 3 (F1-F4); section 2 labels; section 5.5 meta |
| Costs: S2 move, review figure, network volume | Section 1 notes and decisions D3, D6; section 5.8 |
| Parameters in prose, not YAML | Section 5.2 |
| Missing: batch check, m15 real re-derivation, freeze test, holm and non-null tests, alignment, P11 on real data, stronger unseen check, small caches, meta keys, replication rule, decisions list | T1f; T5j; T5g and T5i; T4d and T4f; T5a and T5k; T1e; T2b; `seen_tensors`; T5a; section 3 replication row; section 1 decisions |
| Wording on `uncertainty.py`; design-rationale doc | Section 2 files table |

## 7. What this does not cover

- The TFB and Laplace fixes, and scoring C2, C4-LAP and fixed TFB on the new set. Both are S2, with families F3 and F4 frozen here.
- A deterministic-LoRA + TFB row, so that TFB is truly training-free (P18). That is S5 or S2, with its own frozen family.
- One-pass baselines, the combined-score test, surface-rate and NLL-matched bins, the MI vs expected-entropy correlation, calibration with CIs (P15), the noise floor for BLoB, and all method-vs-method claims. These are S3.
- The result figure (S4), the LoRA seeds and ensemble (S5), and generated tokens (S6, P7).
- Paper text, README.md and report.md (m3, m11, m13).
- Held-out Wikipedia. The base model trained on all 100M cached Wikipedia tokens. Fresh Wikipedia documents after the cut could be added later at little cost.
- A dev split.
- Whether the Hugging Face mirror still serves the same snapshot and order. S1-T2a reports it. If the mirror is gone, the test set cannot be built. This spec has no fallback source.
- Contamination below the unseen check's reach: paraphrases, or shared spans shorter than half a block. The window-hit rate is reported, not controlled.
- Monte Carlo noise of AUROC at N = 20 beyond the one comparison in S1-T1 and the optional spread re-runs.
- Length bias from the 257-token rule. The rule is reported, and S3 controls for length.
- Rebuilding models from `ckpt["config"]` (K29), a `--ckpt-root` flag (K9), and CI lint on `scripts/`.

## 8. Confidence

| Claim | Confidence | Why |
|---|---|---|
| The old D1 set is 500 StackExchange + 500 arXiv blocks from 9 papers, and the T1 targets are right | high | The refuter confirmed it by a CPU forward pass and by decoding, and the AUROCs recompute from the saved files |
| MI AUROC reproduces within ±0.01 | medium | The old VI, BLoB and MC-dropout draws were unseeded, and Monte Carlo noise at N = 20 is unmeasured. C0 and C4-TFB should match closely. |
| The 200-block re-run is bitwise identical (T1d) | medium-high | Per-block reseeding and the determinism flags should make GPU inference repeatable at batch 1. It has not been run. |
| The Hugging Face stream reproduces the cache | low-medium | It is a community mirror. The `datasets` shuffle depends on the shard list and the library version. The P19 probe moved to session 03 and has not run. |
| The unseen check catches exact and near duplicates without a stream match | high | It tests the tokens themselves against all 11 cached tensors, and every hash hit is confirmed exactly |
| 1,000 eligible test documents per domain can be reached | medium-high for StackExchange, HackerNews, FreeLaw and PubMed; medium for arXiv | arXiv needs 17-31M tokens past the cut, estimated from 9 papers in 128K tokens. Network time is unmeasured. |
| Weighted AUROC with a stratified document bootstrap is the right unit | medium-high | It is a standard cluster bootstrap. The probe shows both the equality and the wider CIs. |
| The T4 thresholds pass as written | high | The probe ran the exact fixtures: ratio 1.59 against a 1.4 threshold; equalities to 1e-15 |
| HackerNews is the right primary ID for C3 and C4 | medium | It is their fine-tuning domain. Their base model also trained on StackExchange, so family F2 keeps that view. |
| GPU 2.0-5.6 h for S1 | low-medium | It uses CA10 batch-1 rates, and blocks per document are estimates. S0 replaces it. |
| S2 re-score 4.1-4.7 GPU h at batch 1 | low-medium | Same rates and block estimates. It is outside S2's roadmap row. |
| Engineering 8.5-20 h | medium-low | It is an allocation from options m4a-m15. The window scan, the stripped copy and the separate re-derivation may push it up. |
| Review 1.0-1.25 h | low | CA1's ratio gives up to 3.3 h at the top of the engineering range |
| G carries information beyond M | unknown | This rests on theory only (metrics-assessment.md section 4.2), which is why it is pre-registered |
