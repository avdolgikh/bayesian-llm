# S2 Post-hoc fixes: TFB rotation and tolerance, scaled diagonal Laplace (`specs/i2-posthoc-fixes.md`)

Status: DRAFT (not approved). Date: 2026-09-26 (revision 1, after one critic pass). Owner: Alexey approves; agents implement.

## 1. Header

| Field | Value |
|---|---|
| Items | **m6**: fix the TFB sampler (SVD rotation, relative tolerance), test it, and refit C4-TFB on held-out ID data. **m7**: scale diagonal Laplace by $N_{\mathrm{seq}}T^2$, and sweep the prior precision $\lambda$ to TFB's relative ΔNLL budget on held-out ID data, for C2 and C4-LAP. |
| Issues | **P4**: diagonal Laplace was never scaled (no $N\cdot T^2$ factor; damping 1.0). **P5**: the TFB sampler skips the SVD rotation and uses an absolute tolerance of 0.1 nats, not 0.3% relative. |
| Approve by | Tue 2026-09-29 (roadmap Section 3, row S2) |
| Build | Session 03, Wed 2026-09-30 |
| Run | Session 04, Mon 2026-10-05, and that night: the refits first, then the re-score of the three new states with S1's scorer. The refits need S1's manifest (re-stream on the night of Thu Oct 1). |
| Depends on | S1 (`specs/i2-eval-rebuild.md`): the manifest with `split: val` rows, the test blocks, the scorer, and families F3 and F4 (interface table in Section 2). S0: the batching gain, which resets the GPU estimate below. |
| Feeds | S1 families F3 and F4: the score sets `c2_refit`, `c4_lap_refit` and `c4_tfb_fixed`. S3-T2: ΔNLL per method for the noise control. S5/b2: the same script with `base_kind: det_lora`. m3/m13: the handoff list in Section 7. |

Costs. The roadmap column is roadmap Section 3, row S2 (m6 plus m7 from options-and-costs.md Section 4.1). The next column is this spec's own estimate.

| Cost | Roadmap row S2 | This spec | Difference and why | Assumption |
|---|---|---|---|---|
| Alexey, approve | in m2 | 0.25 (in m2) | none | CA1 |
| Alexey, review | 0.75 h (m6 0.25; m7 0.5) | 0.75-1.5 h (m6 0.25; m7 0.5-1.25) | up to +0.75 h, because the engineering range is larger (1 h of review per 6-9 engineering h) | CA1 |
| Engineering | 4.5-7.5 h (m6 1.5-2; m7 3-5.5) | 6-9.5 h (m6 2-3; m7 4-6.5) (est.) | +1.5-2 h: sampler versioning, call-site edits and T5 (+0.5-0.75); fit record, own base loader and SHA-256 check (+0.5-0.75); hooking the three new score sets into S1's scorer and drawing v2 Laplace noise on the GPU (+0.5) | CA2 |
| GPU (RTX 4070), S2's own work (m6, m7) | 0.7-1.5 GPU h | 0.8-3.5 GPU h (est.) | The upper end is +2.0 h. The roadmap's 1.5 h top holds only if batching gives at least 1.7x (12 λ-equivalents) or 2.3x (19). S0 measures the gain. | CA3, CA10 |
| GPU, re-score moved from S1 (m4 work; S1 decision D6) | not in row S2 | 1.4-4.7 GPU h; 0.65-2.0 if only the first block of each document is scored | +1.4-4.7 h against row S2. S1 section 5.8 computes it and assigns it here. It holds only if v2 Laplace noise is drawn on the GPU (Section 5.4). | CA10; S1 section 5.8 |
| GPU, S2 run in total | 0.7-1.5 GPU h | 2.2-8.2 GPU h | +1.5-6.7 h | CA3, CA10 |
| Opus agents | 5-8 | 5-8 | none | CA6 |
| Cloud | $0 | $0 | none | |

GPU breakdown of S2's own work (est.). The high end uses CA10 rates at batch 1. Those are 14.35 ms per pass for C2 (C1's 287 ms / 20) and 23.4 ms per pass for TFB and Laplace-LoRA (469 ms / 20). The low end divides the same total by 3, which is the batching gain m4 assumes at its low end. Every row assumes one weight draw per ($\sigma_q$ or $\lambda$, seed), reused for all 640 blocks (Section 5.6). Without that rule, one draw per block would add about 5.3 h to the C2 sweep alone ($153{,}600 \times 0.125$ s for 33.5M normals on the CPU).

| Pass | Count | GPU h at batch 1 (CA10) |
|---|---|---|
| TFB search, C4 | 18 steps × 10 samples × 640 blocks = 115,200 passes | 0.75 |
| TFB ΔNLL re-measure | 20 × 640 = 12,800 passes; 25,600 if the SE rule doubles $S$ | 0.08-0.17 |
| MAP anchors and curvature | 3 × 640 MAP passes; 480 + 960 windows forward and backward | 0.05 or less |
| λ sweep, C2 | 12-19 λ-equivalents × 20 × 640 passes at 14.35 ms | 0.61-0.97 |
| λ sweep, C4-LAP | 12-19 λ-equivalents × 20 × 640 passes at 23.4 ms | 1.00-1.58 |
| **Total** | | **2.5-3.5 at batch 1; 0.8-1.2 at a 3x batching gain** |

| Term | Value |
|---|---|
| λ-equivalents, typical | 12 (8 grid values + about 4 bisection steps) |
| λ-equivalents, worst case | 19 (8 grid + 8 bisection + 1 extension + 2 for one SE doubling) |
| Why batch 1 is an upper bound | CA10 rates include a weight draw per pass. The forward-only rates are 8.1 ms (C0) and 14.8 ms (C3 mean weights) (report.md:90-91). |
| CPU draw cost (measured, session 02) | 0.125 s per draw of 33.5M normals (C2); 0.0021 s per draw of 655,360 (C4) |
| Hidden cost in the re-score | S1's scorer draws per block (20 draws per block). With a CPU generator, `c2_refit` would spend $20 \times 0.125 = 2.5$ s per block on the CPU, or 5.6-6.4 h over its 8,000-9,200 test blocks. Section 5.4 moves the v2 Laplace draw to the GPU to remove this. |
| Mon Oct 5 night | S2 in total (2.2-8.2 GPU h) plus S3 (0.5-1.5) needs 2.7-9.7 GPU h. The roadmap plans 1.2-3.0 GPU h for that night. One night holds 8-10 GPU h (CA3). At the top end, S1's D6 fallback (first block of each document) brings the total to 5.0-7.0 GPU h. |

## 2. Scope and non-scope

What changes:

| File | Change |
|---|---|
| `minigpt/tfb.py` | v2 sampler with the rotation (Section 5.3). New option `epsilon_rel`. `sampler_version` is a required keyword with no default in `TFBState` and in `fit_tfb`. Every search step goes into `search_log` with its sampler version. On the `epsilon_rel` path: a pre-check at `search_max`, and an error if no step is accepted. Save and load keep the new fields. Sampling a `v1_legacy` state emits a `UserWarning`. |
| `minigpt/laplace.py` | `sampler_version` is a required keyword with no default in `LaplaceState`. `fit_laplace` keeps its signature and estimator and returns a `v1_legacy` state (raw $\hat F$, unscaled). New `scale_laplace_state(...)` returns a v2 state. New `posterior_std(state)`, used by the sampler and by T3. Save and load keep the new fields. Sampling a `v1_legacy` state emits a `UserWarning`. |
| `minigpt/posthoc_refit.py` (new, about 200 lines, est.) | `load_posthoc_base(cfg)`, `build_fit_blocks(...)`, `measure_delta_nll(...)`, `match_prior_precision(...)`, `write_fit_record(...)`, and the end-of-run checks (Section 5.5). The three new score-set entries in S1's scorer import the loader, and so does S5/b2. |
| `scripts/refit_posthoc.py` (new, about 40 lines) | CLI with `--config`. Calls the module. Exits non-zero when a check fails. |
| `configs/i2_posthoc_c4_tfb.yaml`, `configs/i2_posthoc_c4_lap.yaml`, `configs/i2_posthoc_c2.yaml` (new) | Every key is explicit (Section 5.5). |
| `experiments/c_pipeline.py:278`, `experiments/b3_post_hoc_lora.py:162` | One keyword each: `sampler_version="v1_legacy"`. Nothing else changes. T5(a) shows the samples stay bitwise equal. |
| `scripts/eval_c_checkpoints.py` (S1's scorer) | Three new score sets, `c2_refit`, `c4_lap_refit` and `c4_tfb_fixed`, in the loader (`_checkpoint_paths` :116-131, `load_model` :153-214). They load the base through `load_posthoc_base` and the state from `data/checkpoints/i2_posthoc/<cell>/`. Nothing else changes, and the existing score sets behave as before. |
| `configs/i2_eval.yaml` (S1's file), `scoring` section only | For the three score sets: `n_samples` 20; add them to `score_sets.test` and, for the two C4 sets, to `score_sets.test_hn`; `labels` (Section 5.5). The frozen `analysis` block is not touched; it already names them in F3 and F4. |
| `tests/test_tfb.py` (extend), `tests/test_laplace.py` (extend), `tests/test_posthoc_refit.py` and `tests/test_posthoc_versions.py` (new) | T1-T5. The existing `fit_tfb` calls in `tests/test_tfb.py` get `sampler_version="v1_legacy"`. `tests/test_laplace.py:77-85` checks the legacy formula and is pinned to `v1_legacy`. |
| `data/checkpoints/i2_posthoc/{c2,c4_tfb,c4_lap}/` (new, gitignored) | New states and `fit_record.json` files |
| AGENTS.md, README.md, `agents/technical-reference.md`, `agents/design-rationale.md` | Doc updates under the rules below. The design-rationale edit also corrects line 25 ("Curvature at convergence is flat"). |

What must not change:

- The saved states and checkpoints in `data/checkpoints/{c0,c2,c3,c4_tfb,c4_lap,b3_lora}/` and `data/checkpoints/laplace_state.pt`. They are read-only (PG11). T3, T5 and S1-T1 read them.
- The curvature estimator in `fit_laplace` (minigpt/laplace.py:69-137).
- The behaviour of `experiments/c_pipeline.py` and `experiments/b3_post_hoc_lora.py`, their configs and their tests. Only the keyword above is added.
- The scorer's behaviour for its existing score sets, its seeding rule (one draw per block, scripts/eval_c_checkpoints.py:243), `scripts/eval_mc_dropout.py`, the eval set and the block builder. S1 owns them. The S2 loader lives in `minigpt/posthoc_refit.py`.
- The `analysis` block of `configs/i2_eval.yaml` (frozen by S1; its sha256 is in `prereg.json`).
- `minigpt/uncertainty.py`, the model, and the BLoB and variational layers.
- `paper/` and report.md (m3, m11, m13). README.md changes only for the test count and the new script.

Interface with S1, as drafted in `specs/i2-eval-rebuild.md` (revision 2). The session-02 consistency check (task 6) must confirm each row.

| S2 needs | What S1 provides |
|---|---|
| Held-out ID documents for the fits | Manifest rows with `split: val`, stored as token ranges only. StackExchange val documents lie fully inside [60M, 80M) and HackerNews val documents fully inside [80M, 90M) of the cached tensors. Documents that straddle a range edge go in no split. |
| No eval document in a fit set | Test documents come from the stream after the 100M cut and pass S1's unseen check against every cached tensor. So a val range cannot hold a test document. T2(c) still checks the IDs at run time. |
| The OOD endpoint for the S2 rows | Families F3 (primary: `c2_refit` on StackExchange; `c4_lap_refit` and `c4_tfb_fixed` on HackerNews; Holm over 9) and F4 (secondary: the two C4 sets on StackExchange; Holm over 6). They are frozen in S1 before any S2 score exists. |
| A scorer that records what was loaded | S1's score-file `meta` holds `checkpoint_sha256` per file, `sampler` and `adapter_source`. S2 sets `sampler: v2` in the labels. |
| The re-score itself | S1 assigns it to S2 (decision D6): S2 runs S1's scorer, unchanged apart from the three loader entries, on S1's `test` and `test_hn` blocks. |

If the re-stream does not match the cache for a domain (S1-T2a, P19), its val documents get no IDs. The fit then draws 640 non-overlapping blocks uniformly from the val range, and the fit record says `fit_units: token_range`.

AGENTS.md rules that apply:

- No extra Bayesian libraries. Only torch, with a manual implementation. No laplace-torch.
- Explicit configs. The script reads every key as `cfg[...]`. It checks the full key list in Section 5.5 before it calls any helper. These helpers have `.get` defaults the script must not rely on: `build_lora_config` (minigpt/config.py:203-212; the rank default is 8, while the C4 configs use 16) and `load_pile_data` (minigpt/data.py:185-192). Also, `experiments/c_pipeline.py:233-236,273-275` break the rule today, and `search_min`/`search_max` are ignored at c_pipeline.py:278-285. Those files are out of scope, so this is recorded here and not fixed.
- LaTeX for every formula in this spec.
- Keep repo docs fresh. When the test count changes, update AGENTS.md and README.md. Add `minigpt/posthoc_refit.py` to the AGENTS.md structure block and the new script to its scripts line. Record the technical change in `agents/technical-reference.md`. Record these decisions in `agents/design-rationale.md`: ΔNLL matching, sampler versioning, the one-draw rule, and withdrawing the 4L rows.
- 4-space indent, 100-character lines, type hints on public APIs. Run `ruff check` and `pytest` before each commit. Conventional Commits. Never mention AI assistants.

Open choice for the approver (the default applies if Alexey says nothing):

| # | Choice | Default | Alternative | Cost of the alternative |
|---|---|---|---|---|
| C1 | The 4L post-hoc rows: B1 (Laplace FFN), B3-TFB and B3-LAP (AG News) | Withdraw them. paper.tex has no 4L rows today. m13 drops them from report.md (:13, :15, :16, :67, :68, :122) and README.md (:39) and says they came from the flawed samplers. No refit. | Refit B1, B3-TFB and B3-LAP with the v2 samplers on a positional AG News validation split | +1.5-2.5 engineering h (est.: AG News data path, 3 more YAMLs, no S1 manifest for AG News). GPU under 0.2 h (est.). +0.25 Alexey h of review. |

The default departs from item 1 of the P5 refuter ("re-run the $\sigma_q$ search for C4-TFB and B3"). The reason: withdrawing a row costs less than refitting it. No v1 claim needs the 4L rows once the MI-ratio "scaling" comparison is dropped (m3).

## 3. Pre-registered endpoint

S2 owns two open results, E1 and E2 below. The OOD endpoint is not one of them. S1 fixes the primary score (the per-block realized-token Jensen gap $G$), the ID sets and the families F3 and F4 for the S2 rows. S3-T2 flags a method whose CI overlaps its noise control. S2 does not restate those rules.

Frozen settings. None of them changes after any refit or OOD score is seen. A later change is reported as post hoc, next to the frozen result.

| Setting | Value | Source |
|---|---|---|
| TFB tolerance | $\epsilon_{\mathrm{rel}} = 0.003$ of $\ell_0$, two-sided: $\lvert\bar\ell-\ell_0\rvert \le \epsilon_{\mathrm{rel}}\,\ell_0$ | TFB paper Appendix E ("0.3% of relative performance change"). Eq. 8 and Algorithm A take the absolute value. This is a deliberate change from roadmap S2 test 2 ("rises by 0.3% or less"): a fall of more than 0.3% is also rejected. |
| TFB samples per search step | $S=10$, seeds 0-9 at every step | configs/c4_tfb.yaml:72; minigpt/tfb.py:123-124 |
| TFB search range | $[10^{-4}, 1]$, linear bisection to a precision of $10^{-5}$ (17 steps), plus a pre-check at 1 | This spec. The old $\sigma_q^\star = 0.030$ lies above the paper's initial range $[0.001, 0.015]$. |
| ΔNLL for the budget | $S=20$, seeds 0-19, on the same 640 blocks | The eval uses $N=20$. |
| Draws | One weight draw per ($\sigma_q$ or $\lambda$, seed), reused for all 640 blocks. The same seeds at every $\sigma_q$ and every $\lambda$. | fit_tfb already reuses seed $s$ for every batch (minigpt/tfb.py:121-124). |
| Fit set | 640 non-overlapping 256-token blocks. Each lies inside one S1 `split: val` document, with at most 3 per document, drawn with `fit_seed` 0. HackerNews 80M-90M for C4; StackExchange 60M-80M for C2. | These ranges are the `val` splits (minigpt/data.py:207-212; S1 section 2) |
| Curvature set | `data["train"]` only. 30 × 32 = 960 windows for C4-LAP and 30 × 16 = 480 for C2, with `curvature_seed` 0. | configs/c4_lap.yaml:44,80; configs/c2.yaml:45,81 |
| $N_{\mathrm{seq}}$, $T$ | 312,500 (C4-LAP) and 625,000 (C2); $T=256$ | 80M/256 and 160M/256 training tokens |
| λ grid | $10^{0},\dots,10^{7}$, one value per decade. That is 8 values over 7 decades, as in roadmap S2 test 4. Then log-bisection of at most 8 steps, which stops at the first value inside the band. One extension to $10^{8}$ if $\Delta\mathrm{NLL}(10^{7})$ is still above the band. | This spec. The median scaled precision is $5.1\times10^{3}$ (C2) and $7.2\times10^{3}$ (C4-LAP). |
| Match band | $\pm10\%$ of $\Delta\mathrm{NLL}^\star_m$ | roadmap S2 test 4 |
| SE rule | If $\mathrm{SE} > 0.05\,\Delta\mathrm{NLL}^\star$ at the chosen value, re-measure there once at $S=40$ (seeds 0-39). There is no second doubling. The same rule applies to the TFB re-measure. | This spec |
| Bases | C2: the C0 best checkpoint (`base_kind: full`). C4-TFB and C4-LAP: the C3 BLoB checkpoint, with `lora_A_mu` mapped to `lora_A` (`base_kind: blob_mean`). | P18 is S5's issue. S2 records the base and labels the rows. |
| Temperature | `sample_scale` = 1, never tuned | |

Primary quantities:

| ID | Quantity | Margin or rule | Used by |
|---|---|---|---|
| E1 | $\sigma_q^\star$ for C4-TFB (v2 sampler, $\epsilon_{\mathrm{rel}}$), and $\rho_{\mathrm{TFB}} = \Delta\mathrm{NLL}_{\mathrm{TFB}}(\sigma_q^\star)/\ell_0^{\mathrm{C4}}$ at $S=20$ | No margin; reported as found. If $\rho_{\mathrm{TFB}} \le 0$, the run exits non-zero, because there is no budget to match. | S3-T2; T4 |
| E2 | $\lambda^\star_m$ and $\Delta\mathrm{NLL}(\lambda^\star_m)$ for $m \in \{\mathrm{C2}, \mathrm{C4\text{-}LAP}\}$, against $\Delta\mathrm{NLL}^\star_m = \rho_{\mathrm{TFB}}\,\ell_0^{m}$ | `match_status`, below | S3-T2; S1 |

Match status. The script writes one status per Laplace model.

| `match_status` | Condition | Exit code | What happens |
|---|---|---|---|
| `matched` | $\lvert\Delta\mathrm{NLL}(\lambda^\star)-\Delta\mathrm{NLL}^\star\rvert \le 0.10\,\Delta\mathrm{NLL}^\star$, with $\mathrm{SE} \le 0.05\,\Delta\mathrm{NLL}^\star$ at $S=20$, or at $S=40$ after one doubling | 0 | The row is scored. |
| `matched_noisy` | Matched at $S=20$. After the one doubling, the $S=40$ SE is still above $0.05\,\Delta\mathrm{NLL}^\star$, or the $S=40$ ΔNLL falls outside the band. | 0 | The row is scored. $\lambda^\star$ is kept. Both ΔNLLs and both SEs are reported. S3 uses the $S=40$ ΔNLL. |
| `tighter_than_budget` | $\Delta\mathrm{NLL}(\lambda=1) < 0.90\,\Delta\mathrm{NLL}^\star$ | 0 | $\lambda^\star = 1$. Both ΔNLLs are reported. No tempering. The row is scored. |
| `not_matched` | $\Delta\mathrm{NLL}(\lambda=10^{8}) > 1.10\,\Delta\mathrm{NLL}^\star$ | 0 | The row is not scored. The paper says so. |
| `bisection_failed` | A bracket exists, but 8 steps end outside the band | non-zero | Nothing is scored. The builder investigates. |

Framing, written in advance. These are drafts for m13; S3's result picks the ending.

- `matched` or `matched_noisy`: "With $N_{\mathrm{seq}}T^2$ scaling and a prior precision chosen so that the sampled models raise held-out ID NLL by the same fraction as TFB ($\rho_{\mathrm{TFB}}$ = <x>), diagonal empirical-Fisher Laplace (FFN weights of C0; LoRA A of the C3 BLoB posterior mean) gives AUROC <a> [<CI>] on <domain>." If the CI contains 0.5 or overlaps the noise control, add: "This is a negative result for this diagonal approximation at this budget. It says nothing about KFAC, GGN or linearized Laplace, which we did not test."
- `tighter_than_budget`: "Even with a negligible prior ($\lambda = 1$), the scaled Laplace posterior raised held-out NLL by only <x>, which is below TFB's budget of <y>. We report it at $\lambda = 1$, without tempering: AUROC <a> [<CI>]."
- `not_matched`: "No prior precision up to $10^{8}$ brought the scaled Laplace posterior inside TFB's budget, so we do not score this row."
- TFB: "The published TFB rows came from a variant that skipped the SVD rotation and used an absolute tolerance of 0.1 nats. That is 2.4% of the anchor NLL, 8 times the paper's 0.3%. It was fitted on the C3 BLoB posterior mean, so this cell is 'BLoB-Mean + TFB', not training-free. The corrected sampler gives $\sigma_q^\star$ = <s> and AUROC <a> [<CI>]."
- In every case, the old "definitive negative" sentence is withdrawn. It rested on a posterior std about 40 times the median weight (Section 6, P4).

## 4. Acceptance tests

| ID | Given | When | Then | Runs in |
|---|---|---|---|---|
| S2-T1 | **Case 1.** `g = torch.Generator().manual_seed(0)`. Draw, in this order: $U$ = the Q of a QR of `randn(64, 8)`, $V$ = the Q of a QR of `randn(8, 8)`, $A_{\mathrm{MAP}}$ = `randn(8, 32)`. Set $d$ = `logspace(0, -0.5, 8)`, $B = U\,\mathrm{diag}(d)\,V^\top$ and $\sigma_q = 0.1$, so $\mathrm{tr}^\star = r\,n\,\sigma_q^2 = 2.56$. **Case 2.** The same $U$ and $A_{\mathrm{MAP}}$, $d$ = `logspace(1, -2, 8)` (10 down to 0.01), and $B = U\,\mathrm{diag}(d)\,R$ with $R$ = `eye(8).flip(0)`, so the SVD returns `Vh` $= R$ up to sign. | `sample_tfb_params` runs on a `TFBState(sampler_version="v2")` built from `torch.linalg.svd(B)`. It draws 10,000 samples in case 1 and 2,000 in case 2, at seeds $0,\dots,S-1$. | (a) In both cases, $\widehat{\mathrm{tr}} = \frac{1}{S}\sum_s \lVert B(A^{(s)}-A_{\mathrm{MAP}})\rVert_F^2$ is within 2% of 2.56, that is, in 2.509-2.611. (b) In both cases, the empirical covariance of the columns of $U^\top B(A^{(s)}-A_{\mathrm{MAP}})/\sigma_q$, over all samples and all $n$ columns, is within 0.05 of $I_r$ in every entry. (c) On the case-1 inputs, the `v1_legacy` sampler gives $\widehat{\mathrm{tr}} > 1.10\,\mathrm{tr}^\star$, and the analytic legacy ratio from the Section 5.1 formula is also above 1.10. Session-02 probe values: (a) 1.0008x and 1.0025x; (b) 0.0048 and 0.0096; (c) 1.601x empirical and 1.600x analytic. | `tests/test_tfb.py` (new test functions, CPU, CI) |
| S2-T2 | A 2-layer fixture: `GPTConfig(n_layer=2, n_head=1, n_embd=32, block_size=16, vocab_size=100)`, `torch.manual_seed(0)`. It is trained for 200 AdamW steps (lr 3e-3, batch 16) on a period-50 token pattern. Then a deterministic LoRA (rank 4, alpha 8, FFN) is injected. Each `lora_B` is set to a non-zero $U\,\mathrm{diag}(d)\,V^\top$ with $U$ and $V$ from QRs of Gaussian draws (generator seed 1) and $d$ = `logspace(0, -0.5, 4)`. `lora_B` starts at zero (minigpt/lora.py:138), and with $B=0$ the noise has no effect. A synthetic manifest in S1's format has 40 documents: 20 with `split: val`, 20 with `split: test`. A YAML with `epsilon_rel: 0.003`, `search_min: 1.0e-4`, `search_max: 1.0`, `search_precision: 1.0e-5`. | `scripts/refit_posthoc.py` runs the TFB search on the fixture, once with `sampler_version: v2` and once with `v1_legacy` | (a) The record logs every step ($\sigma_q$, $\bar\ell$, $\ell_0$, accepted, `sampler_version`). $\sigma_q^\star$ equals the largest accepted $\sigma_q$, and every rejected $\sigma_q$ is above $\sigma_q^\star$ (yes/no). Every step of the v2 run records `v2` (yes/no). (b) The pre-check rejects `search_max` (probe: $\lvert\bar\ell-\ell_0\rvert$ = 4.79 against a tolerance of $1.9\times10^{-4}$). A fixture YAML with `search_max` inside the tolerance raises "search range too small" (yes/no). One where `search_min` already fails raises "no sigma_q accepted" (yes/no). (c) Every block the fit reads lies inside a `val` document, and no fit document ID is also a `test` ID (yes/no). A manifest in which one ID appears in both splits, or a `val` range overlaps a `test` range, makes the script exit non-zero with "eval document in fit set" (yes/no). (d) A YAML without `epsilon_rel`, or without `lora.rank`, fails with KeyError (yes/no). (e) $\lvert\sigma^\star_{q,\mathrm{v2}}-\sigma^\star_{q,\mathrm{legacy}}\rvert > 10 \times$ `search_precision` (probe on this fixture design: 0.00493 against 0.00535). No direction is asserted. The real C4-TFB refit runs checks (a)-(c) at the end and exits non-zero if any fails. | `tests/test_posthoc_refit.py` (CPU, CI); end-of-run checks in `scripts/refit_posthoc.py` |
| S2-T3 | The saved `data/checkpoints/c4_lap/laplace_state.pt` (655,360 LoRA-A entries), with $N_{\mathrm{seq}}=312{,}500$ and $T=256$. The saved `data/checkpoints/c2/laplace_state.pt` (33,554,432 FFN entries), with $N_{\mathrm{seq}}=625{,}000$. A synthetic state. | `scale_laplace_state(...)` builds a v2 state with $\lambda=0$, and `posterior_std` returns $\sigma_j=\tau_j^{-1/2}$ with $\tau_j = N_{\mathrm{seq}}T^2\hat F_j+\lambda$. $\lambda=0$ is valid because neither state has a zero curvature entry (session-02 probe). No sampling is done. | (a) The median over all entries of $\sigma_j$ is $0.0118 \pm 0.002$ for C4-LAP and $0.0140 \pm 0.002$ for C2. Use `torch.median` on the flat tensor; `torch.quantile` fails above $2^{24}$ entries. (b) For the `v1_legacy` state of each file, the median of $(\hat F_j+1)^{-1/2}$ is $1.00 \pm 0.01$. (c) A synthetic state with $\hat F=10^{-6}$, $N_{\mathrm{seq}}=1{,}000$, $T=4$ and $\lambda=2$ gives $\sigma = 2.016^{-1/2} = 0.704295$ to within $10^{-6}$. | `tests/test_laplace.py`. Part (c) runs in CI. Parts (a) and (b) skip when `data/` is absent, and fail instead when `REQUIRE_POSTHOC_DATA=1` (pre-merge command below). |
| S2-T4 | (i) `match_prior_precision` with synthetic ΔNLL functions. (ii) The T2 fixture model with a v2 Laplace state (end to end). (iii) The real v2 states for C2 (C0 base) and C4-LAP (C3 mean base), each from a curvature refit, with targets $\Delta\mathrm{NLL}^\star_m=\rho_{\mathrm{TFB}}\,\ell_0^{m}$ and $\rho_{\mathrm{TFB}}$ from the C4-TFB record. | The sweep runs over the grid, bisects, and applies the SE rule | (a) With $\Delta\mathrm{NLL}(\lambda)=1/(1+\lambda/100)$ and SE = 0: target 0.05 gives `matched`, within ±10%. Target 2.0 gives `tighter_than_budget` with $\lambda^\star=1$. Target $10^{-7}$ gives `not_matched`. All three exit 0. A step function that jumps from twice the target to half the target between two bisection points gives `bisection_failed` and a non-zero exit (yes/no each). (b) With SE = 0.08 × target at $S=20$ and 0.03 × target at $S=40$: `matched`, and exactly one $S=40$ evaluation. With SE = 0.08 × target at both: `matched_noisy`, exit 0, and no second doubling (yes/no each). (c) The record holds each of these fields (yes/no each): base path; base kind; SHA-256 of the bytes loaded, which equals the SHA-256 of `base_checkpoint` and differs from that of a decoy checkpoint in the same folder; the weights equal the file's weights; $N_{\mathrm{seq}}$; $T$; every λ tried; ΔNLL and SE per λ; $\ell_{\mathrm{BMA}}$ per λ (log only); `match_status`; a hash of the fit-set document IDs; seeds; git commit; `sampler_version`; the median $\hat F$. A `blob_mean` checkpoint with one `lora_A_mu` key removed raises KeyError (yes/no). (d) Both real runs exit 0 with a status other than `bisection_failed`. Each refit's median $\hat F$ is within 0.5x-2x of the saved state's median ($3.5\times10^{-7}$ for C4-LAP; $1.24\times10^{-7}$ for C2). After the re-score, each of the three score files has a `checkpoint_sha256` for the base that equals the fit record's SHA-256 (yes/no each). | `tests/test_posthoc_refit.py` for (a)-(c) (CPU, CI); the real runs of `scripts/refit_posthoc.py` for (d), with its last check run by `scripts/refit_posthoc.py --check-scores` on the three S1 score files |
| S2-T5 | The three pre-fix files (`c2/laplace_state.pt`, `c4_lap/laplace_state.pt`, `c4_tfb/tfb_state.pt`), synthetic legacy fixtures, and one v2 state of each kind from the T2 and T4 fixture runs. The test module holds a verbatim copy of `sample_tfb_params` (minigpt/tfb.py:160-194) and `sample_laplace_params` (minigpt/laplace.py:140-173) from commit a7a73b6 as the reference. | The states are loaded with `load_tfb_state` / `load_laplace_state` and sampled on CPU at seeds 0-4 | (a) The pre-fix files load with `sampler_version == "v1_legacy"`. Their samples equal the reference copies with `torch.equal` for every tensor. So S1-T1 can still reproduce 0.917 for C4-TFB. (b) The v2 states keep every new field through a save and a load, with equal values. (c) Sampling a `v1_legacy` state, TFB or Laplace, emits a `UserWarning` (yes/no each). (d) Every state written by the T2 and T4 fixture runs reads back with `sampler_version == "v2"` (yes/no). (e) Passing both `epsilon` and `epsilon_rel`, or neither, raises ValueError (yes/no). (f) Sampling a v2 Laplace state with `sample_scale != 1.0` raises ValueError (yes/no). (g) Building `TFBState` or `LaplaceState`, or calling `fit_tfb`, without `sampler_version` raises TypeError (yes/no). | `tests/test_posthoc_versions.py`. The synthetic parts run in CI. The saved-file part of (a) skips when `data/` is absent, and fails instead when `REQUIRE_POSTHOC_DATA=1`. |

Pre-merge command. The builder runs it on the machine that holds `data/`:

```bash
REQUIRE_POSTHOC_DATA=1 uv run pytest tests/test_tfb.py tests/test_laplace.py tests/test_posthoc_refit.py tests/test_posthoc_versions.py -rs
```

It passes when the exit code is 0 and the summary line has no "skipped". Without the variable, as in CI, the saved-state parts skip.

For C2 there is no TFB fit, because C0 has no LoRA. The roadmap's "TFB's ΔNLL on the same data" is read here as TFB's relative rise $\rho_{\mathrm{TFB}}$ applied to C0's own held-out NLL.

## 5. Implementation notes for the builder

### 5.1 TFB formulas (Shi et al. 2024, arXiv 2412.05723, Eqs. 4-8, Theorem 4.1, Algorithm A)

The LoRA update is $\Delta W = s\,BA$ with $s=\alpha/r$ (minigpt/lora.py:135,146). The formulas below drop $s$.

$$B = U\,\mathrm{diag}(d)\,V^\top,\qquad U\in\mathbb{R}^{m\times r},\ V\in\mathbb{R}^{r\times r},\ d\in\mathbb{R}^{r}_{>0}\qquad\text{(Eq. 4)}$$

$$B' = U\,\mathrm{diag}(d),\qquad A' = V^\top A\qquad\text{(Eq. 5)}$$

$$q(A'_{ij}) = \mathcal{N}\!\left(A'_{ij}\mid M_{ij},\,\Omega_{ij}^2\right),\qquad M = V^\top A_{\mathrm{MAP}},\qquad \Omega_{ij} = \sigma_q/d_i\qquad\text{(Eqs. 6-7)}$$

The same distribution in the stored $A$ coordinates. `torch.linalg.svd` returns `Vh` $=V^\top$, so $V$ = `Vh.T`:

$$A^{(s)} = A_{\mathrm{MAP}} + V\,\mathrm{diag}(\sigma_q/d)\,E^{(s)},\qquad E^{(s)}_{ij}\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,1)$$

$$B\left(A^{(s)}-A_{\mathrm{MAP}}\right) = \sigma_q\,U\,E^{(s)},\qquad \mathbb{E}\left\lVert B\left(A^{(s)}-A_{\mathrm{MAP}}\right)\right\rVert_F^2 = r\,n\,\sigma_q^2$$

This matches Theorem 4.1: $\Sigma_q = \sigma_q^2\, I_n\otimes \mathrm{diag}(I_r, 0_{m-r})$ in the $U$ basis.

The legacy sampler (minigpt/tfb.py:188-192) draws $A^{(s)} = A_{\mathrm{MAP}} + \mathrm{diag}(\sigma_q/d)\,E^{(s)}$. That gives

$$\mathbb{E}\lVert B(A^{(s)}-A_{\mathrm{MAP}})\rVert_F^2 = n\,\sigma_q^2\sum_{i,j}P_{ij}\,\frac{d_i^2}{d_j^2},\qquad P_{ij} = (V^\top)_{ij}^2 .$$

$P$ is doubly stochastic. By the Birkhoff-von Neumann theorem and the AM-GM inequality, $\sum_{i,j}P_{ij}\,d_i^2/d_j^2 \ge r$. Equality holds if and only if $d_i = d_j$ wherever $(V^\top)_{ij}\neq 0$. So the legacy trace is never below the target. This matches the refuter's result that all 1,000 random draws were above 1.

The search (Eq. 8, Algorithm A):

$$\sigma_q^\star = \max\ \sigma_q\quad\text{s.t.}\quad \left\lvert\bar\ell(\mathcal{D}_{\mathrm{fit}};\sigma_q)-\ell_0\right\rvert \le \epsilon_{\mathrm{rel}}\,\ell_0,\qquad \epsilon_{\mathrm{rel}}=0.003$$

$$\ell(\mathcal{D}\mid\theta) = -\frac{1}{\lvert\mathcal{D}\rvert\,T}\sum_{n\in\mathcal{D}}\sum_{t=1}^{T}\log p_\theta(x_{n,t}\mid x_{n,<t}),\qquad \ell_0=\ell(\mathcal{D}_{\mathrm{fit}}\mid\theta_{\mathrm{MAP}}),\qquad \bar\ell(\mathcal{D};\sigma_q)=\frac{1}{S}\sum_{s=1}^{S}\ell\!\left(\mathcal{D}\mid\theta^{(s)}_{\sigma_q}\right)$$

### 5.2 Laplace formulas (Yang et al. 2023, arXiv 2308.13111, Eqs. 7-9)

$$p(\theta\mid\mathcal{D})\approx\mathcal{N}(\theta_{\mathrm{MAP}},\Sigma),\qquad \Sigma = \left(-\nabla^2_\theta\log p(\mathcal{D}\mid\theta)\big|_{\theta_{\mathrm{MAP}}}+\lambda I\right)^{-1}\approx (F+\lambda I)^{-1}$$

This spec uses a diagonal approximation with the empirical Fisher (observed tokens), one term per training sequence:

$$\sigma_j^{-2}=\lambda+\sum_{n=1}^{N_{\mathrm{seq}}}g_{n,j}^2,\qquad g_n=\nabla_\theta\log p_\theta(x_n)=-T\,\nabla_\theta\bar\ell_n,\qquad \bar\ell_n=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_{n,t}\mid x_{n,<t})$$

The code averages the squared gradients of the mean-token loss over $M$ sampled windows (minigpt/laplace.py:104-123; the mean reduction is at minigpt/model.py:169):

$$\hat F_j=\frac{1}{M}\sum_{m=1}^{M}\left(\frac{\partial\bar\ell_m}{\partial\theta_j}\right)^2\ \Longrightarrow\ \sum_{n=1}^{N_{\mathrm{seq}}}g_{n,j}^2\approx N_{\mathrm{seq}}\,T^2\,\hat F_j$$

$$\tau_j=N_{\mathrm{seq}}\,T^2\,\hat F_j+\lambda,\qquad \sigma_j=\tau_j^{-1/2}$$

The legacy sampler (minigpt/laplace.py:167-168) uses $\sigma_j=(\hat F_j+1)^{-1/2}\approx 1$. Check: $3.5\times10^{-7}\cdot312{,}500\cdot65{,}536\approx7.2\times10^{3}$, so $\sigma\approx0.0118$.

The budget match:

$$\Delta\mathrm{NLL}(\psi)=\bar\ell(\mathcal{D}_{\mathrm{fit}};\psi)-\ell_0,\qquad \rho_{\mathrm{TFB}}=\frac{\Delta\mathrm{NLL}_{\mathrm{TFB}}(\sigma_q^\star)}{\ell_0^{\mathrm{C4}}},\qquad \Delta\mathrm{NLL}^\star_m=\rho_{\mathrm{TFB}}\,\ell_0^{m}$$

$$\mathrm{SE}=\frac{\mathrm{sd}_s\,\ell\!\left(\mathcal{D}\mid\theta^{(s)}\right)}{\sqrt{S}},\qquad \ell_{\mathrm{BMA}}(\lambda)=-\frac{1}{\lvert\mathcal{D}\rvert T}\sum_{n,t}\log\!\left(\frac{1}{S}\sum_{s=1}^{S}p_{\theta^{(s)}}(x_{n,t}\mid x_{n,<t})\right)$$

$\ell_{\mathrm{BMA}}$ is Laplace-LoRA's validation criterion. Here it is logged only. No $\arg\min_\lambda$ is reported, and it plays no part in the choice of $\lambda$. It needs only the realized-token probability of each sample.

For C4-LAP the target equals $\Delta\mathrm{NLL}_{\mathrm{TFB}}$ itself, because it has the same base and the same fit set as C4-TFB.

### 5.3 TFB changes (m6)

1. At minigpt/tfb.py:73-74, rename to `U, S, Vh = torch.linalg.svd(...)` and cache `(U, S, Vh)`. Fix the comment at :23.
2. At minigpt/tfb.py:188-192, the v2 branch is `a_map + Vh.T @ (std * eps.to(a_map.device))` with `std = (sigma_q / S.clamp(min=1e-6)).unsqueeze(1)`. Keep the current line exactly as it is as the `v1_legacy` branch. Keep the RNG call order: one `torch.randn(a_map.shape, generator=gen, dtype=a_map.dtype)` per name, in `param_names` order, from one generator seeded once. An unknown `sampler_version` raises ValueError.
3. At minigpt/tfb.py:131-136, when `epsilon_rel` is given, accept a step if `abs(avg_noisy_loss - anchor_loss) <= epsilon_rel * anchor_loss`. Exactly one of `epsilon` and `epsilon_rel` must be passed; both or neither raises ValueError. The C pipeline keeps passing `epsilon`.
4. `TFBState` (minigpt/tfb.py:19-27) gets `sampler_version: str` as a keyword-only field with no default (`field(kw_only=True)`; Python is 3.11 or later), plus `epsilon_rel: float | None` and `search_log: list[dict]`. `epsilon` becomes `float | None`. Save and load (:197-219) keep them. A file without `sampler_version` loads as `v1_legacy`. A file with an unknown value raises.
5. `fit_tfb` gets `sampler_version` as a required keyword. The temporary state at :111-118 passes it on, so the search itself runs the chosen sampler. Each step appends `{sigma_q, avg_loss, anchor_loss, delta, accepted, sampler_version}` to `search_log`.
6. `fit_tfb` takes optional `anchor_batches`. When they are given, it skips the random draw at :84-89. The TFB re-measure and the Laplace sweep then use exactly the same 640 blocks.
7. On the `epsilon_rel` path only, so that the legacy callers keep their behaviour: evaluate `search_max` before the loop and raise "search range too small" if it passes. After the loop, raise "no sigma_q accepted" if no step passed.
8. `sample_tfb_params` on a `v1_legacy` state emits `warnings.warn(..., UserWarning)`.
9. After the search, the refit re-measures $\Delta\mathrm{NLL}_{\mathrm{TFB}}(\sigma_q^\star)$ at $S=20$ (seeds 0-19), applies the SE rule, and writes the value to the fit record. It is the target in T4 and the budget in S3-T2.

### 5.4 Laplace changes (m7)

1. `LaplaceState` (minigpt/laplace.py:19-26) gets `sampler_version: str` as a keyword-only field with no default, plus `n_data_seqs: int | None`, `tokens_per_seq: int | None` and `prior_prec: float | None`. Save and load (:213-233) keep them. A file without these fields loads as `v1_legacy`.
2. `fit_laplace` keeps its signature and estimator (:69-130). Its return at :131-137 adds `sampler_version="v1_legacy"`, because a raw $\hat F$ is unscaled.
3. Add `scale_laplace_state(state, *, n_data_seqs, tokens_per_seq, prior_prec) -> LaplaceState`. It returns a v2 state and sets `damping` to 0.0, which v2 ignores.
4. Add `posterior_std(state) -> dict[str, torch.Tensor]`. For `v1_legacy` it keeps the exact legacy expression, `(1.0 / (curvature + damping)).sqrt() * sample_scale`, so that T5(a) stays bitwise equal. For v2 it returns `(n_data_seqs * tokens_per_seq**2 * curvature + prior_prec).rsqrt()`, and raises ValueError if `sample_scale != 1.0`.
5. `sample_laplace_params` (:140-173) uses `posterior_std`. On a `v1_legacy` state it keeps the CPU generator and the RNG call order, and emits a `UserWarning`. On a v2 state it draws the noise on the parameter's device, from `torch.Generator(device=phi.device)` seeded with the same seed. A C2 draw on the CPU takes 0.125 s. S1's scorer draws 20 times per block, so a CPU draw would add 5.6-6.4 h to the `c2_refit` re-score (Section 1). The v2 samples then depend on the device type. That is acceptable, because only the legacy path must be bitwise stable (T5a). Update the docstring at :146.
6. $N_{\mathrm{seq}}$ goes in YAML: `n_data_seqs: 312500` for C4-LAP (80M HackerNews training tokens / 256) and `625000` for C2 (160M / 256). The script asserts `n_data_seqs == len(data["train"]) // T` (minigpt/data.py:207-212).
7. The curvature pass reads only `data["train"]`. It runs in fp32, without autocast. It calls `torch.manual_seed(curvature_seed)` before `fit_laplace`, because `get_batch` uses the global RNG. It uses `n_curvature_batches` × `curvature_batch_size` from YAML, which are 30 × 32 = 960 windows for C4-LAP and 30 × 16 = 480 for C2. The script asserts that no eval document's token range lies inside the train slice.

### 5.5 Refit module, script and configs

`scripts/refit_posthoc.py --config configs/i2_posthoc_<cell>.yaml` calls `minigpt/posthoc_refit.py`.

Base loading does not reuse `_blob_to_deterministic_lora` (scripts/eval_c_checkpoints.py:133-150). That function always loads `c3/ckpt_best.pt` (:137), takes no path, and silently keeps initial weights for missing keys (:146, :148). The new `load_posthoc_base(cfg)` works like this:

| `base_kind` | Used by | Load rule |
|---|---|---|
| `full` | C2 (C0 checkpoint) | Build MiniGPT from the YAML model keys. `load_state_dict(strict=True)`. |
| `blob_mean` | C4-TFB, C4-LAP (C3 checkpoint) | Inject a deterministic LoRA. Map `.lora_A` from `.lora_A_mu`. Every other target key must be in the checkpoint. Strict load. The dropped `lora_A_g` keys are listed in the record. |
| `det_lora` | S5/b2 | Inject a deterministic LoRA. Keys map one to one. Strict load. |

A missing key raises KeyError. The loader reads the file bytes once, hashes them with SHA-256, and then calls `torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)`. So the recorded hash is the hash of the file that was loaded.

Required YAML keys. The script checks every key before it calls any helper.

| Section | Keys |
|---|---|
| Top level | `method` (`tfb` or `laplace`), `cell`, `out_dir`, `device`, `autocast` (fp16 on CUDA for the ΔNLL passes, as in the scorer, scripts/eval_c_checkpoints.py:246-248; $\ell_0$ runs under the same autocast) |
| `base` | `base_checkpoint`, `base_kind` |
| `model` | `block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias` |
| `lora` (for `blob_mean` and `det_lora`) | `rank`, `alpha`, `target` (16, 32.0 and `ffn` for C4, configs/c4_lap.yaml:83-85) |
| `data` | `dataset`, `pile_id_domains`, `pile_ood_domains`, `pile_id_tokens`, `pile_ood_tokens`, `val_fraction`, `test_fraction`, `data_seed` (the Pile shuffle seed, 1337, passed to `load_pile_data` as `cfg["train"]["seed"]`) |
| `fit` | `manifest_path` (S1's `data/eval_i2/manifest.jsonl`), `fit_split: val`, `fit_domain`, `n_fit_blocks: 640`, `max_blocks_per_doc: 3`, `fit_seed: 0`, `batch_size: 32` |
| `tfb` | `epsilon_rel`, `n_search_samples`, `search_min`, `search_max`, `search_precision`, `n_delta_samples: 20`, `se_threshold: 0.05`, `se_max_doublings: 1` |
| `laplace` | `selection_mode`, `n_curvature_batches`, `curvature_batch_size`, `curvature_seed`, `n_data_seqs`, `tokens_per_seq`, `prior_prec_grid`, `grid_extension: [1.0e8]`, `bisect_max_steps: 8`, `match_tol: 0.10`, `tfb_record_path`, `n_delta_samples: 20`, `se_threshold: 0.05`, `se_max_doublings: 1` |

`match_prior_precision(delta_fn, target, grid, extension, bisect_max_steps, match_tol, se_threshold)` is a pure function. `delta_fn(lam, n_samples)` returns (ΔNLL, SE). T4(a)-(b) test it with synthetic functions. Log-bisection uses $\lambda_{\mathrm{mid}}=\sqrt{\lambda_{\mathrm{lo}}\lambda_{\mathrm{hi}}}$. If ΔNLL is not monotone on the grid, the function takes the first bracket from the small-λ side and records `monotone: false`.

Outputs go to `data/checkpoints/i2_posthoc/<cell>/`. The script never writes into `data/checkpoints/{c2,c4_tfb,c4_lap}/`. S5/b2 runs the same script with `base_kind: det_lora` and one YAML per seed, with outputs under `data/checkpoints/i2_posthoc/b2/`.

`--check-scores` reads the three S1 score files and compares their `meta.checkpoint_sha256` for the base with the fit records (T4d).

Entries for the `scoring` section of `configs/i2_eval.yaml`:

| Score set | `display` | `sampler` | `adapter_source` | Blocks |
|---|---|---|---|---|
| `c2_refit` | "C2 diag. Laplace FFN (scaled)" | `v2` | `none` | `test` |
| `c4_lap_refit` | "BLoB-mean + diag. Laplace LoRA-A (scaled)" | `v2` | `blob_mean` | `test`, `test_hn` |
| `c4_tfb_fixed` | "BLoB-mean + TFB (fixed)" | `v2` | `blob_mean` | `test`, `test_hn` |

### 5.6 Draws and loop order

- For each $\sigma_q$ (TFB) or $\lambda$ (Laplace), and each seed $s$: draw the weights once, apply them with `apply_sampled_params`, run all 640 blocks in batches of `batch_size`, then restore.
- The TFB search may keep `fit_tfb`'s batch-outer loop. A seeded draw of 655,360 entries costs milliseconds and gives the same weights for every batch.
- The Laplace sweep must loop sample-outer. One C2 draw is 33.5M normals, which takes 0.125 s on the CPU.
- The same seeds $0,\dots,S-1$ are used at every $\sigma_q$ and every $\lambda$. With fixed noise draws, ΔNLL changes smoothly with $\lambda$, which keeps the bisection stable.
- The eval scorer draws per block (seed = seq_idx × N + s, scripts/eval_c_checkpoints.py:243). That is S1's rule and does not change here.

### 5.7 Numbers checked while writing this spec

Session 02, CPU only. Scripts in the scratchpad: `s2_probe_posthoc.py`, `s2_probe_adversarial.py` (the first draft), and `s2r_fixture_probe.py`, `s2r_search_probe_c.py` (this revision). They import the real `sample_tfb_params`.

| Quantity | Value |
|---|---|
| Toy trace (64×8×32, $\sigma_q=0.1$, random B, 10,000 samples): legacy / fixed / target | 3.061 / 2.562 / 2.560 |
| Toy with `Vh` equal to the reversal (d from 10 to 0.01, 2,000 samples): legacy / fixed / target | $3.27\times10^{5}$ / 2.566 / 2.560 (from `s2_probe_adversarial.py`) |
| Saved `c4_tfb`: $\sigma_q$; $\epsilon$; anchor loss | 0.0300; 0.1 absolute; 4.173 nats on training windows. So the tolerance was 2.4% relative; 0.3% would be 0.0125 nats. |
| Saved `c4_tfb`: legacy trace / target per layer (32 layers) | min 1.063, median 1.150, max 2.020; pooled 1.287 |
| Singular values of C4's $B$ | 0.249-1.788, median 0.820 |
| `c4_lap`: $\hat F$ median; median $\lvert\phi\rvert$; legacy std; scaled std q10/q50/q90 at $\lambda=0$ | $3.5\times10^{-7}$; 0.0233; 1.00; 0.0066 / 0.0118 / 0.0309 |
| `c2`: the same quantities | $1.24\times10^{-7}$; 0.0247; 1.00; 0.0085 / 0.0140 / 0.0251 |
| Median scaled precision $N_{\mathrm{seq}}T^2\hat F$: C4-LAP; C2 | $7.2\times10^{3}$; $5.1\times10^{3}$ |
| C4-LAP median std at $\lambda=10^{4}$ / $10^{5}$ / $10^{6}$ | 0.0076 / 0.0031 / 0.0010 |
| Zero curvature entries | 0 in both states |

## 6. Verification evidence from session 02

Two refuters tried to break P4 and P5. Both claims were **confirmed**. A third refuter's D1 verdict applies in part. The revision probes pinned the test fixtures.

### P4 (Laplace scale): confirmed

| Evidence | File:line | Finding |
|---|---|---|
| Fisher is a mean, not a sum | minigpt/laplace.py:112-123 | Squared per-sequence gradients are added, then divided by `total_samples`. No data-size factor. |
| Extra $1/T^2$ | minigpt/model.py:169 | The loss is token-mean cross-entropy, so each $\hat F$ entry carries $1/T^2$ compared with the summed log-likelihood. |
| Std formula | minigpt/laplace.py:167-168 | variance = 1/(curvature + damping); std = sqrt(variance) × sample_scale |
| Damping fixed at 1, never tuned | configs/c2.yaml:79-81; configs/c4_lap.yaml:78-80; experiments/c_milestones.py:64-69,130-135,307-309 | damping 1.0, sample_scale 1.0, 30 curvature batches. `max_runs_for` returns 1 for c2 and c4_lap. |
| Fit size | experiments/c_pipeline.py:244-252 | 480 sequences (C2) and 960 (C4-LAP) |
| Saved states | data/checkpoints/c2/laplace_state.pt; data/checkpoints/c4_lap/laplace_state.pt (CPU) | See the numbers table below |
| Near-uniform predictions | report.md:39,42,45; data/d1_scores.pt | NLL 9.10 (C2) and 9.73 (C4-LAP), against 2.79 for C0 |
| Same pattern at 4L | configs/b1_laplace_agnews.yaml:50; configs/b3_lora_agnews.yaml:64; data/checkpoints/laplace_state.pt; data/checkpoints/b3_lora/laplace_state_lora.pt | std median 0.99999; median $\lvert w\rvert$ 0.019 (B1) and 0.031 (B3-LAP) |
| Paper reads it as a real result | paper/paper.tex:49,93,155,310; agents/design-rationale.md:25 | "flat curvature", "definitive negative result", and $(\mathrm{diag}(F)+\lambda I)^{-1}$ with no $N$ scaling |

| Number | C2 | C4-LAP |
|---|---|---|
| Entries | 33,554,432 | 655,360 |
| $\hat F$ median / max | $1.244\times10^{-7}$ / $2.336\times10^{-3}$ | $3.495\times10^{-7}$ / $1.775\times10^{-4}$ |
| Entries with $\hat F$ > damping | 0 | 0 |
| Stored std median (min) | 1.000000 (0.998834) | 1.000000 (0.999911) |
| Median $\lvert\phi\rvert$ | 0.02471 | 0.02332 |
| Noise-to-signal energy $\sum\sigma^2/\sum w^2$ | 686 | 694 |
| Median std, $N_{\mathrm{seq}}T\hat F+1$ | 0.219 (still too wide) | 0.186 (still too wide) |
| Median std, $N_{\mathrm{seq}}T^2\hat F+\lambda$, any $\lambda\le1$ | 0.0140 | 0.0118 |
| Mean predictive entropy (uniform = 10.82) | 9.40 | 9.16 |
| MI ratio OOD/ID from the D1 scores | 1.014 | 0.997 |
| CPU probe (6 ID + 6 OOD sequences, 8 samples): MAP NLL; stored NLL of $\bar p$; per-sample NLL | 2.350; 10.54; 15.25 | 2.603; 10.27; not reported |
| CPU probe, $N_{\mathrm{seq}}T^2$ scaling: NLL at $\lambda$ = 1 / 100 / $10^4$ | 9.00 / 2.43 / 2.36 (MI ratio 1.15 and 1.23 at the last two) | 2.606 at $\lambda=1$ (MI 0.0049 nats) |

Wording for the paper (P4 item 5): the MC-averaged prediction is near uniform (entropy 9.4 of 10.8). Single samples have NLL 15-16, which is worse than uniform. So the samples are confidently wrong, not uniform. The probe's MI ratios come from 6 sequences and are indicative only.

What S2 takes from P4: the $N_{\mathrm{seq}}T^2$ factor (not $N_{\mathrm{seq}}T$), a λ sweep that is required rather than optional (C2 needs $\lambda \ge$ about $10^2$), T3's medians 0.0118 and 0.0140, and the handoff list in Section 7. The erratum must cover all four Laplace rows (B1, B3-LAP, C2, C4-LAP). Under choice C1 the 4L rows are withdrawn.

### P5 (TFB rotation): confirmed

| Evidence | File:line | Finding |
|---|---|---|
| The stored "V" is `Vh` | minigpt/tfb.py:23,73-74 | On the real C4 state, $U\,\mathrm{diag}(S)\,V$ rebuilds C3's `lora_B` with relative error at most $3.4\times10^{-6}$ on all 32 layers; $U\,\mathrm{diag}(S)\,V^\top$ gives 1.28 or more. |
| Rotation never used | minigpt/tfb.py:178,188-192 | `V` is unpacked but unused. The sample is `a_map + (sigma_q/S)[:,None] * eps`. The noise on $A$ is row-scaled, not applied to $A' = V^\top A$. |
| B never swapped | minigpt/laplace.py:186-210 | `apply_sampled_params` replaces only `lora_A`, so nothing undoes the missing rotation. |
| Test cannot see it | tests/test_tfb.py:71-72 | The test sets $V = I$. With $V=I$ the legacy sampler matches the target (ratio 1.001). |
| Same sampler in search and eval | minigpt/tfb.py:124; scripts/eval_c_checkpoints.py:245 | The saved $\sigma_q$ was tuned on the wrong covariance shape, so the search must be re-run, not only the sampling. |
| C4-TFB base is BLoB | experiments/c_pipeline.py:107-110,123-159; .pipeline-state/c4_tfb.json | The largest absolute difference between `a_map` and C3's `lora_A_mu` is 0 on all 32 layers, and B equals C3's `lora_B`. C4-TFB is "BLoB-Mean + TFB". |
| B3 has the same flaw | experiments/b3_post_hoc_lora.py:361-385 | B3 is a true deterministic LoRA + TFB, but its saved state also has a trace ratio of 1.072-1.173. |
| Paper claim | paper/paper.tex:51,75,94,159,326 | "zero Bayesian training" and "no Bayesian training" for TFB |

| Number | Value |
|---|---|
| Target trace, random B (64×8, A 8×32, $\sigma_q$ 0.1) | 2.56 |
| Legacy sampler, 10,000 draws: seed 0; seed 1 | 2.922 (1.141x); 3.357 (1.311x) |
| Legacy analytic ratio over 1,000 random B | min 1.048, median 1.154, max 1.425; all above 1 |
| Rotated sampler | 2.564 (1.001x); eigenvalues / $\sigma^2$ 0.993-1.008 |
| Real C4-TFB: $\sigma_q$; per-layer trace ratio | 0.030007; 1.063-2.020 (median 1.150) |
| Real C4-TFB: per-direction variance / $\sigma_q^2$ inside col(B) | about 0.1x to 8.95x; the target is 1 |
| Real B3 (8 layers): trace ratio | 1.072-1.173 (median 1.115) |

What S2 takes from P5: the fix `a_map + Vh.T @ (std * eps)`; a random-orthogonal test (T1); a refit of the search, not a re-sample; the label "BLoB-Mean + TFB" for C4; and the base path in YAML. The effect on AUROC 0.917 is unknown until the GPU re-score, so this spec predicts no direction. The B3 refit that P5 asks for is replaced by withdrawal under choice C1 (Section 2).

### D1 (eval data): confirmed; applies to S2 in part

| Evidence | File:line | What S2 uses |
|---|---|---|
| The eval always uses C0's split | scripts/eval_c_checkpoints.py:96-109; minigpt/data.py:202,207-213 | The old ID set was StackExchange 80M-100M for every method. S1 names per-method ID sets; Section 2 lists the ones S2 needs. |
| The base path is hardcoded | scripts/eval_c_checkpoints.py:116-131,133-150 | S2 has its own loader with a path, a kind and a SHA-256 (Section 5.5). |
| No document boundaries in the caches | minigpt/data.py:162-166,177 | The fit set needs S1's manifest; the P19 fallback is in Section 2. |

### Revision probes (this pass; CPU only)

| Probe | Result | Used in |
|---|---|---|
| T1 case 1 (seed 0, d = logspace(0, -0.5, 8)); max abs(V - I) = 1.337 | legacy 4.097 (1.601x; analytic 1.600x); v2 2.5621 (1.0008x); covariance max deviation 0.0048 | T1(a)-(c) |
| T1 case 1 with d = logspace(0, -1, 8) | legacy 4.366x (analytic 4.369x) | Shows the threshold 1.10 is not borderline |
| T1 case 2 (Vh = reversal) | legacy $3.267\times10^{5}$ (analytic ratio $1.275\times10^{5}$); v2 2.5665 (1.0025x); covariance max deviation 0.0096 | T1(a)-(b) |
| The first draft's other reversed-order toy (`s2_probe_posthoc.py`: ascending d, then the reversal) | The SVD re-sorts it to Vh = I, so the legacy ratio is exactly 1. It does not test the bug. | The T1 case-2 definition |
| Untrained 2-layer fixture | $\lvert\bar\ell-\ell_0\rvert$ = 0.0071 at $\sigma_q = 1$ and 0.0084 at 100, both below the tolerance. The pre-check would fail. | T2 fixture must be trained |
| Trained 2-layer fixture (200 steps) | $\ell_0$ = 0.0635; tolerance $1.9\times10^{-4}$; pre-check $\lvert\Delta\rvert$ 4.60 (legacy) and 4.79 (v2); $\sigma_q^\star$ 0.00535 (legacy) and 0.00493 (v2), ratio 0.922; 17 steps | T2(b), T2(e) |

## 7. What this does not cover

- KFAC, GGN, Laplace over $B$, the linearized predictive (Laplace-LoRA Eqs. 11-13) and the marginal-likelihood prior (Eq. 14). The published Laplace-LoRA uses KFAC over A and B with linearized prediction, so the paper must call this row "diagonal empirical-Fisher Laplace on LoRA A".
- The sampled-label Fisher (W2, metrics-assessment.md C5). m7 asks only for the $N_{\mathrm{seq}}T^2$ scaling. The sampled-label Fisher would be a new item with its own cost.
- TFB variants: the pseudo-label anchor set (TFB Appendix E), the accuracy metric, last-layer TFB and any $\epsilon$ sweep. A sweep would tune on outcomes.
- Deterministic-LoRA bases (S5/b2). The S2 C4 fit records are interim (`base_kind: blob_mean`). S5-T2 ("none records a BLoB checkpoint") applies to the b2 fit records under `data/checkpoints/i2_posthoc/b2/`, not to S2's records.
- Which rows the paper uses. If S5 runs (Recommended), the Post-hoc × LoRA cell uses the b2 rows (deterministic LoRA + TFB or Laplace, 3 seeds). The S2 C4 rows stay as one labelled secondary row, "BLoB-Mean + TFB" and "BLoB-Mean + diagonal Laplace", as in the TFB paper's own Table 1. Under Minimal, the S2 C4 rows are the only Post-hoc × LoRA rows. They carry the same labels, and the method matrix (paper.tex:75) must not call the cell "no Bayesian training".
- The 4L rows B1, B3-TFB and B3-LAP. Under the default of choice C1 they are withdrawn, not refit.
- The OOD endpoint, the AUROCs, the baselines, the noise control and any corrected-vs-legacy TFB comparison. S1 and S3 own them.
- The analysis of the re-score. S2 runs the re-score (S1 decision D6), but the AUROCs, CIs and Holm tests for F3 and F4 come from S1's analysis code, and S3 adds the noise control.
- The C pipeline, its gates and its made-up `mi_ood` / `flip_rate` fields.
- Model selection. `ckpt_best` was chosen on validation loss (minigpt/train.py:314), and the fit sets are the `val` splits. So the fits reuse data that model selection has already seen. They read no eval data.
- Whether 480-960 curvature windows give a stable diagonal Fisher. T4(d) only compares the refit median with the saved median.
- The measured MC noise of ΔNLL at $S=20$. It is estimated, not measured. The SE rule bounds what happens.

Handoff to m3 and m13. These texts are wrong under P4 or P5. S2 does not edit them, except `agents/design-rationale.md:25`, which the S2 builder corrects in the same edit that records the S2 decisions.

| Text | Where | Why |
|---|---|---|
| "curvature is flat", "endemic", "near-zero Fisher" | paper.tex:49,113,310; report.md:122; agents/design-rationale.md:25 | P4: $\hat F$ was never scaled, and damping 1.0 dominated it |
| "definitive negative result" | paper.tex:93 | P4 |
| Posterior $(\mathrm{diag}(F)+\lambda I)^{-1}$ with no $N_{\mathrm{seq}}T^2$ | paper.tex:155 | P4 |
| "1.97M parameters" for Laplace on LoRA | paper.tex:155,310 | The c4_lap state has 655,360 LoRA-A entries. 1.97M equals $3 \times 655{,}360$, which is BLoB's count (A mean, A scale, B). |
| "zero Bayesian training", "no Bayesian training" for TFB | paper.tex:51,94,326; the cell at paper.tex:75 | P5 item 4: C4-TFB started from the C3 BLoB mean |
| "$\Omega_{ij}=\sigma_q/d_i$" applied to A | paper.tex:159 | It applies to $A' = V^\top A$ |
| "uniform" samples | wherever it appears | Samples are confidently wrong (per-sample NLL 15-16). Only the MC average is near uniform (entropy 9.4 of 10.8). |
| 4L post-hoc rows | report.md:13,15,16,67,68,122; README.md:39 | Choice C1 (withdrawn by default) |

## 8. Confidence

| Claim | Confidence | Why |
|---|---|---|
| The legacy TFB sampler skips the rotation, and the v2 formula hits the trace target | high | minigpt/tfb.py:73-74,188-192. Probes: legacy 1.14-1.60x on random B, $1.3\times10^{5}$x with Vh = reversal; v2 within 0.25% in every case. |
| The legacy trace is never below the target | high | Doubly stochastic $P$ plus AM-GM (Section 5.1). 1,000 of 1,000 random draws were above 1. |
| On the real C4-TFB state, the legacy sampler gives 1.06-2.02x the intended trace per layer | high | Computed from the saved `svd_cache` |
| The old tolerance was 8 times looser than the paper's | high | configs/c4_tfb.yaml:71 ($\epsilon = 0.1$); saved anchor loss 4.17 nats, so 2.4% against 0.3% |
| $N_{\mathrm{seq}}T^2$ scaling gives a median std of 0.012 (C4-LAP) and 0.014 (C2) | high | The saved states; minigpt/model.py:169; minigpt/laplace.py:112-123 |
| The one-draw rule gives the same ΔNLL estimator that `fit_tfb` already uses | high | minigpt/tfb.py:121-124 reuse seed $s$ for every batch |
| The T1 thresholds cannot fail on an unlucky draw | high | Seeds and $d$ are pinned. The analytic legacy ratio is 1.600 against a threshold of 1.10. |
| The T2 fixture shows different $\sigma_q^\star$ for v2 and legacy | medium | The probe gave 0.00493 against 0.00535 (42 precision steps apart). The builder's fixture will differ in detail. |
| The fixes change the OOD result for Laplace or TFB | unknown | No re-score has run. The scaled std is about half the median weight. The Fisher is empirical, from 480-960 windows. |
| A λ in $10^{0}$-$10^{7}$ brackets TFB's budget | medium | The median scaled precision lies inside the grid. The C2 CPU probe restored the MAP NLL at $\lambda = 10^2$. ΔNLL at the grid ends is not measured. |
| The ±10% match resolves at $S=20$ | medium-low | The MC SE is not measured. The SE rule gives an end state either way (`matched_noisy`). |
| Engineering fits 6-9.5 h | medium | Built up from the source items plus the additions listed in Section 1 |
| S2's own GPU fits 0.8-3.5 GPU h | medium-low | CA10 rates include per-pass draws, so batch 1 is an upper bound. The batching gain is not measured (S0). |
| The moved re-score fits 1.4-4.7 GPU h | medium-low | S1's block counts are estimates. It holds only with the GPU draw for v2 Laplace; a CPU draw adds 5.6-6.4 h (measured draw time, estimated block count). |
| The fit sets share no document with S1's eval sets | medium-high | In S1's draft, test documents come after the 100M cut and pass an unseen check against every cached tensor. T2(c) checks the IDs at run time. It depends on S1 being approved as drafted. |
