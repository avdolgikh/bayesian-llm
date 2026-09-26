# S0: Timing probe (`specs/i2-timing-probe.md`)

Status: DRAFT (not approved). Date: 2026-09-26, revised after the session-02 critic pass. Owner: Alexey approves; agents implement.

## 1. Header

| Field | Value |
|---|---|
| Items | m0 |
| Issues | None directly. S0 replaces three cost assumptions with measurements: CA4 (LoRA fine-tune cost), CA5 (355M cost and VRAM) and CA10 (re-score rates). |
| Feeds | S1: the m4 GPU estimate, the scorer batch size, the G0 cap on blocks per document, and a hand-off check that runs before m4. S2: sampler cost per draw. S5: the fine-tune GPU estimate and one deterministic LoRA adapter it may reuse. Ambitious: 355M VRAM. |
| Approve by | Tue 2026-09-29 (roadmap.md:85) |
| Build and run | Session 03 builds it on Wed Sep 30 and runs it that night (roadmap.md:57). Alexey reads the result at G0 on Thu Oct 1 (roadmap.md:58). The hand-off check runs on Fri Oct 2, before m4 (roadmap.md:60). |
| Script | new `scripts/timing_probe.py` |
| Configs | new `configs/i2_timing_probe.yaml` and `configs/i2_timing_probe_det_lora.yaml` |
| Output | `data/i2/timing_probe.json` (gitignored, like all of `data/`) |
| Adapter | `data/checkpoints/i2_probe/det_lora/ckpt_best.pt`, kept (Section 2) |
| Pytest | new `tests/test_timing_probe.py` (CPU only, 5 tests) |

Costs (one set; the roadmap differences are listed below):

| Cost | Range | Assumption |
|---|---|---|
| Alexey, approve | 12-18 min, inside m2 (1-1.5 h for S0-S4, roadmap.md:94) | CA1 |
| Alexey, read the result at G0 | 0.25 h (roadmap.md:85). CA1's review rate (1 h per 6-9 eng h) on 1.5-2.5 eng h gives 0.17-0.42 h. CA1 rounds items of 0.5 h or less to one 0.25 h step. | CA1 |
| Engineering | 1.5-2.5 h (est.) | CA2 |
| GPU | 0.3-1.1 GPU h on the RTX 4070, in one night | CA3; rates from CA10 with the C2 correction in Section 6 |
| Cloud | $0 | CA7 |

GPU count by part (estimates):

| Part | Test | GPU minutes | Basis |
|---|---|---|---|
| 355M-shape and 76M-shape training steps | S0-T3 | 1-6 | 2 shapes × 13 steps, plus model build. Batches 2 and 1 run only if batch 4 does not fit. |
| Re-score timing, 7 score sets, 3 batch sizes | S0-T1 | 8-33 | Low end: batch-1 rates from Section 6, with a speed-up of 4x at batch 8 and 10x at batch 32 (assumed). High end: no batching gain at all. Plus 1-2 min of model loads. |
| Deterministic LoRA fine-tune | S0-T2 | 8-29 | 7-27 min (options-and-costs.md:185, b2 source), plus 1-2 min for setup and the separate eval timing. CA4's top (4 × 26.9 min = 108 min) is the tail. With that tail the probe is still one night (2.5 GPU h at most). |
| Estimates and hand-off check | S0-T4, S0-T5 | 0 | CPU arithmetic |
| **Total** | | **18-68 (0.3-1.1 GPU h)** | |

The assumptions S0 measures (from options-and-costs.md Section 3):

| ID | Assumption today | What S0 measures |
|---|---|---|
| CA4 | New LoRA fine-tunes cost 1-4x the BLoB cell (27 min per seed; MLflow run 3b22bfcf gives `train_time_sec` 1614.1 s = 26.9 min). | Minutes (same clock as C3), step and eval time, and peak VRAM of one deterministic LoRA fine-tune (S0-T2) |
| CA5 | GPT-2 355M costs 4-5x the 76M runs per step. Peak VRAM of one BLoB step is unknown. | Peak VRAM and ms per step at the 355M shape, and the 355M/76M step-time ratio (S0-T3) |
| CA10 | Batch-1 rates from report.md: C0 8.1 ms; at N=20, C1 287 ms, C3 316 ms, TFB 469 ms. C2 and MC dropout are assumed to match C1, and Laplace-LoRA to match TFB. All 7 sets take about 2.1 s per block. The batching gain is unknown. | ms per block for all 7 score sets at batch 1, 8 and 32 (S0-T1), and the m4 estimate built from them (S0-T4) |

Differences from roadmap Section 3, for Alexey to accept at G0:

| # | Roadmap text | This spec | Why |
|---|---|---|---|
| R1 | S0 eng 1-2 h (roadmap.md:85) | 1.5-2.5 h | 3 more score sets, the hand-off check and 5 CPU tests |
| R2 | S0 GPU 0.5-1 h (roadmap.md:85, :57) | 0.3-1.1 GPU h | Part count above |
| R3 | Reproduction 0.2-0.6 GPU h (roadmap.md:60, :218) | 0.75 GPU h at batch 1 (CA10), less with batching. This is an S1 cost. S0 only writes the estimate. | S1-T1 scores 4 sets × 1,000 blocks, twice (roadmap.md:111). The roadmap priced 7 sets once. |
| R4 | S0 test 3: "9.5 GB" (roadmap.md:102) | 9.5 GiB = 9,728 MiB. This is the authoritative unit. | torch reports MiB. The card has 12,282 MiB (nvidia-smi, 2026-09-26). The "12 GB" in compute-feasibility.md:19 is also GiB. |
| R5 | S0 test 1: 4 methods (roadmap.md:100) | 7 score sets | CA10 assumes the C2, C4-LAP and MC dropout rates. The benchmark never ran them (`scripts/benchmark_inference.py:48-54`). |
| R6 | S0 test 3: one step (roadmap.md:102) | 3 warm-up and 10 timed steps, at the 355M shape and at the 76M shape | One step cannot give a stable time or the CA5 ratio |
| R7 | S0 test 2: "at the step count planned in S5" (roadmap.md:101) | 10,000 steps, C3's schedule (`configs/c3_phase2.yaml:43,51`) | S5 is not written yet. S5 rescales with the measured step time and eval time (Section 5). |
| R8 | Assumption 4: about 2.1 s per block (roadmap.md:37) | 4.0-5.1 s per block at batch 1 | C2 alone costs 2.2-3.3 s (Section 6) |
| R9 | S5 seeds: "3 BLoB + 3 deterministic" (roadmap.md:63, :151) vs "2 more BLoB seeds" (options-and-costs.md:184) | S0 writes the S5 estimate for both counts | S5 picks the count. S0 does not decide it. |
| R10 | S0 test 4: "the estimate is written into S1" (roadmap.md:103) | A script check with an exit code (S0-T5). It runs before m4 on Fri Oct 2. | The next agent session after the probe is Mon Oct 5 (roadmap.md:62), which is after the m4 run. |

What Alexey needs to do:

| # | Action | By | Minutes | Default if Alexey is silent |
|---|---|---|---|---|
| A1 | Approve S0 | Tue Sep 29 | 12-18 | S0 is not built. G0 uses CA10 with the Section 6 correction, so the cap applies. |
| A2 | Decide Q3 (branch and gitignore policy). Suggested: code, configs, specs and tests on a branch, pushed after review (options-and-costs.md:331). | Before session 03, Wed Sep 30 | part of m1 | Q3 has no default (session-02-plan.md:19). Approving S0 also approves the Q3 fallback in Section 2. |
| A3 | Optional: set NVIDIA "Prefer No Sysmem Fallback" | Before Wed night | 2 | `sysmem_fallback_confirmed: unknown`. The step-time and peak-VRAM checks stand in (S0-T3). |
| A4 | Optional: set a power cap (`nvidia-smi -pl`, needs admin) | Before Wed night | 2 | No cap. Today the limit is 200 W, which is the default (nvidia-smi, 2026-09-26). |
| A5 | Optional: pause Windows Update | Before Wed night | 1 | Not paused. A reboot loses only the part in flight, because each part saves its own section. |

Q1 keeps its default (b), so S0-T3 runs. The session-03 agent owns the sleep setting (Section 5, pre-flight).

## 2. Scope and non-scope

What changes:

| File | What |
|---|---|
| `scripts/timing_probe.py` | New. The probe. CLI: `--config`, `--part {shape355m, rescore, finetune, estimate, check, all}`, plus `--m4-config` and `--g0-ack` for `check`. |
| `configs/i2_timing_probe.yaml` | New. Every probe parameter (AGENTS.md: explicit configs). |
| `configs/i2_timing_probe_det_lora.yaml` | New. Training config for S0-T2: a copy of `configs/c3_phase2.yaml` with the changes in Section 5. |
| `tests/test_timing_probe.py` | New. 5 CPU tests (Section 4). |
| `data/i2/timing_probe.json` | New output. Each part merges its own section. |
| `data/checkpoints/i2_probe/det_lora/ckpt_best.pt` | New adapter from S0-T2. It is kept. S5 may reuse it as a deterministic seed (Section 5). Nothing deletes it (PG11). |
| `AGENTS.md` | Structure block only: add `timing_probe` to the scripts line (:34) and update the test count (:35, :124). The rules text does not change. |
| `README.md` | Test count only (:92, :107) |
| `agents/technical-reference.md` | Test count (:53) and one line on the probe and its output |

What must not change:

| Path | Rule |
|---|---|
| `minigpt/`, `experiments/` | No edits. The probe only imports from them. |
| Existing `scripts/` (`benchmark_inference.py`, `eval_c_checkpoints.py`, `eval_mc_dropout.py`, `profile_c_gpu.py`) | No edits. The probe imports their functions. |
| Existing `configs/` | No edits |
| `data/checkpoints/{c0,c1,c2,c3,c4_tfb,c4_lap}`, `data/d1_scores.pt`, `data/mc_dropout_scores.pt`, `data/pile/*.pt` | Read only. Nothing is deleted (PG11). |
| `data/checkpoints/` root folder | No new files. 58 stray `ckpt_step*.pt` files already sit there. |
| `mlflow.db` | No MLflow runs are logged |
| `paper/`, `report.md`, the result numbers in `README.md` | No edits. S0 numbers are planning numbers, not paper numbers. |

Q3 fallback (approved with S0). If Q3 is still open when session 03 starts, the agent writes the script, both YAML files and the tests in the session scratchpad. They import the repo read-only. The only writes are the gitignored outputs under `data/i2/` and `data/checkpoints/i2_probe/`. Once Q3 is decided, the agent moves the files to `scripts/`, `configs/` and `tests/` and makes the three doc edits (about 0.25 eng h, inside the engineering range). No commit is made until Q3 is decided.

Out of scope: the production batched scorer (S1 owns it, in m4), the TFB and Laplace fixes (S2), the paper latency table (S3-T1 and b13), the resume fix, the `save_checkpoint` fallback path (avoided here, not fixed), and any sampler change.

AGENTS.md rules that apply:

| Rule | How S0 meets it |
|---|---|
| No notebooks | `.py` only |
| No extra Bayesian libraries | torch only. `nvidia-smi` and `powercfg` run through `subprocess`, read only. No pynvml. |
| No modern transformer tricks | The 355M shape is plain MiniGPT: learned positions, GELU, LayerNorm |
| Explicit configs | Every parameter sits in the two YAML files. The fine-tune YAML sets `patience_evals` and `patience_min_delta`. Otherwise `minigpt/config.py:231-232` would default them. |
| Coding style | ruff, 100-char lines, type hints on public functions |
| Keep repo docs fresh | The three doc edits above |
| Commits | None until Q3 is decided. No mention of AI assistants in code or comments. |

## 3. Pre-registered endpoint

S0 has no scientific endpoint. It is a measurement. The rules below are fixed before the run, so nobody can read the result to fit a plan. Each rule writes a named JSON boolean.

| Quantity | Rule | JSON field | If the rule fails |
|---|---|---|---|
| C0 anchor (forward only, batch 1) | Last attempt in $[6.075,\ 10.125]$ ms, which is 8.1 ms × (1 ± 0.25) from the YAML | `rescore.anchor.anchor_pass` | If attempt 1 fails, the script waits 60 s, reads `nvidia-smi` again and re-runs once by itself. If attempt 2 also fails, `anchor_pass` is false. The measured rates are kept, because they describe the machine as it is. An anchor failure does not block S0. G0 notes it. |
| m4 at 3 blocks per document | $G_{m4}(3) \le 10$ GPU h | `estimates.m4.cap_applied` = ($G_{m4}(3) > 10$) | `blocks_per_doc` = 1 (G0 rule, roadmap.md:181) |
| m4 at 1 block per document | $G_{m4}(1) \le 10$ GPU h | `estimates.m4.escalate_g0` = ($G_{m4}(1) > 10$) | Alexey decides at G0. If he is silent, m4 runs at 1 block per document and N=20, split over `m4_nights_needed` nights from Fri Oct 2. Sat Oct 3 and Sun Oct 4 have no GPU work (roadmap.md:61). If more than 3 nights are needed, N drops to 10 (compute-feasibility.md:208). |
| Roadmap m4 low end (1 GPU h, options-and-costs.md:133) | $G_{m4}(1) \le 1.0$ GPU h | `estimates.m4.low_end_holds` | At G0, the measured $G_{m4}(k)$ replaces the roadmap's m4 range. |
| 355M shape fits | At batch 4: no out-of-memory error, peak reserved at most 9,728 MiB, and no suspected spill | `shape355m.fits`, `shape355m.spill_suspected` | Ambitious cannot run MiniGPT with an fp32 base on the 4070 at batch 4. v1 is not affected. Batches 2 and 1 are then measured and recorded. |
| CA4 | `wall_min` at most 107.6 min (4 × 26.9) | `finetune.ca4_exceeded` | S5 and b11 GPU ranges are re-costed from the measured value |
| CA5 | 355M/76M step-time ratio in $[4.0,\ 5.0]$ | `shape355m.ca5_out_of_range` | Ranges built on CA5 (a5: 16-65 GPU h) are rescaled by ratio / 4.5 |

The speed-up is reported but decides nothing. `speedup_vs_ca10` compares with the 2.1216 s CA10 sum. `speedup_measured` compares with the measured batch-1 sum.

Negative framing, stated now: "If $G_{m4}(1) > 1.0$ GPU h, the m4 low end in the roadmap is wrong. If $G_{m4}(3) > 10$ GPU h, the re-score runs at 1 block per document."

## 4. Acceptance tests

S0 is accepted when S0-T1(a), (c) and (d), S0-T2, S0-T3, S0-T4 and S0-T5 pass. S0-T1(b) is recorded but does not block.

| ID | Given | When | Then |
|---|---|---|---|
| S0-T1 | The 7 score sets: C0, C1, C2, C3, C4-TFB and C4-LAP (checkpoints and states in `data/checkpoints/`), plus MC dropout on C0. Blocks 0-159 from `load_eval_data(160)`: 256-token ID blocks, StackExchange tokens 80,000,000-80,040,961. N=20 (N=1 for C0). | `python scripts/timing_probe.py --config configs/i2_timing_probe.yaml --part rescore`. First the anchor, on its own fresh C0 load. Then, for each cell, one warm-up batch and the timed blocks (blocks 1-64 at batch 1, 8-135 at batch 8, 32-159 at batch 32). Then an untimed C0 NLL pass over blocks 0-63 at each batch size. | (a) `rescore.cells` holds 21 cells (7 sets × 3 batch sizes). Each cell has `ms_per_block`, `n_blocks_timed`, `peak_reserved_mib` and `peak_allocated_mib`, or `over_budget: true`. The C2, C4-TFB and C4-LAP cells also have `sampler_ms_per_sample`. Each of the 7 sets has at least one cell with `over_budget: false`. Yes/no. (b) `anchor.attempts_ms` holds 1 or 2 values, and `anchor_pass` follows the Section 3 rule. Yes/no (recorded, not blocking). (c) `nll_check.max_abs_diff_nats` is at most 0.01 at batch 8 and at batch 32 against batch 1, over all 64 blocks. Pass/fail. An indexing bug gives more than 0.1. Runs in `scripts/timing_probe.py`. (d) `tests/test_timing_probe.py::test_batched_scorer_consistency`, on a 2-layer CPU fixture in fp32 with 16 blocks: per-block NLL at batch 8 equals batch 1 to 1e-5. With 3 identical draws, max MI and max $g_t$ are at most 1e-6. `::test_json_merge_schema`: two parts merged into a file under `tmp_path` keep both sections and all header fields, and no `.tmp` file remains. A file without `rescore.cells` fails validation with `SchemaError`. Pass/fail. |
| S0-T2 | `configs/i2_timing_probe_det_lora.yaml`: C3's phase-2 schedule (10,000 steps, batch 32, block 256, lr 3e-4, warmup 500, eval every 500 steps with 20 iters, seed 1337), with the Section 5 changes. A deterministic LoRA (rank 16, alpha 32, FFN) on `data/checkpoints/c0/ckpt_best.pt`. HackerNews train split. `torch.manual_seed(1337)` runs before the model is built and the LoRA is injected. | `--part finetune` trains once, from a fresh optimizer, to the last step. It never resumes. Then it times `estimate_loss` on its own, 3 times. | The JSON `finetune` section holds `wall_min` (= `train_time_sec` / 60), `train_step_ms`, `eval_s`, `n_evals`, `peak_reserved_mib`, `config_sha256`, `seed_applied` and `checkpoint`. Yes/no checks: `steps_completed` = 10,000; `n_evals` = 21; `trainable_params` = 1,310,720; `best_val_loss` is below the step-1 val loss; `seed_applied` = 1337. `ca4_ratio` = `wall_min` / 26.9 and `ca4_exceeded` are written. `estimates.s5_finetune_gpu_h` holds values for `n_blob_2` and `n_blob_3`. Runs in `scripts/timing_probe.py`. |
| S0-T3 | Q1 = (b), the default. MiniGPT at 24 layers, 16 heads, width 1024, block 512, vocab 50,257, dropout 0.1, random weights. BLoB LoRA (rank 16, alpha 32, prior 0.2, init_g 0.1) on the FFN. AMP fp16, AdamW, grad clip 1.0. | `--part shape355m` runs 3 warm-up and 10 timed steps at batch 4, with a synchronize after each step. Then it runs the same at the 76M shape (16 layers, 8 heads, width 512, block 512, batch 4). If the 355M shape does not fit at batch 4, it repeats at batches 2 and 1. | Yes/no parameter counts. 355M: base = 354,298,880 and BLoB trainable = 5,898,240. 76M at block 512: base = 76,432,896 and BLoB trainable = 1,966,080. Each shape records `peak_reserved_mib`, `peak_allocated_mib`, `median_step_ms` and `max_step_ms`. `spill_suspected` = (`max_step_ms` > 2 × `median_step_ms`). `fits` = no out-of-memory error, and `peak_reserved_mib` ≤ 9,728, and not `spill_suspected`. `ca5_ratio` = median step (355M) / median step (76M), with `ca5_out_of_range`. If Q1 = (a), the section holds `skipped: "Q1=a"` and nothing runs. Runs in `scripts/timing_probe.py`. |
| S0-T4 | The `rescore` and `finetune` sections. The S1 planning counts: 1,000 documents per domain; k = 1, 2 or 3 blocks per document; 5 sets scored by all 7 methods (StackExchange, arXiv, markup-stripped arXiv, FreeLaw, PubMed); HackerNews scored by the 3 LoRA rows. | `--part estimate` (CPU only) computes $b^*_m$, $G_{m4}(k)$, $G_{\mathrm{repro}}$, the speed-ups and $G_{S5}$ (Section 5). | `estimates.m4` holds `gpu_h` for k = 1, 2, 3, plus `batch_size` ($b^*_m$ per set), `cap_applied`, `blocks_per_doc`, `escalate_g0`, `low_end_holds`, `m4_nights_needed`, `speedup_vs_ca10` and `speedup_measured`. `estimates.repro` holds `gpu_h_b1` and `gpu_h_bstar`. If a set has no cell within budget, the part exits with `NoFeasibleBatchError`, which names the set. `tests/test_timing_probe.py::test_estimate_arithmetic`, with the CA10 rates as fixture: $G_{m4}(1)$ = 3.29 ± 0.01, $G_{m4}(3)$ = 9.88 ± 0.01, `cap_applied` false, `blocks_per_doc` 3, `m4_nights_needed` 2, `low_end_holds` false, `escalate_g0` false. Rates × 1.02: $G_{m4}(3)$ = 10.08 ± 0.01, `cap_applied` true, `blocks_per_doc` 1, `m4_nights_needed` 1. Rates ÷ 3.5: $G_{m4}(1)$ = 0.94 ± 0.01, `low_end_holds` true. $G_{\mathrm{repro}}$ at batch 1 = 0.75 ± 0.01. With `wall_min` = 20.0: $G_{S5}$ = 1.897 ± 0.005 (`n_blob_2`) and 2.345 ± 0.005 (`n_blob_3`). `::test_batch_choice`: set a (batch 1: 0.30 s, 500 MiB; batch 8: 0.05 s, 2,000 MiB; batch 32: 0.03 s, 10,000 MiB) gives $b^*$ = 8. Set b (batch 8 over budget; batch 32: 0.04 s, 9,000 MiB) gives $b^*$ = 32. Set c (all cells over budget) raises `NoFeasibleBatchError`. Pass/fail. |
| S0-T5 | `data/i2/timing_probe.json` and S1's m4 YAML, which sets `blocks_per_doc` explicitly | `python scripts/timing_probe.py --part check --m4-config <S1 m4 YAML>` runs as the first line of the Fri Oct 2 night list, before the m4 re-score | Exit 0 if the YAML's `blocks_per_doc` ≤ the JSON's `blocks_per_doc`, and either `escalate_g0` is false or `--g0-ack` is given. Exit 2 on a blocks mismatch, with a message that names both values. Exit 3 if `escalate_g0` is true and `--g0-ack` is missing. Exit 4 if the JSON is missing. `tests/test_timing_probe.py::test_check_plan`, with synthetic files in `tmp_path`: (YAML 3, JSON 1) gives 2; (1, 1) gives 0; (1, 3) gives 0; escalated without ack gives 3; escalated with ack gives 0; no JSON gives 4. Pass/fail. |

## 5. Implementation notes for the builder

**Smallest change.** Write one new script that imports existing functions. Do not copy logic that already exists.

| Need | Reuse |
|---|---|
| Forward-only anchor | `benchmark_latency` and `_load_model("c0", device)` in `scripts/benchmark_inference.py:186-234` and `:61`. This code produced the 8.1 ms (report.md:90). |
| Model loading | `load_model` in `scripts/eval_c_checkpoints.py:153-208` for C0 to C4-LAP. For MC dropout: `load_c0_model` in `scripts/eval_mc_dropout.py:84-92`, plus `enable_dropout` in `minigpt/layers.py:230-244`. |
| Old blocks | `load_eval_data(160)` in `scripts/eval_c_checkpoints.py:96-109`. The ID set is the first windows of `test_id`, which is StackExchange from token 80M (`minigpt/data.py:207-212`). |
| Samplers | `sample_tfb_params` (`minigpt/tfb.py:160-194`), `sample_laplace_params` (`minigpt/laplace.py:140-173`) and `apply_sampled_params` (`minigpt/laplace.py:186-211`), unchanged |
| 355M training step | The step logic of `scripts/profile_c_gpu.py:58-104` (GradScaler, clip 1.0, `_configure_optimizer`), with dropout 0.1 instead of 0.2 (`profile_c_gpu.py:123`) |
| Fine-tune | `train()` in `minigpt/train.py:165`, `estimate_loss` in `minigpt/train.py:54`, `inject_lora(..., bayesian=False)` in `minigpt/lora.py:153-207`, and `build_gpt_config`, `build_lora_config` and `build_train_config` in `minigpt/config.py:183-240` |

**Run order.** `--part all` runs each part as its own subprocess: `shape355m`, then `rescore`, then `finetune`, then `estimate`. The short memory test runs first, so an out-of-memory or spill problem shows up in minutes. One process per part frees VRAM and isolates crashes (code-assessment.md:268). Each part merges its section into the JSON through a temp file and `os.replace`, as in `experiments/pipeline_runner.py:578-582`. `check` is not part of `all`. It runs on Fri Oct 2.

**Anchor.** Run it first, on its own fresh load from `_load_model("c0", device)`. Do not change that model's state, and do not call `requires_grad_(False)` on it. It then stays comparable with report.md. The scoring cells load their models again. Record `other_compute_procs` from `nvidia-smi --query-compute-apps=pid --format=csv,noheader` before each attempt.

**Batched scoring (probe-local; S1 owns the production scorer).** A batch holds B blocks, $x \in \mathbb{Z}^{B \times 256}$, with targets $y$. For C1, C2, C3, C4-TFB and C4-LAP, each forward call uses one weight draw for all B blocks. `BayesianLinear` samples per call (`minigpt/layers.py:75-85`) and so does BLoB (`minigpt/lora.py:69-79`). TFB and Laplace apply one sampled set per batch. MC dropout differs: its masks are drawn per element, so each block gets its own mask. For sample $s = 1, \dots, N$ and position $t$:

$$\ell_{s,t} = \log \operatorname{softmax}(z_{s,t}), \qquad p_{s,t} = e^{\ell_{s,t}}, \qquad \bar p_t = \frac{1}{N}\sum_{s=1}^{N} p_{s,t}$$

$$\mathrm{MI}_t = H[\bar p_t] - \frac{1}{N}\sum_{s=1}^{N} H[p_{s,t}], \qquad H[p] = -\sum_{v} p_v \log p_v$$

$$g_t = \operatorname{logsumexp}_{s}\, \ell_{s,t}(y_t) - \log N - \frac{1}{N}\sum_{s=1}^{N} \ell_{s,t}(y_t)$$

- Keep the running sums of $p_{s,t}$ and $H[p_{s,t}]$ in fp32, updated in place. Do not chunk. The probe measures the simple path. S1 may chunk.
- Inside the timed region, copy these to the CPU: per-token $\mathrm{MI}_t$, $H[\bar p_t]$, $\max_v \bar p_{t,v}$, $g_t$, and the $N$ values $\ell_{s,t}(y_t)$. S1 saves the per-sample realized-token log-probs, so their cost belongs in the rate.
- Call `requires_grad_(False)` on all parameters before the scoring cells. Autocast caches fp16 copies of weights that require grad, which inflates VRAM (paper-critique.md:62).
- Seeds: call `torch.manual_seed(seed)` at the start of each cell. For TFB and Laplace, the draw seed is (batch index) × N + s. The old scorer used (block index) × N + s (`scripts/eval_c_checkpoints.py:243`). Timing does not depend on this.

**Timing.** For method $m$ at batch size $b$ over $n_b$ timed blocks:

$$r_m(b) = \frac{T_m(b)}{n_b}$$

- $T_m(b)$ runs from `torch.cuda.synchronize()` before the first timed batch to the point after the last `.cpu()` copy.
- It includes sampling and host copies. It excludes model loading and the warm-up batch.
- The timed blocks are 64 at batch 1, 128 at batch 8 (16 batches) and 128 at batch 32 (4 batches). The earlier draft used 256 at batches 8 and 32. The smaller counts cap the worst case (no batching gain) at about 33 GPU min.
- Peak VRAM is `torch.cuda.max_memory_reserved()` after `reset_peak_memory_stats()` in each cell. Also record `max_memory_allocated()`.
- On `torch.cuda.OutOfMemoryError`, or when peak reserved is above 9,728 MiB, mark the cell `over_budget: true`, call `empty_cache()` and go on.
- For C2, C4-TFB and C4-LAP, time the draw plus the parameter swap alone: 1 warm-up and 5 timed draws, with a synchronize. Record the result as `sampler_ms_per_sample`. S2 can then judge what a GPU noise draw would save.

**NLL check.** This is a separate, untimed pass for C0 at N=1 over blocks 0-63, at batch 1, 8 and 32. Use the same autocast settings as the timed cells. Compute the per-block mean NLL of $y$ and record `max_abs_diff_nats` for batch 8 and batch 32 against batch 1.

**Fine-tune config.** `configs/i2_timing_probe_det_lora.yaml` equals `configs/c3_phase2.yaml` except in these fields:

| Field | Value | Why |
|---|---|---|
| `experiment.name`, `experiment.run_name` | `i2_probe_det_lora` | Keeps it apart from C3 |
| `train.checkpoint_dir` | `data/checkpoints/i2_probe/det_lora` | Never write into C3's or C0's folders |
| `train.checkpoint_interval` | 0 | No periodic checkpoints (`minigpt/train.py:385`). C3 used 5,000 (`configs/c3_phase2.yaml:53`). |
| `train.kl_weight`, `train.kl_annealing_steps` | 0.0, 0 | Deterministic LoRA |
| `train.patience_evals`, `train.patience_min_delta` | 0, 0.001 | The run reaches the last step (`minigpt/train.py:373`). Without these fields the code defaults to 10 (`minigpt/config.py:231-232`). |

- Order: `torch.manual_seed(cfg["train"]["seed"])`, then build `MiniGPT`, then `load_checkpoint(data/checkpoints/c0/ckpt_best.pt, model)`, then `inject_lora(..., bayesian=False)`, then `train()`. `train()` never seeds (`minigpt/train.py:165-232`). The LoRA A matrix gets a random Kaiming init (`minigpt/lora.py:142`). The pipeline seeds at `experiments/experiment_setup.py:44`, before it builds the model. The probe copies that order, so `seed_applied` is the seed that was actually used.
- C3's phase 1 is a copy of C0's checkpoint (`experiments/c_pipeline.py:81-91`). So S0 uses the same base as C3.
- Call `train(..., mlflow_run=None, config_dict=cfg, kl_weight=0.0, num_train_tokens=0)`. Never pass `resume_ckpt`, because resume drops the AdamW state (code-assessment.md Section 2.5).
- Before starting, assert that `data/pile/hackernews_100000000.pt` and the three OOD caches (`arxiv_10000000.pt`, `freelaw_10000000.pt`, `pubmed_abstracts_10000000.pt`) exist. A missing cache would stream from the network and write a new cache (`minigpt/data.py:149-151`).
- Set `wall_min` = `metadata["train_time_sec"]` / 60. This is the same clock as C3's 1614.1 s. It runs from `minigpt/train.py:229` to `:399` and includes the 21 evaluations and the best-checkpoint reload (`:393-394`). Also record the time of the whole `train()` call as `train_call_min`, for information only.
- Eval time: after training, time `estimate_loss` on the train split and the val split (batch 32, block 256, 20 iters), 3 times, and take the median of each pair as `eval_s`. Then:

$$t_{\mathrm{step}} = \frac{T_{\mathrm{train}} - n_{\mathrm{eval}}\, t_{\mathrm{eval}}}{10{,}000}, \qquad t_{\det}(n, i) = \frac{n\, t_{\mathrm{step}} + \left(\lfloor n / i \rfloor + 1\right) t_{\mathrm{eval}}}{60}\ \text{min}$$

  Here $T_{\mathrm{train}}$ is `train_time_sec`, $n_{\mathrm{eval}}$ = 21, and $i$ is the eval interval. S5 uses $t_{\det}(n, i)$ for any other step count. A linear rescale by steps alone would be wrong, because the evals sit inside the timed region. Best-checkpoint saves land in $t_{\mathrm{step}}$.
- Reuse by S5. The adapter is kept together with `config_sha256` and `seed_applied`. The P5 verdict needs a true deterministic LoRA + TFB row, and S5-T2 needs 3 deterministic seeds. S5 may count this adapter as one of them if its own deterministic config hashes to the same `config_sha256` and its seed script seeds at the same point. That saves 7-27 GPU min. Otherwise S5 retrains. S5 decides.

**355M shape.** Build `GPTConfig(vocab_size=50257, block_size=512, n_layer=24, n_head=16, n_embd=1024, dropout=0.1, bias=True)` with no Bayesian layers. Apply `inject_lora(LoRAConfig(16, 32.0, "ffn", 0.2, 0.1), bayesian=True)`. Use random tokens, `torch.randint(0, 50257, (1_000_000,))`, as in `profile_c_gpu.py:208`. The base stays fp32 with fp16 autocast, because MiniGPT has no bf16 path. Batches 2 and 1 run only if batch 4 does not fit. With the fallback policy unknown, a VRAM spill would not raise an error. It would run several times slower (code-assessment.md:271). So spill detection uses the step-time spread:

$$\text{spill suspected} \iff \max_j t_j > 2\,\operatorname{median}_j t_j, \qquad \rho_{\mathrm{CA5}} = \frac{\operatorname{median}_j t_j(355\mathrm{M})}{\operatorname{median}_j t_j(76\mathrm{M})}$$

Both shapes use the same settings (block 512, batch 4). The C runs used block 256 and batch 32.

**Estimates (`--part estimate`, CPU).** The best batch size for each set is the fastest one that fits:

$$b^*_m = \operatorname*{arg\,min}_{b \in \{1, 8, 32\}:\ \text{not over budget}} r_m(b)$$

$$G_{m4}(k) = \frac{d\,k}{3600}\left[\, n_{\mathrm{all}} \sum_{m \in \mathcal{M}} r_m(b^*_m) + n_{\mathrm{L}} \sum_{m \in \mathcal{L}} r_m(b^*_m) \right]$$

- $d$ = 1,000 documents per domain and $k$ = blocks per document.
- $n_{\mathrm{all}}$ = 5 sets and $n_{\mathrm{L}}$ = 1 set (HackerNews).
- $\mathcal{M}$ = the 7 score sets. $\mathcal{L}$ = {C3, C4-TFB, C4-LAP}.
- $r_m$ is in seconds per block. C0 counts one pass.

$$S_{\mathrm{CA10}} = \frac{2.1216\ \mathrm{s}}{\sum_{m \in \mathcal{M}} r_m(b^*_m)}, \qquad S_{\mathrm{meas}} = \frac{\sum_{m \in \mathcal{M}} r_m(1)}{\sum_{m \in \mathcal{M}} r_m(b^*_m)}, \qquad n_{\mathrm{nights}} = \left\lceil \frac{G_{m4}(k^*)}{8} \right\rceil$$

Here $k^*$ = `blocks_per_doc`, and 8 GPU h is the low end of a night (CA3).

$$G_{\mathrm{repro}}(b) = \frac{2 \times 1000}{3600} \sum_{m \in \{\mathrm{MCD},\, \mathrm{C1},\, \mathrm{C3},\, \mathrm{TFB}\}} r_m(b), \qquad G_{S5}(n_{\mathrm{BLoB}}) = \frac{26.9\, n_{\mathrm{BLoB}} + n_{\det}\, t_{\det}}{60}$$

- $G_{\mathrm{repro}}$ is reported at $b = 1$ and at $b^*_m$. It is an S1 cost (R3).
- $n_{\det}$ = 3 and $n_{\mathrm{BLoB}} \in \{2, 3\}$ (R9). $t_{\det}$ = `wall_min`.

Fixture numbers for `tests/test_timing_probe.py` (CA10, batch 1, seconds per block): C0 0.0081; C1, C2 and MC dropout 0.2869; C3 0.3156; C4-TFB and C4-LAP 0.4686. The 7-set sum is 2.1216 s and the LoRA sum is 1.2528 s. So $G_{m4}(k) = k \times 3.2947$ GPU h.

**Tests.** `scripts/` is not a package, and no test imports a script yet (P20). Import the probe the way `tests/test_c_pipeline.py:37-42` imports the pipeline: `monkeypatch.syspath_prepend(<repo>/scripts)` and then `importlib.import_module("timing_probe")`. Keep the pure functions (`choose_batch`, `estimate`, `merge_section`, `validate_schema`, `check_plan`, batched statistics) at module level. Import the GPU-side helpers inside the part functions, so that importing the module does no work. The tests must not read `data/`, which is gitignored and absent in CI. Every fixture is built in the test or in `tmp_path`.

**Probe YAML (sketch; every field explicit).**

```yaml
probe:
  output: data/i2/timing_probe.json
  q1_backbone: b
  vram_budget_mib: 9728
  sysmem_fallback_confirmed: unknown   # yes | no | unknown; set by the operator, not measured
  windows_update_paused: unknown       # yes | no | unknown; set by the operator
rescore:
  methods: [c0, c1, c2, c3, c4_tfb, c4_lap, mc_dropout]
  n_samples: 20
  n_samples_c0: 1
  block_size: 256
  n_blocks_loaded: 160
  batch_sizes: [1, 8, 32]
  blocks_timed: {1: 64, 8: 128, 32: 128}
  warmup_batches: 1
  seed: 0
  sampler_timing_draws: 5
  nll_check: {blocks: 64, tol_nats: 0.01}
  anchor: {warmup: 5, repeats: 20, ref_ms: 8.1, tolerance: 0.25, max_attempts: 2, retry_wait_s: 60}
finetune:
  train_config: configs/i2_timing_probe_det_lora.yaml
  lora_type: deterministic
  base_checkpoint: data/checkpoints/c0/ckpt_best.pt
  blob_reference_min: 26.9             # C3 train_time_sec 1614.1 s, MLflow run 3b22bfcf
  n_blob_seeds: [2, 3]                 # options b1 vs roadmap S5-T2; S5 picks
  n_det_seeds: 3
  ca4_max_ratio: 4.0
  eval_timing_repeats: 3
shape355m:
  n_layer: 24
  n_head: 16
  n_embd: 1024
  block_size: 512
  batch_size: 4
  fallback_batch_sizes: [2, 1]
  dropout: 0.1
  bias: true
  lora: {rank: 16, alpha: 32.0, target: ffn, prior_std: 0.2, init_g: 0.1}
  compare: {n_layer: 16, n_head: 8, n_embd: 512}
  warmup_steps: 3
  timed_steps: 10
  max_step_ratio: 2.0
  lr: 0.0003
  weight_decay: 0.0
  grad_clip: 1.0
  ca5_range: [4.0, 5.0]
  seed: 0
estimate:
  docs_per_domain: 1000
  blocks_per_doc: [1, 2, 3]
  sets_all_methods: 5
  sets_lora_only: 1
  lora_methods: [c3, c4_tfb, c4_lap]
  cap_gpu_h: 10.0
  low_end_gpu_h: 1.0
  night_gpu_h: 8.0
  ca10_sum_s: 2.1216
  repro: {blocks: 1000, runs: 2, methods: [mc_dropout, c1, c3, c4_tfb]}
```

**JSON fields.**

| Section | Fields |
|---|---|
| header | `schema` (`i2-timing-probe/1`), `created_utc`, `git_head` and `git_dirty` (read-only git), `gpu_name`, `driver_version`, `torch_version`, `cuda_version`, `memory_total_mib`, `power_limit_w`, `power_default_limit_w`, `other_compute_procs`, `sleep_disabled` (read back from `powercfg`), `sysmem_fallback_confirmed` and `windows_update_paused` (both copied from the YAML as operator statements, yes, no or unknown), `config_sha256` for both configs |
| `rescore` | `anchor` {`attempts_ms`, `anchor_pass`, `other_compute_procs`}, `cells` [{`method`, `batch_size`, `ms_per_block`, `n_blocks_timed`, `peak_reserved_mib`, `peak_allocated_mib`, `over_budget`, `sampler_ms_per_sample`}], `nll_check` {`max_abs_diff_nats`, `pass`} |
| `finetune` | `wall_min`, `train_call_min`, `train_step_ms`, `eval_s`, `n_evals`, `steps_completed`, `trainable_params`, `best_val_loss`, `first_val_loss`, `peak_reserved_mib`, `config_sha256`, `seed_applied`, `checkpoint`, `ca4_ratio`, `ca4_exceeded` |
| `shape355m` | per shape: `base_params`, `trainable_params`, `peak_reserved_mib`, `peak_allocated_mib`, `median_step_ms`, `max_step_ms`, `oom`; then `fits`, `spill_suspected`, `ca5_ratio`, `ca5_out_of_range`, `fallback` (batches 2 and 1, only if not `fits`), or `skipped` |
| `estimates` | `m4` {`gpu_h`, `batch_size`, `cap_applied`, `blocks_per_doc`, `escalate_g0`, `low_end_holds`, `m4_nights_needed`, `speedup_vs_ca10`, `speedup_measured`}, `repro` {`gpu_h_b1`, `gpu_h_bstar`}, `s5_finetune_gpu_h` {`n_blob_2`, `n_blob_3`} |

**Night pre-flight (code-assessment.md:271).** Each setting has an owner and a default:

| Setting | Owner | Default |
|---|---|---|
| Sleep off: `powercfg /change standby-timeout-ac 0` | Session-03 agent, as the first line of the Wed night list (permission prompt) | If it is refused, the probe runs anyway and records `sleep_disabled` as false |
| No other compute process | The probe itself (`nvidia-smi`) | Recorded; the anchor retry re-checks it |
| NVIDIA "Prefer No Sysmem Fallback" | Alexey (A3) | `unknown`; the spill checks in S0-T3 stand in |
| Power cap | Alexey (A4) | None (200 W, the default). If a cap is set later for the nights, the S0 rates are optimistic. Re-run `--part rescore` under that cap (8-33 GPU min). |
| Windows Update pause | Alexey (A5) | Not paused |

## 6. Verification evidence from session 02

My own checks. All are CPU only or read only. The scripts sit in `scratchpad/s02/`. No GPU job ran, and nothing in the repo changed.

| Check | Result | Evidence |
|---|---|---|
| C2 sampler cost | The C2 state holds 33,554,432 entries in 32 tensors (damping 1.0, scale 1.0). One CPU draw of that size took 149 ms in a 3-repeat run and 102-104 ms in a 10-repeat run (min 102, median 102, max 104), on 8 threads. The sampler draws on the CPU with a CPU generator, then copies to the device. | `minigpt/laplace.py:156-171`; `data/checkpoints/c2/laplace_state.pt`; `s0_cpu_inspect.py` |
| What this means for CA10 | At N=20 and batch 1, C2 costs at least 20 × (102 + 8.1) ms = 2.2 s per block: the draw plus a C0-sized forward. That is 7.7x CA10's 287 ms. With the 149 ms draw and C1's forward rate it is 3.3 s. The 7-set sum becomes 4.0-5.1 s per block. At batch 1, $G_{m4}(1)$ = 6.0-7.4 GPU h and $G_{m4}(3)$ = 17.9-22.3 GPU h. So the cap applies at batch 1, but escalation does not. For the cap not to apply, batching must give about 1.8-2.2x overall. For the roadmap's low end to hold, it must give about 6-7.4x. These are estimates. S0-T1 measures them. | Arithmetic on the row above; CA10 fixture in Section 5; report.md:90-101 |
| The report.md benchmark covers 5 configurations | No C2, C4-LAP or MC dropout rows | `scripts/benchmark_inference.py:48-54` |
| The report.md rates are forward passes only | The benchmark times `model(x)` on random tokens. The scorer adds a full-vocabulary fp32 softmax and entropy for each sample, at batch 1. So CA10 rates are lower bounds for the scoring path. | `scripts/benchmark_inference.py:186-234` vs `scripts/eval_c_checkpoints.py:216-277` |
| TFB timing includes the CPU noise draw | Yes, inside the timed loop | `scripts/benchmark_inference.py:200-205`; `minigpt/tfb.py:169-192` |
| One weight draw per forward, shared across the batch | Yes for C1 and BLoB. Not for MC dropout, which draws its masks per element in train mode. | `minigpt/layers.py:75-85, 230-244`; `minigpt/lora.py:69-79` |
| Parameter counts (meta device) | C0 shape, block 256: 76,301,824. 76M shape, block 512: 76,432,896. 355M shape, block 512: 354,298,880. Deterministic LoRA: 1,310,720 trainable. BLoB: 1,966,080 (76M) and 5,898,240 (355M). | `minigpt/model.py:106-139`; `minigpt/lora.py:153-207`; `s0_cpu_inspect.py` |
| Post-hoc LoRA states | C4-LAP: 655,360 entries. TFB: 655,360 entries in 32 tensors, sigma_q = 0.0300. | `data/checkpoints/c4_lap/laplace_state.pt`; `data/checkpoints/c4_tfb/tfb_state.pt` |
| C3's time base | `train_time_sec` = 1614.1 s (26.9 min). The run itself took 52.8 min from start to end, including the post-training evals. `best_val_step` = 7000. There are 21 val points up to step 10,000, so all steps ran. The patience default of 10 did not trigger: 6 evals came after the best one. `checkpoint_interval` was 5000. | `mlflow.db` run 3b22bfcf (read-only sqlite); `minigpt/train.py:229, 393-399`; `minigpt/config.py:231` |
| Seeding | `train()` never calls `torch.manual_seed`. The pipeline seeds before it builds the model. The deterministic LoRA A matrix gets a random Kaiming init. | `minigpt/train.py:165-232`; `experiments/experiment_setup.py:44`; `minigpt/lora.py:142` |
| Test import pattern | No `scripts/__init__.py` and no `conftest.py`. One test already imports a non-package module through `syspath_prepend`. | `tests/test_c_pipeline.py:37-42` |
| Machine today | RTX 4070, 12,282 MiB, power limit 200 W (the default), driver 591.86 | `nvidia-smi --query-gpu` (read only) |
| Checkpoint fallback path | Without `config_dict`, periodic checkpoints go to `data/checkpoints/`. 58 `ckpt_step*.pt` files already sit there. | `minigpt/train.py:96-99` |

Refuter verdicts from session 02:

| Refuter | Verdict | Evidence and numbers | What it changes in S0 |
|---|---|---|---|
| P4, Laplace scale | confirmed | The Fisher is a mean over 480 (C2) or 960 (C4-LAP) sequences of squared gradients of the token-mean loss (`minigpt/laplace.py:115-123`). The sampler uses std = $\sqrt{1/(F + 1)}$ (`:167-168`), with damping 1.0 (`configs/c2.yaml:79-81`). In the saved states, the median std is 1.000 against a median $\lvert w \rvert$ of 0.0247 (C2) and 0.0233 (C4-LAP). The D1 NLL is 9.10 and 9.73 (report.md:42,45). The fix is $N_{\mathrm{seq}} T^2 F + \lambda$. C2 also needs $\lambda \ge$ about $10^2$. | The fix changes the variance values, not the draw. The draw is still a CPU `randn` over 33,554,432 (C2) and 655,360 (C4-LAP) entries (`minigpt/laplace.py:156-171`). So the S0 rates for C2 and C4-LAP hold after S2, unless S2 moves the draw to the GPU. `sampler_ms_per_sample` prices that option. The $\lambda$ sweep is an S2 cost. |
| P5, TFB rotation | confirmed | `V` is really `Vh` (`minigpt/tfb.py:73-74`). It is never used, and the noise on A is row-scaled by $\sigma_q/S_i$ (`:178, 188-192`). With random B, the trace is 2.92 and 3.36 against a target of 2.56. On the real C4-TFB state, it is 1.06-2.02x per layer. C4-TFB's A equals C3's `lora_A_mu` exactly (difference 0), so C4-TFB is BLoB-Mean + TFB. | The fix $A = A_{\mathrm{MAP}} + V_h^{\top}(\sigma \odot \varepsilon)$ adds one $16 \times 16$ by $16 \times d_{\mathrm{in}}$ matmul per layer per draw. That is about $10^7$ multiply-adds over 32 layers, which is negligible next to a 21.2 ms forward (report.md:99). So the S0 TFB rate holds after S2. S0 times the saved C4-TFB state as it is; the rate does not depend on the base. Because the verdict needs a true deterministic LoRA + TFB row, the S0-T2 adapter is kept for S5 (Section 5). |
| P1/P19, eval split | confirmed | The eval uses C0's split for every method (`scripts/eval_c_checkpoints.py:96-109`). ID = StackExchange[80,000,000:80,128,001] (`minigpt/data.py:202, 207-213`). OOD = arxiv[0:128,001], which holds 9 papers. The MI ratios are a hardcoded dict (`scripts/eval_c_checkpoints.py:71-78`). The caches hold 0 EOT tokens. | S0 times ID blocks 0-159, which are StackExchange tokens 80,000,000-80,040,961. Every block has 256 tokens, so the cost does not depend on the domain. The set counts in $G_{m4}$ come from S1's rebuilt plan: 5 sets for all methods, plus HackerNews for the 3 LoRA rows, as the P1/P19 fix requires. S0 reads no MI or AUROC. |

## 7. What this does not cover

- AUROC or any score quality. In a batch, C1, C2, C3, C4-TFB and C4-LAP share one weight draw across the blocks. MC dropout does not. S1-T1 tests whether the shared draw moves AUROC.
- The paper's latency or cost table. S0 numbers are planning numbers. S3-T1 and b13 own the paper numbers. Merged-LoRA timing is also out.
- Timing of generation or answer rescoring (S6-T5).
- The steps a 76M model needs to memorize fictional facts. CA4 for b10 and b11 stays an extrapolation.
- The TFB fit time after S2 changes the sigma_q search, and the cost of S2's $\lambda$ sweep.
- Whether C2's noise draw moves to the GPU (S2 decides). Whether S5 reuses the S0 adapter (S5 decides).
- The content of S1's m4 YAML and of the Fri Oct 2 night list. S0 gives only the check (S0-T5). S1 and session 03 must call it.
- HF + PEFT at 355M (compute-feasibility.md estimates 7.0-7.1 GB in bf16). S0 measures MiniGPT at the 355M shape with random weights, an fp32 base and fp16 autocast. The GPT-2 loader (a1) is not built.
- Kaggle or rental speed factors (CA7).
- Variation over a whole night (heat, power). S0 takes one sample of the machine state.
- Monte Carlo noise at N=20.
- A timing clash in the roadmap: session 03 builds the S5 seed script before S5 is approved (Thu Oct 1). S0-T2 does not depend on S5 or its script. It calls `minigpt.train.train()` directly.

## 8. Confidence

| Claim | Confidence | Why |
|---|---|---|
| The probe fits 0.3-1.1 GPU h | medium | Part count. The fine-tune range comes from options b2, not from a measurement. The re-score high end assumes no batching gain. CA4's top would raise the total to about 2.5 GPU h, still one night. |
| C2's batch-1 rate is at least 7.7x the CA10 assumption | high | The CPU draw was measured twice (102-104 ms and 149 ms), and the code draws on the CPU for every sample (`minigpt/laplace.py:156-171`) |
| At batch 1, $G_{m4}(3)$ is above 10 GPU h (17.9-22.3) | medium-high | It follows from the C2 cost. The other rates are forward-only lower bounds from report.md. |
| At batch 1, $G_{m4}(1)$ is at most 10 GPU h (6.0-7.4), so no escalation | medium | Same basis. The scoring path adds fp32 vocabulary work that report.md did not time. |
| Batching lifts the cap ($G_{m4}(3) \le 10$) | medium-low | It needs about 1.8-2.2x overall. C2's CPU draw is shared across the batch, which helps. But the fp32 vocabulary work grows with B, and batch 32 may go over 9,728 MiB. |
| The roadmap's m4 low end holds ($G_{m4}(1) \le 1.0$) | low | It needs about 6-7.4x over the measured batch-1 rates, or 3.3x over CA10 |
| The C0 anchor passes (±25%) | medium-high | Same machine, and the power limit is the default today. Risks: driver 591.86 may differ from the March driver, or another GPU process may run. |
| The 355M shape fits in 9,728 MiB at batch 4 | medium | My rough count is 5-7 GB: fp32 base 1.42 GB, cached fp16 weights about 0.7 GB, activations about 2 GB, logits and loss about 1.5 GB. It is not measured. |
| The CA5 ratio falls in [4.0, 5.0] | medium-low | The parameter ratio is 4.64. The small shape may be launch-bound at 2,048 tokens per step, which would lower the ratio. |
| The deterministic LoRA takes 26.9 min or less | medium | Same schedule as C3, with no sampling, no KL term and no periodic checkpoints. C3's 26.9 min is a single run. |
| Every Then gives a number or a yes/no, and names its script or test | high | All five rows name `scripts/timing_probe.py` or `tests/test_timing_probe.py`. The S1 hand-off is an exit code (S0-T5), not a manual copy. |
| The check runs before the Fri Oct 2 m4 run | medium | S0 provides and tests the check. S1 and the session-03 night list must call it. |
