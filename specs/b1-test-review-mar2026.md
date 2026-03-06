# B1 Test Review - `tests/test_laplace.py`

Date: 2026-03-06  
Scope reviewed: `tests/test_laplace.py` (TDD Stage #2 for B1)

## Summary

`tests/test_laplace.py` is a strong unit-test start for B1 Laplace internals (curvature, sampling, state
roundtrip, selection scope, side effects). The main gaps are integration-level coverage and strictness in a
few assertions.

## Findings (ordered by severity)

1. High - "MI pipeline integration" test does not exercise the real pipeline
- File: `tests/test_laplace.py` (`TestMIPipelineIntegration`)
- Issue: The test recomputes MI manually instead of calling existing uncertainty APIs in
  `minigpt/uncertainty.py` (`compute_uncertainty_metrics` / `score_sequence`).
- Risk: Can pass while true pipeline wiring is broken.

2. Medium - B1 end-to-end acceptance behavior is not covered
- Spec reference: `specs/b1-laplace-tech-spec-mar2026.md` (B1-4 + acceptance criteria)
- Missing coverage:
  - checkpoint -> Laplace fit/load -> MI eval integration smoke
  - logging contract checks for Laplace + standard uncertainty metrics
- Risk: Unit tests pass while experiment/script integration fails.

3. Medium - `select_params` mode tests are under-constrained
- File: `tests/test_laplace.py` (`TestSelectParams`)
- Issue:
  - `head` mode: only checks `len(selected) >= 1` and name substring.
  - `all` mode: checks only "has some MLP and some attention".
- Risk: Incorrect extra/missing params can slip through.

4. Low - context-manager exception safety is untested
- File: `tests/test_laplace.py` (`TestParamSelectionScope`)
- Issue: Restoration is checked only on normal exit from `apply_sampled_params`.
- Risk: Exception path could leak sampled params into model state.

5. Low - a couple of tests are potentially flaky
- File: `tests/test_laplace.py`
- Cases:
  - `different seed => different samples`
  - `MI mean > 0` on tiny synthetic random data
- Risk: Intermittent failures depending on implementation details.

## Coverage vs B1 behaviors

- B1-1 Curvature fitting: Covered well.
- B1-2 Posterior sampling: Covered well.
- B1-3 Uncertainty evaluation: Partially covered (math path), not via canonical uncertainty pipeline.
- B1-4 End-to-end script: Not covered in this file.

## Recommended follow-up tests

1. Add a pipeline-wiring test that calls `compute_uncertainty_metrics` (or `score_sequence`) with Laplace
   sampling applied, instead of manual MI math.
2. Add a lightweight integration smoke test (`tests/test_b1_pipeline.py`) for:
   - load deterministic checkpoint
   - fit/load Laplace state
   - run uncertainty eval
3. Tighten selection-mode assertions:
   - exact expected set for `head`
   - explicit include/exclude lists for `all`
4. Add exception-safety test for `apply_sampled_params`:
   - raise inside context
   - assert full parameter restoration in `finally`.
5. Stabilize stochastic assertions:
   - compare aggregate deviation thresholds (or repeated trials), avoid brittle single-shot checks.

## Notes

- Current red state is expected in TDD: `minigpt.laplace` is not implemented yet.
- Local run also hit sandbox temp-dir permission errors for tests using `tmp_path`; this is environment-related
  and separate from test design quality.
