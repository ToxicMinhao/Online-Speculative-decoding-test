# Implementation validation — correction bank v0.1

## Material Passport

- Status: CPU implementation checks passed; submitted for review.
- Scope: algorithm implementation, causal orchestration and reporting. No GPU qualification or model-performance result.
- Definition: user-supplied v0.1 specification, SHA-256 `453b142466d8c6d8025391cf3d942426460876e34aa5bcb80c3f78c7b8314e12`, with the three subsequent clarifications.
- Research status: existing stage unchanged; execution waits for review.

## Checks completed

| Check | Result |
| --- | --- |
| New correction-bank tests | 52 passed |
| Existing per-turn and follow-up tests, in the same invocation | 123 passed, 2 optional artifact/tokenizer checks skipped |
| Earlier two-update tests, in a separate invocation | 64 passed, 17 optional installed-source checks skipped |
| Approved six-stream package plan | Passed; 116 contexts, 110 updates per learning arm, 1,044 main timed executions |
| Python syntax, configuration JSON and Git whitespace checks | Passed |

The main invocation finished in 23.41 seconds. The earlier suite finished in
1.94 seconds. Test environment: Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3,
pytest 8.3.5 and a CPU-used PyTorch 2.11.0+cu130 installation, with CUDA devices
hidden and one BLAS/OpenMP thread. This is not the native runtime qualification.
The skipped historical tests require explicit local installed-source/tokenizer
paths; none of the new bank tests is skipped. No historical scientific source
was modified.

Coverage includes:

- Fixed projections, radius, first-two-turn fallback, exact fusion and the
  scale-relative error floor.
- All-owner evaluation, majorizer/radius acceptance, immutable cumulative
  protection and served-record insertion before candidate trials.
- Failed/nonfinite solver proposals, feasible no-ops, zero eligible exposure
  without resampling, and strict-JSON failure diagnostics.
- Compressed versus direct loss changes and gradients, finite-difference checks,
  exact sampled Fisher curvature, and direct historical-loss violations.
- Native teacher-head logits with explicit float64 probability normalization;
  differences from native probability rounding are recorded.
- Author anchor sampling, causal context masks, newest-response-only supervision,
  exact target-position alignment and retained RNG state.
- A three-turn training-service round trip with actual mathematical writes,
  independent historical-loss checks, immutable artifacts and no final write.
- Independent BANK/OSD histories, identical paired inputs, replay isolation,
  clean restart after qualification, mismatch retention and failure gates.
- Chunked-prefill descriptor routing, state activation, native correction hooks,
  request/adaptation cost boundaries and cumulative storage admission.
- Sum-based reporting, frozen fallback visibility, separate source-fit and
  transfer evidence, and a labeled reconstructed frozen episode-cost reference.

A synthetic full-size check also exercises four 64x64 modules, four owner
candidates, the second selected-owner step, the next read and snapshot recovery
with unchanged solver defaults. It took approximately 0.093 seconds, with about
93 MiB peak process RSS and a 2.37 MiB snapshot. This is a geometry/allocation
check, not an estimate of native training or serving performance.

## Test-harness repairs

Combining both historical directories in one pytest process initially exposed
their duplicate top-level test module names. They were run in separate processes;
historical test files were not changed. A new CLI test also needed to patch the
absolute native-module namespace actually imported by the runner. The final
commands above passed after that test-only correction.

## Still required after review

Native GPU qualification must check actual model features, masks/conditioning,
zero-output preservation, compiled corrections, compressed/direct objectives,
historical losses, source-fit capacity and full-context memory use. CPU tests
cannot substitute for that qualification. Useful transfer, activation frequency,
native output preservation and complete-runtime savings remain unmeasured.

Storage admission assumes no future offloading or deletion. With the pinned
draft size, its conservative full-retention requirement is approximately
125.58 GiB free, including qualification, six streams, rolling overlap and an
8 GiB reserve. Before execution, sufficient capacity or an explicit verified
offload/retention plan is required. No storage was deleted or moved for this
code-review submission.

Only task code, tests and documentation are submitted. Dataset records, model
weights, runtime credentials, the source PDF and experiment artifacts remain
outside the upload.
