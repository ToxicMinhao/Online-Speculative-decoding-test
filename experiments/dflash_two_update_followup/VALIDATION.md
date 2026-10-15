# Implementation validation — October 15, 2026

CPU implementation checks and the subsequently authorized native GPU qualification
have passed. These are execution/correctness checks, not scientific efficacy
results. The full ten-session campaign has not started.

## Completed checks

- **50 follow-up tests passed, zero skipped.** Includes pinned-author float32
  and bfloat16 sampler/mask/loss/gradient comparisons, response-specific masking,
  future-history isolation, persistent Adam/RNG, fixed evaluation windows,
  original-versus-own trajectory evidence, retained-checkpoint validation,
  capture-disabled timing and nonoverlapping cost reporting.
- **81 existing two-update regression tests passed, zero skipped.** Installed
  source checks were enabled, including chunked-prefill ownership, full-context
  guards, publication accounting, native source pins and replay interfaces.
- All ten original compact R1/R2 result bundles passed reference validation
  against their exact dataset rows. No original weights were loaded and no
  archived response was regenerated.
- The actual metadata-only CLI plan completed, admitting ten development
  sessions and reserving R3/R4. Its combined matrix has 1,680 checkpoint scores;
  the final F/C1/C2 comparison is reused rather than run twice.
- `git diff --check` passed. No model, dataset, experiment output or private
  runtime configuration is included in this code change.

CPU numerical checks used Python 3.10 and PyTorch 2.11.0 with CUDA devices hidden.
The build has CUDA support, but these tests execute only CPU tensors. Enable
`ADAFLASH_SOURCE_ROOT` for the pinned author differential tests and
`TWO_UPDATE_WORKSPACE` for the existing installed-source regression checks.
Missing dependencies are explicit skips, not passes.

## Native GPU qualification

The reviewed implementation at `ae6be2b6169d583023abc941de769a4d66705e97`
completed the bounded qualification on two 80 GB A800 GPUs. The same 131 CPU tests
also passed in the installed A800 environment with zero skips.

- All 29 native serving calls preserved the required exact outputs and weight
  identities, with feedback capture disabled. The sequence exercised frozen,
  retained C1/C2 and newly trained A1/A2 publication and cache refresh.
- Two disposable updates completed eight optimizer steps each, maintaining the
  same draft, Adam state and anchor RNG. Each used two passes over its eight fresh
  responses; only each example's current response received supervision.
- Native teacher features took exactly one target forward for each of 36 calls.
  The target and its head/embedding remained frozen.
- Maximum-input serving and training backward passed at 31,744 input + 256
  response tokens, including a backward check with Adam states resident. The
  nine extra draft-only padding slots were masked. Cumulative training-process
  PyTorch peaks were 33.90 GiB allocated / 37.24 GiB reserved, not whole-device
  measurements.
- The actual author-wrapper BF16 differential had 0.1069% relative loss difference
  and maximum 2.8538% relative gradient L2 error over 58 parameters, within declared
  tolerances (loss rtol 1%, atol .001; gradient relative L2 5%). A 40-token
  last-token perturbation preserved all earlier teacher feature rows exactly.
- Successful attempt wall time was 472.76 seconds. Worker cleanup passed and
  both GPUs were idle afterward.

The first disposable qualification launcher failed before any inference/training
because its multiprocessing re-import did not register the private native module.
Its wrapper was repaired and independently checked before retrying. Both failed
and successful attempts are retained privately with raw measurements, exact source
and configuration hashes. The production learning/decoding source and scientific
settings were unchanged. No models, request data or raw result bundle are uploaded
with this documentation.

A800 used Python 3.12.14, PyTorch 2.9.1+cu130, Transformers 4.57.1 and Triton 3.5.1;
the qualified SGLang development installation remains bound by the existing source
manifest. Local CPU checks above used their separately declared environment.

## Limits and next gate

The source review and qualification confirm that teacher supervision uses the normalized final layer and all five
draft-conditioning layers. Teacher replay is the author's full-sequence extend;
the serving engine independently retains its 8,192-token chunked-prefill setting.

The native qualification gate is satisfied. The full campaign still needs its
own storage admission and declared three-repeat comparison. Disposable updates,
synthetic capacity fixtures and single-call qualification timings do not establish
future drafting benefit or a training-cost advantage. R3/R4 remain reserved.
