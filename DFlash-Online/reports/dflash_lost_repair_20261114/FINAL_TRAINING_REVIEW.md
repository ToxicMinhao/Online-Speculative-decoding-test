# Final training-exposure review

Verification status: ANALYZED from retained compact evidence; no additional model execution. The reviewer authored the training-probe implementation and its focused tests, so this is a same-team audit with authorship overlap, not an independent native replication.

**Result: PASS within the reviewed read-only scope.** 742,225 CPU assertions passed on the sealed schedule, aggregate and individual presentation receipts, scope counters, native coefficient arithmetic and denominator coverage. No implementation or configuration changes were made during this review.

- Exact retained K+6 recovery SHA256: `15ac28e1235851a15158e789af1480fd2d82535903b1dec5d992ec843c9ae3a1`. INPUTS records step 768, 1,536 previous presentations, horizon 7,424, accumulation 2, response-mean reduction and empty accumulation. Its copied configuration rehashes to the recorded identity and is declared verbatim equal to the recovery configuration.
- Execution commit: `29b6a15c5f5f16135663ea61e3d31f21b3f1d465`. The prior source review confirms strict `draft_state_dict` loading and use of the saved `rng`, not a new seed or another checkpoint's schedule. POSTFLIGHT records unchanged inputs, checkpoints and source.
- Both passes contain each of the 128 scheduled requests exactly once. All 256 presentation identities and complete position/keep arrays exactly equal SCHEDULE.json and their individual durable receipts. The schedule digest matches the execution result.
- All 64,564 kept anchor occurrences are present exactly once per presentation (32,356 distinct response/start identities); no padded slots are counted. All 51,846 first errors, including 1,939 unavailable-successor cases, remain in the first-error denominator. There are 16,466 high and 33,441 low valid-successor occurrences.
- Independent arithmetic recomputed every response's K+ coefficient formula, every weighted first-error occurrence, positional decay, full selected weight and full first-error mass. Maximum coefficient difference is 0; maximum per-response first-error-mass difference is 3.17692757e-05; maximum selected-weight difference is 0.000699838177. All pass the existing mass tolerance, relative 2e-6 plus absolute 2e-7. Native response aggregates remain authoritative: total first-error weight 27646.563011169; total selected weight 288152.924682617.
- Exactly 256 native response forwards and 8,170 vocabulary-head chunks completed. Backward calls, optimizer creation/steps, temporary steps, teacher-head forwards and startup response forwards/backwards are zero. Budget attempted/completed training_forward counts both equal 256; failed and unknown work are empty.

## RNG and state evidence

The executed source independently generates a sampler-only two-pass schedule from recovery RNG, restores that RNG, then redoes normal CPU permutation and native CUDA anchor sampling during actual frozen forwards. Each anchor binding is checked before the draft forward; every presentation reports no RNG use after sampling. Completion requires the full final Python/NumPy/CPU/CUDA RNG to equal the sampler-only final state and caller RNG/modes to restore. Parameter and gradient version/identity checks also passed. These are source-backed execution checks: compact delivery does not include the full recovery tensor or final RNG tensor, so this audit does not independently regenerate the native CUDA draws or rehash the 6.3GB recovery locally.

## Cost accounting

Read-only forward/observation phase: 109.792s. Disposable sampler schedule: 1.651s. Worker initialization/input loading: 33.409s. Total worker phase: 144.853s; supervisor process phase: 150.098s. These nested times must not be summed as separate experiment totals. Peak allocated/reserved memory: 5.199/6.357GiB. The whole probe occupied 2853.504s of its 14,400s ceiling; this includes native census work outside this training-exposure review.

## Evidence and limits

Source receipts: `delivery/retained/campaign_001/training/INPUTS.json`, `SCHEDULE.json`, `RESULT.json`, `STATUS.json`, all 256 `presentations/*.json`, plus root PLAN/PREFLIGHT/POSTFLIGHT/BUDGET/RESULT. The audit checks serialized margins for finiteness, nonnegativity and tie consistency; it does not independently recompute those margins from private full-vocabulary logits. No GPU, remote operation, optimizer step or experiment retry was performed. Training exposure is a frozen-checkpoint measurement, not evidence that a candidate update repairs lost errors or improves serving.
