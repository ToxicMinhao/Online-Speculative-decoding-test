# Controlled initialization and active-ranking qualification: passed

## Material Passport

ARS experiment-agent run/validate; October 26, 2026 (America/Edmonton).
Authorized deliverable: bounded correctness qualification only. A800 supervisor
2140570 exited 0. No main U/P training, serving generation, teacher replay or
new feedback collection occurred. The original failed qualification is preserved.

## Result

**Both gates passed.** With the declared one-step process warm-up followed by
fresh model/optimizer construction and restoration of the pinned complete
pre-sampling RNG state, ordinary OSD repeats and ranking-disabled U/P match
exactly. Active U/P also pass the implemented mask, gradient, reference-isolation,
finite-state and memory checks. No numerical tolerance, objective, optimizer,
learning rate, attention implementation, anchor rule or data setting was changed.

| Trial | Outcome |
|---|---|
| Ordinary warm-up | Two retained presentations, one disposable step; no learned state inherited. |
| Oa | Fresh released state validated; original first-window scalar metrics reproduced. |
| Ob | Exactly matches Oa at initial state, both backward boundaries, pre-optimizer gradients, final state and numerical records. |
| U0, ranking disabled | Same exact Oa comparisons pass. |
| P0, ranking disabled | Same exact Oa comparisons pass. |
| U1, ranking active | Prefix, coefficient allocation, full-block alignment, rival gradients, fixed reference and RNG checks pass. |
| P1, ranking active | Prefix, coefficient allocation, rival gradients, fixed reference and RNG checks pass. |

There are 27 saved exact-comparison receipts, all passing. Initial comparisons
include draft parameters/buffers, fresh Adam, modes, counters and Python/NumPy/
Torch CPU/CUDA RNG. Every trial uses the original first-pass 128-response
permutation and its first two recorded anchor draws. The two distinct responses
are `train_00072` and `train_00121`; fourteen presentations are their repeated
use across seven trials, not fourteen independent responses. The entire retained
256-presentation schedule is validated as an artifact; only the first two anchor
draws are executed here.

## Active-ranking evidence

| Response | Sampled blocks | Supervised position occurrences | Protected positions | Weighted protected fraction rho |
|---|---:|---:|---:|---:|
| train_00072 | 363 | 5,340 | 2,710 | 0.63790023 |
| train_00121 | 411 | 6,060 | 2,526 | 0.55400604 |

U1 and P1 produce identical reference masks and rho values. An independent
sequential oracle matches every reference mask position. U/P weighted coefficient
mass agrees under the existing tolerance; rho is weight-based, not the unweighted
protected-position fraction. The existing CPU fixtures separately cover seeds,
first mismatches, later matches, invalid positions and EOS boundaries.

Four detached eval/no-grad forwards compare reference/current full-block hidden
features for the two retained draws, with exact equality. Each active arm also
passes two strongest-rival gradient checks: eight rows per presentation across
the full 151,936-token vocabulary. All 32 row-check executions have zero observed
error against the analytic expected gradient. These are bounded implementation
checks, with repeated inputs across arms, not independent statistical samples.
The projection is BF16; margin subtraction and softplus are FP32. Existing
BF16 gradient tolerances remain rtol=0.016, atol=1e-5.

The reference remains independently stored, frozen, in eval mode, gradient-free
and equal to the released weights after the temporary optimizer step. Its
construction and forwards preserve current-model modes and complete RNG.
Reference activations are released before current autograd. Frozen target table
version/layout guards pass. All inspected gradients, parameters and optimizer
states are finite. Active updates are intentionally not required to equal OSD.

## Work and cost

- Exactly **7 temporary optimizer steps, 14 response presentations and 14 backwards**.
- **14 current training forwards, 4 active-reference forwards and 4 detached comparison forwards**.
- Internal work is separately recorded: 70 current-layer forwards including
  recomputation, 20 observed reference-layer forwards and 2,940 observed vocabulary
  head calls including recomputation. These are implementation counters, not
  additional generations or teacher-backbone forwards.
- **0 main optimizer steps, 0 generations, 0 new teacher-model forwards**.
- Qualification driver: **586.17 s (9m46s)**. Supervisor including preflight and
  A800 CPU tests: **643 s (10m43s)**, 08:21:09–08:31:52 UTC / 02:21:09–02:31:52 MDT.
- Summed model loading: 135.29 s. Instrumented response-training spans: 199.76 s,
  **including** 141.63 s of gradient-snapshot persistence. These overlapping
  timings must not be added together. State checks and other persistence account
  for further qualification time; this is not a production training benchmark.
- Peak CUDA allocated: **23.250 GB (21.65 GiB)**; peak reserved: **26.521 GB**.
  Loading peaks are separately retained. Active U/P allocated peaks are 23.250
  and 23.178 GB. This qualifies the tested shapes, not every possible larger input.
- Raw qualification evidence occupies 102.769 GB (95.71 GiB). It remains on A800;
  only the compact evidence package was retrieved to Fenrir. No storage was deleted.
  A800 has 216.475 GB (201.61 GiB) free afterward; both GPUs are idle.

The runner has no checkpoint-export, publication, generation or main-campaign
entry. Temporary states remain diagnostic evidence and are never serving or
main-training initializations. The warm-up trainer is destroyed before Oa.

## Review, validity and retained artifacts

Eighteen existing helper/ranking CPU tests passed. Five new CPU tests passed,
including full seven-window execution and intentional initial/post-step failures
that preserve evidence and stop. The five new tests also passed independently
and on A800 (11.335 s). An initial local fixture invocation left GPUs visible and
failed because CUDA RNG appeared after its CPU pin; this test-isolation issue is
recorded in CODE_REVIEW.md. Corrected CPU launchers hide GPUs, and the fixture
forbids CUDA initialization. No GPU equality requirement was weakened.

Preflight pinned models, runtime, configuration, retained data and code. The final
hash audit confirms **105 production files and 132 retained O artifacts unchanged**,
plus the new qualification files and both pinned diagnostic dependencies.
A separate internal evidence audit passed 78 checks against the compact receipts;
it did not download and independently recompare the model-sized remote tensors.
No edits were made to DFlash-Online production code or Fenrir DFlash-Research.

Artifacts:
- `runner.py`, `test_runner.py`, `SEAL.json`: implementation, tests and exact inputs.
- `raw/qualification/RESULT.json`, `WORK.jsonl`, per-trial receipts: raw work,
  exposure, gradient checks and all reached exact gates.
- `COMPARISON_INDEX.json`: all 27 exact-comparison receipts.
- `QUALIFICATION_TRIALS.csv`: per-trial costs, memory and check coverage.
- `raw/FINAL_RETENTION_AUDIT.json`: source/artifact preservation and final resources.
- `CODE_REVIEW.md` and `RESULT_AUDIT.md`: separate internal code/evidence review.
- Full tensors: `/data/sources/minhao/runs/dflash_prefix_ranking_controlled_qualification_20261026/qualification/`.

## Interpretation and stopping point

The controlled startup makes the checked ordinary/ranking-disabled executions
exactly agree. Warm-up and complete RNG restoration were introduced together;
this does not identify the precise operation responsible for the earlier
first-versus-later discrepancy or attribute the effect separately to either control.
The old failed gate remains failed under its old procedure.

This is one-window correctness evidence. It does not establish equivalence to
the historical O checkpoint's entire 128-step numerical trajectory. Any later
reuse of that immutable checkpoint must retain this initialization-procedure
limitation. Native output preservation and useful drafting/latency gains for
trained U/P remain untested here because there were no generations.

**Stop reached: both qualification gates pass; main U/P training and the
768-generation F/O/U/P comparison remain unlaunched.**


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_controlled_qualification_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
