# Projected-target component: implementation and CPU qualification

Date: 2026-10-22. Scope: the first objective modification in the supplied
[handoff](HANDOFF.md), using the newly qualified online-feedback foundation.
ARS experiment workflow guided bounded implementation and independent mathematical,
integration and test reviews. No dataset selection, GPU generation, scientific
campaign, dependency installation or GitHub publication occurred.

## Delivered code

Active A800 root: `/data/sources/minhao/DFlash-Online`.
Local review copy: `worktree/` beside this report.

| File | Purpose |
| --- | --- |
| `online_feedback/projected_target.py` | 67 lines including documentation: fixed c=1.1 preservation/repair targets, detached FP32 water filling. |
| `online_feedback/objective.py` | 166 lines: matched hard/soft losses, immutable pre-event reference, shared author anchors/conditioning, bounded checkpointed chunks and work timers. |
| `online_feedback/training.py` | Small opt-in integration: objective selection, verified canonical labels, reference lifetime and resume identity. |
| `tests/test_projected_target.py` | 11 mathematical and numerical checks. |
| `tests/test_objective_integration.py` | 8 real tiny-DFlash forward/backward checks. |
| `tests/test_target_event.py` | 8 event-boundary checks using the real TrainerCore/BF16Optimizer with an instrumented scalar fixture. |
| `README.md` | Usage, debugging entry points and accounting boundaries; preserves the completed upstream GPU qualification record. |

Only these seven files were installed. Inference, buffer/controller, upstream
SpecForge, existing configurations and both clean DFlash-Research copies remain
unchanged. The default `objective="upstream"` preserves the existing training path.
Use `objective="hard_ce"` for the matched FP32 control and `objective="projected"`
for the new target. Reuse the existing controller and event trainer; no second
feedback loop was introduced.

## Actual update procedure

1. Existing serving/target replay supplies exact response tokens and verified
   target argmax IDs. The new objective checks that binding before training.
2. At event start, copy the current pre-update draft once. Freeze/detach this
   reference; share the already frozen target embedding and vocabulary head.
3. Each presentation uses one author anchor draw. Both current and reference
   forwards use those exact anchors and the same context/attention construction.
4. Correct reference decisions preserve q0. Incorrect decisions receive the
   handoff's projected distribution with c=1.1. Masks, decay, per-response
   normalization and accumulation remain shared with hard CE.
5. Backpropagate through the current draft and use the existing optimizer.
   Reference inference does not advance training RNG. The reference stays fixed
   across every response, pass and optimizer step in this event, including new
   sampled anchors, and is released after backward finishes.
6. Existing checkpoint/export/publication handles the updated draft. The reference
   is unregistered and absent from optimizer state and checkpoint keys. A later
   event captures a new pre-event reference from the then-current draft.

Model/head arithmetic remains native BF16; softmax, projection and loss arithmetic
are FP32. Argmax uses original logits and the lowest token ID on ties. Underflow
zeros are retained without flooring or renormalization. Probability/margin test
atol is 2e-6; independent FP64-oracle/soft-CE-gradient checks use atol 2e-7,
rtol 2e-5. Full-vocabulary and extreme finite-logit cases are included.

## Validation

- Local: **113 tests passed**, 3.274 seconds test execution; [log](local_tests.log).
- A800 environment, explicitly CPU-only: **113 tests passed**, 3.277 seconds;
  [log](a800_tests.log).
- Total consists of 86 existing tests plus 27 new tests. Existing upstream
  optimizer/restart/evaluation tests remain passing.
- Real tiny-DFlash tests cover loss/mask/denominator equivalence, reference
  immutability, matched anchors/RNG, backbone gradients, frozen target/reference,
  and chunked versus single-chunk results in FP32 and BF16.
- Event tests verify lifetime across all backwards, cleanup on failure, optimizer
  continuity, recorded costs, target provenance and objective-aware resume.
- Independent integration review found no blocking issue. Its two documentation
  findings (overlapping timers and stale README status) were resolved.

These tests do not qualify full-model CUDA memory fit, DDP/native export/reload or
serving performance for the new objective. The prior successful GPU loop test
used upstream hard CE; it must not be relabeled as projected-target qualification.
No adaptation-benefit conclusion follows from these CPU results.

## Costs and storage

One reference draft copy adds event-local memory. Sorting full-vocabulary
probabilities and checkpointed loss recomputation add work; small source-code size
does not imply negligible training cost. No response-by-vocabulary target cache
is retained.

`objective_work.reference_setup_s` is outside `training_s`.
`reference_forward_s`, `reference_head_s`, and `projection_s` are already inside
`training_s`, including recomputation during backward. They must not be summed
onto training time again. Event wall time includes setup and all event work;
checkpoint persistence/export/publication retain separate accounting. Component
timers synchronize CUDA and are diagnostic cost measurements.

A800 before: 26,579,922,944 free bytes. After source/tests: 26,579,755,008 bytes.
Both GPUs remained idle; no deletion was needed or performed. Fenrir had
85,299,642,368 free bytes at the checked point. Existing user debugger processes
were left untouched. Old source is preserved in `before.tar.gz`, locally and at
`/data/sources/minhao/runs/projected_target_component_20261022/before.tar.gz`.
The seven-file update and exact transfer list are retained as `source_update.tar.gz`
and `TRANSFER_FILES.txt`. No SHA checks were added to the runtime.

## Pending before the feasibility campaign

Review this component, then bind the separate eight-step campaign configuration
and its actual LR schedule, seal the 48-request manifest and artifact identities,
and admit peak storage before native execution. The current four-step
infrastructure config is not the eight-step handoff config. The training context
limit is 2048; the planned rendered-prompt/output caps are 1024/512, subject to
explicit native position accounting. The campaign's first complete group must
qualify the new objective within the declared 144-generation budget. No broader
mechanism, tuning or historical campaign restart is implied by this delivery.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_component_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
