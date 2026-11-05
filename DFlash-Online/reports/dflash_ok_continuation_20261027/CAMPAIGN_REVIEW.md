## Material Passport

- Origin Skill: ARS academic-research-suite, experiment-agent run/validate.
- Scope: independent source review of the continuation orchestration, plus CPU-only mocked execution.
- Verification Status: ANALYZED for scientific execution; 15 executable CPU orchestration checks passed. No GPU or remote operation, model import, scientific forward, optimizer update or generated response.
- Review provenance: this reviewer authored the continuation data preparation and earlier objective implementation, but did not author the campaign driver under review. This is not an external or blind review.

The reviewed orchestration has **no remaining launch-blocking issue found within this scope**. After the stronger sealing and runtime checks, all 15 tests pass in [CAMPAIGN_CPU_TESTS_RESEAL.log](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/CAMPAIGN_CPU_TESTS_RESEAL.log); current source compilation and hashes are in [CAMPAIGN_SOURCE_CHECK_RESEAL.json](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/CAMPAIGN_SOURCE_CHECK_RESEAL.json). Both shell wrappers pass `bash -n`. Earlier passing logs/source checks remain retained separately.

The full mocked campaign exercises three 128-request windows. Each window is generated once using O's checkpoint before that event, with no K-generated collection. O and K receive the same immutable feature paths and bytes; K receives O's presentation/anchor reference receipt. Each arm restores its own prior state and advances 128→256→384→512. Both updates finish before the next collection, and trainers close before evaluation. Only events 2 and 4 are evaluated. Total mocked calls are exactly 384 feedback plus 768 evaluation generations, with 768 new optimizer steps and 1,536 presentations; event 1 is never repeated. Initial state files remain unchanged.

Two failure-accounting defects found during review were fixed by the parent before these passing tests: top-level counters now reconcile partial trainer/evaluator work from durable child records, and trainer cleanup preserves an existing training exception while retaining its own cleanup error. Tests cover interrupted feedback, partial training, partial evaluation, trainer and backend cleanup failures, including cleanup failure after otherwise completed work. Unknown and unattempted generation work remains visible. An interrupted optimizer call is not assumed to have done no work. Failure recovery retains the previously committed state and partial counters/errors; the driver does not claim a complete snapshot of failed live state or automatic retry capability.

Negative tests also reject repeated inputs, event-1 replay, a changed learning horizon, bad checkpoint publication, changed shared features and ineligible feedback without replacement. An existing campaign directory is rejected without overwriting its evidence.

The actual seal/preflight implementation is tested inside an isolated fake remote directory tree, with model/native operations mocked. It binds the correct original `dflash_osd_one_event_20261025/sealed/MANIFEST.json`, both own complete parent-state paths, new continuation source, windows, runtime and dependency paths; preflight and postflight pass, then reject a changed sealed source file. These mocks verify path and integrity orchestration, not actual remote artifact availability or real tensor equality.

The strengthened sealing gate additionally rejects five independently corrupted fixtures: incomplete selection, mismatched prepared-output hash, changed request order, changed parent event receipt, and changed historical O restart bytes. Parent receipt identities are checked against the earlier seal; the complete O state is checked against its retained historical digest. K's full state receives a fresh identity and structural/export/RNG/Adam validation; no missing historical K full-state digest is invented. Runtime verification now runs at both preflight and postflight, which the integration test explicitly counts and compares. Loading memory peaks and training-only peaks are logged separately and checked by the mocks.

The old fixtures first failed because they lacked the new `SELECTION_AUDIT.json` and mock `cuda.is_initialized` capability. Those expected fixture failures are preserved in [CAMPAIGN_FIXTURE_MISMATCH.log](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/CAMPAIGN_FIXTURE_MISMATCH.log); updating fixture contracts fixed them without changing scientific code or tolerances. They are not GPU or production experiment failures.

Storage admission includes native/prepared feature rows, full state and export writes, atomic overlap, a 10% tensor margin, 2 GiB compact evidence, 2 GiB work allowance and 8 GiB reserve. Tests verify its exact threshold and durable rejection before generation. Admission deliberately covers the next phase only; later phases account for then-current free space and actual collected lengths. Keeping all native/prepared pools means the first admission does not guarantee every later worst-case phase will fit. The driver must stop at a later storage gate if needed; this review does not substitute for remote free-space checks or authorize deletion.

The real numerical resume gates and retained-F native evaluator are separate components with their own tests/reviews. The orchestration harness replaces those dependencies before importing the driver and cannot establish GPU/Flex arithmetic, actual checkpoint/RNG equality, native output equality, the first-48 execution gate or efficacy. No scientific API or qualified setting was changed for these tests. Existing research artifacts are preserved, and Stage 1/project None remains unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
