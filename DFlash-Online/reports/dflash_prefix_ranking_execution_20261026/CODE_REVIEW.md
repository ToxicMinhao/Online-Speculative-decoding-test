# Independent main U/P execution review

**Verdict: no blocking implementation finding remains.** The reviewed code is ready for the authorized one-event campaign once fresh remote access, resource, storage and source-seal checks pass. This review performs no remote launch, native generation or GPU training.

## Scope preserved

- Reuse the 128 retained frozen-feedback responses/features and original 256-presentation/anchor schedule. No feedback recollection or O retraining.
- Verify the completed controlled qualification, its exact equality gates, configuration/ordinary-receipt identity, pinned initial state and helper implementation. Do not invoke the superseded failed qualifier again.
- Execute one ordinary process warmup: two presentations, two backwards and one discarded temporary optimizer step, accounted separately. It neither exports a serving checkpoint nor supplies main-arm state.
- Construct U and P independently from released weights/fresh Adam. Restore the pinned complete pre-sampling RNG after construction, then verify initial weights, optimizer, counters, modes and RNG exactly. The existing trainer creates each event's reference afterward.
- Train each arm at lambda 1 for two passes: 256 presentations and 128 optimizer steps per arm, **512 presentations and 256 main steps total**. The unchanged sampler must reproduce the original ordering and every anchor draw; the receipt is an assertion, not a replacement sampler.
- Close training before loading the native serving backend. Evaluate F/O/U/P freshly on the same 64-request panel, three rotating repeats: **768 generations**. The first eight requests' 96 generations form an embedded correctness gate, without adding executions.

## Reviewed implementation boundaries

The new `controlled_startup.py` binds the previous passed qualification and all 27 equality receipts. The modified `run.py` uses it in place of the old failed qualification call. The shell supplies all five explicit qualification/pin/helper paths. Root preflight separately seals the unchanged scientific sources and reviewed orchestration changes; the old qualification does not imply approval of arbitrary later source edits.

A warmup cleanup issue found during review was repaired: model, observer, bound backend methods and closure references are cleared before garbage collection and cache release. Thus warmup tensors are not intentionally retained into the main constructors. No loss, precision, learning-rate, optimizer or sampling change was needed.

The unchanged evaluator verifies exact prompt IDs, output token IDs, termination, stop tokens, block/context geometry, cache policy, capture-disabled status and trace accounting. Publication checks invalidate caches and validate checkpoint identity; native weight equality checks cover all expected exported/native tensors. Fresh F/O verification counts must reproduce their retained first-event controls. Raw records are saved before interpretation; a correctness failure stops execution while preserving invalid/incomplete evidence. Valid null or harmful outcomes continue.

Reporting independently revalidates the durable rows and native records. It uses one trajectory per method/request for count analysis, all three timing repeats for latency, and pooled sums rather than averaged percentage changes. P versus O remains primary, with P versus U and P versus F included. All regressions remain visible. Timed request latency includes prefill, drafting, verification and request cache work, while excluding training and publication.

Loading, initial-state verification, ordinary warmup, training, reference computation, persistence, export, activation and complete campaign time remain separately recorded. Reference costs are nested within event cost and must not be added twice. Reused feedback/teacher preparation has zero **new** cost, not zero historical cost. Benchmark checkpoint switching is a diagnostic cost, not deployment adaptation cost.

## Independent CPU verification

Final focused suite: **25 tests passed in 12.622 seconds**, with `CUDA_VISIBLE_DEVICES=''`. This includes six startup tests, sixteen evaluator/report tests and three accounting tests. The startup fixture runs real tiny-model U/P events of 128 steps each, validates the full original schedule, checks independent initial state despite disturbed constructor RNG, and rejects changed initial weights. Orchestration coverage forbids invoking the old qualifier and verifies warmup → U → P → native evaluation order. Evaluation coverage includes the full 768-execution fixture, exact validity failures, harmful/null continuation, balanced rotation and complete incomplete-work accounting.

An earlier focused run had 24 passes and one test-only `NameError` from misplaced fixture code. It was corrected without changing production behavior; the failure log remains in `INDEPENDENT_CPU_TESTS_FIXTURE_FAILURE.txt`. Final output is in `INDEPENDENT_CPU_TESTS.txt`. CPU fixtures do not establish native main-run output preservation or the efficacy of ranking.

Command and environment:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree:/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree/tests:/home/minhao2/DFlash-Research/training:/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_readiness_20261025/source_snapshot/inference \
/home/minhao2/.venvs/dflash-specforge/bin/python -m unittest \
  test_prefix_ranking_controlled_startup test_prefix_ranking_evaluation test_prefix_ranking_run -v
```

Reviewed SHA-256:

| File | Digest |
|---|---|
| `experiments/prefix_ranking/run.py` | `472f0e67a84aa885505ff13c8ce094e2800c0198fec9daf72477f388024dbc58` |
| `experiments/prefix_ranking/controlled_startup.py` | `5cce6c0bbd9870bd4ddad5bd9c4d64dfc49fbdfb0c74c27d25998972147e62be` |
| `tests/test_prefix_ranking_controlled_startup.py` | `2b378b017052ea20549c17087456b526787b9275320141c33b8a3371f082089f` |

## Interpretation boundary

Controlled startup establishes exact ordinary/lambda-zero behavior for the qualified one-step window. It does not retroactively pass the original failed gate, identify its kernel-level cause, or establish equality to the retained O artifact's full historical 128-step trajectory. That limitation is preserved in the startup receipt and must remain in the final comparison report. The approved experiment reuses the historical O artifact; this review does not authorize retraining it, adding events, tuning lambda or changing the numerical protocol.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
