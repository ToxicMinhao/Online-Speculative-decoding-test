# Full C58 qualification stop

## Material Passport

ARS experiment execution / evidence review; research stage unchanged. Executed source: `2897d483241e0c24b64deab31db09cc10669d9c7`, PR54. This report covers the first native attempt and CPU diagnosis, not a completed experiment. The preserved original failure remains a failure.

## Outcome

**The full campaign has not started.** CPU authentication, source staging, model/runtime identities, storage and hardware admission passed. The native qualification then stopped during the exact released-start initialization comparison, before startup forwards, loss computation, backward, AdamW or serving. The reported discrepancy is in the Python random state, not a measured loss, gradient or BF16 weight update.

The first eight reported paths are `state.rng.python.1.0` through `state.rng.python.1.7`. Fresh-boundary validation and released-export tensor equality had completed before this comparison. The checker raised before saving the candidate snapshot, so its complete RNG values and full discrepancy list were not retained. The eight displayed paths do **not** establish that every other RNG component or scientific field matched. That limitation is preserved explicitly.

## What the evidence supports

Original PR25 independently constructed its released initial state, compared it with K0 and recorded `rng_overwritten=false`; it did not load the witness RNG to manufacture agreement. A new CPU read of the authenticated original witness confirms its Python and NumPy states are exactly fresh seed17. Its Torch CPU state is advanced by model construction, so resetting every generator to17 after construction would be incorrect.

The new initializer still calls the original seed17 path and exact comparator. Import timing is a candidate explanation. On Fenrir, an import-only trace found a logging-library import consuming one Python `getrandbits(24)` draw after seeding. **The pinned A800 import-only control did not reproduce that effect:** the library was already imported before seeding, both original and preloaded routes retained Python state index624, and neither recorded a random draw during the inspected stages. The local trace is not an established explanation of the native failure, and no speculative preload fix was applied.

The A800 import controls used the pinned Python3.12.15 / Torch2.13.0+cu130 / Transformers5.12.1 runtime, with CUDA uninitialized and no model construction. They test imports only; they do not replay the failed constructor. We cannot yet identify which construction operation changed Python RNG or whether other RNG components also differed. There is no evidence here that C's mechanism helps or fails.

## Actual work and resources

| Measurement | First attempt |
|---|---:|
| Completed trainer initializations | 1 |
| Response forwards / backwards | 0 / 0 |
| Adam applications | 0 |
| Complete-response generations | 0 |
| Production events | 0 / 58 |
| CPU preflight | 30.246 s |
| Launcher wall time, including preflight | 60.925 s |
| Occupied GPU-card time | 25.506 s |
| Peak GPU allocated / reserved | 6.691 / 9.209 GB |
| Post-stop shared free space | 517.442 GB |

The code's reused runtime-compatibility receipt contains historical wording about three trainer constructions; it is a setting receipt, not actual work accounting. The actual work ledger records one constructor and no pending operation. Both physical GPUs were idle after shutdown. Nineteen retained evidence files were transferred and individually authenticated; an independent CPU audit passes50 work/resource/claim checks. No scientific retry has occurred.

The original campaign clock and failure records remain intact. CPU prelaunch inventory/cleanup is separate calendar work. Subsequent diagnosis is additional CPU work, not a reset of the original campaign clock or a new GPU allowance.

## Cleanup and remaining scope

Verified cleanup reclaimed **62.743 GB allocated** on A800 from eight completed PR53 recoveries and512 feature packets, plus **2.600 GB** on Fenrir from12 redundant published source worktrees. Final PR53 exports, selected paper checkpoints, controls, source and compact evidence remain authenticated. Exact retired optimizer continuation and feature replay are deliberately unavailable; fingerprints are not backups. Separately,1.903 GB of newly created redundant proof files was removed.

The sealed new-run estimate is71.011 GB under the unchanged100 GB ceiling; the shared reserve is400 GB. The full scientific work remains58C events/7,424Adam steps/7,473own responses, followed by18,432scored endpoint executions and576warmups. No baseline training or settings have changed.

## Next bounded action

Failure saving is now repaired in the review checkout: compact original/current RNG states and complete comparison differences are durable before the unchanged comparator raises. Ten focused CPU tests pass (four new failure-evidence tests plus six trainer tests), including preservation of the failing gate and exact passing behavior. Three additional CPU observer tests verify the manual tracer preserves RNG progression, handles nested seed calls and restores the profiler on failure. The overlapping seven-test tracer/evidence suite also passes. These changes have not been executed natively and do not fix or waive the RNG mismatch.

The prepared diagnostic code commit is `a5521481b29e76cf4c25b6944dbda3c958e5f204`; its manual entry is `experiments.full_c_gsm8k.initialization_trace`. The proposed next action is to trace global Python-random consumption through one exact native constructor, with **zero forwards, backwards, Adam steps or generations**. This diagnostic must charge its constructor/runtime to the existing ledgers and preserve the first attempt. A native retry requires the targeted review/retry decision; the full campaign remains stopped.

The previous **15–18hour execution estimate** remains a conditional planning estimate after the blocker is resolved, not a completion promise. The sealed raw estimate was16.61hours; measured qualification admission remains unfinished. Any resumption must recheck remaining work against the original24hour elapsed and40card-hour caps using the prescribed1.3×remaining-work plus30minute contingency.

## Evidence / interpretation boundary

- [Readiness and fixed design](READINESS.md)
- [CPU implementation review](IMPLEMENTATION_REVIEW.json)
- [Storage before/after](STORAGE_BEFORE_AFTER.json)
- [Qualification work summary](QUALIFICATION_ATTEMPT_001.json)
- [Failure-evidence CPU tests](INITIAL_EVIDENCE_CPU_TESTS.json)
- [Independent work/resource audit](INITIAL_FAILURE_WORK_AUDIT.json)

Private logs, exact RNG payloads, source/input bindings and failure records remain outside Git. This report supplies a concrete execution blocker, not a new scientific result or an authorization to weaken the equality gate. Native K+ remains primary; C's incremental utility is unresolved.
