# Completed AdaFlash C1 training, cost and retention audit

## Material Passport

- **Status: ANALYZED; saved-record audit passed.** 8,321,740 checks, zero discrepancies.
- Execution commit: `681336c5e5c5caa0aa67560e46ffe83ce81e9351`; published PR #29 code.
- Audit finished: 2026-11-06T21:40:23.044736+00:00; 21.963 seconds of read-only CPU work.
- Scope: saved A800 records under `/data/sources/minhao/runs/dflash_adaflash_c1_20261105/production_001`.
- Same-team auditor wrote the campaign implementation; authorship overlap is disclosed. The arithmetic reconstruction does not call the experiment's report functions.
- No model forwards, gradients, optimizer steps, new generations, tensor loading, cleanup or source edits were performed.
- Evidence: [machine-readable audit](AUDIT_FULL.json), [independent audit script](audit_training.py), [recomputed per-event costs](EVENT_COSTS_RECOMPUTED.csv). The earlier, narrower successful pass remains in `AUDIT.json`.

## Work and state accounting

The work log has exactly matched, sequential started/completed entries for **7,473 online generations, four inherited correctness generations, 7,424 terminal teacher-preparation forwards, 14,848 training forwards/backwards and 7,424 native Adam steps**. No unknown work is present. The last 49 requests use A58 with capture disabled and no training.

All 58 events contain exactly 128 new feedback identities, two shuffled passes, 256 presentations and 128 updates. Each optimizer window averages its own two separately normalized response losses. Each step retains LR 3e-4. The final state is at step 7,424 with 14,848 presentations. Initial and all 58 boundary receipts report exact model/export/optimizer/RNG integrity; no disposable optimizer steps are recorded. Full tensor equality is supported by those retained execution receipts, not freshly re-established by this audit.

All 7,424 training responses match the sealed request order and exact prompt IDs, were generated from A's correct preceding checkpoint, and retain exact token/teacher-preparation bindings. The final 49 request identities and A58 checkpoint identities also match. All 14,848 naturally sampled anchor/order records match the historical comparator records, without a forced schedule. This does not establish identical hidden features between arms.

Six tiny GPU qualification cases recorded 12 synthetic loss forwards and backwards, zero model forwards and zero optimizer steps. Their retained receipt passed. Actual objective instrumentation totals 14,848 draft forwards, 74,240 layer forwards, 944,660 projection-loss chunks and 1,889,320 vocabulary-head calls. Chunk counts include native checkpoint recomputation; they are not additional response presentations.

## Exposure and clipping

| Quantity | Recomputed total |
|---|---:|
| Response presentations | 14,848 |
| Active sampled anchors | 3,728,050 |
| Supervised position occurrences | 54,368,391 |
| Vocabulary entries receiving diagnostics | 8,260,515,854,976 |
| Upper-clipped entries | 897,525,502 |
| Positions with at least one clipped entry | 44,662,694 |
| Negative reverse-KL entries left unchanged | 39,748,176,264 |

The clipping rate is **0.01086525% of eligible vocabulary entries**, or **82.148273% of supervised positions** with at least one clipped entry. Denominators differ and must not be interchanged. Position occurrences were independently reconstructed from each anchor, the 15-proposal block and the saved sequence boundary. Each original response observation contributes exactly once; optimizer accumulation rows and checkpoint recomputation do not inflate exposure.

The recomputed teacher argmax differs from the saved canonical token at **616 sampled occurrences across 46 response presentations**, 0.00113301% of supervised occurrences. All native preparation labels still match canonical outputs. As specified, the CE component uses the recomputed frozen-teacher argmax. This counter is a training-label diagnostic, not a native-serving validity failure, and this audit does not newly establish its numerical cause.

All saved numeric response/window metrics are finite. Minimum response mixture loss is 0.035171311; no negative total response losses were recorded. Clipping counts inherit native FP32 diagnostic accumulation; loss values and gradients cannot now be recomputed from retired feature tensors.

## Measured costs

| Cost | Sum (seconds) |
|---|---:|
| Training-step time, independently summed over presentations | 18398.732252 |
| Frozen teacher-head work, nested inside training | 5944.035382 |
| Checkpoint save | 1106.264240 |
| Export | 896.716344 |
| Terminal teacher preparation | 290.857885 |
| Prepared-feature persistence | 336.573791 |
| Activation, including native refresh | 85.222067 |
| Native activation sub-timer | 21.275417 |
| Whole update call, encompassing several rows above | 21517.471817 |
| Publication/state validation | 1465.547601 |
| Rolling retention | 540.700250 |
| Entire A worker, including collection and final tail | 31265.087089 |

The trainer loaded in 18.113522s; native backend loading took 19.257000s; initial scientific-state validation took 10.211775s. Peak training-device memory was **19.254281 GiB allocated / 22.337891 GiB reserved**. This is the measured trainer peak, not combined two-GPU memory.

Feedback collection for the 7,424 responses used in training took 5373.033030s in summed complete-request latency, including 4965.903223s native generation. It generated 1,907,459 output tokens with 8 cap hits. Capture-host and native-feature-persistence timers are nested in native serving; they must not be added to complete-request latency.

Costs reconcile with `results/TRAINING.json`, the 58 original `RESULT.json` receipts, per-presentation records, teacher-preparation receipts and activation records. Compiler timers and teacher-head clocks are nested. Do not add every row into a deployment total or treat a residual as a kernel-time breakdown. Retained O/K+ training times, if shown, are historical context rather than a fresh matched cost comparison.

## Retention and remaining capabilities

Exactly **22,386 distinct tensor paths** were retired after the recorded replacement/successor checks:

| Kind | Paths |
|---|---:|
| Raw native capture | 7,424 |
| Temporary terminal features | 7,424 |
| Prepared response packets | 7,424 |
| Previous event optimizer/model states | 57 |
| Previous event serving weight files | 57 |

Every retired path is absent, every removal matches its prewritten plan, and **59,864 unique retained compact-evidence files match their recorded byte counts and SHA-256 identities**. The cumulative logical size retired is 719,335,880,035 bytes; this is turnover across the trajectory, not simultaneous disk usage or a measured increase in free space.

A's initial state, final A58 complete state and final serving export remain. Intermediate event1–57 tensor-level replay and optimizer continuation are unavailable. Original native/terminal/prepared feature tensors were deleted, not archived. Compact tokens, anchors, measurements, preparation-equivalence receipts, state-validation receipts and source provenance remain. No additional cleanup was performed by this audit; final observed A800 free space was 128.881680 GB.

## Interpretation safeguards

All 11 ARS statistical-fallacy categories were considered for this accounting review: Simpson's paradox and ecological inference do not arise from a descriptive work sum; benchmark-selection limits remain; no post-outcome covariate conditioning or base-rate classification is used; no extreme-response selection or regression-to-mean claim is made; failed/unknown work is checked rather than discarded; clipping/activity counters are not selectively promoted to efficacy outcomes; the fixed endpoint and loss are unchanged; no causal mechanism or reverse-causality claim is inferred from the counters. Efficacy intervals and per-request native outcomes belong to the separate endpoint audit.

**No training/accounting/cost/retention blocker was found.** This establishes consistency of retained evidence, not a second GPU execution or a full-author AdaFlash reproduction.
