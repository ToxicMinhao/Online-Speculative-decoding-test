# Full K/K_HALF training, accounting and handoff review

## Material Passport

- Origin Skill: ars-codex:academic-research-suite 3.24.0 / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-02
- Verification Status: ANALYZED
- Version Label: full_k_training_cost_review_v1
- Execution: `0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34`, A800 `production_002`
- Review provenance: separate same-team agent, shared handoff/context; independently written standard-library audit, without importing campaign validators. No independent GPU reproduction, model-state deserialization, training restart, or scientific-file edits.

**Passed: 1,849,884 explicit conditions, zero discrepancies.** This comprises 1,846,975 raw-training conditions, 2,905 source/correctness/cost addendum conditions, and four exact local report cross-file equalities. The main audit read or rehashed 76,107 distinct compact files; it checked 75,296 retirement-evidence hash bindings. The addendum independently rehashed all 2,856 sealed source/configuration files. No scientific correctness or accounting defect was found within this scope.

Overall confidence: solid for recorded work/accounting, with the state and scientific-generalization limits below. Source/runtime/model pins are retained receipts; the model and runtime binaries were not independently imported or model-tested by this reviewer. Independent final-evaluation reconstruction and bootstrap calculations belong to the separate evaluation review.

## Completed work and causal ordering

Each arm has one fresh worker segment and one native attempt, with no resumed scientific segment. Its 104,810 work-ledger rows reconstruct exactly 58 events, 7,424 committed Adam steps, 14,848 forward/backward response presentations, 7,424 terminal-feature preparations, 7,473 online generations, four inherited correctness generations and 384 intermediate generations. All starts complete once; no unknown or repeated work is recorded. Both native directories contain exactly 7,861 generation traces.

Every event uses the next 128 original request identities, served with that arm's previous published checkpoint. Preparation follows all 128 responses, then each response appears exactly once in each of two passes. Each Adam update follows two original forwards and two backwards; learning rates are constant 3e-4 and 1.5e-4 respectively. All 29,696 response observations and 14,848 optimizer windows agree with event receipts. Prepared full-context tokens and native target labels bind to the arm's own responses; supervision stays response-only, sequence lengths remain at most 2,272, and the terminal feature supplies no replacement label. The final 49 responses use event-58 weights without another update or feedback capture.

Intermediate state-unchanged receipts cover all six prescribed event-14/29/44 evaluations. All eight inherited capture/publication/reload correctness controls have matching tokens/termination and required call equality, with no feedback collection or disposable optimizer step. Final-evaluation receipts account for 12,288 completed generations. Thus **14,946 online + 768 intermediate + 12,288 final + 8 correctness = 28,010 complete generations**. Feature preparation and teacher-head operations are separate work, not extra complete generations.

This supports the intended full online-updater protocol. K/O is the same-LR full-updater comparison, and K_HALF/O compares different recipes. It is not shared-feedback replay or an isolated mechanism ablation; historical O features were not recreated.

## Exposure and costs

| Reconstructed exposure | K | K_HALF |
|---|---:|---:|
| Response presentations | 14,848 | 14,848 |
| Kept sampled-anchor occurrences | 3,728,050 | 3,728,050 |
| Available position occurrences | 54,368,391 | 54,368,391 |
| Selected/supervised position occurrences | 33,320,668 | 32,063,446 |
| Selected position weight / loss denominator | 18,028,154.566895 | 17,609,142.998901 |
| Original draft forwards | 14,848 | 14,848 |
| Layer forward invocations | 74,240 | 74,240 |
| Objective chunks | 944,660 | 944,660 |
| Vocabulary-head invocations | 1,889,320 | 1,889,320 |

Exposure counts include repeated presentations and overlapping block positions; they are not unique tokens. Selection totals were reconstructed from each optimizer boundary once, and separately reconciled with original response observations. The small accounting comparison tolerance handles materialized FP32 accumulation; it changes no training qualification criterion. Equal aggregate anchor exposure does not itself prove identical anchor sequences.

| Recorded seconds | K | K_HALF |
|---|---:|---:|
| Training loop | 18,707.203 | 18,672.490 |
| Blocking updates | 21,735.125 | 21,712.566 |
| Teacher preparation | 287.347 | 290.067 |
| Prepared-feature persistence | 322.006 | 338.222 |
| Checkpoint saving | 1,161.561 | 1,149.953 |
| Export | 897.491 | 883.928 |
| Activation | 93.995 | 90.953 |
| Controller serving | 7,204.021 | 7,267.020 |
| Capture host work | 3,880.848 | 3,913.494 |
| Capture persistence | 348.745 | 351.471 |
| Teacher-head work | 5,932.073 | 5,930.544 |
| Selection work | 762.921 | 741.831 |
| Initial loading | 42.121 | 40.381 |
| Publication validation | 1,540.941 | 1,457.228 |
| Retention maintenance | 447.862 | 460.966 |
| Online worker wall time | 31,423.154 | 31,417.314 |

These rows overlap. Preparation, training, persistence/export and activation are inside blocking updates; capture is inside serving; teacher-head and selection work are inside training. Do not sum the table. Compiler timers are retained by name, can nest, and add no independent wall time. Serving-worker compilation is included in loading/serving and is not separately identifiable. Final worker loading is 28.762 s; campaign wall time is 73,416.448 s and contains the 90.002 s transfer-clearance wait. All named raw costs reconcile with the delivered cost records.

Both arms' maximum update-window allocation is 20,675,251,200 bytes (19.255 GiB), with 24,052,236,288 bytes (22.400 GiB) reserved. These trainer-process windows exclude initial loading, later boundary validation and the separate serving worker; they are not whole-system peaks.

Historical archive copying overlapped training/capture. Those adaptation measurements therefore describe the observed campaign under that I/O load and cannot isolate uncontended adaptation costs. The transfer barrier paused transport before final timed serving. The later archive-resume SSH failure is operationally separate; its root cause remains unknown, and it supplies no evidence that final timing overlapped transfers. Final request-time comparisons measure checkpoint-ready serving, not training-inclusive or asynchronous speedup. Historical F/O adaptation costs are unmatched.

## State evidence and retention

The audit checked both initial validation receipts, all 116 event-boundary/live-equality receipts, all six intermediate unchanged-state receipts, and both final validation receipts. Initial receipts certify fresh released weights, empty Adam and seed-17 RNG; K_HALF's paired check reports exact equality after declared LR/metadata differences without overwriting RNG. Final receipts report 58 Adam entries, step 7,424, presentation 14,848 and one CUDA RNG state, with exact live/persisted/export equality. These are original execution-time numerical checks, not repeated tensor checks in this review.

Both initial complete states, both event-58 complete states and both final serving weight files still exist. Final state/export sizes match their saved SHA-256 identity receipts, and each latest-state path is the same inode as its retained event-58 state. This review did **not** recompute the large tensor-file hashes or deserialize their contents. Exact restart plumbing is supported by the saved checks and reviewed source; no real interrupted-and-resumed scientific segment occurred in this campaign.

For each arm, retirement records identify exactly 7,424 native-capture tensors, 7,424 prepared-feature tensors, 7,424 terminal-feature tensors, 57 superseded export-weight files and 57 superseded complete optimizer/RNG states. All 44,772 combined retired paths are absent; their retained compact evidence rehashes correctly, and every predecessor-state retirement names a validated successor. The summed logical file lengths retired over the whole campaign are 1,439,217,442,630 bytes; this is cumulative turnover, not simultaneous occupied space or a final free-space measurement.

These artifacts were deleted, **not archived**. Exact intermediate model reload/optimizer continuation and feature-tensor replay are unavailable; compact outputs, observations, metrics and validity evidence remain. This new-run retirement is separate from the ten historical intermediate states retired before launch and the unfinished twelve-file historical archive. No deletion was performed during this review.

A800 free space was 76,780,556,288 bytes at review entry and 76,780,548,096 bytes at the final check; both GPUs remained idle with no compute process. Authoritative local/remote A800 rules matched at entry.

## Handoff/report gap checklist

| Handoff requirement | Review disposition |
|---|---|
| Exact released initialization, own full streams, fixed optimizer work | Passed within receipt/record scope above |
| Final tail, inherited controls, fixed intermediate points | Passed; no added generation or checkpoint selection |
| Separate preparation/capture/training/persistence/publication costs | Passed; nested scopes must remain explicit |
| Source/configuration seal and pinned compatibility | All 2,856 sealed file identities rehashed; runtime/model execution remains original qualification evidence |
| Retained initial/final states and bounded retirement | Passed existence/size/receipt checks; numerical tensor reinspection not performed |
| Final evaluation counts and inference | Receipt totals reconcile; independent evaluation/bootstrap reviewer supplies raw evaluation checks |
| Training/capture archive overlap | Add the explicit disclosure above to the final narrative; generated report alone does not explain it |
| Retention/restart limitations | Add exact retirement counts and distinguish saved restart receipts from an exercised interrupted resume |
| One next recommendation and research stage | Preserve separately agreed follow-up language; no run, policy deployment or stage advance is authorized by this review |

## Fallacy scan

Coverage: 11/11 ARS categories considered for this review's accounting/claim scope. Simpson's paradox: no training-seed/domain aggregation inference supported here. Ecological fallacy: repeated positions are explicitly exposure, not unique tokens or independent samples. Berkson/selection bias: reused benchmark and fitted seed remain selected evidence. Collider bias: no added adjustment or outcome-conditioned subgroup. Base-rate neglect: not a diagnostic classification analysis. Regression to the mean: fixed full horizon, no performance-selected checkpoint. Survivorship: all intended work completes and unknown work is zero in the ledgers. Look-elsewhere: retain all prescribed endpoints and descriptive secondary intervals. Forking paths: no efficacy-driven retry/tuning is recorded. Causal overreach: full-updater and different-LR contrasts do not isolate selection, normalization or LR. Reverse causality: recorded collection/update/publication order is correct, but order alone does not establish the unique mechanism of endpoint gains.

Evidence: [main reconstruction](TRAINING_COST_REVIEW.json), [source/correctness/cost addendum](TRAINING_RECEIPT_ADDENDUM.json), [independent audit source](code/training_review.py), [addendum source](code/training_receipt_addendum.py), and private retained input-identity ledger `review_20261102/INPUT_IDENTITIES.json`. Scripts ran through stdin with CUDA hidden, using the approved helper and existing system Python; they performed only read operations on A800.
