# Final execution and cost review

## Material Passport

- Mode: ARS experiment-agent validation of completed execution.
- Verification Status: ANALYZED from authenticated retained receipts; no new model, tensor replay, optimizer, generation, or remote work.
- Actual scientific execution: `2549f02d9b1edba8fb86f8e91a497e68c224d586`, PR47, execution_002.
- Evidence collection: 62,530 files / 640,935,301 raw bytes, verified against authenticated inventory `7841887b536584b0261823dcb22ca9b84f89acdcf5a4320a89472135f69717a0`.
- Review: 6,915 explicit receipt/accounting checks passed; zero failed. This is a same-team code-author review, not an independent external replication.

## Outcome

The prescribed execution completed with exit 0. All six GPU qualification cases passed. All 128 native N steps and 48 disposable E/C/H steps completed, with 16 fixed windows and 80 fresh endpoint exports. The exact final-native comparison confirms retained historical NATIVE7 equality for weights, Adam moments/counters and RNG at 896 cumulative steps and 1,792 presentations. No new complete-response generation occurred.

The earlier execution_001 failure remains preserved: its H checker rejected an OrderedDict/plain-dict schema before reaching tensor values. The repaired execution_002 H comparison now passes both all58 weight tensors and optimizer state. This does not retroactively relabel the earlier attempt a pass.

## State, source objective and role checks

- Qualification order is exactly N, ALL, E, C, H and restored N_repeat. Origin restoration passed.
- Main trajectory retained 176 exact state checks: 80 source-observer checks, 48 pre-branch checks and 48 restorations to authoritative N. All passed. Source observers therefore did not advance the surviving optimizer/RNG/model trajectory.
- All32 E/C same-state forward parity receipts passed: displayed native loss, native selection/masks, anchors and denominator remain unchanged.
- The64 E/C response normalization checks exactly match the native response denominator and full objective in retained ordinary observations. No active-role denominator substitution is present.
- Across1,070 unique recorded projection chunks per E/C role, all excluded direct token-loss gradients are zero and all gradients are finite. E records6,356 active nonzero first-error occurrences; C records66,993 active nonzero correct-prefix occurrences. These are repeated training occurrences, not distinct examples or evaluation samples.
- The independently accumulated E-enabled plus C-enabled objective differs from the displayed native objective by at most8.94e-8 over the32 response instances; this scalar observation is not a revised gradient-sum acceptance test.
- Each H update advances exactly one native Adam step, performs zero model forward/backward, makes zero sampling draws and reports zero current gradient. Its RNG remains unchanged. H retains the ordinary native gradient-present pattern and old Adam history; it is not the static P endpoint.
- The160 source observations comprise two responses at each of80 P/N/E/C/H endpoints, with zero backward. They are source-fit measurements only.

## Exact work accounting

| Category | Response forwards | Response backwards | Adam applications |
|---|---:|---:|---:|
| Disposable startup | 2 | 2 | 0 |
| Successful six-case qualification | 10 | 10 | 6 |
| Surviving native trajectory | 256 | 256 | 128 |
| Main disposable E/C/H branches | 64 | 64 | 48 |
| Source observations | 160 | 0 | 0 |
| Current attempt | **492** | **332** | **182** |
| Preserved failed attempt | 10 | 10 | 5 |
| Cumulative | **502** | **342** | **187** |

Current-attempt forward/backward totals include startup; excluding the two startup executions there are330 ordinary/qualification response forwards with backward and160 source-only forwards. All durable starts have matching completions; zero unknown or failed operations remain in execution_002.

The physical nested counters are492 draft forwards,2,460 layer forwards,21,220 objective-chunk calls and47,790 vocabulary-head calls. The head count reconciles to twice the objective-chunk count plus5,350 source-observer head calls. These include recomputation/nested work and must not be added to response counts as extra examples.

Independent native scoring completed20,480/20,480 conceptual and physical probes, with zero invalid, reused or unattempted rows. The execution-wide native budget also includes reconstruction and qualification:23,425 draft blocks,23,484 target calls,448,601 context forwards and84 model loads. These are diagnostic operations, not complete-response generations. The prior failed attempt's119 draft blocks,117 target calls,178 context forwards and2 loads remain separately retained.

## Time and memory

- Current occupied-card accounting: **6,800.969879s (113.35min)**.
- Preserved failed occupied-card time: **203.070064s (3.38min)**.
- Cumulative: **7,004.039942s (116.73min)**, below the10,800-second cap.
- Current campaign wall time: **6,872.811598s (114.55min)**, a different clock from occupied-card time.
- Native probe qualification phase29.614s; training qualification plus native trajectory phase3,609.132s; independent native probe phase2,707.378s. Remaining campaign overhead includes preparation/authentication/transitions; no detailed kernel breakdown is inferred.
- Trainer loading31.653s, disposable startup18.684s, six-case training qualification122.478s. Main trajectory including branches, state operations and diagnostics3,416.391s; total trainer3,603.319s.
- Eighty endpoint export timers sum210.931s. Source-observer internal timers sum58.258s; enclosing durable response-operation timers sum62.408s. The difference reflects timer scope. These and optimizer/forward/backward timers are nested; do not add them into another total.
- Recorded native event `training_s` is436.925s, checkpoint persistence19.730s and final native export13.565s. These component timers are not a full accounting of the intervention study.
- Main training measured PyTorch peak: **19.079GiB allocated /24.121GiB reserved**. Qualification maximum: **19.916GiB allocated /21.805GiB reserved** (H includes the numerical oracle).
- Largest observed physical GPU0 footprint in108 monitor snapshots: **54.652GiB**. This is sampled `nvidia-smi` process/card usage during native inference, not a measured allocator peak. GPU1 remained idle in those snapshots; both were idle at completion.

## Actual BF16 displacement across sixteen windows

Norms are computed from the realized BF16 endpoint parameters, accumulated in FP64 scalar accounting. The ratio column averages the16 paired per-window ratios; it is not the ratio of pooled parameter norms.

| Endpoint | Mean update norm | Min–max norm | Mean norm / corresponding N | Mean changed-coordinate fraction |
|---|---:|---:|---:|---:|
| N | 1.106223 | 0.803080–1.380595 | 100.00% | 10.13% |
| E | 1.042700 | 0.754216–1.317305 | 94.26% | 9.77% |
| C | 0.989848 | 0.733633–1.232016 | 89.65% | 9.44% |
| H | 0.957501 | 0.707132–1.203732 | 86.72% | 9.21% |

E is about5.74% smaller than N by the mean paired norm ratio, C about10.35% smaller, and H about13.28% smaller. H still makes a substantial nonzero update because prior Adam moments and decay survive when the current gradient is zero. These differences do not identify a harmful gradient direction, establish a scale-matched comparison, or imply that reducing the step alone would reproduce any outcome. Per-window values remain in `FINAL_UPDATE_NORMS.csv`.

## Limits

This review checks exact retained receipts and independently recomputes their arithmetic/count reconciliation. It does not load retired or remote model tensors to replicate their comparison. It establishes execution/accounting consistency; independent-probe efficacy and uncertainty are reviewed separately. There is no serving-latency result or new complete-response evaluation in this diagnostic. No implementation, tolerance or scientific setting was changed during this review.

## Evidence

- `delivery/execution_002_stream/VERIFIED.json`, authenticated `REMOTE_MANIFEST.json`.
- `retained/training/qualification/RESULT.json`, each case result, and `RESTORED_ORIGIN.json`.
- `retained/training/FINAL_NATIVE_PARITY.json`, `STATE_PARITY.json`, per-window forward-parity receipts.
- `retained/training/UPDATE_RECEIPTS.json`, `RESULT.json`, `SOURCE_WORK.json`, `PHYSICAL_WORK.jsonl`.
- `retained/STATUS.json`, `WORK_AND_COSTS.json`, `probes/STATUS.json`.
- Private review outputs `FINAL_EXECUTION_COST_CHECKS.json` and public-safe scalar summary `FINAL_EXECUTION_COST_SUMMARY.json`.
