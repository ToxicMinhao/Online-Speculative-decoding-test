# Continuation-repair pilot: incomplete comparison

**Status: BLOCKED. GPU qualification and TAIL training passed; serving evaluation has not run. There is no evidence yet that TAIL improves over BASE.**

Training execution: `b3637cfe4ac3fb4096368cc0c93896b420bdb9a8`. First orchestration repair: `b3a467200af0b255fe2a491924d03c2c1f0c286b`. These identities are distinct; the repair does not alter the trained weights.

| Work | Completed | Remaining |
|---|---:|---:|
| Disposable qualification Adam steps | 3 | 0 |
| Main TAIL Adam steps / presentations | 128 / 256 | 0 |
| Matched endpoint readout forwards | 74 | 63 |
| Scored serving generations | 0 | 768 |
| Declared serving warmups | 0 | 24 |

Qualification passed exact native BASE/disabled gradients, weights, Adam/RNG and restoration checks, retained BASE compatibility, coefficient invariants and the independent FP64 allocation oracle. TAIL completed the prescribed warm event, verified its recovery/export, and ended at cumulative step 256 / presentations 512 / event 2. All 256 presentations were active, changing 22,781 HIGH coefficient occurrences. Activity and distinct weights are not serving benefits.

The first readout attempt stopped after 73 primary reads plus one included repeat because supervisor and child wrote the same atomic storage temporary file. The per-process receipt repair addresses that race. Its resume checker then rejected the saved error before model loading: Python's compact `repr(FileNotFoundError)` omits the filename, whereas the full saved traceback contains it. This second error was missed by a synthetic test fixture and the initial independent review. That review is explicitly superseded, with its original receipt retained. Neither failure is evidence of a mechanism failure.

All historical failures, completed readouts, original source/plan and final TAIL state remain preserved. The rejected restart added 8.846716 charged process seconds, zero model forwards, zero Adam steps and zero generations. Cumulative charged process time is 1308.517726 seconds. The original four-hour clock continues; elapsed time at the last process exit was 1876.408622 seconds and does not reset while awaiting approval.

Main training measured 607.428 seconds; checkpoint persistence 19.076 seconds and export 15.791 seconds are separate nested costs. The complete training process measured 776.923 seconds. Recorded training peak allocation/reservation was 20.555 / 27.781 GB. Historical BASE training is not a fresh overhead control, and no training-inclusive speedup can be claimed.

Before execution, three obsolete CODE event 57 rollback states were retired after verifying retained final event 58 states and compact evidence, freeing 19.243552768 GB. Direct event 57 optimizer/RNG rollback was lost, not archived. All pilot dependencies remain protected. A800 retained more than 415 GB free after staging and training, above the 400 GB reserve.

The remaining fixed work is 63 readouts plus 792 serving executions. Estimate: 30–50 minutes of GPU work followed by CPU audit, conditional on approval and the remaining original resource limits. No training or completed readout needs repetition. The handoff's one localized implementation retry is consumed; another execution requires a scoped user decision. The corrected checker at `d819c243235e5a650ce5292f2fa26ebfa563a758` passes all 50 pilot CPU tests. A800 CPU preflight authenticates the original and revised source, unchanged final recovery/export, original clock and completed work; it passes with zero model loads or forwards. A800 free space is 415.078 GB after staging. A separate actual-record check verifies all 74 saved reads without model loading or changes to raw evidence. An independent partial audit passes 4,872 checks of completed qualification/training. Whole-campaign and serving audits remain unexecuted.

Until the full valid comparison exists, retain native C+K+PAIR. This report provides no serving improvement, confidence interval, or adoption claim.
