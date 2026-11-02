# Original handoff coverage

Execution commit: `0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34`.
Scope: the October 31 **full GSM8K K/K_HALF** handoff, with the November 1 operational amendment allowing archive transfer during training and requiring transfer quiescence before final timing.

| Requirement | Completed work and evidence |
|---|---|
| New K and K_HALF from released weights, fresh seed-17 Adam/RNG | Separate new trajectories; initial-state, configuration and live state-validation receipts; independent training review |
| Retained F and O58, no retraining or recreated old feedback | Used only as final evaluation controls; retained export identities and old canonical records checked |
| Original 7,473 request stream per arm, own feedback, final 49 without training | 14,946 online generations; 116 causal feedback events; each arm has 49 tail requests; raw online and WORK chronology |
| 58 events and 7,424 Adam steps per arm | Exactly 116 events, 14,848 total Adam steps, 29,696 response presentations/forwards/backwards; [accounting](ACCOUNTING.json) |
| Existing numerical objective, optimizer, masks, anchors and serving setup | Execution source seal, [configs](configs/), pinned runtime/model files, per-presentation and window observations |
| Terminal teacher preparation counted separately | Exactly 14,848 preparation forwards; no misclassification as complete generations; costs and native-work observations |
| Intermediate events 14/29/44 on first 128 requests, count-only | 768 valid executions; [learning curve](LEARNING_CURVE.csv); state/RNG isolation receipts; no performance-based stopping |
| Final F/O/K/K_HALF on 1,024 original held-out requests, three repeats | 12,288 valid executions, zero invalid/unknown/unattempted; [all measurements](ALL_EVALUATIONS.csv), [paired requests](PAIRED_REQUESTS.csv) |
| Fixed 64-request method blocks and balanced eight-order schedule | 48 block/repeat orders; each method has 12 appearances in every position; pairwise precedence 24/24; independent raw audit |
| Exact outputs, termination, traces, weights and consistent repeat counts | Raw record/native-trace reconstruction, canonical equality and publication receipts; [evaluation audit](EVALUATION_AUDIT.json) |
| Eight inherited correctness generations, zero disposable Adam steps | Four per new arm, included in work ledger; no additional generation warmups or scientific retries |
| No training or artifact transfer during final timing | Both trainer workers exited; source-bound transfer clearance and process-quiescence receipts precede final backend creation |
| Primary K_HALF/O; K/O, K_HALF/K, both versus F mandatory | All five comparisons in [report](REPORT.md) and [summary](SUMMARY.json); K_HALF/O explicitly uses different learning rates |
| Ratios of sums and three-repeat request-mean timing | Reconstructed per request; all repeat totals retained in [TIMING_REPEAT_TOTALS.csv](TIMING_REPEAT_TOTALS.csv) |
| 10,000 paired bootstrap, PCG64 seed 20261102, linear 95% intervals | Independently reconstructed at the request level; no timing-repeat pseudoreplication; secondary intervals unadjusted/descriptive |
| Gross gains/harms and fixed learning curves | All requests, regressions and ties retained; all intermediate endpoints and same-subset event-58 endpoint reported |
| Training, teacher, capture, compilation, persistence and publication costs | [COSTS.json](COSTS.json), [EVENT_COSTS.csv](EVENT_COSTS.csv), [RESOURCES.json](RESOURCES.json); nesting and unavailable separate worker-compilation timing disclosed |
| Actual training exposure | [EXPOSURE.json](EXPOSURE.json), [EVENT_EXPOSURE.csv](EVENT_EXPOSURE.csv); overlapping position/anchor occurrences distinguished from unique tokens |
| Retention and exact recovery | Initial/final states and final exports remain; verified successor retirement receipts and lost intermediate tensor-level replay are disclosed |
| Code, exact commands and source commit | Published PR #23; [execution/provenance](EXECUTION_AND_PROVENANCE.md); every review/report identifies the scientific execution commit |
| Independent record review | Separate [evaluation/statistics review](reviews/RAW_STATISTICS_REVIEW.md) and [training/accounting review](reviews/TRAINING_COST_REVIEW.md); CPU evidence review, not independent GPU reproduction |
| One recommendation, no automatic expansion | One proposed second-seed confirmation of same-LR K/O, subject to separate agreement; no follow-on run, tuning, merge or deployment |

The scientific run is complete. Unfinished historical archive copying is separate housekeeping: it failed after final timing and successful experiment exit, with originals preserved. This report does not represent that archive as completed.
