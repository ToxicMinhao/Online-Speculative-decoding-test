# PR53 bounded-stop audit

**Independent audit passed: 17,114 checks, zero errors.** Source execution is `31b36fd27976fb46bb6754d8b8b7e769c3d6e5f6`. The authenticated compact delivery is `delivery/execution_001`; the complete audit result and consumed-file identities are in [INDEPENDENT_STOP_AUDIT.json](INDEPENDENT_STOP_AUDIT.json).

The run stopped after both datasets passed all six native training cases and serving checks on both physical cards. This includes exact bare-native/wrapper/cap-zero agreement, a nonzero active-C intervention, exact own-state reload, original-state restoration, CPU export equivalence and exact cross-card output/count signatures. The independent reader revalidated the native trace accounting and exact canonical output/termination for all 16 qualification generations and four origin warmup references. Those qualification measurements are correctness evidence, not serving-performance evidence.

| Work | Completed | Unattempted in the declared campaign |
|---|---:|---:|
| Adam applications | 12 | 1,024 |
| Response forwards | 28 | 2,064 |
| Response backwards | 28 | 2,064 |
| Training initializations | 2 | 8 |
| Feedback generations | 0 | 512 |
| Teacher preparations | 0 | 512 |
| Canonical generations | 4 | 512 |
| Qualification generations | 16 | 0 |
| Final warmup executions | 0 | 96 |
| Scored executions | 0 | 3,072 |

There are zero invalid or unknown model operations. No production task artifact exists, and no model lease remains active. Unattempted work is not relabelled invalid. The eight prospective production initializations account for the 16 additional disposable startup forwards/backwards in the unattempted response totals.

## Time admission

The initial forecast passed at 13,277.516 elapsed seconds (221.292 minutes). After qualification, the persisted forecast was independently reconstructed exactly:

`1,393.915513 elapsed + 1.3 × 9,702.483436 remaining raw + 600 margin = 14,607.143980 seconds`.

That exceeds the unchanged 14,400-second limit by **207.143980 seconds (3 minutes 27.144 seconds)**. The aggregate occupied-card forecast is 16,342.552596 seconds, leaving 1,657.447404 seconds under the 18,000-second card limit. The elapsed forecast alone caused this refusal; the actual elapsed clock had not exhausted four hours.

The underlying calculation does not add historical whole-task time to a new component estimate. It takes their maximum. The measured event estimate is loading plus disposable startup plus 128 times the slowest qualified pair with explicit diagnostic observer times removed, plus the prospective 45-second publication/boundary allowance. This repriced each GSM8K event to 552.344896 seconds and each MathQA event to 905.791333 seconds. The exact persisted price matches the planned parallel schedule: elapsed is each concurrent phase's maximum, while card time sums both workers.

Completed qualification is already represented by actual elapsed/occupied time; its old prospective allowance is not charged again. One future 300-second CPU-export reconstruction allowance is included inside the 1.3 multiplier. The 600-second margin is applied once to the remaining forecast. Thus the precise formula is `actual + 1.3 × (grouped remaining work + 300) + 600`, rather than an unmultiplied additional 300 seconds. The future reconstruction is distinct from the ordinary event's original publication allowance. No duplicate charge was identified.

These are prospective planning estimates, not guaranteed execution bounds. The existing policy correctly stopped instead of silently reducing the contingency, changing the hard cap, omitting tasks, or resetting the clock.

## Interpretation review — 11/11 categories checked

| Category | Bounded-stop interpretation |
|---|---|
| Simpson's paradox | No pooled or dataset performance comparison exists; keep GSM8K/MathQA unscored. |
| Ecological fallacy | Qualification correctness cannot establish per-request or population drafting benefit. |
| Berkson's selection | The sealed panels were not executed or changed after this stop; no result-conditioned sample claim. |
| Collider bias | No post-outcome subset or conditioning is introduced. |
| Base rate neglect | Completed, invalid, unknown and unattempted denominators are shown explicitly. |
| Regression to the mean | Active-C changed the fixed qualification fixture; that is not evidence of general performance improvement. |
| Survivorship bias | All completed operations are preserved; production is unattempted, not discarded for unfavorable results. |
| Look-elsewhere effect | No favorable qualification count is promoted to efficacy; no additional panel or analysis is selected. |
| Forking paths | The published formula and limits explain the stop; no post-hoc margin or policy change is applied. |
| Correlation/causality | A resource forecast says nothing about C's benefit or failure. |
| Reverse causality | There is no performance outcome from which to infer a mechanism. |

Native K+ remains primary. C's prior contribution-assessment status and performance-preservation uncertainty remain unchanged. The implementation passed qualification, but the planned production experiment did not run. No scientific conclusion, training-inclusive speedup or restart authorization follows from this audit.

## Reader-only correction

The first independent reader attempt expected a flat file-identity mapping; this terminal archive instead stores `COMPACT_STOP_MANIFEST.files` as a list. It stopped at the first read with an `AttributeError` before authenticating a scientific record. The original reader source and failure are retained in `stop_audit_001`. A narrow list-to-mapping reader correction and focused fixture were added; four reader/pricing tests pass in `STOP_READER_REPAIRED_TESTS.log`, then the separate audit attempt passed. No production code, model operation, scientific record or metric was changed.
