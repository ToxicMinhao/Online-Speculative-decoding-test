# One-event F/O/P/R checkpoint comparison

Execution: **completed**. Valid common requests: **64/64**.

All values use fresh capture-disabled measurements on the same saved development inputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| F | 3140 | 54.309900 | 160.6385 |
| O | 3075 | 53.059998 | 157.3131 |
| P | 3069 | 53.044502 | 157.0062 |
| R | 3075 | 53.087305 | 157.3131 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| R_vs_O | +0.0000% | -0.0515% | 25 / 25 / 14 |
| R_vs_P | -0.1955% | -0.0807% | 15 / 17 / 32 |
| R_vs_F | +2.0701% | +2.2511% | 38 / 18 / 8 |
| O_vs_F | +2.0701% | +2.3014% | 40 / 19 / 5 |
| P_vs_O | +0.1951% | +0.0292% | 24 / 25 / 15 |

**R versus O is primary.** Other comparisons do not replace the ordinary OSD control. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 1173.957889s; native weight validation: 12.427635s. These are not charged to inference-only latency. Adaptation and complete campaign costs belong to the separate run report.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_margin_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
