# One-event prefix-ranking checkpoint comparison

Execution: **completed**. Valid common requests: **64/64**.

All values use fresh capture-disabled measurements on the same saved development inputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| F | 3140 | 54.069934 | 160.6385 |
| O | 3075 | 52.814925 | 157.3131 |
| U | 3082 | 53.012440 | 157.6713 |
| P | 3069 | 52.807258 | 157.0062 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| P_vs_O | +0.1951% | +0.0145% | 24 / 25 / 15 |
| P_vs_U | +0.4218% | +0.3870% | 28 / 22 / 14 |
| P_vs_F | +2.2611% | +2.3353% | 36 / 19 / 9 |
| O_vs_F | +2.0701% | +2.3211% | 40 / 19 / 5 |

**P versus O is primary.** Beating U alone does not establish improvement over ordinary OSD. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 1234.588522s; native weight validation: 12.961454s. These are not charged to inference-only latency. Adaptation and complete campaign costs belong to the separate run report.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
