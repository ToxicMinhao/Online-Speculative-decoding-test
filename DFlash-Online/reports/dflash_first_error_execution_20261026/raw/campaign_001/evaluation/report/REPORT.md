# One-event O/C/K/A checkpoint comparison

Execution: **completed**. Valid common requests: **64/64**.

All values use fresh capture-disabled measurements on the same saved development inputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| O | 3075 | 52.777255 | 157.3131 |
| C | 3102 | 53.257954 | 158.6944 |
| K | 3027 | 52.020989 | 154.8575 |
| A | 3058 | 52.456160 | 156.4434 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| K_vs_O | +1.5610% | +1.4329% | 33 / 21 / 10 |
| A_vs_C | +1.4184% | +1.5055% | 35 / 17 / 12 |
| C_vs_O | -0.8780% | -0.9108% | 14 / 30 / 20 |
| A_vs_K | -1.0241% | -0.8365% | 21 / 31 / 12 |

K/O and A/C test position selection; C/O and A/K test loss choice. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 975.354071s; native weight validation: 9.068209s. These are not charged to inference-only latency. Adaptation and complete campaign costs belong to the separate run report.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.

Retained F supplies exact canonical tokens/termination and historical verification counts only (HISTORICAL_F_COUNTS.json). No fresh F latency comparison is made.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
