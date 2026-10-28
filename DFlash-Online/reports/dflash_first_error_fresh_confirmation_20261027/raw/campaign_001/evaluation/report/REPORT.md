# Fresh O/K checkpoint confirmation

Execution: **completed**. Valid common requests: **64/64**.

All values use fresh capture-disabled measurements on inputs new to the current O/K study. Older project exposure is disclosed in the input manifest. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| O | 3191 | 56.101814 | 155.2420 |
| K | 3131 | 55.324202 | 152.3230 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| K_vs_O | +1.8803% | +1.3861% | 31 / 20 / 13 |

**K versus O is the sole comparison.** First fresh O establishes canonical tokens and termination; no historical call count is a gate. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 367.024468s; native weight validation: 5.984612s. These are not charged to inference-only latency. There is no new adaptation. Initial backend loading and complete campaign costs are reported separately.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_fresh_confirmation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
