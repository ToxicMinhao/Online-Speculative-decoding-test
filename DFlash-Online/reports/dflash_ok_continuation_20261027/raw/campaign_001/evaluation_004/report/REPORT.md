# O/K continuation evaluation

Execution: **completed**. Valid common requests: **64/64**.

O/K values use fresh capture-disabled measurements on the original development panel. All returned tokens and termination are checked against retained F outputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| O | 2994 | 51.484646 | 153.1693 |
| K | 2942 | 50.517116 | 150.5090 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| K_vs_O | +1.7368% | +1.8793% | 34 / 20 / 10 |

| Historical comparison | Verification-call reduction | Calls improved / regressed / tied |
|---|---:|---:|
| O_vs_F | +4.6497% | 44 / 13 / 7 |
| K_vs_F | +6.3057% | 50 / 8 / 6 |

Common-cohort output lengths: 146–515; EOS requests: 64; output-cap hits: 0.

| Arm | Accepted draft tokens | Committed progress | Accepted / call | Progress / call |
|---|---:|---:|---:|---:|
| O | 16546 | 19483 | 5.5264 | 6.5073 |
| K | 16601 | 19483 | 5.6428 | 6.6224 |

**K versus O is the primary comparison.** Retained F supplies canonical tokens, termination and historical call context only; no F latency is compared. No historical O count is a gate. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 275.820633s; native weight validation: 4.956613s. These are not charged to inference-only latency. Continuation training, initial backend loading and complete campaign costs are reported separately.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
