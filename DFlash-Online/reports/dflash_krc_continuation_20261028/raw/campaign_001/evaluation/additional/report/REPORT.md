# KRC2/K2 event-2 continuation evaluation

Execution: **completed**. Valid common requests: **64/64**.

KRC2/K2 values use fresh capture-disabled measurements on this fixed panel. All returned tokens and termination are checked against this panel’s retained canonical outputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| K2 | 3136 | 54.191128 | 157.3665 |
| KRC2 | 3116 | 53.853610 | 156.3629 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| KRC2_vs_K2 | +0.6378% | +0.6228% | 31 / 20 / 13 |

| Count-only history | Reference calls | Candidate calls | Reduction |
|---|---:|---:|---:|
| KRC1_vs_K1 | 3150 | 3125 | +0.7937% |
| KRC2_vs_KRC1 | 3125 | 3116 | +0.2880% |
| K2_vs_K1 | 3150 | 3136 | +0.4444% |

Historical event-1 values are call counts on the same valid requests. Relative KRC/K advantage and KRC’s own improvement are distinct; historical-versus-fresh timings are not a learning curve.

Common-cohort output lengths: 122–872; EOS requests: 64; output-cap hits: 0.

Native generation and cache-maintenance timers are components of complete request latency, not additive costs. See the separate timer columns in the CSV/JSON tables.

| Arm | Accepted draft tokens | Committed progress | Accepted / call | Progress / call |
|---|---:|---:|---:|---:|
| K2 | 16789 | 19864 | 5.3536 | 6.3342 |
| KRC2 | 16808 | 19864 | 5.3941 | 6.3748 |

**KRC2 versus K2 is the primary comparison.** The development panel requires retained K2 counts; the additional panel has no historical K2 count gate. No K1 or F generation or fresh latency is included. This fixed panel is reported separately from the other panel. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 380.328783s; native weight validation: 7.690254s. These are not charged to inference-only latency. KRC2 training, initial backend loading and complete campaign costs are reported separately. Historical K training is not a fresh matched cost control.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_continuation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
