# KR/KRC ordinary reference-replay evaluation

Execution: **completed**. Valid common requests: **64/64**.

KR/KRC values use fresh capture-disabled measurements on this fixed panel. All returned tokens and termination are checked against this panel’s retained canonical outputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| KR | 3255 | 55.876409 | 163.3380 |
| KRC | 3125 | 53.695815 | 156.8145 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| KR_vs_KRC | -4.1600% | -4.0610% | 9 / 44 / 11 |

Common-cohort output lengths: 122–872; EOS requests: 64; output-cap hits: 0.

Native generation and cache-maintenance timers are components of complete request latency, not additive costs. See the separate timer columns in the CSV/JSON tables.

| Arm | Accepted draft tokens | Committed progress | Accepted / call | Progress / call |
|---|---:|---:|---:|---:|
| KR | 16658 | 19864 | 5.1177 | 6.1026 |
| KRC | 16795 | 19864 | 5.3744 | 6.3565 |

**KR versus KRC is the primary comparison.** Fresh KRC verification counts must reproduce their retained per-request values. No K1 or F generation or fresh latency is included. This fixed panel is reported separately from the other panel. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 355.293366s; native weight validation: 6.743343s. These are not charged to inference-only latency. KR training, initial backend loading and complete campaign costs are reported separately. Historical KRC training is not a fresh matched cost control.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_replay_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
