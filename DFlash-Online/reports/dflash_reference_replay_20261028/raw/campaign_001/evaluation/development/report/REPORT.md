# KR/KRC ordinary reference-replay evaluation

Execution: **completed**. Valid common requests: **64/64**.

KR/KRC values use fresh capture-disabled measurements on this fixed panel. All returned tokens and termination are checked against this panel’s retained canonical outputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| KR | 3109 | 53.764256 | 159.0525 |
| KRC | 3005 | 51.864162 | 153.7320 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| KR_vs_KRC | -3.4609% | -3.6636% | 14 / 44 / 6 |

Common-cohort output lengths: 146–515; EOS requests: 64; output-cap hits: 0.

Native generation and cache-maintenance timers are components of complete request latency, not additive costs. See the separate timer columns in the CSV/JSON tables.

| Arm | Accepted draft tokens | Committed progress | Accepted / call | Progress / call |
|---|---:|---:|---:|---:|
| KR | 16426 | 19483 | 5.2834 | 6.2666 |
| KRC | 16535 | 19483 | 5.5025 | 6.4835 |

**KR versus KRC is the primary comparison.** Fresh KRC verification counts must reproduce their retained per-request values. No K1 or F generation or fresh latency is included. This fixed panel is reported separately from the other panel. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 359.263418s; native weight validation: 8.197206s. These are not charged to inference-only latency. KR training, initial backend loading and complete campaign costs are reported separately. Historical KRC training is not a fresh matched cost control.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_replay_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
