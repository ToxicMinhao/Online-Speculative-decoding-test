# K/KRC K-plus-correction evaluation

Execution: **completed**. Valid common requests: **64/64**.

K/KRC values use fresh capture-disabled measurements on the original development panel. All returned tokens and termination are checked against retained F outputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| K | 3027 | 52.275888 | 154.8575 |
| KRC | 3005 | 51.829177 | 153.7320 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| KRC_vs_K | +0.7268% | +0.8545% | 31 / 15 / 18 |

Common-cohort output lengths: 146–515; EOS requests: 64; output-cap hits: 0.

| Arm | Accepted draft tokens | Committed progress | Accepted / call | Progress / call |
|---|---:|---:|---:|---:|
| K | 16513 | 19483 | 5.4552 | 6.4364 |
| KRC | 16535 | 19483 | 5.5025 | 6.4835 |

**KRC versus K is the primary comparison.** Retained F supplies canonical tokens and termination; optional F counts are historical context only. No F latency is compared. No historical K count is a gate. This original development panel is held out from feedback training but was already used in development. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 280.493267s; native weight validation: 5.589430s. These are not charged to inference-only latency. KRC training, initial backend loading and complete campaign costs are reported separately.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_event_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
