# KRC2/KRC2_OFF event-2 correction-removal evaluation

Execution: **completed**. Valid common requests: **64/64**.

KRC2/KRC2_OFF values use fresh capture-disabled measurements on this fixed panel. All returned tokens and termination are checked against this panel’s retained canonical outputs. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|
| KRC2_OFF | 3138 | 54.049624 | 157.4669 |
| KRC2 | 3116 | 53.740707 | 156.3629 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| KRC2_vs_KRC2_OFF | +0.7011% | +0.5715% | 23 / 14 / 27 |

| Count-only history | Reference calls | Candidate calls | Reduction |
|---|---:|---:|---:|
| KRC2_OFF_vs_KRC1 | 3125 | 3138 | -0.4160% |
| KRC2_vs_KRC1 | 3125 | 3116 | +0.2880% |
| KRC2_OFF_vs_K2 | 3136 | 3138 | -0.0638% |

Historical KRC1 and K2 values are counts on the same valid requests. Correction benefit versus OFF is distinct from OFF improvement beyond KRC1; historical-versus-fresh timings are not a learning curve.

Common-cohort output lengths: 122–872; EOS requests: 64; output-cap hits: 0.

Native generation and cache-maintenance timers are components of complete request latency, not additive costs. See the separate timer columns in the CSV/JSON tables.

| Arm | Accepted draft tokens | Committed progress | Accepted / call | Progress / call |
|---|---:|---:|---:|---:|
| KRC2_OFF | 16787 | 19864 | 5.3496 | 6.3301 |
| KRC2 | 16808 | 19864 | 5.3941 | 6.3748 |

**KRC2 versus KRC2_OFF is primary: positive reduction favors retaining correction.** Both panels require reproduction of retained per-request KRC2 counts. No K1, K2 or F generation or fresh latency is included. This fixed panel is reported separately from the other panel. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Benchmark-only activation: 380.476701s; native weight validation: 7.615636s. These are not charged to inference-only latency. OFF training, initial backend loading and complete campaign costs are reported separately. Historical KRC2 training is not a fresh matched cost control.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_off_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
