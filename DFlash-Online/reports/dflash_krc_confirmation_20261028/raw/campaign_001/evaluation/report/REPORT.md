# Fresh K1/KRC checkpoint confirmation

Execution: **completed**. Valid common requests: **64/64**.

All values use fresh capture-disabled measurements on inputs new to the current K1/KRC study. Older project exposure is disclosed in the input manifest. Request latency includes prefill, drafting, target verification and request cache maintenance. Training and activation are excluded; these are results with an available checkpoint.

| Arm | Verification calls | Accepted draft tokens | Output tokens | EOS / capped | Sum of mean times (s) | Calls / 1,000 tokens |
|---|---:|---:|---:|---:|---:|---:|
| K1 | 3150 | 16769 | 19928 | 64 / 0 | 55.833497 | 158.0690 |
| KRC | 3125 | 16795 | 19928 | 64 / 0 | 55.088521 | 156.8145 |

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| KRC_vs_K1 | +0.7937% | +1.3343% | 25 / 14 / 25 |

**KRC versus K1 is the sole comparison.** First fresh K1 establishes canonical tokens and termination; no historical call count is a gate. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.

Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.

Gross calls saved: 42; gross calls added: 17; net saved: 25. Leaving each request out once gives call reductions from +0.6783% to +0.8698%. 0/64 omissions are nonpositive. This sensitivity analysis does not replace the complete-panel primary result.

Benchmark-only activation: 282.785738s across 193 publications; native weight validation: 8.544533s. These are not charged to inference-only latency. There is no new adaptation. Initial backend loading and complete campaign costs are reported separately.

The canonical agreement gate uses the native verifier and is not an independent target-only reference. This fixed-checkpoint panel tests input transfer; it does not establish robustness across training seeds or feedback windows.

Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv, LEAVE_ONE_OUT.csv and SUMMARY.json. Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_confirmation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
