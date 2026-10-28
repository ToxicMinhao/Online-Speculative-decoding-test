# Checkpoint-component arithmetic report

## Material Passport

ARS experiment-agent; CPU analysis of supplied measurements. Correctness qualification belongs to the runner; no model execution or training by this module.

**FULL**: 64/64 declared requests in the common four-arm/all-repeat cohort. Full-panel claims require all 64 requests.

Calls and token statistics use designated repeat 0. Timing pools all three repeats. Every table and bootstrap uses the same common cohort.

| Arm | Requests | Calls | Saved vs F | Extra vs O | Gross saved | Gross harm | Improved/tied/regressed | Retained ratio | Call reduction % | Time reduction % |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| F | 64 | 3140 | 0 | 345 | 0 | 0 | 0/64/0 | 0 | 0 | 0 |
| O | 64 | 2795 | 345 | 0 | 359 | 14 | 56/3/5 | 1 | 10.9873 | 11.3193 |
| C-only | 64 | 3108 | 32 | 313 | 78 | 46 | 29/11/24 | 0.0927536 | 1.01911 | 1.39834 |
| B-only | 64 | 2896 | 244 | 101 | 266 | 22 | 55/1/8 | 0.707246 | 7.7707 | 7.36486 |

Retained-ratio denominator status: positive_at_least_2pct.

| Conditional effect | Calls saved | Pointwise 95% interval |
|---|---:|---|
| M_C_given_BF | 32 | [-7.0, 72.0] |
| M_C_given_BO | 101 | [56.97500000000002, 145.0] |
| M_B_given_CF | 244 | [185.0, 303.0] |
| M_B_given_CO | 313 | [253.0, 373.0] |
| I | 69 | [17.0, 118.0] |

Bootstrap denominator states (no resamples filtered):

- D_O < 0: 0 (0).
- D_O = 0: 0 (0).
- 0 < D_O / N_F < 0.02: 0 (0).
- D_O / N_F >= 0.02: 10000 (1).

Limitations:

- Fixed-panel totals are exact observations conditional on passed validity gates.
- Bootstrap intervals are exploratory pointwise request-composition sensitivity; they are not simultaneous guarantees, training-seed uncertainty, or fresh confirmation.
- A partial common cohort does not correct selection bias from invalid or missing requests.
- Retained-gain ratios with negative O gain do not express retention of positive benefit; small positive denominators are unstable. Use absolute effects alongside harm and time.
- Endpoint substitution does not establish restricted trainability, necessity, training savings, or net online acceleration. Follow-up experiments require separate review.

The CSV/JSON artifacts retain absolute intervals, ratio interval status, cross-tabs, every timing repeat, missing/invalid slots, and mismatch evidence. No outcome automatically selects or authorizes a later training experiment.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_checkpoint_components_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
