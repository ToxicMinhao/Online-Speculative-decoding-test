# OSD baseline follow-up

Planned 96 generations; observed 96; qualified valid measurements 96.

A measures fit to the trained source requests; B measures transfer to held-out requests. Negative changes mean fewer verification calls or less request time. Ratios use sums, not mean request percentages. Incomplete comparisons have no full-planned percentage; the machine-readable tables label available-pair results separately.

| Group | Set | Valid/planned | F / OSD calls | Call change | Request-time change | Gains/regressions/ties |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | A | 8/8 | 356 / 380 | +6.74% | -0.30% | 1/6/1 |
| gsm8k | B | 8/8 | 303 / 361 | +19.14% | +17.58% | 1/6/1 |
| codealpaca | A | 8/8 | 456 / 460 | +0.88% | +4.73% | 5/3/0 |
| codealpaca | B | 8/8 | 598 / 659 | +10.20% | +10.14% | 0/8/0 |
| sharegpt | A | 8/8 | 1440 / 1275 | -11.46% | -10.34% | 7/1/0 |
| sharegpt | B | 8/8 | 1072 / 1136 | +5.97% | +5.42% | 1/7/0 |
| ALL | A | 24/24 | 2252 / 2115 | -6.08% | -5.51% | 13/10/1 |
| ALL | B | 24/24 | 1973 / 2156 | +9.28% | +8.76% | 2/21/1 |

Qualification:

- gsm8k: passed.
- codealpaca: passed.
- sharegpt: passed.

Only fresh F/OSD timings are compared. Each request has one instrumented observation, including prefill, drafting, verification and compact tracing; large feedback capture, training, publication and file persistence are outside that timer. These measurements do not establish robust serving speedup or adaptation-inclusive acceleration. No historical C/P timings enter these ratios.

PER_REQUEST.csv retains missing, invalid and unqualified work. POOLED.csv includes calls per 1,000 output tokens, accepted draft tokens and committed progress per call, cap/EOS counts and separate timing denominators. COSTS.json preserves actual campaign wall time, all observed serving costs, reused historical teacher preparation, new training, checkpoint/export and activation. Teacher-head work is nested inside training and is not added twice.

A source improvement alone does not establish transfer. A B improvement over frozen supports useful adaptation under this fixed recipe. Null or harmful results remain evidence; this report does not retune the method or establish general superiority of forward KL.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
