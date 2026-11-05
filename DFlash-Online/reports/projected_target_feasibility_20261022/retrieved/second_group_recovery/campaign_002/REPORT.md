# Projected-target feasibility report

Planned 144 generations; present 96; qualified and valid 96.

A is the trained source set; B is held out. Negative changes mean fewer verification calls. Ratios use summed calls, never the mean of request ratios. Incomplete rows retain their planned denominator; their full-planned percentage is unavailable.

| Group | Role | Comparison | Valid/planned | Arm/base calls | Change | Gains/regressions/ties |
|---|---|---|---:|---:|---:|---:|
| gsm8k | A | C vs F | 8/8 | 351/356 | -1.40% | 4/4/0 |
| gsm8k | A | P vs F | 8/8 | 394/356 | +10.67% | 1/6/1 |
| gsm8k | A | P vs C | 8/8 | 394/351 | +12.25% | 1/7/0 |
| gsm8k | B | C vs F | 8/8 | 367/303 | +21.12% | 1/6/1 |
| gsm8k | B | P vs F | 8/8 | 350/303 | +15.51% | 1/6/1 |
| gsm8k | B | P vs C | 8/8 | 350/367 | -4.63% | 5/1/2 |
| codealpaca | A | C vs F | 8/8 | 424/456 | -7.02% | 6/2/0 |
| codealpaca | A | P vs F | 8/8 | 496/456 | +8.77% | 3/5/0 |
| codealpaca | A | P vs C | 8/8 | 496/424 | +16.98% | 0/8/0 |
| codealpaca | B | C vs F | 8/8 | 656/598 | +9.70% | 0/7/1 |
| codealpaca | B | P vs F | 8/8 | 637/598 | +6.52% | 1/6/1 |
| codealpaca | B | P vs C | 8/8 | 637/656 | -2.90% | 6/1/1 |
| sharegpt | A | C vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | A | P vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | A | P vs C | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | B | C vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | B | P vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | B | P vs C | 0/8 | unavailable | unavailable | 0/0/0 |
| ALL | A | C vs F | 16/24 | 775/812 | unavailable | 10/6/0 |
| ALL | A | P vs F | 16/24 | 890/812 | unavailable | 4/11/1 |
| ALL | A | P vs C | 16/24 | 890/775 | unavailable | 1/15/0 |
| ALL | B | C vs F | 16/24 | 1023/901 | unavailable | 1/13/2 |
| ALL | B | P vs F | 16/24 | 987/901 | unavailable | 2/12/2 |
| ALL | B | P vs C | 16/24 | 987/1023 | unavailable | 11/2/3 |

Token and termination totals across groups (qualified valid requests only):

| Role | Arm | Valid/planned | Accepted draft tokens | Output tokens (mean) | EOS/cap |
|---|---|---:|---:|---:|---:|
| A | F | 16/24 | 3436 | 4251 (265.7) | 14/2 |
| A | C | 16/24 | 3474 | 4251 (265.7) | 14/2 |
| A | P | 16/24 | 3357 | 4251 (265.7) | 14/2 |
| B | F | 16/24 | 4229 | 5131 (320.7) | 13/3 |
| B | C | 16/24 | 4103 | 5131 (320.7) | 13/3 |
| B | P | 16/24 | 4138 | 5131 (320.7) | 13/3 |

Qualification:

- gsm8k: passed.
- codealpaca: passed.
- sharegpt: qualification_missing_file.

PER_REQUEST.csv preserves every planned arm/request and its validity reason. SUMMARY.json records accepted draft tokens, output lengths and EOS/cap rates with explicit denominators. POOLED.csv includes the explicitly labeled available-valid-pair ratios for incomplete comparisons.

COSTS.json separates shared teacher preparation, training, nested reference/projection work, exposure, checkpoint/export, activation and serving. Timings are single instrumented observations and do not establish speedup. Losses from different objectives are not compared. Peak-memory and end-to-end receipts are retained where supplied.

Benefit versus F supports adaptation on this screen. Beating C while remaining worse than F is reduced harm. Little source learning leaves preservation inconclusive; this report does not select a new direction.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
