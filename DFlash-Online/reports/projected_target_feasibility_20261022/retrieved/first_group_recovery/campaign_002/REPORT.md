# Projected-target feasibility report

Planned 144 generations; present 48; qualified and valid 48.

A is the trained source set; B is held out. Negative changes mean fewer verification calls. Ratios use summed calls, never the mean of request ratios. Incomplete rows retain their planned denominator; their full-planned percentage is unavailable.

| Group | Role | Comparison | Valid/planned | Arm/base calls | Change | Gains/regressions/ties |
|---|---|---|---:|---:|---:|---:|
| gsm8k | A | C vs F | 8/8 | 351/356 | -1.40% | 4/4/0 |
| gsm8k | A | P vs F | 8/8 | 394/356 | +10.67% | 1/6/1 |
| gsm8k | A | P vs C | 8/8 | 394/351 | +12.25% | 1/7/0 |
| gsm8k | B | C vs F | 8/8 | 367/303 | +21.12% | 1/6/1 |
| gsm8k | B | P vs F | 8/8 | 350/303 | +15.51% | 1/6/1 |
| gsm8k | B | P vs C | 8/8 | 350/367 | -4.63% | 5/1/2 |
| codealpaca | A | C vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| codealpaca | A | P vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| codealpaca | A | P vs C | 0/8 | unavailable | unavailable | 0/0/0 |
| codealpaca | B | C vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| codealpaca | B | P vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| codealpaca | B | P vs C | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | A | C vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | A | P vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | A | P vs C | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | B | C vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | B | P vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| sharegpt | B | P vs C | 0/8 | unavailable | unavailable | 0/0/0 |
| ALL | A | C vs F | 8/24 | 351/356 | unavailable | 4/4/0 |
| ALL | A | P vs F | 8/24 | 394/356 | unavailable | 1/6/1 |
| ALL | A | P vs C | 8/24 | 394/351 | unavailable | 1/7/0 |
| ALL | B | C vs F | 8/24 | 367/303 | unavailable | 1/6/1 |
| ALL | B | P vs F | 8/24 | 350/303 | unavailable | 1/6/1 |
| ALL | B | P vs C | 8/24 | 350/367 | unavailable | 5/1/2 |

Token and termination totals across groups (qualified valid requests only):

| Role | Arm | Valid/planned | Accepted draft tokens | Output tokens (mean) | EOS/cap |
|---|---|---:|---:|---:|---:|
| A | F | 8/24 | 1641 | 1998 (249.8) | 8/0 |
| A | C | 8/24 | 1646 | 1998 (249.8) | 8/0 |
| A | P | 8/24 | 1601 | 1998 (249.8) | 8/0 |
| B | F | 8/24 | 1545 | 1849 (231.1) | 8/0 |
| B | C | 8/24 | 1480 | 1849 (231.1) | 8/0 |
| B | P | 8/24 | 1495 | 1849 (231.1) | 8/0 |

Qualification:

- gsm8k: passed.
- codealpaca: qualification_missing_file.
- sharegpt: qualification_missing_file.

PER_REQUEST.csv preserves every planned arm/request and its validity reason. SUMMARY.json records accepted draft tokens, output lengths and EOS/cap rates with explicit denominators. POOLED.csv includes the explicitly labeled available-valid-pair ratios for incomplete comparisons.

COSTS.json separates shared teacher preparation, training, nested reference/projection work, exposure, checkpoint/export, activation and serving. Timings are single instrumented observations and do not establish speedup. Losses from different objectives are not compared. Peak-memory and end-to-end receipts are retained where supplied.

Benefit versus F supports adaptation on this screen. Beating C while remaining worse than F is reduced harm. Little source learning leaves preservation inconclusive; this report does not select a new direction.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
