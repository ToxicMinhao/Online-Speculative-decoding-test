# Projected-target feasibility report

Planned 144 generations; present 16; qualified and valid 0.

A is the trained source set; B is held out. Negative changes mean fewer verification calls. Ratios use summed calls, never the mean of request ratios. Incomplete rows retain their planned denominator; their full-planned percentage is unavailable.

| Group | Role | Comparison | Valid/planned | Arm/base calls | Change | Gains/regressions/ties |
|---|---|---|---:|---:|---:|---:|
| gsm8k | A | C vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| gsm8k | A | P vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| gsm8k | A | P vs C | 0/8 | unavailable | unavailable | 0/0/0 |
| gsm8k | B | C vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| gsm8k | B | P vs F | 0/8 | unavailable | unavailable | 0/0/0 |
| gsm8k | B | P vs C | 0/8 | unavailable | unavailable | 0/0/0 |
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
| ALL | A | C vs F | 0/24 | unavailable | unavailable | 0/0/0 |
| ALL | A | P vs F | 0/24 | unavailable | unavailable | 0/0/0 |
| ALL | A | P vs C | 0/24 | unavailable | unavailable | 0/0/0 |
| ALL | B | C vs F | 0/24 | unavailable | unavailable | 0/0/0 |
| ALL | B | P vs F | 0/24 | unavailable | unavailable | 0/0/0 |
| ALL | B | P vs C | 0/24 | unavailable | unavailable | 0/0/0 |

Token and termination totals across groups (qualified valid requests only):

| Role | Arm | Valid/planned | Accepted draft tokens | Output tokens (mean) | EOS/cap |
|---|---|---:|---:|---:|---:|
| A | F | 0/24 | unavailable | unavailable | 0/0 |
| A | C | 0/24 | unavailable | unavailable | 0/0 |
| A | P | 0/24 | unavailable | unavailable | 0/0 |
| B | F | 0/24 | unavailable | unavailable | 0/0 |
| B | C | 0/24 | unavailable | unavailable | 0/0 |
| B | P | 0/24 | unavailable | unavailable | 0/0 |

Qualification:

- gsm8k: qualification_missing_file.
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
