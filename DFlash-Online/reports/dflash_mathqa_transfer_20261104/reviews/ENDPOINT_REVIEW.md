# Independent MathQA endpoint calculation

## Material Passport

ARS experiment-agent validate. Separate NumPy/standard-library calculation of saved scalar evidence. Statistics implementation authorship overlap is disclosed. No production statistics helpers were imported; no model was rerun and native trace reconstruction is covered by the separate production audit.

Status: **passed**, 179617 numerical/identity/accounting checks. Execution `12583640baf2210f767e108e3632c0e5f9b8caaa`. All 1,024 requests and 9,216 timed executions retained, plus three historical controls.

| Arm | Request seconds (sum of means) | Native seconds | Cache seconds | Calls once/request | Output tokens | Cap hits |
|---|---:|---:|---:|---:|---:|---:|
| O | 1358.127369 | 1293.337401 | 54.501466 | 74786 | 519261 | 45 |
| K | 1328.923629 | 1266.912263 | 51.289979 | 73117 | 519261 | 45 |
| KPLUS | 1330.256461 | 1266.213117 | 53.215191 | 73132 | 519261 | 45 |

K+/O (primary): complete-time reduction +2.052157% [+1.652048, +2.452528]%; call reduction +2.211644%. Request wins/regressions/ties 635/249/140.

K+/K (confirmatory): complete-time reduction -0.100294% [-0.439884, +0.242661]%; call reduction -0.020515%. Request wins/regressions/ties 407/396/221.

Both ordered outcomes exactly reproduce the generated report within the declared floating arithmetic comparison tolerance (rtol 1e-12, atol 1e-9). This tolerance applies only to CPU report reconstruction, never to the scientific pass rule or native correctness.

## ARS fallacy scan — 11/11 checked

| Type | Result and boundary |
|---|---|
| Simpson’s paradox | The primary is the prespecified pooled ratio. All three repeat totals are retained; request and repeat heterogeneity must not be replaced by a universal per-request claim. No new subgroup tests were added. |
| Ecological fallacy | Request wins/regressions/ties above qualify the aggregate effect; pooled benefit does not imply benefit on every request. |
| Berkson’s paradox | Panel membership was fixed using input-only eligibility/hash ordering; conclusions concern eligible MathQA requests, not all math prompts. |
| Collider bias | No selection or adjustment by generated lengths, cap hits or observed performance. All valid capped/slow outputs remain included. |
| Base-rate neglect | Exact 1,024-request, 9,216-timing and output/cap denominators are reported; no diagnostic prevalence claim is made. |
| Regression to the mean | No request selection by an extreme observed outcome; this single frozen-checkpoint confirmation does not establish training-seed robustness. |
| Survivorship bias | Every planned execution is accounted and valid. No failed or incomplete request was removed to obtain the endpoint. |
| Look-elsewhere effect | Two fixed ordered latency contrasts; both reported. Count/component intervals are descriptive, not additional confirmatory discoveries. |
| Garden of forking paths | Source, panel, checkpoints, sequence and analysis were sealed before generation; no tuning, extra arm or selective retiming appears in the accounted work. |
| Correlation/causation | This establishes only the conditional serving comparison of fixed checkpoints; it does not isolate why training differed or establish adaptation-inclusive gains. |
| Reverse causality | Checkpoints precede the MathQA panel outputs; no inference from these outputs to an established training mechanism is made. |

All intervals use one shared set of 10,000 PCG64 seed-20261103 request resamples. Each request carries all three methods and three-repeat means. Timing repeats are not independent samples. A zero-crossing interval does not establish equivalence. Historical GSM8K latencies are not pooled. No broad-domain, multi-seed, online-adaptation or training-inclusive claim follows.

Machine-readable reconstruction, source hashes, paired rows and repeat totals accompany this review.
