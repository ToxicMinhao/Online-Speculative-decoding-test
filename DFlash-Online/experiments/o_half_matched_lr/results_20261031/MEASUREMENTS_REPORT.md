# Ordinary OSD at matched lower learning rate: four-event coding comparison

Execution source: `19ee22faf985e0576d287eaa8c8624b5a44b46f5`.

Question: does K retain useful drafting benefit over ordinary OSD at the same LR1.5e-4? Before execution, O_HALF was expected potentially to recover part of ordinary O’s deficit; the relative outcome was unknown. Only O_HALF was trained.

**Decision A: Useful matched-LR advantage for K_HALF.**

This reuses the PR21 development panel that already informed the choice of K_HALF. It is development/ablation evidence, not fresh confirmation. F/K/K_HALF counts are verified archived repeat-zero trajectories; O/O_HALF counts come from new two-repeat evaluation. The compatibility bridge is excluded from the performance sample.

| Scope | Method | LR | Count provenance | Calls | Output tokens | Calls / 1,000 tokens | Accepted tokens | Progress / call |
|---|---|---:|---|---:|---:|---:|---:|---:|
| combined | F | released | archived PR21 | 16300 | 88266 | 184.6691 | 71950 | 5.3994 |
| combined | O | 3e-4 | new two-repeat | 16379 | 88266 | 185.5641 | 71873 | 5.3733 |
| combined | K | 3e-4 | archived PR21 | 16303 | 88266 | 184.7031 | 71948 | 5.3984 |
| combined | O_HALF | 1.5e-4 | new two-repeat | 16042 | 88266 | 181.7461 | 72209 | 5.4862 |
| combined | K_HALF | 1.5e-4 | archived PR21 | 15865 | 88266 | 179.7408 | 72387 | 5.5474 |
| 17 | F | released | archived PR21 | 8150 | 44133 | 184.6691 | 35975 | 5.3994 |
| 17 | O | 3e-4 | new two-repeat | 8177 | 44133 | 185.2809 | 35949 | 5.3816 |
| 17 | K | 3e-4 | archived PR21 | 8140 | 44133 | 184.4425 | 35985 | 5.4060 |
| 17 | O_HALF | 1.5e-4 | new two-repeat | 8035 | 44133 | 182.0633 | 36091 | 5.4767 |
| 17 | K_HALF | 1.5e-4 | archived PR21 | 7949 | 44133 | 180.1147 | 36177 | 5.5359 |
| 29 | F | released | archived PR21 | 8150 | 44133 | 184.6691 | 35975 | 5.3994 |
| 29 | O | 3e-4 | new two-repeat | 8202 | 44133 | 185.8473 | 35924 | 5.3652 |
| 29 | K | 3e-4 | archived PR21 | 8163 | 44133 | 184.9636 | 35963 | 5.3908 |
| 29 | O_HALF | 1.5e-4 | new two-repeat | 8007 | 44133 | 181.4289 | 36118 | 5.4958 |
| 29 | K_HALF | 1.5e-4 | archived PR21 | 7916 | 44133 | 179.3669 | 36210 | 5.5590 |

| Scope | Comparison | Call reduction | Paired 95% interval | Wins / regressions / ties | Gross saved / added |
|---|---|---:|---:|---:|---:|
| combined | K_HALF_vs_O_HALF | +1.1034% | [+0.4548, +1.8059]% | 125 / 82 / 49 | 336 / 159 |
| combined | K_vs_O | +0.4640% | [-0.2227, +1.1112]% | 97 / 112 / 47 | 315 / 239 |
| combined | K_HALF_vs_K | +2.6866% | [+1.9233, +3.5338]% | 175 / 48 / 33 | 570 / 132 |
| combined | O_HALF_vs_O | +2.0575% | [+1.4279, +2.7140]% | 145 / 66 / 45 | 467 / 130 |
| combined | O_vs_F | -0.4847% | [-1.3753, +0.4021]% | 100 / 124 / 32 | 304 / 383 |
| combined | K_vs_F | -0.0184% | [-1.0284, +0.8982]% | 107 / 113 / 36 | 369 / 372 |
| combined | O_HALF_vs_F | +1.5828% | [+0.8581, +2.3211]% | 137 / 73 / 46 | 418 / 160 |
| combined | K_HALF_vs_F | +2.6687% | [+1.8411, +3.5566]% | 154 / 62 / 40 | 560 / 125 |
| 17 | K_HALF_vs_O_HALF | +1.0703% | [+0.3810, +1.8128]% | 63 / 39 / 26 | 160 / 74 |
| 17 | K_vs_O | +0.4525% | [-0.3479, +1.2013]% | 48 / 53 / 27 | 150 / 113 |
| 17 | K_HALF_vs_K | +2.3464% | [+1.4715, +3.2975]% | 86 / 28 / 14 | 267 / 76 |
| 17 | O_HALF_vs_O | +1.7366% | [+1.0655, +2.4142]% | 72 / 34 / 22 | 211 / 69 |
| 17 | O_vs_F | -0.3313% | [-1.2717, +0.6044]% | 51 / 59 / 18 | 159 / 186 |
| 17 | K_vs_F | +0.1227% | [-0.9545, +1.1381]% | 57 / 58 / 13 | 195 / 185 |
| 17 | O_HALF_vs_F | +1.4110% | [+0.6927, +2.1460]% | 67 / 40 / 21 | 200 / 85 |
| 17 | K_HALF_vs_F | +2.4663% | [+1.6011, +3.3875]% | 77 / 33 / 18 | 274 / 73 |
| 29 | K_HALF_vs_O_HALF | +1.1365% | [+0.4068, +1.9180]% | 62 / 43 / 23 | 176 / 85 |
| 29 | K_vs_O | +0.4755% | [-0.4166, +1.3404]% | 49 / 59 / 20 | 165 / 126 |
| 29 | K_HALF_vs_K | +3.0258% | [+2.1274, +3.9901]% | 89 / 20 / 19 | 303 / 56 |
| 29 | O_HALF_vs_O | +2.3775% | [+1.5827, +3.2035]% | 73 / 32 / 23 | 256 / 61 |
| 29 | O_vs_F | -0.6380% | [-1.5722, +0.2822]% | 49 / 65 / 14 | 145 / 197 |
| 29 | K_vs_F | -0.1595% | [-1.2376, +0.8147]% | 50 / 55 / 23 | 174 / 187 |
| 29 | O_HALF_vs_F | +1.7546% | [+0.9553, +2.5633]% | 70 / 33 / 25 | 218 / 75 |
| 29 | K_HALF_vs_F | +2.8712% | [+2.0234, +3.7764]% | 77 / 29 / 22 | 286 / 52 |

The reverse O_HALF/K_HALF reduction is calculated with K_HALF calls as its denominator:

- combined: -1.1157% [-1.8391, -0.4569]%.
- 17: -1.0819% [-1.8463, -0.3824]%.
- 29: -1.1496% [-1.9555, -0.4085]%.

Objective-by-LR interaction uses the common F-call denominator; positive values mean K’s call advantage over O is greater at lower LR. It is not a subtraction of differently normalized reductions.

- combined: +0.6196% [-0.1758, +1.5232]%; numerator 101 calls / F 16300.
- 17: +0.6012% [-0.3467, +1.6207]%; numerator 49 calls / F 8150.
- 29: +0.6380% [-0.3502, +1.7258]%; numerator 52 calls / F 8150.

The 0.5% target is a research-priority point-estimate threshold, not deployment break-even. A positive interval alone does not satisfy that threshold or prove the true effect exceeds it. Practical equivalence requires the entire primary interval within ±0.5 percentage points; the provisional-base decision also requires both seed points within that band.

All intervals resample 128 joint request identities carrying both fixed seeds and all methods, using 10,000 PCG64 seed20261102 draws and linear percentiles. The 256 seed/request outcomes and repeats are not independent samples. Intervals are nominal/descriptive, conditional on these fitted trajectories and reused panel; they do not adjust for earlier K_HALF selection. Secondary intervals are unadjusted.

| Scope / candidate vs comparator | Retained gain | Lost gain | Added gain | Prevented harm | Added harm | Net saved |
|---|---:|---:|---:|---:|---:|---:|
| combined / K_HALF_vs_O_HALF | 309 | 109 | 251 | 85 | 50 | 177 |
| combined / O_HALF_vs_O | 219 | 85 | 199 | 268 | 45 | 337 |
| combined / K_HALF_vs_K | 261 | 108 | 299 | 271 | 24 | 438 |
| 17 / K_HALF_vs_O_HALF | 152 | 48 | 122 | 38 | 26 | 86 |
| 17 / O_HALF_vs_O | 113 | 46 | 87 | 124 | 23 | 142 |
| 17 / K_HALF_vs_K | 135 | 60 | 139 | 128 | 16 | 191 |
| 29 / K_HALF_vs_O_HALF | 157 | 61 | 129 | 47 | 24 | 91 |
| 29 / O_HALF_vs_O | 106 | 39 | 112 | 144 | 22 | 195 |
| 29 / K_HALF_vs_K | 126 | 48 | 160 | 143 | 8 | 247 |

Positive gains and harms are formed within seed/request before pooling. The identity added gain − lost gain + prevented harm − added harm equals net saved calls is checked at every paired outcome. Gross gains/harms, retained fractions and paired intervals are retained in SUMMARY.json. This accounting does not identify gradient conflict, memorization or a preservation mechanism.

## Work and cost

New training budget: 1,024 Adam steps, 2,048 ordinary response presentations/forward-backward equivalents, zero reference/diagnostic passes and zero disposable optimizer qualification steps. New serving budget: 48 compatibility-bridge plus 1,024 O/O_HALF executions = 1,072; all 112 gate/bridge executions are included. Actual completion/failure counters appear in SUMMARY.json.

| O_HALF seed | Valid events | Training seconds | Event wall seconds | Loading/resume seconds | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| 17 | 4 | 1423.212 | 1580.863 | 155.751 | 19.234 |
| 29 | 4 | 1419.227 | 1574.747 | 155.637 | 19.234 |

Historical O/K/K_HALF updater costs are unmatched context in COST_SUMMARY.json. Nested forward/head/compile/setup timers are not additive to their parent totals. Response losses are separately normalized observations captured before accumulation merging; window gradient/clipping records are counted once. O and K have different masks/normalization, so their raw losses are not evidence of serving improvement.

**No comparative serving latency or throughput result is supported.** New evaluations batch checkpoints and use two repeats; reused methods were measured previously with three. Request, native-generation, cache, loading and activation times are operational accounting only. No old/new timing pooling, call-count-to-speedup conversion, or training-inclusive speedup is claimed.

## Scope and one next action

Keep K_HALF as the provisional coding base; recommend a separately agreed bounded portability or online-setting check of this fixed base.

This recommendation does not authorize a follow-on experiment or a project-stage change. The fixed O-generated feedback stream is shared replay, not an O_HALF-controlled online deployment. Equal LR neither matches Adam displacement nor isolates K’s selection from its normalization. Lower LR also changes decoupled weight decay. No general learning-rate optimum, convergence, domain-shift/stochastic-decoding benefit or novelty claim follows.

Files: [paired counts](PAIRED_REQUESTS.csv), [all new evaluations](ALL_NEW_EVALUATIONS.csv), [summary](SUMMARY.json), [bridge audit](BRIDGE_REPORT.json), [training audit](TRAINING_AUDIT.json), [costs](COST_SUMMARY.json), [training costs](TRAINING_COSTS.csv), [evaluation accounting](EVALUATION_COSTS.json).
