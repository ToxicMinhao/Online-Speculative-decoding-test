# Released-prefix protection v1: fixed coding comparison

Execution source: `d10521a1aa7d256bb75b37a6aae07586ae7cc8b3`.

Hypothesis: protecting released-DFlash prefixes may remove more harmful changes than useful K gains. The expected pattern allowed partial harm reduction together with lost gains; a modest or unresolved effect was anticipated.

mixed fitted-seed directions.

| Scope | Method | Calls | Output tokens | Calls / 1,000 tokens | Accepted draft tokens | Progress / call |
|---|---|---:|---:|---:|---:|---:|
| combined | F | 16300 | 88266 | 184.6691 | 71950 | 5.3994 |
| combined | K | 16303 | 88266 | 184.7031 | 71948 | 5.3984 |
| combined | K_HALF | 15865 | 88266 | 179.7408 | 72387 | 5.5474 |
| combined | KP | 16247 | 88266 | 184.0686 | 72001 | 5.4170 |
| 17 | F | 8150 | 44133 | 184.6691 | 35975 | 5.3994 |
| 17 | K | 8140 | 44133 | 184.4425 | 35985 | 5.4060 |
| 17 | K_HALF | 7949 | 44133 | 180.1147 | 36177 | 5.5359 |
| 17 | KP | 8144 | 44133 | 184.5331 | 35979 | 5.4034 |
| 29 | F | 8150 | 44133 | 184.6691 | 35975 | 5.3994 |
| 29 | K | 8163 | 44133 | 184.9636 | 35963 | 5.3908 |
| 29 | K_HALF | 7916 | 44133 | 179.3669 | 36210 | 5.5590 |
| 29 | KP | 8103 | 44133 | 183.6041 | 36022 | 5.4307 |

| Scope | Comparison | Call reduction | Paired 95% interval | Request-time reduction | Wins / regressions / ties |
|---|---|---:|---:|---:|---:|
| combined | KP_vs_K | +0.3435% | [+0.0182, +0.6919]% | +0.1770% | 102 / 80 / 74 |
| combined | KP_vs_F | +0.3252% | [-0.6386, +1.2365]% | +0.3620% | 112 / 110 / 34 |
| combined | KP_vs_K_HALF | -2.4078% | [-3.2861, -1.6281]% | -2.2350% | 57 / 163 / 36 |
| combined | K_HALF_vs_K | +2.6866% | [+1.9233, +3.5338]% | +2.3592% | 175 / 48 / 33 |
| combined | K_HALF_vs_F | +2.6687% | [+1.8411, +3.5566]% | +2.5402% | 154 / 62 / 40 |
| combined | K_vs_F | -0.0184% | [-1.0284, +0.8982]% | +0.1853% | 107 / 113 / 36 |
| 17 | KP_vs_K | -0.0491% | [-0.4838, +0.4037]% | -0.0916% | 44 / 41 / 43 |
| 17 | KP_vs_F | +0.0736% | [-0.9387, +1.0477]% | -0.0939% | 58 / 56 / 14 |
| 17 | KP_vs_K_HALF | -2.4531% | [-3.3587, -1.6254]% | -2.4150% | 27 / 80 / 21 |
| 17 | K_HALF_vs_K | +2.3464% | [+1.4715, +3.2975]% | +2.2687% | 86 / 28 / 14 |
| 17 | K_HALF_vs_F | +2.4663% | [+1.6011, +3.3875]% | +2.2664% | 77 / 33 / 18 |
| 17 | K_vs_F | +0.1227% | [-0.9545, +1.1381]% | -0.0023% | 57 / 58 / 13 |
| 29 | KP_vs_K | +0.7350% | [+0.1276, +1.3442]% | +0.4462% | 58 / 39 / 31 |
| 29 | KP_vs_F | +0.5767% | [-0.4730, +1.5482]% | +0.8172% | 54 / 54 / 20 |
| 29 | KP_vs_K_HALF | -2.3623% | [-3.4349, -1.3898]% | -2.0541% | 30 / 83 / 15 |
| 29 | K_HALF_vs_K | +3.0258% | [+2.1274, +3.9901]% | +2.4500% | 89 / 20 / 19 |
| 29 | K_HALF_vs_F | +2.8712% | [+2.0234, +3.7764]% | +2.8136% | 77 / 29 / 22 |
| 29 | K_vs_F | -0.1595% | [-1.2376, +0.8147]% | +0.3727% | 50 / 55 / 23 |

Combined decisions count 256 seed/request pairs, while uncertainty resamples 128 joint request identities carrying both fitted seeds and all methods. The same 10,000 PCG64 seed-20261102 draws are reused everywhere. Intervals are conditional on these two trajectories/panel; secondary intervals are descriptive and unadjusted.

| Candidate | Retained K gain | Lost gain | Added gain | Prevented harm | Added harm | Retained fraction |
|---|---:|---:|---:|---:|---:|---:|
| KP | 301 | 68 | 91 | 112 | 79 | 0.8157 |
| K_HALF | 261 | 108 | 299 | 271 | 24 | 0.7073 |

| Seed / event | Mean K loss | Mean protection loss | Protected positions | Active / protected | Incorrect / protected | Wrong ties |
|---|---:|---:|---:|---:|---:|---:|
| 17 / 1 | 0.266816 | 0.047718 | 437100 | 0.03800732097918096 | 0.02822695035460993 | 772 |
| 17 / 2 | 0.305113 | 0.066748 | 458577 | 0.042769262304912806 | 0.031837619418331055 | 805 |
| 17 / 3 | 0.314923 | 0.097615 | 417831 | 0.05106849419980806 | 0.03840069310319244 | 826 |
| 17 / 4 | 0.288120 | 0.068167 | 421876 | 0.04675070399833126 | 0.035064805772312246 | 737 |
| 29 / 1 | 0.266132 | 0.051421 | 437289 | 0.03717907379330374 | 0.026998163685800467 | 776 |
| 29 / 2 | 0.310282 | 0.071404 | 458705 | 0.04063613869480385 | 0.029687925791085774 | 882 |
| 29 / 3 | 0.318180 | 0.090745 | 417681 | 0.04911403678884124 | 0.036614066715986604 | 882 |
| 29 / 4 | 0.288901 | 0.069593 | 421118 | 0.04536258245907323 | 0.034140074753394534 | 781 |

| Trajectory | Completed events | Training seconds | Reference forward/head seconds (included) | Peak allocated GiB |
|---|---:|---:|---:|---:|
| 17/K_HALF | 4 | 1490.584 | 0.000 | 19.286 |
| 17/KP | 4 | 1847.024 | 345.130 | 21.239 |
| 29/K_HALF | 4 | 1492.825 | 0.000 | 19.259 |
| 29/KP | 4 | 1848.319 | 343.737 | 21.212 |

Historical original K training seconds (unmatched context): seed 17: 1486.487, seed 29: 1486.279.

All losses/exposures in TRAINING_RESPONSES.csv are original-response observations before accumulation merging. TRAINING_WINDOWS.csv records one native gradient norm/clipping decision per optimizer window. TRAINING_AGGREGATES.json averages normalized response losses and sums exposure without counting checkpoint recomputation twice. These are training-block metrics, not native acceptance or a measured matched-K preservation counterfactual.

Serving totals use one validated trajectory per request; times average its three repeats. Complete-request, native-generation and cache-maintenance scopes remain separate. TRAINING_COSTS.csv preserves cold compilation, reference setup/forwards/heads, extra loss, teacher heads, persistence, export, state validation and memory scopes. Nested subcosts overlap parent timers and must not be added to them. EVALUATION_COSTS.json separates publication/switching from serving. Historical K training costs are context, not fresh matched costs.

This is a new short-prompt CodeAlpaca panel under the same raw-row project partition. Source answers were unused; all selected IDs/tokens and exclusions were sealed before model work. No O comparison, general convergence, stochastic-decoding, multi-turn domain-switching, asynchronous deployment or training-inclusive speedup claim follows.

Next action: stop at this fixed comparison. Review its paired uncertainty, gain/harm profile and lower-LR control before agreeing to any subsequent experiment. No automatic tuning or extension.
