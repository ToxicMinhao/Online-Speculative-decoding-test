# K+ successor assignment control

## Material Passport

ARS experiment-agent / validate; ANALYZED. Native-record reconstruction; one continuing seed17 and one fixed permutation.

Status: **complete**. Decision: **informative_placement_supported**.

Execution commit: `4d977e9ffd2cab9d58ae8806c29df11d843353da`.

Sealed N=256; 1536 planned timed completions. All three real step640 exports are timed afresh.

The panel excludes the full development set, training feedback and the768 SURVIVAL confirmation questions. Historical project exposure is documented; global unusedness is not claimed.

| Contrast | Calls candidate/reference | Gross saved/added | Call reduction % [95% CI] | Complete-time reduction % [95% CI] |
|---|---:|---:|---:|---:|
| TRUE_vs_SHUFFLED | 11522/11591 | 189/120 | 0.595289 [0.17096392277157424, 1.0154080568468404] | 1.245038 [-0.14040819047782588, 2.562570905845744] |
| TRUE_vs_K | 11522/11564 | 166/124 | 0.363196 [-0.0719358460567526, 0.7844907845903006] | 0.357830 [-0.9794432445858803, 1.6532584454771675] |
| SHUFFLED_vs_K | 11591/11564 | 134/161 | -0.233483 [-0.6535370677246283, 0.19540752795703814] | -0.898393 [-2.2747559699946827, 0.4515231030091096] |

| Arm | Calls once | Summed mean request seconds | Cap-hit requests |
|---|---:|---:|---:|
| K | 11564 | 208.689084 | 0 |
| TRUE | 11522 | 207.942331 | 0 |
| SHUFFLED | 11591 | 210.563931 | 0 |

| Contrast | Sweep1 complete-time reduction % | Sweep2 reduction % |
|---|---:|---:|
| TRUE_vs_SHUFFLED | 1.949924 | 0.531610 |
| TRUE_vs_K | 0.932937 | -0.222639 |
| SHUFFLED_vs_K | -1.037212 | -0.758280 |

Training, extra student projections, reference preparation, parent diagnostics and serving costs are retained separately in RESOURCES.json/RESOURCE_TIMERS.csv. Missing values are not imputed. Nested timers are not added. No training-inclusive speedup is inferred.

Matched-start repair counts and all parent-bit denominators are in MATCHED_START.json and MATCHED_START_COUNTS.csv. They address only the common first conditioning prefix; all later rounds contribute to endpoint calls.

The primary endpoint is TRUE versus SHUFFLED call reduction. An interval wholly above zero supports informative placement for this fork/permutation; a supported TRUE versus K call gain additionally establishes incremental verifier benefit. A resolved unfavorable TRUE/SHUFFLED contrast counts against this prioritization rule. An interval spanning zero leaves the successor-bit question open and does not establish equivalence.

Jointly resample whole requests across all arms:10,000 PCG64 draws, seed20261111, linear2.5/97.5 percentiles. Calls use one repeat after agreement; latency averages repeats within requests. Intervals are unadjusted. Latency benefits require their own favorable interval and both favorable sweep directions.

The historical58-event gain was only0.28% fewer calls; this one-event screen may lack precision at that scale. Manipulation strength and local repair do not override serving evidence. All11 fallacy checks are retained. Stop after event five and this sealed panel; no automatic follow-up.

| Manipulation quantity | Recorded value |
|---|---:|
| presentations | 256 |
| eligible_errors | 45255 |
| ineligible_errors | 2286 |
| mixed_strata | 3396 |
| flipped_bits | 18452 |
| uniform_expected_flips | 18416.406527254767 |
| reassigned_weight | 1827.4145187735558 |
| projection_calls | 7994 |
| projection_s | 50.37267164885998 |
| assignment_s | 8.139749048277736 |
| labels_changed | True |
| final_export_changed | True |

The uniform-permutation benchmark is the sum of2k(n−k)/n over eligible strata; the executed permutation is one fixed deterministic hash assignment. Counts across diverged arms need not match.
