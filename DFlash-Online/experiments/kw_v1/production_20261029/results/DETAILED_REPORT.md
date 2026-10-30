# KW-v1: learned response weights through event 12

## Material Passport

ARS experiment-agent run/validate. ANALYZED by raw-record reconstruction; no independent GPU replication. Stage 1 / project None unchanged.

Execution source: `708e974e06f5d9f4b186c32607c16b37cbd4b14f`.

**Mixed/inconclusive**. Primary KW12 versus K12 new-final64 call reduction: **+0.2154%**. Final four-panel pooled reduction: **-0.0700%**.

Paired request bootstrap percentile 95% interval: [-0.6897%, +1.0527%]. 10,000 resamples of 64 request pairs, isolated NumPy PCG64 seed 20261030, ratio of summed calls. This interval is conditional on this run/checkpoint and panel; it does not establish training-seed robustness.

| Event | Panel | Comparison | Valid requests | Calls (reference → candidate) | Call reduction | Wins/losses/ties | Gross saved/added | Request-time reduction | Native-time reduction |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | development | KW_vs_K | 64 | 2871 → 2856 | +0.5225% | 25/23/16 | 44/29 | +0.9435% | +0.5175% |
| 10 | development | KW_vs_O | 64 | 2884 → 2856 | +0.9709% | 32/21/11 | 71/43 | +1.0025% | +0.9359% |
| 10 | development | K_vs_O | 64 | 2884 → 2871 | +0.4508% | 29/29/6 | 66/53 | +0.0595% | +0.4206% |
| 10 | additional | KW_vs_K | 64 | 3019 → 3030 | -0.3644% | 17/24/23 | 23/34 | -0.3607% | -0.3671% |
| 10 | additional | KW_vs_O | 64 | 3080 → 3030 | +1.6234% | 37/21/6 | 102/52 | +1.5771% | +1.5485% |
| 10 | additional | K_vs_O | 64 | 3080 → 3019 | +1.9805% | 37/15/12 | 96/35 | +1.9308% | +1.9086% |
| 10 | fresh | KW_vs_K | 64 | 2921 → 2927 | -0.2054% | 22/23/19 | 32/38 | -0.1877% | -0.1917% |
| 10 | fresh | KW_vs_O | 64 | 2980 → 2927 | +1.7785% | 30/20/14 | 97/44 | +1.7068% | +1.7378% |
| 10 | fresh | K_vs_O | 64 | 2980 → 2921 | +1.9799% | 31/20/13 | 104/45 | +1.8910% | +1.9258% |
| 12 | development | KW_vs_K | 64 | 2831 → 2822 | +0.3179% | 28/24/12 | 52/43 | +0.3826% | +0.3195% |
| 12 | development | KW_vs_O | 64 | 2925 → 2822 | +3.5214% | 43/10/11 | 131/28 | +3.4170% | +3.4175% |
| 12 | development | K_vs_O | 64 | 2925 → 2831 | +3.2137% | 43/12/9 | 122/28 | +3.0461% | +3.1080% |
| 12 | additional | KW_vs_K | 64 | 2958 → 2971 | -0.4395% | 23/26/15 | 37/50 | -0.5607% | -0.4103% |
| 12 | additional | KW_vs_O | 64 | 3091 → 2971 | +3.8822% | 43/10/11 | 134/14 | +3.6864% | +3.7664% |
| 12 | additional | K_vs_O | 64 | 3091 → 2958 | +4.3028% | 42/10/12 | 152/19 | +4.2234% | +4.1596% |
| 12 | fresh | KW_vs_K | 64 | 2859 → 2869 | -0.3498% | 17/28/19 | 31/41 | -0.3336% | -0.3290% |
| 12 | fresh | KW_vs_O | 64 | 2979 → 2869 | +3.6925% | 42/10/12 | 135/25 | +3.6092% | +3.5708% |
| 12 | fresh | K_vs_O | 64 | 2979 → 2859 | +4.0282% | 44/10/10 | 142/22 | +3.9297% | +3.8870% |
| 12 | final | KW_vs_K | 64 | 2786 → 2780 | +0.2154% | 25/20/19 | 40/34 | +0.2302% | +0.2194% |
| 12 | final | KW_vs_O | 64 | 2849 → 2780 | +2.4219% | 35/20/9 | 104/35 | +2.3086% | +2.3479% |
| 12 | final | K_vs_O | 64 | 2849 → 2786 | +2.2113% | 34/18/12 | 105/42 | +2.0832% | +2.1332% |
| 12 | pooled four panels | KW_vs_K | 256 | 11434 → 11442 | -0.0700% | 93/98/65 | 160/168 | -0.0766% | -0.0559% |
| 12 | pooled four panels | KW_vs_O | 256 | 11844 → 11442 | +3.3941% | 163/50/43 | 504/102 | +3.2684% | +3.2895% |
| 12 | pooled four panels | K_vs_O | 256 | 11844 → 11434 | +3.4617% | 163/50/43 | 521/111 | +3.3425% | +3.3436% |

| Arm | Four-event measured training (s) |
|---|---:|
| O | 1224.929605 |
| K | 1277.774415 |
| KW | 1802.709259 |

KW versus K measured training overhead: +41.0820%. COSTS.json retains every event cost and controller/objective-work receipt, including loading, checkpoint persistence, export and memory. Timers are nested; do not add them twice. Updater cost is separate from request latency. This shared-feedback campaign does not measure deployed asynchronous queue/lag behavior or training-inclusive speedup.

| KW measured component | Four-event sum (s) |
|---|---:|
| ordinary_training_s | 1265.873100 |
| capture_host_s | 1.019443 |
| credit_and_controller_s | 103.797570 |
| previews_s | 295.506516 |
| weighting_s | 96.713562 |
| live_update_s | 34.046991 |
| window_observed_s | 1804.372538 |

Ordinary training observation includes loading/metric/capture components. Capture host time records enqueue/clone time and is nested in that observation. Window-observed time overlaps all window components. These rows are not additive independent costs.

Raw audit: 4032/4032 valid generations; 7/7 complete valid fixed panel endpoints. Every output/termination, native count trace and repeat count is revalidated. The event-10 first-eight-development gate uses 72 included generations. Retained F/K1/O8 canonicals remain fixed; new final canonicals come from first O12, deferring earlier K/KW rows. This is native-verifier agreement, not extra target-only verification.

Counts use one validated repeat-0 trajectory per request; request and native times use each request’s three-repeat mean. Repeats are not independent samples. EVENT_PANEL.csv and SUMMARY.json include exact calls, calls per 1,000 output tokens, accepted-draft and committed progress per call, every panel/endpoint and O comparison. PER_REQUEST.csv and ALL_REPEATS.csv retain rejected/incomplete evidence. A gain over O without a gain over K does not support the added component.

KW adds a bounded five-feature allocation controller to unchanged K. Two isolated actual BF16 Adam previews produce a finite secant, credited with the next ordinary unweighted training-window gradient. That next window is shuffled training data, not held-out or necessarily the next chronological user. This is an approximate first-order loss-transfer signal, not an exact derivative through BF16, a serving objective, or future Adam history. Generic reweighting and next-batch gradient credit have prior art; no established novelty or causal-mechanism claim is made.

ARS fallacy scan, 11/11: Simpson (all panels visible beside pooled result); ecological (paired requests); Berkson (fixed selected population limits generalization); collider (no performance filtering); base-rate neglect (pooled denominators); regression to mean (no extreme-outcome selection); survivorship (invalid/missing evidence visible); look-elsewhere (all seven endpoints); forking paths (fixed primary and bootstrap); correlation/causation (no mechanism attribution); reverse causality (training credit is not evidence of the serving cause).

Stop at event 12 and this report; no fourth arm, tuning or event 13.
