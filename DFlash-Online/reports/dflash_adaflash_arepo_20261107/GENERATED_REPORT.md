# Released AdaFlash loss under our matched online protocol

## Material Passport

ARS experiment-agent run -> validate; status **valid_complete**. This analyzes retained native records and is not independent GPU replication.

Execution commit: `7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a`.

Configuration binding: `63ca016c7bd7dc460aba2c48c0eb1373db89aacf7660352e46c058f3a4571e54`. SUMMARY.json records checkpoint identities, source/configuration pins, qualification and work.

## Fixed comparison

A_repo is the only newly trained arm: released DFlash, fresh seed-17 BF16 AdamW, 58 own-feedback events, 7,424 Adam steps and 14,848 response presentations. O58, K+58 and paper-A58 are retained checkpoints, measured freshly together with A_repo58.

The loss ports AdaFlash 478bf0ff7a3a5232b13d0f14e752d5fe1da0b718: 0.8 teacher-argmax CE + 0.2 reverse KL, temperature 1. It preserves the source probability operations and intermediate dtypes, sums RKL over vocabulary, divides by the full response weight sum plus 1e-6, caps normalized positions at 0.01, then applies weights. Separately reduced CE and RKL are mixed last. All valid proposals receive loss; there is no K+ mask or adaptive-length head. This is not PR30's FP32-cap/rescaling proxy.

Each learning recipe collected its own feedback; this is an online-recipe endpoint comparison, not identical-feature isolation or a full AdaFlash system reproduction.

## Fresh endpoint measurements

| Measurement | O58 | K+58 | Paper-A58 | A_repo58 |
|---|---:|---:|---:|---:|
| Verification calls | 44730 | 43267 | 45083 | 43721 |
| Output tokens | 314847 | 314847 | 314847 | 314847 |
| Complete-request seconds | 808.8613966275007 | 785.696731566141 | 816.0723522882909 | 795.3523886085798 |
| Native-generation seconds | 757.9408401226004 | 735.7823551477243 | 764.8649670823166 | 742.1921015356978 |
| Cache-maintenance seconds | 36.55770078177253 | 36.56589214627941 | 36.72599864502748 | 39.508712972203895 |
| Runtime accepted counter / verification | 6.0159400849541695 | 6.253172163542653 | 5.961005256970477 | 6.17785503533771 |
| Matched accepted draft tokens / verification | 6.037044489157165 | 6.275498647930293 | 5.9818113257769 | 6.198966171862492 |
| Committed progress / verification | 7.0159400849541695 | 7.253172163542653 | 6.961005256970477 | 7.17785503533771 |

The native accepted counter is committed progress minus one per verification round. Matched acceptance clips the canonical-matching draft prefix at the stopping boundary. EOS/caps can make these counters differ. Committed progress excludes each request's initial target-prefill token. Every ratio uses summed counts; all counters remain in the tables.

**KPLUS_vs_AREPO:** complete-request reduction 1.214010% [95% paired CI [0.5722085389220982, 1.861133599296821]]; call reduction 1.038403%. Request wins/regressions/ties 472/370/182; gross calls saved/added 1256/802.

**KPLUS_vs_O:** complete-request reduction 2.863861% [95% paired CI [2.175156480205154, 3.563900052452744]]; call reduction 3.270736%. Request wins/regressions/ties 633/250/141; gross calls saved/added 1917/454.

**AREPO_vs_O:** complete-request reduction 1.670126% [95% paired CI [0.9716167763817916, 2.3544437722097635]]; call reduction 2.255757%. Request wins/regressions/ties 593/258/173; gross calls saved/added 1481/472.

**AREPO_vs_A:** complete-request reduction 2.538986% [95% paired CI [1.8371000110119753, 3.241923748613239]]; call reduction 3.021094%. Request wins/regressions/ties 663/202/159; gross calls saved/added 1714/352.

Primary decision: **favors_KPLUS**. The paired K+/A_repo complete-request-time interval determines this label. Other intervals are unadjusted descriptive comparisons.

Time totals sum per-request means over three repeats. Count totals use one validated trajectory per request after repeat consistency. The bootstrap uses 10,000 paired-request draws, PCG64 seed 20261102 and linear 2.5/97.5 percentiles. Repeats are not independent requests.

## Costs, validity and limits

Qualification is separately retained in SUMMARY.json: at most six disposable AdamW steps before complete scientific-state reset. It does not count toward the 7,424 main steps. Four inherited native correctness completions remain separate from the 12,288 timed completions.

A_repo worker wall: 31668.327 seconds. TRAINING.json and EVENT_COSTS.csv separate training, teacher preparation, persistence, activation and publication; nested timers must not be added. Exposure records count original response observations once, excluding checkpoint recomputation and accumulation copies.

O/K+/paper-A training costs are historical context, not fresh matched training-overhead measurements. Fresh endpoint serving gains do not establish training-inclusive or asynchronous deployment speedup. Loading/switching remain outside request latency. No detailed kernel breakdown is inferred from aggregate request times.

The result concerns one seed and a reused train-disjoint mathematics panel. It does not establish other-domain or seed robustness, full-system AdaFlash performance, or an isolated causal explanation of any difference. All measured regressions remain in the paired totals. No tuning or follow-up is authorized.

Deliverables: SUMMARY.json, VALIDITY.json, ALL_REPEATS.csv, PAIRED_REQUESTS.csv (only for a valid complete endpoint), REPEAT_TOTALS.json, TRAINING.json, TRAINING_EXPOSURE.csv and EVENT_COSTS.csv. Raw failures and unattempted/unknown counts remain in the original status and validity evidence.
