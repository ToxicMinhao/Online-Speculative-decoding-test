# Completed VFD endpoint and native-exposure review

Both prepared private CPU audits passed on the verified production mirror, without script repairs or model work. Execution remains `05e4b30488fb267a196edb1a81a5f6aaf613a838`. The endpoint reconstruction imported no production evaluation/statistics/report functions. It passed 646,094 checks, authenticated 4,058 source files against the original inventory, recounted 89,374 native verification blocks, and matched the saved SUMMARY.json. The exposure supplement checked 31 retained training/scoring receipt files; every source size and SHA-256 also matches the frozen original inventory. A separate child review found no discrepancy across 247 statistic comparisons, 701 exposure arithmetic checks and 145 production pass-table checks. The full mirror’s separate retrieval gate covers 10,049 files / 613,289,482 bytes.

Authorship overlap: this reviewer authored the production data, evaluation, statistics and reporting modules. These are separate same-team CPU calculations and evidence checks, not blind independent implementation review or independent GPU replication. A separate child reviewer checked the supplement’s formulas and record-binding safeguards.

All 2,000 scheduled executions for 200 requests × five arms × two repeats are included. Each request’s two timings are averaged before the fixed 10,000 paired request-bootstrap draws (PCG64 seed 20261104, shared indices, linear 2.5/97.5 percentiles). All arms retain the one capped request and 199 EOS responses. Each has 59,952 output tokens over the 200 unique requests; count totals below count each repeated-identical request once.

| PROGRESS64 versus | Request-time reduction | 95% paired interval |
|---|---:|---:|
| FULL | -0.831263% | [-1.538366%, -0.146340%] |
| STATIC64 | -0.392136% | [-1.042477%, +0.261770%] |
| RANDOM64 | +0.012889% | [-0.698658%, +0.743770%] |
| ONCE | +0.741453% | [-0.079564%, +1.557417%] |

PROGRESS64 is slower than FULL in this screen. The predeclared decision is **otherwise**: advancement fails, and the efficiency criterion also fails because the primary lower bound −1.5384% is below the allowed −0.5%, despite a 21.4587% measured updater-time saving. STATIC/RANDOM comparisons do not establish added scoring value. No simpler arm satisfies the declared efficiency rule either. No automatic extension or tuning is authorized; Stage 1 / project None does not advance.

| Policy | Verification calls | Mean request ms | Logical updater s | Native training blocks | Selected supervised occurrences |
|---|---:|---:|---:|---:|---:|
| FULL | 8,872 | 759.955657 | 1414.155988 | 264,848 | 2,214,949 |
| ONCE | 9,008 | 771.996877 | 700.573933 | 132,424 | 1,008,301 |
| RANDOM64 | 8,946 | 766.371664 | 1053.034990 | 199,476 | 1,616,557 |
| STATIC64 | 8,916 | 763.279793 | 1087.225616 | 198,170 | 1,599,854 |
| PROGRESS64 | 8,945 | 766.272885 | 1110.696630 | 201,294 | 1,631,313 |

Native training exposure is recovered from each original forward once: nonempty blocks equal no-error blocks plus the first-error-position histogram, cross-checked against K+ priority-error counts and sampler geometry. Counts are occurrences, not unique tokens. No empty native block occurred. Training uses up to 512 anchors per response; selector score blocks come from its separate up-to-16-anchor forwards.

| Policy | Second-pass response token occurrences | Second-pass native blocks | Selected score blocks |
|---|---:|---:|---:|
| FULL | 137,308 | 132,424 | not scored |
| ONCE | 0 | 0 | not scored |
| RANDOM64 | 71,212 | 67,052 | not scored |
| STATIC64 | 66,763 | 65,746 | 4096 |
| PROGRESS64 | 71,758 | 68,870 | 4096 |

Equal 64-response selections therefore did not imply equal native exposure. The supplemental FORWARDS.csv retains every actual selected length and exposure count; SELECTIONS.csv and pass tables summarize them. ONCE’s second-pass exposure is zero. A blank score count for an unscored policy is not zero native exposure.

Physical main work reconciles to 1,664 optimizer steps / 3,328 presentations / 866,020 native blocks / 7,105,426 selected supervised-position occurrences out of 12,643,929 available. Logical standalone work is 1,920 steps / 3,840 presentations / 996,212 native blocks / 8,070,974 selected occurrences out of 14,543,049 available. The difference attributes event-1 shared first-pass work to each policy logically while counting it once physically. Qualification remains separate: two steps / four presentations / eight score calls / 592 native blocks / 3,717 selected occurrences.

All 512 feedback generations, 512 teacher preparations, 1,408 main score forwards and 58 missing-reference generations reconcile. The 58 references are excluded from endpoint timing and distinct from the 512 feedback responses. No unknown or excess work was introduced by this analysis.

Physical training-pass wall is 4,596.303964 s, exactly matching the production cost total; its nested compute timer is 4,575.960361 s. Separate production totals are preparation 44.617481 s, scoring 78.569348 s, selection 0.279676 s, persistence 730.563776 s, export 291.426833 s and activation 149.298713 s. Logical updater timers sum pass wall plus scoring and selection, including packet loading/setup, and match the original report for all five policies. Event-1 policy worker wall excludes the earlier shared worker; logical updater costs attribute the relevant shared work. Nested or asynchronous worker timers are not added to endpoint latency.

Limits: one seed, four common-feedback events, and a reused train-disjoint benchmark panel. This establishes neither broad-domain robustness nor a causal explanation for individual score-selected responses. Earlier independent normalization checks found no overlap between the original 512 training questions and this 200-question panel after stripping the fixed training suffix; historical project exposure remains disclosed (four unmatched / 196 previously used in available inventories).

Evidence: [endpoint audit](endpoint_audit_001/AUDIT.json), [paired reconstruction](endpoint_audit_001/PAIRED_RECONSTRUCTION.csv), [native exposure audit](training_exposure_001/TRAINING_EXPOSURE_AUDIT.json), [logical exposure](training_exposure_001/LOGICAL_ARM_TOTALS.csv), [selected exposure](training_exposure_001/SELECTIONS.csv), [worker costs](training_exposure_001/WORKER_COSTS.csv).
