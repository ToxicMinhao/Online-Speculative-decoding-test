# First-group cost audit

ARS bounded accounting review of the retrieved GSM8K recovery evidence; no execution or scientific-validity judgment. The saved `COSTS.json` reflects 48 completed generations, including 16 reused F records. The subsequently retrieved `STATUS.json` is still running in CodeAlpaca; recovery and whole-campaign wall time are not yet available.

| Observed GSM8K boundary, seconds | C | P |
|---|---:|---:|
| Trainer loading, outside event | 17.943 | 16.906 |
| Training steps, including objective work | 31.526 | 31.687 |
| Checkpoint construction/save | 38.068 | 37.728 |
| Export | 18.113 | 18.197 |
| Event wall through receipt construction | 87.859 | 88.899 |
| Native activation | 0.366 | 0.408 |
| Instrumented serving, A / B | 14.029 / 13.974 | 16.058 / 13.480 |

Shared teacher preparation for eight A responses is **13.874 s**, with **0.378 s** feature persistence, charged once for the executed campaign. This replay made 372 target forwards, including prefill and final-feature forwards; they are preparation work, not additional generated answers. F serving is **16.123 s A + 11.974 s B = 28.097 s**, measured in the original attempt. Sources: [preparation](retrieved/first_group_recovery/campaign_002/gsm8k/F/PREPARATION.json), [C training](retrieved/first_group_recovery/campaign_002/gsm8k/C/TRAINING.json), [P training](retrieved/first_group_recovery/campaign_002/gsm8k/P/TRAINING.json), [cost receipts](retrieved/first_group_recovery/campaign_002/COSTS.json).

The table contains nested boundaries and must not be summed down the columns. `training_s` times each batch preparation/train step and metric synchronization, excluding its preceding feature-file load. Event wall includes those file loads, validation, reference setup, training, checkpoint/save and export; it stops before writing `RESULT.json` and committing the checkpoint rename. Checkpoint/save includes state construction as well as persistence. P's reference forward **0.376 s**, head **1.624 s**, and projection **0.850 s** are already inside training; their **2.850 s** sum is an instrumented subcost, not additional training time. Reference setup **0.008 s** is outside step timing but inside event wall. Chunk-checkpoint recomputation can contribute to head/projection counters. Synchronization and instrumentation prevent treating these as isolated kernel costs or subtracting them to obtain an uninstrumented counterfactual. See [trainer](repair_001/online_feedback/training.py) and [objective](repair_001/online_feedback/objective.py).

Preparation includes replay, feature assembly and CPU transfer; feature persistence separately times the tensor payload save. Its following JSON receipt write is outside both timers. Native serving includes fresh-cache generation, RNG preservation, compact trace collection and synchronization; tokenization, output validation, JSON persistence, preparation and adaptation are outside request latency. These are single sequential instrumented observations, with no timing repetitions or extra warmups. They do not establish deployment speedup. `activation_s` and `native_activation_s` are aliases of the same interval; count one. Zero `cache_maintenance_s` describes request-local cache policy, not zero cost for constructing request caches. See [native boundaries](repair_001/online_feedback/native.py).

The current [reporter](report.py) retains the major nested boundaries correctly, but `COSTS.json` is not an exhaustive decomposition. It omits top-level `TRAINING.json.model_loading_s` and `diagnostic_validation_s` (**0.745 s C; 1.834 s P**), plus `CHECKS.json.publication_validation_s` (**0.804 s C; 0.838 s P**). The diagnostic aggregate mixes initial checks outside event wall with P reference checks inside it, so adding the entire aggregate to event wall would overlap. Publication validation is separate from the reported activation interval. Frozen resets for later groups, retirement/cleanup, report writes and other bookkeeping also lack dedicated entries in `COSTS.json`; completed runner wall time will contain most of them. Do not label a sum of listed components “end to end.” Sources: [runner](runner.py), [C checks](retrieved/first_group_recovery/campaign_002/gsm8k/C/CHECKS.json), [P checks](retrieved/first_group_recovery/campaign_002/gsm8k/P/CHECKS.json).

GPU peaks are **28.002 / 28.459 GiB allocated/reserved for C** and **36.897 / 37.850 GiB for P**. The runner resets allocator peaks before constructing each trainer and reads them after training/export validation, before activation and adapted serving. These are process-wide peaks for that interval, including the already-resident native target/draft; reserved memory also reflects allocator caching. They are neither incremental objective memory nor whole-campaign peaks, and must not be summed across arms. They exclude CPU optimizer masters/moments, host feature storage and disk. The observed difference does not isolate the reference model's causal memory cost.

The original failed attempt's runner wall is **37.984 s**, already containing its **9.072 s** native loading and **28.097 s** F serving. The remaining **0.816 s** includes failed preparation and other overhead; there is no separate failed-preparation timer, so do not attribute the entire residual to replay. All 16 copied F JSON records match the retained originals. For total consumed runner time, eventually report **original `wall_s` + recovery `wall_s`**; do not add the reused F serving total again. Recovery wall excludes the original F execution, while its logical result tables retain those original timings. A component-level campaign ledger should likewise deduplicate F once and identify its original-attempt provenance. The failed attempt and engineering pause remain explicit; summing runner intervals excludes the repair pause, development, external preflight and retrieval. Sources: [original status](retrieved/campaign_001/STATUS.json), [recovery status](retrieved/first_group_recovery/campaign_002/STATUS.json), [failure analysis](repair_001/FAILURE_ANALYSIS.md).

## Final stopped-run accounting

This final section supersedes the earlier pending-time statements. Recovery stopped on the exact-output gate at `sharegpt_A_07` in P after **136/144 observed generations**, comprising 16 reused F records and 120 new generations. All six training events had completed: **96 response presentations and 48 optimizer steps**. The eight unexecuted requests are ShareGPT P's held-out B set; their cost is unobserved, not measured zero. The invalid A07 generation and all ShareGPT preparation, training, export and publication work remain charged. See [final status](retrieved/final_recovery/campaign_002/STATUS.json).

Consumed runner time is **37.984244345 + 1175.149142241 = 1213.133386586 s** across the original and recovery attempts. Recovery wrapper elapsed time is **1177.474 s**, a broader boundary for that same recovery process, not another additive cost. The 16 reused F timings remain part of the original interval and are not added again. Across 136 distinct observed generations, instrumented serving sums to **455.378843 s**: **28.096905 s** original F plus **427.281938 s** new recovery serving. The failed ShareGPT P A07 request contributes **7.865683 s** to those totals. These are consumed-cost observations, not a speedup result. Sources: [wrapper resource receipt](retrieved/final_recovery/RESOURCE_USAGE.txt), [final cost receipts](retrieved/final_recovery/campaign_002/COSTS.json).

| Event | Loading, s | Step training, s | Event wall, s | Observed serving requests / s | GPU peak allocated / reserved, GiB |
|---|---:|---:|---:|---:|---:|
| GSM8K C | 17.943 | 31.526 | 87.859 | 16 / 28.003 | 28.002 / 28.459 |
| GSM8K P | 16.906 | 31.687 | 88.899 | 16 / 29.538 | 36.897 / 37.850 |
| CodeAlpaca C | 17.432 | 30.284 | 86.731 | 16 / 42.389 | 34.231 / 34.785 |
| CodeAlpaca P | 17.601 | 32.577 | 89.932 | 16 / 44.195 | 36.792 / 37.582 |
| ShareGPT C | 17.414 | 27.768 | 83.300 | 16 / 91.626 | 34.572 / 35.551 |
| ShareGPT P, stopped during serving | 17.015 | 31.014 | 87.317 | 8 / 55.689 | 35.843 / 36.461 |

All six events contribute **184.856 s step training**, **227.633 s checkpoint/save**, and **106.995 s export**, nested within **524.037 s event wall**; trainer loading adds its separately measured **104.311 s** boundary. Shared teacher preparation totals **78.405 s**, plus **1.366 s** feature persistence, across the three groups. These component observations are not a complete partition of runner wall. The prior overlap and peak-memory qualifications still apply. [COST_BOUNDARIES.csv](COST_BOUNDARIES.csv) preserves all six raw receipt rows, including loading, mixed-overlap diagnostic validation, publication validation, reference/projection subcosts, and explicit peak scope; no valid-result filter removes failure costs.

ShareGPT P stopped before retirement. Its restart state remains retained under the stopped-run disposition; no P retirement receipt exists in the retrieved evidence. Earlier completed arms retain their recorded retirement history. This audit performs no retirement, restart, additional generation, or efficacy inference.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
