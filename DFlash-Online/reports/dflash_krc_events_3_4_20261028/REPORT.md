# KRC continuation through event 4

Execution **completed**; independent audit **passed**.

KRC2’s complete corrected weights, own BF16 Adam state, RNG and history continue through the retained O-generated event-3/4 feedback. O is the main OSD-on-DFlash baseline; K is the correction ablation. The panels remain separate.

At every ordinary K update, the next disjoint response pair in the same pass supplies its current correct prefixes. The qualified post-K reference correction adjusts draft weights while leaving Adam moments/counters unchanged. Optional diagnostic probes remain disabled.

Both panels completed the declared endpoint comparison and passed the retained-evidence audit.

## development

Valid common requests: 64/64.
Against O, the advantage persists and shrinks: event 2 +3.0075% → event 4 +1.4696% (-1.5379 percentage points). Against K, the advantage reverses: event 2 +0.2018% → event 4 -0.2719% (-0.4737 percentage points). KRC4 versus KRC2 alone changes calls by 0.5730% reduction; that within-arm change does not establish relative benefit.

| Comparison | Baseline / method calls | Call reduction | Request-time reduction | Wins / losses / ties | Saved / added |
|---|---|---|---|---|---|
| KRC4 vs O4 | 2994 / 2950 | 1.4696% | 1.5959% | 35 / 21 / 8 | 85 / 41 |
| K4 vs O4 | 2994 / 2942 | 1.7368% | 1.9263% | 34 / 20 / 10 | 94 / 42 |
| KRC4 vs K4 | 2942 / 2950 | -0.2719% | -0.3369% | 24 / 28 / 12 | 46 / 54 |

| Event | Measurement | O calls | K calls | KRC calls |
|---|---|---|---|---|
| 1 | historical | 3075 | 3027 | 3005 |
| 2 | historical | 3059 | 2973 | 2967 |
| 4 | fresh | 2994 | 2942 | 2950 |

| Arm | Complete request s | Native generation s | Cache maintenance s |
|---|---|---|---|
| O4 | 51.916256 | 50.705822 | 0.031297 |
| K4 | 50.916211 | 49.827974 | 0.031645 |
| KRC4 | 51.087724 | 49.954032 | 0.031378 |

| Repeat | Arm | Requests | Calls | Request s | Native s | Cache s |
|---|---|---|---|---|---|---|
| 0 | O4 | 64 | 2994 | 51.93922242708504 | 50.73367438651621 | 0.02863982506096363 |
| 0 | K4 | 64 | 2942 | 50.883026998490095 | 49.79804778471589 | 0.029491189867258072 |
| 0 | KRC4 | 64 | 2950 | 51.043193869292736 | 49.92043260484934 | 0.029074950143694878 |
| 1 | O4 | 64 | 2994 | 51.909881027415395 | 50.696115566417575 | 0.03268829360604286 |
| 1 | K4 | 64 | 2942 | 50.93310144171119 | 49.84254748001695 | 0.032854270190000534 |
| 1 | KRC4 | 64 | 2950 | 51.116484047845006 | 49.97461906634271 | 0.03266379237174988 |
| 2 | O4 | 64 | 2994 | 51.89966591447592 | 50.68767518736422 | 0.03256319463253021 |
| 2 | K4 | 64 | 2942 | 50.93250500969589 | 49.84332773089409 | 0.032590351998806 |
| 2 | KRC4 | 64 | 2950 | 51.103493232280016 | 49.967045433819294 | 0.03239448554813862 |

Valid / invalid / unknown / unattempted executions: 576 / 0 / 0 / 0.
Publication/switching 954.559 s and native weight checks 11.152 s are outside request latency.

## additional

Valid common requests: 64/64.
KRC4 improves on O4 (+1.8850%). Earlier O counts are absent, so persistence against O cannot be assessed. Against K, the advantage reverses: event 2 +0.6378% → event 4 -0.4251% (-1.0629 percentage points). KRC4 versus KRC2 alone changes calls by 1.4442% reduction; that within-arm change does not establish relative benefit.

| Comparison | Baseline / method calls | Call reduction | Request-time reduction | Wins / losses / ties | Saved / added |
|---|---|---|---|---|---|
| KRC4 vs O4 | 3130 / 3071 | 1.8850% | 1.8135% | 36 / 20 / 8 | 100 / 41 |
| K4 vs O4 | 3130 / 3058 | 2.3003% | 2.3625% | 33 / 18 / 13 | 100 / 28 |
| KRC4 vs K4 | 3058 / 3071 | -0.4251% | -0.5623% | 22 / 26 / 16 | 37 / 50 |

| Event | Measurement | O calls | K calls | KRC calls |
|---|---|---|---|---|
| 1 | historical | absent | 3150 | 3125 |
| 2 | historical | absent | 3136 | 3116 |
| 4 | fresh | 3130 | 3058 | 3071 |

| Arm | Complete request s | Native generation s | Cache maintenance s |
|---|---|---|---|
| O4 | 53.902598 | 52.971794 | 0.030819 |
| K4 | 52.629160 | 51.779177 | 0.030790 |
| KRC4 | 52.925095 | 51.996986 | 0.030913 |

| Repeat | Arm | Requests | Calls | Request s | Native s | Cache s |
|---|---|---|---|---|---|---|
| 0 | O4 | 64 | 3130 | 53.86613707430661 | 52.94277428276837 | 0.027227947488427162 |
| 0 | K4 | 64 | 3058 | 52.58907931484282 | 51.74776675365865 | 0.02708185836672783 |
| 0 | KRC4 | 64 | 3071 | 52.88470652140677 | 51.96307056583464 | 0.02781154029071331 |
| 1 | O4 | 64 | 3130 | 53.91890982724726 | 52.98308398202062 | 0.032691119238734245 |
| 1 | K4 | 64 | 3058 | 52.65389481373131 | 51.79983270727098 | 0.03263415768742561 |
| 1 | KRC4 | 64 | 3071 | 52.94824920594692 | 52.01482719741762 | 0.03247163072228432 |
| 2 | O4 | 64 | 3130 | 53.922746904194355 | 52.98952235840261 | 0.03253927640616894 |
| 2 | K4 | 64 | 3058 | 52.644507298246026 | 51.78993194550276 | 0.03265530243515968 |
| 2 | KRC4 | 64 | 3071 | 52.94232994504273 | 52.01306027546525 | 0.032456034794449806 |

Valid / invalid / unknown / unattempted executions: 576 / 0 / 0 / 0.
Publication/switching 963.385 s and native weight checks 11.342 s are outside request latency.

Counts use one validated trajectory per request; reductions are 100 × (baseline total − method total) / baseline total. Three timing repeats are not independent training runs. Request totals sum each request’s repeat mean. Native-generation and cache-maintenance timers are nested inside request time. Historical counts support the event-1/2/4 curves; historical and fresh latency are never used as a learning curve.

## Training and retained states

| Event | Adam / presentations | Ordinary forwards / backwards | Reference capture / replay / backwards | BF16 nonzero | Branches |
|---|---|---|---|---|---|
| 3 | 128 / 256 | 256 / 256 | 256 / 256 / 224 | 127 | {"applied": 127, "zero_violation": 1} |
| 4 | 128 / 256 | 256 / 256 | 256 / 256 / 220 | 125 | {"applied": 125, "zero_violation": 3} |

| Event | Wall s | Inclusive train s | Reference/correction s | Integrity s | K + other residual s | Save s | Export s | Retention s | Peak allocated / reserved GB |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 693.529 | 653.578 | 296.045 | 6.704 | 350.829 | 21.101 | 15.296 | 0.000 | 38.231 / 40.624 |
| 4 | 681.568 | 642.801 | 288.806 | 6.665 | 347.330 | 20.284 | 15.473 | 0.000 | 39.855 / 42.488 |

KRC3 state: `/data/sources/minhao/runs/dflash_krc_events_3_4_20261028/campaign_001/KRC/events/event_003/training_state.pt`; serving export: `/data/sources/minhao/runs/dflash_krc_events_3_4_20261028/campaign_001/KRC/events/event_003/export`.
Audited reference presentations / anchors / available / protected occurrences: 256 / 66818 / 975476 / 516445.
Audited ordinary anchors / available / selected position occurrences: 66818 / 975476 / 566927.0. These overlapping occurrences are not independent examples.

KRC4 state: `/data/sources/minhao/runs/dflash_krc_events_3_4_20261028/campaign_001/KRC/events/event_004/training_state.pt`; serving export: `/data/sources/minhao/runs/dflash_krc_events_3_4_20261028/campaign_001/KRC/events/event_004/export`.
Audited reference presentations / anchors / available / protected occurrences: 256 / 65996 / 963417 / 504525.
Audited ordinary anchors / available / selected position occurrences: 65996 / 963417 / 554534.0. These overlapping occurrences are not independent examples.

Inclusive training time sums the presentation timers, including ordinary K, reference/correction and integrity hooks. Response-tensor loading occurs outside those timers and remains in event wall time. The residual subtracts measured reference/correction and integrity timers; it includes ordinary K and unseparated overhead, not an independently measured pure-K time. Save/export/retention are nested in event wall time. Saved-state validation and artifact hashing follow the event timer and are separately retained in TRAINING.csv and campaign costs. The complete states retain the 7,424-step horizon; task orchestration stops at 512. Reference exposure adds computation, so training is not compute matched.
Disposable startup: `{"backwards": 2, "exact_initial_state_restored": true, "loading_s": 23.35354275070131, "optimizer_steps": 0, "peak_cuda_allocated_bytes": 16491343360, "peak_cuda_reserved_bytes": 18845007872, "response_evaluations": 2, "runtime_closed_before_K_construction": true, "separate_disposable_runtime": true, "status": "passed", "wall_s": 42.81947931088507, "work_counts": {"draft_forwards": 2, "head_forwards_including_recompute": 392, "layer_forwards_including_recompute": 10, "response_model_forwards": 2, "startup_backward_attempted": 2, "startup_backward_completed": 2, "startup_response_evaluation_attempted": 2, "startup_response_evaluation_completed": 2}}`.
Backend loading: 22.007 s; campaign wall: 4428.999 s.

## Validity and provenance

KRC2 live restoration: `True`; final state retention: `passed`; storage admission: `True`.
Correctness requires exact canonical output tokens/termination and repeat-consistent counts. Development additionally reproduces retained O4/K4 per-request counts; additional-panel O4/K4 counts are new. The first eight development requests supply the included 72-execution native gate within the 1,152-generation budget.
Execution commit: `11b7718a04416a89025a6df69c0c33e558e5fe1b`. Postprocessing commit: `ea4f289c6e18383b7560c8c4103a9d91ca3943bf`; script SHA-256 `3c8e8462a15b7490bc074732a69b13216816d156977f5116d7c62b5e3d20a454`.
Execution source: `/data/sources/minhao/DFlash-Online`; dirty files: `[]`.
Audit issues: `[]`. Source/retrieval verification: `{"local_files_rehashed": 1053, "remote_only_files": 606, "retrieved_files_rehashed": 3845, "sealed_files": 1659}`.
Exact inputs, model/config/runtime pins and commands are retained in sealed plan (retained private evidence; not published), preflight (retained private evidence; not published), postflight (retained private evidence; not published), [launch record](LAUNCH.json), [launch script](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/11b7718a04416a89025a6df69c0c33e558e5fe1b/DFlash-Online/provenance/run_tools/dflash_krc_events_3_4_20261028/run_campaign.sh), independent audit (full record retained privately; [compact status and integrity counts](SUMMARY.json)), and complete costs (full record retained privately; [compact costs](SUMMARY.json) and [event cost table](TRAINING.csv)). Machine-readable root SUMMARY.json/TRAINING.csv and analysis panel files preserve all figures, paths, request rows and repeats.

KRC4 versus K4 compares their complete correction histories; it does not isolate corrections in events 3 and 4 because the event-2 states already differ. This is one shared-feedback trajectory and fixed-checkpoint serving evidence, not seed robustness or training-inclusive asynchronous deployment speedup. No tuning, new arm, event-3 serving evaluation or continuation beyond event 4 follows. Stage 1 / project None remains unchanged.

## Public evidence availability

This public copy preserves the report’s result values. Only the recorded evidence links and this availability note differ from the retained original. Compact metrics and execution identities are supplied alongside the report; the existing launch and review records remain in this report directory. Raw plans, prompts, responses, token trajectories, complete audit and runtime records, replay features, model weights and optimizer states are outside this public package. Absolute paths, usernames and host metadata are operational provenance, not public download locations. Original and public file hashes, missing compact files and the exact text transformations are recorded in [publication manifest](PUBLICATION_MANIFEST.json). The execution and analysis revisions retain their distinct roles.
