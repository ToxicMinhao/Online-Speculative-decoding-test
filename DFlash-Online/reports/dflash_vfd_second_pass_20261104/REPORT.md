# VFD second-pass selection: four-event result

## Material Passport

ARS experiment-agent run/validate. Complete and analyzed from retained native evidence; no independent GPU replication. Execution commit: [`05e4b30488fb267a196edb1a81a5f6aaf613a838`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/05e4b30488fb267a196edb1a81a5f6aaf613a838), [PR #27](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/27). Native K+/VFD scientific baseline remains `e0a9d3c235e1efd15404c549282d21c99243a21c`.

**Decision 3: do not advance this progress-based selector from this screen.** PROGRESS64 was slower than full two-pass VFD and did not establish an advantage over random or initial-opportunity selection. This conclusion concerns the tested second-pass rule, not VFD generally. No tuning, extension or additional run was performed.

## Fixed experiment

Five policies started from the authenticated event-zero weights, fresh optimizer and seed-17 RNG. Across four events, the preceding FULL checkpoint supplied one shared pool of 128 responses per event. All policies learned every response once; FULL revisited all 128, ONCE none, and RANDOM64/STATIC64/PROGRESS64 revisited 64 in FULL’s recorded order. Event 1’s first pass was physically shared; subsequent policies retained their own weights, Adam state, RNG and counters. The VFD objective, optimizer and hyperparameters were unchanged.

The final endpoint used 200 presealed train-disjoint GSM8K test requests, five checkpoints and two rotating/reversed timing sweeps. BF16 greedy Qwen3-8B/DFlash-b16 serving used block 16, FA3, a 4,096-token context limit and 2,048 new tokens or EOS, with capture disabled. Training and bulk transfers were absent during timing. The panel has four requests unmatched in available prior inventories and 196 with older project exposure; it is not described as globally fresh.

## Serving result

Times below sum the two-repeat mean for each of the 200 requests. Calls use one validated trajectory per request. Positive percentage reduction favors PROGRESS64.

| Policy | Request time, s | Verification calls | Accepted draft tokens/call | Logical updater time, s |
|---|---:|---:|---:|---:|
| FULL | 151.991 | 8,872 | 5.7557 | 1414.156 |
| ONCE | 154.399 | 9,008 | 5.6528 | 700.574 |
| RANDOM64 | 153.274 | 8,946 | 5.7003 | 1053.035 |
| STATIC64 | 152.656 | 8,916 | 5.7226 | 1087.226 |
| PROGRESS64 | 153.255 | 8,945 | 5.7010 | 1110.697 |

| PROGRESS64 versus | Request-time reduction | 95% paired interval | Calls saved / added | Requests with fewer / more / equal calls |
|---|---:|---:|---:|---:|
| FULL | -0.8313% | [-1.5384%, -0.1463%] | 145 / 218 | 70 / 101 / 29 |
| STATIC64 | -0.3921% | [-1.0425%, +0.2618%] | 148 / 177 | 72 / 87 / 41 |
| RANDOM64 | +0.0129% | [-0.6987%, +0.7438%] | 175 / 174 | 79 / 80 / 41 |
| ONCE | +0.7415% | [-0.0796%, +1.5574%] | 243 / 180 | 92 / 82 / 26 |

**Primary:** PROGRESS64 took 0.8313% more complete request time than FULL; its reduction interval was entirely negative. It also needed 73 additional verification calls (8,945 versus 8,872; 0.8228% more). Both timing sweeps show the same direction.

**Selective value:** versus RANDOM64, PROGRESS64 saved one verification call and only 0.0129% request time, with a wide interval spanning zero. Versus STATIC64, it added 29 calls and was 0.3921% slower on the point estimate; that interval also spans zero. These controls establish neither added scoring value nor equivalence.

**Cost trade-off:** PROGRESS64 saved 303.459 s of measured logical updater time versus FULL (21.4587%), but cost 57.662 s more than RANDOM64 (5.4758%). The efficiency-only criterion still fails: the primary lower confidence limit, −1.5384%, is below the prespecified −0.5% acceptable slowdown. Cheaper updating does not establish serving acceleration.

The fixed analysis averages repeats within request and uses 10,000 paired request-bootstrap draws (PCG64 seed 20261104). Repeats are not independent samples or training replications. Control intervals are descriptive and unadjusted. All 59,952 output tokens and the one capped request remain included. Accepted draft tokens/call and committed progress/call are distinct; both are retained in SUMMARY.json.

## Selection, exposure and cost evidence

Event-1 STATIC64/PROGRESS64 selected-set overlap was **37/64**. Later overlaps were 41, 38 and 39, but those selectors ranked under different learned states. All zero/tie rates, opportunities, repaired/persistent/broken counts and selected lengths remain in the score and selection tables. The field `selected_valid_blocks` in the original selector table counts the separate up-to-16-anchor scoring blocks; supplemental native-exposure tables separately report actual training blocks and supervised occurrences.

Logical standalone optimizer steps were FULL 512, ONCE 256 and each half-selection policy 384. Shared event-1 work is charged to every applicable policy for cost comparisons; the actual physical main work was **1,664 Adam steps, 3,328 response presentations and 1,408 scoring forwards**. Feedback work was exactly 512 generations and 512 teacher preparations. Qualification was separate: eight scoring forwards, four presentations/backwards and two discarded Adam steps, with no qualification generation.

Measured physical training-pass wall was 4,596.304 s (4,575.960 s native training compute); scoring was 78.569 s, selection 0.280 s, state persistence 730.564 s and export 291.427 s. Shared feedback collection wall was 407.169 s; preparation wall was 44.617 s, including 20.077 s teacher preparation and 23.642 s feature persistence. Capture-host subtime (306.365 s) is part of collection and must not be added again. Loading, activation, checkpoint validation and unpartitioned controller/initialization overhead remain separately scoped in COST_ACCOUNTING.json and worker receipts. These are not a kernel-time breakdown.

**Duration:** wrapper start 2026-11-04 19:02:15 UTC, end 21:41:29 UTC: **2 h 39 m 14 s**, within the initial 2.5–4 h estimate. Campaign wall was 9,436.984 s; subsequent local retrieval/review is outside that runtime. Qualification took 60.667 s. Peak training allocation was 20.745 GB; this excludes the separate target-serving GPU.

## Correctness and evidence

All 58 focused CPU tests and the bounded GPU qualification passed. All four own-state continuations, shared forks, selection identities, exact budgets, source/runtime pins, final checkpoint immutability, canonical outputs/termination and repeat-count checks passed. All 2,000 timed executions and 58 missing canonical generations completed; no invalid, unattempted or unknown work remains. The wrapper, scientific campaign, postflight and report all exited successfully.

A separate calculation reproduced the reported endpoint statistics with 646,094 checks across 4,058 source files and 89,374 native verification blocks. Same-team authorship overlap is disclosed; this is an artifact/calculation audit, not an independent GPU replication. Native-exposure and interpretation reviews accompany the report.

The compact archive contains 10,049 files / 613,289,482 bytes, verified against original file identities. Raw token/trace evidence is private; final model exports, complete recovery states and fixed initialization remain on A800. Retired intermediate feature/state capabilities are explicitly recorded in retention receipts, not described as archived. No source evidence was deleted during retrieval. Before launch, seven audited obsolete intermediate states were retired to free 44.052 GB; exact removed paths and lost historical resume capabilities are recorded under storage/. Current final checkpoints and recovery states remain retained.

## Interpretation and stopping point

This one-seed, four-event common-feedback screen argues against the specified progress rule at this budget. It does not identify why particular responses transferred, prove that all selective replay policies fail, establish seed robustness, or measure five independent online deployments. No further experiment or mechanism change follows automatically. Stage 1 / project None remains unchanged.

Machine-readable delivery: SUMMARY.json, CONFIGURATION_PIN.json, COST_ACCOUNTING.json, PAIRED_REQUESTS.csv, ALL_REPEATS.csv, TRAINING_PASSES.csv, SELECTIONS.csv, SCORES.csv, SCORE_BLOCKS.csv, SCORE_SUMMARY.csv, state/qualification/endpoint audits, and supplemental native-exposure tables.
