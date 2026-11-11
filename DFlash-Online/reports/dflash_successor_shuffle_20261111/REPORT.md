# K+ successor-shuffle control: completed event-five result

## Material Passport

ARS-Codex 3.24.2 experiment-agent, run → validate. Status: **complete and ANALYZED**. Actual execution commit: [`4d977e9ffd2cab9d58ae8806c29df11d843353da`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/4d977e9ffd2cab9d58ae8806c29df11d843353da), [PR #39](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/39). Independent checks recompute retained records; they are not additional GPU reproductions. The later report publication does not change the execution revision.

**TRUE K+ used 69 fewer verification calls than SHUFFLED: a 0.5953% reduction, with a paired 95% interval of [0.1710%, 1.0154%].** This supports informative placement of the successor-based priority for this training fork and fixed permutation. The incremental benefit over ordinary K remains unresolved: TRUE saved 42 calls, but its interval crosses zero. No complete-request latency benefit is established.

## Controlled fork and panel

All arms start from the same complete original PR #32 KPLUS4 state at step 512, including native BF16 AdamW moments, counters and RNG. K performs ordinary prefix-plus-first-error KL without redistribution. TRUE uses native K+ priority assignment. SHUFFLED permutes those assignments within each response and first-error proposal position while retaining the native reduction. The permutation uses seed 20261111 independently of training RNG; fixed points remain allowed.

PR #37's TRUE export at step 640 was authenticated and reused. New K and SHUFFLED arms each completed one event: 128 AdamW steps, 256 response presentations and two passes over the same 128 saved responses and teacher packets. Their natural presentation order and anchor draws match the retained schedule. No feedback was regenerated. The inherited seed-17 trajectory and 7,424-step horizon remain intact. Training retains BF16 KL/AdamW, constant LR 3e-4, weight decay 0.01, clip 1, microbatch 1, accumulation 2, up to 512 anchors, decay gamma 7, Flex attention and eight-block vocabulary chunks.

The 256 test requests were selected by the prescribed input hash from the 295 remaining PR #38-eligible rows, excluding its 768 measured requests and inherited feedback/development/probe exclusions. All 256 have documented older-project exposure; this is not a globally unused panel. Inputs, order and checkpoints were sealed before new outcomes. We reused 221 canonical references and prepared 35 separately through TRUE with the pinned native target verifier. These are verifier-backed references, not a new target-only implementation.

Fresh serving uses Qwen3-8B/DFlash on one physical A800, BF16/FA3, greedy block 16, context 4,096, output cap 2,048 or EOS, concurrency 1 and the qualified cache flushing. Training-data capture is disabled; compact verification traces remain enabled equally for all arms. Two counterbalanced sweeps reverse arm order within eight-request blocks. Training ended before serving, and transfers were absent during timing.

## Primary and control comparisons

Positive reductions favor the first named arm. Calls count one validated trajectory per request; timing averages the two repeats within each request before pooling. Intervals jointly resample whole requests across all arms, with 10,000 PCG64 draws, seed 20261111 and unadjusted 95% linear percentile intervals.

| Comparison | Calls, candidate / reference | Gross saved / added | Requests improved / regressed / tied | Call reduction, % [95% CI] | Request-time reduction, % [95% CI] |
|---|---:|---:|---:|---:|---:|
| **TRUE / SHUFFLED — primary** | 11,522 / 11,591 | 189 / 120 | 111 / 78 / 67 | **+0.5953 [0.1710, 1.0154]** | +1.2450 [−0.1404, 2.5626] |
| TRUE / K | 11,522 / 11,564 | 166 / 124 | 96 / 75 / 85 | +0.3632 [−0.0719, 0.7845] | +0.3578 [−0.9794, 1.6533] |
| SHUFFLED / K | 11,591 / 11,564 | 134 / 161 | 82 / 105 / 69 | −0.2335 [−0.6535, 0.1954] | −0.8984 [−2.2748, 0.4515] |

| Measure | K | TRUE | SHUFFLED |
|---|---:|---:|---:|
| Output tokens | 79,078 | 79,078 | 79,078 |
| Accepted draft tokens | 67,495 | 67,534 | 67,474 |
| Committed output progress | 78,822 | 78,822 | 78,822 |
| Calls / 1,000 output tokens | 146.2354 | 145.7042 | 146.5768 |
| Accepted draft tokens / call | 5.8366 | 5.8613 | 5.8212 |
| Committed progress / call | 6.8162 | 6.8410 | 6.8003 |
| Complete-request seconds, summed repeat means | 208.6891 | 207.9423 | 210.5639 |
| Native-generation seconds, same aggregation | 194.8017 | 194.7615 | 195.7818 |
| Cache-maintenance seconds, same aggregation | 7.5988 | 6.6598 | 8.2541 |

Committed progress excludes the target-generated first token. Accepted draft tokens include accepted proposals subject to terminal clipping; the separate native `progress − calls` proxy is retained in the machine-readable summary and is not substituted for that count. All 256 requests ended at EOS; none hit the output cap.

| Sweep | K request seconds | TRUE request seconds | SHUFFLED request seconds | TRUE / SHUFFLED reduction |
|---|---:|---:|---:|---:|
| 1 | 209.6575 | 207.7015 | 211.8321 | +1.9499% |
| 2 | 207.7207 | 208.1832 | 209.2958 | +0.5316% |

Both sweeps favor TRUE over SHUFFLED in complete-request time, but the paired interval includes zero. Native-generation reduction is also unresolved: +0.5212% [−0.1465%, 1.1983%]. Cache maintenance contributes to the complete-time point difference. These aggregate clocks do not identify kernel-level causes. TRUE/K timing reverses direction between sweeps, and no fresh training-inclusive speedup is claimed.

## The shuffled control was active

Every one of the 256 SHUFFLED presentations passed the same-state selection, denominator, class-count and native FP32 mass/coefficient checks. There were 45,255 eligible first errors, 2,286 ineligible errors and 3,396 mixed strata. The permutation changed **18,452 bits**, versus the descriptive uniform-permutation benchmark of 18,416.4065, and reassigned 1,827.4145 units of first-error weight. TRUE and SHUFFLED have different final weight hashes.

These invariants compare SHUFFLED's assignment with unpermuted labels at its own current state. They do not require identical class totals across diverged trajectories. Both new arms saw 918,693 available supervised-position occurrences; selected exposure was 539,081 for K and 539,711 for SHUFFLED. Changed predictions can change the selected prefix despite identical sampled anchors. Full per-presentation records remain private, with aggregate checks in [MANIPULATION_SUMMARY.json](MANIPULATION_SUMMARY.json).

## Matched-start diagnostic

The 64 input-hash-selected starts contain 10 parent successor-high errors, 53 successor-low errors and one all-correct parent. Successor-ineligible and seed-EOS groups both have zero cases. There are 63 old errors and 64 nonempty prefixes eligible for preservation analysis.

| Child | Old first errors repaired / 63 | Earlier prefixes broken / 64 | Prefix tokens gained / lost | Net prefix gain | Tied starts |
|---|---:|---:|---:|---:|---:|
| K | 11 | 6 | 62 / 7 | +55 | 47 |
| TRUE | 13 | 6 | 62 / 7 | +55 | 45 |
| SHUFFLED | 12 | 6 | 62 / 7 | +55 | 46 |

Canonical-prefix and native accepted-prefix checks agree throughout, so every counted repair is also natively verified. No acceptance advance is mislabeled as repair. In the high group, TRUE repairs 5/10 versus 6/10 for each control; in the low group, it repairs 8/53 versus 5/53 for K and 6/53 for SHUFFLED. The diagnostic therefore does **not** establish that preferential repair of the parent high group explains the endpoint result. All arms have the same net gain on these starts. Later verification rounds contribute to full-request calls; local frequencies cannot be converted into a share of time saved.

## Work, costs and validity

All 1,536 timed executions are valid, with zero invalid, unknown, extra or unattempted executions. Outputs and termination match exactly, repeat counts agree, and pre/postflight source and checkpoint checks pass. Separate reference preparation completed 35 generations; the parent diagnostic completed 64 one-block evaluations. Neither supplied learning feedback.

| Cost scope | K | SHUFFLED |
|---|---:|---:|
| Native training seconds | 328.0647 | 400.9850 |
| Checkpoint persistence seconds | 20.4195 | 20.7444 |
| Export seconds | 14.3261 | 14.1468 |
| Initialization seconds | 36.8810 | 36.9930 |
| Disposable startup seconds | 19.7433 | 18.5149 |
| Training worker wall, including admission checks | 480.4780 | 554.5859 |
| Peak allocated / reserved GPU GiB during event | 19.1235 / 24.5566 | 19.1236 / 24.5566 |

SHUFFLED's observed native-training difference versus K is +72.9202 seconds (+22.2274%). Its 7,994 extra student-head projections took 50.3727 seconds and assignment took 8.1397 seconds, both nested inside training. The arm difference also includes K+ reduction, logging and different trajectories; it is not a pure assignment-overhead estimate or the intrinsic cost of native TRUE K+. TRUE was reused and has no fresh training clock.

The main event used exactly 256 new AdamW steps and 512 response forwards/backwards. Qualification separately used three disposable steps and six ordinary forwards/backwards, plus six startup forwards/backwards. Main startup added four forwards/backwards and zero steps. Qualification added 94 extra detached head projections; no qualification state became a training checkpoint. Projection chunks and checkpoint recomputation are retained separately in the cost records.

Qualification wrappers took 432.1474 seconds. Reference preparation took 66.6238 seconds, including loading and 28.1346 seconds of generation wall. Parent probes took 26.4382 seconds including loading. The serving-stage wall was 1,669.7175 seconds, including 20.7569 seconds loading and 302.1289 seconds for 161 checkpoint publications. These are nested scopes and must not be added again. Total runtime was **3,330.4673 seconds (55.51 minutes)**; wrapper and report completion spanned 10:25:41–11:21:22 UTC, or 55 minutes 41 seconds. Implementation, transfer and independent analysis are outside that runtime.

Storage admission reserved 103.199 GB against 714.132 GB free; 665.540 GB remained after execution. No artifacts were retired. All 5,013 compact evidence files (454,125,542 bytes) were hash-verified on transfer; model states and tensors remain on A800.

## Independent checks and conclusion

The independent scalar audit exactly reproduces all 129 checked summary fields, all 12 intervals, sweep totals, 52,224 raw/CSV fields and 161 publications. A separate raw audit checks all 1,536 output/termination pairs, native sidecars and **69,354 verification blocks**. The matched-start audit reproduces all 2,796 reported fields. Native qualification passes exact identity parity and the real-permutation checks; 63 new CPU tests pass locally and on A800, with 50 inherited K+ regression tests also passing locally.

Reviewers are members of the implementation team, with authorship overlap disclosed. The scalar and direct-loop checks use separately written arithmetic; they are not blinded external reviews or new GPU executions. One matched-start audit reader initially rejected valid extra subset annotations; its schema handling was corrected with the original failed attempt preserved. This changed no production code, measurement, input or scientific setting. All 11 ARS interpretation checks are recorded.

**The prespecified primary comparison supports informative priority placement for this fork and one permutation.** It does not establish an incremental call benefit over ordinary K, a latency improvement, first-block mediation, training-seed robustness or a general advantage across tasks. Unresolved contrasts do not establish equivalence. This bounded event-five experiment is complete; no tuning, additional event or panel was run.

Companion evidence: [paired requests](PAIRED_REQUESTS.csv), [every repeat](TIMING_REPEATS.csv), [sweeps](SWEEPS.csv), [sealed IDs/hashes](PANEL_MANIFEST.json), [summary](SUMMARY.json), [cost scopes](COST_ACCOUNTING.json), [qualification](IMPLEMENTATION_AND_QUALIFICATION.md), [raw audit](audit/RAW_VALIDITY_001.json), [scalar audit](audit/SUCCESSOR_AUDIT_001.json), [matched-start audit](audit/MATCHED_START_AUDIT.json), and [compact evidence inventory](COMPACT_DELIVERY_MANIFEST.json). Full rendered inputs, outputs, traces and per-presentation data remain outside Git in the retained execution evidence.
