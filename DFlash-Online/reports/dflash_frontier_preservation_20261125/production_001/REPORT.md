# Frontier preservation: completed one-event comparison

**Status: numerical and native integration qualification passed; the full event completed; serving efficacy is unresolved, with no demonstrated benefit.** FRONTIER_P saved only 2 of CONTROL's 7,639 verification calls. Complete-request latency was essentially unchanged, while training took 4.32 times as long. This is a valid completed negative development screen, not an execution failure or evidence of equivalence.

## Material Passport

- Origin Skill / Mode: experiment-agent / run + validate
- Origin Date: 2026-11-25
- Verification Status: ANALYZED — authenticated native execution and independent CPU recomputation; no second training or serving run
- Version Label: frontier_preservation_v1 / native_token_contraction_v2 / production_001
- Actual production execution: `225802657d630c0985abd68d6e7453f3e47c5b74`

## Serving result

The fixed panel contains 128 reused OpenCode development requests. All 768 scored executions and 24 declared warmups passed exact retained-F token/termination validation; all three repeats have consistent per-request counts. There are zero invalid, unknown, or unattempted executions. All scored responses terminated by EOS, with identical 50,226 output tokens per arm. Length range is 144–1,314 tokens; mean 392.390625.

| Measurement | CONTROL | FRONTIER_P | FRONTIER_P reduction versus CONTROL, 95% CI |
|---|---:|---:|---:|
| Verification calls, one trajectory/request | 7,639 | 7,637 | +0.0262% [−0.4392%, +0.4625%] |
| Sum of per-request mean complete latency | 135.9984 s | 136.0213 s | **−0.0168% [−0.4871%, +0.4100%]** |
| Sum of per-request mean native generation time | 131.9757 s | 131.9972 s | −0.0163% [−0.4699%, +0.4049%] |
| Sum of per-request mean cache-maintenance time | 0.05752 s | 0.05651 s | Descriptive; already included in request timing |
| Coupled accepted draft tokens / call | 5.55819 | 5.55991 | Counts are coupled to committed progress and calls |
| Committed progress / call | 6.55819 | 6.55991 | Same 50,098 committed tokens per arm |
| Stop-clipped accepted proposal count | 42,580 | 42,583 | Separate counter; not interchangeable with coupled acceptance |

FRONTIER_P wins on calls for **50 requests**, regresses on **41**, and ties on **37**. Gross savings of **73 calls** are nearly canceled by **71 added calls**. The primary outcome is complete-request time, not a favorable secondary metric. No positive serving conclusion follows from these intervals; absence of a demonstrated difference is not proof of equivalence.

Complete-request totals for repeats 1/2/3 are CONTROL **136.00099 / 136.03724 / 135.95696 s**, and FRONTIER_P **135.95300 / 136.24043 / 135.87039 s**. Native-generation and cache totals are retained in [SERVING_SUMMARY.json](SERVING_SUMMARY.json). [PER_REQUEST.csv](PER_REQUEST.csv) preserves all 128 paired rows; [ALL_REPEATS.csv](ALL_REPEATS.csv) preserves all 792 executions, identifying warmups separately.

The estimate is `100 × (1 − sum(candidate request means) / sum(control request means))`. Confidence intervals use 10,000 paired whole-request PCG64 resamples, seed 20261125, and linear 2.5/97.5 percentiles. Counts use one validated trajectory after checking repeats; timing repeats are not extra independent requests. Secondary intervals are descriptive and unadjusted. No minimum gain threshold was imposed.

## What was held fixed

Both arms independently restored the same authenticated PR59 **PAIR1 warm state**, including corrected weights, 58 warm Adam entries, counters, RNG, configuration and event-one history. They started at step128/presentations256/event1 and finished at **step256/presentations512/event2**. Neither optimizer nor seed was reset. Each received 256 new response presentations and 128 native AdamW updates.

CONTROL is **native C + K+PAIR**, with preservation alpha0.5/cap0.25. FRONTIER_P replaces only C's preservation allocation; K+ first-error repair coefficients and PAIR remain enabled. This comparison is not FRONTIER_P versus plain OSD or plain K+.

One common-parent collection supplied all128 feedback responses, using original PR56 `train_requests[128:256]` and source indices128–255. One immutable prepared feature pool was shared. Both arms used identical presentation orders and sampled anchors. Current roles are refreshed per actual two-response window; scoring uses frozen roles and isolated RNG, while the committed update replays cached anchors and ends at the ordinary post-sampling RNG state. Final coefficients are detached, and only current prefix/first-error positions receive direct teaching loss. Assessment and continuation positions do not add a committed auxiliary loss.

The target is pinned Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; released DFlash origin is `z-lab/Qwen3-8B-DFlash-b16` revision `9b41424b7109f9c5413454f481b09a82b85333f4`. Native BF16 KL/head, five trainable draft layers, Flex training attention, constant LR3e-4, weight decay0.01, clipping1, gamma7, K+eta0.5, microbatch1/accumulation2, up to512 anchors and eight-block head chunks remain unchanged. The inherited constant-LR horizon remains7424 steps; only this continuation event ran.

Serving retained SGLang/FA3, BF16 greedy block16, 2,048 tokens or EOS, logical context4,096/physical4,112, concurrency1, capture disabled, full cache flush and fused-helper refresh. Feedback/evaluation ran on physical GPU A; serial training ran on B. Training processes closed before evaluation; B stayed quiet and no transfers occurred during timing. No extra arm, event, tuning, or earlier-endpoint selection occurred.

## Qualification and the numerical interpretation

The [separate qualification report](../allocation_qualification_001/REPORT.md) retains the prospective contract, all58 serial reference contractions, four complete predetermined pools, coefficients, two fast repeats and independent audits. Reference scores contract the actual native unweighted token-FKL gradient with represented FP32 shadow direction `w` in FP64, then divide by `2 Z_response`. The fast route casts `w` to BF16 without rescaling; secondary effective-direction contractions disclose that rounding separately.

All allocation invariants, coefficient-fidelity and local-alignment gates passed. Maximum coefficient error was **1.2592% of beta**, below the prospective2.5% allowance plus roundoff; minimum reference local alignment was **99.8463%**, above90%. Fast allocations repeated exactly. Four disposable native updates verified CONTROL/disabled equality, beta-reference equality, exact state restoration, and a BF16 endpoint change attributable to allocation. Production contains no serial token-gradient oracle and no extra Adam update for scoring.

**The historical hidden-state scalar check remains FAILED:** 2.047197% exceeded its unchanged1.6% criterion. The user replaced its production-admission role with the declared allocation-level contract; it was not relabeled a pass. Earlier failure records and all attempts remain intact. There were no new failed production attempts or retries.

The continuous FP32 optimizer shadow is a sensitivity surrogate, not an exact derivative or prediction of discrete BF16 training. The separate same-coefficient integration diagnostic measured a shadow/native displacement difference of0.893947 relative to shadow norm, with1,420,195 differing coordinates after casting the shadow. This limitation remains unresolved; the present result does not isolate it as the cause of absent serving benefit.

## Fixed training witnesses

Two diagnostic sets were sealed at the common parent: eight new-buffer responses and eight disjoint parent-event training responses, each with eight native anchors (64 overlapping block occurrences per set). They are training/mechanism evidence, not held-out transfer samples. Origins, valid positions, P/E, L/e/H/K and labels remain fixed; endpoints recompute strongest wrong rivals.

| Witness measurement | New buffer: CONTROL → FRONTIER_P | Parent training: CONTROL → FRONTIER_P |
|---|---:|---:|
| Original first errors / old prefix positions / usable followers | 44 / 488 / 60 | 31 / 676 / 33 |
| Raw first-error repairs | 33 → 36 | 10 → 11 |
| Useful repairs: error fixed and preceding matches survive | 33 → 35 | 9 → 10 |
| Damaged old-prefix positions | 3 → 3 | 14 → 14 |
| Blocks with old-prefix damage | 2 → 2 | 10 → 7 |
| Surviving usable followers | 58 → 59 | 30 → 29 |
| Reachable followers in the new consecutive prefix | 39 → 52 | 12 → 13 |
| Total matching-prefix length | 656 → 676 | 675 → 690 |
| Fixed-K shortfall | 35 → 21 | 75 → 59 |
| Response-mean assessment, lower is better | 3.51655 → 3.14953 | 6.58770 → 6.91978 |

New-buffer useful-repair comparisons contain3 gains/1loss/60ties; matching-prefix comparisons contain5gains/3losses/56ties. Parent-training useful repairs contain1gain/0losses/63ties; matching-prefix comparisons contain3gains/1loss/60ties. These modest local differences did not produce demonstrated serving benefit. Parent-training assessment worsened despite somewhat longer matching prefixes, so the witness evidence is mixed.

For new-buffer assessed positions (592), current strongest-wrong mean margin is6.49008→6.61951; margin ties4→1, wrong-rival ties24→25, and origin-rival changes216→222. For parent-training assessed positions (740), margins7.05752→7.10591; margin ties4→4, wrong-rival ties25→29, rival changes175→176. Full origins, all three pairwise comparisons, denominator definitions and raw/useful-repair distinction are summarized in [WITNESS_SUMMARY.json](WITNESS_SUMMARY.json). A raw repair failing old-prefix survival is not called useful.

## Training and total work

| Measured scope | CONTROL | FRONTIER_P |
|---|---:|---:|
| Training timer, including candidate scoring | 564.545 s (9.41 min) | 2,439.323 s (40.66 min) |
| Recovery persistence | 18.111 s | 19.723 s |
| Serving export | 14.609 s | 15.197 s |
| Event wall time before final commit | 600.821 s | 2,482.177 s |
| Discarded startup, zero Adam updates | 25.969 s | 18.673 s |
| Complete training subprocess occupancy | 700.047 s | 2,588.085 s |

FRONTIER_P training costs **4.32086× CONTROL (+332.09%, +1,874.778 s /31.25min)**. Its allocation-score timers total1,860.161s; complete candidate-window timers total2,443.155s. The latter includes per-window bookkeeping, so it is not identical to the per-response training timer. These scopes overlap and must not be summed. No detailed kernel-time breakdown is inferred.

Committed teaching across both arms is512 response forwards/backwards and256 Adam updates. Candidate scoring adds1,024 backbone forwards (including256 directional evaluations),512 first-order backwards,128 FP32 optimizer-shadow evaluations and128 optimizer VJPs, but zero additional Adam steps. Training totals are therefore **1,536 backbone forwards /1,024 first-order backwards /256 Adam steps**, before startup and witnesses. Eight-block native projection layout totals11,790 chunks per arm's response schedule; combined module hooks record200,430 vocabulary-head calls and70,740 objective-chunk calls, including recomputation. These are module-call counts, not kernels or extra full-model passes.

Startup adds4 forwards/4backwards/0Adam; fixed witness reads add48 forwards and48head calls/0backwards/0Adam. Common collection adds128 generations and128 preparations; serving adds768scores+24warmups. Qualified numerical/integration work remains separate:58 token-oracle backwards and4 disposable Adam updates. All pending-work ledgers close normally. [COSTS_SUMMARY.json](COSTS_SUMMARY.json), [TIMING_BREAKDOWN.json](TIMING_BREAKDOWN.json) and [ALLOCATION_WINDOWS.csv](ALLOCATION_WINDOWS.csv) retain the measured scopes and128 allocation-window records.

Shared collection serve calls cost220.382s, comprising161.347s generation/validation/recording and59.034s buffer persistence; native request time160.449s is nested within those calls. Teacher preparation cost8.105s and prepared-feature persistence10.301s. Shared serving-worker load was19.981s. Evaluation backend loading cost22.009s (worker initialization21.849s nested), and12 checkpoint publications cost12.444s (native activation4.383s nested). These are outside the timed serving comparison. [TIMING_BREAKDOWN.json](TIMING_BREAKDOWN.json) gives the full scopes.

[ALLOCATION_SPARSE_DIAGNOSTICS.json](ALLOCATION_SPARSE_DIAGNOSTICS.json) gives updates1/64/128, including current scoring branches, mass versus beta/old C, contrast, predicted alignment, preview assessment and final native gradient norms. All three final updates were unclipped. The full signed per-token contrast range was not retained; the reported contrast range is the per-block maximum absolute centered score.

## Resource, integrity and reporting limits

Production source and all5,205 staged files were checked before execution and rehashed afterward. Parent recovery/export, complete PR56 input manifest and both final recoveries/exports still match their identities. [IDENTITIES.json](IDENTITIES.json) records actual execution commits, endpoint counters/hashes and the private archive binding. Both GPUs were idle at postflight.

The new phase completed in **6,612.312s (1h50m12s)**, inside4h. Production occupancy was4,624.346cardseconds. Including new numerical/integration work and the original267.424cardseconds, cumulative occupancy was **5,175.047cardseconds (1.43751card-hours)**, below8card-hours. The earlier elapsed interval3,465.673s and intervening review/CPU-preparation gap4,095.307s remain separate; neither was silently erased or charged as GPU work. Offline retrieval/audit/publication follows model-work completion.

A800 postflight free space was **407.983GB**, above400GB. Charged A800 study storage including compiler growth was24.113GB; current local delivery copies bring combined usage to about25GB, below40GB. No inherited artifact was deleted; all parent states, old failures, feedback and witnesses remain retained. The only removed objects were verified redundant new captured-feature temporaries handled by the existing preparation path; their exact path-level preparation/retirement receipts remain private, with counts and bytes in [RETENTION.json](RETENTION.json). [RESOURCES.json](RESOURCES.json) records scopes and limits.

The integration fixture's measured peak was37.658GB allocated/48.566GB reserved after startup. Production monitoring observed a maximum61,187MiB (about59.75GiB) on GPU B. This is a sampled device-memory observation, not an exact whole-run allocator peak. Exact production training allocator peak and separate trainer-loading duration were not instrumented and are unavailable. Shared serving and evaluation allocator peaks remain in the cost records.

**Another reporting gap:** requested actual/shadow parameter-displacement norms at updates1/64/128 were not saved. Gradient norms, clipping, score scales/contrast, allocation movement, predicted alignment and preview assessment were retained. The disposable integration norm diagnostic is not substituted for missing production measurements; no extra update or rerun was added.

The private delivery contains1,134 hash-verified compact files, all raw evaluation records and canonical references. Published tables omit prompts, source IDs, token content, anchors and tensors. The offline audit checks792 executions/47,640 native blocks and all saved serving/witness/cost summaries; the independent arithmetic audit and11/11 interpretation checks are in [INDEPENDENT_RESULT_AUDIT.json](INDEPENDENT_RESULT_AUDIT.json) and [INTERPRETATION_REVIEW.md](INTERPRETATION_REVIEW.md). CPU audit code and regression fixtures are in [audit_code](audit_code). Full witness logits were not saved, so the audit reconstructs metrics from retained native predictions/margins rather than independently re-projecting logits.

This is one warm continuation, one shared feedback buffer and one reused development panel. The intervals condition on these two endpoints; they do not establish training-seed robustness, fresh-panel confirmation, a universal failure of sensitivity allocation, or an equivalence bound. Lower training assessment or more local repairs alone cannot establish serving success. The fixed screen is complete; no additional event, tuning or new mechanism was run.

Execution used the archived [RUN.sh](RUN.sh); the coordinator assigned physical GPUs by phase from the sealed plan. Main implementation passed140 focused CPU tests; the offline audit/retrieval fixtures passed15 CPU tests. [AUDIT_SUMMARY.json](AUDIT_SUMMARY.json) distinguishes model execution, scalar validation and unrecorded quantities. Supplemental report extraction initially rejected a relative-versus-absolute receipt-path representation; that CPU-only extractor was corrected, the first error was retained, and no scientific computation or raw evidence changed.
