# Track2 final evidence interpretation

## Material Passport

Mode: run→validate. Verification status: **ANALYZED**. Execution: `8a4cbcc8b78adfee845a0a1409db4a0d29c54058`.

Source handoff: `TRACK2_K_REPLAY_EXECUTOR_INSTRUCTIONS_20261102.txt` (2026-11-02). This is a separate CPU reconstruction by the same team, with authorship overlap in the original selection, evaluation and reporting implementation. The auditor imports no experiment code. It is not an independent GPU reproduction.

## Finding

The retained evidence passes **2,917,883 checks with zero discrepancies**. All six audit sections passed on the first run; no production file was changed and no audit repair was needed. The measured result does not establish a useful overall improvement from this replay allocation. R saves 12 calls against U but uses 21 more than K; both pooled intervals span harm and benefit. R's coding advantage over U is accompanied by a math regression. Coverage rises at every replay event, without a demonstrated overall serving benefit.

The complete result supports parking this candidate in its current form. The evidence does not show the consistent across-domain/future-request benefit needed to motivate an unchanged seed replication under the handoff. Changing the replay dose or descriptor is not authorized by this review. This is an inconclusive/negative utility result from a valid run, not an implementation failure and not proof of equivalence.

## Reconstructed endpoint

Positive reduction means fewer calls or less time for the candidate. Counts use 256 unique requests once, after exact repeat equality; time uses each request/method's mean of three repeats. The independent 10,000-draw bootstrap uses PCG64 seed 20261102, jointly paired indices, 128 math and 128 coding samples per draw, and linear percentiles. Every reconstructed numerical field agrees with the saved endpoint using relative and absolute tolerances of 1e-12.

| Panel | Contrast | Call reduction % [95% interval] | Wins / regressions / ties | Gross saved / added | Time reduction % |
|---|---|---:|---:|---:|---:|
| pooled | R/U | 0.0920 [-0.4775, 0.6432] | 101 / 107 / 48 | 241 / 229 | -0.6691 |
| pooled | R/K | -0.1615 [-0.8027, 0.4487] | 101 / 112 / 43 | 241 / 262 | -0.6441 |
| pooled | U/K | -0.2538 [-0.9200, 0.3989] | 110 / 106 / 40 | 250 / 283 | 0.0249 |
| math | R/U | -1.3259 [-2.1867, -0.5011] | 38 / 66 / 24 | 73 / 147 | -2.0587 |
| math | R/K | -0.7842 [-1.7928, 0.1968] | 45 / 63 / 20 | 104 / 148 | -2.2101 |
| math | U/K | 0.5347 [-0.5361, 1.6081] | 62 / 45 / 21 | 146 / 116 | -0.1484 |
| coding | R/U | 1.1534 [0.4136, 1.8845] | 63 / 41 / 24 | 168 / 82 | 0.3937 |
| coding | R/K | 0.3111 [-0.5140, 1.0909] | 56 / 49 / 23 | 137 / 114 | 0.5501 |
| coding | U/K | -0.8522 [-1.6863, -0.0428] | 48 / 61 / 19 | 104 / 167 | 0.1570 |

These are conditional request-panel intervals for one seed and one fitted trajectory on a reused development panel. They do not include training-seed or adaptive-trajectory uncertainty. Secondary/domain/time intervals are unadjusted and descriptive; no p-values or multiplicity-adjusted confirmatory findings are asserted.

| Arm | Calls | Accepted draft tokens | Accepted / call | Committed progress / call | Output / call | Mean-repeat total request seconds |
|---|---:|---:|---:|---:|---:|---:|
| K | 13004 | 62888 | 4.836050 | 5.817441 | 5.837127 | 230.281434 |
| U | 13037 | 62844 | 4.820434 | 5.802715 | 5.822352 | 230.224155 |
| R | 13025 | 62869 | 4.826795 | 5.808061 | 5.827716 | 231.764660 |

Each arm commits 75,650 post-anchor tokens and returns the same 75,906 complete output tokens. Accepted draft tokens exclude target corrections; committed progress and full output are distinct quantities. Counts are identical in all three repeats.

| Repeat | K request seconds | U request seconds | R request seconds | R/U time reduction % | R/K time reduction % |
|---|---:|---:|---:|---:|---:|
| 0 | 229.607639 | 229.805746 | 231.559838 | -0.7633 | -0.8502 |
| 1 | 230.803345 | 230.475837 | 231.429655 | -0.4138 | -0.2714 |
| 2 | 230.433319 | 230.390883 | 232.304487 | -0.8306 | -0.8120 |

R is slower than U and K in each complete final repeat. Mean-time R/U is -0.6691% [-1.7590, 0.3440]%; R/K is -0.6441% [-1.7911, 0.4262]%. Neither establishes a serving speedup. Publication and loading are outside request timing; the 36 batch boundaries plus the final boundary all carry passing transfer/updater quiescence and resource receipts.

## Future-serving stream

All 3,072 pre-update records are valid. Events 1 and 2 have exact neutral arm counts; replay can first affect event 3. Events 3-8 are descriptive observations of this adaptive trajectory, without a bootstrap. K has capture on; U/R have capture off. Their pre-update wall times are not speedup comparators.

| Event | Domain | K calls | U calls | R calls | R/U reduction % | R/K reduction % |
|---|---|---:|---:|---:|---:|---:|
| 1 | math | 5338 | 5338 | 5338 | 0.0000 | 0.0000 |
| 2 | math | 5475 | 5475 | 5475 | 0.0000 | 0.0000 |
| 3 | coding | 6492 | 6470 | 6487 | -0.2628 | 0.0770 |
| 4 | coding | 7547 | 7519 | 7529 | -0.1330 | 0.2385 |
| 5 | math | 5299 | 5213 | 5231 | -0.3453 | 1.2833 |
| 6 | math | 5337 | 5333 | 5363 | -0.5625 | -0.4872 |
| 7 | coding | 8380 | 8368 | 8370 | -0.0239 | 0.1193 |
| 8 | coding | 7317 | 7277 | 7268 | 0.1237 | 0.6697 |

Events 3-8 total K/U/R = 40,372 / 40,180 / 40,248 calls: R/U -0.1692%, R/K +0.3071%, U/K +0.4756%. R loses to U on five of these six events. Future math totals are 10,636 / 10,546 / 10,594 and coding totals 29,736 / 29,634 / 29,654; R uses more calls than U in both domain aggregates. This does not support treating the small final pooled R/U point estimate as a consistent selector benefit.

## Selection, memory and training evidence

All 1,024 4,096-dimensional descriptors are finite and unit-normalized in FP64, and match their packet receipts. Those receipts identify the last prompt row and a finite positive original norm. Every selected identity, sorted pass plan, coverage score and marginal trajectory is independently reconstructed: clipped nonnegative cosine, 32 distinct greedy choices, exact score ties by original ordinal, common fresh96 seed [20261102,e,0], and uniform old32 seed [20261102,e,1]. For events 2-8, R coverage exceeds U respectively by 0.006149, 0.014710, 0.019605, 0.010082, 0.006201, 0.017103 and 0.030149.

The shared capacity128 reservoir processes all 1,024 unique fresh records in chronological order. All 896 post-fill Algorithm R draws, counter values, slot identities, eviction receipts and before/after PCG64 state match exactly under the separate seed [20261102,0,2]. No current/future record enters old replay; insertion receipts follow all three arm commitments. Final memory contains 63 math and 65 coding records. Domains are joined afterward for reporting, not supplied to selection.

All 24 training workers complete 256 response forwards, 256 backwards and 128 Adam steps. Their 6,144 presentation records match the exact two-pass selections and original path/hash bindings. Every two-microstep accumulation boundary, optimizer counter, fixed 3e-4 LR receipt, replay flag, anchor count and finite metric is checked. Event-1 neutral decoded-state receipts and all 24 state-boundary receipts pass. The audit does not newly decode model, Adam or RNG tensors.

| Arm | Presentations | Adam steps | Replay presentations | Prompt tokens | Response tokens | Active anchors |
|---|---:|---:|---:|---:|---:|---:|
| K | 2048 | 1024 | 0 | 114374 | 588760 | 542308 |
| U | 2048 | 1024 | 224 | 116786 | 585134 | 543435 |
| R | 2048 | 1024 | 224 | 115765 | 587487 | 545614 |

Totals are 6,144 presentations, 3,072 Adam steps, 448 replay presentations, 346,925 prompt tokens, 1,761,381 response tokens and 1,631,357 active anchors. Root work reconciles 3,072 pre-update generations, four capture parity generations and 1,024 preparation forwards, each with zero unknown operations. Final evaluation adds 2,304 generations: 5,380 complete generations overall. Equal response counts do not imply equal token work.

## Native validity, costs and retained capability

The audit reconstructs all 270,623 native verification blocks: 153,239 pre-update, 117,198 final and 186 parity. It verifies first-error decisions, committed prefixes, causal anchors/positions, complete output/termination, accepted/progress/call metrics and exact native sidecar equality. All final outputs match the retained canonical tokens and stop semantics. All four fixed capture controls match K. Publication helper/cache receipts and complete native weight inventories pass; no new GPU inference or weight comparison was run.

The parent retrieval verified all 19,639 compact files. This auditor reads and hashes 14,932 evidence files; `FINAL_AUDIT_INPUTS.json` records their identities and the auditor source hashes. Original bulk tensors are outside this retrieval. Descriptor extraction and scientific-state equality therefore rely on their retained execution receipts; they are not fresh tensor-level replications. The audit's first execution took 16.77 seconds with approximately 608 MiB peak RSS.

Recorded campaign costs remain separate and may nest: campaign 17,525.49 s; adaptation-ready 15,327.05 s; training workers 10,016.63 s; final evaluation 2,191.94 s; export 434.00 s; selection 0.683 s; descriptors 0.453 s. These do not sum into an independently measured total or prove training-inclusive/asynchronous speedup. Detailed costs remain in the original status/report.

There are 2,986 retirement receipts, recording 208,569,153,843 logical bytes and 208,574,205,952 allocated bytes cumulatively. They state retired, not archived, with reasons and retained-evidence references. This audit does not verify current remote existence or reconstruct retired features/intermediate states. Intermediate tensor replay/resume is lost; compact receipts remain. No raw prompt or response text is included here, and no production file was modified.

## Statistical fallacy scan: 11/11

| Category | Assessment | Evidence and limit |
|---|---|---|
| Simpson's paradox | CAUTION | No classic all-strata reversal. R/U is +0.0920% pooled, -1.3259% math and +1.1534% coding. Opposing domain effects nearly cancel; the pooled number must not hide math harm. |
| Ecological fallacy | CAUTION | The ratio of call totals is not an average individual benefit. R/U has 101 wins, 107 regressions and 48 ties, with 241 calls saved and 229 added. R/K has 101/112/43 and 241/262. |
| Berkson's paradox | CAUTION | The fixed, filtered, previously used math/code panel is a selected population. Neither 256 panel requests nor the designed recurrence stream estimates natural conversation traffic. |
| Collider bias | NOTE | Every planned final slot and pre-update row is present and valid, so no success-conditioned subset was used. The selector uses current/past prompt descriptors; final outcomes and labels do not enter the reconstructed choices. |
| Base rate neglect | CAUTION | Final sampling is 128 math plus 128 coding requests, while future events 3-8 contain 256 math and 512 coding. Ratios weight by reference call totals; neither mix is an estimate of deployment prevalence. |
| Regression to the mean | NOTE | Contemporaneous fixed K/U controls and all prespecified events are retained. Samples were not chosen after an extreme efficacy result. Three timing repeats concern the same trained endpoints, not three training replications. |
| Survivorship bias | NOTE | All 2304 final slots, 3072 pre-update generations, four parity controls and 24 worker ledgers reconcile. No invalid, unknown or unattempted final generation is dropped; no unknown worker/root operation remains. |
| Look-elsewhere effect | CAUTION | Pooled R/U is primary; R/K is required. Neither interval excludes zero. Domain, U/K and timing intervals are descriptive and unadjusted; coding R/U alone cannot turn the overall endpoint into a positive result. |
| Garden of forking paths | CAUTION | The presealed recipe, fixed seeds, eight events, 32-record allocation, exact selection and bootstrap agree. No outcome-based rerun appears in this evidence. The reused panel and prior research choices still preclude a fresh confirmatory claim. |
| Correlation versus causation | CAUTION | The controlled R/U contrast compares selectors at the same allocation under shared K-generated supervision. It does not identify why domain effects differ, establish seed robustness, or show asynchronous or training-inclusive speedup. Coverage improved at all seven replay events without a demonstrated overall serving gain. |
| Reverse causality | NOTE | All selected old records precede their training event; admission follows all three commits. Future-serving counts first differ at event 3 after the event-2 update. Temporal order is correct, but it does not prove a general mechanism explanation. |

Machine-readable details: `FINAL_REVIEW_DETAIL.json`. Reconstruction evidence: `FINAL_EVIDENCE_AUDIT.json`; input provenance: `FINAL_AUDIT_INPUTS.json`. No stage or project direction is advanced by this review.
