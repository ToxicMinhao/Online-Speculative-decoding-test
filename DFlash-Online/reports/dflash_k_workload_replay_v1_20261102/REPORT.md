# Track2: workload-conditioned replay with K

## Material Passport

- **Completed, valid. Verification status: ANALYZED.** ARS-Codex 3.24.0 experiment-agent run → validate; CPU reconstruction of retained records, not an independent GPU reproduction.
- Execution `8a4cbcc8b78adfee845a0a1409db4a0d29c54058`; [PR #24](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/24). Scientific base `4d94f30c42701d0f86ac04e2829c834593c16d8e`; qualified parent `0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34`.
- A800 run: `2026-11-02T21:01:53Z` to `2026-11-03T01:56:37Z`; wrapper, campaign, report and postflight all exit 0. No retries, tuning or additional arms.
- Authority: [unchanged handoff](HANDOFF.txt). Reporting commit is separate from the execution commit; see [provenance](PROVENANCE.json).

## Finding and decision

**Coverage-selected replay R did not establish an improvement over K or uniform replay U on the balanced final panel.** R used 13,025 verification calls, U 13,037 and K 13,004. R/U saved only 12 calls (+0.0920%, interval −0.4775% to +0.6432%); R/K added 21 calls (−0.1615%, interval −0.8027% to +0.4487%). These intervals do not establish equivalence either.

The domain split matters: versus U, R saved 86 coding calls but added 74 math calls. R was slower in the pooled fresh timing measurements. During future-serving events 3–8, U also used fewer calls than R. The coverage score increased in every replay event, but this did not establish useful overall selection. **Recommendation: retain K and park this fixed coverage-replay allocation.** This is not a claim that every replay strategy fails. No follow-on run, smaller dose, mixed selector or tuning was performed or authorized by the outcome.

## What was implemented and kept fixed

All arms began at released Qwen3-8B-DFlash-b16 with identical decoded state, empty Adam, seed 17 and an empty common replay memory. Target/tokenizer Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; draft revision `9b41424b7109f9c5413454f481b09a82b85333f4`. Eight events followed **M1, M2, C1, C2, M3, M4, C3, C4**, each with 128 K-generated verified responses. Math and coding are controlled recurring workloads, not naturally sampled conversation traffic.

K uses two complete fresh passes. From event 2, U and R use the same 96 fresh responses plus 32 past responses in pass two; U samples uniformly, R greedily maximizes prompt-descriptor coverage. Their shared 128-record reservoir admits current records only after all three event commits. The descriptor is the normalized FP64 frozen target last-prompt hidden vector already in the prepared packet; no model scoring forwards, domain labels or future evaluation information enter selection. New code is additive under `experiments/k_workload_replay_v1`; historical full-K guards and native numerical training operations remain unchanged.

Each arm completed **1,024 Adam steps and 2,048 presentations**. K's objective remains forward KL over its current matching prefix plus first error, gamma 7, per-response selected-weight normalization, two-response mean accumulation, BF16 native AdamW, LR 3e-4, betas .9/.999, epsilon 1e-8, decay .01 and clipping 1. The inherited horizon remains 7,424 steps / max_events 58; stopping at event 8 did not alter scheduling. Microbatch 1, accumulation 2, up to 512 native anchors, projection chunks 8 and qualified Flex attention are unchanged. Each arm maintains its own continuous native RNG; later replay anchors are not forcibly matched across different inputs.

Serving preserves greedy BF16, block 16, 2,048 tokens or EOS, full prompts, context 4,096 and concurrency 1; training context is 2,272. The final panel contains the prescribed 128 math plus 128 coding requests, interleaved, with three capture-disabled repeats and the prescribed blockwise counterbalance. All training workers had exited before final serving; 37 before-batch/after-final barriers observed no other GPU process or owned artifact transfer. These barriers are sampled checks, not a claim of continuous process monitoring. Checkpoint publication/loading was outside request timing. Final comparisons share exact output tokens and termination; all 256 unique final requests ended with EOS, with no cap hits.

## Final endpoint: counts, acceptance and fresh timing

Counts use one validated trajectory per request after all three repeats agree. Request time is the sum of each request's three-repeat mean, not the sum of all repeated executions.

| Panel | Arm | Calls | Output tokens | Accepted draft tokens/call | Committed progress/call | Request time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| pooled | K | 13004 | 75906 | 4.8361 | 5.8174 | 230.281 |
| pooled | U | 13037 | 75906 | 4.8204 | 5.8027 | 230.224 |
| pooled | R | 13025 | 75906 | 4.8268 | 5.8081 | 231.765 |
| math | K | 5611 | 37380 | 5.6605 | 6.6391 | 99.632 |
| math | U | 5581 | 37380 | 5.6943 | 6.6748 | 99.780 |
| math | R | 5655 | 37380 | 5.6087 | 6.5874 | 101.834 |
| coding | K | 7393 | 38526 | 4.2103 | 5.1938 | 130.650 |
| coding | U | 7456 | 38526 | 4.1663 | 5.1499 | 130.445 |
| coding | R | 7370 | 38526 | 4.2269 | 5.2100 | 129.931 |

“Accepted draft tokens/call” counts accepted proposals. “Committed progress/call” counts all verifier-loop progress (accepted proposals plus target commits), excluding the initial target seed. Thus progress is 75,650 for 75,906 output tokens over 256 requests. `output_tokens/call`, separately retained in SUMMARY, includes those 256 seeds. Calls per 1,000 output tokens are K 171.3172, U 171.7519 and R 171.5938. The three methods have identical outputs, so throughput ratio equals reference total time / candidate total time.

Positive reductions favor the candidate:

| Panel | Candidate / reference | Call reduction [95% interval] | Wins / harms / ties | Gross saved / added calls | Time reduction |
| --- | --- | --- | --- | --- | --- |
| pooled | R / U | +0.0920% [-0.4775, +0.6432] | 101/107/48 | 241/229 | -0.6691% |
| pooled | R / K | -0.1615% [-0.8027, +0.4487] | 101/112/43 | 241/262 | -0.6441% |
| pooled | U / K | -0.2538% [-0.9200, +0.3989] | 110/106/40 | 250/283 | +0.0249% |
| math | R / U | -1.3259% [-2.1867, -0.5011] | 38/66/24 | 73/147 | -2.0587% |
| math | R / K | -0.7842% [-1.7928, +0.1968] | 45/63/20 | 104/148 | -2.2101% |
| math | U / K | +0.5347% [-0.5361, +1.6081] | 62/45/21 | 146/116 | -0.1484% |
| coding | R / U | +1.1534% [+0.4136, +1.8845] | 63/41/24 | 168/82 | +0.3937% |
| coding | R / K | +0.3111% [-0.5140, +1.0909] | 56/49/23 | 137/114 | +0.5501% |
| coding | U / K | -0.8522% [-1.6863, -0.0428] | 48/61/19 | 104/167 | +0.1570% |

Pooled R/U time reduction is -0.6691% [-1.7590, +0.3440]; R/K is -0.6441% [-1.7911, +0.4262]. Their throughput ratios are 0.993353 and 0.993600, respectively. There is no demonstrated serving-time gain. All intervals and domain throughput/count ratios are in [SUMMARY.json](SUMMARY.json).

| Repeat | K request/native/cache (s) | U request/native/cache (s) | R request/native/cache (s) |
| --- | --- | --- | --- |
| 0 | 229.608 / 221.054 / 4.397 | 229.806 / 222.408 / 3.317 | 231.560 / 222.789 / 4.560 |
| 1 | 230.803 / 221.880 / 4.761 | 230.476 / 222.648 / 3.750 | 231.430 / 222.889 / 4.335 |
| 2 | 230.433 / 220.996 / 5.282 | 230.391 / 222.753 / 3.554 | 232.304 / 223.194 / 4.906 |

Native generation and cache maintenance are measured components of complete request time, not a detailed GPU-kernel decomposition. The remaining measured request overhead is not assigned to an unmeasured cause. All repeats, including regressions, appear in [ALL_FINAL_ROWS.csv](ALL_FINAL_ROWS.csv); unique paired rows appear in [FINAL_PAIRED.csv](FINAL_PAIRED.csv).

Intervals use exactly 10,000 paired stratified PCG64 resamples, seed 20261102: 128 math and 128 coding identities per draw, shared across all methods; linear 2.5th/97.5th percentiles. They are conditional on this reused panel and one fitted seed/trajectory. Primary R/U and required R/K are both reported; domain, U/K and time intervals are descriptive and unadjusted, not simultaneous familywise coverage. Timing repeats are not independent requests or training seeds.

## Future serving and selector behavior

These requests were served before learning from their own event. K capture-on times cannot be compared with U/R capture-off times; the table uses calls only. Events 1 and 2 have exact neutral equality as required. Replay first affects serving at event 3.

| Event/workload | K calls | U calls | R calls | R/U reduction | R/K reduction | U/K reduction |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / math | 5338 | 5338 | 5338 | +0.0000% | +0.0000% | +0.0000% |
| 2 / math | 5475 | 5475 | 5475 | +0.0000% | +0.0000% | +0.0000% |
| 3 / coding | 6492 | 6470 | 6487 | -0.2628% | +0.0770% | +0.3389% |
| 4 / coding | 7547 | 7519 | 7529 | -0.1330% | +0.2385% | +0.3710% |
| 5 / math | 5299 | 5213 | 5231 | -0.3453% | +1.2833% | +1.6229% |
| 6 / math | 5337 | 5333 | 5363 | -0.5625% | -0.4872% | +0.0749% |
| 7 / coding | 8380 | 8368 | 8370 | -0.0239% | +0.1193% | +0.1432% |
| 8 / coding | 7317 | 7277 | 7268 | +0.1237% | +0.6697% | +0.5467% |

Across events 3–8 (768 distinct requests: 256 math, 512 coding), K/U/R used **40,372 / 40,180 / 40,248 calls**. R/K saved 0.3071%, U/K saved 0.4756%, while R used 0.1692% more calls than U. Both replay arms helped on the first math return (event 5), but R's advantage over K reversed at event 6. R was worse than U in five of six future events. These are descriptive observations along an adaptive trajectory; no independent-request trajectory bootstrap was applied.

The selection rule executed as specified:

| Event | Memory math/code | U replay math/code | R replay math/code | U coverage | R coverage |
| --- | --- | --- | --- | --- | --- |
| 2 | 128/0 | 32/0 | 32/0 | 0.955248 | 0.961397 |
| 3 | 128/0 | 32/0 | 32/0 | 0.751500 | 0.766210 |
| 4 | 90/38 | 24/8 | 3/29 | 0.919856 | 0.939462 |
| 5 | 65/63 | 17/15 | 32/0 | 0.947906 | 0.957987 |
| 6 | 77/51 | 14/18 | 32/0 | 0.952064 | 0.958265 |
| 7 | 82/46 | 17/15 | 0/32 | 0.921832 | 0.938936 |
| 8 | 71/57 | 21/11 | 0/32 | 0.909640 | 0.939789 |

R increased measured coverage in all seven replay events. Once both domains were available, it selected 29/32 current-domain records at event 4 and 32/32 at events 5–8. The descriptor therefore distinguished these workloads; it was not simply uninformative. At the final coding event, R replayed no math records despite 71 math records being available, whereas U replayed 21. This is consistent with a current-workload versus retention tradeoff, but does not isolate its causal mechanism. It does not prove a different split or replay dose would work.

Exact member/selection/insertion ledgers, pass bindings and RNG states are in [ledgers](ledgers); [SELECTION_SUMMARY.csv](SELECTION_SUMMARY.csv) includes ages and assignment concentration. The full descriptor vectors and original memory snapshots remain in the private hash-verified archive; public memory ledgers omit only those arrays and preserve their source hashes. No selector change followed these results.

## Work, exposure and costs

| Work | Planned | Completed |
|---|---:|---:|
| Pre-update generations (includes 1,024 K feedback generations) | 3,072 | 3,072 |
| Final capture-off generations | 2,304 | 2,304 |
| Capture parity controls | 4 | 4 |
| Total complete generations | 5,380 | 5,380 |
| Terminal teacher preparations | 1,024 | 1,024 |
| Adam steps | 3,072 | 3,072 |
| Ordinary response forwards/backwards | 6,144 / 6,144 | 6,144 / 6,144 |
| Replay presentations (included above) | 448 | 448 |
| Extra scoring passes / disposable Adam steps / retries | 0 | 0 |

All planned executions returned valid results; there are zero invalid, unknown or unattempted slots. Capture parity passed on all four fixed controls. Event 1 complete weights/Adam/RNG/counters/anchor equality passed; event 2 mixed-pass exposure passed. No separate GPU rehearsal was added. Startup/compilation occurred within the included operations; CPU checks and preflight/postflight add no model work. [WORK_ACCOUNTING.json](WORK_ACCOUNTING.json) retains attempt/completion ledgers.

| Arm | Prompt-token presentations | Response-token presentations | Active anchors | Native training (s) | Worker + initialization + export (s) |
|---|---:|---:|---:|---:|---:|
| K | 114,374 | 588,760 | 542,308 | 2,806.719 | 3,533.509 |
| U | 116,786 | 585,134 | 543,435 | 2,764.578 | 3,492.420 |
| R | 115,765 | 587,487 | 545,614 | 2,775.998 | 3,505.994 |

Actual selected proposal occurrences across all presentations were K 4,328,864 of 7,924,658 available, U 4,401,565 of 7,940,991 available, R 4,418,085 of 7,973,717 available. These include repeated/overlapping anchors; they are not unique supervised tokens. [Per-pass exposure](TRAINING_EXPOSURE.csv) retains weighted sums and first-error positions, counted only from each original response observation.

Exposure counts repeated presentations, not unique tokens. Equal presentations do not imply equal token FLOPs. The native-training timer includes forwards, backwards, optimizer and observation work; isolated forward/backward/Adam wall times were **not measured**. Their work counts are known, but a finer time breakdown cannot be reconstructed. Each complete-worker total includes training, restoration, input validation, persistence, state validation and cleanup; the rightmost subtotal sums only disjoint initialization-worker, training-worker and later-export scopes, excluding shared costs. Nested cost rows must not be added together.

R's observed native training was 11.421 s (0.413%) longer than U and 30.721 s (1.095%) shorter than K. This is not an equal-FLOP or isolated overhead comparison: lengths/anchors differ, workers run K/U/R, and qualified compiler caches are reused. Compilation is included; named compiler timers are nested and do not define additional work. Complete per-arm construction, restoration, input I/O, persistence, export and process costs are in [the cost review](reviews/FINAL_TRAINING_COST_REVIEW.md) and [event cost rows](reviews/cost_retention_final_001/TRAINING_COST_ROWS.csv).

| Shared or outer measured scope | Seconds | Interpretation |
|---|---:|---|
| All three arms adaptation-ready wall | 15,327.051 | Initialization, feedback/serving, training, publication and checks |
| Final evaluation wall | 2,191.939 | Includes 2,304 serving executions, switching and checks |
| Campaign wall | 17,525.487 | Outer total; 4 h 52 m 5 s; excludes separate pre/postflight and report |
| Full wrapper wall | 17,684 | 4 h 54 m 44 s; includes CPU tests, pre/postflight and report |
| Teacher preparation | 84.036 | Includes loading, terminal forward, concatenation and validation |
| Prepared-feature persistence | 48.736 | Not all capture I/O |
| Descriptor extraction | 0.453 | Existing target features; no model forward |
| Selection total | 0.683 | Includes 0.548 scoring + 0.082 greedy search |
| Reservoir maintenance | 23.960 | Commit, eviction and associated retirement |
| Deferred exports / verification | 434.004 | Across all arms/events; outside native training |
| Pre-update activation / validation | 91.322 | Outside per-request timing |
| Final activation / weight validation | 52.432 / 7.057 | Outside per-request timing |
| Resource scans | 1,384.528 | Included in campaign wall; safeguards, not model work |

These shared and outer times overlap the per-arm scopes and are not additive. Teacher preparation is shared once, not charged three times. K feedback capture overhead is embedded in its pre-update request path; its isolated incremental overhead was **not measured**. The four parity controls checked outputs and counts, not a matched capture-cost timing study. Selection was cheap here (maximum counted persistent NumPy arrays 8,520,704 bytes), but low selector cost does not rescue the absent final drafting benefit. No asynchronous or training-inclusive deployment speedup was measured.

Storage admission passed with 76,723,990,528 free bytes against 75,023,917,685 required, including 8 GiB reserve, 4 GiB additional compiler allowance, 3 GiB compact allowance, 2 GiB workspace and 5% tensor headroom. The compiler allowance was declared before results; actual growth was 100,327,424 bytes. Minimum free space over 3,150 discrete checks was 43,447,312,384 bytes; postflight had 45,090,390,016 bytes free. Training allocator maxima were 20,723,064,832 / 20,788,358,144 / 20,788,358,144 allocated bytes for K/U/R, with reserved maxima about 27.58 GB. These peaks start after construction/restore; full-process GPU and host-memory peaks were not measured. Discrete storage samples are not continuous peaks.

## Correctness review, provenance and retention

The local focused CPU suite passed 58 tests; final integration checks passed 32; the remote launch suite passed 59. An earlier local fixture-only invocation had three test errors, repaired before launch and preserved in the private implementation record. No GPU failure or scientific retry occurred. Preflight and postflight each checked all 3,453 sealed source/input/model/runtime identities. The published implementation is the exact execution code at `8a4cbcc8b78adfee845a0a1409db4a0d29c54058`.

Historical coding canonical provenance has one disclosed limit: its original direct byte-hash receipt was unavailable. The copied canonical bytes were sealed with a current hash, and complete native/logical/output validation matched the retained hash-bound scalar, validity and termination receipts. This is not described as verification of a missing old byte hash. Math canonical byte receipts were available and verified. Exact output agreement is under the qualified native verifier, not a new target-only oracle.

Separate CPU reconstruction passed **2,917,883 checks with zero discrepancies**, independently recomputing native/output validity, source/input bindings, exact selections/reservoir state, exposure/work ledgers, final ratios and the prespecified bootstrap. [Audit](reviews/final_reconstruction_001/FINAL_EVIDENCE_AUDIT.md), [machine-readable checks](reviews/final_reconstruction_001/FINAL_EVIDENCE_AUDIT.json). A further [handoff/resource review](reviews/FINAL_HANDOFF_REVIEW.md) performed 51,679 checks across 9,594 hashed/read files, reconstructed final and future-serving results separately and checked delivery coverage. Cost extraction consumed 3,293 hashed receipts with no missing required receipt or issue.

Review provenance is explicit: reviewers are separate agents on the same team, with shared history and disclosed implementation overlap; the trainer author prepared the cost extraction, and the selector author prepared the reconstruction auditor. The handoff reviewer separately checked raw results. This is not blinded external validation, independent GPU reproduction or a new decode of retired tensors. All 11 ARS statistical-fallacy categories were assessed, including domain aggregation, reused-panel selection, all-attempt reporting, multiplicity, prespecification and causal limits. No positive headline is inferred from a domain-only gain or a confidence interval crossing zero.

The raw run is retained privately at `/home/minhao2/EAGLE_SPEC/research/dflash_k_workload_replay_v1_20261102/retrieved_001/production_001` and on A800 under `/data/sources/minhao/runs/dflash_k_workload_replay_v1_20261102/production_001`. Retrieval verified all **19,639 compact files** (1,919,536,120 bytes) and the archive SHA-256. Private prompts, responses, output token arrays and model tensors are not published to Git. Bulk files remain on A800; [FINAL_BULK_PRESENCE.json](FINAL_BULK_PRESENCE.json) records a fresh existence/size/inode check, not a second tensor decode. Final state/export identities and production exact-live-state checks are in [FINAL_STATE_RECEIPTS.json](FINAL_STATE_RECEIPTS.json).

Fresh filesystem inspection confirms 138 bulk paths on 135 unique device/inode pairs: one shared initial scientific state, three complete final states, three final serving exports, 128 final-reservoir packets, and three latest-state hardlink aliases. Retained unique bulk allocation is 29,558,915,072 bytes. U/R retain their independently validated initial restore metadata pointing to the identical shared initial payload.

Exactly 2,986 new-run bulk paths were retired after their prescribed validation: 1,024 native capture tensors, 1,024 terminal tensors, 896 evicted prepared packets, 21 intermediate exports and 21 predecessor complete states. They were **deleted, not archived**. Compact outputs/native traces, descriptor and anchor records, selection/membership/RNG receipts and validity evidence survive. Direct tensor replay of evicted feedback and exact intermediate event-1–7 resume/serving are no longer available. Final complete scientific states and reservoir remain available. No historical artifacts were deleted or offloaded to fit this run. Nominal removed-file sizes do not establish measured physical space reclaimed; [exact retirement ledger](reviews/cost_retention_final_001/RETIREMENT_PATHS.csv) and [retained artifacts](reviews/cost_retention_final_001/RETAINED_ARTIFACTS.csv) preserve paths and capability limits. The older interrupted full-K offload remains separate and was not resumed.

## Prespecified decision boundaries

- **A:** not met: neither R/U nor R/K has a positive lower bound; future R/U is negative and common-mode timing does not improve.
- **B:** not met: neither replay arm establishes balanced-final superiority over K.
- **C:** only its point-estimate sign pattern appears. The tiny positive R/U estimate is not an established selection advantage; the recorded recurrence observations do not justify automatically proposing a smaller dose.
- **D:** a descriptive specialization/retention pattern is visible relative to U (coding improves, math worsens), with explicit current-domain selection. It does not establish a balanced-endpoint benefit or a proven tradeoff versus K. No mixed-selector experiment follows.
- **E/G:** applicable practical conclusion: coverage improves without a demonstrated overall serving gain, and endpoint effects remain inconclusive. Future directions are not consistently favorable for R relative to U.
- **F:** no robust pooled call gain was established, and there is no pooled time gain. Selector overhead is small; removing that overhead would not establish improved drafting.

**One next action proposed: keep K and park this fixed coverage-selected replay mechanism.** Reopening requires a separately agreed concrete mechanism question or new evidence; this completed probe authorizes no further run. The unchanged raw generated report is preserved as [GENERATED_REPORT_ORIGINAL.md](GENERATED_REPORT_ORIGINAL.md). Its automatic C label was based only on point-estimate signs; this complete evidence review supersedes that label without changing any numerical result. Stage 1 / project None remains unchanged.
