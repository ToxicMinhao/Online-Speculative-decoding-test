# Fresh MathQA endpoint interpretation audit

## Material Passport

- Mode: ARS experiment-agent / validate. Verification status: **ANALYZED**.
- Scientific execution: `96f0a13d189b8f5a1d3cfcf103a0ff1097f18cd8`.
- Reviewer: separate interpretation agent, previously responsible for the reporting checklist; no authorship of the endpoint runner or its numeric reducer. Same project team; no independent GPU replication.
- Scope: completed compact tables and receipts, with **64,573 local numeric consistency checks passed**. No model work, network, remote operation or raw-token replay. The separate final evidence auditor supplies full raw/statistical verification.
- Bootstrap intervals were checked against that separate auditor receipt; this reviewer did not independently regenerate its draws.
- Status: **no interpretation blocker found**. Native K+ remains the project baseline; this report does not select a new project stage.

## Supported outcome

For these fixed checkpoints on the prespecified fresh MathQA panel, C58 has lower complete-request latency and fewer verification calls than AREPO58. The primary latency reduction is **1.61027%**, paired 95%CI **[1.33745%,1.88395%]**, with a corresponding **1.68030%** call reduction, CI **[1.40895%,1.95158%]**. This is about 22.705 ms saved per request under this measurement setup. Secondary comparisons also favor C over O and AREPO over O. These are endpoint comparisons of complete own-feedback training recipes, not causal isolation of an allocation rule or loss term.

| Comparison | Time reduction %,95%CI | Call reduction %,95%CI | Call wins / regressions / ties | Gross saved / added | Net saved |
|---|---|---|---|---|---|
| KPLUS_C58_vs_AREPO58 | 1.61027 [1.33745,1.88395] | 1.68030 [1.40895,1.95158] | 599 / 295 / 130 | 2178 / 798 | 1380 |
| KPLUS_C58_vs_O58 | 2.34365 [2.09596,2.59922] | 2.42523 [2.18043,2.67390] | 672 / 233 / 119 | 2476 / 469 | 2007 |
| AREPO58_vs_O58 | 0.74538 [0.50000,0.99483] | 0.75766 [0.52071,0.99992] | 516 / 335 / 173 | 1512 / 885 | 627 |

All request-level categories use1,024 paired requests. Their sum is 1,024 for every contrast; gross saved minus added equals the pooled call difference. In particular, the positive primary pooled result retains 295 regressions—it is not universal improvement.

## Arithmetic, repeats and counters

Counts use one validated deterministic trajectory per request. Each request/arm has exactly three timing observations, averaged before pooling. Local checks reconcile the 3,072 per-request arm rows and 9,216 repeat rows with the completed summary. In every arm, output tokens total 573,061, committed progress 572,037,951 requests end at EOS and 73 at the 2,048-token limit. No cap-hit requests are excluded.

| Arm | Calls | Native accepted / call | Stop-clipped matched / call | Committed progress / call | Sum of request-mean latency,s |
|---|---:|---:|---:|---:|---:|
| O58 | 82755 | 5.91241617 | 5.92285662 | 6.91241617 | 1454.700303 |
| AREPO58 | 82128 | 5.96518849 | 5.97502679 | 6.96518849 | 1443.857261 |
| KPLUS_C58 | 80748 | 6.08422500 | 6.09461535 | 7.08422500 | 1420.607286 |

The exact inherited definitions in `online_feedback/sglang_hooks.py:129–157` matter. Let `accepted_end` be the first mismatching proposal slot (or block width 16) and `commit_rows` be actual emitted progress. The native counter is `commit_rows−1`; the stop-clipped matching trace counter is `min(accepted_end−1,commit_rows)`. Consequently, the latter can exceed the native counter by one when stopping clips the round. The raw pre-stop matching count is a third quantity, `accepted_end−1`. Do not call native acceptance an untruncated matching count or assume the stop-clipped summary must be smaller. Per request, summed committed progress is output length minus the one prefill-produced token. Native pooled acceptance and progress differ by exactly 1 token/call; call and acceptance changes are linked accounting views, not independent confirmations.

| Comparison | Repeat 1 time reduction % | Repeat 2 % | Repeat 3 % |
|---|---:|---:|---:|
| KPLUS_C58_vs_AREPO58 | 1.63424 | 1.63328 | 1.56350 |
| KPLUS_C58_vs_O58 | 2.36111 | 2.37666 | 2.29338 |
| AREPO58_vs_O58 | 0.73894 | 0.75573 | 0.74147 |

All three repeat totals favor the same direction. This is repeated measurement of the same 1,024 requests, not three independent datasets or training seeds. C/AREPO and C/O latency favor C in all 48 scheduling groups; AREPO/O in 44, with 4 contrary groups retained. These descriptive slices do not enlarge the inferential sample.

Repeat 3 cache-maintenance totals rise to 8.440 / 9.443 / 9.805 s for O/AREPO/C, versus roughly 0.50–0.56 s in each earlier repeat. All measurements remain included. The primary complete-request timing result still favors C in every repeat. This audit does not infer the cause of that cost variation or a detailed kernel breakdown from aggregate timings.

## Statistical limits and claim wording

The bootstrap resamples whole paired requests with all arms and repeat times together: 10,000 PCG64 draws, seed 20261119, lexicographic request-ID input order, and linear 2.5/97.5 percentiles. Intervals are marginal 95% intervals conditional on these fixed fitted endpoints and timing observations. Secondary intervals are descriptive and unadjusted. The bootstrap does not establish training-seed robustness, a temporal-noise model or simultaneous coverage for all metrics. No 0.5% threshold or post hoc request removal is used.

This is fresh-input transfer within mathematics from GSM8K-trained weights. Freshness is bounded by the authenticated known-project exposure inventory, not a guarantee about inaccessible histories, semantic near-duplicates or pretraining. Checkpoints have distinct own-feedback histories. The comparison neither isolates the causal effect of C's allocation/cap nor represents AdaFlash's complete high-concurrency system. Historical KPLUS58 timing must not be added as a fourth fresh arm. Serving improvements exclude training/adaptation costs.

Supported wording: “On this prespecified fresh MathQA panel, fixed C58 has lower complete-request time and fewer verification calls than fixed AREPO58, with positive paired 95%lower bounds.” Avoid universal-request, mechanism-causal, cross-task-family, training-seed or training-inclusive claims. An unresolved contrast would not establish equivalence or nondegradation; none of the three reported main time/call contrasts is unresolved here.

## ARS statistical fallacy scan — 11/11 checked

| Category | Finding |
|---|---|
| Simpson's paradox | All three complete-repeat contrasts favor the reported direction. C/AREPO and C/O favor C in 48/48 scheduling groups; AREPO/O in 44/48. Four contrary AREPO/O groups are retained heterogeneity, not evidence of universal improvement. Scheduling groups/repeats are not independent replications. |
| Ecological fallacy | Report the pooled estimand together with request wins/regressions/ties. A pooled benefit does not mean every request improves; primary comparison has 295 request regressions. |
| Berkson's paradox | The input-only sealed eligibility/exclusion protocol does not select by generated answer, length or performance. Nevertheless, findings concern the eligible prospectively unused MathQA pool, not all possible requests. Historical exposure coverage and semantic-near-duplicate/pretraining limits remain. |
| Collider bias | Primary analysis uses the complete sealed panel, including 73 cap hits per arm, without filtering on output length, success or speed. No post-treatment covariate adjustment is used. |
| Base rate neglect | No diagnostic sensitivity/specificity claim. Denominators are explicit: 1,024 requests, 3,072 request-arm records, 9,216 scored repeats; 951 EOS and 73 length terminations per arm. |
| Regression to the mean | Requests were sealed by input-only rank, not extreme prior performance. Frozen endpoints are measured contemporaneously. Prior development-based mechanism selection remains historical context; fresh requests do not create independent training replications. |
| Survivorship bias | Completed work/status receipts report all 10,552 planned complete generations, zero invalid/unknown/unattempted scored/warmup work, and no extra model work. This interpretation audit checks compact table coverage; separate raw audit supplies raw-token verification. |
| Look-elsewhere effect | C/AREPO complete-request latency is the single prespecified primary contrast. C/O, AREPO/O and drafting intervals are descriptive and marginal/unadjusted. Report all; do not treat coupled call/acceptance metrics or repeats as independent confirmations. |
| Garden of forking paths | Selection seed 20261119, exact panel, schedule, three repeats, whole-request bootstrap and endpoints were fixed before generation. No 0.5% success threshold, outlier removal, favorable-repeat selection or extra panel is applied. Report all completed measurements. |
| Correlation versus causation | These measurements establish comparative serving behavior for fixed trained endpoints under this protocol. Distinct own-feedback histories and complete recipes do not isolate C allocation, a cap, or an AdaFlash loss term as the causal source. No seed-robustness or training-inclusive-speedup inference. |
| Reverse causality | Checkpoints and input panel precede scoring, so serving outcomes did not choose these weights or requests. Temporal ordering does not identify why a training component helped; no learning-mechanism causal claim follows. |

## Evidence and verification boundary

Local sources: `evidence/SUMMARY.json`, `PER_REQUEST.csv`, `ALL_REPEATS.csv`, `REPEAT_TOTALS.csv`, completed work/status/cost receipts, postflight identities and the separately executed `FINAL_EVIDENCE_AUDIT.json`. Source hashes and individual numeric findings are retained in `INTERPRETATION_AUDIT.json`.

The separate auditor receipt reports 10,552 raw generations, 864,732 raw verification blocks and a completed independent raw/statistical audit. This reviewer checked that receipt and its comparison numbers, but did not itself reread private responses or reconstruct native trajectories. There is no independent GPU replication claim. The final publication/privacy and compact-delivery checks remain the executor's closeout tasks.
