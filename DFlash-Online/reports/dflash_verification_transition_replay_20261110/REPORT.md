# K+ verification-transition replay: completed diagnostic

**Verdict: the proposed expensive-restart bottleneck is not supported on either dataset. Do not pursue the proposed newly-reached-start allocation branch on these results.** Actual advances generally left an equally costly or cheaper continuation under both fixed checkpoints. One MathQA request shows an expensive farther restart under both checkpoints; it is an isolated case, not the repeated pattern required by the handoff.

ARS experiment-agent, execution and evidence validation. Actual clean execution commit: **`d7b1311e7b2f910e935bf81b4e40282767d7540c`**. This report supersedes the earlier implementation-only status, while preserving that report and all earlier evidence. No algorithm, checkpoint, selection rule or tolerance changed during execution. No training occurred.

## Native gate and completion

All **128 selected requests** completed, with **256 authenticated original walks, 256 crossed suffixes and eight neutral suffixes**. There were zero invalid, failed, unknown or unattempted prescribed executions. Both GPUs were idle after the run.

The mandatory neutral constructor gate passed for both datasets and both checkpoints. All eight reconstructed suffixes reproduce every proposal, native position, committed step, output token, termination and remaining verification count: **267 rounds and 4,005 proposal occurrences**. Native receipts also record exact valid-prefix target KV and neutral draft KV equality. The separate CPU audit verifies those receipts and downstream traces; large KV tensors were not transferred for another tensor-level comparison. See [native gate review](NATIVE_GATE_REVIEW.md).

All 256 fresh original walks reproduce their bound historical native traces, including all proposals and counts. Every crossed continuation preserves the exact canonical output, original cap and termination. All **1,421** postflight source/input checks pass, including 1,139 source files and 282 artifact identities. The checkout remains clean at the reviewed commit; all four draft exports are unchanged. [Execution provenance](EXECUTION_PROVENANCE.json), [source/input checks](SOURCE_AND_INPUT_CHECKS.json).

## What was compared

For each dataset, ordinary **KPLUS4 (512 Adam steps)** and **KPLUS6 (768)** were frozen. The sample contains the first 64 requests under the prespecified SHA256 ordering from each original 256-request development panel. IDs were sealed before inspecting outcomes. These are reused development requests.

Serving used the pinned target/tokenizer/runtime, five-layer native DFlash, BF16/FA3, greedy block 16, concurrency one, context 4,096, and each original 2,048-token/EOS budget. At the first round where actual committed progress differed, the two native successor states were retained. Source target state and committed feature-ingestion history stayed fixed, while the destination draft KV was rebuilt through its own native context ingestion. An offset alone was not treated as complete state.

The four cells count **subsequent** verification calls: C44 = KPLUS4 at its own successor; C46 = KPLUS4 at KPLUS6's successor; C64 = KPLUS6 at KPLUS4's successor; C66 = KPLUS6 at its own successor. Thus:

- R4 = C46 − C44; R6 = C66 − C64: change from starting at the KPLUS6 successor, holding the draft fixed.
- M4 = C64 − C44; M6 = C66 − C46: checkpoint effects at the two fixed successors.
- D = C66 − C44 = R4 + M6 = M4 + R6 = original total-call difference.

All identities reproduce for every request. Positive R means extra remaining calls; negative R means fewer. An **actual advance** means KPLUS6 committed farther at that round. A **verified repair** separately requires repairing KPLUS4's old first rejected token while keeping the preceding canonical matches. No actual advance is removed if this check fails. In this sample, all 31 GSM8K and 40 MathQA advances independently pass the repair check; the two concepts are still distinct.

## Endpoint and transition counts

| Dataset | Selected / complete | KPLUS4 → KPLUS6 calls | Fewer / more / tied total calls | First advance / regression / no divergence |
|---|---:|---:|---:|---:|
| GSM8K | 64 / 64 | 2,761 → 2,748 | 29 / 29 / 6 | 31 / 33 / 0 |
| MathQA | 64 / 64 | 4,330 → 4,252 | 39 / 19 / 6 | 40 / 24 / 0 |

Total-call ties still contain a progress divergence. Earlier proposal differences with equal progress occurred on 54 GSM8K requests (132 rounds) and 30 MathQA requests (84 rounds); these remain recorded. None of the selected successors is within 16 canonical tokens of termination, so the all-advance and prespecified interior-advance strata coincide. No terminal-zero or missing-cell substitution was needed.

## Restart effects after actual advances

Means are changes in **remaining verification calls**, not percentages or latency changes. Brackets are two-sided 95% percentile intervals.

| Dataset | Interior advances / verified repairs | Both R4 and R6 positive | Mean R4 [95% CI] | Mean R6 [95% CI] |
|---|---:|---:|---:|---:|
| GSM8K | 31 / 31 | 0 / 31 | −0.387 [−0.700, −0.071] | −0.613 [−0.857, −0.390] |
| MathQA | 40 / 40 | 1 / 40 | −0.525 [−0.727, −0.303] | −0.700 [−1.026, −0.410] |

| Dataset / effect | Negative / zero / positive | Median | Range | Unconditional contribution per selected request [95% CI] |
|---|---:|---:|---:|---:|
| GSM8K R4 | 12 / 17 / 2 | 0 | −3 to +2 | −0.188 [−0.344, −0.031] |
| GSM8K R6 | 17 / 14 / 0 | −1 | −3 to 0 | −0.297 [−0.438, −0.172] |
| MathQA R4 | 23 / 14 / 3 | −1 | −2 to +1 | −0.328 [−0.469, −0.188] |
| MathQA R6 | 25 / 12 / 3 | −1 | −5 to +1 | −0.438 [−0.656, −0.250] |

The unconditional quantity is I[actual advance] × R divided across all 64 selected requests. Regressions contribute zero to this specific quantity but remain in the full table and endpoint analysis. All negative effects and all cases with positive effects are retained.

The lone both-positive MathQA case is `mathqa-transfer-0583`: progress 9→10; C44/C46/C64/C66 = **10/11/11/12**; R4 = R6 = +1. It is an interior verified repair with D = +2. This establishes that the phenomenon can occur, not that it is a useful bottleneck for the proposed algorithm.

The two GSM8K advances expensive under KPLUS4 are `eval_00508` (R4/R6 = +1/0) and `eval_00560` (+2/−1). Their extra restart cost is absent under KPLUS6. MathQA also retains four mixed-sign/zero cases in addition to the lone both-positive case. The [full four-cell table](PER_REQUEST.csv) includes them all.

Of the endpoint regressions, 19/29 in GSM8K and 11/19 in MathQA start with a local progress regression. The other 10 and eight start with an advance. This is a descriptive breakdown; an endpoint regression after an advance is not by itself evidence of an expensive restart. The crossed cells distinguish that question from the checkpoint effect.

## Uncertainty and concentration

The original analyzer and a separately coded CPU audit reproduce all raw rows and intervals. Each dataset uses **10,000 paired full-request bootstrap resamples**, PCG64 seed **20261110**. Each resample draws the original 64 rows and recomputes the advance stratum; neither rounds nor four-cell entries are independently resampled. There are **zero empty-stratum draws**. Eligible samples are 31 and 40, not 256 continuations or thousands of rounds.

All 64 leave-one-request-out conditional means remain negative for each dataset/effect. Largest absolute-contributor shares are 16.7%/15.8% for GSM8K R4/R6 and 7.4%/14.7% for MathQA. Positive contributions are sparse: GSM8K R4's largest positive share is 66.7%, R6 has none; MathQA's shares are 33.3% for each effect. Full medians, quartiles, standard deviations, signed/absolute sums, contributor IDs, empty-draw counts and leave-one-out values are in [SUMMARY.json](SUMMARY.json).

These are prespecified nominal intervals, not simultaneous family-wise guarantees. The all/interior strata are identical here and are not independent replications. This one selected subset of reused development panels does not establish generality to other checkpoints, datasets, seeds or transitions within training presentations.

## Work and costs

| Work | Actual | Ceiling |
|---|---:|---:|
| Selected requests | 128 | 128 |
| Fresh original completions | 256 | 256 |
| Crossed suffixes | 256 | 256 |
| Neutral suffixes | 8 | 8 |
| Draft-block / standalone verifier evaluations | 27,800 | 32,768 |
| Occupied GPU seconds | 1,301.658 | 14,400 |
| Optimizer steps / backward passes / feedback collection | 0 / 0 / 0 | 0 / 0 / 0 |

All actual work completed; zero failed/unknown attempts. The 4,968 unused block evaluations are unused allowance, not unattempted selected requests. Total block work includes **27,749 native rounds, three startup draft evaluations and 48 standalone startup verifier evaluations**. There were 28,053 target calls, 29,342 context-forward operations and 521 model initialization/publication operations; these overlap with the enclosing round/suffix intervals and are not additive kernel times. Context work processed 274,116 token occurrences, including repeated ingestion: 23,888 target-prefill tokens, 221,306 ordinary/continuation draft tokens, and 28,922 constructor tokens in 1,081 reconstruction segments. The ledger's `rebuild=True` also covers continuation appends; it must not all be called constructor work. Detailed accounting is in [WORK_REVIEW.md](WORK_REVIEW.md) and the complete [compressed work ledger](WORK.jsonl.gz).

Wrapper wall time was **22 min 18 s** (02:39:37–03:01:55 UTC, November 10), including 57 CPU tests. Occupied GPU time was **21 min 41.658 s**, including native loading, startup, reconstruction and model work. The separately recorded final full-identity CPU audit took 23.871 s, outside GPU execution. Implementation preparation's earlier elapsed interval was 1,472.190 s (including concurrent review/documentation), not CPU utilization. Final CPU analysis/audit costs are recorded separately in the review receipts. No request-time or serving-speedup comparison was conducted.

Storage admission had 75.658 GB free against an 8 GiB reserve plus 2 GiB artifact allowance. After native execution, 75.496 GB remained. All 1,971 compact evidence files (157,281,371 bytes) were copied privately and rehashed; checkpoint weights and raw native evidence remain preserved. Large transient successor KV was not archived. No scientific artifact was deleted during this run.

## Review and decision limits

Native execution used the exact reviewed clean checkout. Same-team reviews independently checked gate traces, endpoint arithmetic/bootstrap and the complete work ledger; some reviewers also authored implementation components. These are separate code/evidence checks, not independent GPU replication. The initial gate auditor's 15-proposal/16-verifier-row schema mistake, a work-auditor helper lookup error exposed by a CPU fixture, and an audit-helper insertion error were corrected with their failed audit logs preserved. None was a native execution failure or a scientific-code change.

The result meets the handoff's **generally zero or beneficial restart** branch on both datasets. It does not establish that every farther restart is beneficial, that local repair always improves the full request, or that future allocation methods cannot work. It does not establish a unique percentage attribution of learning gains, nor an untimed-replay serving speedup. The positive isolated MathQA case does not warrant expansion under this handoff.

**Stop at this report.** Retain ordinary K+; do not implement or launch the proposed frontier-allocation pilot on this evidence. No new algorithm, tuning, sample expansion or training run follows automatically. Research stage remains unchanged.

## Delivered evidence

- [Selected IDs and fixed configuration](SELECTED_IDS.json); [original handoff](HANDOFF.txt).
- [Per-request four-cell counts, boundary and repair flags](PER_REQUEST.csv); [paired summaries/intervals](SUMMARY.json).
- [Native checks](NATIVE_CHECKS.json); [gate audit](NATIVE_GATE_REVIEW.md).
- [Endpoint audit](ENDPOINT_REVIEW.md); [work/provenance audit](WORK_REVIEW.md); [11-category interpretation review](INTERPRETATION_REVIEW.md).
- [Actual execution provenance](EXECUTION_PROVENANCE.json); [source/input checks](SOURCE_AND_INPUT_CHECKS.json); [execution totals](EXECUTION_RESULT.json); [complete work ledger](WORK.jsonl.gz).

Prompts, canonical token streams, tensors and private runtime logs remain outside Git. All reports identify the scientific execution commit; later report-publication commits do not replace it.
