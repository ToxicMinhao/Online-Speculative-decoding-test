# Independent raw-record and statistical review

## Material Passport

- Origin Skill: ars-codex:academic-research-suite 3.24.0 / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-02
- Verification Status: ANALYZED
- Version Label: full_k_raw_statistics_review_v1
- Execution source: `0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34`, production_002.
- Review: separately implemented CPU audit by a delegated same-family agent with shared task context. The original report and source schemas were visible. This is not a blind or cross-model review, independent GPU reproduction, re-decoding, or tensor-state reinspection.

## Verdict and evidence

**PASS for the record/statistical scope; no scientific discrepancy found.** Independently reconstructed statistics match the original SUMMARY.json exactly: maximum absolute numerical difference is 0.0 across compared values. This verdict is limited to the saved evidence. Overall inferential confidence remains **CAUTION** because this is one fitted seed on a previously used benchmark, with descriptive unadjusted secondary intervals.

The audit read all 12,288 final and 768 intermediate new evaluations, all 1,024 original frozen reference records, their separately saved native traces, all seven schedules and their measurement/status/validation records, plus recorded checkpoint publication and weight checks. It checked 579,882 new native blocks and 50,584 historical frozen blocks. Across 41,271 unique files it reconstructed native first-mismatch decisions, accepted tokens, committed tokens, positions, full output sequences, EOS/cap termination, count scalars, timing validity, request identities and frozen output equality. All three repeats have identical count trajectories per request/method. Every final and intermediate execution remains included; no invalid or missing execution was discarded.

The final balanced schedule matches the independently reconstructed sealed PR21-derived order: 16 blocks × 3 repeats; each method appears in each position 12 times and each ordered method pair has precedence 24 times. Within each method block, all 64 requests retain their original order. Publication/cache-flush and loaded/exported-weight equality are supported by their retained live-check receipts; this review does not load model tensors to repeat those checks.

There are 4,349,703 successful programmatic checks including the local completion described below. The number of checks describes audit coverage; it is not a measure of statistical certainty.

## Independently recovered endpoint effects

Positive percentages mean a reduction relative to the denominator method. Counts use one repeat-consistent trajectory per request; request time uses the three-repeat mean within each of 1,024 requests. Every interval uses 10,000 paired-request bootstrap draws, PCG64 seed 20261102, and linear 2.5/97.5 percentile endpoints. All methods travel together in each resampled request. The five call effects and all fifteen timing-component effects were reconstructed without importing campaign, audit, or report functions.

| Comparison | Call reduction and 95% interval (%) | Request-time reduction and 95% interval (%) | Wins / regressions / ties | Gross saved / added calls |
|---|---:|---:|---:|---:|
| K_HALF/O | +1.6790% [+1.2908, +2.0674] | +1.3703% [+0.6749, +2.0624] | 523 / 345 / 156 | 1512 / 761 |
| K/O | +3.0025% [+2.6667, +3.3457] | +2.1886% [+1.4693, +2.8760] | 630 / 241 / 153 | 1800 / 457 |
| K_HALF/K | -1.3645% [-1.7575, -0.9866] | -0.8366% [-1.4969, -0.2026] | 334 / 535 / 155 | 766 / 1358 |
| K_HALF/F | +13.0575% [+12.5951, +13.5239] | +12.2789% [+11.5071, +13.0387] | 993 / 14 / 17 | 6638 / 33 |
| K/F | +14.2278% [+13.6878, +14.7639] | +13.0067% [+12.1917, +13.8146] | 982 / 18 / 24 | 7230 / 33 |

F/O/K/K_HALF totals are **50,584 / 44,730 / 43,387 / 43,979 calls**, each for exactly 314,847 output tokens. Relative to F, gross gains/harms are O **5,962/108**, K **7,230/33**, K_HALF **6,638/33** calls. All methods retain the same 1,021 EOS-terminated requests and three cap-terminated requests; output length ranges from 94 to 2,048 tokens. Capped outputs are included.

The primary K_HALF/O endpoint interval is above zero, supporting an endpoint gain under this fixed run. K performs better than K_HALF at the final endpoint; this does not imply a general learning-rate ranking across tasks or training seeds. K/O is the same-LR full-updater comparison. K_HALF/O compares different recipes and must not be described as matched-LR evidence. Arm-specific feedback means these results do not isolate mask selection, normalization, or any single component.

The eight fixed curve cells match exactly: K calls at events 14/29/44/58 are **5,405 / 5,274 / 5,164 / 5,123**; K_HALF calls are **5,429 / 5,300 / 5,270 / 5,224**, against retained F **6,046** on the first 128 requests. Event 58 is a subset of existing final repeat-zero records, not another generation. These curves show maintained/increasing benefit against F in this panel; no matched intermediate O evidence supports faster-learning claims.

## Preserved audit implementation failure and local completion

The remote CPU audit completed every raw-record reconstruction and every bootstrap calculation, then exited 1 because its own final comparison guard required a built-in `float` and rejected a computed `numpy.float64`. The initial receipt and script are preserved unchanged. This was an audit-only type-check defect, not a failed experiment or numerical disagreement.

The separate local continuation read those JSON-normalized numbers, verified that the local original SUMMARY.json and CANONICALS.json hashes exactly match the files inspected remotely, and completed all numeric/curve/schedule/length comparisons. Its exit code is 0 and all compared numbers are exactly equal. No experiment, GPU computation, remote audit, archival transfer or scientific retry was performed to resolve the type check. No old receipt was overwritten.

- Complete private audit receipt and per-file hashes: `../../review_20261102/RAW_STATISTICS_AUDIT_FULL.json`, SHA256 `e5a909957957ce2e18373fce50834aea5a8d78e5804bd8c33f058e49d91e4fe1`.
- Read-only audit implementation: [original audit code](code/independent_raw_statistics.py).
- Local completion implementation: [local completion code](code/complete_local_comparison.py).
- Compact review, exact statistics, check coverage, source/script hashes and limitations: [RAW_STATISTICS_REVIEW.json](RAW_STATISTICS_REVIEW.json).

## Statistical fallacy scan

Coverage: **11/11 checked**. No additional scientific analysis or post-hoc filter was introduced.

| Fallacy | Assessment | Scope |
|---|---|---|
| Simpson's paradox | No observed repeat-level reversal for the primary or K/O timing effects; broader subgroup claims unavailable | Ratios of summed paired counts and per-repeat timing totals are retained. One training seed cannot establish seed-level consistency. |
| Ecological fallacy | Not found in the stated endpoint claim | Per-request wins, harms and ties remain visible; an aggregate benefit is not universal request improvement. |
| Berkson's paradox | CAUTION: reused-panel selection | The entire fixed panel is retained, including capped outputs. It remains a selected, previously used benchmark. |
| Collider bias | No outcome-conditioned exclusion identified | All planned valid requests enter analysis; no conditioning on response success or performance. |
| Base-rate neglect | Not applicable to diagnostic classification | Full 1,024-request denominator and call totals are supplied. |
| Regression to the mean | No new extreme-case selection identified | Final F/O controls are contemporaneous; benchmark reuse still limits confirmation. |
| Survivorship bias | No omitted final/intermediate executions | Exact planned counts and all three capped responses are retained. |
| Look-elsewhere effect | CAUTION for secondary intervals | One prespecified primary contrast; four other call contrasts and timing components are descriptive and unadjusted. No familywise claim is supported. |
| Garden of forking paths | Fixed endpoints and analysis preserved | No checkpoint selection, interim stopping, tuning or outlier removal appears in these records. Historical reuse prevents treating this as fresh confirmation. |
| Correlation versus causation | CAUTION for component attribution | Controlled full-updater endpoints support recipe-level comparisons; arm-specific feedback and changed LR in K_HALF/O do not isolate an individual mechanism. |
| Reverse causality | Not an applicable endpoint ambiguity | Fixed training precedes evaluation. The record review alone does not identify the causal source of training benefits. |

The request bootstrap quantifies uncertainty over this request panel conditional on the trained trajectories; it does not include training-seed uncertainty. Timing reflects checkpoint-ready serving with the required cache handling and three repetitions. It is neither training-inclusive net speedup nor proof of asynchronous deployment advantage. This review authorizes no further experiment or policy change.
