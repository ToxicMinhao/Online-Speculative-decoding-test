## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Origin mode: validate
- Origin date: 2026-11-19
- Verification status: VERIFIED for scalar reduction and paired-bootstrap recomputation only; no native/model replay
- Version label: PR54 final-primary-review v1
- Execution commit: `b4459820b3822c3d760bb3fd880e4d139fa2bbc2`
- Source: authenticated retained PR54 endpoint summaries and scalar measurement records
- Scope: CPU-only statistical validation; no GPU execution, model work, additional endpoint samples, or second independent native-token audit

## Conclusion

C58 has a small demonstrated reduction in verification calls relative to native K+58 on each fixed panel under the prespecified marginal paired-bootstrap analysis. MathQA also has a small demonstrated serving-time reduction. GSM8K's timing interval spans zero, so its time benefit remains unresolved. This updates the earlier interim statement, which had only point estimates.

| C58 versus K+58 | GSM8K | MathQA |
|---|---:|---:|
| Verification-call reduction | 0.31664%, 95% interval [0.05007%, 0.58292%] | 0.29536%, [0.09995%, 0.49243%] |
| Native accepted proposals/call, K+58 → C58 | 6.253172 → 6.276211 | 6.086323 → 6.107315 |
| Relative native acceptance increase | 0.36844%, [0.05791%, 0.68058%] | 0.34490%, [0.11650%, 0.57615%] |
| Request-time reduction | 0.28353%, [−0.25287%, 0.82400%] | 0.29850%, [0.09764%, 0.49908%] |
| Request wins / regressions / ties | 416 / 366 / 242 | 430 / 370 / 224 |
| Gross calls saved / added | 780 / 643 | 974 / 758 |
| Net calls saved | 137 | 216 |

These gains are small in practical magnitude. Both panels were reused and only one seed-17 fitted training trajectory was examined. The intervals are conditional on these fitted endpoints and panels; they do not include variation from rerunning training. They do not establish a universal benefit, training-inclusive speedup, or a particular internal mechanism. Native acceptance and verification-call effects are mathematically coupled under identical outputs, so they are not independent confirmations.

## Statistical contract checked

Counts use one trajectory per request after checking three-repeat consistency. Timing uses each request's three-repeat mean, then sums across requests. For C versus K, call reduction is `1 - sum(C_calls) / sum(K_calls)`; time reduction uses the analogous ratio of summed mean times. Native acceptance is total native accepted proposals divided by total verification calls. Stop-clipped matched acceptance is a distinct metric and is not substituted for the native counter.

The bootstrap resamples 1,024 whole paired requests 10,000 times, preserving all arms and repeat timings together. Request IDs are in lexicographic order. The prespecified NumPy PCG64 seeds are 20261118 for GSM8K and 20261119 for MathQA. Marginal 95% percentile intervals use linear quantiles. No p-values or family-wise significance claim are added. No inference that one panel's effect is larger than the other's follows from comparing separate interval labels.

## ARS interpretation review: 11/11 categories

| Category | Assessment |
|---|---|
| 1. Simpson's paradox | Datasets remain separate; no pooled cross-dataset headline. Repeat timing points all favor C on each dataset, but individual request regressions remain visible. No universal per-request advantage is inferred. |
| 2. Ecological fallacy | The pooled ratios describe total workload, not improvement for every request. Wins, regressions, ties, and their 1,024-request denominators are reported. |
| 3. Berkson's paradox | These are fixed reused benchmark panels rather than a population sample. Conclusions remain conditional; no selection by current outputs or gains was introduced by this review. Historical benchmark selection limits external validity. |
| 4. Collider bias | No conditioning on post-treatment success, generated length, or observed gain is introduced. All prespecified valid requests remain included; validity evidence is retained separately. |
| 5. Base-rate neglect | Full request counts, wins/regressions/ties and gross saved/added calls are shown. No rare-event conditional rate is presented without its denominator. |
| 6. Regression to the mean | The comparison uses contemporaneous fixed C/K endpoints on the same panel, rather than selecting the worst historical requests. Prior favorable small screens and reused inputs limit confirmation claims; they are not evidence of independent replication. |
| 7. Survivorship bias | The retained completed evaluation reports all 9,504 executions per dataset: 9,216 scored and 288 warmups, with no invalid/unknown/unattempted endpoint work. Earlier startup failures remain separate and are not erased. Scalar review verifies receipt completeness; it does not recreate the raw token audit. |
| 8. Look-elsewhere effect | The report retains prespecified C/K, C/O and K/O comparisons, both datasets and unresolved outcomes. This review focuses on primary C/K. Intervals are marginal, without family-wise correction; no experiment-wide or across-project significance claim is made. |
| 9. Garden of forking paths | Exact frozen seeds, ordering, paired units, metrics and quantile method are reused. No tolerance change, request exclusion, alternative bootstrap selection or post-result tuning occurred. Startup/code fixes were correctness repairs with historical failures preserved. |
| 10. Correlation versus causation | The native K+ control tests the complete added-allocation recipe under the retained training/serving setup. C/O is the whole recipe comparison. Neither conditional endpoint result proves the hypothesized internal gradient mechanism, independent-data robustness, or training-inclusive deployment gain. |
| 11. Reverse causality | Training endpoints precede evaluation and evaluation requests never become training feedback in this run. Temporal direction is defined; there is no reverse-causation claim to infer from the endpoint associations. |

Overall interpretation confidence: **CAUTION** because the endpoint gains are small, panels are reused, training-seed robustness is untested, and multiple reported intervals are marginal. This label does not negate positive fixed-panel call intervals. A GSM8K time interval crossing zero is unresolved, not evidence of equivalence or preservation. No automatic algorithm change or follow-up experiment is recommended or authorized by this audit.

## Reproducibility boundary

The retained private audit script independently constructs paired arrays from retained `MEASUREMENTS.jsonl`, verifies file identities against the retrieved remote identity receipt, checks all repeats and request denominators, and recomputes the four primary C/K interval metrics without importing production analysis helpers. Both datasets passed: all four point estimates and 95% interval endpoints exactly match the retained summaries, and request-order and bootstrap draw-matrix identities match. [INDEPENDENT_PRIMARY_BOOTSTRAP.json](INDEPENDENT_PRIMARY_BOOTSTRAP.json) records the results. The public [verify_published_tables.py](verify_published_tables.py) separately reproduces the primary metrics from the numeric CSV tables. This does not rerun training/serving, authenticate model weights independently, or repeat the native raw-token audit; those remain supported by the retained experiment validity receipts.
