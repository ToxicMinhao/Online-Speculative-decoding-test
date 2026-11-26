# Independent serving audit

## Material Passport

- Task: Point3 cumulative-maximum assessment, retained-result validation.
- Verification status: ANALYZED; independent CPU statistical recomputation passed. No new GPU replication.
- Actual execution commit: `6f7f3a5d1769a630e4ec852205eb06ca55047341`.
- Data: 128 paired reused development requests; three timing repeats; three fixed endpoints.
- Public identities: fresh 1–128 request ordinals and 1–2 warmup ordinals only.

## Results

Positive reductions favor the candidate. Timing uses summed per-request means; call totals use one validated trajectory per request.

| Candidate / reference | Complete latency reduction, % (95% CI) | Call reduction, % (95% CI) | Call wins / regressions / ties | Gross calls saved / added |
|---|---:|---:|---:|---:|
| F_C_MAX / F_C_SUM | 0.389084 (-0.520682, 1.298616) | -0.026199 (-0.425882, 0.360460) | 42 / 42 / 44 | 62 / 64 |
| F_C_SUM / C | -0.744147 (-2.372349, 0.775624) | 0.065454 (-0.435564, 0.554723) | 51 / 49 / 28 | 86 / 81 |
| F_C_MAX / C | -0.352167 (-1.765539, 0.985743) | 0.039272 (-0.436860, 0.496931) | 49 / 44 / 35 | 77 / 74 |

The prespecified unrounded selection chooses **C**. This is a development-selection rule, not an equivalence test or a claim that all other settings fail.

## Recomputed evidence and limits

The independent script imports no production experiment modules. It reconstructs each request's three-repeat means, one-trajectory counts, acceptance/progress identities, all three contrasts, repeat totals and the selection. The paired bootstrap uses 10,000 shared PCG64 resamples, seed 20261125, and linear 2.5/97.5 percentiles. The draw digest is `f6306d2449d775e38609cb8ba45833ca588815736e0605bed18ebaae5c8659a7`.

The script verifies retained scalar calculations, not unsaved training-vector contractions. Exact token/termination checks, native qualification, checkpoint authentication, physical-card isolation, and work/resource accounting are covered by the companion evidence audit. Export identity status for this statistical audit: `independently_verified_cpu_postflight`. The independent CPU postflight parsed the stored safetensors headers and native tensor bytes, without loading a model framework or using a GPU. All three parameter identities match; the exports are distinct. This authenticates stored endpoints, not independent training re-execution.

No p-values were requested or computed. The MAX-versus-SUM complete-request latency interval is the primary component contrast. All 12 displayed timing/count intervals are retained; the nonprimary intervals are descriptive and unadjusted. The three whole-request bootstrap pairs are not independent replications. This reused panel and one inherited training trajectory do not establish fresh-panel, seed, feedback-window or domain robustness. Serving differences do not establish training-inclusive speedup. Lower MAX than SUM at the same endpoint follows from construction and is not evidence of learning benefit.

## Statistical fallacy checks — 11/11

| Check | Finding / interpretation limit |
|---|---|
| 1. Simpson's paradox | Descriptive scheduling-group direction check, without changing pooling or selection: F_C_MAX/F_C_SUM: group 0 +0.463% latency / -0.098% calls, group 1 +0.305% latency / +0.056% calls; F_C_SUM/C: group 0 -1.648% latency / +0.147% calls, group 1 +0.268% latency / -0.028% calls; F_C_MAX/C: group 0 -1.177% latency / +0.049% calls, group 1 +0.572% latency / +0.028% calls. Mixed subgroup signs are not a reason to drop requests. |
| 2. Ecological fallacy | Aggregate ratios are accompanied by all request-level outcomes; no claim that every request improves. |
| 3. Berkson's paradox | This is a sealed, previously used development panel. No outcome-based request filtering occurred; population generalization remains limited. |
| 4. Collider bias | No post-treatment covariate adjustment or favorable-output subgroup selection enters the primary result. Every scored execution passed the fixed validity gates. |
| 5. Base-rate neglect | Report 128 request denominators, wins/regressions/ties, gross calls saved/added, output counts and 1,152 scored executions plus 36 warmups. |
| 6. Regression to the mean | All endpoints are evaluated contemporaneously on the same requests; historical serving times are not used as a learning curve. Reused-development selection still limits confirmation. |
| 7. Survivorship bias | The planned 1,188 returned measurements are retained. No failed, unfavorable or slow request is excluded from the reported paired panel. |
| 8. Look-elsewhere effect | All three prespecified contrasts and timing/count intervals are reported, including negative estimates. Secondary intervals are unadjusted; broad mechanism-search history is not fresh confirmatory evidence. |
| 9. Garden of forking paths | Fixed arms, panel, repeat pooling, seed, bootstrap, precision and unrounded freeze rule are retained. No post-result tuning or alternative request selection was run. |
| 10. Correlation versus causation | The paired arm contrast measures these complete implementations under matched conditions; local training-assessment changes are not asserted to cause a percentage of serving gains. No isolated mechanism or deployment-wide causal claim is made. |
| 11. Reverse causality | The checkpoints were fixed before serving evaluation, and evaluation results never trained them. The endpoint comparison has temporal ordering; local metric associations do not independently establish mechanism. |

Confidence classification: **CAUTION** for generalization and multiple comparisons. The arithmetic is independently checked; a development point estimate or an interval crossing zero is not equivalence, noninferiority, absence of harm, or proof of mechanism failure.

## Table conventions

`PER_REQUEST.csv` contains 128 paired rows; `ALL_REPEATS.csv` contains all 1,152 scored rows; `WARMUPS.csv` contains all 36 included warmups. `ARM_TOTALS.csv`, `REPEAT_TOTALS.csv` and `CONTRASTS.csv` provide aggregate and three-repeat accounting. `GROUP_DIRECTION_CHECK.csv` is a descriptive audit of the two scheduling groups. Public ordinals follow first scheduled appearance. The audit JSON's `bootstrap.public_request_order` supplies the exact ordinal permutation needed to reproduce the intervals from these public tables. Repeat, group and activation columns remain zero-based. Native and stop-clipped accepted tokens remain separate. Loading, activation and warmups do not enter primary serving latency.
