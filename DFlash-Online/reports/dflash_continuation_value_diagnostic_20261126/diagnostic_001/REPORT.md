# Continuation-value information diagnostic

**Completed; decision: INCONCLUSIVE.** The prespecified analysis does **not establish added predictive value** for the richer continuation score beyond the fixed controls. Its adjusted estimate is positive, but the response-bootstrap interval crosses zero. This does not establish absence of information or equivalence. **Keep native C + K+PAIR unchanged; no allocation pilot or training follows automatically.**

Actual native and analysis execution: `b6f2a2c8d487bcc2d32a5e716b870c10eb5067b3`, based on PR63 publication `281516426411ac567129dae7af15ea443a3d0c94`. ARS experiment-agent run + validate; verification status ANALYZED. Source, qualification, raw retrieval, numerical analysis and independent cross-audits passed.

## Question and retained comparison

Does the extent and strength of the parent's consecutive correct followers add information about subsequent useful repair progress, beyond hard run length, immediate-follower confidence, initial error difficulty and prefix fragility?

The fixed transition is **PR59 PAIR1 → PR60 CONTROL**, both native C + K+PAIR: step128/presentation256/event1 → step256/presentation512/event2. PR63 C is the same child, not independent evidence. The endpoints and original training trajectory were reused without modification.

The primary cohort comprises **all 128 retained OpenCode event-2 training-buffer responses**, original source indices128–255. Each uses64 distinct anchors selected by the prescribed SHA256 ranking from its authenticated first-pass presentation. All **8,192 paired block occurrences** remain included: **2,114 HIGH**,4,215 LOW and1,863 no-error; none are empty. Every response contributes4–29 HIGH blocks. These are TRAIN-buffer endpoint observations, not held-out transfer, intermediate-step effects or 8,192 independent trials. The inference unit is the response, conditional on one fixed checkpoint transition.

At each parent HIGH error, the score is `u = sum(running_minimum(FP32 max(0,tanh(margin/2)))) / 14`. Eligibility, roles, weights and reachable endpoint K are fixed from the parent. The signed outcome is `Y = min(child matching-prefix length, K) − parent matching-prefix length`; regressions remain in the analysis. Neither u nor Y is native acceptance, verification calls or serving latency.

## Prespecified result

All estimates below concern matching-prefix positions under fixed training conditioning. Positive coefficients indicate positive association, not the causal effect of increasing loss weight.

| Analysis | Estimate per0.1 increase in u | 95% whole-response bootstrap interval |
|---|---:|---:|
| **Primary: full controls** | **+0.30627** | **[−0.25390,+0.82485]** |
| Without the four immediate-confidence terms | +0.19449 | Not estimated; descriptive secondary fit |
| Unadjusted weighted association | +1.27109 | Not estimated; descriptive only |

Full-model θ is **3.06272** in original u units. One fixed, analysis-weighted SD of baseline-residualized u is0.05343, corresponding to **+0.16364 positions [−0.13566,+0.44071]**. The raw weighted u/Y correlation is0.64763; this raw relationship cannot override the full-model interval.

The full model contains response fixed effects, exact joint(e,r,n) categories, difficulty and prefix-fragility linear/quadratic terms, tie/empty-prefix indicators and all four declared immediate-confidence terms. Each response has total weight1, with native beta weighting within its HIGH blocks. FP64 SVD uses the prospectively fixed relative rank cutoff1e−10 on the weighted design. Continuous scales use parent predictors only and remain fixed for resampling.

All support/inference gates pass: **127 supported response clusters**,1,824 residual degrees of freedom, identifiable θ, and **10,000/10,000 identifiable bootstrap draws**. The model retains152 exact strata,40 with only one occurrence, without merging. Maximum response share of residual-score energy is7.573%. All128 leave-one-response-out coefficients are positive (range1.57275–3.80913); removing one response at a time does not eliminate the wider response-resampling uncertainty. The interval crossing zero determines the inconclusive result.

The bootstrap uses whole responses with multiplicity, PCG64 seed20261126,10,000 fixed draws and linear2.5/97.5 percentiles. All draws used the validated factor/SVD acceleration; first20 were also compared with direct SVD. All prespecified bootstrap draws were retained; no invalid-draw replacement, regularization or model selection occurred. [Model results](MODEL_RESULTS.json), [all bootstrap draws](BOOTSTRAP.json), [influence](INFLUENCE.csv).

## What the continuation channel and outcomes contain

Later bottlenecks exist: **828/2,114 HIGH blocks (39.17%) have D>0**;1,286 have D=0, including all847 one-follower runs. Within6,764 correct-run terms,3,856 are controlled by the immediate follower and2,908 by a later follower;208 terms have tied bottlenecks. Terms beyond the next mismatch contribute zero. This is score variation, not evidence of useful reweighting. All128 selected diagnostic response pools meet the mixed-HIGH/LOW, at-least-two-HIGH and score-variation conditions; this does not reconstruct their historical512-anchor allocations. [Support](CONTINUATION_SUPPORT.json), [coverage](EXPOSURE.csv), [response support](PER_RESPONSE.csv).

| Descriptive primary-HIGH outcome | Count / denominator |
|---|---:|
| Raw old-error repairs |1,799 /2,114 blocks |
| Useful repairs with preceding matches preserved |1,741 /2,114 blocks |
| Blocks losing old prefix agreement |89 /2,114 blocks |
| Positive progress / lost consecutive-prefix occurrences |7,042 /290 |
| Signed progress through frozen K |+6,752 |
| Damaged old-prefix positions, a different count |127 |

Thus58 raw repairs are blocked by earlier damage. The unconstrained child-prefix increase is+10,239 occurrences, but only the fixed-K outcome enters the primary analysis. Descriptive full-model coefficients per0.1u are+0.01978 for useful-repair incidence and−0.000503 for prefix-damage incidence; no secondary confidence intervals or success claims are attached.

The legacy new-buffer set has8responses/64blocks and overlaps the primary cohort:33 useful repairs,+69 signed fixed-K positions. The separate8-response/64-block parent-training set has9 useful repairs and−11 signed positions. Both reproduce old integer summaries; neither is pooled as independent confirmation. [Outcome decomposition](OUTCOME_DECOMPOSITION.csv).

## Correctness, cost and limits

Exact legacy-eight-anchor labels, validity, proposals, roles and retained native margins pass at both endpoints on both A800s. New64-anchor repeated readouts and cross-card comparisons are exact; each primary pair stays on its assigned physical card. Reduced-anchor readouts are not claimed to reproduce historical512-anchor training forwards. Endpoint file hashes, exact export loading, parameter-version/gradient checks, modes, caches and RNG checks pass. No optimizer or trainer was instantiated.

| New work | Completed |
|---|---:|
| Qualification response forwards / head calls |16 /72 |
| Primary response forwards / head calls |256 /2,048 |
| Physical block readouts, including qualification |16,960 |
| Backwards / Adam / teacher / generations |0 /0 /0 /0 |
| Invalid / unknown / unattempted / retries |0 /0 /0 /0 |

Native execution used **367.237 occupied card-seconds (0.1020 card-hours)** and **189.099 elapsed seconds (3m09s)**. Qualification/primary parallel phase walls were81.369/103.100s. Across cards, loading totaled274.256s and readouts72.940s; these overlap across processes and are not additive wall time. Compiler timers are nested. Peak memory was9.383GB allocated/9.624GB reserved. Full CPU extraction/analysis/public-table writing took94.760s, including89.716s for the bootstrap. File authentication/staging were CPU-only; their complete wall duration was not separately measured.

Both GPUs are idle. Postflight A800 free space is405.510GB, preserving400GB. Measured remote source/run/cache allocation is 0.477 GB; actual local study allocation is 0.590 GB. Including a 10 MB publication/Git allowance, the all-location charge is 1.077 GB, below the 2 GB cap. The prelaunch 750 MB local envelope remains conservative. No inherited file was deleted or new model/feature tensor copied. Final actual all-location accounting is in [WORK_AND_RESOURCES.json](WORK_AND_RESOURCES.json).

All49 implementation CPU tests pass locally and on A800. Cross-audits verify21,973 work/provenance checks,3,369,528 metric checks and25,372 statistical checks; these are overlapping checks, not new samples. One CPU audit checker initially mishandled a source-manifest symlink and was repaired with its failure retained. No scientific execution was retried. [Audit scope and limitations](AUDIT.md).

This is one retained training transition with overlapping sampled blocks and a specified linear adjustment model. Polynomial controls do not eliminate every nonlinear relationship or unmeasured confounder. Even a positive association would not establish an optimal allocation law, causal training benefit or serving improvement. Here the primary interval remains unresolved, so the diagnostic supplies **no demonstrated basis for changing the working method**.
