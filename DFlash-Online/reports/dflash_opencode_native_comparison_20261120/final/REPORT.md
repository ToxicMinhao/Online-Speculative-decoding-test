Execution commit: `1551e5e5d0dd7736d8441977aa3dea250535da7b`. Report revision is recorded separately by its Git commit.

Status: **passed**. This report adds no model executions.

An explicitly authorized startup retry retains 56.0318 prior occupied-card seconds in the original clock. The failed AR piecewise-graph initialization performed additional uninstrumented forwards; their exact count remains unknown. No scientific generation or optimizer step executed before that retry. The original failure evidence is preserved.

The original AR/speculative equality check failed and remains retained as a historical failure. The user subsequently authorized separate AR and frozen-F canonical references. Only the remaining qualification work was resumed under that explicit output-policy amendment; the prior unknown execution and consumed time remain in the accounting. The original cross-path equality criterion is not retrospectively reported as passed.

The user authorized 96 elapsed hours and 120 occupied GPU-card-hours, both on the original clock. Passed qualification and all 2,048 precheck generations from `658d734c422a1a200fe400dceb256492f529204e` were reused without repeating model work. Production uses the execution commit above; the source review verifies unchanged scientific code and settings. The earlier budget stop, failure evidence and consumed costs remain retained.

C reduces complete-request latency against AREPO on this fixed CODE panel. This supports the relative endpoint comparison in this one-seed setting, not an isolated loss-component explanation.

Primary C/AREPO complete-request time reduction: **1.3939% (95% CI 1.0667, 1.7186)**.

| Method | Calls | Native accepted/call | Committed/call | Complete time (s) | Native time (s) | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| O_CODE58 | 58112 | 6.07011 | 7.07011 | 1042.219 | 1009.161 | 395.197 |
| AREPO_CODE58 | 57074 | 6.19869 | 7.19869 | 1022.148 | 991.089 | 402.957 |
| C_CODE58 | 56302 | 6.29740 | 7.29740 | 1007.901 | 977.180 | 408.653 |
| F | 65306 | 5.29127 | 6.29127 | 1164.198 | 1129.039 | 353.790 |
| AR | N/A | N/A | N/A | 4990.564 | 4973.609 | 83.322 |

| Contrast | Complete-request reduction | Call reduction | Wins / regressions / ties |
|---|---:|---:|---:|
| C_CODE58_vs_AREPO_CODE58 | 1.3939% (95% CI 1.0667, 1.7186) | 1.3526% (95% CI 1.0272, 1.6785) | 530 / 327 / 167 |
| C_CODE58_vs_O_CODE58 | 3.2928% (95% CI 2.9794, 3.6117) | 3.1147% (95% CI 2.8050, 3.4241) | 664 / 224 / 136 |
| AREPO_CODE58_vs_O_CODE58 | 1.9257% (95% CI 1.6404, 2.2058) | 1.7862% (95% CI 1.5133, 2.0565) | 595 / 271 / 158 |
| O_CODE58_vs_F | 10.4775% (95% CI 9.8676, 11.1023) | 11.0158% (95% CI 10.3977, 11.6463) | 908 / 76 / 40 |
| AREPO_CODE58_vs_F | 12.2015% (95% CI 11.5954, 12.8311) | 12.6053% (95% CI 11.9889, 13.2409) | 942 / 53 / 29 |
| C_CODE58_vs_F | 13.4253% (95% CI 12.8568, 14.0180) | 13.7874% (95% CI 13.2037, 14.3889) | 967 / 33 / 24 |
| O_CODE58_vs_AR (descriptive AR comparison) | 79.1162% (95% CI 78.6808, 79.5501) | N/A | N/A |
| AREPO_CODE58_vs_AR (descriptive AR comparison) | 79.5184% (95% CI 79.0887, 79.9475) | N/A | N/A |
| C_CODE58_vs_AR (descriptive AR comparison) | 79.8039% (95% CI 79.3888, 80.2238) | N/A | N/A |
| F_vs_AR (descriptive AR comparison) | 76.6720% (95% CI 76.2326, 77.1200) | N/A | N/A |

AR and frozen F agree on tokens and termination for 286/1024 requests. Differences remain for 738 requests: token sequences 738, termination 4, and output length 669. Every AR repeat matches its own AR canonical; every F/O/AREPO/C execution matches the frozen-F canonical. The user-authorized numerical contract permits cross-path output differences only. The full-panel AR timing rows are descriptive comparisons of the actual native outputs; they are not identical-output speedup claims. AR_SPEC_OUTPUT_COMPARISONS.csv preserves every request, including zero-based first-difference offsets.

| Endpoint versus AR | Identical-output requests | Identical-output subset speedup |
|---|---:|---:|
| O_CODE58 | 286 / 1024 | 5.1592× (95% CI 5.0085, 5.3242) |
| AREPO_CODE58 | 286 / 1024 | 5.2751× (95% CI 5.1213, 5.4440) |
| C_CODE58 | 286 / 1024 | 5.3238× (95% CI 5.1740, 5.4876) |
| F | 286 / 1024 | 4.5443× (95% CI 4.4193, 4.6878) |

These subset ratios exclude all differing-output requests; full-panel O/AREPO/C/F contrasts retain every request. Subset membership is determined only by AR/F token and termination agreement, so it is a descriptive subset rather than the original full-panel estimand. All repeats have consistent native counts. The independent reducer reproduces the request means, summed ratios, 10,000 paired bootstrap draws and intervals. Secondary intervals are descriptive and unadjusted.

The custom panel comes from the prospectively bounded OpenCodeInstruct reservoir, not the original author 1,024-row slice. All three online arms start from release with seed 17 and use their own feedback. Training context is 4,096 for this study; the output cap remains 2,048. Freshness is relative to the authenticated inventory, with historical coverage limits disclosed in READINESS.md.

Counts use one validated trajectory per request; three repetitions estimate timing, not independent samples. Native acceptance is progress minus verification calls; stop-clipped matched drafts are counted directly from the trace. The initial prefill token is excluded from committed progress. Zero-call requests remain in output and latency totals. AR speculative counts are N/A.

Training cost fields are individually preserved in TRAINING_COSTS.csv. Nested timers overlap and must not be summed arbitrarily. Own-feedback curves are descriptive pre-update trajectories, not controlled endpoint comparisons. The event and engine costs include initialization/publication work separately from scored request latency; no kernel-time breakdown is inferred from aggregate latency.

Endpoint serving performance does not establish training-inclusive or asynchronous throughput. AREPO implements the released loss in the common framework, not the complete AdaFlash concurrent-serving system. One trained seed per arm does not establish seed robustness. No result triggers tuning or an additional run.

Audit scope: independent CPU reconstruction of retained raw evidence by a collaborating agent with shared project context. This is not independent GPU replication. Private prompts, generated tokens and native traces remain outside the publication directory.
