# Full GSM8K K/K_HALF online-training comparison

Execution source: `0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34`.

## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-02T06:10:13.219265+00:00
- Verification Status: ANALYZED
- Version Label: full_k_gsm8k_report_v1

CPU record reconstruction and statistical analysis; not an independent GPU reproduction or reinspection of retired model/feature tensors. Separate raw-evaluation/statistical and raw-training/accounting reviews are delivered in `reviews/`.

**Completed:** the fixed campaign exited successfully at 06:10:50 UTC on November 2, 2026 (11:10:50 p.m. Edmonton time on November 1), after 20 h 27 m 27 s including wrapper checks. Both new arms completed all 58 events. All 12,288 final and 768 intermediate evaluation executions are valid. No new experiment was launched for this report.

The fixed protocol trains only K and K_HALF, each from released DFlash with fresh seed-17 Adam state. Each is assigned its own original 7,473-request stream, 58 updates and 7,424 Adam steps, with the final 49 requests served without another update. F and completed O58 are retained controls. This is an arm-specific online-feedback comparison; old O feedback features were not reconstructed.

The target is pinned Qwen3-8B and the draft is Qwen3-8B-DFlash-b16. K uses constant LR 3e-4; K_HALF uses LR 1.5e-4. Both retain the qualified current-prefix-plus-first-error KL objective, selected-weight normalization, BF16 AdamW, response-only supervision, two passes, microbatch 1 and accumulation 2. Greedy block-16 serving uses 2,048 tokens or EOS, context 4,096, concurrency 1 and full per-request cache flushing.

**The primary K_HALF/O endpoint is positive; same-LR K/O shows the larger gain.** K uses 3.00% fewer verification calls and 2.19% less request time than O. K_HALF uses 1.68% fewer calls and 1.37% less request time. K_HALF does not outperform K in this full GSM8K run.

Primary: K_HALF58 versus O58. This compares different learning recipes; K58/O58 is the same-LR full-updater control. The benchmark was used previously and only seed 17 is fitted here.

| Method | Calls | Output tokens | Calls / 1,000 tokens | Accepted tokens | Progress / call |
|---|---:|---:|---:|---:|---:|
| F | 50584 | 314847 | 160.6622 | 264170 | 6.2040 |
| O | 44730 | 314847 | 142.0690 | 270037 | 7.0159 |
| K | 43387 | 314847 | 137.8034 | 271387 | 7.2331 |
| K_HALF | 43979 | 314847 | 139.6837 | 270814 | 7.1357 |

| Comparison | Call reduction | Paired 95% interval | Wins / regressions / ties | Gross saved / added |
|---|---:|---:|---:|---:|---:|
| K_HALF/O | +1.6790% | [+1.2908, +2.0674]% | 523 / 345 / 156 | 1512 / 761 |
| K/O | +3.0025% | [+2.6667, +3.3457]% | 630 / 241 / 153 | 1800 / 457 |
| K_HALF/K | -1.3645% | [-1.7575, -0.9866]% | 334 / 535 / 155 | 766 / 1358 |
| K_HALF/F | +13.0575% | [+12.5951, +13.5239]% | 993 / 14 / 17 | 6638 / 33 |
| K/F | +14.2278% | [+13.6878, +14.7639]% | 982 / 18 / 24 | 7230 / 33 |

| Comparison | Complete request-time reduction | Paired 95% interval | Native-generation reduction | Cache-time reduction |
|---|---:|---:|---:|---:|
| K_HALF/O | +1.3703% | [+0.6749, +2.0624]% | +1.2730% | +2.5453% |
| K/O | +2.1886% | [+1.4693, +2.8760]% | +2.7388% | -11.3971% |
| K_HALF/K | -0.8366% | [-1.4969, -0.2026]% | -1.5071% | +12.5159% |
| K_HALF/F | +12.2789% | [+11.5071, +13.0387]% | +12.5269% | +2.2115% |
| K/F | +13.0067% | [+12.1917, +13.8146]% | +13.8256% | -11.7786% |

Calls use one repeat-consistent trajectory per request; timing uses three-repeat request means. All ratios divide summed measurements. The 10,000 paired-request bootstrap uses PCG64 seed 20261102 and linear percentile endpoints; its unit is one of 1,024 requests carrying all methods, not a timing repeat. Intervals are conditional on this fitted seed and previously used benchmark. Secondary intervals are descriptive/unadjusted. No equivalence threshold was introduced.

| Repeat | F request seconds | O request seconds | K request seconds | K_HALF request seconds |
|---|---:|---:|---:|---:|
| 1 | 911.059 | 812.889 | 791.559 | 801.473 |
| 2 | 912.238 | 810.081 | 795.574 | 797.583 |
| 3 | 911.228 | 809.112 | 791.722 | 799.700 |

| Method versus F | Gross gain calls | Gross harm calls | Net saved calls |
|---|---:|---:|---:|
| O | 5962 | 108 | 5854 |
| K | 7230 | 33 | 7197 |
| K_HALF | 6638 | 33 | 6605 |

Gross gains and harms retain request identities. They describe useful and harmful endpoint changes, not a unique cause such as memorization or gradient conflict.

## Average acceptance and output progress

Both columns divide total tokens by total verification calls across the 1,024 requests, using one repeat-consistent trajectory per request. Parentheses give the increase over freshly measured frozen DFlash.

| Method | Accepted draft tokens / verification | Committed output progress / verification |
|---|---:|---:|
| F | 5.2224 (baseline) | 6.2040 (baseline) |
| O | 6.0370 (+15.60%) | 7.0159 (+13.09%) |
| K | 6.2550 (+19.77%) | 7.2331 (+16.59%) |
| K_HALF | 6.1578 (+17.91%) | 7.1357 (+15.02%) |

Accepted draft tokens exclude verifier-provided correction/bonus tokens. Committed progress includes those tokens when committed and excludes the first output token produced during prefill. EOS/cap clipping means the columns need not differ by exactly one. These increases are not latency reductions. The formulas and native count definitions remain unchanged.

## Fixed learning-curve observations

| Arm | Event | Requests | Calls | Retained F calls | Reduction vs F | Wins / regressions / ties |
|---|---:|---:|---:|---:|---:|---:|
| K | 14 | 128 | 5405 | 6046 | +10.6021% | 115 / 6 / 7 |
| K | 29 | 128 | 5274 | 6046 | +12.7688% | 117 / 6 / 5 |
| K | 44 | 128 | 5164 | 6046 | +14.5882% | 124 / 4 / 0 |
| K | 58 | 128 | 5123 | 6046 | +15.2663% | 120 / 4 / 4 |
| K_HALF | 14 | 128 | 5429 | 6046 | +10.2051% | 119 / 2 / 7 |
| K_HALF | 29 | 128 | 5300 | 6046 | +12.3387% | 125 / 2 / 1 |
| K_HALF | 44 | 128 | 5270 | 6046 | +12.8349% | 123 / 2 / 3 |
| K_HALF | 58 | 128 | 5224 | 6046 | +13.5958% | 125 / 1 / 2 |

Events 14/29/44 use the fixed first 128 requests once. Event 58 in this table uses those same 128 requests from the final repeat-zero records, with no extra generation. The curve uses retained F count context; the full 1,024-request final comparisons above use fresh F. These observations did not select checkpoints or stop training. No matched intermediate O measurements are supplied, so this is not evidence of faster learning than O.

## Work, exposure and costs

All 116 training events, 14,848 Adam steps, 29,696 response presentations, 14,946 online responses, 14,848 terminal-feature preparations, 768 intermediate evaluations, 12,288 final evaluations and eight inherited correctness generations are separately accounted for. Complete generations total 28,010. No disposable Adam qualification step or new efficacy warmup is included.

| Arm | Training seconds | Blocking updates seconds | Controller serving seconds | Loading seconds | Online wall seconds |
|---|---:|---:|---:|---:|---:|
| K | 18707.203 | 21735.125 | 7204.021 | 42.121 | 31423.154 |
| K_HALF | 18672.490 | 21712.566 | 7267.020 | 40.381 | 31417.314 |

Teacher preparation, capture, feature persistence, optimizer work, checkpoint/export, publication and retention costs are retained separately in COSTS.json. Training/preparation/persistence/publication are nested within blocking updates; capture is nested within serving. Do not add nested timers twice. Actual anchor and supervised-position occurrence exposure appears in EXPOSURE.json; repeated positions are exposure, not unique tokens. F/O58 historical adaptation costs are not fresh matched costs.

| Arm | Update-window peak allocated GiB | Update-window peak reserved GiB |
|---|---:|---:|
| K | 19.255 | 22.400 |
| K_HALF | 19.255 | 22.400 |

These are trainer-process CUDA peaks, reset before each controller update and measured through preparation, training, checkpoint/export and publication. They exclude initial loading, subsequent boundary validation and the separate serving worker; they are not whole-system memory peaks.

RESOURCES.json preserves each event and compiler timer by name. TorchDynamo timers can nest: do not sum different timer names or add them to training/update wall time. Serving-worker compilation is included in loading/serving but is not separately identifiable. Final worker loading and campaign wall time are separate fields in COSTS.json, when their receipts are available.

Final request timing is contemporaneous checkpoint-ready serving. It does not establish net asynchronous deployment or training-inclusive speedup. Aggregate timers are not a kernel-time breakdown.

## Actual supervised exposure

| Arm | Sampled anchor occurrences | Available position occurrences | Selected supervised occurrences | Selected fraction |
|---|---:|---:|---:|---:|
| K | 3,728,050 | 54,368,391 | 33,320,668 | 61.29% |
| K_HALF | 3,728,050 | 54,368,391 | 32,063,446 | 58.97% |

These are occurrences across presentations and overlapping sampled blocks, not unique response tokens. Both arms generate their own feedback and select loss positions from their own current predictions. Equal aggregate anchor counts do not turn this into a shared-feedback ablation. [Event exposure](EVENT_EXPOSURE.csv) retains all 116 events, including loss numerators/denominators and actual forward/chunk/layer counts. [Event costs](EVENT_COSTS.csv) retains every blocking update and named component; nested timers must not be summed twice.

## Execution validity, retention and review

The fixed work reconciles to 28,010 complete generations, 14,848 Adam steps and 29,696 ordinary response forwards/backwards. Every final output and termination agrees with its canonical reference; within-method call counts repeat exactly. Final timing uses the sealed 64-request method-block schedule: each method occupies each position 12 times and each pair has 24/24 precedence. Intermediate evaluation preserves the live trainer state/RNG and adds no optimizer work.

Archived-file transfer overlapped training/capture under the user's operational amendment, so adaptation timings may include I/O contention. Final evaluation began only after both training workers exited and transfers were verified stopped; the separate transfer-clearance wait was 90.002 s. A transfer-resume failure occurred after successful experiment exit and affected only unfinished historical archiving. It neither changed scientific settings nor overlapped final timed serving.

Both initial complete states, both final complete states and both final serving exports are retained. Execution-time live/persisted/export equality checks passed; this review checked their receipts, existence and final sizes/inodes without reloading or rehashing the large tensors. No interrupted-and-resumed scientific segment occurred, so saved recovery checks are not an exercised restart test. Exactly 44,772 new bulk paths were retired after successor validation: per arm, 7,424 native-capture tensors, 7,424 prepared-feature tensors, 7,424 terminal-feature tensors, 57 prior export weights and 57 prior optimizer/RNG states; they were not archived and no longer support tensor-level replay at those boundaries. Raw compact event/evaluation/native records, anchors, validity/failure receipts and source/configuration provenance remain. The separate same-team reviewers independently reconstructed records with shared task context; they were not blind or cross-model reviewers. Their audits passed 4,349,703 evaluation/statistical checks and 1,849,884 training/accounting checks, respectively, with no scientific discrepancies. All 11 statistical-fallacy categories were assessed. These check counts measure audit coverage, not statistical certainty. The reviewers did not recreate deleted tensors or rerun a GPU experiment. See [handoff coverage](HANDOFF_COVERAGE.md), [execution and retained artifacts](EXECUTION_AND_PROVENANCE.md), and the two record reviews linked below.

## One next recommendation

Propose one separately approved second-training-seed confirmation of the same-LR K versus O comparison. K is the stronger mathematics endpoint here, while the present request-bootstrap intervals do not measure training-seed robustness. No such run is launched by this report. `SUMMARY.json` preserves the original generated interpretation unchanged; this final reviewed recommendation supersedes its generic next-action wording without changing any measurement.

This recommendation authorizes no further event, tuning, panel expansion or new mechanism. A positive endpoint is not training-seed robustness, convergence or conference novelty. Valid negative and inconclusive outcomes complete the fixed experiment.

Files: [independent evaluation/statistics review](reviews/RAW_STATISTICS_REVIEW.md), [independent training/accounting review](reviews/TRAINING_COST_REVIEW.md), [summary](SUMMARY.json), [paired requests](PAIRED_REQUESTS.csv), [all evaluation records](ALL_EVALUATIONS.csv), [learning curve](LEARNING_CURVE.csv), [raw-evidence audit](EVALUATION_AUDIT.json), [work accounting](ACCOUNTING.json), [costs](COSTS.json), [resources](RESOURCES.json), [exposure](EXPOSURE.json).
