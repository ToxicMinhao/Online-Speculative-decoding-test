# Independent final R campaign audit

**Audit passed with zero issues. R did not demonstrate a useful advantage over ordinary OSD or P in this one-event screen.** The complete machine-readable result is [FINAL_AUDIT.json](FINAL_AUDIT.json). This is an analysis of retained execution evidence, not a new GPU/model reproduction.

All 768 fresh F/O/P/R generations are present in the prescribed rotation, with all 64 requests in the common valid cohort. Independent reconstruction of 37,077 native blocks checks longest accepted prefixes, EOS/cap clipping, committed tokens, positions and scalar totals. Every output and termination matches the saved canonical reference. Raw identities, capture-disabled traces, four native tensor-equality checks, 705 publication receipts and the embedded 96-generation qualification gate pass. There are no unknown or unattempted generations.

| Arm | Verification calls | Sum of mean request times (s) |
|---|---:|---:|
| F | 3,140 | 54.309900 |
| O | 3,075 | 53.059998 |
| P | 3,069 | 53.044502 |
| R | 3,075 | 53.087305 |

Counts use one validated repeat-0 trajectory per request; time averages the three fresh repeats. Every arm emits the same 19,547 output tokens across the 64 requests. Comparisons are ratios of pooled sums, not averages of request percentages.

| Comparison | Calls saved | Call reduction | Requests improved / regressed / tied | Request-time reduction |
|---|---:|---:|---:|---:|
| R/O, primary | 0 | 0.000000% | 25 / 25 / 14 | −0.051465% |
| R/P | −6 | −0.195503% | 15 / 17 / 32 | −0.080693% |
| R/F | 65 | 2.070064% | 38 / 18 / 8 | 2.251144% |

R/O's equal aggregate calls hide 50 requests with different verification-call counts: 40 calls saved and 40 added. R takes 0.027307 s more in the sum of mean request times; all three repeat totals are slightly slower. R/P adds six calls and 0.042803 s. These small timing differences are descriptive; no statistical significance is asserted. R/F has exactly the aggregate call benefit already attained by ordinary O, so it supplies no incremental R advantage.

All fresh F/O counts reproduce the original first-event control. A separate [control reproduction check](RETAINED_CONTROL_REPRODUCTION.json) confirms all 576 F/O/P rows match the previous campaign on calls, accepted draft tokens, committed progress and output tokens. The independent second reviewer also checked all 192 P rows. Historical timings are not pooled with current timings.

R completed exactly 128 optimizer steps and 256 presentations. The retained 128-response feature pool, two-pass order, anchors, update boundaries and learning rates match O. Direct anchor reconstruction gives 65,096 sampled anchors and 949,560 supervised position occurrences. Fresh released initialization and clean trainer closure pass. The earlier [qualification audit](QUALIFICATION_AUDIT.md) remains valid: five disposable steps, ten presentations, four reference and four detached comparison forwards, twenty detached-logit gradient calls, eleven qualification equality receipts and one fresh-main initial-state receipt.

Boundary-only diagnostic aggregation gives 241,673 active protected occurrences / 429,510 protected occurrences (**56.2671%**), and 38,062 reference-correct/current-incorrect occurrences / 545,042 supervised reference-correct occurrences (**6.9833%**). The latter includes eligible matches after prefix failure. The auxiliary numerator is positive in 127 of 128 update windows. Pooled weighted KL and auxiliary diagnostics are 0.8900274131 and 0.0263597608; optimization remains response mean. These are sampled training occurrences, not unique examples or final-checkpoint preservation rates. They show that the penalty was exercised; absent matched O/P regression diagnostics, they do not establish reduced regression relative to either comparator.

R's complete event costs **419.579671 s**: 383.806407 s training, 17.652472 s checkpoint persistence and 15.474607 s export, plus event overhead. Reference work (66.828352 s) and teacher-head work (103.769067 s) are nested in the event. Qualification costs 366.870457 s; R loading and initialization control cost 19.246277 s and 8.078538 s; native loading costs 54.016945 s; evaluation costs 1,836.384784 s. Disjoint phases sum to 2,704.176672 s, within the 2,710.198808 s campaign wall time. Activation and weight validation are nested in evaluation. Serving comparisons exclude adaptation and do not establish net online acceleration. Feedback has zero new collection/preparation cost, not zero historical cost.

[Postflight](raw/POSTFLIGHT.json) reports unchanged contents for all 110 source files, nine execution files, 32 dependencies and 140 retained artifacts, with supervisor exit zero. The auditor checks those coverage counts against the seal and retention manifest and binds the O/P receipts and qualification inputs. [Retrieval verification](RETRIEVAL_AUDIT.json) passes for all 2,386 compact files / 176,302,541 bytes. Tensor-state equality is established here through the retained exact-comparison receipts; the auditor does not reload tensor snapshots. No source was changed or experiment rerun during this review.

ARS interpretation coverage: all 11 statistical-fallacy patterns were reviewed. The fixed cohort/analysis and complete negative comparisons avoid outcome-based exclusions and selective reporting; 768 executions are not treated as independent examples. Pooled results are accompanied by request directions. Diagnostic base rates are retained; no adjusted causal model or reverse-causality claim is made. Selection/generalization and causal attribution remain limited by one seed, one event, the reused development panel, and the preserved historical O startup caveat. The exact fresh R0/Oa window does not prove historical O's entire 128-step trajectory identical. O remains the practical primary control; R/P shares the qualified warm-start design. No follow-up tuning, project adoption or workflow-stage change follows from this result.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_margin_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
