## Material Passport

- Origin Skill: ARS experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-27 UTC (October 26 in Edmonton)
- Verification Status: **ANALYZED** — local raw-record and receipt audit; no new GPU reproduction
- Version Label: OCKA_compact_audit_v1
- Source: completed `dflash_first_error_execution_20261026/campaign_001`

## Audit result

**Passed, with zero inconsistencies.** The prepared auditor passed on its first
execution against the complete retrieved results. No audit-script correction,
raw-data change, new training, generation or remote operation was needed.
[Machine-readable audit](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/FINAL_AUDIT.json).

The campaign completed all **384 main optimizer steps** and **768 presentations**
across C/K/A. All **768 O/C/K/A evaluations** passed on the common **64/64-request**
development cohort. The audit independently reconstructed **36,786 native
verification blocks**, exact canonical outputs and termination, repeated call
counts, all four comparisons, and their CSV/JSON agreement. Retained O reproduced
its historical **3,075 calls**. There were no missing, duplicated, unknown or
extra evaluation slots and no fresh F generation.

The supervisor exited 0 after finishing at 2026-10-26 23:47:41 UTC. The remote
postflight receipts cover **117 source files, 10 execution files, 32 dependencies
and 144 retained artifacts**, with no changes. A separate local hash/size check
after this audit confirmed all **2,395 retrieved compact files** still match the
retrieval manifest. [Postflight](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/raw/POSTFLIGHT.json),
[local integrity check](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/FINAL_AUDIT_LOCAL_INTEGRITY.json).

## Recomputed measurements

Calls use one validated repeat-0 trajectory per request; all three repeats have
the same counts. Time is the sum over 64 requests of each request's mean across
three fresh repeats. It includes complete capture-disabled requests and excludes
training, checkpoint loading and activation.

| Arm | Verification calls | Sum of mean request times (s) |
|---|---:|---:|
| O — full forward KL | 3,075 | 52.777255 |
| C — full canonical CE | 3,102 | 53.257954 |
| K — first-error forward KL | 3,027 | 52.020989 |
| A — first-error canonical CE | 3,058 | 52.456160 |

Positive reductions mean fewer calls or less request time than the reference.

| Contrast | Calls saved | Call reduction | Time reduction | Call improved / regressed / tied |
|---|---:|---:|---:|---:|
| K versus O: selection under KL | 48 | 1.560976% | 1.432939% | 33 / 21 / 10 |
| A versus C: selection under CE | 44 | 1.418440% | 1.505491% | 35 / 17 / 12 |
| C versus O: CE versus KL, full | −27 | −0.878049% | −0.910806% | 14 / 30 / 20 |
| A versus K: CE versus KL, selected | −31 | −1.024116% | −0.836529% | 21 / 31 / 12 |

The signs of all four time contrasts agree across each of the three repeat
totals. For K/O, 92 calls saved on improving requests are offset by 44 extra
calls on regressing requests; the net is 48. For A/C, 90 saved are offset by 46
extra; the net is 44. These are bounded descriptive findings for this one event,
seed and development panel, without a new significance test in this audit.
[Per-request evidence](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/raw/campaign_001/evaluation/report/PER_REQUEST.csv),
[repeat totals](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/raw/campaign_001/evaluation/report/REPEAT_TOTALS.csv).

Historical F contributes **3,140 calls** and canonical outputs/termination on
this same cohort. It supplies no fresh latency comparison.
[Historical F context](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/raw/campaign_001/evaluation/report/HISTORICAL_F_COUNTS.json).

## Training and qualification checks

Every new arm used the same 128 retained feature identities and paths, exact
256-presentation order and sampled anchors, two passes, accumulation boundaries,
learning rates and 128 optimizer steps. Fresh initialization receipts match the
pinned released state and restored full RNG; all trainers closed successfully.
Each arm sampled 65,096 blocks with **949,560 available proposal occurrences**.

| Arm | Selected occurrences | Occurrences selected | Selected decay weight | Weight selected |
|---|---:|---:|---:|---:|
| C | 949,560 | 100.000000% | 423,956.893433 | 100.000000% |
| K | 534,063 | 56.243207% | 298,984.750916 | 70.522441% |
| A | 542,343 | 57.115190% | 302,315.797363 | 71.308145% |

The audit reconstructs available occurrence counts and decay weights from the
retained anchor schedule, checks the first-error histograms, selection bounds,
loss numerator/denominator consistency and one response epsilon, and sums only
the 128 optimizer-boundary rows. It does not count checkpoint recomputation or
non-boundary accumulation rows as extra exposure. Small rounding allowances in
these scalar reconstruction checks reflect FP32 arithmetic; the GPU exact
equality qualification gates retain exact comparison.

The separate qualification used exactly seven discarded windows: 14 current
training forwards, 14 backwards, seven temporary optimizer steps and 30 bounded
logit VJPs, with no generations or reference-model forwards. All seven gates,
22 exact comparison receipts and the compact evidence hashes for 35 retired
temporary tensor files pass. All-selected K/O and A/C equivalence, current-mask
oracle checks, qualified BF16 arithmetic, detached selection and excluded direct
prediction gradients were covered. C/A incurred zero teacher-head projection
cost; no arm incurred reference or auxiliary-objective work.
[Qualification evidence](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/raw/campaign_001/qualification/RESULT.json).

## Cost accounting

| Arm | Presentation training (s) | Complete event (s) | Peak allocated GPU memory (GB, decimal) |
|---|---:|---:|---:|
| C | 209.963450 | 241.748416 | 20.472524 |
| K | 325.094455 | 356.562822 | 20.472180 |
| A | 213.625288 | 245.658185 | 20.472524 |

Complete events include training, selection/teacher work, checkpoint persistence
and export. Selection timers are 9.550/13.072/13.267 seconds for C/K/A; K's
103.525-second teacher-head timer is already inside its training/event time.
Loading and fresh-state checking are recorded separately.

Qualification costs 642.040073 seconds including outer setup, with its nested
driver taking 639.096408 seconds. Native backend loading costs 21.257318 seconds;
evaluation wall time is 1,625.394339 seconds, including 705 checkpoint activations
and weight checks. The disjoint phase sum is 3,203.371447 seconds, below the
3,209.824443-second actual campaign wall time. Activation, selection and teacher
timers are not added twice. Reused features have zero new preparation cost,
which does not erase their historical acquisition cost. These serving results
do not establish an adaptation-inclusive speedup.
[Cost receipt](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/raw/campaign_001/COSTS.json).

## Interpretation limits and ARS fallacy scan

Compact records cannot reconstruct unsaved logits from every main-training
forward. Their accounting is checked here; mask/arithmetic/gradient correctness
rests on the qualified implementation and retained independent oracle evidence.
Exact tensor comparisons are read from their retained receipts rather than
rerun. O remains the retained practical control; equality to its entire
historical 128-step numerical trajectory was not claimed or established.

**Coverage: 11/11 statistical fallacy types checked.** The audit's completion and
consistency result is passed; generality and causal interpretation remain
bounded by the single-seed development experiment.

| Type | Assessment |
|---|---|
| Simpson's paradox | All contrasts use the same cohort; repeat-level signs agree. Per-request gains and regressions are retained. No subgroup-generalization claim is made. |
| Ecological fallacy | Aggregate improvements are not treated as improvements on every request; direction counts are reported. |
| Berkson's paradox / selection | The fixed development panel defines the scope. Results are not extrapolated to an unselected workload population. |
| Collider bias | No post-outcome covariate adjustment or performance-based cohort filtering was used; all 64 requests passed. |
| Base-rate neglect | Counts, denominators and gross gains/losses are explicit; no diagnostic-accuracy claim is made. |
| Regression to the mean | Matched full-block controls are retained; this is not an uncontrolled pre/post improvement claim. |
| Survivorship bias | All planned arms, requests and repeats completed; no invalid or incomplete request was silently discarded. |
| Look-elsewhere effect | All four planned contrasts are reported, including both unfavorable CE contrasts. No significance claim or winner-only inference is made. |
| Garden of forking paths | The sealed schedule, objectives and comparison axes were fixed before execution; this audit required no script or data correction. Broader exploratory history remains relevant to generality. |
| Correlation versus causation | The matched-arm experiment supports the observed contrast pattern; the historical O startup caveat, one seed and one event limit exclusive or universal causal claims. |
| Reverse causality | Training preceded evaluation under the fixed plan; evaluation outputs did not select or modify the trained arms. |

No further experiment, retuning, artifact retirement or project-stage change is
implied by this audit.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
