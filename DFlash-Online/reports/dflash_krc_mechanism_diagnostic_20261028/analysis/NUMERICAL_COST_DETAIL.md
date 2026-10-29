# Numerical predictions and diagnostic accounting

Execution: `c9461345189b91d241ff21590027500e6ac7dab4`. Validated analysis: `220c62e2c52489cf114125b679ae7188eea09218`. This supplement contains descriptive numerical and accounting checks only; it makes no mechanism recommendation.

## Fitted-objective values

Every cell is **minimum / median / maximum across ten fixed windows** of that event. R candidates fit R; D_FULL fits D. Formal ideal is (1−alpha)V even for an inactive branch; the applicable ideal and its null reason are retained separately in JSON. Negative linearized predictions are retained.

| Event | Candidate | Before V | Formal ideal | Proposed linear V | Realized linear V | Actual V |
|---:|---|---|---|---|---|---|
| 2 | R_FULL | 2.35742e-05 / 0.000148802 / 0.000438345 | 0 / 0 / 0 | -4.51899e-11 / -6.34794e-13 / 1.85723e-11 | 1.9525e-05 / 0.000105628 / 0.000314812 | 6.98324e-06 / 8.4044e-05 / 0.000338114 |
| 2 | R_HALF | 2.35742e-05 / 0.000148802 / 0.000438345 | 1.17871e-05 / 7.44011e-05 / 0.000219172 | 1.17871e-05 / 7.44011e-05 / 0.000219172 | 2.239e-05 / 0.000133557 / 0.00040206 | 2.7933e-05 / 0.000136569 / 0.000345977 |
| 2 | D_FULL | 5.15627e-05 / 0.000125215 / 0.000349379 | 0 / 0 / 0 | -3.89024e-11 / -7.08169e-12 / 2.21012e-11 | 4.05261e-05 / 9.73271e-05 / 0.000244631 | 4.76903e-05 / 7.42295e-05 / 0.000279503 |
| 4 | R_FULL | 2.32934e-05 / 0.000119065 / 0.000559761 | 0 / 0 / 0 | -4.09701e-11 / 1.9058e-12 / 1.83929e-11 | 1.92221e-05 / 8.15531e-05 / 0.000372487 | 1.22321e-05 / 3.38426e-05 / 0.000368732 |
| 4 | R_HALF | 2.32934e-05 / 0.000119065 / 0.000559761 | 1.16467e-05 / 5.95323e-05 / 0.000279881 | 1.16467e-05 / 5.95323e-05 / 0.000279881 | 2.18903e-05 / 0.000107192 / 0.000509896 | 0 / 0.000100243 / 0.000519778 |
| 4 | D_FULL | 3.8761e-05 / 7.32979e-05 / 0.000259553 | 0 / 0 / 0 | -3.60309e-11 / -6.26912e-12 / 1.56514e-11 | 3.32288e-05 / 5.68645e-05 / 0.000168948 | 1.737e-05 / 7.10537e-05 / 0.000177589 |

## Per-window ratios

Cells again show minimum / median / maximum, computed per window before summarizing. Undefined denominators yield null with an explicit reason; they are never replaced by zero. Ratios are descriptive and can exceed one or be negative.

| Event | Candidate | Actual V / before V | Realized/proposed dot | Realized/proposed norm | Actual reduction fraction |
|---:|---|---|---|---|---|
| 2 | R_FULL | 0.08 / 0.743939 / 1.17857 | 0.142052 / 0.245012 / 0.294693 | 0.438126 / 0.569757 / 0.60732 | -0.178571 / 0.256061 / 0.92 |
| 2 | R_HALF | 0.32 / 0.902503 / 1.6 | 0.0430939 / 0.130497 / 0.224652 | 0.235033 / 0.414926 / 0.526679 | -0.6 / 0.0974967 / 0.68 |
| 2 | D_FULL | 0.52 / 0.78125 / 1.66667 | 0.15168 / 0.228765 / 0.327144 | 0.442882 / 0.549206 / 0.625249 | -0.666667 / 0.21875 / 0.48 |
| 4 | R_FULL | 0.107143 / 0.596032 / 1.33333 | 0.0881894 / 0.275262 / 0.334561 | 0.338187 / 0.576719 / 0.659724 | -0.333333 / 0.403968 / 0.892857 |
| 4 | R_HALF | 0 / 0.791209 / 1.21739 | 0.0466777 / 0.178846 / 0.215736 | 0.242945 / 0.481845 / 0.51272 | -0.217391 / 0.208791 / 1 |
| 4 | D_FULL | 0.333333 / 0.728469 / 1.55556 | 0.142727 / 0.217148 / 0.349082 | 0.410859 / 0.532619 / 0.674514 | -0.555556 / 0.271531 / 0.666667 |

The JSON also retains proposed/realized linear V divided by before V, each discrepancy, applicable-ideal nulls, all zero-denominator reasons, and every window’s scalar values. Proposed-versus-realized differences describe rounding of the actual displacement; actual-versus-realized residuals do not uniquely separate nonlinear/nonsmooth effects from finite-precision gradient/forward effects.

## Wrong ties after R hinge decreases

| Event | Candidate | R hinge falls: windows | Those windows with wrong ties | Protected wrong-tie positions | Blocking protected-tie blocks | Old-prefix tokens lost in those blocks | R prefix delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | R_FULL | 8/10 | 8 | 41 | 41 | 115 | +0 |
| 2 | R_HALF | 6/10 | 6 | 33 | 33 | 75 | +2 |
| 2 | D_FULL | 4/10 | 4 | 19 | 18 | 46 | +10 |
| 4 | R_FULL | 8/10 | 7 | 27 | 27 | 72 | +120 |
| 4 | R_HALF | 7/10 | 6 | 15 | 14 | 37 | +78 |
| 4 | D_FULL | 6/10 | 5 | 17 | 16 | 50 | +34 |

A blocking tie is the candidate’s first native mismatch, with zero canonical margin, inside the originally protected prefix. These counts establish local remaining errors, not a causal explanation of all missed progress or a serving-speed claim. D_FULL is selected here by its cross-reference R hinge, not its fitted D hinge.

## Phase costs

Campaign wall time: **3106.167 s**.

| Timer / scope (seconds) | Event 2 | Event 4 |
|---|---:|---:|
| Episode through endpoint comparison | 1567.154 | 1505.066 |
| Parent-state validation, inside episode | 9.374 | 21.492 |
| Disposable startup wall, inside episode | 43.850 | 34.758 |
| Actual load/live resume, inside episode | 26.606 | 26.494 |
| Training-event wall before commit | 1452.952 | 1388.920 |
| Training windows, nested in event | 1417.298 | 1353.927 |
| Checkpoint save, nested in event | 17.134 | 17.148 |
| Export, nested in event | 15.002 | 14.906 |
| Decoded endpoint comparison, inside episode | 6.601 | 6.821 |
| Verified replay retirement, after episode timer | 16.324 | not recorded / not applicable |
| Original pre_reference_s, nested in training | 76.889 | 73.459 |
| Original post_k_s, nested in training | 200.521 | 194.957 |
| Original correction_s, nested in training | 19.575 | 19.707 |
| Original integrity_s, nested in training | 4.209 | 4.218 |
| Original pre_reference_integrity_s, nested in training | 2.592 | 2.593 |
| Diagnostic candidate_sequence_s, sum over ten windows | 452.633 | 437.196 |
| Diagnostic final_restore_and_check_s, sum over ten windows | 109.402 | 107.288 |
| Diagnostic full_continuation_snapshot_s, sum over ten windows | 53.230 | 47.916 |
| Diagnostic post_K_parameter_snapshot_s, sum over ten windows | 13.549 | 4.703 |
| Diagnostic pre_capture_s, sum over ten windows | 73.367 | 70.978 |
| Diagnostic pre_capture_snapshot_s, sum over ten windows | 51.905 | 37.712 |
| Diagnostic validation_and_block_metrics_s, sum over ten windows | 15.024 | 14.032 |
| Diagnostic disjoint outer timers (not added again) | 754.086 | 705.793 |
| Candidate-sequence other instrumentation | 18.595 | 17.740 |

| Event | Candidate | Restore sum | Construction/gradient sum | State hash sum | Scoring sum |
|---:|---|---:|---:|---:|---:|
| 2 | R_FULL | 0.000 | 0.000 | 46.345 | 16.928 |
| 2 | R_HALF | 48.296 | 1.091 | 46.896 | 16.551 |
| 2 | D_FULL | 48.219 | 18.714 | 46.998 | 16.543 |
| 2 | K_ONLY | 48.836 | 0.000 | 47.043 | 16.553 |
| 4 | R_FULL | 0.000 | 0.000 | 45.800 | 14.990 |
| 4 | R_HALF | 47.308 | 1.092 | 46.334 | 14.993 |
| 4 | D_FULL | 47.554 | 16.322 | 46.446 | 14.972 |
| 4 | K_ONLY | 48.156 | 0.000 | 46.491 | 14.966 |

- Episode wall is recorded before replay-state identity hashing and event-2 tensor retirement; campaign wall includes those later operations. Do not assume episode sums cover the whole campaign.
- Startup wall includes startup loading. Event wall contains training, checkpoint save and export; objective and original correction component timers nest inside training.
- The disjoint diagnostic outer sum contains pre-capture snapshot, pre-capture, post-K parameter snapshot, full-continuation snapshot, candidate sequence and final restore/check. Validation/block metrics are inside candidate sequence, so they are not added again.
- Candidate restore, construction/gradient, state identity and scoring are inside candidate sequence. Its residual also includes direction-evidence reductions, recorder setup, receipt writes and other instrumentation; it is not a separately measured kernel category.
- D_FULL construction_and_gradient_s includes its gradient replay, direct correction and gradient alignment. A standalone D-gradient timer is unavailable and remains null.
- R_FULL construction timing is bookkeeping only: the actual original full correction was already applied and charged to original correction_s.
- These are recorded host wall timers. Not every subphase is separately CUDA-synchronized; a later CPU copy/hash can absorb preceding asynchronous work. Do not interpret construction/hash components as isolated GPU-kernel costs.
- Candidate and diagnostic component timers overlap their enclosing sequence/training/event/campaign timers. No overlapping timer totals are added.

## Work and recomputation

| Event | Phase | Draft forwards | Head calls including recomputation | Layer calls | Layer excess over one/layer/draft |
|---:|---|---:|---:|---:|---:|
| 2 | pre_reference | 256 | 8482 | 1280 | 0 |
| 2 | mechanism_pre_A | 20 | 643 | 100 | 0 |
| 2 | mechanism_pre_D | 20 | 638 | 100 | 0 |
| 2 | learning_K | 256 | 33928 | 1280 | 0 |
| 2 | post_k | 256 | 16228 | 1280 | 0 |
| 2 | mechanism_R_FULL | 60 | 1913 | 300 | 0 |
| 2 | mechanism_R_HALF | 60 | 1913 | 300 | 0 |
| 2 | mechanism_D_FULL | 80 | 3189 | 400 | 0 |
| 2 | mechanism_K_ONLY | 60 | 1913 | 300 | 0 |
| 2 | separate disposable startup | 2 | 392 | 10 | 0 |
| 4 | pre_reference | 256 | 8348 | 1280 | 0 |
| 4 | mechanism_pre_A | 20 | 546 | 100 | 0 |
| 4 | mechanism_pre_D | 20 | 555 | 100 | 0 |
| 4 | learning_K | 256 | 33392 | 1280 | 0 |
| 4 | post_k | 256 | 15820 | 1280 | 0 |
| 4 | mechanism_R_FULL | 60 | 1728 | 300 | 0 |
| 4 | mechanism_R_HALF | 60 | 1728 | 300 | 0 |
| 4 | mechanism_D_FULL | 80 | 2809 | 400 | 0 |
| 4 | mechanism_K_ONLY | 60 | 1728 | 300 | 0 |
| 4 | separate disposable startup | 2 | 392 | 10 | 0 |

Recorded core work: 2136 response forwards, 996 backward equivalents, 256 ordinary Adam steps and 512 presentations. Separate startup: 4 forwards / 4 backwards / 0 Adam. Serving and feedback generations: zero.

The head/layer calls above are internal work, not additional response equivalents. The D_FULL hook phase includes its D gradient replay plus six candidate-scoring responses per window. R extra post-K head evaluations (beyond its same-layout pre-capture count) are 7746/7472 for events 2/4; corresponding D extra evaluations are 638/526. Ordinary K head calls include its own objective/teacher work and recomputation; this receipt does not separate all those contributions.

All original/candidate counters, candidate timing distributions, startup counters, byte-based memory peaks and scalar objective component timers remain in the JSON. Earlier KRC1 local non-monotonicity was already documented; this supplement makes no first-discovery claim.
