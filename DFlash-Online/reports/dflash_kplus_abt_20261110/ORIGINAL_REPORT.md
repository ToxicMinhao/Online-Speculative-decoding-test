# K+ adversarial block training: bounded events 5–6

**Evidence status: complete.** inconclusive_or_dataset_dependent.

## Material Passport

- Origin Skill: ARS-Codex 3.24.2 / experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED (CPU evidence reconstruction; no model rerun)
- Version Label: kplus_abt_report_v1

Execution source: `c64dfe3bf50d8bce84f2cca3576826c8d12a0087`; binding `5afd12aa4873b4c1c36e1b9cff2b787866ae474887da6f6f6e440f468f6b0669`.
Each dataset restores its complete PR32 KPLUS4 state, then continues four arms from 512 to 768 Adam steps. KPLUS4 supplies event-5 feedback and this campaign’s KPLUS5 supplies event-6 feedback. The arms share packets, ordinary order and anchors; this is not four independent feedback trajectories.

Rho is fixed at 0.01. KPLUS uses one view; CLEAN2, RANDOM and ADV use two equally weighted views with one native clip/Adam update per ordinary two-response window. Serving remains clean, greedy BF16 block-16, concurrency one, context 4096 and 2048 generated tokens or EOS. The user-scoped D24 exception does not establish a FreeLB reference reproduction.

## Completeness and failures

Valid arm-events: 16/16. Dataset endpoint states: gsm8k: passed, mathqa: passed.
All expected tasks and qualification cases remain in `RUN_STATUS.csv`, including failures and absent receipts. Absence is not imputed as zero work. A dataset with incomplete training or failed native output/termination audit receives no paired inference.

The separately retained failed attempt consumed 1 disposable Adam step and 4 forwards/4 backwards (2/2 case work plus 2/2 discarded startup), with 0 serving checks and 0 main updates. Its endpoint was not saved. The fresh qualification still requires 12 Adam steps, 68 forwards/68 backwards and 8 serving checks. The retry is sealed to reuse its four feedback packets: 0 new qualification feedback generations/teacher preparations; main feedback remains 512 unique packets, comprising 4 retained plus 508 newly collected. Prior qualification wall time was 161.684 s; nested feedback preparation 48.856 s, failed case 63.075 s, and startup 18.594 s are not additive. See `PRIOR_ATTEMPT.json` for the authenticated counts and execution source.

## Endpoint comparisons

Positive reduction means fewer calls or less complete-request time. Each time is summed after averaging the two repeats within each request. Ratios are ratios of sums, not means of request ratios. Calls use one validated deterministic trajectory per request.

| Dataset / ADV versus | Calls reduction % [95% CI] | Time reduction % [95% CI] | Call wins/regressions/ties |
|---|---:|---:|---:|
| gsm8k / KPLUS | 0.4077 [-0.0992, 0.9025] | 0.5091 [-0.0292, 1.0508] | 109/92/55 |
| gsm8k / CLEAN2 | 0.2084 [-0.2854, 0.6797] | 0.4156 [-0.1139, 0.9321] | 113/89/54 |
| gsm8k / RANDOM | 0.0501 [-0.3978, 0.4992] | 0.0600 [-0.4111, 0.5288] | 91/90/75 |
| mathqa / KPLUS | 0.1728 [-0.1571, 0.4958] | 0.2679 [-0.3592, 0.8919] | 101/89/66 |
| mathqa / CLEAN2 | -0.0836 [-0.4370, 0.2737] | -0.0022 [-0.6443, 0.6401] | 93/97/66 |
| mathqa / RANDOM | -0.2933 [-0.6457, 0.0458] | -0.2701 [-0.8911, 0.3371] | 90/106/60 |

Paired bootstrap: 10,000 PCG64 samples, seed 20261110, linear percentiles, shared request draws across all three contrasts. ADV/KPLUS is primary; CLEAN2 and RANDOM are mechanism controls. Intervals are unadjusted and conditional on one continuing training seed and historically reused development panels. Timing repeats are not independent requests or training seeds. No pooled favorable number or equivalence claim is made.

## Training cost and numerical observations

| Dataset / arm | Intrinsic training s | Diagnostic observer s | Observed windows with preclip norm >1 |
|---|---:|---:|---:|
| gsm8k / KPLUS | 637.704 (2/2 events) | 12.516 (2/2 events) | 0/256 |
| gsm8k / CLEAN2 | 1258.303 (2/2 events) | 12.482 (2/2 events) | 0/256 |
| gsm8k / RANDOM | 1262.459 (2/2 events) | 12.517 (2/2 events) | 0/256 |
| gsm8k / ADV | 1257.843 (2/2 events) | 12.497 (2/2 events) | 0/256 |
| mathqa / KPLUS | 1029.466 (2/2 events) | 16.073 (2/2 events) | 0/256 |
| mathqa / CLEAN2 | 2036.728 (2/2 events) | 16.082 (2/2 events) | 0/256 |
| mathqa / RANDOM | 2043.474 (2/2 events) | 16.130 (2/2 events) | 0/256 |
| mathqa / ADV | 2038.886 (2/2 events) | 16.122 (2/2 events) | 0/256 |

Intrinsic training subtracts only measured diagnostic observer time from its enclosing training interval. Worker loading, discarded startup, checkpoint save/export, shared collection and endpoint activation remain separate in `TRAINING_COSTS.csv` and `RUN_STATUS.csv`; nested timers must not be added. The clipping column counts reported preclip norms above the native threshold 1, not a different optimizer setting. Serving speed is not adaptation-inclusive speedup.

| Dataset | CLEAN2 exact native comparison | Different tensors / coordinates |
|---|---|---:|
| gsm8k | False | 203 / 521213127 |
| mathqa | False | 202 / 502417625 |

CLEAN2 is the actual two-view BF16 measurement; mathematical redundancy does not imply bitwise identity. Disabled and zero-radius checks are separate exact-native gates. Full numerical differences remain in qualification evidence.

| Dataset / arm | Recorded two-view responses | Mean clean→second teacher-loss change | Mean intended / realized block norm |
|---|---:|---:|---:|
| gsm8k / CLEAN2 | 512/512 | 0 | 0 / 0 |
| gsm8k / RANDOM | 512/512 | 1.62765e-05 | 0.0217435 / 0.0220384 |
| gsm8k / ADV | 512/512 | 0.0210587 | 0.0217435 / 0.0219404 |
| mathqa / CLEAN2 | 512/512 | 0 | 0 / 0 |
| mathqa / RANDOM | 512/512 | 7.149e-06 | 0.0219407 / 0.0222384 |
| mathqa / ADV | 512/512 | 0.0155211 | 0.0219407 / 0.0221356 |

Intended FP32 radii and realized BF16 input changes are distinct. `AUGMENTATION.csv` also preserves coverage, zero-gradient/no-eligible blocks and clean-to-second prefix changes. Loss increase under a finite nonlinear attack is an observation, not an automatic software-validity test or evidence of clean serving benefit.

## Fixed clean post-update diagnostics

| Dataset / arm | Recorded response observations | First errors repaired with prefix intact | New earlier errors | Net prefix change |
|---|---:|---:|---:|---:|
| gsm8k / KPLUS | 16/16 | 562 | 20 | 1662 |
| gsm8k / CLEAN2 | 16/16 | 524 | 20 | 1587 |
| gsm8k / RANDOM | 16/16 | 555 | 18 | 1635 |
| gsm8k / ADV | 16/16 | 555 | 12 | 1644 |
| mathqa / KPLUS | 16/16 | 600 | 23 | 1705 |
| mathqa / CLEAN2 | 16/16 | 582 | 36 | 1521 |
| mathqa / RANDOM | 16/16 | 587 | 26 | 1646 |
| mathqa / ADV | 16/16 | 622 | 28 | 1690 |

Windows 1/32/65/128 are fixed in both events and every arm. These are the exact fitting responses at fixed anchors, with clean support reused after the committed update. They are overlapping sampled-block occurrences on divergent training states, not held-out transfer, independent tokens or unique causal attribution of endpoint changes. See `DIAGNOSTICS.csv` for every observation.

## Interpretation and limits

All four arms and both datasets must remain visible. ADV must earn improvement over CLEAN2 and RANDOM before the direction itself is supported. Similar or unresolved differences do not establish equivalence. Better perturbed loss without better clean serving does not meet the project objective. This short shared-feedback continuation does not establish the full 58-event ranking, superadditivity, training-seed robustness or deployment performance.

ARS statistical/methodological scan: 11/11 categories addressed in `FALLACY_CHECKS.json`, with remaining panel-selection and multiple-comparison limitations explicit. This is a bounded report review, not an external independent replication. No automatic radius sweep, method adoption, longer run or research-stage advance follows. Stage 1 / project None remains unchanged.

Tables: `COMPARISONS.csv`, `PAIRED_REQUESTS.csv`, `TIMING_REPEATS.csv`, `TRAINING_COSTS.csv`, `OPTIMIZER_WINDOWS.csv`, `AUGMENTATION.csv`, `DIAGNOSTICS.csv`, `QUALIFICATION.json`, `RUN_STATUS.csv`, `SUMMARY.json`, `INPUT_MANIFEST.json`.
