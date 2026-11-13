## Material Passport

- Status: **ANALYZED; completed bounded experiment, no new execution or statistical analysis**.
- Sources: the final `SUMMARY.json`, `REPORT.md`, `STATISTICS_AUDIT.json` and `REPORT_SELF_CHECKS.json` under `delivery/final_001/early_report/final_report`; the original handoff and sealed analysis rules.
- Training: `562f63fdff6955d5ffcf6086bf9b2f8f948dba8b`. Evaluation recovery: `345b1321856c40caab7ee3af7a9905dece3d0229`.
- Authorship: same-team reviewer with objective/report implementation overlap. Independent arithmetic is not an independent native replication.

## Supported conclusion

**This fixed recent-repair penalty did not demonstrate an improvement over ordinary K+ on this fork. The predeclared +0.50% useful accepted-length effect is unsupported, and the version should not advance as a K+ upgrade.** The intervention ran, fired naturally and changed BF16 updates, so this is not a zero-treatment result. Its intended next links—better retention on held-out starts and better later-request drafting—were not demonstrated.

The result is narrower than “retention never works” or “RECENT is proven worse.” The accepted-length intervals contain zero, but RECENT/NATIVE's upper bound is only +0.1044%, below the chosen +0.50% screen. There is neither an established speed benefit nor an established slowdown. No dose change, retuning or further experiment is justified automatically by these results.

The implementation uses **0.10 as the auxiliary-loss coefficient** and **`0.50*min(parent_margin,1.0)` as each fixed margin floor**. These constants must remain distinct in the final report's mechanism description.

## Completed work and primary endpoints

All three arms completed 128 production Adam steps and 256 presentations from the same full K+6 state (768→896). All 1,536 timed completions on 256 requests were valid, with zero invalid, unknown or unattempted executions. Counts use one validated trajectory per request; the two sweeps provide timing means, not extra statistical samples.

| Arm | Clipped accepted proposals | Verification calls | Accepted proposals/call | Summed request-mean time (s) |
|---|---:|---:|---:|---:|
| NATIVE | 62,402 | 11,026 | 5.659532 | 189.156668 |
| RECENT | 62,367 | 11,055 | 5.641520 | 189.568193 |
| CONTROL | 62,375 | 11,049 | 5.645307 | 189.563942 |

| RECENT versus | Primary accepted-length gain, % [95% CI] | Call reduction, % [95% CI] | Time reduction, % [95% CI] | Call wins / regressions / ties | Calls saved / added |
|---|---:|---:|---:|---:|---:|
| NATIVE | −0.318266 [−0.747809, +0.104368] | −0.263015 [−0.629433, +0.097513] | −0.217558 [−0.601650, +0.161451] | 80 / 94 / 82 | 107 / 136 |
| CONTROL | −0.067093 [−0.506686, +0.372561] | −0.054304 [−0.429268, +0.318863] | −0.002242 [−0.383925, +0.376357] | 74 / 74 / 108 | 110 / 116 |

Accepted-length gain and call reduction are different metrics: both accepted proposals and verification counts can change under identical final tokens. They must not be relabeled interchangeably. The paired speedups are 0.997829 versus NATIVE and 0.999978 versus CONTROL, each computed from complete-request mean times.

## Which mechanism links held or failed?

1. **Prospective tags and placement were feasible.** The frozen join supplied 815 distinct recent slots and 1,629 training occurrences. CONTROL switched 114 distinct slots across 70 requests, or 16.4569% of assigned raw weight, with absolute margin SMD 0.012456. All placement support thresholds passed; 701 recent slots remained shared by both active arms. This makes the prespecified placement test supported, not a test of every recent slot independently.
2. **The penalty reached the learned model.** RECENT fired on 341 occurrences in 107 of 128 update windows. Its first-firing active/off check changed preclip gradients, Adam moments and 297,041 BF16 parameter entries, with exact active replay and final restoration. Final RECENT differs from NATIVE at 296,587,104 coordinates. Those changes demonstrate treatment delivery, not useful direction or effect size. The 0.335772% assigned native weight is an exposure fraction; it does not prove that inadequate dose caused this result.
3. **Held-out retention did not improve in the required direction.** On 431 common repaired starts, RECENT had 123 exact-slot relapses and 275 accepts-through, compared with NATIVE's 121/278 and CONTROL's 118/281. Earlier breaks were 33 versus 32/32 and remain in the denominators. The relevant local contrast intervals include zero; the observed directions fail the screen without establishing every local difference as a reliable harm.
4. **Additional repairs did not produce a net local advantage.** Across 2,775 starts, RECENT acquired 1,356 and lost 1,361 prefix occurrences, net −5; NATIVE acquired 1,327 and lost 1,270, net +57; CONTROL acquired 1,383 and lost 1,274, net +109. RECENT passed 435 of 2,410 old blockers versus 442/439. Against NATIVE, the extra 29 acquired occurrences were accompanied by 91 extra lost occurrences. This describes the retained local outcomes; it neither proves mediation nor converts those counts into time saved.
5. **Later-request acceptance and placement advantage were not demonstrated.** Both primary intervals contain zero, and the NATIVE contrast's upper bound misses the useful-size target. The controlled extra term altered learning without demonstrating the intended endpoint benefit.

One marginal local interval—lost prefix occurrences versus NATIVE—excludes zero. All 16 local contrasts remain reported. This isolated secondary interval should not be promoted to a family-wise-confirmed explanation of the primary result.

## Prespecified decision audit

| Requirement | Outcome |
|---|---|
| RECENT/NATIVE accepted-length point at least +0.50% | Fails: −0.318266% |
| RECENT/NATIVE accepted-length lower bound above zero | Fails: −0.747809% |
| Placement support | Passes all four screens |
| RECENT/CONTROL accepted-length lower bound above zero | Fails: −0.506686% |
| Fewer held-out exact relapses and more accepts-through than both controls | Fails observed directions |
| Timing interval does not establish a slowdown | Passes: upper bound +0.161451%; speed remains unresolved |

The code's decision `prespecified_useful_size_unsupported_on_this_fork` follows the agreed order. It is supported by the native accepted-length upper bound below +0.50%, not by treating a negative point as proof of harm. All 22 prescribed intervals pass the independent 27,486-check arithmetic audit with 10,000 valid and zero undefined draws each. The full 11-category ARS interpretation scan is recorded in `ARS_FALLACY_CHECKLIST.md`.

## Cost, recovery and scope

Main training took 333.3912 s for NATIVE, 339.3717 s for RECENT and 338.9773 s for CONTROL. The difference includes actual divergent trajectories and instrumentation; it is not an isolated auxiliary-kernel estimate. Reversible captures, first-firing checks, frozen reference preparation, loading, persistence/export and serving costs are separately retained. Historical feedback collection, teacher preparation and K5/K6 label replay were reused, not newly timed; they are not free future online work.

The original publication-budget failure and its 24.235622 occupied GPU seconds are preserved. No parent generation began in that failed phase. Approved recovery reused the completed training and added zero optimizer steps. Summed execution wall was 6,258.9913 s; recorded allocated phase windows were 1.666006 card-hours, including loading/cleanup and excluding gaps. These are not kernel-busy time or full calendar elapsed duration.

All requests are training-held-out for this fork but have disclosed older-project endpoint exposure. The bootstrap conditions on one trained seed, shared feedback and fixed control placement. It cannot establish training-seed robustness, multiple own-feedback events, global freshness, or a training-inclusive speedup.

**Final decision:** stop this version at the agreed milestone and retain native K+. Reopening would require separately agreed evidence that addresses the failed retention/transfer link and meets the prespecified meaningful endpoint and control comparisons; the current experiment supplies no basis for automatically increasing the coefficient or expanding the run.
