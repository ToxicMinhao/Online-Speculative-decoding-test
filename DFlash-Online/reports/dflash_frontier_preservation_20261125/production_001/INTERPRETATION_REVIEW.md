# Independent result interpretation

## Material Passport

- Mode: experiment validation. Verification status: **ANALYZED**.
- Evidence: completed execution `225802657d630c0985abd68d6e7453f3e47c5b74`; all 1,134 retrieved compact files (442,061,947 bytes) independently hash-checked against the retained manifest, plus the archive hash.
- Work: independent measurement-row reduction and witness arithmetic, with no exporter or experiment imports and no model rerun. The separate exporter passes all 792 generations and 47,640 native blocks.
- Review provenance: a separate team agent with shared protocol context, not a blinded external replication. Aggregate results and checks are in `INDEPENDENT_RESULT_AUDIT.json`; the independent script and raw evidence remain local.

**The event completed, but serving efficacy is unresolved.** The independent calculation matches the saved primary result: FRONTIER_P complete-request latency reduction is **−0.016821%**, paired 95% CI **[−0.487073%, +0.410045%]**. The point estimate is slightly slower; the interval supports neither an improvement claim nor a resolved regression. It does not establish equivalence, because no equivalence margin was declared.

| Metric | CONTROL | FRONTIER_P | Candidate reduction, 95% CI |
|---|---:|---:|---:|
| Sum of per-request mean complete times | 135.998396 s | 136.021272 s | −0.016821% [−0.487073%, +0.410045%] |
| Sum of per-request mean native generation times | 131.975671 s | 131.997154 s | −0.016278% [−0.469856%, +0.404869%] |
| Verification calls, one checked repeat | 7,639 | 7,637 | +0.026181% [−0.439243%, +0.462544%] |
| Output tokens, one checked repeat | 50,226 | 50,226 | Identical output length and termination |

There are 50 request-level call wins, 41 regressions and 37 ties: 73 gross calls saved and 71 added, for two net calls saved. These small aggregate differences do not demonstrate serving benefit. Native accepted proposals/call are 5.558188 versus 5.559906; stop-clipped accepted proposals/call are separately 5.574028 versus 5.575881. Committed progress/call is 6.558188 versus 6.559906 and includes the verified correction token. These coupled counters are not separate independent successes.

The bootstrap uses **128 paired requests**, averaging each request's three timing repeats first, then drawing 10,000 whole-request resamples with PCG64 seed 20261125. Independent compensated-sum calculations reproduce the reported intervals within floating-point roundoff. The three repeats are timing observations on fixed endpoints, not three trained-model replications or 384 independent requests. Their latency-reduction point estimates are +0.035285%, −0.149358% and +0.063672%, consistent with an aggregate close to zero. The interval is conditional on one trained fork and this reused development panel; it does not include training-seed variation or establish population-wide generalization.

Witnesses provide a mixed diagnostic picture. Each set contains eight training responses and 64 overlapping fixed block occurrences, not 64 independent held-out examples:

| Witness measure | New buffer: CONTROL / FRONTIER_P | Parent training: CONTROL / FRONTIER_P |
|---|---:|---:|
| Original error blocks | 44 | 31 |
| Raw repair | 33 / 36 | 10 / 11 |
| Useful repair: error fixed and original prefix retained | 33 / 35 | 9 / 10 |
| Damaged original prefix positions | 3 / 3 of 488 | 14 / 14 of 676 |
| Surviving original followers | 58 / 59 of 60 | 30 / 29 of 33 |
| Reachable original followers | 39 / 52 | 12 / 13 |
| Matching-prefix length sum | 656 / 676 | 675 / 690 |
| Fixed-K shortfall sum | 35 / 21 | 75 / 59 |
| Mean response assessment, lower is better | 3.516551 / 3.149530 | 6.587701 / 6.919777 |

The candidate has favorable discrete repair/prefix measurements on these small sets, but the parent-training assessment is worse than CONTROL. Both endpoints also worsen that assessment relative to the common parent's 4.628858; both improve the new-buffer assessment relative to 6.844945. This prevents a blanket claim that the candidate preserved all prior behavior or improved every diagnostic. Damaged-prefix positions count actual old positions now wrong, not the entire prefix lost after one error. Surviving followers need not be reachable from the endpoint prefix. No witness significance test or independent-evidence claim is warranted.

The review independently reconstructs integer witness metrics from retained proposals/labels and checks the assessment formula from stored margins. Full vocabulary logits were not retained, so strongest-wrong rival identities and ties cannot be independently re-projected. Qualified derivative/allocation fidelity and activity establish implementation behavior, not serving benefit. The historical hidden-state 1.6% gate remains failed. Selected production updates did not record actual/shadow parameter-displacement norms; the integration BETA displacement diagnostic must not substitute for that missing production quantity. Training overhead remains separately reported and cannot be offset by a claimed serving win from these results.

## Statistical fallacy scan: 11/11 checked

| Check | Disposition |
|---|---|
| Simpson's paradox | No demonstrated paradox. Repeat signs vary around zero; keep the two witness sets separate because their assessment changes differ in direction. |
| Ecological fallacy | Aggregate ratios do not imply benefit on each request. Overlapping blocks do not increase the number of independent evaluation units. |
| Berkson's selection | The reused first-128 panel and first-eight witness sets limit representativeness. No outcome-conditioned exclusions are used. |
| Collider bias | Panel, common feedback and witness origins were fixed before branch outcomes. Do not restrict the primary result to repaired or faster cases. |
| Base-rate neglect | Preserve the 44/31 original-error and 488/676 original-prefix denominators; raw counts alone can mislead. |
| Regression to the mean | The contemporaneous CONTROL comparison is essential. Before/after fitting to the new buffer alone does not establish candidate benefit. |
| Survivorship bias | All 792 planned generations are valid and retained; no invalid cases were dropped. Earlier failed qualifications remain disclosed. |
| Look-elsewhere effect | Latency is the primary outcome. Secondary intervals are unadjusted; the favorable cache-maintenance interval or selected witness metrics cannot replace it. |
| Garden of forking paths | Endpoints, counts, estimator and bootstrap seed are fixed. Disclose the prospective numerical-contract revision and retain the earlier failed gate. |
| Correlation versus causation | The controlled fork supports the scoped comparison. Witness associations do not establish a causal explanation for general serving performance. |
| Reverse causality | Training precedes serving and origin roles were frozen; no reverse-time selection is evident. Post-update margins alone do not identify a causal mediator. |

No factual or arithmetic blocker was found for reporting **completed but efficacy unresolved**, with the limitations above. This review does not authorize another event, tuning, or a revised scientific claim.
