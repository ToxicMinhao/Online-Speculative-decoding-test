# Frozen event-seven replay: independent scalar and interpretation audit

## Material Passport

- Verification status: **ANALYZED; scalar and raw-receipt reductions passed**. No new GPU reproduction was performed by this reviewer.
- Scientific execution: `10d89bf8ba657cd905d7ad4fb31bf65a0a851656`.
- Scope: retained NATIVE7 and RECENT7, all 128 fitted requests, 815 sealed slots, 1,629 paired occurrences.
- Method: separately authored CPU reductions do not import the production analysis implementation. The reviewer inherited project context and inspected the implementation; this is same-family/team review, not blinded or cross-model replication.
- Model forwards, backwards, optimizer steps and generations added by this audit: **zero**.

## Audit results

The independent scalar audit passed **51,521 checks**, with no failures. It reconstructs the three-way outcomes from prefix lengths; checks native sign/tie consistency; reduces paired rows to exact equal-slot fractions; verifies headroom, rescue/loss and per-request totals; and reproduces the prescribed 10,000-draw request bootstrap through integer multiplicities and independently calculated linear quantiles.

The separate raw-receipt audit passed **58,525 checks**, with no errors. It binds every public scalar to its private raw observation and prequalified assignment, compares all 512 natural presentation/order/anchor/keep-mask receipts against the retained original schedule, checks work sums and confirms installed endpoint/runtime/shared-component equality receipts. It authenticates the original schedule by its archived SHA256. Tensor equality itself was performed by the executed scorer; this CPU audit verifies its source-bound receipts rather than claiming a second tensor-level reproduction.

Completed scientific work is exactly **512 response forwards / 16,340 vocabulary chunks / 3,258 arm observations**, with zero backward, optimizer, teacher-head, startup-response or generation work. Both endpoints score every planned presentation; no cases are filtered by outcome or reachability. Endpoint processes occupy **263.331328 seconds** in total; campaign wall is **275.951025 seconds**. The two synchronized scoring intervals sum to **161.698418 seconds**. These nested elapsed intervals include host work and must not be added or called measured kernel-busy time.

## Independently reproduced finding

| Quantity | Result |
|---|---:|
| Native accepts-through slot equivalents | 740 / 815 = 90.797546% |
| RECENT accepts-through slot equivalents | 745 / 815 = 91.411043% |
| Native headroom | 75 / 815 = 9.202454 percentage points |
| RECENT rescues / losses | 13 / 8 slot equivalents |
| Net RECENT minus native | +0.613497 percentage points; exact 100/163 |
| Paired request-cluster 95% interval | [−0.494453, +1.668574] percentage points |
| Raw occurrence rescues / losses | 25 / 16, among 1,629 pairs |
| Requests with retention disagreement | 19 / 128 |

The primary direction remains **unresolved**. The positive point estimate is not a demonstrated fitted-retention advantage because its interval includes zero. Conversely, this is not proof of no benefit or equivalence: the prespecified interval remains compatible with modest negative or positive differences on represented fitted requests. The interval does not describe training-seed uncertainty.

Native has 19 earlier-break and 56 tagged-mismatch slot equivalents. RECENT has 14 and 56 respectively. The paired 3×3 table shows that 6 of the 13 rescue equivalents arise from native earlier-break cases, while 7 arise from native first mismatches at the tag. Of the 8 losses, 2 become earlier breaks and 6 become tagged mismatches. These distinguish prefix accessibility from correctness of the tagged decision; they do not identify the training update responsible.

The singleton is a rescued tag and retains full slot weight. This explains why the raw 1,629-occurrence difference (+0.552486 pp) differs from the prespecified 815-slot difference (+0.613497 pp). Both are reported; the singleton is neither dropped nor duplicated.

Mean tagged margin rises by 0.012807 in the equal-slot descriptive reduction, while direct tagged correctness rises by only one slot equivalent (+0.122699 pp). These summaries have no inferential interval and do not establish a surrogate mechanism or native serving benefit. PR44's independently rechecked held-out outcome remains 8 rescues versus 11 losses, with 275/431 RECENT accepts-through versus 278/431 native. The original serving result stands.

## ARS statistical fallacy scan: 11/11 checked

| Category | Check and disposition |
|---|---|
| Simpson's paradox | Both pass-specific point differences are positive (+0.491400 and +0.613497 pp), as is the equal-slot aggregate. No sign reversal across these prescribed passes. Request heterogeneity remains visible in the per-request table; no uniform benefit is claimed. |
| Ecological fallacy | Primary unit is the equally weighted sealed slot; requests are the dependence clusters. Aggregate fitted retention is not attributed to every token/request or converted to free-running serving performance. |
| Berkson's paradox | The cohort deliberately consists of previously repaired, prequalified training tags. Its retention estimates describe this selected fitted population, not arbitrary positions or unseen requests. |
| Collider bias | Membership comes from archived pre-endpoint assignments. Endpoint correctness, reachability and loss firing do not determine inclusion. No post-outcome conditioning is used to explain causation. |
| Base-rate neglect | Native retention and its 9.202454 pp headroom, earlier/tagged misses, all 815 slots and all 1,629 occurrence denominators are reported. Rescue percentages are not presented without the original failure base. |
| Regression to the mean | Tags were selected by historical repair. A same-input native endpoint is retained as the comparison; parent-to-endpoint changes alone are not treated as a retention-treatment effect or proof of fragility cause. |
| Survivorship bias | All sealed occurrences are retained, including failed or unreachable tags and the singleton. Work and receipt audits confirm no incomplete or discarded cases in this completed run. |
| Look-elsewhere effect | One prespecified primary contrast receives a cluster interval. Pass, outcome, direct-correctness and margin summaries are descriptive; no favorable subgroup is substituted for the primary result. This remains a follow-up diagnostic prompted by prior negative serving evidence. |
| Garden of forking paths | Endpoints, archived cohort, slot averaging, PCG64 seed 1707, 10,000 draws and linear quantiles were fixed before scoring. No algorithm, coefficient, eligibility or reduction change was made after outcomes. |
| Correlation versus causation | The endpoint contrast describes this fixed experimental fork. It cannot establish why fitted and held-out results differ, whether more weight would help, or that local retention mediates serving performance. |
| Reverse causality | Archived tags and parent margins precede endpoint scoring; future outcomes cannot set tags. Temporal ordering does not locate the damaging update or validate an earlier intervention. |

## Supported endpoint conclusion

Native leaves **75 of 815 slot equivalents of headroom**. RECENT rescues **13** but loses **8**, producing **five net retained slot equivalents**. This is a small favorable fitted point estimate whose direction is statistically unresolved under the prespecified request sensitivity analysis. The evidence does not demonstrate strong fitted preservation that then fails to transfer, nor does it establish that the mechanism cannot preserve fitted repairs. PR44's lack of serving benefit is unchanged. No coefficient change, further training or new experiment follows from this audit.
