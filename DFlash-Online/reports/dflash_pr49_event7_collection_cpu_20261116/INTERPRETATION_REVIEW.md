# Independent interpretation and reproducibility audit

Coverage: all 11 ARS statistical fallacy categories checked. Confidence is **caution** for generalization: this is retrospective analysis of one origin/seed and a reused request population, with three unadjusted descriptive contrasts. It is not a new experiment or independent scientific replication. No p-values, standardized effect labels, efficacy threshold or multiplicity adjustment were added.

The separate reader saved its raw-derived results before reviewing the main reducer. It checked 112,536 native evidence assertions with no errors. The subsequent comparison passes **7,523 checks**, with exact agreement for every one of 384 arm/request rows, all 128 paired differences and population identities; all summary points and interval endpoints agree within an absolute tolerance of 1e-12. Ordered-source and common bootstrap draw hashes agree exactly. No counting or statistical disagreement remains.

| Contrast | Native acceptance change | Paired 95% interval | Call reduction | Paired 95% interval | Fewer / more / tied requests |
|---|---:|---:|---:|---:|---:|
| E versus N | −1.7261% | [−3.0112%, −0.4853%] | −1.4993% | [−2.6504%, −0.4165%] | 38 / 61 / 29 |
| S versus N | +0.2394% | [−0.4733%, +0.9768%] | +0.2045% | [−0.4078%, +0.8293%] | 44 / 42 / 42 |
| E versus S | −1.9608% | [−3.1797%, −0.7500%] | −1.7073% | [−2.8053%, −0.6447%] | 35 / 67 / 26 |

E saves 73 calls and adds 139 relative to N: **66 net additional calls**. S saves 61 and adds 52: **nine net calls saved**, with unresolved direction in the descriptive interval. E adds 75 net calls relative to S. These aggregates do not imply that every request changes in the same direction.

E's collection deficit is present before its own event-eight update, so that later update is not necessary for this observed deficit. This does not identify the cause of the final-panel deficit: the collection and final panels are different populations. S has a small favorable point estimate here, but this analysis does not demonstrate an advantage on this population or a uniform benefit across conditions. Its final-panel advantage must not be described as independently confirmed. Neither neutrality nor equivalence is established.

Acceptance and call savings are algebraically linked under exact common output tokens, since A = P − V. They are not independent evidence. Native accepted proposals A are primary; terminal matching count M is an audit quantity and differs here. Collection latency is excluded because capture and concurrent work were present. A decoding-work percentage is not a measured serving-speed percentage.

| ARS category | Assessment for this analysis |
|---|---|
| 1. Simpson's paradox | No predefined population subgroup comparison is introduced. All three arm contrasts and pooled totals are reported. The all-request pooled metric is not replaced with an average of individual ratios; final-panel and collection effects are kept separate. |
| 2. Ecological fallacy | Request wins/losses/ties and gross changes accompany pooled results. No claim that every request or individual loss row benefits or regresses. |
| 3. Berkson's paradox | Population is the fixed eligible collection window, not a random population sample. Retrospective reuse and scope are disclosed; no outputs or favorable lengths were used to choose this analysis subset. |
| 4. Collider bias | All 128 sealed pairs pass and are retained. No filtering on agreement, success, length, acceptance or outcome is used to produce a favorable comparison. Core validity failure would suppress the primary comparison rather than select a passing subset. |
| 5. Base-rate neglect | Denominators are explicit: 128 requests per arm, 384 records, 30,648 output tokens and 30,520 committed tokens per arm. All are EOS, with zero cap-hit and zero zero-call requests; boundary fixtures still cover those cases. |
| 6. Regression to the mean | This is not a remeasurement of selected worst requests, and no pre/post learning effect is inferred from the separate final panel. Method selection from earlier project work remains a limitation. |
| 7. Survivorship bias | All expected records and pairs authenticate; none are removed. Unknown/missing evidence paths were checked to produce incomplete reports without pooled efficacy. |
| 8. Look-elsewhere effect | All three declared contrasts and all required intervals are published, including unresolved S/N. These are not multiplicity-adjusted or corrected for earlier mechanism exploration. |
| 9. Garden of forking paths | Handoff-fixed population, metrics, identity sorting, common PCG64(1708) draws and linear percentiles are followed. No threshold, subgroup search or additional sweep is added after seeing results. Retrospective status remains explicit. |
| 10. Correlation versus causation | This count comparison does not isolate gradient direction, optimizer moments, repair history or the cause of final-serving outcomes. No collection timing or causal mechanism claim is warranted. |
| 11. Reverse causality | Each arm's immutable event-seven collection precedes its own event-eight learning. The later own update cannot have generated these predictions; cross-arm wall-clock overlap does not invalidate that narrow temporal statement. Other explanations remain unresolved. |

Native K+ remains the primary method, persistent E remains parked, and S remains a fixed update-strength control. This audit supports no further model work, tuning, checkpoint reconstruction or automatic next experiment. The narrow lesson is that E's deficit was already visible in its fixed event-seven collection, while S's advantage is not demonstrated uniformly across the two evaluated populations.
