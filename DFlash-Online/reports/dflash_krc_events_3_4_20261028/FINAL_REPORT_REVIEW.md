# Final report review — KRC events 3 and 4

## Material passport

ARS-Codex 3.24.0 experiment-agent, validate mode. Verification status: **ANALYZED**. Local same-team independent review of the completed report, retained audit and arithmetic; this was not a model rerun, blinded external review or new experiment. No remote activity, source edits, model operations or extra generations occurred. Only this review document was written.

Execution: `11b7718a04416a89025a6df69c0c33e558e5fe1b`. Actual analysis revision: `ea4f289c6e18383b7560c8c4103a9d91ca3943bf`. Report-builder SHA-256: `3c8e8462a15b7490bc074732a69b13216816d156977f5116d7c62b5e3d20a454`; the local builder and its object at the analysis revision match. These identities have different roles and are kept separate.

**Review outcome: no numerical or interpretation discrepancy requiring a report-builder or execution-code change was found.** The report answers the declared question and preserves the unfavorable correction comparison.

## Scientific interpretation

KRC4 uses fewer target verification calls than O4 on both panels: development 2,994→2,950, **1.4696% reduction**; additional 3,130→3,071, **1.8850% reduction**. K4 performs slightly better than KRC4 on both: development 2,942 versus 2,950 (KRC reduction **−0.2719%**, eight added calls); additional 3,058 versus 3,071 (**−0.4251%**, thirteen added calls). Thus the observed benefit of the full method over O does not establish a benefit of its full correction history over K at event 4.

On development, the KRC advantage over O persists but shrinks from event 2's 3.0075% to 1.4696%, a **−1.5379 percentage-point** change. Additional O1/O2 counts are absent, so persistence against O cannot be assessed there. The KRC/K advantage reverses on both panels: development +0.2018%→−0.2719%; additional +0.6378%→−0.4251%. These describe this trajectory, without a significance or equivalence claim.

KRC itself improves from event 2 to 4: 2,967→2,950 (**0.5730%**) and 3,116→3,071 (**1.4442%**). K improves more: 2,973→2,942 (**1.0427%**) and 3,136→3,058 (**2.4872%**). The report correctly separates within-arm movement from the relative advantage. Historical events 1/2 and fresh event 4 stay labeled; panels remain separate.

KRC4/K4 compares complete correction histories because their event-2 states already differ; it does not isolate corrections added in events 3/4. Shared O-generated feedback, one training trajectory and additional reference computation limit the conclusion. No seed robustness, universally useful correction schedule, compute-matched training benefit or training-inclusive asynchronous acceleration follows. The protected-position counts describe **pre-correction reference exposure**; disabled post-correction probes supply no direct final-prefix preservation measurement.

## Arithmetic and timing review

Direct local calculations made 9,189 scalar accounting/arithmetic assertions over the 128 per-request rows, 1,152 execution rows, 18 repeat totals, six comparison rows, both count curves, training costs and source identities. All passed. This count describes calculation checks, not additional executions or independent scientific observations. Per-request means match their three repeats; counts are repeat-consistent; pooled reductions, wins/losses/ties and gross saved/added reconcile. Gross savings minus additions equals each pooled call difference. Historical curve and within-arm figures were independently recomputed from the retained per-request columns.

The following signs are candidate reductions for repeats 0/1/2; `+` means less time and `−` means more time. All fresh request and native-generation directions agree with call-count directions. Cache maintenance has its own small, sometimes mixed component differences; it does not establish an overall latency direction.

| Panel | Candidate / reference | Complete request | Native generation | Cache maintenance |
|---|---|---|---|---|
| Development | KRC4 / O4 | + / + / + | + / + / + | − / + / + |
| Development | K4 / O4 | + / + / + | + / + / + | − / − / − |
| Development | KRC4 / K4 | − / − / − | − / − / − | + / + / + |
| Additional | KRC4 / O4 | + / + / + | + / + / + | − / + / + |
| Additional | K4 / O4 | + / + / + | + / + / + | + / + / − |
| Additional | KRC4 / K4 | − / − / − | − / − / − | − / + / + |

Summed mean request-time reductions for KRC4/O4 are **1.5959% / 1.8135%**; KRC4/K4 reductions are **−0.3369% / −0.5623%**, development/additional respectively. Three timing repeats are repeated measurements of fixed checkpoints, not three training replicates. Native and cache timers are nested inside request time. Publication/switching and weight checks are outside it. Historical latency and historical O/K training costs are not fresh matched controls or a latency learning curve.

## Work, costs and validity

Both events contain 128 new Adam steps and 256 ordinary presentations. Together they add **256 Adam steps, 512 ordinary presentations, 1,536 core response forwards and 956 backwards** (512 ordinary plus 444 reference backwards). Event 3 has 127 realized nonzero corrections and one zero-violation branch; event 4 has 125 and three. All diagnostic-forward counts are zero. Reference exposure is 256 presentations over 128 responses per event; the two reference forward phases are not 512 independent examples. Separate startup contributes two forwards/two backwards and zero Adam steps.

The event walls are **693.529 / 681.568 s**, totaling **1,375.097 s**. Reference/correction work is **296.045 / 288.806 s**, totaling **584.851 s**, already within inclusive training. The reported residual equals inclusive training minus reference/correction and integrity timers; it includes ordinary K and other overhead and is not an independent pure-K measurement. Save/export/retention are nested in event wall; the remaining wall components are about 3.554/3.011 s. State validation and artifact hashing follow the event timer and remain separately recorded. Both-event saves total 41.385 s and exports 30.769 s. The near-zero rounded retention entries are nonzero hardlink operations, 0.000069838/0.000073783 s, preserved precisely in TRAINING.csv. Peak memory is reported as decimal GB, with event-4 maxima 39.855 allocated/42.488 reserved.

The completed summary records 1,152 valid generations, zero invalid/unknown/unattempted work, and the included 72-generation gate. Development reproduces retained O4/K4 counts. The independent audit passes with no issues and reconstructs 54,435 native blocks. Complete-state receipts confirm global steps/presentations 384/768 and 512/1,024, the unchanged 7,424-step horizon, own-state restoration and retained KRC3/KRC4 states/exports. Full tensor checks were performed remotely by the campaign/audit receipts; this local report review did not rehash nonlocal tensor payloads. Its source-integrity basis is 1,053 local counterparts plus 606 remote-only identities among 1,659 seals, and 3,845 retrieved files rehashed.

One inherited compatibility receipt retains stale prose about three fresh K/G trainer constructions. Actual startup, event and state receipts establish a disposable startup plus one continuous KRC trajectory; this label is metadata, not evidence of extra training. The final audit records this limitation. No new arm, event-3 serving evaluation, tuning or continuation beyond event 4 is implied. Stage 1 / project None remains unchanged.

## Statistical fallacy coverage

All 11 ARS categories were checked: Simpson's paradox (panels kept separate); ecological inference (pooled differences accompanied by request directions, no universal per-request claim); Berkson/selection bias (fixed panels, no population-wide generalization); collider bias (no outcome-dependent adjustment); base-rate neglect (not a diagnostic-probability analysis); regression to the mean (relative controls shown beside within-arm change); survivorship (all planned executions accounted); look-elsewhere effects (all declared arms/panels and the unfavorable ablation retained); forking paths (predeclared endpoint, no retuning or checkpoint selection); correlation/causation (full-history, shared-feedback scope explicit); reverse causality (training precedes fixed-checkpoint evaluation). No p-values, confidence intervals or unsupported statistical-significance claims were introduced. This is descriptive evidence from one trajectory.

## Reviewed artifact identities

| Local artifact | SHA-256 |
|---|---|
| REPORT.md | `39bb0fc6d94aad629b5c3ed39cc727cf2e41757e41665d82baabef465461da7c` |
| SUMMARY.json | `cf689ca508c97f9e3aa3868c0fc551ac7534404d6fe0b911a04cebec0c667233` |
| TRAINING.csv | `1ddb7d4fccc8f6496ec798a0b607bae8dc37cf8fca75d9e0935c03f94ff54ae6` |
| FINAL_AUDIT.json | `e0a9bea4c6f7ff512171890dc72006f5f282bf2c6e5329943598f99ef4d6dd2e` |

Inputs also include HANDOFF.txt, FINAL_REVIEW_CHECKLIST.md, the ten analysis-panel CSVs, raw campaign COSTS.json and the declared source objects. Publication transformations and the exact final public bytes are reviewed separately; this document records the original local report identities.
