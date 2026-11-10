# Independent arithmetic and interpretation cross-review

## Material Passport

ARS experiment-agent validate; status **ANALYZED**. Completed GSM8K evidence only, source `6440e8ff5af1a845f830d65c884f6711d7e4a687`. Same-team reviewer with earlier authorship of numerical helpers/tests and sealing code; this is not external review or independent GPU replication. Zero model loading, model forwards/backwards, optimizer steps, generations, remote work, or active-source changes.

## Finding

No material numerical, control-label, or causal-interpretation error was found in `diagnostics/FINDINGS.md` or `production/PRODUCTION_REVIEW.md`. Their conclusions remain appropriately bounded to this GSM8K panel and the eight specified local windows. The report may say the diagnostics are **consistent with suppressed useful repair/progress and harmful local redirection**. It cannot say these records establish the unique cause of the endpoint regression or quantify how much endpoint latency/calls that mechanism explains.

## Independent checks

A separate script (`endpoint_review/review_diagnostics.py`) read all eight original diagnostic `RESULT.json` records and all 512 original FSG/SCALAR production step receipts. It did not import either analysis implementation. All 160 diagnostic window/pair/candidate rows reproduce: **4,320 count/KL cells** agree with the supplied CSV. All eight retain successful completion, exact state-restoration receipts, and zero extra Adam steps. The arithmetic pass took 0.0624 s; review and source-reading time are separate.

- FSG has **256/256 genuine accepted endpoints**, each distinct from native and pre-state, not merely an accepted boolean. All reasons are measured protected-KL decrease. There are 254 feasibility searches.
- Its committed norm/native ratio has minimum **0.999893208515**, median **0.999999999578**, maximum **0.999999999998**. Thus negligible total update norms or mostly inactive corrections do not explain this result. These ratios do not imply every useful direction was retained.
- SCALAR has **34/256 genuine writes**, 188 numerical-ambiguity and 34 measured-KL-increase fallbacks. Its committed norm/native median is 1.0. This is its independent evolving trajectory, not the same-state scalar diagnostic.
- Across all eight windows, committed-minus-native A net progress is **−158 occurrences**, worse in **7/8**, better in one. D is **−126**, worse in **6/8**, better in one, tied in one. Useful repair totals change by **−36 A / −38 D**. Old-prefix losses change by **−4 A / +17 D**; new earlier errors by **+2 A / +7 D**. Those sums concern overlapping block occurrences, not independent output tokens or requests.
- A protected KL decreases in 8/8; D decreases in 7/8. The stronger supported statement is suppressed repair/progress with inconsistent useful-prefix preservation, not successful behavioral preservation inferred from KL.

## A–F interpretation

**A:** The stated over-preservation interpretation is carefully qualified. Lower drift coexists with reduced repair/progress, but old-prefix preservation is weak or inconsistent. No claim that all suppressed directions were useful is warranted.

**B:** A improves over its own pre-state, but FSG generally reduces A's benefit relative to native. Therefore “fits A but fails to transfer” alone is incomplete; the report correctly distinguishes fit from incremental FSG benefit. D remains buffered training data.

**C:** Same-state scalar/FSG norm mismatch is at most **2.4267894e−12** relative. FSG has worse local net progress than this scalar in A 7/8 and D 6/8, supporting a local redirection effect beyond total norm. The scalar is explicitly **pre-guard**: 2 accepted, 3 numerical ambiguity, 3 measured KL increase. Six rejected scalar candidates would commit native. Their candidate scores must not be described as eight committed scalar updates, and the report does not do so. Independently trained SCALAR6 is a different comparison.

**D:** Feasible-minus-raw net progress sums **+47 A / −39 D**; raw FSG is already worse than native in each event/role aggregate. Realization changes outcomes and can contribute locally, but does not alone explain the deficiency. The raw norm-infeasible candidate is not a deployable control.

**E:** Uniform-versus-feasible FSG D net progress changes **+25**, but windows split 4 better/4 worse; A changes −66. This is mixed and not a new-method selection.

**F:** Expanded-versus-feasible FSG D net progress changes **+31** (5 better/3 worse), while useful repairs change −4; A net changes −13. Norm/effective strength also change. This is not evidence that sketch rank is the isolated cause.

## Cost and source review

All six event cost records and 30 copied cost fields agree exactly with their original `RESULT.json`. All six training-minus-observer subtraction identities agree. `native_pilot.py:298–300` explicitly subtracts observer wall time from the training envelope, and `functional_geometry.py:545` records a hook envelope that includes the native optimizer and committed observer. The published production review correctly labels parameter probes as nested inside realization, realization inside training, and avoids adding hook totals to training. Training ratios are implementation costs, not equal-compute claims or a kernel decomposition. Memory peaks include diagnostic/export activity as disclosed.

A minor inherited diagnostic free-text label calls `raw_fsg` “unadjusted a=1; expanded second direction” (`pilot_diagnostics.py:350`). The executable branch is the production raw FSG endpoint, keyed `raw_fsg`; the later expanded construction is separate. The analyzed table and report correctly use the executable candidate identity rather than this generic label. This does not change measurements or the conclusions and does not justify modifying the active run.

The endpoint and KPLUS4 arithmetic/provenance checks are separately documented in `endpoint_review/NUMERICAL_REVIEW.md`. All 11 ARS inference safeguards remain covered there and in both reviewed reports: preserve dataset and event strata, request versus occurrence units, fixed-window selection, all activity/fallbacks, denominators, fixed endpoints, missingness, multiplicity, unchanged analysis decisions, local-versus-trajectory causal scope, and temporal ordering. MathQA and the full two-dataset decision remain pending.
