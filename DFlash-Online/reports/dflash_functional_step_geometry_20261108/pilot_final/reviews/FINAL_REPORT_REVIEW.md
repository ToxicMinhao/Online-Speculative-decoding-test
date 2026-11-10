# Final assembled FSG R1 report review

## Material Passport

ARS experiment-agent validate; **ANALYZED**. Reviewed the assembled `PILOT_REPORT.md` against the independently reconstructed GSM8K/MathQA endpoint and origin arithmetic, authenticated completion receipts, `production/PRODUCTION_COST_WORK_REVIEW.md`, and all-dataset `diagnostics/FINDINGS.md`. Same-team reviewer previously contributed mathematical helpers/tests and preparation code; this is not external review or independent GPU reproduction. Zero new model/remote work, no active-source edits.

## Verdict

**No material numerical, validity, cost-accounting, comparison-label or causal-interpretation issue found.** The final report faithfully identifies a valid negative for this tested FSG R1 implementation and prespecified two-event pilot, with both datasets separately unfavorable against both controls. It does not claim a universal failure of geometry, target-answer degradation, equivalence of SCALAR/KPLUS, or a unique mechanism cause.

One minor wording correction was sent to the parent: replace “intervals are in percentage points” with **“interval bounds are expressed as percentages of the reference metric.”** The estimands are relative changes in calls/time, not differences between two percentage-valued measurements. No numeric value or interval needs changing.

Referenced artifacts are being assembled; this review intentionally does not treat temporarily missing publication links as missing scientific evidence. Final publication should verify those links and all private-input identity references.

## Factual and control checks

- Independent MathQA reconstruction validates all1,536 raw executions and reproduces original bootstrap points/intervals exactly; the earlier independent GSM8K review does the same. All six arm endpoint totals, mean times, matched acceptance/progress values, caps, win/regression/tie counts, gross saved/added calls, and positive-increase signs match the assembled report.
- All four primary latency intervals favor controls rather than FSG. SCALAR/KPLUS remains unresolved on both datasets. The narrative correctly separates these facts from the generic automated `inconclusive_or_mixed_pilot` label, whose code lacks a dedicated negative branch.
- Both origin comparisons authenticate exact panel/canonical/runtime/export compatibility and use counts only: GSM8K origin12,116 versusFSG12,752 (+5.2493%); MathQA origin17,074 versusFSG17,667 (+3.4731%). Neither historical latency nor an approximate replacement checkpoint is used.
- Completion receipts authenticate exit0, end23:02:04UTC,18ordered tasks, no live attempt/retry, and15,122-file postflight. The reported17h59m21supervisor duration differs legitimately from64,567.09scontroller time. Idle GPUs are a status observation, not evidence of scientific validity.
- The production audit supports512/512genuineFSGwrites and74/512SCALARwrites; active endpoint identity is distinct from pre-state and native. All1,024gate decisions and recorded continued state/export receipts pass. Nearly native norm ratios do not imply directions or BF16 coordinate changes are negligible.
- R1's norm-feasible construction, own-stateSCALAR matching and subsequent actualKLguard remain the implemented policy. Lower protectedKL is partly imposed, not independent success evidence. Divergent production trajectories are not same-state causal comparisons.
- Same-state diagnostic scalar scores are correctly labeled pre-guard and distinguished from the separately trainedSCALAR arm. All16windows and320pair/candidate rows remain included. The report retains the favorableMathQAuniformexception alongside lower repair and other adverse outcomes; it does not select a successor from diagnostic hindsight.
- Both new-work and feedback-exposure ledgers reconcile:512shared responses=508new+4retained;12events/1,536Adam/3,072ordinaryF/B;6,144additionalF/2,048B;16diagnostics640F/64B;24startupF/B andzero startupAdam;3,072endpoint generations. Shared feedback is not an own-policy deployment comparison.
- Intrinsic training excludes observer time exactly once. Parameter-probe/realization/hook components remain nested; loading/startup/preparation/persistence/activation are separate. Peakmemory is measured event memory and may include diagnostics/export. No equal-compute, fundamental-complexity or training-inclusive speedup claim is made.
- Retired event5bulk artifacts are disclosed as retired, not archived/replayable; final states/exports and compact certificates are retained. Native state equality is based on original certificates/source review, not a new tensor reload.

The original `raw_fsg` free-text label remains a harmless descriptive metadata defect: source builds the raw production sketch before separately constructing the expanded diagnostic. The final report refers to the earlier annotation, preserves original evidence, and does not confuse these candidates.

## Eleven statistical-fallacy checks

| Category | Review outcome |
|---|---|
|1. Simpson/aggregation|Datasets and events remain separate; a pooled favorable effect cannot conceal either regression.|
|2. Ecological/pseudoreplication|Paired requests are endpoint units; repeats are averaged, block occurrences and updates are not independent requests/seeds.|
|3. Berkson/selection|Fixed reused panels and16prespecified windows disclosed; no outcome-driven request/window replacement.|
|4. Collider/conditioned selection|All accepted/rejected candidates and production fallbacks retained; no accepted-only efficacy claim.|
|5. Base rates|Request, execution, block/error/prefix occurrence and teacherKLdenominators remain in report/tables.|
|6. Regression to mean|Fixed event6endpoint and diagnostic windows; no best-checkpoint selection.|
|7. Survivorship|Completed/failed/unknown/unattempted work and historical failures remain visible; no invalid request omitted.|
|8. Look-elsewhere/multiplicity|All planned contrasts/variants disclosed; intervals explicitly unadjusted; no familywise or new-variant success claim.|
|9. Forking paths|Original settings, paired bootstrap and supports preserved; no tuning triggered by null/harmful findings.|
|10. Causal overclaim|Endpoint harm is demonstrated for fixed runs; local redirection/repair evidence is supportive, not unique causal attribution.|
|11. Reverse causality|Pre-state supports precede update evaluation; lower postupdateKL is not retrospectively treated as proof of useful preservation.|

## Release boundary

After the small unit-wording edit and artifact-link check, the report is ready for additive publication. Preserve the unchanged automated recommendation and earlier interim/qualification/failure reports. No further scientific run, parameter revision or stage advancement follows from this review.

Executor resolution: the interval-unit wording was corrected before publication. Primary report/index links and public-payload scan pass; see `PUBLICATION_CHECK.json`.
