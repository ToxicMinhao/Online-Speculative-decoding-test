# AdaFlash C1: independent statistics and interpretation review

**Status: ANALYZED.** All **204,139 numerical, pairing, scheduling and table checks passed**, with **zero discrepancies**. This separate same-team review recomputed statistics directly from the retained CSV rows, using no production statistical functions and no model/GPU execution. It is independent arithmetic, not independent training replication or raw-token revalidation. The reviewer previously reviewed the implementation and supplied its analytic loss fixture; that authorship overlap is disclosed.

Execution: `681336c5e5c5caa0aa67560e46ffe83ce81e9351`, PR #29. Inputs: `completion_review_001/results/{ALL_REPEATS.csv,PAIRED_REQUESTS.csv,SUMMARY.json,REPEAT_TOTALS.json,VALIDITY.json}`. Input identities, recomputed values and check counts are preserved in `STATISTICS_AUDIT.json`; the standalone CPU calculation is `recompute.py`, with output in `RECOMPUTE.log`.

## Supported result

**K+58 retained a clear endpoint advantage over paper-defined AdaFlash-OPD A58 in this experiment:** complete-request time was **4.02845% lower**, with paired 95% interval **[3.32888%, 4.73465%]**. Verification calls were **4.02813% lower**. This supports the prespecified K+/A endpoint decision for these checkpoints and the fixed C1 panel.

| Comparison (positive favors first arm) | Complete-request time reduction | Paired 95% interval | Call reduction | Request call wins / regressions / ties |
|---|---:|---:|---:|---:|
| K+58 versus A58 — primary | +4.02845% | [3.32888%, 4.73465%] | +4.02813% | 687 / 195 / 142 |
| K+58 versus O58 — secondary | +3.25001% | [2.56777%, 3.94162%] | +3.27074% | 633 / 250 / 141 |
| A58 versus O58 — secondary | −0.81112% | [−1.38820%, −0.24470%] | −0.78918% | 298 / 468 / 258 |

Secondary intervals are the prespecified unadjusted paired intervals, not a separately multiplicity-controlled family claim. A58 required more calls and more complete-request time than O58 on this endpoint. No request or repeat was selected to obtain these results.

| Pooled quantity | O58 | K+58 | A58 |
|---|---:|---:|---:|
| Verification calls | 44,730 | 43,267 | 45,083 |
| Output tokens | 314,847 | 314,847 | 314,847 |
| Complete-request seconds, sum of three-repeat means | 810.06916 | 783.74184 | 816.63976 |
| Accepted draft tokens / verification | 6.03704 | 6.27550 | 5.98181 |
| Committed output progress / verification | 7.01594 | 7.25317 | 6.96101 |

K+/A saved **2,181 calls** on improved requests and added **365** on regressions, for **1,816 net calls saved**. The pooled result does not imply that every request improved. All three capped requests remain included; output lengths range from 94 to 2,048 tokens.

The fresh K+/A complete-request reductions are **4.18730%, 4.20784%, 3.68872%** in repeats 0–2. All sixteen predeclared 64-request blocks have a positive descriptive K+/A time reduction, ranging from **0.89823% to 6.90708%**. These checks reveal no aggregate/repeat/block direction reversal; they are descriptive, not additional hypothesis tests or alternate endpoints.

Native-generation time separately favors K+ over A by **4.07587% [3.60088%, 4.56171%]**. The cache-maintenance interval crosses zero. Thus the native-generation measurements support the endpoint difference; aggregate clocks do not identify a particular kernel or training mechanism responsible for it.

## Statistical and accounting checks

The reconstruction confirms exactly **9,216 valid table rows**, **1,024 distinct paired requests**, three distinct repeats per method/request, stable checkpoint identities and repeated counts, equal output lengths/termination, and the full sealed order. Every one of the six method orders occurs eight times. Counts use one trajectory; latencies use the mean of three repeats. The bootstrap resamples whole paired requests with **PCG64 seed 20261102, 10,000 draws, linear 2.5/97.5 percentiles**. All twelve declared metric/comparison intervals reproduce the report; there are zero undefined draws.

The retained summary/validity receipts report zero invalid, unknown, unattempted, pending or extra endpoint executions. Their training accounting reconciles **58 events, 7,424 Adam steps, 14,848 presentations, 7,424 terminal preparations, 7,473 online generations and four separate correctness generations**, followed by 49 final requests without learning. This review checked the arithmetic and reported receipts; separate raw-record and training-state reviews establish the underlying evidence.

Training/adaptation cost remains separate. A's worker wall is **31,265.087 seconds**, including collection and other operations; it is not pure training time. Nested compiler, checkpoint, update-call and worker timers must not be summed as independent costs. Historical O/K+ training times do not constitute fresh matched overhead measurements. The endpoint result alone does not establish that adaptation costs are repaid.

## ARS statistical fallacy scan — 11/11 checked

| Category | Disposition and evidence |
|---|---|
| 1. Simpson's paradox | **Not identified in the declared groupings.** K+/A is positive in all three repeats and all sixteen predeclared schedule blocks. This does not rule out other unexamined subgroup patterns. The primary remains the ratio of summed paired request means. |
| 2. Ecological fallacy | **Interpretation limitation retained.** The aggregate win coexists with 195 request regressions. A one-seed checkpoint result cannot be generalized to all requests, training seeds or workloads. |
| 3. Berkson's paradox | **Selection limitation disclosed; no outcome selection identified.** All 1,024 sealed requests remain, including three capped requests. This is a previously used, train-disjoint GSM8K panel, not a globally fresh or population-representative sample. |
| 4. Collider bias | **Not identified in the primary calculation.** There is no filtering by latency, call improvement, cap status or clipping activity. No correctness-invalid rows were selectively discarded; reported invalid counts are zero, with raw validation reviewed separately. |
| 5. Base-rate neglect | **Not identified.** Request counts, call totals, gross saved/added calls and acceptance/progress denominators are explicit. The primary sample size is 1,024 paired requests, not 9,216 independent trials. Clipping ratios require eligible-entry/position denominators and must not be equated with efficacy. |
| 6. Regression to the mean | **No historical-to-fresh improvement claim used.** All three checkpoints were retimed together on the fixed panel. The conclusion does not rely on choosing a prior extreme request, repeat or historical latency. |
| 7. Survivorship bias | **Not identified in the retained endpoint tables.** Complete schedule and accounting are present, zero unknown/unattempted/invalid records are reported, and all regressions are retained. Completion and source evidence remain separate audit responsibilities. |
| 8. Look-elsewhere effect | **Primary preserved; wider-project selection remains a limitation.** K+/A complete-request time was fixed before execution. All declared secondary comparisons are reported. The project has examined multiple candidates; this result does not erase that history or turn this reused panel into independent confirmation. |
| 9. Garden of forking paths | **No within-run analysis change identified.** Fixed seed, estimator, balanced order and all repeats reproduce the declared analysis. No alternative filtering, favorable subset or bootstrap was substituted. Prior implementation repairs/failures remain disclosed rather than treated as scientific retries. |
| 10. Correlation versus causation | **Mechanism attribution unsupported.** The endpoint compares complete own-feedback recipes. It does not isolate the causal contribution of reverse KL, entry clipping, CE mixture or feedback differences. Associations between loss/clipping and acceptance cannot explain the mechanism by themselves. |
| 11. Reverse causality | **No reversed temporal interpretation needed for the endpoint.** The retained protocol trains before freezing and timing the checkpoints; evaluation does not supply feedback. Within-training associations would still need a separate causal test. |

No statistical-reporting blocker was identified. The interval is sufficiently separated from zero for the declared endpoint decision, but it captures request-level uncertainty conditional on these checkpoints; it does not measure training-seed uncertainty or all system drift.

## What this does and does not establish

This is evidence that **K+ retains its serving advantage against the specified paper-defined AdaFlash-OPD loss under the matched PR25 protocol**. The experiment uses 0.8 teacher-argmax CE plus 0.2 entry-wise upper-clipped reverse KL; the pinned public trainer instead sums vocabulary divergence, normalizes, and then caps at position level. A58 therefore tests the requested paper formula in DFlash-Online, **not that public-code variant or the complete AdaFlash method**. No adaptive-length head was trained or served.

The result does not establish broad-domain robustness, another training seed's ordering, other concurrency settings, an isolated benefit of K+'s reweighting over K, training-inclusive speedup, or a general failure of AdaFlash. The appropriate completion is the fixed comparison report and evidence. No tuning or additional run follows automatically.
