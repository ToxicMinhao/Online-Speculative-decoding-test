## Material Passport

- Origin skill/mode: ARS academic-research-suite, experiment-agent **validate**.
- Date: 2026-11-04. Verification status: **ANALYZED**. Version: vfd_final_interpretation_v1.
- Execution: **05e4b30488fb267a196edb1a81a5f6aaf613a838**; native K+/VFD baseline remains e0a9d3c235e1efd15404c549282d21c99243a21c.
- Sources: [fixed handoff](../HANDOFF.md), [prepared checklist](INTERPRETATION_CHECKLIST.md), [native summary](../SUMMARY.json), its accompanying score/selection/pass/paired-request tables and audits, [preparation review](../PANEL_PREPARATION_REVIEW.json), and [final delivery report](../REPORT.md).
- Scope: retained-evidence interpretation; no remote/model work or public-source edits. Stage 1 / project None remains unchanged.

## Decision and uncertainty

**Decision 3: this screen argues against the specified progress selector at this budget. Neither advancement nor the efficiency-only claim passes.** PROGRESS64's complete-request reduction versus FULL is **−0.8313% [−1.5384%, −0.1463%]**. It took 153.2546 s versus 151.9911 s across 200 per-request repeat means: 1.2634 s more, or 6.317 ms/request. The interval excludes zero for these trained checkpoints; this is a small adverse effect, not evidence of seed-level or broad-domain harm.

| PROGRESS64 versus | Request-time reduction | Paired 95% interval | Supported interpretation |
|---|---:|---:|---|
| FULL | −0.8313% | [−1.5384%, −0.1463%] | Adverse primary endpoint. |
| STATIC64 | −0.3921% | [−1.0425%, +0.2618%] | Point estimate worse; added value unestablished. |
| RANDOM64 | +0.0129% | [−0.6987%, +0.7438%] | One net verification call saved; neither advantage nor equivalence established. |
| ONCE | +0.7415% | [−0.0796%, +1.5574%] | Uncertain benefit. |

PROGRESS saves **303.459 s / 21.4587%** of measured logical updater time versus FULL, but the primary lower limit −1.5384% fails the fixed **> −0.5%** efficiency criterion. ONCE and RANDOM64 also fail that criterion: their lower limits versus FULL are −2.4740% and −1.6381%. Neither simpler arm clearly dominates under the sealed operational definition, since each PROGRESS-versus-arm upper limit is positive. No policy qualifies for the prescribed efficiency preference. STATIC is not adopted from its favorable point estimate.

The analysis unit is **200 paired requests**. Two timings are averaged within arm/request before the ratio of summed times; 10,000 paired bootstrap draws use PCG64 seed 20261104. The repeats are neither 400 independent observations nor two training replications. Six contrasts and four metrics produce 24 reported intervals: one primary and 23 descriptive, unadjusted secondary intervals. No p-values or multiplicity-adjusted claim is inferred. Overall confidence is **CAUTION**, reflecting small magnitude, one seed and secondary uncertainty; the prescribed decision itself is clear.

## Score and cost evidence

| Event | Valid scoring blocks | Initial opportunities | Opportunity repairs | Positive-credit blocks | Positive-score responses / 128 | Selected zero-score responses / 64 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | 578 | 436 | 44 | 37 | 27 |
| 2 | 2048 | 548 | 410 | 58 | 45 | 19 |
| 3 | 2048 | 554 | 405 | 69 | 53 | 11 |
| 4 | 2048 | 531 | 371 | 75 | 53 | 11 |

Across PROGRESS states, 244/8192 sampled valid blocks earn positive credit. There are 2211 initial opportunities, 1622 opportunity repairs and 385 qualifying opportunities. All valid blocks, including 1429 fully matching blocks, remain in the response denominators. Repair and broken-prefix flags can overlap and must not be added as mutually exclusive categories. Every recorded response has 16 valid score blocks. The cutoff is zero in every event; fixed ties fill **68/256 selected slots**. These facts describe sparse scoring; they do not establish why the policy was slower.

Event-1 STATIC/PROGRESS overlap is **37/64** under the same initial state. Later overlaps 41/38/39 also reflect diverged models. Second-pass selected response-token totals differ despite identical response counts: RANDOM 71,212; STATIC 66,763; PROGRESS 71,758. These lengths are not equal compute. The selector's `selected_valid_blocks` counts separate scoring blocks; supplemental exposure tables report native training blocks and occurrences.

Logical updater seconds are FULL 1414.156, ONCE 700.574, RANDOM 1053.035, STATIC 1087.226 and PROGRESS 1110.697. The shared event-1 first pass is charged to every policy; shared entry scoring to both scored policies, and post-scoring to PROGRESS. Logical steps/presentations are **1920/3840**, versus physical **1664/3328**. Final per-policy steps are 512/256/384/384/384. The updater scope includes pass setup, packet loads, scoring and selection. Initialization, restoration, feedback preparation, persistence and export remain outside this comparison. Savings therefore establish neither training-inclusive serving speedup nor full deployment throughput.

The source work audits record exactly 3328 main backwards/presentations, 1664 Adam steps, 1408 scoring forwards, 512 shared feedback generations/preparations, 2000 timed generations and 58 missing canonical generations. Qualification separately used eight scores, four presentations/backwards and two discarded steps, with initial restoration and fork equality. Physical pass wall is 4596.304 s, scoring 78.569 s and selection 0.280 s; source receipts separate compute, preparation, persistence/export and activation. Counts represent fixed work, not independent replications.

## Validity, scope and final-report claims

The native endpoint audit records all 2000 executions valid, exact canonical output/termination parity, stable repeated counts and no omitted or unknown work. All 200 requests and the one capped question remain. Counts agree with timing: PROGRESS adds 73 calls versus FULL, with 145 gross saved and 218 added. **Call** wins/regressions/ties are 70/101/29; latency wins are 89/200. Both timing sweeps are adverse (−0.7983%, −0.8642%). Eight of ten prespecified 20-request blocks are adverse and two favorable; these are descriptive checks, not new tests or exclusions.

The panel is train-disjoint, with **196 prior-project-use questions, four unmatched in incomplete available inventories and 142 old-panel overlaps**. It is not historically fresh. One seed-17/four-event trajectory uses common feedback from preceding FULL; this does not measure five independent live deployments. Request-bootstrap intervals do not include training-seed uncertainty.

The final delivery report's decision, point estimates/intervals, one-call random difference and cost scopes are consistent with the reviewed evidence. The root corrected the serving table header to **Requests with fewer / more / equal calls**, avoiding a latency-win interpretation. No remaining claim-level or cost-scope blocker was found. Separate data and native-exposure auditors own the completed numeric/raw checks; this interpretation review does not claim to have rerun the model, independently replayed complete tensors, or replaced those audits.

## Fallacy scan — 11/11 checked

| Category | Severity | Actual finding |
|---|---|---|
| Simpson's paradox | NOTE | No sweep-level reversal: both sweeps are adverse. Eight blocks are adverse and two favorable (range −2.5009% to +1.0833%). Heterogeneity exists; no all-subgroup reversal was observed. Keep the prespecified ratio-of-totals primary. |
| Ecological fallacy | CAUTION | Aggregate harm is not universal request harm: call wins/regressions/ties are 70/101/29 and latency wins 89/200. Policy totals do not reveal each training response's causal value. |
| Berkson's paradox | CAUTION | Input-only selection avoids outcome filtering, but prior exposure and eligibility restrict population inference. No specific Berkson artifact is established; the panel is not historically fresh. |
| Collider bias | NOTE | The primary includes every sealed request without conditioning on post-training scores, acceptance, calls or length. Those fields remain diagnostics, not adjustment variables. |
| Base rate neglect | CAUTION | All 8192 valid blocks remain in denominators; 244 have positive credit. Zero ties fill 68 selected slots. Scores are not calibrated probabilities; correlated blocks do not increase the replication count. |
| Regression to the mean | CAUTION | Choosing first-pass movement does not establish persistent benefit. The endpoint is adverse versus FULL and uncertain versus both selectors; score movement is not efficacy evidence. |
| Survivorship bias | NOTE | All 2000 executions and 200 requests, including the capped question and regressing requests, remain. Source audits report zero invalid, unattempted or unknown endpoint work. |
| Look-elsewhere effect | CAUTION | One primary and 23 descriptive unadjusted secondary intervals are reported. No favorable-only reporting, primary switch, equivalence claim or new post-result correction is justified. |
| Garden of forking paths | NOTE | The sealed handoff/source specify scores, hashes, panel, timers, bootstrap and thresholds. No tuning or extension follows. The exact execution pin remains 05e4b30488fb267a196edb1a81a5f6aaf613a838. |
| Correlation versus causation | CAUTION | The controlled policy comparison supports a narrow endpoint comparison. Sparse scores, repairs and lengths do not isolate why it is slower, establish per-response attribution or prove general mechanism failure. |
| Reverse causality | NOTE | Entry/immediate-post-pass-1 scores precede selection/pass 2; held-out evaluation is after event 4. Current feedback supplies scores, not later events or endpoint outcomes. Temporal precedence does not prove score movement causes later benefit. |

## Authorship and stopping point

This is a **same-team review with implementation overlap**. The reviewer authored the scoring module and scoring/training/worker tests, contributed narrow training/worker state and cost fixes, and previously reviewed campaign/storage/seal. The interpretation is not an independent implementation or GPU replication. The separate endpoint/bootstrap/raw/exposure audits are distinct deliverables.

The result argues against this fixed second-pass progress rule at this budget; it does not reject VFD generally or all selective replay. No tuning, new panel, seed replication, longer run or adoption follows automatically.
