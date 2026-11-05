# O_HALF final interpretation and source review

## Material Passport

- ARS 3.24.0, experiment-agent validate mode. Verification status: **ANALYZED**.
- Execution: `19ee22faf985e0576d287eaa8c8624b5a44b46f5`; review concerns completed `production_002`.
- Scope: read-only handoff/source/result review and independent arithmetic from all 256 paired CSV rows. I previously authored the narrow archive/evaluator modules; I did not author this report/statistics implementation.
- This is not an independent GPU rerun, full native-trace audit, bootstrap rerun or decoding of recovery tensors. Those stronger evidence scopes must be attributed to the separate auditors. No source changes, remote access or model work occurred.

## Verdict

**No blocking interpretation finding. Decision A is consistent with the prespecified rule.**

K_HALF uses 15,865 calls versus O_HALF's 16,042: 177 fewer, or **+1.1034%**. The reported joint-request interval is **[0.4548, 1.8059]%**, with positive seed points **1.0703% / 1.1365%**. The point clears the 0.5% priority threshold and the interval clears zero. Its lower endpoint is below 0.5%, so this does **not** establish that the true improvement exceeds 0.5%.

Lower LR improves both implemented objectives here: O_HALF/O is +2.0575% and K_HALF/K is +2.6866%. K_HALF also improves over F by +2.6687%; O_HALF/F is +1.5828%. The descriptive objective-by-LR interaction is +0.6196% with an interval spanning zero; do not infer an established interaction. Wins/regressions/ties for the primary are 125/82/49 across the 256 seed/request outcomes, with gross 336 saved and 159 added calls. These counts are not 256 independent bootstrap units.

## Interpretation boundaries checked

1. **Counts, not new speedup evidence.** Archived F/K/K_HALF and fresh checkpoint-batched O/O_HALF evaluations support count comparisons only. The report explicitly excludes latency, throughput and training-inclusive speedup conclusions; all new timings are operational costs.
2. **Implemented objectives, not isolated token selection.** K changes both selected positions and the response denominator. Matching LR does not match displacement norms or separate these components.
3. **Lower LR also changes decoupled weight decay.** Weight-decay coefficient 0.01 is retained; a smaller LR is not exactly a half-sized parameter update.
4. **Conditional uncertainty.** The reported 10,000 PCG64/20261102 bootstrap resamples 128 joint request identities with both fitted seeds/all methods. Its interpretation is conditional on the two fitted trajectories and reused panel, unadjusted for prior K_HALF selection. Secondary intervals remain descriptive/unadjusted; no population-level seed robustness or fresh confirmation is claimed.
5. **Feedback and next action.** These are four-event shared O-generated feedback trajectories, not new O_HALF-controlled online deployments. Keep K_HALF as the provisional coding base and recommend one separately agreed bounded portability/online-setting check of that fixed base. This does not authorize a follow-on run, new controller or stage change.

## Pre-launch issues and their relevance

- The original JSON transport converted three provider tuple fields to lists. The narrow repair decodes those specified fields before strict contract comparisons; actual states and historical validators remain typed and unchanged. The retained actual-state CPU receipt verifies both seeds' initial weights/export, empty Adam/counters/history and RNG, with zero model/optimizer work. This was metadata transport, not a loss/gradient discrepancy or algorithm retuning.
- `production_001` failed CPU admission because 35 committed PR21 result-publication files were absent from the deployed source. The failure receipt records zero model work. Exact committed files were copied and `production_002` was freshly sealed at the same `19ee22f` execution commit. Postflight verifies all 2,790 sealed identities. The earlier failure and a superseded ad-hoc diagnosis remain preserved; no failed training/serving path was silently retried.
- The generated compact `REPORT.md` does not narrate these operational issues. Include this lineage or link the repair evidence in the final delivered report without modifying the raw original report.

## Checks actually performed

I independently recomputed from the paired CSV all five method call totals, all eight point effects, wins/regressions/ties and gross saved/added counts for the combined scope and each seed. All agree. Each of the 256 joined rows has the expected archived/new provenance and output-length/termination equality. Source review confirms the A/B/C/D decision boundaries and bootstrap units. I inspected migration, failed-admission, deployment, postflight and completion receipts.

Reported completion is eight valid O_HALF jobs, 1,024 Adam steps, 2,048 presentations and 1,072 valid generations (48 bridge + 1,024 final), with zero invalid or unknown work. New O_HALF updater time is 2,842.439 s summed over both seeds; historical O/K/K_HALF times are not fresh matched cost controls. Nested timers are not additive.

All **11/11 ARS statistical-fallacy categories** were checked. Reused-panel selection, conditional intervals and unmatched timing remain disclosed limitations, not removed uncertainties. Category details and exact reviewed file SHA256 identities are in `FINAL_INTERPRETATION_REVIEW.json`. No new model or CPU test suite was run; the only executed validation in this final review was the described independent table/receipt assertions.
