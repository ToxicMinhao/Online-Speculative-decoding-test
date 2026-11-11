## Material Passport

- Workflow: ARS-Codex 3.24.2, experiment-agent validate.
- Verification status: **ANALYZED**; local evidence interpretation and limited scalar rechecks, with no new model execution, bootstrap rerun, remote operation or production edit.
- Execution: PR39, `4d977e9ffd2cab9d58ae8806c29df11d843353da`.
- Authorship: this reviewer authored the core SHUFFLED implementation and its CPU tests. This reviewer did not author this experiment's endpoint/statistical report. This is a separate same-team interpretation review, not blinded external review or independent experimental replication.
- Inputs: the handoff, retained final `SUMMARY.json` and `MATCHED_START.json`, scalar and raw-validity audit receipts, matched-start audit interpretation, cost notes, qualification/work receipts, and the public `REPORT.md`. Reviewed-file identities are in `FINAL_INTERPRETATION_SOURCE_PINS.json`.

The reviewed public report has no remaining factual or inferential blocker within this review's scope. Its bounded conclusion follows the handoff. One terminology error was found in the separate cost notes and reported to the parent: the 35 references were generated through native DFlash with TRUE and the target verifier, so “target-only canonical generations” is inaccurate. The public report already describes them correctly; the cost-note author is correcting that wording and provenance. This does not affect measured work or validity.

## Endpoint interpretation

TRUE's 11,522 calls versus SHUFFLED's 11,591 give 69 net calls saved and a 0.595289% reduction, with the prespecified paired, unadjusted 95% interval [0.170964%, 1.015408%]. This supports informative priority placement **for this shared training fork and fixed permutation**. It is a small aggregate benefit, with 111 improved, 78 regressed and 67 tied requests; it is not a benefit on every request or evidence across training seeds/permutations.

TRUE versus ordinary K saves 42 calls: +0.363196% [−0.071936%, 0.784491%]. SHUFFLED versus K adds 27 calls: −0.233483% [−0.653537%, 0.195408%]. Both remain unresolved. The supported primary contrast does not establish an incremental verifier benefit over K, and unresolved intervals do not establish equivalence. I independently recomputed all three point reductions and net call differences from retained arm totals; they match exactly.

All complete-request latency intervals cross zero. TRUE/SHUFFLED has +1.245038% [−0.140408%, 2.562571%], with favorable points in both sweeps. TRUE/K has +0.357830% [−0.979443%, 1.653258%], with opposite sweep directions. SHUFFLED/K has −0.898393% [−2.274756%, 0.451523%]. None supports a latency advantage or slowdown. The native-generation contrast is separately unresolved; cache-maintenance differences do not establish a kernel-level cause or training-inclusive speedup.

The supplied scalar audit reports exact agreement for 129 checked summary fields, 12 intervals, 52,224 raw/CSV fields and 161 publications. The raw audit separately reports all 1,536 output/termination pairs, sidecars and 69,354 verification blocks checked. These receipts support the report's stated validation, but this review did not repeat those complete audits or native execution. The repeated call counts reconcile to 2 × (11,564 + 11,522 + 11,591) = 69,354 blocks.

## Manipulation and matched starts

The control acted: 18,452 changed bits over 256 presentations, 45,255 eligible errors and 3,396 mixed strata; final weights differ. The 18,416.4065 uniform-permutation benchmark is descriptive, not a permutation p-value or evidence of robustness across maps. The same-state mass/selection checks do not imply equal masks or class totals across diverged arms.

The matched subset has 10 successor-high errors, 53 low errors and one all-correct parent. Successor-ineligible and seed-EOS groups have zero cases; no rates are inferred for them. There are 63 old errors and 64 nonempty earlier prefixes. Direct reduction of all retained matched rows reproduces K/TRUE/SHUFFLED repairs of 11/13/12, six prefix breaks each, and 62 tokens gained minus seven lost: **net +55 for every arm**. The all-correct parent is a tie, not an opportunity for repair. Canonical and native accepted-prefix agreement is an observed property of these records, not a general identity between definitions.

The report correctly avoids claiming preferential high-group repair: TRUE repairs 5/10 there versus 6/10 for both controls; its higher pooled repair count arises with 8/53 low-group repairs versus 5/53 and 6/53. These descriptive starts neither explain nor contradict the full-request call contrast, which includes later conditioning states. No local repair count is converted to a latency contribution or causal mediation claim.

## Cost and study limits

Main work is 256 new AdamW updates and 512 ordinary response forwards/backwards. Three disposable qualification updates and six ordinary qualification forwards/backwards are separate, with six qualification and four main discarded startup forwards/backwards. The two extended qualification cases add 94 detached projections relative to native TRUE. No disposable state enters main training.

K and SHUFFLED intrinsic training clocks are 328.064734s and 400.984953s. Their 72.920219s difference includes redistribution, logging and divergent trajectories; it is not pure assignment cost and cannot be assigned to native TRUE. The 50.372672s extra projection and 8.139749s assignment clocks are already nested inside SHUFFLED training. TRUE was authenticated and reused, so it has no contemporaneous main-training clock. A delegated read-only cost check reproduced 16 directly checkable clocks and confirmed this nesting.

The 3,330.467275s current runtime encloses qualification, training, reference preparation, parent diagnostics and serving. Its nested clocks must not be summed again. Staging, transfer, historical TRUE training and independent analysis are outside that runtime. The public report distinguishes these scopes. The 256 requests are held out from the training fork and SURVIVAL panels but have older project exposure; one continuing seed, one fixed permutation, one event and one task limit generalization. Completion does not authorize tuning, additional events/panels or a stage change.

## Fallacy coverage

All 11 ARS categories were checked.

| Category | Review disposition |
|---|---|
| Simpson's paradox | Both timing sweeps and parent strata are disclosed; differing stratum orderings are not converted to a pooled mechanism claim. |
| Ecological fallacy | Aggregate gains are not attributed to every request; wins/regressions/ties are reported. |
| Berkson's paradox | Input-hash selection and restricted eligible population are disclosed; no globally unused-panel claim. |
| Collider bias | The matched grouping uses the shared parent state; no conditioning on child success is introduced. |
| Base-rate neglect | Group sizes, 63 repair opportunities, 64 prefixes and zero-case strata remain explicit. |
| Regression to the mean | Parent-to-child improvement alone is not credited to TRUE; both controls are retained. |
| Survivorship bias | All 1,536 executions and all 64 matched starts are retained; zero cap hits is observed, not an exclusion rule. |
| Look-elsewhere effect | The prespecified primary contrast and all controls are shown; intervals are unadjusted, with no familywise claim. |
| Garden of forking paths | Sealed selection, fixed seed/map and bounded event are preserved; audit-reader correction is disclosed as analysis-only. |
| Correlation versus causation | The controlled contrast is bounded to its intervention/fork; local repair and timing components are not claimed as endpoint mediators. |
| Reverse causality | Parent training state precedes child outcomes; no additional directional claim is drawn from descriptive strata. |
