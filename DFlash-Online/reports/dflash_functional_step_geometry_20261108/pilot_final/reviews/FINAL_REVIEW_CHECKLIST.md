# FSG R1 final two-dataset review checklist

## Material Passport

ARS experiment-agent **validate**. Prepared November 9, 2026 from the original handoff, the adopted BF16 realization R1 supplement, and executed-source reporting/statistics code. This is a CPU/source review checklist, **not a statement that final MathQA evidence has already passed**. No remote/model work or scientific-source edits. Same-team reviewer previously contributed implementation/tests; disclose this overlap, and distinguish independent arithmetic from independent GPU replication.

Scientific execution: `6440e8ff5af1a845f830d65c884f6711d7e4a687`; GSM8K interim publication: `e8f7c796fb556e5bdafcf2118b718eb60c4405ef`. Pilot binding: `8ec5fed0d4ed9b40dab7c26d5089cd42b01f357ab0379193da42baa93f0b3dc5`. Prior failures and interim conclusions remain historical evidence.

## 1. Completion, identity and causal training controls

- Authenticate a stable final compact snapshot by exact paths/sizes/hashes; verify supervisor exit, final status, source/input postflight, and every planned phase. Failed/unknown/unattempted work must remain visible; absence of an exception is not completion evidence.
- Confirm **six independent dataset/arm continuations**, each restored from its own dataset's complete PR32 KPLUS4 state at cumulative Adam step512, then events5–6 ending at768. No qualification state, optimizer reset, seed reset, cross-dataset origin or later-checkpoint substitution.
- Confirm **12 training events, 1,536 native Adam applications, 3,072 ordinary presentations**, 128 responses/two passes/128 steps per event. Preserve native moments/counters/RNG, active-tensor membership and ordinary K+ objective/arithmetic/settings.
- Four shared feedback windows (dataset × event), each128 responses, **512 total feedback generations including qualifying packets reused within budget**. Existing launch admitted four reused packets/508 new; audit actual identities/counts rather than assume this. Evolving KPLUS4/KPLUS5 supplies each dataset's event5/event6 feedback. All three arms share byte-identical prepared packets, presentation order and native anchors. This is shared-feedback continuation, not three own-policy deployments.
- Preserve historical failures and the approved numerical/startup revisions; no claim that original failed cross-layout gates passed. Reference the successful supplementary R1 qualification separately (four disposable Adam cases; at most8 ordinaryF/B,32 additionalF/16B,4 serving; zero feedback/teacher).

## 2. Serving endpoint, primary estimands and validity

- Event6 only: **256 existing development requests/dataset × three arms × two sweeps =1,536 executions/dataset, 3,072 total**. Previously used development panels are not new/unseen test sets.
- Audit exact request/prompt identities, export identity, canonical token IDs and termination for every record, all cap cases, and repeat-consistent deterministic counters. No selective dropping of mismatches or unfavorable requests. Distinguish qualification/startup records from endpoint observations.
- Confirm native BF16/five-layer/greedy block16, context4096,2048output tokens-or-EOS, concurrency1, cold-cache policy, disabled capture/detail tracing, fixed balanced order and no concurrent GPU training during timing.
- Average two times **within each request**, then take ratios of aggregate sums. Use one validated deterministic counter trajectory/request/arm. Never count repeats as independent requests.
- Run the unchanged `statistics.reduce_stage/summarize` with event6. Independently verify arithmetic. For each dataset, preserve FSG/KPLUS, FSG/SCALAR and SCALAR/KPLUS.
- Bootstrap: **10,000 paired-request resamples, PCG64 seed20261108, linear percentile95% intervals**, same sampled request indices across all contrasts within dataset. Intervals are unadjusted and conditional on these reused panels and one continuing training seed.
- Report total/mean complete-request latency, native-generation/cache components, calls, matched accepted proposals/call, separately native accepted counter/call, committed progress/call, output lengths/totals, EOS/cap/validity counts, wins/regressions/ties, gross calls saved/added and separate sweep totals. Keep activation/loading/validation outside request latency; do not infer kernel attribution.
- Preserve reduction sign `100*(1−candidate/reference)`; if also showing increase, negate point and reverse/negate interval endpoints.

## 3. Production behavior and R1 correctness

- Per dataset, each FSG/SCALAR has256windows (512filtered windows/dataset;1,024total). Verify each saved trace against the event receipt and independently recompute gate decisions.
- Count genuine activity by exact committed identity: distinct from **both pre-state and native endpoint**, not accepted flag alone. Separate eligibility/nonzero displacement, empty support/sketch, native/zero/lambda bypasses, rounded-to-prestate, ambiguity and measured KL rejection.
- R1 is a **norm-only 24-iteration lower-feasible-bracket search of the original continuous displacement**, not scaling an already-rounded delta or KL-guided backtracking. Already-feasible a1 stays exact; strict actual norm/KL checks and deadbands are unchanged. Final feasible write must match measured bound. Record failed/no-op paths honestly.
- Report raw continuous/native, rawBF16/native, feasible/native and committed/native norms; search incidence/alpha/brackets/probes; exact endpoint identities; direction cosines; changed coordinates including native-zero coordinates; per-response/pooled protectedKL and margin-floor/tie coverage/weight fractions.
- SCALAR computes its **own** feasible hypothetical target at its evolving state, then its original closest-bracket24-step matcher. Report target/realized mismatch and its own final candidate KL/guard; do not borrow FSG targets or claim exact norm matching if rounded mismatch remains. Independent SCALAR trajectory is not the local same-state scalar counterfactual.
- Other recorded geometry: c, gradient norms/dots, `(g·u)/c`, Gram/eigenvalues/sigma/solve residual/cross-response cosine and projected energy. A small residual or lower protectedKL is not useful-learning evidence. Missing optional ordinary-gradient directional products stay unavailable.

## 4. All16 prespecified mechanism windows

- Both datasets × events5/6 × windows1/32/65/128 = **16**. Report every window and candidate, including harmful/null outcomes. Current A fitting pair and D32pair-positions-ahead within the same pass; original deterministic isolated diagnostic anchors fixed across candidates. D is buffered training data, not held-out evidence.
- Preserve prestate/native/rawa1FSG/norm-feasibleFSG/committedFSG/matchedscalar/uniformraw+feasible/expandedraw+feasible identities. Separate candidate scores from guard-selected endpoints, especially rejected scalar/uniform/expanded candidates. Include physical committed-state restoration, RNG/modes/grads/optimizer integrity.
- Per A/D/candidate: useful first-error repairs with prior prefix intact; native repairs and gained prefixes retained/lost/added; old-prefix occurrences lost/new earlier errors/net contiguous-prefix change; ties/canonical margins; full-vocabulary old-draft KL; fixed-support teacherKL prefix/first-error numerators/denominators. Report response/block/occurrence denominators; overlapping occurrences are not independent tokens or test requests.
- R1 raw versus feasible realization must be visible; expanded second direction is the original raw a1FSG displacement. Uniform changes proposal weights only and uses the common original weighted guard. Expanded covariance whitening uses cutoff1e−6 and production sigma; changed norm/effective strength prevents an isolated-rank claim.
- Diagnose whether lower drift sacrifices repair, whether A and D differ, whether same-state scalar explains directional effects, and whether rounding/weighting/expanded sketch helps consistently. These local observations do not identify a unique endpoint cause or convert to a fraction of latency/calls. Do not select another variant from diagnostic hindsight.

## 5. Work, resource and cost accounting

- Main response bounds: **3,072 ordinaryF/B +6,144 additionalF/2,048 additionalB = at most9,216F/5,120B**. Diagnostic bounds separate:40F/4B perwindow, at most640F/64B. Count attempted failures, head projections/chunks/checkpoint recomputation, teacher preparation, startup/compile and extra work separately.
- Confirm main feedback512 and timed serving3,072, with no unbudgeted missing-reference generations. Existing canonical proofs avoided the optional reference-generation allowance; audit actual count.
- Separate preparation/capture/teacher work, ordinary training, geometry/VJP/solve/guard, feasibility/scalar scans/copies, observer diagnostics, loading/startup/persistence/export/activation and total campaign wall time. **Do not sum nested timers.** Intrinsic training is recorded training envelope minus diagnostic observer; memory peaks may include diagnostics/export and must be labeled.
- Compare fresh within-dataset serving contemporaneously. Historical KPLUS4 calls may be used only after exact provenance/panel/runtime agreement; never treat historical latency as fresh learning progress. Current endpoint speed is checkpoint-ready, not training-inclusive speedup. Costs are measured implementation costs, not a fundamental lower bound.
- Preserve selected final exports/recovery and audit raw/source/provenance. Any retirement needs a separate exact receipt, preserved compact evidence and disclosure of lost replay/resume capability; status review does not imply cleanup authorization beyond standing rules.

## 6. Prespecified decision and reporting-code limitation

Primary endpoint is **complete-request latency**, FSG versus BOTH controls, separately on BOTH datasets. Strong support requires all four latency-reduction lower95% bounds above zero **and positive useful-progress/call effects**. This does not establish seed robustness, unseen-panel generalization or conference-level validation by itself, and does not authorize automatic longer training.

User's clarified wording governs: **if FSG and SCALAR both improve over KPLUS but FSG has no demonstrated advantage over SCALAR, directional necessity remains unestablished; this does not establish equivalence.** If SCALAR's improvement is also unresolved, a shrinkage explanation is not established.

Clear regression with numerically faithful, genuinely active corrections is a **valid negative result** under original handoff§11: retain K+, and do not extend to58events simply because the mechanism is elaborate. Mixed domains or small unresolved effects remain mixed/inconclusive; report each dataset rather than pool away harm. Lower fittedKL or accepted activity alone never establishes success.

**Reporting-code caveat:** `statistics.recommend()` contains the two positive branches but no dedicated negative branch. Its fallback string is `inconclusive_or_mixed_pilot` even for clear regression. Preserve the raw automated output, but the reviewed narrative must use the actual per-dataset effects and the original handoff's explicit negative-result branch. Do not weaken demonstrated harm into mere uncertainty because of that generic label; no scientific-code change is needed for this review.

## 7. Final delivery and research integrity

Publish additive final report to PR33 with exact execution/publication identities, sealed contract, referenced qualification history, raw endpoint-repeat and paired-request tables, all1,024filtered-window traces or compact tables, all16diagnostic tables, scientific checkpoint identities, separated costs and source/state/numerical/work audits. Keep private prompts/tokens/tensors outside Git and preserve authenticated private availability. Keep GSM8K interim unchanged as the earlier scoped record; final report supersedes only its pending-MathQA/full-review status.

State same-team authorship overlap and what arithmetic was independently reproduced; do not imply independent GPU replication. Use ARS **ANALYZED** for evidence-only review, with explicit zero new model/GPU work. Check all11 fallacies (strata/Simpson, occurrence-to-request ecological inference, selected windows/Berkson, activity-conditioned collider selection, base-rate denominators, regression-to-mean, survivorship, multiple comparisons, forking paths, causal overclaim, reverse causality). Update experience/current status only after audited results; no automatic research-stage or mechanism-selection change.

## Sources inspected

- `research/dflash_functional_step_geometry_20261108/HANDOFF.md`, especially§§1,3–11 and mathematical limits.
- `bf16_realization_r1_001/HANDOFF.txt`, especially§§2–7.
- Execution source `experiments/functional_step_geometry/statistics.py` (`reduce_stage`, `summarize`, `recommend`) and inherited evaluation contracts.
- Prior reviewed GSM8K interim and its endpoint/production/diagnostic cross-audits, published `e8f7c79`.
