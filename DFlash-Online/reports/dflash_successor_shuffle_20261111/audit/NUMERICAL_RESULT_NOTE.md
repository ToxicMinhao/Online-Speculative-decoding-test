# Independent successor-shuffle result audit

## Material Passport

ARS3.24.2 experiment-agent / validate. Verification status: **ANALYZED**. The independently implemented raw-scalar audit passed on its first result run in 0.928463s. No production reduction helpers or model imports, remote operations, native reruns or bulk tensor reads. Same-team implementation-author overlap is disclosed.

Source: published execution `4d977e9ffd2cab9d58ae8806c29df11d843353da`, local hash-verified delivery (5,013 files /454,125,542bytes). [Audit JSON](/home/minhao2/EAGLE_SPEC/research/dflash_successor_shuffle_20261111/analysis/SUCCESSOR_AUDIT_001.json) retains raw input identities, exact command and paired request values. [Numerical note JSON](/home/minhao2/EAGLE_SPEC/research/dflash_successor_shuffle_20261111/analysis/NUMERICAL_RESULT_NOTE.json) retains all contrasts, repeats and costs.

**The prespecified primary call contrast supports informative placement for this one fork and fixed permutation. Incremental benefit versus ordinary K remains unresolved; no complete-request latency contrast demonstrates benefit or slowdown.** The bounded experiment is finished; no automatic follow-up is authorized.

## Endpoint

All256 requests /1,536 timed executions validate. Invalid, unknown and unattempted counts are zero. There are no cap-hit or seed-EOS requests. Calls are counted once after repeat agreement; times average the two repeats within each request. The joint paired-request bootstrap uses10,000 PCG64 draws, seed20261111, linear unadjusted95% intervals. Positive reduction favors the candidate.

| Contrast | Call reduction [95% CI] | Complete-time reduction [95% CI] | Call wins/losses/ties | Complete-time sweeps |
|---|---|---|---|---|
| TRUE vs SHUFFLED | +0.595289% [+0.170964, +1.015408] | +1.245038% [-0.140408, +2.562571] | 111/78/67 | +1.949924%, +0.531610% |
| TRUE vs K | +0.363196% [-0.071936, +0.784491] | +0.357830% [-0.979443, +1.653258] | 96/75/85 | +0.932937%, -0.222639% |
| SHUFFLED vs K | -0.233483% [-0.653537, +0.195408] | -0.898393% [-2.274756, +0.451523] | 82/105/69 | -1.037212%, -0.758280% |

Calls total TRUE11,522, SHUFFLED11,591 and K11,564. TRUE versus SHUFFLED saves189 calls and adds120, net69; versus K it saves166 and adds124, net42. These are pooled ratios, not means of request ratios. TRUE/SHUFFLED's complete-time sweeps are both favorable, but its interval includes zero. TRUE/K's sweep directions differ.

Summed per-request mean complete times are TRUE207.942331s, SHUFFLED210.563931s and K208.689084s; per-request means are0.812275s,0.822515s and0.815192s. Native-generation reduction TRUE/SHUFFLED is+0.521167% [−0.146464,+1.198316]; TRUE/K+0.020677% [−0.662244,+0.681174]. The recorded cache-maintenance contrast TRUE/SHUFFLED is+19.314811% [+1.418006,+33.868271], a secondary unadjusted component result that does not establish complete-request acceleration or unique causal attribution.

The first-native-round diagnostic across all256 requests favors TRUE over SHUFFLED on23 prefixes and loses on20, with213ties,39tokens gained and53lost (net−14). This limited start-local observation does not explain the favorable full-request call contrast. The separate64-request parent-matched analysis is outside this scalar audit.

## Training, manipulation and accounting

The full original PR37 response/anchor schedule and authenticated TRUE640 identity match. K and SHUFFLED each perform128 Adam updates and256 presentations (total256updates/512forward-backward presentations); reused TRUE performs zero new main updates. Separate main startup adds4forward/backward passes withzero updates. Qualification adds3disposable updates,12forward/backward passes including its startup, and3sampler-only draws. Total newly executed training/qualification response work is528forwards and528backwards; no main model state comes from qualification.

All256 SHUFFLED manipulation observations reconcile:45,255eligible errors,2,286ineligible,3,396mixed strata and18,452changed bits, versus18,416.406527 uniform-permutation expected changes. Native recorded reassigned weight sums to1,827.414519. The final export differs from TRUE. Same-forward scalar fields, pinned donor hashes and class/mass/coefficient invariants pass. This is substantial actuation, not evidence by itself of efficacy.

SHUFFLED's7,994extra detached projections take50.372672s; assignment takes8.139749s. Intrinsic training clocks are K328.064734s and SHUFFLED400.984953s. The observed difference includes both K+ machinery and shuffle overhead; it is not solely the cost of assignment. TRUE is historical reuse with no new training cost in this execution.

Current run wall is**3,330.467275s (55.507788min)**. Sequential qualification wrappers total432.147379s; main K/SHUFFLED wrappers480.478028s/554.585870s. Missing-reference preparation is66.623824s (35new/221reused), and64untimed parent blocks take26.438241s. Serving worker wall is1,669.717516s, enclosing1,633.331483s evaluation,20.756898s loading,161publications with302.128923s outer publication clocks, and1,254.390692s summed timed-request clocks. Nested clocks must not be added again or labeled kernel time. No prior failed-launch receipt was supplied to this audit; this is not an imputation that historical failure costs are zero.

## Verification and interpretation limits

All52,224 raw/CSV fields preserve identical values and order. All129 checked report numerical fields,12bootstrap intervals,repeat totals,classification and161publication identities match exactly. No scientific-data discrepancy or result-audit repair was required. Native output/weight/state correctness relies on the separately retained native validators and postflight; arithmetic agreement is not a second native model run.

All11 ARS fallacy categories were considered: (1) pooled and sweep directions are both retained, including the TRUE/K timing reversal; (2) pooled gains do not imply every request wins; (3) this input-selected panel is not a population-random sample; (4) no filtering by post-treatment success is used; (5) denominators, ties and gross gains/losses remain explicit; (6) no favorable pilot/subgroup was selected for confirmation here; (7) all sealed executions are accounted for, with no complete-case filtering; (8) all3contrasts and latency components are retained, with unadjusted intervals disclosed; (9) seed, panel, donor rule and bootstrap were fixed and not retuned; (10) conclusions are bounded to this continuing training seed and permutation, without claiming unique downstream mechanism or general latency benefit; (11) no reverse-causal explanation is inferred from local diagnostics.

An unresolved TRUE/K interval is not equivalence. The supported TRUE/SHUFFLED call result does not demonstrate a new improvement over the incumbent TRUE rule, nor a training-inclusive speedup. Stage1/projectNone and the original stopping rule remain unchanged.
