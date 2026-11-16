# Persistent prefix withholding v2: qualification stopped

## Material passport

- Task: user-authorized persistent N/E/S feasibility campaign, handoff version2.
- Workflow: ARS experiment execution and retained-evidence audit; Stage1/projectNone unchanged.
- Data interpretation: **DEVELOPMENT_FEASIBILITY_REUSED_DATA**.
- Execution commit: `e032cbc1fba90a7a6aa6a3460bc0a8c954810497`.
- Subsequent CPU-tested repair commit: `e50d185dc36efb6fa390026251046e30bef1033d`; not executed on GPU.
- Result: **INCOMPLETE_NO_EFFICACY_CLAIM**. No production event or complete-response generation ran.
- Original v1 readiness/data-gate failure and both v2 attempt records remain unchanged.

## Intended comparison

Carry prefix-gradient withholding through E7/E8, comparing final E8 with native N8 and fixed lower-LR S8. E retains native K+ scalar loss, selection, weighting and normalization but stops the direct current-prefix row gradients. S uses the single declared 0.96 LR multiplier; it is not a per-step norm match. Original N7 is reused. Each arm would generate its own event8 feedback on the same128 inputs, then final N8/E8/S8 would receive1152 balanced scored executions on one quiet physical card.

The sealed128-request panel contains128tier-A requests, all with known older-project exposure. Strict current-adaptation exclusions pass.104 compatible native-verifier references are authenticated for reuse;24 fixed original-K6 references were planned. Neither references nor candidate serving were generated in this attempt. This is a reused-development feasibility comparison, not fresh confirmation.

## What passed, and where execution stopped

The first CPU attempt at `116ae86` stopped because the environment checker compared PyTorch wheel metadata (`2.13.0`) against an imported-module version pin (`2.13.0+cu130`). Actual imported versions match. The narrow checker repair is documented in [CPU_PREFLIGHT_FIX.md](CPU_PREFLIGHT_FIX.md); the failed attempt consumed zero model work.

Attempt002 CPU preflight passed in58.021s: original K6/N7 complete recovery, exports, scientific ancestry, runtime sources, physical GPU identities, S migration and storage admission all passed. It then ran two of the eight disposable GPU qualification cases on physical A800A. The second case, ALL-enabled new adapter versus legacy native K+, failed the required endpoint equality check. Execution stopped automatically. CardB and E/S qualification were unattempted; both cards are now idle.

| Retained comparison | Legacy native N | ALL-enabled adapter | Result |
|---|---:|---:|---|
| Forward values, anchors, selection, weights and denominator | Reference | Identical | Pass |
| Pre-clipping gradients | Reference | Identical for all58 tensors | Pass |
| Adam first/second moments | Reference | Identical for all58 states | Pass |
| Scientific RNG | Reference | Identical | Pass |
| Trainer global step / presentations |769 /1538|769 /1538|Identical |
| Adam internal step, all58 counters |769|770|**Fail** |
| Endpoint weights |Reference|36of58 tensors differ|**Fail** |

The recorded tensor identities permit exact equality/count comparisons; they do not retain full endpoint tensors for an absolute weight-error magnitude. No additional model-sized snapshots or GPU diagnostics were created to obtain one.

## Confirmed implementation cause and narrow repair

The disposable qualification restore reused CPU Adam step tensors by reference. Installed PyTorch2.13.0's `Optimizer._process_value_according_to_param_policy` deliberately returns the original `step` tensor for noncapturable/nonfused AdamW; moving the model and moments to CUDA does not copy this scalar. The old adapter only made an independent backend copy for CPU fixtures. Thus the first case advanced the in-memory restore source from768 to769; the second started with769 and advanced to770 despite its trainer counter being reset to768. Adam's bias correction consequently used a different step number.

Evidence is stronger than a numerical-sensitivity hypothesis: every moment and preclip gradient matches, every step hash independently resolves to769 versus770, the installed loader contains the direct-return branch, and a tiny CPU regression reproduces that aliasing behavior. This is a qualification state-restoration bug. E's withheld-prefix rule was never exercised, so the failure is not evidence for or against it.

The pending GPU repair clones CPU Adam step scalars for every disposable CUDA restore; normal device transfer still copies moments. CPU fixtures keep independent full state. The restored source counters are checked after each case, and the final restoration oracle is recorded before any case can mutate it. No objective, optimizer arithmetic, precision, tolerance, data or budget was changed. CPU test results are supplied in `CPU_REPAIR_TESTS.log`. **The repaired code has not been rerun on GPU and is not yet qualified.**

A second CPU-reviewed bookkeeping repair permits explicitly authenticated prior source snapshots' internal documentation links during allocation accounting. It still rejects undeclared, external and run-state links and counts their allocated bytes. The campaign never reached this storage boundary; this repair did not cause or fix the observed gradient-control failure.

## Work, time and resources

| Work | Completed | Unattempted within original plan |
|---|---:|---:|
| Disposable Adam applications |2|6 |
| Production Adam applications |0|640 |
| Response forwards / backwards, including startup |6 /6|1304 /1304 |
| Training initializations |1|6 |
| Own-feedback generations |0|384 |
| Missing-reference generations |0|24 |
| Serving-qualification generations |0|8 |
| Final warmup / scored generations |0 /0|72 /1152 |
| Teacher preparations |0|384 |

All completed physical operations are accounted for; no unknown work remains. The **validation check** failed after its two optimizer operations completed, so zero failed physical-operation entries does not mean qualification passed. Four response presentations belong to the two test updates; two discarded startup forward/backward responses used zero Adam steps.

- Occupied-card time:89.016s, entirely on physical A. CPU-preflight-to-stop launcher wall time:148.704s. These are nested intervals, not additive costs.
- Model load28.421s; startup23.350s; qualification cases and observations29.795s. Full qualification-worker wall85.230s.
- Peak allocated CUDA memory17,041,992,704B; peak reserved18,580,766,720B. Sampled process-tree host RSS34,904,080,384B; this is a sampled bound.
- Observed physical hooks:6 draft forwards,30 layer forwards,312 objective chunks and624 vocabulary-head calls, including startup. Projection/checkpoint callback counts are retained separately from response-level work. The raw ledger retains per-category counts; these are not complete generations.
- Original K6/N7 recovery file identities still match. All4412 staged source entries were rechecked after the stop.
- Peak storage plan100.158GB, including prior275,750,912B, under110GB with400GB reserve. Postflight run/source allocation544,468,992B plus16,523,264B observed shared-compiler growth (about0.561GB total); this is sampled usage, not a continuous peak. A800 free704.276GB. No additional historical data was deleted. PR48's earlier75.509GB retirement is already complete and is not counted again.

The pre-run **2–3hour elapsed estimate** remains provisional for a future successfully qualified full comparison. There is no current completion ETA: production is stopped, cross-card/serving/resource qualification is incomplete, and measured full-work admission has not run. The four summed-card-hour and four elapsed-hour caps remain unchanged.

## Interpretation and next boundary

No E/N, E/S or S/N serving comparison exists. No loss, speedup, acceptance improvement or practical-pass claim is supported. The original published raw report and all failure records remain intact; the independent partial reducer passed990 checks with no efficacy computation. These checks are an independent CPU implementation within the same team, not another model execution or statistical replicate.

Do not relabel the failed ALL case as passed or reset consumed work. Any GPU retry requires an explicit bounded decision reconciling the two consumed optimizer applications with the original648-total cap; a fresh eight-case rerun plus the full640-step campaign would total650 attempted Adam applications,1316 response forwards/backwards and8 training initializations. This includes the failed attempt’s six forwards/backwards and one initialization, the fresh qualification’s20 forwards/backwards and two initializations, and production’s1290 forwards/backwards and five initializations. No such expansion or retry has been executed. Stop here with the corrected implementation and evidence for review.
