# Qualification004 work and outcome review

Verification Status: **ANALYZED**. Execution `90666a76300da2200cb2062b7bc41ee4327f0dfb`. This same-team retained-evidence audit includes implementation authorship overlap (continuation and work helpers). It is not independent GPU replication. The reviewer performed no GPU/remote work or scientific source edits.

## Decision

The remaining correctness checks pass under the explicitly revised numerical protocol and corrected startup procedure. **Both representative FSG proposals fall back to ordinary K+; no pilot starts.** This is a pass for the exercised correctness paths, not evidence of useful FSG adaptation. The strict BF16 norm guard remains unchanged.

| Dataset | Native norm | Rounded FSG norm | Excess | Serving calls for every method |
|---|---:|---:|---:|---:|
| GSM8K, retained FSG case | 1.458325859395869 | 1.4584726072099505 | 0.0100627588% | 19 |
| MathQA, newly executed case | 1.1423660014910397 | 1.1424440154990803 | 0.0068291605% | 20 |

Nontrivial accepted representative corrections: **0/2**. Both SCALAR cases independently calculate the same rejected raw FSG proposal and fall back **before scalar norm matching**. Thus active native scalar-search behavior and accepted-correction efficacy were not exercised. Rejection occurs before candidate KL evaluation; missing candidate KL must not be reported as passing or failing. The synthetic maximum-shape resource fixture is not an accepted representative correction.

The eight serving executions have identical within-dataset token IDs and EOS termination across the four methods. GSM8K outputs contain127 tokens and108 accepted draft tokens; MathQA175 and155. Native receipts validate these against the saved canonical outputs and check the neutral proposals/verification traces. These one-request-per-dataset checks establish output preservation after fallback, not a timing benefit or statistical equivalence.

## Exact work accounting

| Work | Prior ledger | New continuation | Final cumulative | Ceiling |
|---|---:|---:|---:|---:|
| Adam applications | 3 | 5 | 8 | 8 |
| Ordinary response forwards | 8 | 10 | 18 | 18 |
| Ordinary response backwards | 8 | 10 | 18 | 18 |
| Additional forwards | 34 | 27 | 61 | 67 |
| Additional backwards | 20 | 16 | 36 | 36 |
| Serving generations | 0 | 8 | 8 | 8 |

The latest failed qualification ledger is a byte-exact prefix of this final ledger. All new attempts completed; none is unknown or failed. The only new Adam cases are GSM SCALAR and Math KPLUS/FSG_ZERO/FSG/SCALAR. The prior three GSM Adam applications were reused, never repeated.

New additional work decomposes into **2F/2B discarded GSM startup**, **2F/2B ordinary-reference recapture**, **2F/2B discarded Math startup**, **16F/8B across four filter hooks**, and **5F/2B for the fresh Math same-layout oracle**: total27F/16B. Six potential candidate-scoring forwards were skipped because three raw proposals failed the norm guard; these were branch maxima, not incomplete required work. GSM formula/resource checks were reused with historical cost labels and zero new forwards.

Measured new training-process physical counts reconcile to **37 draft forwards,185 layer forwards,3,829 vocabulary-head calls,614 proxy calls and1,142 ordinary-objective chunks**. These are overlapping accounting levels, not quantities to add into a single pass count. Serving target/draft execution is separate; the eight generations are not included in these training counters. Copied historical GSM FSG_ZERO physical/cost fields were excluded.

Earlier standalone ordinary-gradient debugging remains separately charged at **28F/20B/zeroAdam** across its failed and repaired attempts. Earlier layout/precision/consequence diagnostics likewise remain outside qualification ceilings. No earlier failure is erased or relabeled a pass; historical cross-layout numerical tests remain failed under their original criteria.

## State, numerical and serving checks

The repaired GSM reference now has separately discarded startup with exact state restoration. The resumed SCALAR ordinary gradient matches across58 tensors/1,048,626,432 coordinates, and its native endpoint/Adam state matches across239 tensors/3,145,885,306 coordinates. Equivalent Math neutral/FSG/SCALAR controls pass. Decoded state, restored live state, response/anchor/loss parity, optimizer counters/moments, RNG, runtime and export comparisons pass where required. Both dataset startup restoration receipts pass. No optimizer seed/reset substitution was introduced.

The fresh Math independent native formula oracle uses9 retained anchors in physical chunks8+1, with unchanged tolerance parameters. It has0 violations and0 relative-L2 discrepancy; GSM's prior same-layout pass is reused. Intermediate `SAME_LAYOUT_ORACLE.json` has status`running` because it precedes cleanup; final `oracles/RESULT.json` is the authoritative passed result. These bounded same-layout passes do not establish arbitrary-layout equivalence.

All **7,108 dependencies** pass preflight/postflight. I independently rehashed **201 compact files /11,441,361 bytes** with zero mismatches. The automated ledger/receipt audit passes252 checks; an additional review checked all fresh exact-state receipts and within-dataset native token equality. Model-sized artifacts are authenticated through retained remote receipts, not rehashed again locally.

Five6,291,868,947-byte disposable state payloads were retired after exact decoded/export checks, with individual retirement receipts. They were qualification artifacts, not production recovery checkpoints; they are deleted, not archived. Original origins/recovery states and final serving exports remain available. Six exports were written in this attempt: five new disposable endpoints plus a re-export of the retained ordinary endpoint.

## Costs and resources

Supervisor: **02:41:26–02:58:48 UTC,9November2026**, **1,042s (17m22s)**, exit0. Runner: **896.196s**. Preflight/postflight: **50.056/50.313s**. The launcher CPU suite passed **188 tests in31.743s**. CPU tests/authentication/postcheck sit outside the runner timer; per-case/oracle/hook/export timers are nested components and must not be summed again. Startup work is separately counted, but its total wall time is not isolated by a dedicated timer, so no startup latency is inferred.

Fresh training-process peak allocation/reservation: **38,420,866,048/40,342,913,024 bytes**. The reused synthetic resource fixture's historical47.018/48.564GB peak is separate, not a new measurement. This does not describe an independently measured peak of the serving worker.

Actual retained preflight free space was **55,338,508,288 bytes (55.3385GB)**, against42,949,672,960 bytes required for this qualification including8GiB reserve. Postflight recorded **42,751,885,312 bytes**, safely above reserve. This is not admission for another launch or the main pilot. A55.525GB reading in the draft report refers to an earlier session reading and should be replaced by the exact retained preflight value. Postflight dependency receipt alone does not establish idle GPUs; a separate final-status receipt is needed for that claim.

## Public report cross-check

The draft `../REPORT.md` correctly separates repaired startup, same-layout qualification, reused resource evidence, active-case fallback and no-pilot decision. Work totals and17m22s/896.196s durations match. Requested minor provenance corrections: use actual55.3385GB preflight free space, and persist/link final idle-GPU evidence if reporting that both GPUs were idle. No scientific or accounting discrepancy was found.

## ARS interpretation checks

- **Simpson paradox:** Keep GSM8K and MathQA qualification outcomes separate; do not pool a serving efficacy estimate.
- **ecological fallacy:** Two representative windows do not establish inactivity on all requests or eventual pilot trajectories.
- **Berkson bias:** The fixed qualification cases were not replaced according to outcomes; synthetic resource construction is explicit.
- **collider bias:** All-fallback correctness cases do not establish performance of accepted corrections.
- **base-rate neglect:** Report0/2 representative accepted FSG proposals,8/8 serving checks, and all earlier failed work.
- **regression to mean:** Startup repair is a demonstrated execution correction, not evidence of an improving learning trajectory.
- **survivorship bias:** Original cold-gradient and cross-layout failures remain retained; no earliercosts/attempts discarded.
- **look-elsewhere effect:** No norms/tolerances were adjusted to observedexcess; strictguardacceptedno extraattempts.
- **forking paths:** Explicit source/plan and revisedcumulativebudget preserve everypriorattempt; no pilot follows automatically.
- **causal attribution:** BF16 realizednorm explains the guarddecision on these cases; this does notprovegeometryineffective.
- **reverse causality:** Equal serving counts follow unchanged keptendpoints; they do not validate an activecorrection mechanism.

Evidence: `retained/qualification/QUALIFICATION.json`, cumulative ledger and sealed prior ledger, five fresh case directories, both serving directories, startup receipts, pre/postflight and compact manifest. Detailed audit checks are in `AUDIT_CHECKS.json`; no further execution is authorized by this review.
