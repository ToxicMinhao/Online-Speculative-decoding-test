# R1 supplementary qualification: numerical and work audit

## Material Passport

- ARS-Codex 3.24.2, experiment-agent validation, **ANALYZED**.
- Execution: `95aeb869189da2d7d4f2109b93c05fd3bd9b7944`; revision `norm_feasible_bf16_r1`.
- Scope: source review and arithmetic/accounting checks of retained compact native receipts. No model reload, new GPU work, or independent native replication.
- Reviewer is the same-team agent who implemented the runtime integration and pilot diagnostic runner. Mathematical helpers were implemented by another agent. This authorship overlap and shared context preclude a claim of independent implementation review.
- Sources: `../retained/qualification/QUALIFICATION.json`, all four case `RESULT.json` files, exact comparison/export receipts, raw work ledger, serving records, pre/postflight, and original supplementary handoff. This audit independently rehashed all 206 retrieved files.

## Disposition

**No numerical or accounting blocker found in the completed supplementary qualification.** All 641 receipt/source-bound assertions pass. The four declared live cases and four native serving checks completed within the new allowance. Both FSG cases commit nontrivial feasible corrections; both SCALAR cases exercise their own matching and final replay, then restore native because their measured KL change lies in the original ambiguity band. These outcomes satisfy the supplementary mechanism-activity conditions. Pilot launch still requires its separate native-binding and peak-storage admission; qualification does not establish a serving benefit.

## Norm and decision checks

| Dataset / case | Native norm | Raw BF16 norm | R1 scale | Final proposed norm | Protected KL: native → candidate | Final decision |
|---|---:|---:|---:|---:|---:|---|
| GSM8K FSG | 1.458325859396 | 1.458472607210 | 0.992445290089 | 1.458325857786 | 0.072695263441 → 0.051821266858 | Commit FSG |
| GSM8K SCALAR | 1.458325859396 | 1.458472607210 | 0.992445290089 | 1.458325857788 | 0.072695263441 → 0.072695263446 | Ambiguity; native fallback |
| MathQA FSG | 1.142366001491 | 1.142444015499 | 0.996576011181 | 1.142366000079 | 0.129428663372 → 0.095945269270 | Commit FSG |
| MathQA SCALAR | 1.142366001491 | 1.142444015499 | 0.996576011181 | 1.142366000077 | 0.129428663372 → 0.129428663388 | Ambiguity; native fallback |

Each R1 realization performs exactly 24 probes, selects the lower endpoint of a bracket of width `2^-24`, and ends with `n(lo) <= U < n(hi)`. Its written norm exactly equals the recorded lower-endpoint probe norm. Every case records 26 construction/norm scan passes, two identity scans, and two writes; scan categories overlap and must not be summed as distinct physical passes. Source inspection verifies that both probes and writes call the same `candidate_chunk` and reconstruct the original FP32 continuous direction before BF16 casting, rather than scaling a previously rounded displacement. This is source-and-receipt evidence, not an independent reconstruction of full parameter tensors.

FSG changes 1,957,575 parameter coordinates relative to native on GSM8K and 2,085,900 on MathQA, while also remaining distinct from the prestate. Its pooled protected-KL reductions of 28.71% and 25.87% describe the fitted qualification windows; lower protected KL is a guarded mechanism property, not evidence of held-out usefulness.

SCALAR's target is its own feasible hypothetical-FSG norm, saved before parameter replacement. Its original 24-step closest-bracket matcher remains separate from the R1 lower-feasible-bracket search. Actual norm errors relative to that target are approximately `2.137e-12` and `2.097e-12`. The scalar candidates differ from native in 4,729 and 4,209 coordinates, so both really reach final-candidate scoring. Their KL differences (`+5.50e-12`, `+1.63e-11`) fall inside the original approximately `1.6–1.8e-9` per-risk deadbands. Exact-native fallback is therefore the specified outcome, not a correctness failure or proof that the updates are equivalent. No hypothetical FSG KL veto is applied to SCALAR.

The audit independently recomputes pooled risks from both retained response terms, checks finite quantities and the exact strict acceptance/ambiguity formula, and verifies committed endpoint identity/norm consistency. Per-response records remain available, including protected counts 783/1,830 (GSM8K) and 1,442/3,377 (MathQA), using all 126/247 and 174/512 anchors respectively.

## Scientific state and native output evidence

All four live cases report exact raw ordinary-gradient agreement with the warmed reference, over 58 active BF16 tensors / 1,048,626,432 coordinates. Each ordinary native endpoint matches its authenticated historical export. The first live native Adam state supplies the retained fresh optimizer-state reference; it is not a reconstructed retired optimizer payload. SCALAR's fresh native state matches that reference, and each auxiliary path preserves its ordinary Adam state, RNG, counters, accumulation, gradients and runtime state. All decode/live-restore and final-export membership/value checks pass. Initial origins remain step 512; each disposable case advances Adam once. No qualification checkpoint initializes the pilot.

All four serving outputs match their retained canonical tokens and termination. GSM8K uses 19 calls for both exports; MathQA uses 22 for FSG versus 20 for SCALAR/native. The latter regression remains visible. These are one-request output-preservation checks per dataset, not an endpoint performance panel or evidence that FSG improves drafting. Active corrections may change proposals and call counts under the contract; no native-proposal/count equality is imposed on them.

The earlier failed cross-layout numerical checks and historical all-fallback qualification remain failed/inactive in their original records. This R1 run uses the separately authorized realization revision, unchanged native BF16 head layout and unchanged norm/KL criteria. It does not retrospectively pass the old gates.

## Work and costs

| New work | Attempted | Completed | Declared ceiling |
|---|---:|---:|---:|
| Adam applications | 4 | 4 | 4 |
| Ordinary response forwards | 8 | 8 | 8 |
| Ordinary backwards | 8 | 8 | 8 |
| Additional response forwards | 32 | 32 | 32 |
| Additional backwards | 16 | 16 | 16 |
| Serving generations | 4 | 4 | 4 |

The additional work comprises four discarded startup forwards/backwards, four warmed reference recapture forwards/backwards, and the four live cases' 24 geometry/candidate forwards plus eight VJPs. No feedback generation, teacher preparation, retry, unknown work or additional Adam step appears in the raw ledger. Physical draft/layer/head/chunk counts are retained separately. Metadata preparation failures preceding model execution consumed zero model work and remain visible.

Runner time is 945.407 s. Peak observed allocated/reserved GPU memory is 38.422/40.647 GB; this is a qualification observation, not a maximum pilot guarantee. R1 feasibility construction/search costs 26.22–26.57 s per measured window; SCALAR matching adds 14.88–14.91 s. The 52–68 s hook totals also include qualification observers, so multiplying them directly would overstate intrinsic update cost. Component times and costs remain separately available in the audit JSON and original records.

## Pilot duration estimate

For sequential execution of the unchanged two-dataset, three-arm, two-event pilot, **plan roughly 12–18 hours after actual launch, with about 15–18 hours the safer expectation if most filtered steps require search**. This excludes resolving the separate storage admission before launch.

The calculation uses 256 future steps for each of the four filtered dataset/arm combinations: measured feasibility and SCALAR parameter scans alone extrapolate to approximately 9.62 hours. The separately clocked geometry components add about 2.20 hours. Historical PR32 K+ training for all six two-event arm trajectories adds approximately 1.39 hours; historical shared-feedback and endpoint-serving rates add approximately 1.04 hours. This yields roughly 14.24 hours before unclocked endpoint scans, 16 diagnostic windows, snapshot writes, loading and publication. No-search/bypass frequency can lower this; longer blocks, I/O and diagnostics can raise it. One natural measured window per dataset/arm is insufficient for a tight bound. This is scheduling context, not an observed end-to-end cost or equal-compute comparison.

Historical timing source: `DFlash-Online/reports/dflash_role_loss_pilot_20261107/REPORT.md`, `TRAJECTORY_COSTS.csv` and `REPEAT_TOTALS.csv`, execution `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`. Historical and fresh timings are not pooled into any performance claim.

## Interpretation checks: 11/11

| Check | Disposition |
|---|---|
| Simpson's paradox | Both datasets are reported separately; no pooled efficacy ranking. |
| Ecological fallacy | Two windows and four serving checks do not represent the future trajectory. |
| Selection/Berkson bias | The original two cases are fixed; no favorable case replacement. |
| Collider bias | No outcome-conditioned subset or covariate adjustment is used. |
| Base-rate neglect | Both FSG and both SCALAR outcomes and all work denominators are retained. |
| Regression to mean | No causal improvement claim is made from qualification-before/after KL. |
| Survivorship bias | All four cases complete; historical failures remain disclosed. |
| Look-elsewhere effect | No performance endpoints or alternatives are selected from these checks. |
| Forking paths | R1 is an explicit authorized revision; old policy/result bytes remain unchanged. |
| Correlation versus causation | Guarded KL/activity is not claimed to establish native performance or directional advantage. |
| Reverse causality | The prescribed state/update/measurement order is checked; future benefit remains untested. |

Artifacts: `RUNTIME_NUMERICAL_WORK_AUDIT.json`, `audit_r1_runtime.py`, `RUNTIME_AUDIT_RUN.log`. No source, checkpoint, algorithm, tolerance or scientific state was changed by this audit.
