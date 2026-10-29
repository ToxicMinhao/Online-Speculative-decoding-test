# O/K events 5–8 raw arithmetic review

Material Passport: ARS experiment-agent validate; **ANALYZED**. This is read-only reconstruction of existing evidence, not a new GPU replication. The script was fixed before outcomes and uses a separate standard-library calculation path. Its author also wrote the evaluator/report; this is same-team shared-context review, not an external or blinded review.

**Result: passed, with zero discrepancies in 1,356,407 checks.** All 1,920 native executions, 89,541 verification blocks and 320 request-endpoint pairs agree with the report. The audit consumed and hashed 1,948 inputs. Scientific execution is `04fac901f7af5478b62ff730833c026637ddd2cd`; operational recovery and report analysis are `1cfb44fa3989e5ed2c5e13805007bb20f22189f8`. Source/state/retention verification is a separate review.

K’s call advantage persists at all three measured endpoints on both existing panels, but its event-8 advantage is smaller than at event 4. Development changes **1.7368% → 3.1031% → 1.1668%**, a net decrease of **0.5700 percentage points**. Additional changes **2.3003% → 0.7777% → 1.5589%**, a net decrease of **0.7414 points**. Both paths are nonmonotonic. The fresh event-8 panel favors K by **3.3367%** (2,997 versus 2,897 calls). These are descriptive results from one continued O/K training trajectory.

| Event | Panel | O/K calls | Call reduction | Call wins/losses/ties | Request-time reduction | Native-time reduction | Request-time wins/losses/ties |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | development | 2997/2904 | 3.1031% | 36/18/10 | 3.2116% | 3.0213% | 43/21/0 |
| 6 | additional | 3086/3062 | 0.7777% | 31/21/12 | 0.5287% | 0.7093% | 36/28/0 |
| 8 | development | 2914/2880 | 1.1668% | 31/21/12 | 0.8542% | 1.1284% | 39/25/0 |
| 8 | additional | 3079/3031 | 1.5589% | 33/25/6 | 1.4392% | 1.4972% | 36/28/0 |
| 8 | fresh | 2997/2897 | 3.3367% | 37/13/14 | 3.2227% | 3.2092% | 42/22/0 |

Every row contains 64 valid requests and 384 timed executions. Primary counts use one validated repeat-0 trajectory per request and arm after exact count consistency across repeats. Pooled reductions use summed O calls as the denominator; timings use the sum of per-request three-repeat means. Timing repeats are repeated measurements of the same requests, not independent examples. Request-time wins compare those means.

| Event | Panel | Accepted draft tokens/call O → K | Committed progress/call O → K | Gross calls saved / added |
|---:|---|---:|---:|---:|
| 6 | development | 5.521188 → 5.729683 | 6.500834 → 6.709022 | 127/34 |
| 6 | additional | 5.456902 → 5.506858 | 6.436811 → 6.487263 | 77/53 |
| 8 | development | 5.705216 → 5.784028 | 6.685999 → 6.764931 | 87/53 |
| 8 | additional | 5.469958 → 5.572418 | 6.451445 → 6.553613 | 113/65 |
| 8 | fresh | 5.639640 → 5.867794 | 6.621622 → 6.850190 | 121/21 |

Accepted draft tokens are independently reconstructed before and after stop clipping. Committed progress excludes the initial seed token; each block has progress in [1,16], accepted draft tokens in [0,15], and accepted tokens no greater than committed progress. EOS/length termination, target posterior tokens, committed output tokens, absolute positions, anchors and counter totals all agree. Native acceptance (`progress − 1`) is checked separately from stop-clipped accepted draft tokens.

Retained development F and additional K1 canonical outputs/termination match exactly. Each fresh request uses its own first O8 repeat-0 output as canonical, and all six O/K outputs match it exactly. The 384 fresh executions include these 64 canonical-producing executions; no extra reference slot is present. The first eight development requests at event 6 account for the included 48-execution correctness gate. No historical O4/K4 count gate is applied to new checkpoints.

| Event | Panel | Complete-request reduction, repeats 0 / 1 / 2 | Native-generation reduction, repeats 0 / 1 / 2 |
|---:|---|---:|---:|
| 6 | development | 3.2646% / 3.1658% / 3.2045% | 3.0596% / 2.9934% / 3.0110% |
| 6 | additional | 0.5632% / 0.5938% / 0.4291% | 0.7333% / 0.7316% / 0.6630% |
| 8 | development | 0.8669% / 0.8499% / 0.8457% | 1.1283% / 1.1372% / 1.1198% |
| 8 | additional | 1.4647% / 1.5168% / 1.3362% | 1.5031% / 1.5347% / 1.4538% |
| 8 | fresh | 3.2163% / 3.2742% / 3.1776% | 3.2024% / 3.2230% / 3.2020% |

All 15 pooled comparisons favor K for both complete-request and native-generation time. Cache-maintenance directions vary; their exact totals and repeat reductions remain in the JSON review and audited tables. These nested timers must not be added together or interpreted as a kernel breakdown. The repeat consistency supports the reported within-evaluation direction; it is not a significance test or seed-robustness result. Historical event-4 timings are absent from the call curve. Update/save/export costs are kept separate by the report and are outside this arithmetic audit’s native-evaluation scope.

ARS fallacy scan: **11/11 checked**.

- **Simpson’s paradox:** Panels and endpoints remain separate; each of the five measured strata favors K in pooled calls and both primary timing measures. No combined-panel claim is used.

- **Ecological fallacy:** Paired request wins/losses/ties and gross savings/regressions are retained; a positive pooled reduction does not mean every request improved.

- **Berkson’s paradox:** The fixed eligible task/domain and exclusion recipe limit generalization; the fresh panel is new data from that recipe, not a new model, task family, or training seed.

- **Collider bias:** No performance-based request filtering is applied. All 320 request-endpoint pairs and all 1,920 executions pass, so no outcome-dependent validity cohort is selected.

- **Base-rate neglect:** All pooled reductions expose summed O denominators. Per-request percentages are not averaged to obtain the primary result.

- **Regression to the mean:** Endpoints and panels were fixed before outcomes, with matched O/K comparisons; this does not convert a nonmonotonic curve into a causal training-age effect.

- **Survivorship bias:** Complete six-slot coverage is present for every request-endpoint. Operational stop/transfer failures are preserved separately; they are not represented as scientific retries or discarded bad endpoints.

- **Look-elsewhere effect:** All five prespecified event/panel strata and all three timing repeats are shown; no best-repeat or best-panel selection and no significance claim.

- **Garden of forking paths:** The audit script hash matches its pre-outcome readiness receipt. No denominator, timing aggregation or endpoint rule changed after reading outcomes.

- **Correlation versus causation:** These are measurements of one controlled continuation. The curve alone does not establish a general training-duration effect or adaptation-inclusive asynchronous speedup.

- **Reverse causality:** Checkpoint order precedes evaluation and no reverse-direction causal inference is made; observational curve shape is not assigned a mechanism.

Evidence: `independent_raw_audit/AUDIT.json` binds all consumed files; its tables retain every request and repeat. `INDEPENDENT_RAW_RESULT_REVIEW.json` contains repeat comparisons and file hashes. The audit script hash matches the pre-outcome readiness receipt. No raw token, margin or anchor arrays are included in this review. Stop remains event 8; no additional mechanism, tuning, deployment or project-direction inference follows.
