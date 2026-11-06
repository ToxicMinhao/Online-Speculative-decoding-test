# AdaFlash C1 raw endpoint audit

## Material Passport

ARS experiment-agent, validate mode. Verification status: **ANALYZED; raw-evidence reconstruction passed**. This is a separate same-team stdlib audit, not an independent GPU replication. The auditor also authored the A-loss component; that authorship overlap is disclosed. No experimental code changed, no model was loaded, and no new optimizer step or generation ran.

Execution commit: `681336c5e5c5caa0aa67560e46ffe83ce81e9351`.

Authoritative evidence root: `/data/sources/minhao/runs/dflash_adaflash_c1_20261105/production_001`.

## Validity

All **9,216 raw executions**, **9,216 native sidecars**, and **399,240 verification rounds** passed reconstruction. All 1,024 sealed canonical outputs were also rebound to their original frozen serving records. The auditor performed **3,016,692 checks**, with **zero remaining discrepancies**, reading 28,688 files. It did not import or call the experiment's validator; the saved production audit was checked only as corroborating evidence.

Checks cover exact prompt/output token IDs and termination, checkpoint and request identities, round starts and committed tokens, first-rejection/acceptance accounting, full context and output cap, disabled feedback capture, valid cache-publication receipts, repeated count consistency, scalar tables, and the sealed balanced order. All six arm orders occur eight times. There are exactly 144 method-batch boundary receipts and 139 checkpoint switches. The single native attempt directory contains exactly 9,216 request sidecars.

Completed/valid executions are 9,216; invalid, unknown, unattempted and excess executions are zero. The wrapper and campaign exited zero. Source/postflight and native weight-equality receipts pass. No new tensor-level weight comparison was performed by this CPU auditor; the existing equality/source receipts were checked.

All 144 timed-batch boundaries, plus before/after endpoint boundaries, report no concurrent training, transfers or foreign GPU processes. This supports absence of detected interference at those checkpoints; it is not continuous external process monitoring.

## Independently recomputed endpoint totals

Counts use repeat 0 after confirming all three repeats agree. Times sum the three-repeat mean within each request; repeats are not additional request samples.

| Quantity | O58 | K+58 | A58 |
|---|---:|---:|---:|
| Verification calls | 44,730 | 43,267 | 45,083 |
| Output tokens | 314,847 | 314,847 | 314,847 |
| Complete-request seconds | 810.069161 | 783.741837 | 816.639762 |
| Native-generation seconds | 757.819230 | 734.923634 | 766.150924 |
| Cache-maintenance seconds | 37.876086 | 35.452837 | 36.026159 |
| Committed matched draft tokens | 270,037 | 271,522 | 269,678 |
| Native accepted-counter tokens | 269,093 | 270,556 | 268,740 |
| Verification-committed tokens | 313,823 | 313,823 | 313,823 |
| Committed matched draft tokens/call | 6.037044 | 6.275499 | 5.981811 |
| Native accepted-counter tokens/call | 6.015940 | 6.253172 | 5.961005 |
| Verification-committed progress/call | 7.015940 | 7.253172 | 6.961005 |
| Total output tokens/call, including initial token | 7.038833 | 7.276839 | 6.983719 |

All displayed aggregate counts and time totals reproduce the original report. Differences below the displayed time precision are floating-point summation order only.

The two accepted-token counters are intentionally distinct. At each native round, the instrumentation records `accepted_draft_tokens = min(raw_matching_proposal_prefix, committed_progress)`, whereas `native_accepted_draft_tokens = committed_progress - 1`. When EOS or the cap ends generation inside the matched proposals, the usual bonus-token slot is removed: the first counter can count the last committed matched proposal, while the runtime counter still subtracts one. Thus `accepted_draft_tokens + 1` is not always the committed progress. These differences occur at 944/966/938 stopping rounds for O/K+/A, respectively, in one trajectory per request. The auditor verified both counters directly in every native sidecar rather than deriving aggregate native counts alone.

Verification-committed progress excludes the first response token generated during prefill. Consequently the 1,024 initial tokens account for `314,847 − 313,823`. See the pinned `online_feedback/sglang_hooks.py:120–157` and `online_feedback/sglang_backend.py:137–146` for exact instrumentation. The original report remains unchanged; `ACCEPTANCE_COUNTERS.csv` supplies all 3,072 request/method rows with both counters.

| Comparison | Call reduction | Request-time reduction | Request wins / regressions / ties | Gross calls saved / added |
|---|---:|---:|---:|---:|
| K+ versus A | +4.028126% | +4.028450% | 687 / 195 / 142 | 2,181 / 365 |
| K+ versus O | +3.270736% | +3.250009% | 633 / 250 / 141 | 1,917 / 454 |
| A versus O | −0.789180% | −0.811116% | 298 / 468 / 258 | 500 / 853 |

The confidence-interval computation is a separate analysis; this audit independently verifies the underlying native counts, pairing, reductions and point estimates. It does not replace that statistical review.

## Audit correction and interpretation limits

The first audit pass completed all raw checks but falsely marked one corroborating-receipt check failed: the auditor assumed `FINAL_AUDIT.json['rows']` was a scalar count, but it is the complete measurement list. This auditor-only schema error was repaired by comparing the full list against `MEASUREMENTS.jsonl`. `AUDITOR_SCHEMA_FAILURE.json` and `audit_endpoint_schema_failure.py` preserve that initial failure; the corrected read-only pass is `AUDIT.json`. No scientific result or original evidence was altered.

ARS fallacy scan, 11/11 considered: Simpson aggregation is limited to the declared single panel, with all request outcomes retained; no aggregate-to-individual claim; selection/Berkson limitations include the reused panel; no collider-adjusted analysis; base-rate diagnostic claims are inapplicable; no extreme-performance request selection or pre/post causal claim; zero endpoint attrition; all three prespecified comparisons and regressions are reported; sealed order/configuration prevent post-outcome choice in this endpoint; own-feedback trajectories prevent interpreting the result as isolated identical-data loss causality; and no reverse-causality claim is made. One training seed and a previously used panel remain the principal generalization limits.

This supports the recorded endpoint comparison for these fixed checkpoints. It does not establish a complete AdaFlash reproduction, the causal source of the difference, training-inclusive speedup, seed robustness, or general method superiority. Source/optimizer/training-accounting review is separate from this endpoint audit.
