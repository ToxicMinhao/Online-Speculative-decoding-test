## Material Passport

ARS-Codex 3.24.0, experiment-agent validation. Bounded same-team cross-component review of the local postprocessor against the frozen evaluator, training receipts and independent audit schema. The reviewer implemented the evaluator earlier; this is not blinded external review or a scientific rerun. GPU execution was already running and was not contacted or modified. Stage 1/project None remains unchanged.

The report builder has **no remaining confirmed schema or interpretation blocker** after the fixes below. All **7 focused report tests pass**, and a separate temporary CPU fixture using the actual three-arm evaluator produced all **1,152 stub records** and passed the builder's endpoint/schema checks. These are synthetic fixture results, not the live campaign's scientific measurements.

Execution source remains frozen and clean at `11b7718a04416a89025a6df69c0c33e558e5fe1b`. Only run-local `build_report.py`, `tests_report.py` and this review were changed. No SSH, GPU work, model operations, production edits or Git commit occurred in this review. The parent will version the postprocessing code separately before the final report.

| Reviewed postprocessing file | SHA-256 after review |
|---|---|
| `build_report.py` | `3c8e8462a15b7490bc074732a69b13216816d156977f5116d7c62b5e3d20a454` |
| `tests_report.py` | `f3e6103e4d8c4590ff80484a6cbbde369f28e29d1daa1077978d05adad9c520c` |

Confirmed and corrected:

- A failed window may retain `correction: null`. The previous fallback branch counter called `.get()` on that null, crashing partial-result reporting. Its fallback was also evaluated eagerly when audited branch counts already existed. The builder now uses audited counts when present and records a missing correction as `unavailable` otherwise. Partial Adam/presentation counts remain visible without inventing a saved state, event time or successful correction.
- The previous endpoint predicate needed only campaign `completed`, audit `passed` and 64 valid requests per panel. It could therefore promote missing or contradictory per-repeat artifacts. The auditor reconstructs native repeat totals, but does not compare the exported repeat CSV itself. The builder now requires exact global/per-panel valid/invalid/unknown/unattempted accounting, all four panel report artifacts, nine unique arm/repeat slots, 64 requests per slot, repeat-consistent aggregate calls and repeat-time means matching the summary. Missing or inconsistent evidence blocks the completed-endpoint conclusion.
- Prior nonpositive gaps produced the malformed phrase “the does not show an advantage.” This now reads “the advantage does not emerge.” Positive persistence, shrinkage, reversal and tie behavior are separately tested.
- The report now preserves reference available-position occurrences alongside presentations, anchors and protected occurrences, using the auditor's exact key. Timing prose clarifies that `training_s` sums presentation timers: reference/correction and integrity hooks are included, while response-tensor loading remains outside those timers and inside event wall time. Saved-state validation and artifact hashing follow the event timer and remain separate costs.

Checks that required no arithmetic changes:

- KRC4/O4 is primary; K4/O4 and KRC4/K4 remain separate. All percentage reductions use `(baseline total - candidate total) / baseline total`, never the mean of request ratios. Per-request directions and gross saved/added counts use the evaluator's audited common cohort.
- Development and additional remain separate. Event-1/2/4 curves label historical versus fresh counts. Absent additional O1/O2 values remain null/“absent”; no fabricated persistence claim against O is made there. Relative gaps against O/K and KRC's own event-2-to-4 movement are distinguished.
- Counts use one validated trajectory per request. Timing repeats are not new examples or training runs. Three per-request means aggregate to the average of the three complete repeat totals. Native-generation/cache timers remain nested in complete request time.
- Auditor exposure keys match exactly: `exposure.kept_anchors`, `available_occurrences`, `selection.selected_position_occurrences`, plus reference presentations/anchors/available/protected occurrences and BF16-realized nonzero correction counts. Branch counts are recorded rather than inferred from nominal update frequency.
- `pre_reference_s + post_k_s + correction_s` measures intrinsic reference/correction work. The three integrity timers are separate components of the presentation timers. Their subtraction leaves a labeled K-plus-other-overhead residual, not a fresh pure-K training measurement. Save/export/retention are inside event wall time. Loading and publication remain outside inference latency.
- The conclusion remains conditional on a complete audited endpoint. It states one shared-feedback trajectory and fixed-checkpoint serving evidence, does not isolate only event-3/4 corrections, and makes no training-inclusive deployment, seed-robustness, new-method selection or stage-advance claim.

Validation:

```bash
CUDA_VISIBLE_DEVICES='' /home/minhao2/.venvs/dflash-specforge/bin/python -m unittest discover -s research/dflash_krc_events_3_4_20261028 -p tests_report.py -v
```

Result: **7 tests, 0.030 s, OK**. The integration fixture also passed using the existing `experiments.krc_events_3_4.evaluate.evaluate` function and its CPU backend over two 64-request panels × three arms × three repeats, yielding nine repeat-table rows per panel. All fixture artifacts lived in a temporary directory and were removed afterward. A synthetic audit-success signal was used only to test the report contract; no live audit result was claimed.

The final scientific report still requires the real completed/failed receipts and audit. This review did not examine live performance or infer an outcome while the GPU campaign was running.

Execution source commit: `11b7718a04416a89025a6df69c0c33e558e5fe1b`. No scientific source changes after launch.
