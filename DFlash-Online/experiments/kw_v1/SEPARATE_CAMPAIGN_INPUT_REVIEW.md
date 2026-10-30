# Separate KW-v1 campaign and sealed-input review

Review date: 2026-10-29 America/Edmonton (2026-10-30 UTC). Reviewer: separate same-team `kw_eval` agent. Scope: local, read-only review of the preparation, campaign, resume, preflight, storage and launcher code authored by another agent; independent reconstruction of the selected inputs, ordering and capacity arithmetic. No remote access, GPU execution, model forwards, serving generations, deletion or commit was performed by this reviewer.

**Conclusion: no remaining source-level blocker in this scope after the narrow completion-audit fix described below.** This is not a completed full-model qualification, actual free-space admission, execution result or efficacy finding. Original source reviewed was commit `54e2598b9e947a1606d8f9fb5c6b268a9a129033`. The final review includes the uncommitted completion-check patch identified by exact hashes below; execution must use the subsequently published and resealed source. No algorithm, hyperparameter, panel, schedule or work budget changed in that patch.

The reviewer authored the evaluation/report implementation earlier. That implementation is therefore **not** claimed as independently reviewed here. The report was changed only after the parent explicitly authorized matching its completion audit to the independently identified campaign gap. Preparation/campaign/resume/storage code was not edited by this reviewer. Research instructions and the ARS 3.24 experiment-agent run/validate workflow were applied; Stage 1 / project None is unchanged.

## Finding and resolution

At the original commit, reconciliation preserved attempted KW work, completed lower bounds and unknown work, but the terminal completion gate checked aggregate budgets without explicitly requiring the unknown map to be zero or each completed count to equal the exact attempt budget. Ordinary execution exceptions already caused failure; no actual misclassified production run was observed. Nevertheless, independently accepting a completed claim with unresolved optimizer/controller work was a concrete audit gap.

The campaign author added `validate_completed_work`, called after final reconciliation and before setting `status='completed'`. Both `kw_work_attempts` and `kw_work_completed_lower_bound` must now equal `{previews: 1024, live_steps: 512, credits: 504}`, with actual integer values; `kw_work_unknown` must have those same three keys and all integer zeros. Missing, malformed, incomplete or unknown work raises an error. The existing exact production totals and final own-arm counters remain required. The report independently rejects the same malformed completed claims. I read the final campaign implementation and regression test; the guard is reached on the terminal success path and does not alter execution settings or introduce an efficacy stop.

The campaign author reports nine focused campaign tests passing after the patch. My earlier independent ten-test campaign/resume run passed before this small patch. My seven focused report tests passed after the report patch, including the complete 4,032-generation synthetic campaign and rejection of missing/mismatched/unknown work. These are overlapping CPU suites, not additive model trials. No further test run was needed after the final read-only guard inspection.

## Input reconstruction

Independent local checks used only source questions and the pinned preprocessing/tokenizer. They did not read test answers to select inputs or inspect new model outcomes. A reconstruction script completed 5,489 assertions covering source cursor, original metadata, selection, overlap, input hashes and expanded ordering.

- The inherited source cursor is reconstructed from both accepted and skipped historical source rows, rather than inferred from the number of accepted requests. It advances from 1,024 to 1,536; these particular next 512 rows require no skips. Selected indices are exactly 1,024–1,535 in original order. All 512 original metadata records are preserved except the intentionally assigned new event identifier; the original event identifier is retained separately.
- Training selection excludes historical evaluation inputs by normalized question, actual source IDs and exact token tuple. NFKC normalization and whitespace normalization agree with the implementation. The historical development request-to-source mapping is used where old request ordinals were not actual dataset row IDs. The old 2026-10-27 confirmation panel is included among the four excluded prior panels, totaling 256 inputs.
- All 1,536 current-study training prompt tuples are distinct. All 320 evaluation prompt tuples, including the old confirmation panel and the new final panel, are distinct and disjoint from those training tuples. Both normalized-question and exact-token overlap checks pass.
- The final 64 are independently reproduced by excluding all 1,536 planned training inputs and the 256 previous evaluation inputs, deduplicating the eligible source questions, and shuffling with isolated `random.Random(20261030)`. Their IDs and order match the sealed selection. The source dataset revision is `740312add88f781978c0658806c59bc2815b9866`; all recorded source and ten preparation-output hashes checked match.
- All 64 final token prompts were reconstructed with the pinned local tokenizer and original preprocessing helper. The original question suffix, disabled thinking, no explicit system message, enabled generation prompt and `add_special_tokens=False` are preserved. The helper SHA-256 is `d19d3c1688768e13783150ce23178fcf2ed4c94d9ad7960e96f50c99d8a9d48f`. All 128 original F/K1 retained request rows and all 64 saved O8 canonical outputs for the third retained panel match their source evidence exactly.
- The final panel is fresh to this current study, not globally unseen: 61 of its 64 inputs have recorded older unrelated-project exposure. This limitation is disclosed by the selection audit and must remain visible in reporting.

The initial seal records `sealed_before_training=true`, UTC `2026-10-30T00:17:46.027124+00:00`, and the original reviewed commit. Its initial PLAN file SHA-256 is `d112de8a2e20481fe9077c364a04772f2e2aa5033b14bfba2822dca092141c75`; SEAL file SHA-256 is `b8ec34639d0b8aabf79ed587ca8625ad59af9d394a4f0ebe5a7bd6dc5a974b3c`; selection-audit file SHA-256 is `aa20b7a123f3c04f11f3c3cb203ef5d320dfb5b00a760baa8bbf72dd3bb9f53e`. Source resealing after the completion fix is required; these identities describe the actual inputs reviewed, not a claim that the old execution seal covers modified source.

## Schedule, continuation and fixed work

The prewritten evaluation schedule has 1,344 ordered request/repeat triples, or 4,032 O/K/KW executions, across the three retained panels at event 10 and all four panels at event 12. For each request, every arm occupies each of the three timing positions once. Each 64-request panel has 96 occurrences of each direction of every pairwise precedence. The schedule semantic SHA-256 bound into PLAN is `d5c8bbace22a68734a908d4455765f335f39ec92fca27194d12fd14a6c210b48`; the actual schedule-file SHA-256 is `6a9b62fc971e5d0e77d04fc8cf5d22c4c25d194eac9260da638bc177335b7ed1`. These intentionally differ because the former hashes canonical JSON. There is no outcome-dependent schedule update.

The first eight event-10 development requests supply the included 72-execution native gate. The retained panels retain their canonical outputs. The final panel obtains its canonical output from the included first O12 execution; earlier K/KW measurements remain pending until that reference exists. There are no extra reference generations or generation warmups.

O continues its own complete O8 state; K continues its own complete K8 state; KW starts from a complete K8 copy with the controller migration. State validation checks actual weights, BF16 Adam moments, step/configuration state, full RNG, history, counters and idle accumulation boundary. The inherited horizon remains 7,424 and the terminal cumulative step is 1,536 per arm. Old-state guards remain unchanged. Each event's preceding O checkpoint collects one shared 128-response pool; all three arms consume that same pool. O's natural training schedule and anchor draws provide the match check for K/KW; the code does not force anchors to conceal divergent RNG. Feature hashes are checked before and after training.

The exact new production budgets remain 512 shared feedback generations, 1,536 committed optimizer steps across the three arms, 3,072 presentations, 1,024 isolated previews, 504 delayed-credit opportunities and 4,032 evaluation generations. There are zero additional controller model passes. The numerical/resource qualification is separate and bounded to eight response forward/backwards, sixteen disposable optimizer applications, zero serving generations and zero persistent optimizer updates. Numerical/memory qualification bindings are checked against the sealed source, K8 identity and qualification inputs; this review does not claim that qualification has run.

## Storage and retention

I independently reproduced the full-capacity phase calculation: **215,574,560,848 new bytes (215.575 decimal GB)**, including the 8 GiB reserve, with the maximum at `event_012_KW_pending`. The calculation uses 1,084,785 full-cap feature rows at 49,164 bytes per row, weight size 2,097,259,104 bytes, two retained copies of new capture/prepared features, all six required event-10/12 full states and serving exports, temporary checkpoint/export/workspace overlap, 10% slack and the compact/workspace allowance. The final incremental footprint is 156,998,957,976 bytes; 38 phases are considered. Retain-local mode assumes no successful offload to create capacity. Original O8/K8 are already occupied storage and are never offered for retirement by this campaign.

Saved event states are immutable links; replacement of the live latest-state path is atomic. Rolling event-9/11 state removal occurs only after verified successors and the corresponding event-10/12 evaluation boundary. Before removing rolling states, the code verifies saved decoded state/export/live-boundary correspondence and hashes, and validates all candidates before unlinking. Required event-10/12 states/exports and all new feedback remain. Original event-8 files are rechecked at postflight. The separate historical-retirement review does not authorize removal of any active KW dependency.

Actual remote free space, newly preserved historical evidence and phase-specific admission remain operational checks for the parent. Local arithmetic passing does not establish that the server currently has 215.575 GB available. No retirement was performed in this review.

## Failure and recovery behavior

An independent CPU failure fixture recorded two completed preview attempts, a completed live optimizer call before the ordinary step counter advanced, and a started but incomplete credit. Reconciliation charged two preview attempts and one credit attempt, accounted for the completed live step despite the runtime counter lag, and retained one unknown credit. It did not invent a successful completion or treat unresolved work as zero. Separate window-row counts remain available alongside work-ledger counts.

Recovery is explicit and restricted to a fully verified idle event-9/10/11 boundary with a completed retention receipt, exact state/export hashes, decoded-state validation and no later feedback/artifacts. There is no automatic scientific rerun or mid-pass recovery. Failure records remain preserved. Valid negative or mixed scientific outcomes do not stop the training/evaluation loops; only correctness/resource/execution checks do.

## Final reviewed source identities

Paths below are relative to the shared `DFlash-Online` checkout. The campaign and report hashes include the completion-audit fix; the final publication commit is assigned by the parent after this review.

| File | SHA-256 |
|---|---|
| `experiments/kw_v1/prepare.py` | `26f613fcf90154efb72861ef75777acb1ed9129cec2b322f5867bd5d714246e1` |
| `experiments/kw_v1/campaign.py` | `22f26282e11a85e92dc0062f751dcde03b56602a8c691b89c6145ba73e6cca55` |
| `experiments/kw_v1/resume.py` | `b6fb1238a0a87b222b6b600c3c8c0a51d30117bac34b9f5a0ff0f189d2a6f3b9` |
| `experiments/kw_v1/storage.py` | `f2b9cb0ef16d5e3e1e469e1a156ec86544ed147086d18fd9f76c48073af0549a` |
| `experiments/kw_v1/preflight.py` | `3ffae61a921af6734ed147c68359865a16d3810ebb888487172fb1e7ce39207e` |
| `experiments/kw_v1/report.py` (self-authored) | `069fe3c49f3c1c38993c133ef81c30c4fe0d146b8a9e1106f8037b4238aec73c` |
| `experiments/kw_v1/test_campaign.py` | `ced3d80be774f82529dd9e1a59e66d036989e3f97f16c1314a6a9f1b4f8f0ac9` |
| `experiments/kw_v1/test_report.py` (self-authored) | `d71ddb80252fb49187e0d25834c1dd9699818d550ec01f09d4b3e24321f1c160` |

CPU commands used `/home/minhao2/.venvs/dflash-specforge/bin/python` with `CUDA_VISIBLE_DEVICES=''` and `PYTHONPATH=.:tests`. The independent campaign/resume command was `python -m unittest experiments.kw_v1.test_campaign experiments.kw_v1.test_resume -v` (10 tests, 5.728 s). The updated report command was `python -m unittest experiments.kw_v1.test_report -v` (7 tests, 25.870 s). Warnings about absent Flash Attention came from import fallback; no model/GPU execution occurred.
