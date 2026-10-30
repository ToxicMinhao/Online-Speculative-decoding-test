# KW production state, work and retention review

## Material Passport

ARS-Codex 3.24.0 experiment-agent validation, **ANALYZED**, October 30, 2026. Execution source: `708e974e06f5d9f4b186c32607c16b37cbd4b14f`. Scope: completed O/K/KW events9–12, using retrieved compact records under `raw/`. Reviewer: same-team `kw_campaign`, author of campaign plumbing and part of the qualification repair. This is explicitly an author audit, not an independent implementation review. No remote access, model execution, tensor loading, source modification or new experiment occurred.

**No state, work-accounting, schedule or retention discrepancy was found in the inspected records.** All three arms complete their own cumulative step1536 from step1024, and all specified new work is accounted for. This finding establishes record consistency and the recorded correctness gates; it makes no claim about KW efficacy.

## State continuity and complete boundaries

All twelve arm/event combinations have a passed saved-parent audit, exact live resume with zero mismatches and no RNG reset, complete-event receipt, exact saved endpoint/export audit, and exact live-versus-saved boundary receipt. I checked their paths, hashes and counters against the sealed initial parents and each immediately preceding own-arm endpoint.

O9 starts from original O8 SHA-256 `60fca4a9d32b8fc20ee86f66d45d202c2d6d8570408c6f84b8b0b60f4283735e`; K9 and the one KW migration start from original K8 SHA-256 `1b1a646eb1dac0b61629a0eb3586b424dea847f487a849f63ec800489de6e9ed`. The KW9 live receipt explicitly verifies migration from K8 with exact scientific state and zero initial controller. KW10/11/12 each restore their own preceding KW state and controller exactly; no later migration is reported. The first KW window also records zero controller parameters and zero RMS/credit count, consistent with the migration gate.

Every saved parent/endpoint audit records 58 draft tensors, 58 complete optimizer entries, the expected cumulative Adam step, inherited horizon7424, one CUDA RNG state, and an export exactly matching the restart weights. Every live boundary records zero mismatch, zero additional model/optimizer work and no RNG restoration during verification. All twelve closed trainers are successful with zero accumulation remainder. The inspected source checks residual gradients, accumulation/metric buffers, reference/schedule state and KW controller equality at these boundaries.

| Event | Own-arm step interval | New steps across three arms | New presentations across three arms | KW previews / credits |
|---|---|---:|---:|---:|
| 9 | 1024→1152 | 384 | 768 | 256 / 126 |
| 10 | 1152→1280 | 384 | 768 | 256 / 126 |
| 11 | 1280→1408 | 384 | 768 | 256 / 126 |
| 12 | 1408→1536 | 384 | 768 | 256 / 126 |
| **Total new work** | | **1536** | **3072** | **1024 / 504** |

Each arm finishes with 3072 cumulative presentations: 2048 inherited plus1024 new. The campaign's 3072 new presentations sum the1024 new presentations of each of three arms; they are not the cumulative total across all arms.

## Shared feedback and natural schedules

The four pools contain exactly128 distinct chronological requests apiece, jointly512 distinct requests, exactly matching the sealed windows. Their feedback checkpoint bindings are the preceding O8/O9/O10/O11 exports. Each pool records128 generations and128 native terminal-teacher completion forwards; those required feedback-preparation operations are separate from controller work and do not represent extra KW training passes.

For each arm/event I checked all256 presentation records: each pass is a complete permutation of the same128 feature identities, accumulation steps occur after every second response, reported counters increase correctly, stepping learning rate remains0.0003, and recorded metrics are finite. K and KW match O's feature metadata, response order, pass index, anchor positions and masks for every presentation: **2048 comparison rows** across the two non-O arms. Each arm records **255256 kept anchor occurrences** across its1024 new presentations.

The execution-sealed `training.py` still calls the normal `torch.randperm` sampler. `PresentationSchedule` only asserts the resulting identities and anchor draws; it does not substitute a permutation or anchors. This source behavior, the exact RNG resume receipts and the matching recorded schedules support the intended natural-draw control. I did not rerun a sampler from the remote RNG tensors.

## KW work and pending credit

I independently reconciled all **512 window rows** against **3568 work-log rows** and their corresponding pairs of training records. Every window has two started previews and one completed-pair record, plus one started/completed live step. Each eligible delayed credit has matching started/completed records. Actual ledger totals are exactly1024 previews,512 live steps and504 credits; all attempts have completion evidence and none remains unknown. These totals agree with every event receipt, trainer-close count and final campaign counters.

The first window of each of eight passes has no incoming credit. The other63 windows per pass credit the immediately preceding window in that same pass, with exact producing identities; no credit crosses a pass or event boundary. All eight final-pass-window records explicitly discard pending credit. Controller parameters and RMS state continue across passes/events, while cumulative credit counts end at126/252/378/504. Each event's final controller equals its last logged window.

The successful checkpoint path refuses a pending credit, and the controller state loader rejects a nonempty pending field. Those source checks, all four passed KW saved-boundary validations, eight pass-end discard records and successful own-state resumes support empty pending state at every saved complete boundary. Full controller tensors were not reloaded locally. The twelve retained `PENDING.json` files are original event-intent records; completed RESULT/CLOSED/boundary receipts supersede their operational status, so their presence is not evidence of unresolved credit or an incomplete event.

## Retention and original-state preservation

The final retention receipt contains exactly **18 files** for the six required O/K/KW event10/12 endpoints: six complete restart states, six model exports and six export configurations. Their recorded identities exactly match the corresponding endpoint identity receipts. They total **50360652092 bytes**. Final current-parent bindings point to the three event12 complete states and exports.

Events9 and11 retire nothing immediately. After the complete own event10 and12 successors have passed saved/export/live-boundary checks and required evaluation, each retention intent/progress/final record removes exactly nine predecessor files: three complete predecessor states and their six export files. The **18 retired files**, totaling **50358297980 bytes**, match the previously recorded event9/11 identities and name the correct own-arm successor. Exact serving/resume of those superseded boundaries is explicitly lost, not archived. No original event8 state, required event10/12 endpoint or feedback artifact appears in the retirement list. The normal source performs retirement only after the required evaluation and successor checks.

All four pools retain their prepared-feature identities plus512 captured-response/feature/terminal identities per event. The final retention receipt records all new feedback/features preserved and no assumed offload. Postflight passes all2670 sealed files and eight original artifacts, revalidates original O8/K8 at step1024 with their original hashes, and records exact shared original RNG/history and disjoint new training identities. This is remote runtime evidence carried in receipts; I did not independently hash the full remote tensors again.

## Scope, retries and qualification reuse

The production qualification receipt is byte-identical to the successful `neutral_repair_001` receipt. Production records zero new qualification work and has no new qualification work directory. The successful earlier8/8/14 qualification and original failed6/6/5 qualification remain separate historical work, rather than being added to the production counters. The included event10 native serving gate is within the existing evaluation budget.

The retrieved run contains one `campaign_001`, no failure receipt, no explicit-boundary-resume record, and successful supervisor/campaign/report exit codes. Final counters report no extra optimizer steps, generations or controller model passes. All four KW event contracts match, and seven inspected scientific source files match their execution708 seal. These records contain no evidence of tuning, repeated events or an automatic retry; they cannot prove the absence of hypothetical unrecorded activity outside the sealed run.

## Audit evidence and limits

The local audit rehashed **127 inspected compact files** against `raw/COMPACT_IDENTITIES.json`, independently checked the JSON relationships above, and verified the seven inspected source-file identities against the execution seal. Machine-readable findings and exact inspected identities are in [STATE_WORK_REVIEW.json](STATE_WORK_REVIEW.json); the reproducible read-only analysis is [audit_state_work.py](audit_state_work.py), with [its output](STATE_WORK_AUDIT.log).

This review did not retrieve/recompute full model/Adam/RNG tensors, replay model work, or recheck current remote physical storage. Exact decoded-state and preservation conclusions are therefore attributed to the validated runtime audit receipts. It is an author audit alongside the team's separate result/cost reviews. No statistical hypothesis test, drafting-effect estimate or timing interpretation is made here. The completed run and audit do not advance Stage1/projectNone or authorize another event.
