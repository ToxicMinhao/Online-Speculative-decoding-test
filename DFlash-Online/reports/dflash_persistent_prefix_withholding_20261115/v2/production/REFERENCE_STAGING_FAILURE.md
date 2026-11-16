# PR49 production reference-staging failure review

The production run at `db6e1221d8d1b234d9d07e246025d1afd5dc30d5` stopped because its new output directory never received the 104 retained canonical references. This is a confirmed packaging/integration bug. The 24-request identity join passed; it then tried to read a destination file that did not exist. This is not a DFlash, BF16, optimizer or withholding-mechanism failure.

This review used local retained code, the sealed package/archive and `PRODUCTION_LIVE_013.json`. No scientific source was changed, no GPU or remote operation was performed, and no retry was launched. The original package remains unchanged.

## Confirmed failure chain

1. `prepare_production_continuation.py:23–24` changes `plan.references` to `production_continuation_001/references/CANONICALS.json`.
2. That builder never creates the file or its `references` directory. Its `PACKAGE_ASSEMBLY.json` explicitly contains `copied: {}` (`prepare_production_continuation.py:84`). Inspection confirms that the local package and the sealed delta archive both lack `references/CANONICALS.json`.
3. `stage_production_continuation.py:13–19` extracts the archive and copies only the package assembly entries. An empty copy map cannot provide the missing seed file.
4. `references.py:84–90` authenticates the panel/missing/reused inputs and successfully joins all 24 missing identities to complete sealed requests. Line 91 then reads the absent mutable destination. `reference_task` calls this before `create_backend`, so the failed reference task constructed no serving backend and attempted no reference generation.
5. The supervisor correctly stops after the failed task and interrupts the concurrent S8 worker. The final evaluation never starts. The implementation failure does not supply an efficacy result.

The original generic `prepare.py:43–44` does create this seed destination by copying `REUSED_REFERENCES.json`. However, production used the separate private continuation builder, which bypassed that preparation path.

## CPU reproduction and scope

The actual prepared PLAN and its authenticated retained inputs were remapped to their local counterparts. Calling the real `reference_task` reproduced `FileNotFoundError` at exactly the missing `references/CANONICALS.json` destination. Backend-construction count and ledger-operation count were both zero.

In a disposable CPU-only directory, copying the exact authenticated 104-entry `REUSED_REFERENCES.json` bytes to that destination made `load_missing_requests` succeed: 24 complete missing requests plus 104 retained references. This verifies the narrow missing-file cause; it does not retry or qualify the interrupted training trajectory.

Evidence: `v2_reviews/PRODUCTION_REFERENCE_FAILURE_REPRODUCTION.json` records source/archive identities, the exception, zero model work and the temporary-copy check.

## Why the earlier tests missed it

The reference-join tests correctly exercised the identity-only audit schema and the real consumer. But their fixture explicitly created both the immutable reused file and the mutable destination (`test_persistent_prefix_reference_join.py:62–65`). The actual-file mock-consumer check did the same in its temporary directory. Those tests established join/consumer correctness **conditional on a seeded destination**; they did not establish that the actual continuation builder supplied it. My earlier “actual-file integration” description did not make that limitation sufficiently explicit.

The continuation tests used placeholder scientific files and mocked `run_campaign` (`test_persistent_prefix_production_continue.py:31–34,118–125`). Thus they never reached the reference consumer with the real staged payload. The native CPU preflight verifies listed immutable inputs, states and runtime/storage configuration; the new mutable reference destination was not among those immutable inputs, and no early reference-readiness call checked it. Passing that preflight did not establish complete task-input staging.

## Narrow repair and missing regression test

The actual continuation-package builder should create the new mutable canonical file from the **authenticated, unchanged 104-entry reused-reference artifact**, retaining its seed identity in the package receipt. It must not use candidate outputs or silently overwrite a partially populated destination.

A production-entry CPU readiness check should then run the complete reference loader/join on the **fully staged destination** before launching any training or serving worker. This would catch the same omission at entry instead of waiting until the reference task becomes ready.

The regression must build or unpack the actual continuation package into a temporary root, apply its actual path remapping/copy manifest, then invoke the real readiness/consumer path without manually creating the seed in the test. Verify exact seed bytes and 104/24/128 coverage. Removing the seed, changing its contents or changing its destination must fail before any task/backend/ledger operation. A successful fixture may mock the backend, but it must not mock away package assembly or inject the missing prerequisite.

This is the proposed repair, not an applied change or execution authorization. The failed attempt, interrupted S8 state and all consumed/unknown work must remain recorded. Any continuation decision must address recoverability and the unchanged cumulative budgets separately; this review does not authorize replaying work.

## Accounting note

The live snapshot reports zero reference generations, zero final warmups and zero scored generations. It also retains one pending S8 `response_forward`; its stored `unknown` map is zero despite `attempted` exceeding `completed` by one. The final work audit must derive unknown work from the pending operation and attempted/completed/failed counts, not treat that stale map as evidence of zero unknown work. These are live-snapshot observations pending the compact delivery audit.
