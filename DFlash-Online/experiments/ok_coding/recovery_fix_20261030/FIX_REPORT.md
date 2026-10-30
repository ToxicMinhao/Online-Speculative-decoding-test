## Material Passport

- Origin: ARS-Codex3.24.0 experiment-agent implementation and validation, October30,2026.
- Reviewed baseline: `5ec126a189170235b8e89e390261904ddc691dcc`.
- Corrected implementation: `a0ba4c23e28bef6a16d79866b251fb1b6c480358`.
- Scope: recovery infrastructure and CPU tests only. No A800 operation or scientific GPU execution.

Both recovery fixes and both review clarifications are implemented. The O/K algorithms, prepared inputs, seeds17/29, four-event schedule, optimizer/RNG contract, evaluation endpoints, retention obligations and work budget are unchanged. The original preparation and failed admission reports remain intact.

### What changed

The old recovery path reused `campaign_001/native`, which the unchanged backend correctly refused to create again. Each admitted invocation now exclusively reserves the next greatest numbered `native_attempts/attempt_NNNNNN` parent and passes its absent `native` child to the backend. Attempt receipts retain source, mode, admission references, startup failure and closure. Legacy paths and absolute capture references are preserved. Numbering consumes no scientific RNG and never reuses gaps; unexpected collisions fail rather than retry.

Source/configuration and exact completed-work validation precede resume capacity calculation. Model/runtime/resource checks finish before native allocation. Completed feedback, preparation, training and evaluation are reused; only the next unstarted action executes from the correct own-arm state and preceding O feedback checkpoint. Invalid or partial work cannot be resumed. Both preflight and boundary-admission receipt namespaces use exclusive maximum-number allocation, preserving earlier failures.

Retirement intent/progress without a completed receipt blocks recovery even before any unlink. Missing/changed initialization, prepared/native feedback, endpoint or latest-state artifacts also block. Unrecorded export files are rejected, and completed work after a chronological gap is not admitted.

### Remaining storage, without weakening fresh admission

Fresh admission is unchanged: **366,710,105,681 bytes** required. The historical available capacity **150,373,031,936 bytes** still fails, leaving **216,337,073,745 bytes** short. This task did not remeasure or provision A800 capacity.

Resume now models bounded future live writes against current free space. Completed retained files already occupy disk and are not charged again. It preserves cap2048, both future feedback copies, missing initialization states, Adam state/export and atomic-save overlap, future rolling predecessors, retained endpoints, full16GiB compiler allowance,10% allocation slack,4GiB compact/workspace allowance and8GiB reserve.

The physical inventory counts allocated blocks once per(device,inode), separately from logical size, and records aliases including latest. **All pre-existing allocations receive zero retirement credit**: link/extent information cannot rule out open handles, mappings or snapshots delaying release. They cancel against the current occupied baseline. New future rolling payload still retires only at the established verified-successor/evaluation phase. Admission deletes nothing and assumes no offload or unrelated cleanup. This conservative policy can still reject some physically feasible continuations.

A deterministic hypothetical completed-seed17 boundary requires **209,627,864,400 bytes** remaining under the unchanged full-cap model; the old full-bound-again gate rejects that capacity. This is fixture arithmetic, not an actual recovered campaign or fresh storage measurement. Threshold and one-byte-below checks pass for both modes. The pure accounting test verifies slack before hypothetical release:110−90=20, not11; it does not authorize a positive production credit.

### CPU evidence

**All 58 CPU tests passed in 123.712 seconds on commit `a0ba4c23e28bef6a16d79866b251fb1b6c480358`**, with CUDA hidden. The final complete-suite result is in `CPU_TESTS.log`; 14 final storage tests also passed in 0.056 seconds (`CPU_FINAL_STORAGE.log`). Repeated checks are not additional distinct tests. The successful recovery fixture exercises real source-seal, boundary, capacity and skip decisions with a strict backend fake that creates the actual absent directory. It preserves legacy traces/captures and gapped receipts, reuses completed events and dispatches the next sealed work from the correct O/K parents. Machine/model/subprocess work is simulated; directory behavior and reuse decisions are not.

Coverage includes partial/unknown work, changed/missing identities, unrecorded exports, unfinished retirement, source changes, storage denial, startup failure, exclusive collisions, RNG preservation, sparse files, hard links, other filesystems, caps, compiler allowance, overlap and reserves. Existing tests include tiny CPU model/optimizer fixtures; these are distinct from experimental model training. Shell syntax and diff-whitespace checks pass.

A separate same-team reviewer independently passed four recovery tests and14 storage tests, reviewed the final implementation and bound nine files by hash. No unresolved actionable finding remains. See `SEPARATE_REVIEW.md/json`. These are CPU correctness checks, not native GPU qualification or performance evidence.

### Completion boundary

- Recovery fixes: **passed CPU validation**.
- Fresh storage admission: **still blocked on the published evidence**.
- Scientific GPU work: **zero**; no feedback/evaluation generation, terminal-teacher preparation or experimental Adam updates.
- Data binding preserved: `6def4832d12698fc2733e91c427db77558f3fe50e7f51063e9c2a11f9bc961af`.
- Work unchanged:1,024 feedback generations,1,024 terminal forwards,2,048 Adam steps,4,096 presentations and4,608 evaluation generations. All remain unattempted.
- A future launch must seal the corrected published source and satisfy actual storage/native gates. The old source seal does not cover this change. No GPU launch or campaign restart is part of this task.
