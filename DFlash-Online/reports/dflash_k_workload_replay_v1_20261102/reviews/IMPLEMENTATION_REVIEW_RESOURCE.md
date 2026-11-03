# Track2 implementation review: inputs, selection, native training and resources

## Material Passport

- Mode: ARS experiment-agent, run-mode implementation/preflight review.
- Reviewer: separate same-team agent; this agent also performed the resource preflight. This is a source/CPU review, not an independent GPU reproduction.
- Authority: `HANDOFF.txt`, publication `4d94f30c42701d0f86ac04e2829c834593c16d8e`, and the retained source/pin receipts.
- Scope: `contract.py`, `data.py`, `storage.py`, `selection.py`, `training.py`, plus the newly available `worker.py`.
- Status: all three actionable findings and the callback-receipt correction are resolved by the final source recheck below. No remaining blocking finding in the reviewed scope. Live execution is not covered.
- New scientific executions, GPU/model imports by this review, remote writes and deletions: zero.

## Source identities reviewed

| File | SHA-256 |
|---|---|
| contract.py | `1244dd16004e7e04de32be925aa77b211b3c48848ff74d50a90a3f0618002fc6` |
| data.py | `c76d8ab235d1384f60ba8d9f48e6d3680c2ac42e6ffabe73077e6e586ad7b014` |
| storage.py | `e4e08b001ea97f54f7848015c05c8074d3079972a80f7092ece88e0f7eac328c` |
| selection.py | `cda37680e8b649089a61038767cbfbb316f79377f58d5770e49b14d1362c001e` |
| training.py | `5f18d1cfec46548173f0364e6103b548d8a4d8af205b146d981fa598803a0bd0` |
| worker.py | `3db25a491fdb8d56051b934aac8c90b3d238f5c955dc0b57019580d60c86d17f` |

## Checks and evidence

The actual retained manifests and canonical inventories were passed through the new `data.build_inputs`, using the exact pinned tokenizer file and decoding with special tokens preserved. The call succeeds with 1,024 training and 256 evaluation requests. It preserves original request IDs, source identifiers and original request indices while adding explicit experiment IDs and global ordinals. The eight first IDs are `train_00000`, `train_00128`, `coding_training_00000`, `coding_training_00128`, `train_00256`, `train_00384`, `coding_training_00256`, and `coding_training_00384`. All 1,280 token arrays, normalized rendered prompts and normalized user prompts are distinct. The request selection never reads canonical outcomes to choose samples.

`selection.py` admits only compact identity/chronology/descriptor records. It has no domain, source-answer, draft-loss or response-outcome input. It reads the last prompt row, normalizes all supplied dimensions in float64, uses the specified nonnegative clipped cosine and fixed 128-query coverage objective, resolves exact coverage-score ties by ascending global ordinal, and continues selecting after zero gain. Its three independent PCG64 seed constructions match the handoff. U/R share exactly the same 96 fresh records; both second-pass lists are sorted by original ordinal before training. Algorithm R advances once per unique fresh record in chronological order and requires all three verified arm names before insertion. Saved state contains slot order, seen identities, counter and complete PCG64 state.

The training adapter leaves the inherited no-replay implementation unchanged. It accepts exactly two lists of 128 distinct bindings, requires 96 current plus 32 strictly past packets for U/R after event 1, rejects renamed or changed historical bindings, hashes immutable packets at admission, and checks file identity on subsequent loads. It invokes the unchanged native one-pass loop twice. Each invocation performs its own `torch.randperm(128)` and fresh native anchor sampling; the native train step, BF16 Adam recipe, response-mean accumulation, K objective, 7,424 horizon and native max-events metadata remain inherited. Receipt validators require 256 presentations and 128 updates per event. The decoded-state gate checks weights, Adam, RNG, counters and sampled anchors for neutral equality; it does not overwrite RNG to manufacture equality. Later-arm restoration uses each arm's complete state.

The new `storage.capacity_plan` evaluated on the actual selected inputs exactly reproduces the independently computed requirement of **75,023,917,685 bytes**. Its cap-length event-2 capture peak is **54,066,958,740 tensor bytes**, including one incremental raw/terminal packet alongside the complete current/preexisting reservoir pool. The 5% tensor margin and separate 4 GiB additional compiler growth, 3 GiB compact, 2 GiB workspace and 8 GiB reserve are retained. An export phase can hold 14 weight equivalents including workspace after reservoir eviction; for these pinned dimensions its smaller one-pool footprint remains below the declared capture peak. Inode accounting deduplicates current-state hardlinks.

The shared-initial implementation independently validates U/R initialization against K, retains one K tensor payload plus per-arm restore metadata and its hash, and restores the arm-specific metadata explicitly. Deferred export avoids overlap between pending complete states and export construction. The newly read worker validates each saved state, then makes an event-specific hardlink to the committed `trainer.latest` inode and records its identity. Consequently, a later atomic replacement of `trainer.latest` can preserve the preceding state inode until the campaign has validated and retired that predecessor.

All six reviewed files parse successfully. This review exercised actual data preparation and capacity arithmetic, and read the supplied CPU fixture coverage; it did not repeat the native-training fixture suite or infer a production GPU pass from those fixtures.

## Campaign integration requirements still to verify

These are explicit dependencies of the admitted plan, not observed campaign defects:

1. Preserve the prior event's verified state hardlink before starting any next-event worker. The inherited trainer atomically replaces `trainer.latest` before the campaign's successor check; a single unlinked latest file would not preserve rollback evidence.
2. Retire all preceding serving exports only after all three pre-update serving runs and their validity/parity checks finish, before any new training state is saved.
3. Verify each successor state and its event hardlink before retiring its preceding state. Retire only this run's files. `retire_owned` checks retained evidence **existence**, not its scientific validity or digest; callers must perform those stronger checks first.
4. Insert the fresh event into the common reservoir only after all three commits verify. Persist the resulting membership/RNG state before evicting any old/current prepared packet. Do not retire packets needed by a remaining arm.
5. Incrementally validate prepared packets and compact evidence before deleting raw/terminal tensors; otherwise the cap plan must include complete raw/prepared coexistence.
6. Materialize the next exports after all arms commit and reservoir eviction completes. Keep final complete states, exports, reservoir and initial metadata/payload.
7. Enforce compiler-growth/compact budgets and reserve-preserving write admission throughout the actual process, including checkpoint/export writes and failure paths. Refresh admission immediately before launch; the prior free-space observation is not a reservation.
8. Verify/seal the canonical source checks and retained input identities before model work. The preflight explicitly records that math canonical sources have older byte-hash coverage, while coding canonical/source bytes were freshly hashed and reconciled to the old hash-verified scalar report; do not upgrade that latter provenance claim.

The inherited 8 GiB compiler allowance still fails current capacity. The 4 GiB additional-growth choice is explicit, evidence-backed and monitored; this review does not permit lowering it further, reducing scientific scope or deleting history if actual growth exceeds the plan.

## Campaign review addendum

The subsequently available `campaign.py` and `worker.py` implement the intended 13-weight-equivalent save/capture lifecycle. They finish all preceding-checkpoint serving, retire preceding exports, train one arm at a time, verify its complete boundary and hardlink, and retire that arm's old state before training the next arm. Memory insertion and durable membership receipts follow all three verified commits; packet eviction precedes export construction. This prevents a 17-weight-equivalent overlap. The three current `trainer.latest` paths and event snapshots name the same respective inodes, rather than six independent complete states. Final exports/states and final reservoir remain retained. Native generation budgets and four parity-control locations match the handoff.

Two findings were sent directly to the implementing parent:

1. **P1 — enforce the bounded compiler/compact allowances during serving and final evaluation.** In the initial campaign source, `check_growth` is called only at event entry, after each training arm and before the entire final evaluation. Each serving request calls `require_write_space`, which checks free space/reserve but does not enforce the 4 GiB compiler-growth or 3 GiB compact allowance. All 2,304 final evaluations and report construction can then finish without another allowance check. The new capacity admission specifically depends on these monitored bounds. Add allowance checks during serving packets and at final evaluation block boundaries, plus after exports/report construction, so a crossed bound stops additional scientific work and remains visible. Original affected locations: `campaign.py` lines 310, 335–349 and 373 onward, together with the serving loop.
2. **P2 — validate the decoded packet's target-label binding before retiring raw features.** `prepare_packet` initially calls only `validate_feature` on the persisted tensor. That function checks shape, finite values, response mask and token bounds; it does not validate the serialized `label_binding`. The unchanged `validate_target_binding` currently runs only later inside training, after the native/terminal source tensors have already been deleted. Call it during packet validation before retirement, alongside the existing exact prompt/response checks, so the declared validated successor includes its target-label provenance. Original affected location: `campaign.py` lines 127–153. The backend's construction checks are useful but do not replace validating the decoded durable successor at the retirement boundary.

Neither finding authorizes another GPU probe, scope change or rerun. Both are operational/validation corrections before the first scientific launch.

### Correction recheck and further operational finding

The implementing parent added `validate_target_binding(packet)` immediately after decoded packet geometry validation and before native/terminal retirement; finding 2 is resolved by source inspection. Per-request pre-update resource checks, final 64-request batch boundary checks, an after-final-batch check, after-export checks and a shell postflight check after report construction resolve finding 1's missing enforcement. The initial corrected callback executed the checks but returned `None`; the parent was asked to return its resource/timing receipt so the final-evaluation boundary ledger directly contains its evidence.

**P1 — preserve qualified SIGTERM cleanup for the operational timeout.** The new shell applies `timeout --signal=TERM --kill-after=120s`, but the initially reviewed campaign main did not install the inherited full-K SIGTERM-to-SystemExit handler. Both training workers and the serving backend start independent sessions. A default SIGTERM therefore terminates only the parent without executing its `finally` cleanup and can leave scientific child processes running beyond the timeout. The parent was asked to reuse the qualified handler from `experiments/full_k_gsm8k/campaign.py` before calling `run`, which lets the worker exception handler and backend cleanup terminate their owned process groups. This is a concrete timeout-path defect; no live signal/GPU test was run by this reviewer.

The worker's fresh-initial/resume routing was also rechecked: initialization supplies no prior state and independently validates K/U/R empty-state equality; event 1 restores the new K initial or its per-arm verified shared reference; later events restore the same arm's exact preceding boundary. The saved arm/binding/config/Adam/RNG contract is checked before loading. The two one-pass calls execute the same `randperm(128)` and native anchor RNG operations as the original two-pass loop; additional work between those calls is host bookkeeping and timings, with no RNG draw. The observation pass index derives from the cumulative 256-response order and is checked against the relabeled native rows.

### Final disposition

All three findings are resolved by source inspection. The final campaign installs the SIGTERM handler before `run`; it raises `SystemExit`, allowing the existing worker and backend cleanup paths to run. The final evaluation callback now returns both resource and timing-barrier receipts. The worker's finalizer records cleanup failures without silently claiming success. No live process termination, native scientific run or GPU qualification was performed by this review.

Final integration source identities (superseding earlier hashes for changed files):

| File | SHA-256 |
|---|---|
| campaign.py | `3d7f568b05cc2c8f2dd31cd285183fb09df6711b83dd0e076b8aa4a50a593bf9` |
| worker.py | `d391f4ce32c2005a107da3439ff5703ef37f940ee22cb57d9fc2cd39adb230f6` |
| evaluation.py | `86c55ebf2eb4ce617cf0c46b924aed98a598eb5be1e615f855c79f003f565122` |
| preflight.py | `a3c538177418a974f9133dbed1e925b46521cbdb016bbf7ad6bf3153a4df818c` |
| run_campaign.sh | `be4f564c5e6aa8140984afc8ac9be11a6f81dbee17d3d7c51d40e3ae6e98871c` |

The preflight/shell identities cover their inspected resource postflight and signal-launch integration; they do not claim an independent full source/model/environment seal audit. The required native event-1 equality and event-2 mixed-pass gates remain production obligations. Capacity is still conditional on current free space, the stated lifecycle, and bounded growth; passing this review does not reserve server resources or advance the research stage.
