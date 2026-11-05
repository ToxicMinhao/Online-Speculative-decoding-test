# O_HALF training implementation review

## Material Passport

- ARS 3.24.0 experiment-agent, bounded implementation and CPU verification.
- Status: **12 CPU tests pass** (16.775 s); base `649e0b34066e217a3f918d82012d2a631c447ad0`. Parent owns publication and the exact execution seal.
- Scope: new `contract.py`, `training.py`, `training_worker.py` and narrowly named tests only. No shared numerical path or historical validator was edited.
- Review provenance: implementation-author review plus separate read-only source review from `kp_objective`; no independent GPU rerun.

## Scientific boundary

`OHalfTrainer` directly inherits `SpecForgeEventTrainer` with `objective='osd'`, `response_mean`, and ranking disabled. It uses all valid proposal positions and the native ordinary denominator. No K-specific trainer, selection mask, reference model, protection objective, optimizer preview or correction is introduced.

Original O initialization is validated by the original O gate first. The new contract permits only constant LR 3e-4 to 1.5e-4, its native Adam-group/recipe values, and the new metadata/output identity. Original weights, empty moments, parameter ordering, RNG, counters and history remain unchanged. State validation creates a temporary read-only LR-normalized view for the unchanged historical gate; it does not rewrite the actual state or weaken that gate. Both original algorithm and optimizer recipe identities are bound in PLAN.

Every own continuation validates its complete contract, BF16 Adam state/counters, event history and complete RNG. The runtime restores RNG after construction. Initial and endpoint state/export checks preserve RNG and require an empty accumulation boundary and no resident gradients. The last anchor observation is retained metadata, and does not wrongly invalidate a completed boundary.

The original per-seed/per-event O schedule is mandatory. Inherited shuffling and anchor sampling still run naturally; actual draws are assertions against the retained record. Every event completes exactly 256 presentations and 128 native Adam steps, with no added GPU qualification work. Each fresh worker admits one event only.

## Tests and observations

All 12 focused CPU tests pass. They cover LR/config allowlisting, wrong arm/seed/objective/contract rejection, empty and complete Adam states, exact source immutability, parameter ordering, atomic restore, mandatory schedules, fixed budgets, and ordinary KL gradients after the first mismatch.

The integration fixture uses a real five-layer tiny DFlash, native BF16 Adam and the inherited full event loop under eager CPU attention. It constructs original O state and its natural schedule, runs an O_HALF event, restores its endpoint, then compares with unchanged plain ordinary O at the same lower LR. Weights, moments, RNG, counters, metrics and anchors agree exactly. The successful run hides CUDA and verifies it remains uninitialized. Preliminary failures were test-environment/fixture issues (default Python lacked Torch, fixture target-feature width, and local CUDA RNG visibility); no production numerical gate or tolerance was relaxed.

Instrumentation observes existing forwards only: response loss and numerator/denominator before TrainerCore accumulation; actual draft/layer/head/chunk entries including checkpoint reentry; host optimizer-call duration; response observation and nested record-write time. It introduces no extra pass or backward. Native preclip norm is retained from ordinary step metrics. Host timers are not a kernel-time profile; nested timers must not be added twice. The post-load memory scope includes training, persistence and endpoint validation.

## Integration review

Parent campaign/preflight/seal/storage modules use the agreed CLI and artifact layout. They check complete chronological recovery boundaries before remaining-capacity admission. New odd-event tensors may be retired only after a verified successor under the sealed policy; no pre-existing tensor receives deletion credit. Incomplete work blocks automatic retry. The compatibility bridge precedes all new training. A report-side LR gate was corrected to preserve native records: non-step rows have `None`, and the 128 optimizer boundaries have `1.5e-4`.

Separate source review found no remaining blocker. CPU verification does not establish native GPU serving compatibility; that is handled by the fixed 48-generation bridge and included evaluation gates, without adding training qualification work.

## Files and interface

- CLI: `python -m experiments.o_half_matched_lr.training_worker ROOT --seed 17 --event 1`.
- Outputs: `campaign_001/seed_SEED/O_HALF/events/event_NNN/{RESULT.json,training_state.pt,export/}`.
- Audit: `campaign_001/seed_SEED/resume_NNN_O_HALF/`, preserving work, worker, costs, natural schedule validation, original response observations, initial migration/resume, live/serialized endpoints and tensor identities.
- CPU evidence: `TRAINING_CPU_TESTS.log`; machine-readable review and source hashes: `TRAINING_IMPLEMENTATION_REVIEW.json`.

No remote access, scientific model execution or GPU experiment was performed by this implementation subtask. The tiny CPU fixture is validation work, not additional scientific training.
