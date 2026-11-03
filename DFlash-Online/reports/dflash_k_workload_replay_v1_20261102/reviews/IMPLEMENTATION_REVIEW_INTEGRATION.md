## Material Passport

- Task: Track2 K/U/R workload-conditioned replay, November 2, 2026.
- Mode: ARS experiment implementation integration review; CPU evidence only.
- Reviewer: same-team trainer/seal implementation agent. This reviewer authored `training.py` and `seal.py`; this is **not an independent review of those implementations**. The scope is the parent-authored campaign/worker/preflight use of their interfaces.
- Status: **PASS for the inspected integration, with one resolved operational finding.** No remaining blocking integration finding was identified.
- Execution: no remote operation, production generation, GPU work, scientific experiment, or production-code edit by this review.

## Integration findings

1. **Output and restore paths agree.** Workers use `campaign_001/arms/{arm}/trainer`; native event receipts and snapshots live under `trainer/events/event_NNN`; `training_state.pt` remains the current committed state. Initial K saves one full tensor snapshot. U/R independently qualify their initializations and retain explicit shared-initial JSON references. Event 1 restores the selected initial receipt; later events restore that arm's immediately preceding boundary.
2. **Scientific state is checked before publication or retirement.** The worker verifies the complete saved state against the live runtime, then hardlinks the committed inode into the event directory and records its identity. A future native atomic replacement of `trainer.latest` cannot overwrite the predecessor inode. The campaign verifies the old identity and retires it only after a verified successor and completion of preceding-checkpoint serving. Initial and final states remain retained.
3. **Explicit replay binding crosses the interface correctly.** The campaign joins selected original record IDs to immutable `{record_id, global_ordinal, path, sha256}` packet bindings and writes two 128-item lists per arm. The worker passes these lists via `pass_plan`. Original IDs and complete prompt/response packets remain intact; only the new trainer admits the declared old records. Domain annotations are added after selection. All three arms commit before reservoir insertion and eviction.
4. **Native work accounting composes with the trainer observations.** WorkLedger wraps the actual response forward, backend backward and underlying Adam step without executing previews. Each successful worker event requires exactly 256 forwards, 256 backwards and 128 Adam updates. Constructors and initialization operations do not call `train_event`. Fresh worker processes restore each arm's complete RNG before training; selection and serving occur in the parent. The native 7,424-step horizon and max-events=58 recipe are preserved while the campaign stops after eight events.
5. **Deferred export lifecycle matches admission.** All old exports finish serving and are retired before training. The trainer commits with an explicit deferred-export status. After every arm commits and reservoir eviction completes, the campaign calls the module-level native exporter and verifies decoded exported weights. `EXPORT_CHECK.json` and arm boundary receipts identify materialized exports. Trainer `RESULT.json` records the earlier training-commit/deferred state; the campaign's boundary/export receipts are the subsequent publication authority.
6. **Sealed plan and preflight agree.** Configuration paths, original configuration, request/canonical paths, model/runtime pins, final evaluation schedule and storage keys match the seal API. Preflight compares every recomputed capacity key while preserving the seal's additional provenance annotations. It verifies source and input identities before model work. The full capacity calculation retains the 8 GiB reserve and explicit post-eviction export workspace.
7. **Resolved operational finding: outer timeout cleanup.** The initial inspected campaign had no SIGTERM handler, while trainer and SGLang children start separate process sessions. A default SIGTERM from the wrapper's 24-hour timeout could bypass Python finally blocks and leave owned work active. The parent added the final `__main__` SIGTERM handler that raises `SystemExit(128 + signum)`. Reinspection confirmed it enters existing worker process-group termination and campaign/backend cleanup paths. The reviewed hash below includes this fix.

## Executed integration check

A temporary CPU-only harness used the existing five-layer tiny native DFlash fixture and the real `worker.run`, WorkLedger, trainer, snapshot validation, neutral comparison and module export. Only device-specific GPU profiling was replaced with a CPU fixture receipt; expected CUDA RNG stream count was set to zero. Released tiny fixture weights were materialized without a forward/backward/update. No production data or settings were changed.

- Three independent worker initializations completed, including U/R shared-initial references.
- K, U and R each completed event 1 with exactly 256 forward/backward presentations and 128 Adam steps. The real campaign neutral-state gate passed for decoded weights, optimizer, RNG, counters and anchor records.
- U completed event 2 with its own restored event-1 state and the explicit 96-fresh/32-old second pass.
- The event-1 hardlink retained its original inode after event 2; the event-2 snapshot and `trainer.latest` shared the new inode.
- The module-level deferred export reproduced the committed event-2 weights exactly at global step 256.
- The harness exited 0. All checks above passed; CUDA was hidden and no GPU work occurred.

This checks integration on tiny CPU tensors. It does not substitute for the included production event-1 neutral gate, capture parity, actual runtime/model pin checks, actual resource admission or final output validity checks. It is neither an independent trainer-method review nor a GPU reproduction.

## Reviewed source identity

Review time: 2026-11-02T20:56:21.615148+00:00. Git HEAD at review: `4d94f30c42701d0f86ac04e2829c834593c16d8e`. The working implementation was inspected by exact file bytes; HEAD alone is not claimed to describe uncommitted additions.

| File | SHA-256 |
|---|---|
| `experiments/k_workload_replay_v1/campaign.py` | `5ccc5c1f021a9e017de4b35f07e25d82b48e37ae4deb9791bfd352f94f927fbd` |
| `experiments/k_workload_replay_v1/worker.py` | `d391f4ce32c2005a107da3439ff5703ef37f940ee22cb57d9fc2cd39adb230f6` |
| `experiments/k_workload_replay_v1/preflight.py` | `a3c538177418a974f9133dbed1e925b46521cbdb016bbf7ad6bf3153a4df818c` |
| `experiments/k_workload_replay_v1/run_campaign.sh` | `be4f564c5e6aa8140984afc8ac9be11a6f81dbee17d3d7c51d40e3ae6e98871c` |
| `experiments/k_workload_replay_v1/training.py` | `5f18d1cfec46548173f0364e6103b548d8a4d8af205b146d981fa598803a0bd0` |
| `experiments/k_workload_replay_v1/seal.py` | `e18a737656cdec83b326b3d290d825832ea4870bd6bc58a44673b4a48efa4ba3` |
| `experiments/k_workload_replay_v1/storage.py` | `73d0d12ec2c45b46c8a0bac2007f4928519800abb3b06f08ec5142fa55128555` |
