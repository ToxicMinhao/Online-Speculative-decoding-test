# Work, provenance and state audit

## Material passport

**Verification status: VERIFIED against retained evidence.** This CPU audit reconstructed the sealed work and state accounting; it did not rerun a model. It does not judge the statistical association or authorize another experiment.

Execution: `b6f2a2c8d487bcc2d32a5e716b870c10eb5067b3`. Fixed transition: native C + K+PAIR at event 1 (128 steps / 256 presentations) to its retained PR60 CONTROL event-2 endpoint (256 / 512). Source base is PR63 publication `281516426411ac567129dae7af15ea443a3d0c94`.

**Authorship disclosure:** the auditing agent authored the read-only native adapter and its CPU tests. It separately reviewed the other agents' preparation, worker and supervisor implementation, and independently reconstructed the retained counts, hashes, schedule and parity comparisons without importing those helpers. This is not an independent GPU replication.

## Result

All **21,973 checks pass**, with no unresolved audit error. All 5,410 source entries and 300 retrieved files (112,133,005 bytes) authenticate. The source manifest is complete against the committed Git tree, records a clean checkout and matches the remote postflight identity. The original PR60 archive binding, training-event receipt, feature identities and both fixed endpoint identities agree.

| Physical work | Qualification | Primary | Total |
|---|---:|---:|---:|
| Response reads attempted/completed | 16 / 16 | 256 / 256 | 272 / 272 |
| Draft forwards attempted/completed | 16 / 16 | 256 / 256 | 272 / 272 |
| Vocabulary-head calls attempted/completed | 72 / 72 | 2,048 / 2,048 | 2,120 / 2,120 |
| Block readouts | 576 | 16,384 | 16,960 |
| Backward / optimizer / teacher / generation | 0 | 0 | 0 |

Invalid, unknown and unattempted scheduled model work: **0**. GPU retries and additional startup model forwards/backwards: **0**. There were eight endpoint loads: parent and child for each of two cards in each of the qualification and primary phases. The primary scientific cohort is 128 responses and 8,192 paired blocks; qualification repetitions are not extra scientific samples.

The audit independently recomputed every response's hash-ranked anchors from the authenticated first-pass training receipt. All 128 responses have 64 selected anchors and remain in source order. Both endpoints for a response ran on the same physical card, with ordinal-parity sharding. There were no replacement examples, skipped responses or post-outcome anchor choices.

Exact legacy labels, validity, proposals, roles and retained margins reproduce on both endpoints/cards. Repeated primary-geometry readouts and cross-card comparisons are exact. The first primary readout after reloading also reproduces its qualified readout. The old eight-anchor geometry and new 64-anchor geometry remain separately labelled.

## State and scope

Every readout records BF16 features/head execution, block size 16, eight-block head chunks and `eval/no_grad`. Expected head shapes and complete block coverage reconcile independently. Every readout retains equal before/after RNG identities, unchanged parameters and gradients, and restored runtime state. All eight endpoint closures report unchanged parameters/modes and no optimizer. The exact 58-tensor export mapping and postflight export file identities match the prescribed parent/child weights.

These checks rely on authenticated exact loads, runtime tensor version/gradient checks, source review and postflight export hashes; they are not full model-sized GPU tensor snapshots or hardware instruction tracing. The explicit seed 17 used while constructing a read-only model preserves the caller's RNG; it is not a claim to reconstruct historical training RNG. Recovery artifacts authenticate lineage and were not loaded into optimizers.

The whole parent feature-manifest file receives a fresh hash. Its eight used legacy feature identities are independently bound by the historically authenticated witness manifest; the fresh whole-manifest hash is not represented as a historical hash.

## Costs and resources

- Occupied GPU-card time: **367.237169 seconds**, below 3,600 seconds, summed from four complete worker-process lifetimes on both physical A800s.
- Original GPU elapsed clock: **189.099294 seconds**, below 5,400 seconds. Qualification precedes primary execution.
- Qualification/primary wall times: 81.368529 / 103.099779 seconds.
- Summed endpoint loading times: 274.256348 seconds across both cards; summed readout times: 72.940314 seconds.
- Peak allocated/reserved CUDA memory: 9,382,659,072 / 9,623,830,528 bytes (9.383 / 9.624 GB).
- Postflight remote free space: **405,509,931,008 bytes**, preserving the 400 GB reserve; both GPUs are idle.
- Retained remote study plus new cache allocation: 476,626,944 bytes, including 17,264,640 bytes of compiler-cache growth.
- Remote charge plus the sealed 750 MB local envelope is 1,226,626,944 bytes, below 2 GB. The current local study occupies 586,821,632 allocated bytes. Final publication and shared Git-object growth still require the root's final all-location accounting.

Loading and readout timers are nested in worker occupation. Compilation timers are passive, nested within those timers and often nested within one another; **do not add their categories together or to occupied time**. Names containing AOT/autograd in compiler telemetry are compiler phases, not evidence of an executed backward. These are diagnostic costs, not serving latency measurements.

## Audit history and limits

The first CPU audit attempt stopped because it treated a source-manifest symlink as a regular file. The audit checker was corrected to use the manifest's `kind`/`target` fields. The failed log is preserved privately; no scientific code, inputs, settings or GPU execution changed. The final check then passed.

The primary responses trained the child. The legacy new-buffer examples overlap this cohort; the parent-training set is a small TRAIN retention description. This audit establishes accounting and fidelity of the retained forward-only diagnostic. It does not establish held-out transfer, causal allocation benefit, serving improvement or training robustness.

Evidence: `AUDIT_WORK.json`, private `AUDIT_WORK_PRIVATE.json`, parameterized CPU checker `audit_work.py`, and retained `AUDIT_WORK_ATTEMPT1.log`. Run the checker with `python audit_work.py --study <private study root>`; it uses the study's authenticated raw receipts and adjacent historical PR60 evidence. It performs no model work.
