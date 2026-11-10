# K+ verification-transition replay: implementation review

## Material Passport

ARS experiment-agent; implementation and CPU validation only. **Native qualification and scientific replay have not run.** Implementation commit: `68cbbdf385162607b18ff467d08a653586a7c9e8`. The adopted [handoff](HANDOFF.txt) and unchanged [configuration/selected IDs](SELECTED_IDS.json) define the scope. This is a new frozen-checkpoint probe; the completed FSG pilot remains closed.

## What this tests

A better proposal block can advance farther yet leave a more costly continuation. For each dataset, compare ordinary KPLUS4 and KPLUS6 on 64 deterministically selected original development requests. At the first actual committed-progress divergence, retain the two native successors. Evaluate each frozen draft at both successors. The two native diagonal suffixes are authenticated; only the crossed continuations require additional suffix execution.

`R4=C46−C44` and `R6=C66−C64` compare successors with the draft fixed. Positive effects indicate an expensive farther restart only when KPLUS6 actually advances and the repair check supports that description. The identities `D=C66−C44=R4+M6=M4+R6` must reproduce the original endpoint call difference. No-divergence effects are zero with NOT_APPLICABLE cells; missing cells remain missing.

## Implemented path

The additive [experiment package](../../experiments/verification_transition/README.md) separates the fixed contract, private input sealing, native successor reconstruction, campaign orchestration, shared work ledger and analysis. Existing training and ordinary serving sources are unchanged.

The native constructor retains the source target KV, seed/positions, request allocator/mapping, original stop budget and committed feature-ingestion segments. It rebuilds the destination's draft KV through the pinned native ingestion method in the original segment shapes. It does not transplant another checkpoint's draft KV or treat a token offset as sufficient state.

The first selected request per dataset supplies both checkpoints' neutral round trips at round 1 and ceil(R/2), deduplicated and nonterminal only. The exact checks cover every proposal, positions, committed progress, final tokens/termination, suffix call counts and inspectable valid cache bytes. No eligible boundary or any mismatch stops the diagnostic; no replacement request or approximate reconstruction is allowed.

A shared process-safe work ledger admits model work before dispatch, retains failed/unknown attempts and enforces the unchanged limits: 256 original completions, 256 crossed suffixes, eight neutral suffixes, 32,768 draft-block/standalone-verifier evaluations and four cumulative occupied GPU-hours. Loading, startup and context rebuilding count. The supervisor reserves 30 seconds for cleanup and terminates only its registered process groups at the cap. Zero training, backward passes, feedback collection or timing sweeps are permitted.

Analysis retains all 64 rows per dataset, boundaries, repair flags, negative effects, missing work, conditional/unconditional effects, concentration and leave-one-request-out signs. It uses 10,000 full-row paired bootstrap resamples, PCG64 seed 20261110. Incomplete execution suppresses a selective-subset go/no-go conclusion. Replay time is work accounting, not a serving-speedup measurement.

## Evidence and review

- **57 new CPU tests passed** ([log](CPU_TESTS.log)); suite time 2.202 s, command wall time 3.408 s. These cover selection, missing/terminal accounting, four-cell identities, bootstrap denominators, process cleanup, sealed inputs, native-stub reconstruction and exact replay gates.
- **69 existing native-serving CPU tests passed** ([log](EXISTING_NATIVE_CPU_TESTS.log)), suite time 2.170 s. No GPU was involved.
- The fixed ID seal has SHA-256 `14a1149c2e15c6bab63e2447e9bc3a78f645dff27387068d5eff194a43649d54`. A separate implementation recomputed the selection exactly.
- CPU schema checks passed on all 128 selected requests and their 256 bound original native records. No crossed continuation or diagnostic outcome was generated.
- Four checkpoint receipts, both original panels, canonical records and seven pinned runtime-file identities were resolved from retained evidence. [Artifact identities](ARTIFACT_IDENTITIES.json) preserve the exact paths and hashes. **Live A800 availability and weight verification remain admission checks.**
- Three same-team agent reviews covered admission/accounting, analysis and native state construction. Reviewers did not independently rerun models; authorship overlap and CPU fixtures are not native qualification.

Review found and fixed detached-process cleanup, missing prepared-seal verification, incomplete-run inference suppression, termination-field validation, a stop-clipped acceptance metadata inconsistency and a test import portability problem. The final integration also explicitly checks native position metadata. Earlier private review logs remain preserved.

## Remaining gate and limits

No GPU/model work, optimizer step, training or checkpoint modification occurred. No empirical answer to the restart question is available yet. The four-hour limit is a hard budget, **not a runtime forecast**; the reconstruction cost has not been measured natively.

Before execution, verify current A800 jobs/storage, synchronize the authoritative operating notes, stage this exact published source and private sealed inputs, and authenticate all live weights/runtime files. Admission reserves 8 GiB plus 2 GiB for compact artifacts; live source snapshots are held in host memory and released per request. Native GPU memory feasibility and exact neutral replay remain unperformed.

The next milestone is the prescribed neutral successor reconstruction using the actual FA3 runtime. Only a valid reconstruction allows crossed continuations. A supported result could justify proposing a separate short allocation pilot; it does not authorize implementing that algorithm here. Sparse, conflicting, boundary-only or incomplete evidence remains inconclusive. No latency or unique causal attribution claim follows.

[Source identities and preparation wall interval](PROVENANCE.json) distinguish this implementation from the prior PR33 scientific execution. Private prompts, canonical token streams, live tensors and raw traces are not published.
