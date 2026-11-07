# Role-loss pilot integration review

## Material Passport

- Workflow: ARS experiment-agent, implementation/reproducibility review within the authorized pilot.
- Review date: November 7, 2026.
- Status: **CPU implementation review passed; native qualification and server admission remain required.** No unresolved launch blocker was found in the final reviewed integration.
- Publication binding: all seven reviewed file hashes were rechecked against local published commit `cdb640543d8fe3a43c8e63af4ccaaa002e09a599` ([PR #32](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/32)); every reviewed byte identity matched.
- Scientific scope: six independently initialized KPLUS/MIX/ROLE trajectories, two datasets, four events, event-2/event-4 evaluation, and the bounded fixed-context diagnostic.
- Evidence: source inspection, retained input metadata, CPU fixtures using the actual tiny author DFlash model/TrainerCore, and focused failure-path tests. No GPU, remote server, model-scale native execution, or production experiment was performed by this reviewer.
- Authorship disclosure: this reviewer authored the new evaluation, statistics, diagnostic and report modules, then reviewed the campaign/training/qualification/preflight/storage implementation written by another same-team agent. This is a separate integration review, not an external review or independent GPU replication.

## Findings resolved before this review closed

| Finding | Consequence before repair | Final repair and verification |
|---|---|---|
| Campaign boundary receipt used `resource`/`timing`, whereas the evaluator consumes `storage`/`quiescence`. | The first timed block would fail on a missing key before any timed generation. | `evaluation_boundary` now supplies the evaluator schema, including the `required_bytes` alias and the unchanged timing barrier result. A direct boundary-to-evaluator CPU test passes. |
| Selected event exports were obtained from trajectory status without rechecking their boundary identities. | A changed event-2/event-4 export could be published successfully under an intended checkpoint label. | `selected_checkpoints` validates completed trajectory/event identity, exact export path and the saved boundary weight hash before and after the stage. The corruption fixture is rejected. |
| The launch script invoked nonexistent `test_role_loss`. | `set -e` would stop the launcher before preflight. | The script invokes `test_role_loss_objective`; the launcher test verifies that every explicitly named test module exists. |
| The outer hard timeout could terminate the parent while its detached training child continued. | Ordinary Python SIGTERM would bypass exception cleanup of the owned training/serving processes. | Campaign and qualification entry points install SIGTERM/SIGINT handlers that raise into failure accounting and cleanup. The parent signals the owned child process group and permits a 90-second cleanup interval inside the outer 120-second grace. A failing trajectory marks its trainer failed and closes serving before the trainer. The handler test verifies that both signals enter the exception path. Forced uncatchable process death is not claimed to run Python cleanup. |

The campaign agent made these repairs. This reviewer did not edit the five reviewed campaign modules.

## Contract checks

**Initialization and continuation.** Each dataset/arm runs in a fresh process. The candidate loads released weights through the unchanged native construction and saves a genuinely empty initial optimizer state. Its scientific state is compared with the authenticated original seed-17 K initialization; the comparison is not a K+58 continuation. Only experiment identity metadata is projected through the historical native validator. Weights, optimizer recipe and moments, RNG, counters, event history, accumulation and the original 7,424-step horizon remain checked. The pilot enforces four events in both the campaign loop and trainer preparation guard; it does not rewrite the stable historical validator to accept a fifth pilot event.

**Response order and backward lifetime.** Native two-pass shuffles, microbatch one and accumulation two are preserved. Inspection of the unchanged `TrainerCore.train_step` confirms that a response's backward returns before the next response's forward can install another saved full-response denominator. The new objective receives the existing native chunk logits and target projections; the campaign introduces no auxiliary training backbone/head pass. Finite loss is checked before backward and finite accumulated gradients before Adam. There is no probability stabilization, example removal or automatic retry.

**Bounded qualification.** The six-record retained reconstruction inventory is authenticated. Only its first two response pairs are used for two disposable updates per arm: six updates and twelve response forwards/backwards in total. Each case restores the authenticated fresh state; temporary decoded weights, moments, counters, core microstep and RNG are compared exactly, then the initial state is restored. The qualification log rejects an attempted seventh update or thirteenth response. Nonfinite first-backward injection stops before any Adam call and retains failed-work evidence. Qualification artifacts never enter production initialization. Native full-size numerical execution remains a required later gate; CPU fixtures do not substitute for it.

**Production work.** Across the six trajectories the ledgers permit exactly 3,072 own-feedback generations, 6,144 ordinary response forwards/backwards and 3,072 Adam steps. Four inherited correctness completions occur only on each dataset's KPLUS trajectory, totaling eight. These records are capture-disabled and excluded from training/timing. Event-2 results do not select a checkpoint, change an objective or stop an otherwise valid trajectory. Event-2 and event-4 exports and event-4 recovery states are retained.

**Diagnostic isolation.** The shared pool is prepared before the own-feedback counting hooks are attached. The worker adds only the idle `roleloss_reconstruct` RPC to the established native worker and calls the previously qualified retained-native-block reconstruction directly; it installs no depth controller or serving wrapper. Each target posterior and committed continuation is checked during replay. Sixteen canonical contexts per dataset are prepared once. Native prefill, verification replay and terminal target forwards are counted separately from the 32 response preparations. Exact decoded row/input/mask equivalence is established before retiring only the duplicate reconstruction and terminal tensor files; the prepared packet and compact receipts remain. Snapshots read the full canonical context directly, avoiding the training adapter's 2,272-token truncation. The native diagnostic runtime restores mode, sampler, RNG, registered buffers and transient caches. A post-diagnostic scientific boundary check verifies the live trainer before continuation.

The retained first sixteen diagnostic records are all eligible: GSM8K has minimum response length 150 and maximum total context length 642; MathQA has minimum response length 213 and maximum total context length 887. Thus this selected input set has no zero-anchor/short-response preparation edge case. The fixed schedule produces 32 shared-initial and 192 post-training draft response forwards, at most 224 total; those records never feed training.

**Evaluation.** All six trajectories finish before the shared evaluation backend starts. Each stage uses the sealed 32-request-block order and the identical pinned native serving path. Publication, weight equality, storage and process/transfer quiescence checks happen outside timed requests. Each generated record is written before correctness validation. Exact canonical IDs and EOS/length termination are required, with all cap cases retained: the selected panels contain one GSM8K cap case and nine MathQA cap cases. Event 2 has 1,536 total timed completions; event 4 has 3,072. Repeats must agree on deterministic counts. Reported endpoint time uses each request's mean of its two repeats; requests, not repeats, are resampled by the fixed PCG64/10,000-draw bootstrap. Per-dataset contrasts and the shared-context explanatory records remain separate.

**Storage.** The phase calculation was checked against the live retention lifecycle. Five completed cells retain six weight-equivalents each (initial state, final model/Adam recovery and two exports). The active cell's worst transition retains ten weight-equivalents, including old/new recovery containers and the retained event-2 export, for a training peak of forty. Evaluation retains thirty-six. The budget additionally includes 128 full-cap training feature packets plus replacement overlap, 32 full-context diagnostic packets, raw/compact evidence, compiler growth, workspace/retrieval staging, a 5% tensor margin and an 8 GiB reserve. Admission uses the maximum of the phase requirements and credits actual validated allocated inodes only; it does not credit a future deletion. Event-1/event-3 exports and obsolete recovery states are retired only after successor publication and identity checks. A failed publication prevents retirement, and the event-2 export survives the next event.

## CPU validation performed

The reviewer ran these focused checks with `CUDA_VISIBLE_DEVICES=''` and the specified local SpecForge environment:

1. Campaign and scientific-state tests: the initial run passed 15 tests, including actual tiny native initialization, a complete event, and own-state resume for every arm.
2. Qualification and worker transport tests: 13 tests passed, including six actual tiny native BF16 Adam updates/twelve presentations, decoded restoration, first-backward nonfinite rejection, inherited serving methods and the single reconstruction RPC.
3. After the integration repairs, campaign plus qualification tests were rerun: **16 tests passed in 12.010 seconds**, including the new boundary, checkpoint corruption, launcher-name and signal-handler checks. Counts across these invocations overlap and are not added as independent evidence.
4. The separate authored evaluation/diagnostic/report suite passed 17 tests, covering all 4,608 scheduled slots, raw/native audit failure paths, unknown/unattempted work, direct bootstrap comparison, cap inclusion, intact-prefix repair/persistence, real author-forward isolation, proven duplicate retirement, and incomplete reports retaining all six cells.

No CUDA runtime was initialized by the tiny native test children. `git diff --check` was clean at review.

## Reviewed source identities

These hashes identify the reviewed bytes and match published commit `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`. Publication is distinct from successful native qualification or experiment execution.

| File under `experiments/role_loss_pilot/` | SHA-256 |
|---|---|
| `campaign.py` | `00fe0d962651eafda3d497348b1107623612fa278bf8e9353aa2cf3a8a958ae9` |
| `training.py` | `ed9a41014e317e972fa275ccc548ddb55190379dee21ac3c9eb5b68716a1d8da` |
| `qualify.py` | `90d14233672dc2945ddc5d8d94dfe9e23de86637976cf585caad6294e438c66f` |
| `preflight.py` | `6aac5184a9fb6ae2bb45ac619c6a68d5378fa00b7898007cba382b34e4141a62` |
| `storage.py` | `22f26542a0a0d405e1d557e1d2e680cc0280b9e11a3356e1496a157efdd0ae2e` |
| `worker.py` | `a3421abc6ccefba750d58f3e3f3a7c5556a3687a512fa80070a466a917bd8cae` |
| `run_campaign.sh` | `b33b5e07ef53bd902b3b958cab73389914b33d50631e23e17ea09041e7db6608` |

## Remaining gates and interpretation limit

Seal the published reviewed source and inputs, then run the prescribed source/runtime/model/input and storage admission checks, bounded native qualification, and the fixed pilot only if those gates pass. The launch wrapper already orders these gates. A native defect remains a technical failure, distinct from a valid null or harmful pilot result. This review establishes no efficacy result and does not authorize a full-event campaign, coefficient sweep, extra arm or project-stage change.
