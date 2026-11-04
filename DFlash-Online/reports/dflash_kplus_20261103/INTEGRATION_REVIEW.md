# K+ objective, trainer and retention integration review

## Material Passport

ARS experiment-agent, run→validate. Verification status: **CPU integration qualified**. This is a same-team review with implementation overlap. It covers source, synthetic logits, CPU toy-model lifecycle and retained-state fixtures. It is not a GPU experiment or independent scientific replication and makes no efficacy claim. Scientific base is `4d94f30c42701d0f86ac04e2829c834593c16d8e`; parent publication will identify the new execution commit. Exact reviewed file hashes are in `INTEGRATION_QUALIFICATION.json`.

The final frozen-source command passed **75 tests, zero failures/errors, exit 0, in 44.725 seconds**: 39 K+ tests and 36 inherited regressions. CUDA was hidden before Python startup. The complete output is `INTEGRATION_CPU_FINAL_001.log`; the exact command, working directory and hashes are in the qualification receipt. No pinned Qwen model or production request was run. Small CPU model/optimizer fixtures test state continuity and native K update equality; they are not additional production training.

## Findings resolved before model work

1. `storage.boundary_files` originally matched `event_001_validation.json` as though it were a boundary manifest. The first post-event storage check would therefore fail after event 1. The scan now accepts only `event_[0-9][0-9][0-9].json`; a fixture retains both real marker and scientific-validation sidecar and passes admission.
2. Admission originally credited logical tensor size and counted compact logical bytes. It now uses `st_blocks*512`, deduplicates device/inode identity and includes directory metadata. Sparse files receive only their physical allocation as credit; hardlinks count once; symlinks fail. No expected future deletion receives credit.
3. The compiler allowance originally had no measured baseline. First admission now writes `COMPILER_BASELINE.json` before model work. Later checks retain the same Torch/Triton/TVM/CUDA/SGLang/FlashInfer cache inventory, measure allocation growth and stop above the fixed 4 GiB allowance. FlashInfer is limited to `FLASHINFER_WORKSPACE_BASE/.cache/flashinfer`, not the whole workspace. Missing baseline after worker work or changed cache roots fails. No automatic allowance extension is added.
4. Prepared-feature retirement originally checked token identity, mask and shape without comparing the retained tensor rows. It now decodes the prepared, native and terminal files on CPU, verifies BF16 and exact native/terminal row equality, then writes an equivalence receipt before unlinking either source. Changing one native-row or terminal-row value prevents both deletions. Validation and retirement wall times remain separately recorded; extra model forwards are zero.
5. Parent fixed the new preflight call to supply the native snapshot validator's required `expected_contract` and `expected_recipe` arguments. Without that repair it would fail before initialization.
6. Parent added physical storage checks beside quiescence checks at final evaluation boundaries. The callback remains outside timed request execution and retains the two receipts.

No unresolved blocker remains in the reviewed objective/storage/trainer/campaign integration. Actual deployed source/input identity, capacity, runtime and GPU correctness gates remain requirements of the later production preflight; CPU tests do not establish those outcomes.

## Objective and native-path preservation

`install_kplus` adds a fixed reducer to the existing BF16 `osd_first_error` object without replacing parameters, teacher/model passes, sampler or optimizer. Chunk outputs retain N_K/S plus detached W1/W2 and differentiable N1/N2. The correction is applied only after all eight-block chunks for a response are summed. The original denominator and single response epsilon are preserved. Correct-prefix coefficients and all-correct blocks stay unchanged; a successor's correctness is read without selecting its KL.

The no-error, single-priority and empty branches return the original N_K tensor directly. Synthetic BF16 loss and gradient equality with native K is exact for those branches. The inherited full-K toy-model test also verifies that observation hooks preserve the plain K update exactly. Existing K registration and two-arm guards remain unchanged. The optional prefix-guard path retains its original term layout when K+ is absent; installation refuses combining K+ with that correction.

The mixed fixtures place the two priority classes in separate homogeneous eight-block chunks with unequal masses, so a mistaken per-chunk correction cannot pass. They compare the explicit weighted loss and gradients with the scalar formula in FP64, FP32 and native BF16 arithmetic under the predeclared tolerances. EOS, anchor, validity-gap and end-of-block cases pass. Diagnostic counts are returned from original forwards, not accumulated by checkpoint recomputation. Cost counters include actual recomputation work and are not presentation counts.

## Trainer and campaign boundaries

The new trainer explicitly installs K+ after the shared native runtime is built and records the complete fixed contract. Its metadata-only view lets the unchanged K state validator inspect weights, Adam, RNG, counters and history; it does not project or replace those scientific values. Fresh initialization is independently created from released weights and compared with the authenticated original K initial state. K58 is never loaded into the candidate. Restore requires its own complete K+ state and unchanged contract.

The controller preserves the 128-response event barrier, two shuffled passes, two-response accumulation and full 7,424-step horizon. It collects each event from the currently published K+ checkpoint, prepares each response once and commits/publishes before the next event. The last 49 stream requests are served without preparation or another update. Natural anchors are recorded and compared when original receipts are available; they are never forced to match a historical schedule.

The four inherited correctness generations are exactly one capture-disabled replay plus updated/frozen/reloaded checks. Initialization, export, state, decoded-feature and publication comparisons add no target/draft training forward or optimizer step. Root ledgers still enforce 7,473 stream generations, four correctness generations, 7,424 terminal preparations, 14,848 ordinary response forwards/backwards and 7,424 K+ steps. K's new optimizer-step budget is zero. Final evaluation adds the fixed 6,144 capture-disabled generations, with no extra efficacy warmup.

Retirement occurs only after durable equivalent preparation or validated successor commitment/publication. Prepared features remain through training. Preceding state/export files are inside the new root and must have the exact predecessor relation and a sole obsolete link before deletion; current/initial states and compact receipts remain. No historical file is selected by these retirement functions. Deleted source/intermediate tensors are retired, not archived, and their lost replay/resume capability is recorded.

## Preserved fixture-environment failure

The first combined command inherited visible-device metadata. Its 73 other tests passed, but two unchanged inherited `FullKTinyIntegrationTests` failed at initial-state validation because their CPU contract expects zero CUDA RNG states. Both failed before any event training. `INTEGRATION_CPU_ATTEMPT_001.json` retains the exact named tests, exception/location, command and count/time. The source and expectations were not changed.

With `CUDA_VISIBLE_DEVICES=''` set before Python imported Torch, those two fixtures passed in 31.364 seconds (`INTEGRATION_CPU_ATTEMPT_002.json`). The final complete CUDA-hidden 75-test command then passed. This was a CPU fixture-environment correction; no production experiment was retried or scientific setting relaxed.

## Final parent follow-up

Before publication, the parent added the report Material Passport, postflight growth checking, copied-input identity checking and a conservative capacity correction that also includes token-ID/mask bytes for overlapping preparation rows. The parent reports 15 focused storage/report tests passed afterward. The final source hashes in `INTEGRATION_QUALIFICATION.json` include those changes; the original 75-test command and its result remain separately retained. This follow-up adds no model work or scientific settings change.
