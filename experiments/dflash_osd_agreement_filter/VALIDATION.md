# Implementation validation — review hold

Validated locally with CPU Torch 2.11. No A800 access, model loading, native GPU
qualification, training experiment or pilot serving occurred. The pinned external
runtime remains an execution prerequisite; CPU tests do not establish its current
GPU compatibility, memory fit, output preservation or execution cost.

| Suite | Passed | Skipped |
|---|---:|---:|
| New agreement-filter package | 59 | 0 |
| Reused per-turn package | 52 | 2 |
| Reused author-path follow-up | 71 | 0 |
| Reused two-update package | 64 | 17 |
| Total | 246 | 19 |

Each suite ran in a separate Python process. Existing historical suites use
colliding top-level test-module names, so collecting all four in one process is
unsupported; no historical files were changed to accommodate that. The new
package's cross-package imports were repaired and its normal pytest command
passes. Skipped inherited checks require optional local artifact/installed-source
bindings; they are not evidence of native qualification. Local Torch emits a CUDA
availability warning; all numerical fixture tensors and tests here use CPU.

## What the checks cover

- Tiny gradient/displacement products cannot underflow into a permitted sign;
  zeros, momentum and weight decay use the actual signed AdamW displacement.
- One global aggregate clip and one Adam state advancement; full native BF16
  state preserved even when no displacement is applied.
- N's own global norm across tensors, independent state and reported BF16
  rounding difference; no G trajectory is consulted.
- Identical sampled forward and original objective, teacher alignment, full
  denominator, fixed absolute-position groups including overlapping anchors,
  normal per-pass sampling, masked older answers and explicit empty-group failure.
- H reproduces the unchanged qualified trainer; isolated filter-disabled
  comparison restores exact optimizer/model/global/anchor RNG state. Its
  qualification checks displacement, weights, moments and exposure, including a
  fixture rejecting zero or half of a nonzero reference update.
- Four independent own-history trajectories, exact common input IDs, 48 main
  events/96 steps/60 scores, no learning from replays, no final update, mismatch
  retention and H-primary interpretation in reports.
- Separate first-stream qualification and new six-stream workers afterward;
  failed fixtures/qualification cannot launch the campaign.
- Complete activation timing remains inherited; new worker dispatch and source
  pin admission are tested. Invalid/incomplete arm evidence blocks retirement;
  valid owned completion evidence is durable before rolling bulk is removed.

## Actual data checks without model execution

Prepared a portable reference bundle from the six **existing completed** per-turn
RESULT files. Approved source text, whitespace, message IDs/order and exact token
arrays pass validation. The bundle contains 12 immutable turn-2/3 contexts:

- File SHA-256: `46087bd9055aa9e8f30389b758ba7ef84c4f470e9112584dc334467c3bdaf4e9`
- Bytes: `519962`
- Manifest SHA-256: `7332c8e81422617ab99f86a0fe391a07f1328aa75620c8718fa1348757a001de`

No raw user text, canonical token bundle, results, checkpoints or private machine
configuration is uploaded. A metadata-only plan passes and records code/reference
hashes plus all main and qualification counts. Text/tokenizer and native GPU
checks remain for the reviewed execution phase.

## Remaining qualification

Before the main campaign, run the declared deterministic guards and three-request
first-stream check. The GPU filter-disabled comparison must satisfy the pinned
native-precision and displacement tolerances, not merely report them. Validate
native outputs, caches and peak memory. Correctness failures stop execution;
unfavorable efficacy does not authorize retuning or block a correct campaign.

**Await review before GPU qualification or experiment execution.**
