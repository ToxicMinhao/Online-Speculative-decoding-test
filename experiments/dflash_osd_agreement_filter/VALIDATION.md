# Implementation and native qualification status

## Direct-full repair: native equivalence passes; N control fails

[Detailed result](NUMERICAL_REPAIR_QUALIFICATION.md), executed code `a142802`.
All 164 package CPU tests and 47 installed-Torch checks pass. Native disabled-filter
weights, Adam moments and actual displacement exactly match unchanged H after
two steps; counters, anchors, RNG and restoration match exactly. All 8 historical
sum checks remain failed diagnostics under the explicit revised protocol.

N fails the unchanged 1% norm gate (31.19% and23.72% errors). Qualification stops
with 10 canonical generations, 8 valid scored replays, 7 events/14 steps plus 4
restored temporary steps. Main campaign absent; no efficacy conclusion. Runtime
recipe metadata now correctly records one response/two passes/two steps.
Third full backwards cost 0.282–0.336s per normal G/N pass; complete instrumented
training and memory costs are reported separately. 300 independent checks audit
the retained evidence without claiming independent raw-tensor reconstruction.

The historical attempts below retain their original outcomes and limits.

## Native retry outcome (qualification failed; pilot not started)

The compatibility repair passes **99 local CPU tests** and **two focused tests
on the installed A800 Torch 2.9.1**, including the original compiled-backward
reproduction and repeated-backward success. Fresh qualification confirms the
compiler setting, then stops after 548.71 seconds at G's **fixed raw-gradient
sum tolerance** during its first update's temporary disabled-filter trial.
H/L complete four events/eight persistent steps and six valid paired scores;
G has no committed update, N is unrun, and the main campaign is not created.

The failure log does not retain the offending parameter/coordinate, gradient
values, error size, or grouped-pass index. BF16 accumulation versus an
implementation discrepancy is unresolved. No tolerance, precision, learning
rate, or mechanism was changed to force continuation. The correct status is
**unqualified**, not evidence that selective filtering helps or fails.

## First native attempt and compiler compatibility repair

The user authorized native qualification and the unchanged bounded pilot only
after every correctness and numerical-control gate passes. The first attempt
on installed Torch `2.9.1+cu130` stopped after 547.60 seconds: H and L completed
their first-stream checks (four update events, eight persistent optimizer steps,
six valid paired scores); G failed during the temporary disabled-filter trial
before a persistent update. N and the main campaign were not started.

The compiled backward rejects `retain_graph=True` when donated saved buffers
are nonempty. The compatibility repair pins
`torch._functorch.config.donated_buffer=False` before model setup/compilation for
**all H/L/G/N trainer workers**, including the ordinary reference trial. It
records the effective flag and Torch version, and rejects later drift. No
precision, loss, learning rate, optimizer, anchor, exposure, inference setting,
or acceptance tolerance is changed. Original author/runtime files are untouched.

A focused CPU AOTAutograd regression reproduces the original donated-buffer
error after an ordinary backward warm-up, then verifies three backward calls
from one compiled forward after the fix. Trainer checks also cover all four
arms, configuration drift, and pinning before native model construction.

The failed attempt and its completed controls remain preserved. The fresh native
retry is reported above; this repair is not a filtering result.

## Reviewer control fixes (historical, before native execution)

After `c5cf3e7`, the updated package passes **93 CPU tests**. Native GPU
qualification remains unrun. The targeted reviewer cases now reproduce and reject
incorrect control/measurement validity:

- BF16 weights `[0.05, 0.05]`, LR `3e-4`, gradients A `[1,1]`, B `[1,-2]` produce
  N's unchanged actual norm about `0.00034527` against intended `0.00024414`.
  The 41.4% discrepancy fails the predeclared 1% per-step norm criterion; the
  implementation does not alter precision or the update to force a pass.
- A score with accepted draft tokens exceeding output tokens retains its failure
  status/errors and cannot enter report gains. Reporting reruns the shared
  validator, rejecting forged passed status and checking the same cost, token,
  count and capture constraints. Nonfinite failure evidence round-trips through
  strict JSON with explicit lossless markers.
- Control tests cover threshold boundaries, zero/tiny/nonfinite norms, missing
  or duplicate steps/events, stale receipts, cumulative invalidity and checkpoint
  ancestry. Failed N qualification cannot start the campaign; failed updates
  retain checkpoint/recovery evidence and do not retire it.
- Descriptive native comparisons remain separate from selective-filter attribution;
  qualified-control denominators are disclosed, and H remains primary.

Independent review also passed 64 focused checks across control, measurement,
reporting, controller and numerical-core tests. All checks are CPU-only; historical
scientific source is unchanged. The original validation below is retained as the
initial implementation record, not a new execution claim.

## Initial implementation (`c5cf3e7`)

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

**Execution is authorized, but the campaign remains gated on a complete native qualification pass.**
