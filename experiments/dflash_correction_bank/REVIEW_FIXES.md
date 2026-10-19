# PR #5 review fixes

Reviewed baseline: `b32e45c45b709ffb6a3b75efa359ff219b260de1`.
All three findings are valid. This revision changes integration and qualification;
it does not redesign the bank or alter the OSD recipe.

## 1. Prefill state mutation — confirmed in pinned source

The locally retained runtime file matches the configured SHA-256 exactly:

- `speculative/dflash_worker.py`: `9397402f5cfbf60b00b7d27d551e5e9e1b605be9b084d028d4ddda76c475dc8c`.
- Its lines 1264–1265 clear `draft_input.ctx_lens` and consume target features.
  Lines 1549–1550 append the target features to draft KV and assign that consumed
  object to `batch.spec_info` before returning.
- `managers/schedule_batch.py`: `e72161a7245e882eef40612a728fd6f72963498666a8caa4fa5b8b8d45413199`.
  Lines 2206–2207 supply the model worker's prefill geometry from `extend_lens`
  and `prefix_lens`.

The reviewer also identified the corresponding behavior in the
[public DFlash runtime](https://github.com/rednote-machine-learning/RedKnot/blob/de97093c81f20622cf330ab0b6199b6b42631799/python/sglang/srt/speculative/dflash_worker.py#L1008).
The local hash check establishes the issue against our configured source;
this revision did not query or execute the live A800 installation.

The wrapper now snapshots the chunk length and prefix offset as Python integers
**before** native forward. It supports initial `spec_info=None`, validates the
returned feature count, and never consults consumed `ctx_lens` for geometry.
Four new tests install the actual wrapper around a mutating runtime stub. They
cover an 8,192 + 239-token prefill, routing exactly once after the final prompt
chunk, feature-count and prefix-gap failures, and unchanged native head behavior
for zero corrections and OSD. Verification logic itself is unchanged.

## 2. BF16 correction survival — qualification gap fixed

The previous compressed/direct checks both evaluated the mathematical objective.
They could not reveal whether a fitted correction vanished at the native hidden
state cast. New qualification-only measurements apply the real serving correction
helper and native vocabulary projection to the already sampled frozen features.

Cases are the served fusion, the fitted singleton when available, zero, and a
feedback-independent control `C = r I / sqrt(rank)`. The control respects the
same radius; it is never published or used for learning. Reported quantities are:

- FP64-reference and native loss changes, each relative to its own uncorrected
  baseline and using identical teacher probabilities.
- Centered logit-change norms, native/reference ratios and discrepancies.
- Feature corrections before and after the native cast, changed coordinates and
  positions, and the fraction of nonzero corrections surviving rounding.
- Compact per-position evidence, execution dtypes and diagnostic time.

Centering removes irrelevant common logit shifts. Undefined ratios are null;
rounding can amplify a change, so magnitude ratios are not bounded percentages.
Tests explicitly detect a tiny correction erased by BF16 rounding and a larger
surviving correction, plus zero, common-shift and empty-exposure cases.

Qualification requires that the measurement was made, not that correction
survival or performance is positive. At least one response must provide actual
nonempty measurements. Its time is included in qualification-only costs. The
retention guarantee remains scoped to the mathematical fixed-feature objective;
these diagnostics do not extend it to native logits or future verification work.

## 3. One-token final answer — explicit empty analysis

A one-token final answer has no eligible sampled block under the inherited
valid-count-minus-one rule. Its exact response and token hashes are audited,
then BANK emits empty features/anchors and a no-op analysis receipt without
calling the teacher, backbone or sampler. RNG is unchanged, no owner is selected,
no write is scheduled, and the unobserved served loss gain is null.

This exception applies only to final analysis. Nonfinal BANK updates and OSD's
training validation, anchors, objective, steps and optimizer are unchanged.
Malformed IDs and empty serving responses are still rejected. Tests cover the
actual feedback-service path, unchanged state, final scheduling and malformed
inputs. Empty final evidence cannot make the aggregate qualification vacuously
pass.

## Validation and scope

- 63 correction-bank tests passed, including 11 added regression tests.
- 123 existing per-turn/follow-up tests passed; two optional artifact/tokenizer
  checks skipped.
- 64 earlier two-update tests passed separately; 17 optional installed-source
  checks skipped.
- The pinned six-stream plan still gives 116 contexts, 110 updates per learning
  arm and 1,044 main timed comparisons.

Only code and CPU checks ran. First-stream GPU qualification and the clean
six-stream campaign remain unexecuted. No learning hyperparameter, scientific
arm, dataset or performance acceptance threshold was changed.
