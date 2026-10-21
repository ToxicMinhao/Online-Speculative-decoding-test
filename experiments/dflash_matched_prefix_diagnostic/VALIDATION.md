# Implementation validation

Status: **schedule-replay repair implemented and CPU-checked; awaiting review
before GPU qualification.** The original cold-prefix attempt failed the all-nine
reconstruction gate and remains unchanged evidence. No adaptation comparison is
available.

## Repair validation

The repair prefills only the original prompt, replays saved verification inputs
through the native target verifier/cache commit path, and then evaluates the
selected draft block. It verifies fresh target predictions and commits; saved
acceptance lengths are checks, never forced outputs. The original all-nine F
proposal equality gate remains intact.

**466 CPU tests pass**: 222 in this diagnostic and 244 inherited PR9/10/11 tests.
The wrapper fixtures cover the original failed cache sequence (87 prompt tokens
plus commits 7/1/1/5), all three checkpoints, chunked original prompts with int32
cache tables/int64 copied indices, target/commit/cache mismatch stops, durable
partial-work counters, unchanged measured-block limits and wrapper restoration.

A source audit also identified an inherited observer incompatibility: D202's
whole-response trace hooks reject verification without an active trace. The
diagnostic now runs D202's source guard and starts its preserved D181 scheduler
without those observation hooks. Native verification/acceptance and D181 weight
publication/cache transactions remain unchanged. Four CPU checks cover this
entry point and its rejection paths. `REPLAY_OBSERVER_COMPATIBILITY.json` records
this adaptation at startup.

An independent **7,610-check** retained-artifact audit confirms every old anchor
field, selected request, checkpoint and evidence binding is unchanged. The v2
plan contains 30 requests, 240 locations, 720 measured draft blocks and **25,854
historical target-verification forwards** for reconstruction. Those additional
forwards are explicitly budgeted and timed; scored-block verifications remain
zero. Per-group reconstruction counts in manifest order are 2,100 / 2,088 /
3,948 / 4,773 / 4,977 / 7,968. No GPU work ran for this repair.

The revised private Fenrir plan hash is:

`ac7d979df1156e4605b7eb1748ceafa9ab03b0a1ec38ea109d06c051666e9777`

The repaired scientific-source digest is:

`25b1e024cea49958356f47a29a182cf28b6395feb36f61decfb013d0dd2cd6d9`

Storage preflight now allows 192 MiB per group plus 1 GiB retrieval staging,
**2.125 GiB planned writes plus the unchanged 8 GiB reserve**. The larger budget
covers duplicated schedules/receipts in plans, events, results and reports and
does not rely on compressed staging. Fresh server space must still qualify.

These tests validate implementation contracts using CPU tensors and stubs. They
do not establish native GPU equivalence or prove why the original token changed.
The first complete group must still pass before further execution; an all-nine
mismatch must stop rather than be waived.

## Preserved original-attempt evidence

## Native qualification attempt

Executed scientific commit `e82681540b719647879583b70692583311bc6c1e` on A800,
without source changes or retuning. All 413 tests also passed with installed
Torch through the existing isolated test-tools directory. Source/runtime/model
bindings and 13 unique checkpoint identities passed admission.

The controller stopped in `gsm8k_00` after ten F evaluations: nine reconstructed
all nine retained proposals, and one differed at the eighth proposal. Its target
prefill was only 101 tokens, so no 8,192-token chunk boundary was crossed. Both
the retained and reconstructed block have a five-token correct prefix; the
difference occurs after their first error. The full reconstruction criterion
still fails and was not relaxed.

The fresh BF16 head has two tied maximum-logit tokens at the changed position.
The retained original logits and the identity of the second fresh tied token
are unavailable here. Numerical sensitivity is a possible explanation, not an
established cause. Target-seed, masks/positions, cache, finite-logit, native
argmax inspection and mutation-guard checks passed. Cold versus incremental
execution and any implementation discrepancy remain unresolved.

There were zero S8/L8 evaluations, valid pairs, training steps, generated response
tokens, target verifications, retries or explicit warmups. All other groups
remain unexecuted; no scientific benefit/harm conclusion follows. Campaign wall
time was 75.97 s; cleanup passed and both GPUs are idle. Peak allocated memory
at the failed probe was 69,924,793,344 bytes. All 77 retrieved compact files were
hash-verified privately. Responses, raw tokens and runtime artifacts remain
outside Git. No automatic restart is authorized by this failure report.

## Original implementation CPU checks

**413 tests passed** across this module and the inherited PR9/PR10/PR11
packages (169 new diagnostic tests and 244 inherited tests). Local test runtime:
Python 3.11, PyTorch 2.7.1+cpu, pytest 8.3.2; GPU execution was not used.

```bash
python -m pytest -q \
  experiments/dflash_matched_prefix_diagnostic \
  experiments/dflash_baseline_learning_curve \
  experiments/dflash_kl_native_diagnostic \
  experiments/dflash_lower_lr_control
```

Coverage includes deterministic lower-median source selection and endpoint
sampling; immutable evidence/plan bindings; EOS-inclusive primary matching
length; non-EOS length and EOS flags; cap censoring; first-error repairs blocked
by earlier errors; all-nine frozen reconstruction including unscoreable tails;
native BF16 head/FP32 inspection and ties; the 8,431-token chunked-prefix fixture
with an int32 cache pool and owned int64 prefix indices; seed mismatch stopping
before drafting; one-block/no-verification guards; weight/cache ownership;
durable nonfinite and unknown-work failures; first-group correctness gating;
continuation on harmful valid results; reporting revalidation; equal-request
aggregation; and missing-arm versus observed-invalid denominators.

Torch/native-wrapper tests use CPU tensors and deterministic runtime stubs.
They verify integration contracts, not actual CUDA kernel correctness or output
preservation on A800. State guards use parameter/buffer identities and PyTorch
mutation versions, explicitly not full tensor-value hashes.

## Original retained evidence and private plan

CPU-only admission of the original PR9/PR10/PR11 compact artifacts passed. The
selected plan contains all **30 requests, 240 locations and 720 planned native
block evaluations**, with 40 locations per group and no zero-round requests.
Source controls use frozen output length and original manifest order only.
All source records, token IDs and plans remain outside Git.

The final plan seals checkpoint-major F→S8→L8 execution within each group,
preserving cold caches at every location and avoiding unnecessary checkpoint
reloads. Its private Fenrir artifact bindings give plan SHA256:

`5ef7ae802f74beef59c63ecab3aef55624596ac056fc315ddaa59db61fee82aa`

The reviewed scientific-source inventory digest is:

`df0c78dd7779ba66085dbc98d8254bf9b0c3a62cfade2bc217cb315c532bf9d6`

Plan bindings include local evidence paths. A server-side plan must be prepared
and sealed against the exact unchanged artifacts at their execution locations;
selection IDs/offsets/proposals must stay identical. The run command requires
that sealed plan, the reviewed source hash, a clean checkout and explicit GPU
execution authorization. It will not silently reselect or rerun a failed probe.

## Remaining native qualification

The first complete group must establish target-seed reconstruction, all-nine
frozen proposal equality, per-arm geometry/cache correctness, native argmax
inspection, immutable weights and actual counters. The repaired original-schedule replay must establish native equality; the old
cold-prefix result does not satisfy this requirement. Missing artifacts or
correctness failures must be retained, not replaced with regenerated evidence.
Valid unfavorable adaptation results do not prevent the remaining groups.

The original preflight budget was 448 MiB plus reserve. The revised budget is
2.125 GiB plus the unchanged 8 GiB reserve, without checkpoint copies, optimizers
or feature tensor exports.
Fresh remote free space, device availability and checkpoint/runtime hashes must
be checked at execution. No current A800 readiness or scientific result is
claimed by these CPU checks.
