# Implementation validation

Status: **CPU checks passed; the first native qualification attempt failed the
all-nine frozen-proposal reconstruction gate.** No adaptation comparison is
available. The CPU-only preparation described below preceded this native run.

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

## Local checks

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

## Retained evidence and private plan

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
inspection, immutable weights and actual counters. Cold-prefill versus original
incremental-cache numerics remain an explicit uncertainty. Missing artifacts or
correctness failures must be retained, not replaced with regenerated evidence.
Valid unfavorable adaptation results do not prevent the remaining groups.

Storage preflight budgets 448 MiB of compact records/staging plus the unchanged
8 GiB reserve, without checkpoint copies, optimizers or feature tensor exports.
Fresh remote free space, device availability and checkpoint/runtime hashes must
be checked at execution. No current A800 readiness or scientific result is
claimed by these CPU checks.
