# Implementation validation

Status: **CPU implementation and retained-evidence admission checked; native GPU
qualification has not run.** No new model evaluations, training, generations,
target verification or response-level KL computations were executed.

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
