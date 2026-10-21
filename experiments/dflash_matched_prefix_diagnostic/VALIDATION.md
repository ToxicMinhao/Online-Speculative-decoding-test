# Implementation validation

Status: **repaired native run completed all six groups, with independent audit**.
Executed scientific commit `133fb74d01633861bf0c31181d45a99669e43cdc`;
subsequent documentation commits do not change that executed code identity.
The original cold-prefix failure remains preserved below.

## Completed native qualification and bounded diagnostic

All **468 CPU tests pass** locally and under the installed A800 Torch runtime.
Admission verifies 13 unique checkpoint identities, pinned model/runtime/source
bindings, and exact parity of all 240 selected locations. Native execution used
server plan `e6f7a8694324095400c365ee8ebdd9ecdb1df42a273ddaa88eb5e2a241b5437e`
and scientific source `6e0ad836d48f9ce9f9ad86ee442e6426cdcfad51f845b1b6690e453fdc7f94b4`.

The first group completed 120 measured blocks and 2,100 historical reconstruction
target forwards. Its native correctness gate passed, so the remaining five
continued unchanged. Total: **720 measured blocks / 25,854 reconstruction target
forwards**, 480 valid F–adapted pairs, zero unknown work, zero unattempted work,
zero invalid comparisons. All **240 F blocks reproduce all nine proposals**,
including the original failed location. Fresh target predictions, commits,
seeds and cache advances also pass their recorded checks.

No training, new response generation, scored-block target verification, retries,
extra warmups, retuning or tolerance changes. The controller exited 0 after
**1,001.03 s (16m41s)** and both GPUs are idle. Independent evidence audit passed
**926,017 checks**; all 1,591 privately transferred files pass size/SHA256 checks.
The audit recomputes recorded evidence and arithmetic, not uninstrumented GPU
behavior or tensor-byte identity.

| Population | Arm vs F | Requests / locations | Mean correct-prefix change | Gain / loss / tie blocks | Useful first-error fixes / eligible F errors | New earlier-error blocks |
|---|---|---:|---:|---:|---:|---:|
| Source controls | S8 (`3e-4`) | 6 / 48 | +2.83333 | 36 / 0 / 12 | 36 / 41 | 0 |
| Source controls | L8 (`3e-5`) | 6 / 48 | +0.47917 | 10 / 0 / 38 | 10 / 41 | 0 |
| Held-out | S8 (`3e-4`) | 24 / 192 | −0.07292 | 36 / 34 / 122 | 36 / 172 | 34 |
| Held-out | L8 (`3e-5`) | 24 / 192 | +0.00521 | 1 / 1 / 190 | 1 / 172 | 1 |

Prefix changes are tokens per block, averaged within requests then equally
across requests. Primary length includes matched EOS; cap-truncated matches
remain censored. The six source controls are illustrative, not an estimate for
all 24 original source responses. S8 shows strong fitting but mixed transfer;
L8 leaves nearly every held-out prefix unchanged. These local changes are not
saved verification calls or serving speedups, and do not establish why response
KL changed. Historical full-response calls remain separate retained evidence.

Diagnostic cost: 129.57 s loading, 519.25 s probe RPC wall, 181.94 s shutdown.
Within execution, recorded prefill is 32.07 s, reconstruction 353.20 s,
activation 44.85 s, cache maintenance 26.87 s and head inspection 3.64 s.
The 306.85 s historical target-verification timer is **inside** reconstruction;
component timers must not be added again to wall time. Peak observed allocated
GPU memory: 70,127,768,064 bytes. A800 free space after packaging: 11.65 GB,
above the unchanged 8 GiB reserve. No artifacts were deleted.

The bounded run is complete and stopped. Private prompts, token IDs, traces,
checkpoint payloads and runtime artifacts remain outside Git. Native success
validates original-schedule replay on these cases; it does not prove the exact
numerical cause of the preserved cold-prefix failure. Earlier campaigns remain
paused; no new mechanism or experiment is started.

The following sections preserve the implementation checks, original failure,
and pre-execution requirements as historical records.

## Repair implementation and pre-execution checks

The repair prefills only the original prompt, replays saved verification inputs
through the native target verifier/cache commit path, and then evaluates the
selected draft block. It verifies fresh target predictions and commits; saved
acceptance lengths are checks, never forced outputs. The original all-nine F
proposal equality gate remains intact.

**468 CPU tests pass**: 224 in this diagnostic and 244 inherited PR9/10/11 tests.
The wrapper fixtures cover the original failed cache sequence (87 prompt tokens
plus commits 7/1/1/5), all three checkpoints, chunked original prompts with int32
cache tables/int64 copied indices, target/commit/cache mismatch stops, durable
partial-work counters, unchanged measured-block limits and wrapper restoration.

The incomplete-run counter now includes remaining reconstruction rounds inside
an attempted arm with a known count: 6 planned, 1 executed, 0 unknown means
5 unattempted. Missing rows, missing counters and `None` remain unknown; they
are not converted to zero executed or known remaining work. Regression coverage
also checks a known-zero activation failure and completed-run zero remainder.
This accounting repair changes no reconstruction or scoring behavior.

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

`6e0ad836d48f9ce9f9ad86ee442e6426cdcfad51f845b1b6690e453fdc7f94b4`

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

## Original native qualification requirements (now completed)

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
