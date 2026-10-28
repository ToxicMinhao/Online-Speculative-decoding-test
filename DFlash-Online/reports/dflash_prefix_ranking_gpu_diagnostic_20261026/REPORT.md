# Lambda-zero GPU discrepancy: bounded diagnostic

## Material Passport

ARS experiment-agent run/validate. A800 execution, October 25 local / October 26
UTC. All bounded work completed; the full U/P campaign remains paused.
No production training/inference code, objective, precision, learning rate,
optimizer, sampling rule or qualification tolerance was changed.

## Decision-relevant result

**The mismatch is not specific to enabling the U/P implementation.** Repeating
ordinary OSD also produced the discrepancy. A later ordinary OSD run and U with
λ=0 matched exactly in initialization, gradients, final weights, Adam state,
RNG, anchors and scalar metrics. All three updated checkpoints produced the
same token IDs and verification counts on four fixed requests.

The evidence supports continuing toward the algorithm experiment after making
qualification compare consistently initialized executions. It does not justify
claiming the old exact gate passed, changing numerical tolerances arbitrarily,
or asserting that numerical differences can never affect a longer run.

## What was executed

O1 and O2 are two independent ordinary OSD trials. U0 is the new uniform-ranking
arm with λ=0. Each started from released DFlash, a fresh optimizer and seed 17,
used the retained event's first two presentations and exact original anchors,
and performed one temporary optimizer step. The full 128-response permutation was
still drawn; only its first two responses were processed.

Ranking is bypassed entirely at λ=0: no reference forward and no literal
ranking-plus-zero computation occurs. The ordinary BF16 forward KL, BF16 AdamW,
Flex attention, response normalization and accumulation are unchanged.

All initial states, gradient boundaries, final states and diagnostic exports were
retained before comparisons. Native replay used F/O1/O2/U0, once each, on the
first four saved development requests: 16 capture-disabled generations, full
saved inputs, greedy block size 16, 2048-token-or-EOS cap, qualified SGLang/FA3 and cold
valid caches. No training ran concurrently with serving.

## Numerical evidence

| Comparison | Initial weights / Adam / Torch RNG | Gradients and final weights | Final Adam / RNG |
|---|---|---|---|
| O1 versus O2 | Exactly equal | Differ | Moments differ; Python RNG differs |
| O1 versus U0 | Exactly equal | Same differences as O1 versus O2 | Same pattern |
| O2 versus U0 | Exactly equal, including Python RNG | **Exactly equal** | **Exactly equal** |

O1's Python RNG state differed after runtime construction; O2/U0 matched. This
is a real initialization difference. The inspected response-order and anchor
samplers use Torch RNG, and the actual retained draws match exactly. Python RNG
has not been established as the cause of the numerical discrepancy.

For O1 versus O2 (identical statistics for O1 versus U0):

- The first observed tensor difference is after the first backward, before any
  optimizer step: 50/58 gradient tensors differ; relative gradient L2 difference
  is 0.2570%. Equal scalar losses do not establish equal forward activations.
- Before AdamW, the relative gradient L2 difference is 0.2504%.
- After the step, 31/58 draft tensors contain differences:911,043 of
  1,048,626,432 weight entries (**0.08688%**).
- Maximum absolute weight difference is 0.0009765625; relative weight L2
  difference is 0.004455% (reference norm 4167.10997, difference norm 0.18563023).
- Scalar losses, accuracy, exposure, norm, order and anchors are exactly equal.
  Do not infer coordinate equality from these rounded summaries.

The numerical summaries separate draft weights from Adam moments and RNG.
Aggregating their norms together would be misleading.

## Exact-RNG backward-only control

To remove the initial Python-RNG confound, a fresh ordinary model evaluated the
same first response three times. Before each fresh forward, all Python, NumPy,
TorchCPU and CUDA RNG states were restored identically, gradients were cleared,
and accumulation bookkeeping was reset. The full original shuffle and anchor
guard remained in place. Optimizer calls were explicitly forbidden.

All three gradients and post-backward RNG states matched exactly. Weights and
Adam stayed unchanged, and the complete initial scientific state was restored
exactly afterward. This does not support a claim of random fluctuation on every
backward. The remaining difference is associated with the first versus later
model/runtime constructions in the initial three-trial process.

**The exact internal operation is unresolved.** Forward activations/logits were
not retained, so the first gradient observation cannot distinguish forward
numerics from backward/accumulation numerics. Compiler, library and reducer
initialization remain possibilities, not established causes. The source review
does not establish that the Triton Flex backward is inherently nondeterministic.
Diagnostic gradient copies also add synchronization and are disclosed here.

## Native output and drafting impact

| Saved request | Output tokens | F calls | O1 calls | O2 calls | U0 calls |
|---|---:|---:|---:|---:|---:|
| eval_00006 |267|41|39|39|39|
| eval_00027 |370|60|59|59|59|
| eval_00039 |219|25|26|26|26|
| eval_00069 |372|68|64|64|64|
| Total |1228|194|188|188|188|

All 16 token sequences and termination records exactly match their saved canonical
references. Fresh F counts reproduce retained F. Publication/native weight checks,
trace geometry and measurement checks pass. The regression on eval_00039 remains
visible. O1/O2/U0 also have identical accepted-token counts per request.

Thus **no output or verification-count effect was detected from these weight
differences on these four requests**. This is a small numerical-impact check,
not a new efficacy claim, timing benchmark or 128-step equivalence proof.

## Accounting and preservation

- Initial diagnostic: 6 training presentations/backwards,3 disposable optimizer
  steps, 30 layer forwards including recomputation, 1176 vocabulary-head forwards.
- Follow-up: 3 first-response presentations/backwards,0 optimizer steps, 15 layer
  forwards including recomputation, 552 vocabulary-head forwards.
- Native generations: 16. New feedback/teacher-model reconstruction:0. Active
  ranking/reference forwards:0. Main U/P optimizer steps:0.
- Six initial CPU tests and one follow-up CPU integration test passed locally and
  on A800. The first local test import omitted the existing tests directory;
  its failure is retained and the test-only import path was corrected.
- Initial trial/state comparison wall time 624.74s; native evaluation 67.09s,
  including 21.01s loading. Initial supervisor 736s including provenance/CPU checks.
  Backward-only control 115.46s; its supervisor 127s. Snapshot persistence and
  exhaustive CPU tensor comparisons are diagnostic costs, not normal update cost.
- Highest recorded training CUDA allocation 21,075,309,056bytes (~19.63GiB).
  Storage preflight reserved 100GiB writes plus 8GiB reserve; final free space
  is 319.41GB. Both GPUs are free. No cleanup was needed.
- All 105 production source files remain unchanged. All 132 existing O artifacts
  retain their size and modification time; full hashes passed before execution.
  Old failed-qualification evidence and its exact-gate outcome are preserved.
  Fenrir's clean DFlash-Research folder remains untouched.

## Recommended next action

Make runtime initialization/warm-up an explicit, symmetric qualification step,
discard its training state, and compare O/U0/P0 from clean released weights and
the original schedule afterward. Retain the O/O repeat and exact state checks;
do not silently loosen a threshold to obtain a pass. Then complete the already
planned active-ranking mask, gradient, reference-isolation and memory checks.

The intended 128-step U/P comparison should follow only that qualification. The
main campaign was not resumed here. These diagnostics give no efficacy result
for λ=1 and do not select a new research direction.

## Evidence

- [Numerical summary](NUMERICAL_SUMMARY.json), [per-request records](PER_REQUEST.csv)
- [All three trial comparisons and work counts](raw/main/trials/RESULT.json)
- [Native validity/accounting](raw/main/evaluation/STATUS.json)
- [Exact-RNG backward-only result](raw/followup/backward_control/RESULT.json)
- [Independent source review](review.md), [independent artifact audit](AUDIT.md)
- [Diagnostic implementation](runner.py), [output check](evaluate.py),
  [backward-only control](backward_control.py)

Complete tensor states/exports remain on A800 at
`/data/sources/minhao/runs/dflash_prefix_ranking_gpu_diagnostic_20261026`.
Compact raw records are retained alongside this report on Fenrir. No private
responses or model tensors were uploaded to GitHub.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_gpu_diagnostic_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
