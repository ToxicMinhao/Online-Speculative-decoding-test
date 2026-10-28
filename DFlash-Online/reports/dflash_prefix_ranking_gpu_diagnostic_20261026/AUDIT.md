# Independent audit of the lambda-zero diagnostic

## Material Passport

ARS experiment-agent, validate mode. **Main and backward-control audits passed: zero
contradictions.** This auditor recomputed aggregates from every retained
per-tensor discrepancy receipt, checked all three trial records against the
original first two response/anchor presentations, and revalidated all 16 raw
generation records and 758 native verification blocks. Source: [AUDIT.json](AUDIT.json).

The multi-GB initial/gradient/final tensors remain on A800 and were not re-read
by this local auditor. Exact tensor equality and per-coordinate errors below
therefore rely on the reviewed runner's retained per-tensor comparison receipts;
their aggregation and internal consistency were independently checked. No GPU,
remote operation or production edit was performed by this auditor. The separate
backward-only RNG control is included below and in AUDIT.json.

## Main findings

**O2 and U0 agree exactly at every recorded boundary:** initialization, both
backward snapshots, normalized pre-optimizer gradients, final draft weights,
Adam tensors, RNG and scalar metrics. O1 differs from both, and the complete
O1/O2 and O1/U0 comparison receipts are identical. Thus this diagnostic does
not associate the discrepancy uniquely with the U0 objective label.

All three trials used the same released draft tensors, optimizer configuration,
module modes, parameter layouts, Torch/NumPy RNG tensors, first two responses
and sampled anchors. **Python RNG state differs for O1 versus O2/U0 already
after construction**; 625 Python-state metadata entries differ. No other initial
metadata differences are recorded. The initial state is therefore not identical
in every respect. Python RNG has not been established as a numerical cause.

Each trial executed two response presentations and one optimizer step. No active
ranking or reference work occurred. The retained scalar losses, accuracies,
exposures and reported gradient norm agree with the original ordinary first
window. The second presentation's exposure is already accumulated across both
responses; it must not be added to the first presentation again.

## Draft weights, separated from Adam and RNG

| Pair | Unequal weight tensors | Unequal coordinates | Coordinate fraction | Maximum absolute difference | Relative weight L2 |
|---|---:|---:|---:|---:|---:|
| O1 / O2 | 31 / 58 | 911,043 / 1,048,626,432 | 0.08687965% | 0.0009765625 | 4.45465155e-5 |
| O1 / U0 | 31 / 58 | 911,043 / 1,048,626,432 | 0.08687965% | 0.0009765625 | 4.45465155e-5 |
| O2 / U0 | 0 / 58 | 0 / 1,048,626,432 | 0% | 0 | 0 |

For the first two pairs, absolute weight-difference L2 is
0.1856302288603 and O1 weight L2 is 4167.109970445. Relative L2 is
`sqrt(sum(per_tensor_difference_l2**2)) / sqrt(sum(per_tensor_O1_l2**2))`.
RNG bytes and optimizer tensors are excluded. No tolerance is introduced;
nonzero differences remain failures of exact equality.

The first recorded tensor discrepancy is already present after the first
backward, before any Adam update:

| O1 versus O2 or U0 | Unequal gradient tensors | Unequal coordinates | Maximum absolute difference | Gradient relative L2 |
|---|---:|---:|---:|---:|
| After first backward | 50 / 58 | 211,720,464 (20.1903%) | 0.0001220703125 | 0.00257033 |
| After second backward | 50 / 58 | 127,623,326 (12.1705%) | 0.0009765625 | 0.00250352 |
| Normalized, before clipping/Adam | 50 / 58 | 127,623,326 (12.1705%) | 0.0009765625 | 0.00250352 |

The small changed-weight fraction must not be described as the fraction of
changed gradient coordinates. Final Adam first moments differ in 50 tensors
and 117,616,909 coordinates; second moments differ in 50 tensors and 127,606,284
coordinates. Adam step counters agree. Torch CPU/CUDA and NumPy RNG tensors
agree; the Python-state metadata difference persists. Full figures are retained
in AUDIT.json.

## Serving output check

All 16 generations match the saved canonical tokens and termination, pass
publication and native-weight checks, and reconstruct correctly from native
verification blocks. The independently checked call vectors are:

| Request | F | O1 | O2 | U0 |
|---|---:|---:|---:|---:|
| eval_00006 | 41 | 39 | 39 | 39 |
| eval_00027 | 60 | 59 | 59 | 59 |
| eval_00039 | 25 | 26 | 26 | 26 |
| eval_00069 | 68 | 64 | 64 | 64 |

F reproduces the saved baseline counts. The distinct one-step updated weights
produce the same verified outputs and call counts on these four requests.
This is a bounded output-impact check, not evidence that the numerical
differences are harmless over 128 steps or other prompts. No timing-speedup
claim is made from the grouped, single-repeat evaluation.

## Additional exact-RNG backward control

The separate control executed three fresh forward/backward graphs in one
ordinary O runtime on the retained first response. Before each pass it cleared
gradients, recreated TrainerCore's accumulation counters and restored the same
Python/NumPy/Torch CPU/CUDA RNG state. An explicit guard prohibited optimizer
steps. BF16, FlexAttention and all numerical settings were unchanged.

All three pairwise comparisons are exact across **58 gradient tensors and
1,048,626,432 gradient coordinates**, plus RNG and metadata. The recorded
responses, anchors and scalar metrics match the original first presentation.
Weights and Adam state remained unchanged; the final scientific state was
restored exactly. The control executed **three additional forwards/backwards,
zero optimizer steps and zero generations**, taking **115.460 seconds** including
loading, persistence and comparison. Its one CPU integration fixture passed;
all five sealed follow-up source files match their retained contents.

This establishes repeatability for those three passes under exact restored RNG
within one runtime. It does not establish that Python RNG caused O1's earlier
difference: runtime reuse, graph/compiler state and RNG control were not varied
independently. It also does not prove the absence of variability in every kernel
or identify a particular kernel. A controlled prewarm followed by complete
scientific-state restoration is a possible future qualification procedure to
review, not an implemented fix or permission to relax the exact gate.

## Interpretation limits and validity checks

- Equal scalar forward losses do not prove coordinatewise equality of hidden
  activations or logits. Those forward tensors were not retained, so the first
  gradient mismatch does not by itself distinguish forward numerical variation
  from backward/accumulation variation.
- O1 ran first; O2/U0 could reuse process-wide compiled kernels. The initial
  Python RNG state also differed. Neither association identifies the cause.
  A specific kernel has not been isolated.
- Gradient persistence adds CPU copies and synchronization. This instrumented
  execution is not timing-identical to the earlier failed qualification.
- One O/O repeat and one U0 comparison support only this window. The original
  failed gate remains failed; this audit does not authorize a main campaign or
  introduce a numerical tolerance.
- All six sealed diagnostic source files match their retrieved contents. The
  work log and result report six training forwards/backwards, three temporary
  optimizer steps, 30 draft-layer forwards including recomputation, and 1,176
  vocabulary-head forwards including recomputation. Main trial processing took
  624.739 seconds including snapshot/export/comparison overhead; evaluation took
  67.089 seconds. These are diagnostic costs, not steady-state speed estimates.
- Including the separate control, total measured work is nine training
  forwards/backwards, three temporary optimizer steps and 16 generations;
  there are no active-ranking/reference forwards or main campaign steps.

ARS fallacy scan: **11/11 checked**. No significance tests or population effects
are asserted. Request-level vectors prevent aggregate-only interpretation;
all three trial pairs and all attempted generations are retained. The four
development requests do not support generalization. Fixed settings and the
original failure remain explicit. Causal attribution to Python RNG, compilation,
FlexAttention or a single kernel is withheld. Detailed applicability notes are
recorded in AUDIT.json.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_gpu_diagnostic_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
