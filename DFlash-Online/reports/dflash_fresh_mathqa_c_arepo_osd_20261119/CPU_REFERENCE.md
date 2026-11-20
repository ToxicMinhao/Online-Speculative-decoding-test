# Inactive direct support-plus-score CPU reference

## Material Passport

- Mode: experiment-agent / run; bounded CPU implementation verification.
- Authority: approved fresh MathQA handoff, section 10.
- Status: **16 tests pass**, including 100 fixed synthetic oracle cases; no skips or failures.
- Scope: a matched adapted reference, **not an official D-PACE/Spec-AUF reproduction**.
- Research position: Stage 1 / project None unchanged. No GPU training, trajectory, fork, parameter sweep or model generation.
- Evidence: [machine receipt](CPU_RESULTS.json), [test log](CPU_TEST_LOG.txt), [operation contract](OPERATION_CONTRACT.json).

## Operation contract

The implementation receives the existing native forward-KL row losses, current
argmax proposals, canonical labels, native beta weights and the same native
BF16 log-softmax already used by the row loss. It performs no model forward or
vocabulary projection. The module has no installer and is not imported by the
endpoint runner or any production model path. It has no mutable response state.

Each input is one response, represented by complete `[block, slot]` arrays.
Chunks may contain different numbers of blocks or different padded widths;
they must not split a block across its slot dimension. A batch/response dimension
is rejected to prevent accidental pooling of multiple responses.

Native validity excludes the known seed at slot zero, stops at the first gap
in positive beta weights and includes canonical EOS but no later position. EOS
in the seed leaves no valid continuation. Support S is each valid matching
prefix plus its first mismatch, or the entire valid block when it has no
mismatch. A mismatch on canonical EOS remains in S. The production mask
functions are imported directly; the scalar oracle implements them separately.

For valid positions, the detached score is

\[
q_{bj}=\exp(\operatorname{FP32}(\log p_{bj}(y_{bj}))),\quad
r_{bj}=\sum_{k=j}^{m_b}\prod_{t=1}^{k}(0.5+0.5q_{bt}).
\]

The score horizon is the entire valid block, including valid positions after
the first error. Invalid factors are one inside the cumulative product, and
**invalid products contribute zero before the reverse sum**. Padding therefore
cannot increase a valid score. The smoothing constant remains 0.5.

For one response, the reference reduces three additive scalars across chunks:

\[
B=\sum_S\beta_{bj},\quad R=\sum_S r_{bj},\quad T=\sum_S r_{bj}\ell_{bj}.
\]

The detached direct coefficient is `B*r/R` on S, zero elsewhere. Loss is
`(B/R)*T/(B+1e-6)`, algebraically the specified coefficient-weighted KL sum.
There is no extra beta multiplier inside R or the coefficient. Native beta
sets total mass; r distributes that mass. Only T carries the native loss graph.
No epsilon is added to positive R. For empty support, the scale and loss are
graph-connected zero and the native `B+1e-6` denominator remains in place.

This is one response-wide pool, with no successor-based error redistribution,
prefix-only subpool or bounded interpolation. Multiple training responses
would still be normalized separately and averaged by the unchanged surrounding
recipe. Tests reject replacing that average with one cross-response pool.

Zero direct suffix coefficients mean zero derivatives with respect to the
suffix loss rows and independent suffix logits. Full-valid-block suffix
probabilities can still affect earlier *detached coefficient values*. Native
shared parameters and attention/context dependencies can also carry gradients
through supported outputs. A small synthetic shared-context example explicitly
demonstrates the latter distinction; it is not a native attention reproduction.

## Exact-source review and provenance

Read-only source checkout: PR54 report commit
`01982251350e015ae69014e04effe47b436aa69e`; scientific execution was
`b4459820b3822c3d760bb3fd880e4d139fa2bbc2`. The five inspected source files have
no diff between those commits. Their exact SHA-256 values and Git path prefix
are retained in [CPU_RESULTS.json](CPU_RESULTS.json); absolute host paths are
excluded from public receipts.

| Source | Reviewed behavior |
|---|---|
| `online_feedback/first_error.py`, lines 12–37 | Contiguous validity, canonical EOS, seed exclusion, matching prefix plus first error |
| `online_feedback/objective.py`, lines 147–164 | Detached native BF16 teacher softmax and forward-KL row arithmetic |
| `online_feedback/objective.py`, lines 293–304 | Native validity/response weights and optional position decay; weights are consumed unchanged |
| `online_feedback/objective.py`, lines 535–547 | One native epsilon and normalization per response |
| `online_feedback/prefix_allocation.py`, lines 14, 47–62, 125–134 | Smoothing 0.5, full-valid-block score, detached canonical probability precision |
| `training/specforge/algorithms/common/dflash_family_model.py`, lines 418–447 | D-PACE score uses invalid multiplicative no-ops and zero invalid suffix-sum terms |
| `online_feedback/kplus.py`, lines 17–34, 70–96 | Existing successor-based error allocation and response pooling, deliberately absent from this direct reference |

The earlier bounded-allocation implementation report was also read to establish
the source's operation and qualification boundaries. No historical efficacy
result is reused as evidence for this new reference. The exact production
score routine is executed in one compatibility test, capturing its score
before its separate bounded allocator. That process-local test patch is
restored immediately; source files and production state are unchanged.

## CPU results and limits

The final run completed 16 tests in 0.456 seconds (suite time) using Python
3.12.15, Torch 2.13.0+cu130 and one CPU thread. CUDA devices were hidden and
CUDA remained uninitialized. The default Python probe initially lacked Torch;
the existing qualified virtual environment was used without installing anything.
An earlier 15-test run passed before adding the shared-context check and changing
the score compatibility check to call actual production source.
The added execution-commit byte check initially used a repository-root path
from its `DFlash-Online` subdirectory and stopped before tests; it now resolves
the actual Git prefix. These development invocations are retained in
[CPU_DEVELOPMENT_NOTES.json](CPU_DEVELOPMENT_NOTES.json).

Coverage includes masks and first error, canonical EOS and EOS seed, empty
support, unequal valid lengths and beta values, 100 fixed randomized scalar
oracle cases, padding with arbitrary invalid content, full-valid suffix score
influence, exact production score compatibility, coefficient detachment,
conserved mass, response-wide chunk reduction, unequal chunk widths, native
BF16 KL rows, finite gradients and zero direct suffix supervision. The separate
scalar oracle uses Python lists/loops and imports neither Torch nor the
implementation. It accepts the native loss rows as numbers and therefore does
not independently validate the upstream KL implementation.

The oracle and reference were written by the same implementation agent. This
is implementation independence, not blind reviewer or independent GPU
replication. Numerical comparisons use FP32-appropriate tolerances; arbitrary
chunk regrouping is not claimed to be bitwise equal. Model integration,
checkpoint recomputation, optimizer behavior, GPU arithmetic, resource fit,
training effectiveness and serving performance remain untested for this
reference. There is no GPU activation path in the delivered module.

No unresolved formula ambiguity remains. A later whole-recipe comparison would
jointly change repair allocation, pool partition and prefix treatment, so it
would not isolate the cap's causal effect. Further method transplantation or
GPU training requires separate scope; this CPU preparation does not waive the
author-reference reproduction rule.

## Reproduce

Use the existing CPU-capable Torch environment and an authenticated DFlash
checkout containing the source files above:

```text
python experiments/fresh_mathqa_endpoints/cpu_reference/verify_cpu_reference.py
```

Run from the DFlash-Online repository directory. The runner hides CUDA, runs
only synthetic CPU tests, and writes the contract, source-bound receipt and
test log in this report directory (or `--output`). An alternate authenticated
checkout can be passed using `--source-root`.
The receipt includes the CPU package's Python file hashes and confirms that the inspected
production source files remain unchanged.
