# FSG retained-gradient diagnosis: BF16 head batching

**Conclusion:** the failed auxiliary-gradient comparison is explained by batch-shape-dependent BF16 head arithmetic on this retained fixture. No incorrect FSG formula, prefix mask, normalization, indexing or detachment has been demonstrated. Checkpointing is not required for the measured discrepancy in this fixture, and repeating the compiled draft backward with an identical incoming gradient is exact. This is a numerical execution limitation under the existing acceptance gate—not evidence that the proposed mechanism helps or fails scientifically.

The original qualification **remains failed**. No tolerance, algorithm, precision policy or production updater was changed. No additional Adam step or pilot was run.

Scientific diagnostic source: [`41db7e31c3c80ee3220e5d919be327039caf2fb9`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/41db7e31c3c80ee3220e5d919be327039caf2fb9), [PR #33](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/33). Original failed qualification source: `cdb4989026f6fa4e818ff70ae993945b008bbd76`. [Compact measurements](GRADIENT_DIAGNOSTIC_SUMMARY.json). This report uses ARS experiment-agent execution/reproducibility validation and same-team source, numerical and accounting reviews; it is not an independent GPU replication by another team.

## What was compared

The runner loaded the immutable GSM8K KPLUS4 origin and the **already saved** first ordinary-Adam endpoint. It reused the first response's original token/teacher-feature packet and first two original anchors, with full conditioning, 16-position blocks, four protected occurrences and identical weights normalized over this two-anchor fixture, as in the failed small oracle. No response or anchor was regenerated. Prefix masks, proposals and normalized weights matched exactly between head layouts.

The head was evaluated three ways on identical hidden tensors: checkpointed one-block chunks, ordinary one-block chunks, and one unchunked two-block projection. These correspond to 16 versus 32 token rows in the native BF16 projection. A conditional second comparison supplied the **same detached incoming gradient** to every head layout. The resulting hidden gradients were then propagated through one shared draft graph, followed by a repeat with the identical incoming hidden gradient.

## Where the mismatch begins

| Controlled comparison | Observation | Supported interpretation |
|---|---|---|
| Chunked versus unchunked head forward | 7,002 / 4,861,952 pre-state logits differ; maximum difference 0.0625; relative L2 0.012749% | The first measured difference occurs in the native head projection, before the draft backward. |
| Chunked versus unchunked head backward, actual branch gradients | 7,740 / 131,072 hidden-gradient coordinates differ; maximum difference 0.0001220703125; relative L2 0.300651% | The different projection layouts also produce different hidden gradients. |
| Same comparison with **identical incoming gradient** | 6,044 / 131,072 hidden-gradient coordinates still differ; relative L2 0.300651% | Head backward batching is sufficient to produce the material hidden-gradient discrepancy. The tiny difference in the forward-derived incoming gradient is not required. |
| Checkpointed versus ordinary chunks | All recorded logits, incoming gradients, hidden gradients, proxy and energy are exactly equal, in both controls | Checkpoint recomputation does not explain this fixture's discrepancy. |
| Identical hidden gradient, repeated draft backward on the same graph | All 58 parameter gradients match exactly | No retained-graph or compiled-backward nondeterminism was observed in this repeat. |

The head's incoming FP32 gradient differed by only `8.6634e-9` maximum, while fixing it exactly left essentially the same hidden-gradient discrepancy. Thus the dominant measured discrepancy is associated with the BF16 head backward's batch shape. This does **not** identify a particular CUDA kernel defect, or establish which numerical path is closer to exact real arithmetic. No precision sweep or alternative attention backend was introduced.

## The original gate is still not satisfied

| Parameter-gradient measurement | Original qualification | Same-graph diagnostic |
|---|---:|---:|
| Coordinates outside `1e-5 + 0.016 × abs(reference)` | 356,523 | 356,522 |
| Total parameter coordinates | 1,048,626,432 | 1,048,626,432 |
| Affected parameter tensors | 55 / 58 | 55 / 58 |
| Global relative L2 discrepancy | 0.6396848% | 0.6396851% |
| Maximum absolute coordinate difference | 0.0078125 | 0.0078125 |

The failure is reproduced to essentially the same scale, **not as an exact reproduction of every original gradient coordinate**. The changed diagnostic decomposition and same-graph execution differ from the original separate-forward oracle; the one-coordinate count difference remains disclosed (only `fc.weight`: 119,865 to 119,864 violations). The global norm passes its separate 1.6% ceiling, but the coordinate criterion fails and cannot be waived by that aggregate result. Hidden-gradient errors can remain below the local coordinate gate yet propagate into parameter-coordinate failures.

Because head-local differences reproduced the parameter gate failure and the fixed-adjoint draft repeat was exact, the conditional fresh-graph check was not needed or executed. This does not claim that all compiled-backward behavior has been exhaustively tested.

## Code findings and repairs

The earlier optimizer-state alias was a real implementation bug and was already fixed in `cdb4989`; it is separate from this numerical gate. Current inspection and the native diagnostic found no further demonstrated **production** bug warranting an algorithm-preserving patch. We did not make the oracle pass by using a common gradient, changing precision, loosening tolerances or deleting failing coordinates.

The new diagnostic code adds durable intermediate evidence, fixed-input head controls, no-Adam tripwires, exact restoration and focused CPU tests. Its prelaunch review corrected diagnostic-only issues in margin reporting, fresh-graph runtime restoration and failure-preserving cleanup. A post-run reporting fix corrects stale descriptive labels inherited from the runtime-pin utility; the original raw receipt remains preserved. The effective setting was always `donated_buffer=False`. No second GPU run was needed for that label-only fix.

## Work, costs and integrity

| New diagnostic work | Executed / ceiling |
|---|---:|
| Response-level draft forwards | 2 / 3 |
| Head-only backward evaluations | 6 / 6 |
| Complete-draft parameter VJPs | 3 / 4 |
| Adam applications | **0 / 0** |
| New feedback / teacher preparations / serving generations | **0 / 0 / 0** |

There were 31 physical head projections: three reference scans plus 28 diagnostic projections, including checkpoint recomputation. Four expected checkpoint early exits reconcile 14 proxy invocations as ten completed plus four partial recomputations; they are not failed scientific operations. These mixed backward counts are not nine full training backwards. No warmup backward or optimizer step was added. Previous qualification work remains charged separately and is not erased by this zero-Adam result.

The supervisor ran for **92 seconds** (22:03:52–22:05:24 UTC, November 8), exit 0. The runner's 74.713-second clock excludes its final dependency-hash postcheck; the supervisor includes the complete process. Peak allocated/reserved GPU memory was **15.593 / 16.039 GB**. This small fixture does not qualify maximum-anchor memory.

Exact restoration passed for model values, Adam state, counters, RNG, runtime and ordinary gradients. Optimizer tensor identities/versions remained unchanged. All **1,511** admission and postflight identity checks matched. All **18 retained files / 1,918,947 bytes** were copied to Fenrir and independently rehashed without differences. Existing inputs, endpoints and original failure records remain intact; no diagnostic checkpoint was published. Both GPUs were idle afterward.

Validation: 19 focused CPU tests passed before execution locally and on A800, plus 43 existing functional-geometry regressions locally. The descriptive-label fix adds one focused test: **20 focused tests now pass**. Same-team accounting review passed 3,114 checks. The 11 ARS interpretation categories were checked: no population efficacy tests or causal generalization are made; selected-case scope, original failure, unexecuted branches and partial numerical reproduction remain explicit.

## Decision

The diagnostic identifies a numerical source of the failed check; it does not supply a justified production-code repair that makes the unchanged gate pass. Keep qualification and the pilot paused. Any proposed change to auxiliary numerical realization or qualification acceptance needs an explicit review; this report makes neither change and authorizes no further experiment.

Private evidence: `/home/minhao2/EAGLE_SPEC/research/dflash_functional_step_geometry_20261108/diagnostic_001/retained/`, including `HEAD.json`, `PARAMETERS.json`, `REFERENCE.json`, `RESULT.json`, `WORK.json`, source/input admission, runtime receipt and process logs. Native source: `experiments/functional_step_geometry/diagnose_gradients.py`, `online_feedback/fsg_gradient_diagnostic.py`; the existing production proxy remains `online_feedback/functional_geometry.py`.
