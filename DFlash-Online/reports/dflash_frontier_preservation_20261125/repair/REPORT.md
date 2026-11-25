# Frontier preservation: bounded derivative repair

**Status: the execution defects are repaired, but native numerical qualification still fails.** The latest scoped derivative path preserves the native hidden output and all 58 ordinary parameter-gradient tensors exactly, and completes the previously unsupported differentiation. Its directional consistency error remains **2.047197%**, above the prospectively fixed **1.6%** criterion. We therefore stopped before the continuation event. There is no serving result, confidence interval or evidence about the candidate's learning benefit. The remaining discrepancy is consistent with BF16 forward/reverse arithmetic sensitivity, but its exact operation-level cause is not established; it must not be relabeled harmless or passed.

## Material Passport

ARS experiment-agent run/validate; bounded implementation repair and native prerequisite checks, with separate source/receipt reviews within the shared-context agent team. These reviews are not independent GPU replications. Latest execution is **`65ae6f95bd73aef8f60542a3d2ed3f5a3e51b48c`** on physical GPU B, UUID `GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1`. All 5,128 staged source files were checked before that launch. The [original failure report](../REPORT.md) and every failed attempt remain retained. This supplement supersedes their current-status conclusion only.

All checks restore the same authenticated PR59 PAIR1 recovery: step128, 256 presentations, event1 and 58 warm Adam entries. They reuse the same fixed training window. The analytic repair, localization and transpose comparisons use one fixed diagnostic direction; the original unsupported probes used their earlier diagnostic construction. No held-out input, new anchor selection based on outcomes, tolerance change, ordinary-loss change or precision change to committed training was introduced. Both frozen responses have 512/269 anchors; only the first entered model derivative checks. This is a bounded prerequisite, not qualification across all production inputs. The first-gradient comparison uses a fixed hidden-output cotangent, not an ordinary KL loss; it does not replace the remaining full-objective controls.

## What was wrong and what was repaired

1. **Missing derivative support in compiled attention.** Ordinary forward/backward works, but the pinned compiled FlexAttention path has no forward-AD rule and rejects differentiating its backward. We added a scoped analytic attention tangent, using native outputs/LSE, the exact mask and GQA mapping, and bounded tiles. Native forward and ordinary backward remain untouched outside the diagnostic scope. FP64/FP32 CPU references pass; the real first-layer Q/K/V/joint checks pass with relative errors between 0.000507% and 0.003090%. This repairs an unavailable operation; it does not replace native serving attention.
2. **An unnecessary graph-retention request in the new reverse wrapper.** The reverse route initially requested retention of a compiled backward graph whose donation policy requires single use. The wrapper needs saved primal values, not a second traversal of that graph. Consuming it once fixes the execution error. `donated_buffer=False` was requested before model construction; why the compiled graph still carried donation metadata is not established. No claim that cache contamination caused it is justified. [Failure](transpose_001/RESULT.json), [pinned donation source](DONATION_SOURCE.json).
3. **BF16 SiLU changes its first backward when asked to build a derivative graph.** After the graph fix, 55/58 gradient tensors differed from ordinary native backward. A controlled CPU fixture reproduces that difference in SiLU: 4,314/8,192 coordinates differ, maximum absolute error 0.03125, relative L2 0.47236%. The explicit native SiLU backward matches ordinary execution. A scoped cotangent-only wrapper now uses that native kernel: the real model's output and **all 58 first-gradient tensors match exactly**. This identifies and repairs this first-gradient mismatch. It does not explain away the separate final 2.047% discrepancy. [CPU fixture](SILU_CONTROL.json), [native before](transpose_002/MODEL.json), [native after](transpose_003/MODEL.json).

The custom reverse interface differentiates only with respect to the incoming output cotangent at fixed primal values. It is not a parameter Hessian or a general second-order FlexAttention implementation. Neither scoped wrapper is installed in ordinary K+PAIR training or serving. Git comparison against the PR59 parent confirms that `online_feedback/`, `training/` and `inference/` are unchanged; repairs are isolated to the new experiment package. [Source parity](ORDINARY_SOURCE_PARITY.json).

## Remaining numerical gate

For the same fixed direction `v` and output cotangent `u`, the check compares `<Jv,u>` with `<v,Jᵀu>`. Scalar products are reduced in FP64 over the actual represented values; `v` is the normalized native gradient, rounded to native dtype. The criterion was sealed before the first repair probe:

`abs(error) <= 1e-5 + 0.016 * abs(native_backward_dot)`.

| Quantity | Analytic forward route | Repaired cotangent-transpose route |
|---|---:|---:|
| Directional contraction | 21.291557381703647 | 21.291557381703647 |
| Native backward reference | 21.736547395977073 | 21.736547395977073 |
| Absolute discrepancy | 0.444990014273426 | 0.444990014273426 |
| Allowed discrepancy | 0.347794758335633 | 0.347794758335633 |
| Relative discrepancy | 2.0471973132% | 2.0471973132% |
| Native output / ordinary first gradients | Exact / exact | Exact / exact |
| Numerical qualification | **Failed** | **Failed** |

The two routes share the analytic attention formula, so agreement is useful consistency evidence, not two independent mathematical oracles. Finite-precision forward and reverse accumulation need not be exact transposes. The native backward contains rounded intermediate calculations; reproducing that backward exactly does not make it an exactly linear map of its incoming gradient. Its rounding-ignored derivative therefore need not reproduce the original backward contraction exactly. This is an explanation consistent with the data, not proof that every remaining operator is correctly implemented. These checks do not differentiate discrete rounding or establish the true derivative of a quantized update. The handoff already distinguishes its mixed-precision shadow from such a derivative; nevertheless, this implementation has not met its declared numerical prerequisite.

A fixed-input localization reconstructs the global residual to 1.44e-15. The five attention residuals sum **+0.00231709**, whereas the total residual is **−0.44499001**. The shared-context merge residual is **+0.00003822**, subtracted in reconstructing the total residual. Most discrepancy is outside the analytic attention operation, spread across the decoder regions; the final normalization contributes −0.01329230. The attention rows are nested within decoder rows and must not be added again. All observed primals and all five local-versus-global parameter-gradient sets match exactly. This rules out the attention adapter or shared-context merge as the dominant measured source, but does not isolate a particular remaining linear, normalization or reduction operation. [Localization](localization_001/LOCALIZATION.json).

We found no demonstrated loss/mask/denominator error in this prerequisite, which does not yet invoke the vocabulary head or loss. Calling it a proven harmless kernel issue would exceed the evidence. Further local operator diagnostics or a revised numerical contract would be a new bounded decision; we did not widen the limit, change derivative precision, replace the score or continue into training.

## Attempts, work and cost

| Attempt / execution commit | Outcome | Occupied card-seconds |
|---|---|---:|
| Original / `da4f4cf` | Missing forward-AD and double-backward support | 40.939 |
| Analytic repair / `534f02d` | Native identity and attention checks pass; full 2.047% gate fails | 51.192 |
| Localization / `f8f8c4d` | Same failure reproduced; residual decomposition completes | 46.193 |
| Reverse route / `0242ccc` | Stops at unnecessary graph retention | 41.252 |
| Single-use graph / `4540d6a` | Attention executes; 55/58 first gradients differ | 41.297 |
| Native SiLU / `65ae6f9` | All first gradients exact; full 2.047% gate still fails | 46.552 |
| **Cumulative, including original failure** | **No production** | **267.424** |

This is **4.457 occupied GPU minutes / 0.07428 card-hours**; elapsed research time includes source inspection, CPU tests, publication and staging, rather than GPU work alone. The latest process receipt records 3,465.67 seconds since the original clock started. There were **zero Adam steps, shadow optimizer previews, vocabulary-head/loss calls, new feedback responses, serving generations or new checkpoints**. Failed/incomplete calls are preserved, not counted as successful full forwards. Across the six attempts, 13 full-model forwards and 10 full-model parameter VJPs completed, with five additional local decoder forwards/VJPs. There are 55 ledger operations attempted, 52 completed and three incomplete primitives. Forward-AD/tangent work and the successful final cotangent transpose are separately named in the ledgers; they are not extra training presentations. Per-attempt primitive ledgers and hooks are in [SUMMARY.json](SUMMARY.json). Supervisor completion and native child status are distinct: a wrapper exit of zero does not turn a failed numerical gate into a pass.

All **66 focused CPU tests pass**, including independent small mathematical references, scoped primal/first-gradient equivalence, native graph single-use behavior, SiLU cotangent differentiation, state isolation, source/resource admission and failure persistence. [Test log](CPU_TESTS.log). Passing them does not override the native failure.

Latest peak memory after parent import is **43.404GB allocated / 47.568GB reserved**. Across attempts, the largest recorded allocation is 46.102GB; initialization peaks were not independently measured because counters were reset after import. No out-of-memory failure occurred. Latest remote cumulative source/run/cache allocation is **1.988GB**; Fenrir study allocation is **0.340GB** (excluding shared Git object-store bookkeeping), about **2.328GB combined**. These are sampled post-run allocations, not continuously measured peaks. No inherited artifact was retired.

Final parent recovery, export, full input manifest and both selected feature packets are byte-identical to their authenticated origins. Every attempt passed full live-state restoration. Postflight A800 free space is **430.119GB**, preserving the 400GB reserve; both cards are idle. [Postflight](POSTFLIGHT.json).

## Completion boundary

The user authorized trying a repair before returning a blocker report. That attempt fixed concrete execution and first-gradient issues, but did not qualify the complete derivative within the bounded investigation. The full coefficient composition through the optimizer shadow, candidate activity, disabled-path equivalence, both-response replay/state integration and production continuation remain unqualified. No claim that FRONTIER_P works or fails as a learning method follows.

**Recommendation: keep the event paused and review this specific numerical mismatch.** There is no supportable full-event ETA until its derivative prerequisite passes. Preserve the fixed inputs, failed results and current native recipe; do not automatically launch another backend investigation or the main event.

Receipt/source audit: [INDEPENDENT_AUDIT.json](INDEPENDENT_AUDIT.json), covering 113 raw files / 351,958 bytes, all six archives and 69 prior receipt bindings. It verifies evidence integrity and accounting; the numerical qualification remains failed.
