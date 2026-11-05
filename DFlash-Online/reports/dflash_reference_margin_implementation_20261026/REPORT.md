# R: reference-margin preservation — implementation review

R is implemented as `osd_reference_margin` in A800's existing DFlash-Online checkout. **423 CPU tests pass on Fenrir and 423 on A800, with no skips, failures or errors.** Both test processes assert CUDA remains uninitialized. No GPU qualification, scientific training/generation, baseline retraining, or feedback collection was performed.

## Implemented objective

The truncated formula in the request is interpreted consistently with its surrounding specification:

\[
m_i^c=z_{i,y_i}^c-\max_{v\ne y_i}z_{i,v}^c,\qquad
\ell_i^R=\operatorname{ReLU}\!\left[
\operatorname{softplus}(-m_i^c)
-\operatorname{stopgrad}(\operatorname{softplus}(-m_i^{ref}))\right],
\]
\[
L_R=\frac{\sum_i w_i\left[\mathrm{KL}(p_i\|q_i)+\lambda g_i\ell_i^R\right]}
{\sum_i w_i+10^{-6}},\qquad\lambda=1.
\]

Each model chooses its **own** strongest noncanonical rival over the full vocabulary. Only the rival index and reference values are detached; both gathered current logits remain differentiable. Native BF16 projection values feed FP32 margin subtraction and stable softplus. `torch.relu` applies separately at every position before weighting/reduction and gives zero derivative at equality.

On a protected position, matching or exceeding the reference margin gives zero auxiliary loss and gradient. A worsened margin activates the penalty; its current-logit gradient then equals original P's auxiliary gradient at that position. Improvements at other positions cannot cancel a penalty. **Zero auxiliary pressure does not stop ordinary KL learning.** This is a soft penalty, not a hard guarantee that every protected prediction will remain correct after an optimizer step.

## Scope and code

Only three production files changed: 97 added lines and 22 removed lines, plus three new test files. No inference, feedback-capture, author training backbone, optimizer, scheduling or evaluation-driver implementation changed.

| File | Change |
| --- | --- |
| `online_feedback/ranking.py` | R identity, P-equivalent prefix allocation, small pointwise reference-margin helper. |
| `online_feedback/objective.py` | Reuse reference projection outputs to retain detached scalar softplus values and correctness flags; apply R in the existing chunked loss; return additive diagnostic numerators/counts. |
| `online_feedback/training.py` | Recognize R as an OSD-derived objective and give its checkpoint a distinct mechanism identity. Existing O/U/P contracts remain unchanged. |

Live source: `/data/sources/minhao/DFlash-Online/online_feedback/` on A800.
Review mirror: `/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/`.

Select R through the existing `SpecForgeEventTrainer(..., objective="osd_reference_margin", ranking_lambda=1.0)` interface. This adds no separate trainer. A future campaign must use its own output directory and the retained ordinary presentation/anchor schedule; the existing U/P campaign driver was not changed or launched.

[Implementation diff](IMPLEMENTATION.diff), [changed-file receipt](SOURCE_CHANGES.json), [integration review](review/CODE_REVIEW.md).

## Unchanged learning contracts

- Ordinary teacher forward KL, BF16 probability arithmetic, position weights, response denominator/epsilon, response averaging, gradient accumulation, optimizer, learning rate and anchor schedule are unchanged.
- R uses exactly P's reference-prefix mask: exclude the known seed, invalid/masked positions, and matches after the first mismatch. Include a correctly matched EOS, then stop the prefix.
- The independent reference draft is fixed for the event, in evaluation mode without gradients. Its existing full-block forward and chunked head projections supply both the mask and new reference scalars. No extra model forward is introduced. Reference hidden activations and vocabulary logits are released before current-model autograd.
- Reference setup/measurement preserves RNG and current module modes. The reference stays outside registered trainable state. Teacher feedback remains detached.
- With `lambda=0`, R bypasses reference/ranking work and follows ordinary O's computation and metrics. Reference-dependent diagnostics are consequently absent in that compatibility mode.
- Resume rejects a different objective, lambda or EOS contract before loading weights/optimizer/RNG. Same-R continuation preserves state and the subsequent fixture event exactly.

## Lightweight logging

All new measurements come from existing reference/current projections. Chunk outputs return additive statistics instead of modifying counters during checkpointed backward recomputation.

| Measurement | Numerator | Explicit denominator |
| --- | --- | --- |
| `protected_activation_rate` | Protected eligible occurrences with current softplus greater than reference softplus | All protected eligible occurrences |
| `reference_correct_to_incorrect_rate` | Eligible occurrences where reference argmax equals canonical token and current argmax does not | **All eligible reference-correct occurrences**, including later matches outside the consecutive protected prefix |
| `forward_kl` | `sum(w * KL)` | `sum(w) + 1e-6` per response |
| `auxiliary_loss` | `lambda * sum(w * g * ell_R)` | Same response denominator |

The rate counts are unweighted; loss components use the original position weights. Counts refer to sampled block-position occurrences, not unique response tokens. The raw counts, both component numerators and `loss_denominator` are retained in `sum_metrics`. Zero count denominators are preserved explicitly; the existing display value of zero then means no eligible observations, not evidence of zero regression.

`ranking_loss` remains the auxiliary numerator without lambda divided by the response denominator; at the prescribed lambda=1 it equals `auxiliary_loss`. Existing component metric aggregation pools numerators/denominators; optimization still uses the unchanged response mean. Intermediate trainer rows describe an individual presentation, while optimizer-boundary rows contain the complete accumulation window: **aggregate the boundary rows**, rather than summing every row and double-counting intermediate presentations.

## CPU evidence

| Evidence | Result |
| --- | --- |
| Full Fenrir suite | 423 passed, 0 skipped; 40.28 s including discovery |
| Full A800 suite, CUDA hidden | 423 passed, 0 skipped; 43.26 s including discovery; exit 0 |
| New focused coverage | 21 tests: 9 helper, 7 objective, 5 trainer integration |
| Preserved-source comparison | 41/41 checks passed across real-reference O/U/P and explicitly injected, nonempty prefix fixtures for active U/P |

New tests cover equal/improved/worsened margins, zero equality derivative, active P-gradient equivalence, changing rivals and ties, detached reference values, pointwise ReLU before unequal weighting, BF16-to-FP32 arithmetic, prefix/mask/EOS/tail boundaries, independent KL+R loss/gradient oracles, logging denominators, no extra reference forward or projection, activation lifetime, RNG/modes, chunk equivalence, schedule rejection and resume isolation.

R(lambda0) and ordinary O have exactly equal tiny-fixture final weights, Adam state, RNG, counters, sampled anchors and numerical metrics. The additional historical-source comparison verifies exact old/new O/U/P losses, metrics, 14 parameter-gradient tensors and final RNG. Its active U/P fixtures deliberately inject six protected positions to exercise nonzero auxiliary arithmetic; they are unit-test fixtures, not experiment observations.

The CPU suite uses tiny synthetic models and eager attention, including BF16 arithmetic where pinned. It does not load released full-size models, use real retained responses, or establish GPU/Flex numerical behavior, memory feasibility, native output preservation, speed or learning benefit.

[Local log](CPU_SUITE_FENRIR.txt), [local summary](CPU_SUITE_FENRIR.json), [A800 log](CPU_SUITE_A800.txt), [A800 summary](CPU_SUITE_A800.json), [historical comparison](review/OLD_NEW_COMPATIBILITY.json), [reproduction script](review/compare_old_new.py).

One initial test-observer failure was caused by a mock retaining head-input activations in its call history. The test was corrected to use a nonretaining observer; no production behavior or acceptance threshold was changed for that failure.

## Preservation and stop point

The live three pre-change production files were verified against the retained snapshot and backed up before installation under `/data/sources/minhao/runs/dflash_reference_margin_implementation_20261026/source_before/`. All 113 other retained source files still match. Existing O/U/P checkpoints, results, failure records and original source snapshots remain intact. Fenrir's clean `/home/minhao2/DFlash-Research` source was not modified; its existing debugger process was left alone.

Remote storage admitted a 512 MiB CPU/source-work allowance plus the existing 8 GiB reserve. Free space was 192.870 GB at preflight and 192.868 GB after CPU tests; no cleanup/deletion was needed. The remote host lacks `rsync`; its two failed transfer attempts wrote no source changes, and the same six-file payload was transferred using tar/SSH and scp instead. [Deployment receipt](SOURCE_DEPLOYMENT.json), [unchanged-source check](PRESERVATION_CHECK.json).

**Ready for code review only.** R has not been GPU-qualified or evaluated. The old U/P GPU receipts remain historical evidence for their original source, not qualification of this changed implementation. No additional experiment or project-direction change follows from the CPU pass.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_margin_implementation_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
