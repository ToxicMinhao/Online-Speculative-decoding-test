# U/P ranking implementation — ready for review

## Material Passport

ARS experiment-agent, bounded implementation and CPU correctness checks. October 25, 2026. Status: **implemented in A800 DFlash-Online; CPU tests passed**. No GPU qualification, released-model training, response generation, new event or performance evaluation was run. Stage 1 / project None is unchanged. An independent same-model-family agent reviewed the source; this is not external or native GPU qualification.

## Agreement with the previous proposal

**Yes: this is the same U/P algorithm reviewed in the earlier readiness handoff.** Ordinary forward KL remains on every supervised position. P adds exact-current-rival softplus on the event reference's consecutive matching prefix; U distributes the same weighted coefficient mass across the whole response presentation. Lambda defaults to 1; lambda 0 is a compatibility path. The change is authorization and data reuse: implementation/CPU tests are now permitted, and future U/P runs reuse the completed O event's retained 128 feedback responses and 256 presentation/anchor entries instead of collecting new feedback.

The reference is an independent copy of the event-start draft. It is frozen and fixed for the event; only the target-owned frozen head/embedding are shared. Reference work runs first under no-grad with preserved RNG, projects logits in chunks while retaining full-block attention, and releases reference activations before the current autograd forward. The current logits provide the full-vocabulary strongest noncanonical rival. The discrete index is detached; gathered canonical/rival logits remain differentiable. Projection and KL retain qualified BF16 arithmetic; margin subtraction and stable softplus use FP32.

The per-response denominator stays sum(w)+1e-6, with the existing response averaging, accumulation, clipping and BF16 AdamW. U uses a single rho across the complete response presentation, without the denominator epsilon. Seed, invalid positions, post-EOS slots and matches after an earlier mismatch are unselected. A valid matching EOS is selected. Empty supervision retains the existing explicit failure rather than resampling.

This auxiliary is always active on its selected positions: it is **not a hard preservation constraint or a guarantee against regressions**. Equal coefficient mass is not equal loss, gradients, displacement or compute. These masks describe sampled training blocks, not automatically native verification acceptance.

## Code and review artifacts

Production root: `/data/sources/minhao/DFlash-Online`.

| File | Change |
|---|---|
| `online_feedback/ranking.py` | Prefix masks, whole-response U/P coefficients, differentiable exact-rival softplus, fixed-anchor forwarding |
| `online_feedback/objective.py` | New osd_uniform/osd_prefix objectives, reference-first execution, unchanged KL plus auxiliary, diagnostic metrics |
| `online_feedback/presentation_schedule.py` | Read retained O receipt; validate exact feature paths/metadata, response order and sampled anchors |
| `online_feedback/training.py` | New objective contracts, frozen-target EOS metadata, OSD strategy/recipe routing, optional retained schedule guard |
| Four focused test modules | CPU numerical, wrapper, lifecycle, schedule and trainer integration checks |

Only these four production files and four tests were installed. The unchanged author inference/training trees, ordinary O artifacts, model/environment settings and Fenrir DFlash-Research were preserved. There is no automatic U/P campaign launcher. Existing O configuration and objective contract remain unchanged. Historical source seals are retained as historical evidence; any future U/P qualification must bind the new source explicitly rather than alter old seals.

- [Implementation diff](IMPLEMENTATION.patch)
- [Source changes](SOURCE_CHANGES.json)
- [Reading order, breakpoints and future API](USE_AND_REVIEW.md)
- [Handoff comparison and proposed budget](HANDOFF_AUDIT.md)
- [Independent code review](INDEPENDENT_CODE_REVIEW.md)
- [Installation receipt](INSTALL.json)

## CPU results

**369 tests passed locally and on A800; zero failures, errors or skips on the final runs. CUDA was not initialized.** The suite includes the 337 pre-existing tests plus 32 new tests. Tests used small synthetic CPU fixtures, not the released full-size model or the real feedback tensors for training.

Covered checks include:

- Prefix/seed/EOS/invalid/cap boundaries and no reopened prefix after a mismatch.
- Exact full-vocabulary rival selection, ties, gradients into both selected logits and FP32 softplus after BF16 values.
- U/P coefficient-weight equality, whole-response rho, direct versus chunked losses/gradients and original normalization.
- Independent immutable reference parameters/buffers, current-mode/RNG isolation, a single ordinary anchor draw, reference activation release before the trainable forward, and no reference recomputation in backward.
- Correct complete-block conditioning, including a negative test for target features at/after the anchor and a positive control for earlier features.
- Lambda-zero bitwise equality to ordinary O in tiny-model loss/gradients, final weights, Adam state, optimizer counters, RNG and sampled exposure.
- Both nonzero arms through the actual tiny DFlash/TrainerCore path; teacher requirements, separate resume contracts and pre-forward rejection of altered ordering/anchors.
- Retained-response path/metadata binding before reference creation/training; all 256 actual retained O schedule entries accepted by the CPU schedule reader.

[Final local log](CPU_ALL_LOCAL.txt), [A800 log](CPU_ALL_A800.txt), [machine-readable A800 result](CPU_TESTS.json).

The first broad local test attempt imported Fenrir's pristine inference copy instead of DFlash-Online's qualified cache-compatible copy, producing 6 failures and 12 errors in existing cache/replay tests. That log is preserved at [CPU_ALL_WRONG_INFERENCE_PATH.txt](CPU_ALL_WRONG_INFERENCE_PATH.txt). Re-running with the actual unchanged DFlash-Online inference dependency resolved them; no production inference code or test tolerance was changed.

New ranking metrics inherit the existing accumulation convention: optimizer-boundary rows contain both responses' accumulated sums. Do not sum intermediate rows again when reporting exposure or coefficient mass. A lower training loss or passing CPU tests is not evidence of improved drafting.

## Proposed remaining GPU qualification — not run

Use the first two presentations from the **full retained 128-response pass**, preserving its original 128-element shuffle and subsequent anchor RNG draws. Do not substitute a newly shuffled two-response pool. Each disposable trial starts at the same released weights with a fresh optimizer and seed 17, stops after one complete accumulation window, and is discarded. Keep the original numerical tolerances.

| Work | Proposed bound |
|---|---:|
| Trials: O, U(lambda0), P(lambda0), U(lambda1), P(lambda1) | 5 |
| Current-model forward/backward presentations | 10 |
| Temporary optimizer steps | **5** |
| Reference forwards inside U1/P1 trials | 4 |
| Extra detached matched-mode reference/current comparisons | 4 forwards |
| New feedback/evaluation generations | **0** |
| New teacher reconstruction | **0 expected**, reuse retained prepared features |

These are 18 top-level draft forwards plus normal checkpointed backward recomputation. Measure full-size peak memory, reference setup/forward/projection and total update cost. Test native Flex/BF16 behavior, reference immutability, sampler/RNG continuity, exact lambda-zero optimizer equivalence and release between trials. No candidate checkpoint becomes a serving checkpoint. CPU success does not establish these GPU properties.

A later one-event comparison, if authorized, would add 256 main optimizer steps / 512 presentations across U and P, with no feedback recollection or O retraining. Measuring F/O/U/P freshly together on the existing panel would require 768 evaluation generations; the first eight requests' 96 generations could supply publication/output qualification within that budget. Historical F/O times must not be pooled with new candidate timings. Those executions remain unapproved and unstarted.

## Retention and stopping point

The exact original 128 prepared feature paths, O receipt, exported checkpoint and full optimizer/RNG state remain present. Source installation verified their sizes and modification times were unchanged. Source backups and transfer bindings are retained. The source/CPU-test allowance is 256 MiB above the existing 8 GiB reserve; A800 has about 403.31 GB free at this session's admission. No artifacts were deleted or offloaded. Fenrir retains about 83 GB free.

**Stop here for review.** The new algorithm is implemented and CPU-tested; GPU correctness, full-size memory and any performance benefit remain untested.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_implementation_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
