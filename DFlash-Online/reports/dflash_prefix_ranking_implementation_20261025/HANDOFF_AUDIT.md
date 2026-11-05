# U/P handoff comparison and review checklist

## Material Passport

ARS experiment-agent, implementation/reproducibility review. Date: October 25, 2026. Scope: independent read-only comparison of the current user handoff with the original readiness handoff and report. No model loading, GPU qualification, training, serving generation or remote operation was performed by this audit. Stage 1 / project None remains unchanged. This file records review criteria and a proposed future budget, not successful execution.

## Comparison with the earlier request

**The mathematical mechanism is the same.** The current handoff authorizes the previously deferred implementation, while keeping GPU execution paused.

| Contract | Original readiness handoff | Current instruction |
|---|---|---|
| Base loss | Ordinary weighted forward KL at every valid supervised position | Unchanged |
| Additional loss | `softplus(z[rival] - z[canonical])`, strongest current noncanonical vocabulary rival, lambda 1 | Unchanged; lambda 0 remains the compatibility control |
| P allocation | Consecutive canonical-matching prefix of an independent, event-fixed reference | Unchanged |
| U allocation | Detached response-wide `rho = sum(w*g)/sum(w)` | Unchanged |
| Normalization | Existing per-response denominator and response averaging/accumulation | Now restates the inspected denominator explicitly: `sum(w) + 1e-6` |
| Reference execution | Sample once per presentation; same full blocks, no-grad/eval, reference first, isolated RNG, release temporaries before trainable forward | Unchanged |
| Experimental start | Matched first event; earlier report proposed released weights/fresh optimizers | Now explicitly retains that choice for U/P |
| Feedback and schedule | Earlier report proposed fresh shared 128-response capture | Reuse the already completed O event's retained 128 responses and presentation/anchor schedule |
| Authorized work | Readiness report only | Implementation and CPU correctness tests only; return for review |

“Sample anchors once” means **once for each response presentation**, shared by its reference and current forwards. The two passes retain their original, separately sampled anchor sets. It does not mean replacing the retained two-pass schedule with one anchor set per response reused in both passes. In this first event, the same released reference, feedback, weights and exact presentation schedule make U/P coefficient masses comparable across arms. Equal coefficient mass does not imply equal loss, gradient norm, optimizer displacement or computation.

The existing O run is complete and must remain immutable. Neither the new implementation nor a future U/P run should resume O's trained state: each new arm begins from released DFlash with a fresh optimizer. Evaluation responses remain outside training. The current authorization does not permit GPU qualification, a full event, serving generation, parameter retuning, another algorithm, or another feedback event.

## Implementation and CPU review checklist

1. **Loss integration:** ordinary KL remains at every supervised position. Add ranking before the existing response denominator; add the denominator epsilon once, never to rho. Preserve response-wise averaging, accumulation, clipping and optimizer behavior. No division by protected count or by `1 + lambda`.
2. **Prefix semantics:** fixtures cover seed exclusion, first-position mismatch, a later match following a mismatch, strict invalid-boundary termination, padding, cap suffixes and EOS. A valid matching EOS may be protected; positions beyond it may not. A cap boundary is neither a fabricated mismatch nor an additional supervised token.
3. **Current rival:** use exact full-vocabulary noncanonical argmax with declared tie behavior. Detach only the selected integer index. Gather both differentiable logits from the original tensor. Verify gradients lower the rival logit and raise the canonical logit when the ranking term is active. Cover ties, extreme margins and BF16 projection followed by FP32 subtraction/softplus.
4. **Response-wide U coefficient:** use all supervised weights across all loss chunks, rather than a separate rho per chunk. Check `sum(w*rho) == sum(w*g)` within a predeclared numerical tolerance, including zero protected mass and nonuniform weights. Respect existing empty-supervision handling without resampling.
5. **Full-block conditioning:** identical token IDs, anchor starts, block positions, masks and target-feature alignment for reference and current. Do not prune slots with zero auxiliary loss or change block attention. Tests should expose a wrapper accidentally resampling anchors or constructing shorter reference blocks.
6. **Independent reference:** immutable independent draft parameter/buffer storage; no optimizer/export ownership; fixed through the event. Frozen target embedding/head sharing is permissible only with verified ownership. Restore original module modes, including shared module flags, and preserve Python/NumPy/Torch RNG states around reference work. Test later anchor draws, not only an initial seed value.
7. **Memory/lifecycle:** project reference logits in the qualified chunks, retain only detached prefix/mask summaries, and finish reference work before building the current graph. Reference inference must not recur inside checkpointed current backward. Release reference state after the event.
8. **Lambda zero:** compare unchanged ordinary O against both new zero-lambda paths on CPU for the same sampled blocks, losses, gradients, actual update, optimizer moments/counters, and RNG. Preserve ordinary objective defaults and exports. CPU success does not qualify compiled Flex/BF16 GPU equivalence.
9. **Replay schedule and artifact isolation:** validate exactly 128 distinct source records, 256 presentations and 128 optimizer boundaries. Preserve retained per-presentation order/anchors and their pass identities; detect schedule mismatch rather than resample. Use new U/P destinations and retain the O export, restart state, source feedback and raw measurements.
10. **Reporting:** separate KL and auxiliary numerators, denominator, rho, protected coefficient mass and real exposure. The completed O logger includes accumulated metrics on optimizer-boundary rows; do not double-count intermediate rows when comparing exposure. No lower-loss or protection guarantee is a serving-performance result.

## Proposed remaining GPU qualification budget — not authorized or run

Use the first two retained response presentations and their exact anchors, reset to the identical released state/fresh optimizer/RNG for each disposable trial, and discard every resulting checkpoint. Preserve the existing scientific tolerances; report failures before changing the protocol.

| Work | Proposed bound | Purpose |
|---|---:|---|
| Temporary ordinary O, U(0), P(0), U(1), P(1) trials | 5 trials × 2 presentations = 10 forward/backward presentations and 5 optimizer steps | Lambda-zero equivalence plus nonzero ranking execution |
| Reference forwards inside those trials | **4** full-block reference forwards | Same two presentations for U(1) and P(1); U(0)/P(0) deliberately take the exact ordinary O fast path without a reference |
| Detached matched-mode reference/current comparison | 4 draft forwards | Two response pairs with identical weights, anchors and deterministic mode |
| New feedback or evaluation generations in this training qualification | **0** | All features/IDs/anchors already retained |
| New teacher-feature reconstruction | **0 expected** | Reuse prepared feedback; report missing artifacts instead of silently reconstructing or regenerating |

This is **18 top-level draft forwards** in total: 10 trainable forward/backward presentations, 4 reference forwards and 4 additional detached forwards. Backward checkpoint recomputation and chunked vocabulary projections are additional internal work and must be costed rather than hidden by that count. Measure peak memory, reference-forward/projection time, total forward/backward cost and exact state restoration. Test reference immutability after temporary nonzero steps. Native Flex, BF16 arithmetic, device fit and actual optimizer equivalence remain unqualified until this proposed GPU work is separately authorized and succeeds.

For a later authorized one-event comparison, U/P require **512 total response presentations and 256 main optimizer steps**, with no additional feedback generations and no retraining of O. A clean F/O/U/P serving comparison would require **64 × 4 × 3 = 768 fresh evaluation generations**, using the retained O checkpoint and new U/P checkpoints; its first eight requests are **96 included generations** for native publication/output checks. Do not combine new candidate timings with historical O/F timings. Those main training and evaluation counts are future planning only, separate from the present CPU-only deliverable and the proposed five discarded qualification steps.

## Sources

- Original user attachment: `/home/minhao2/.codex/attachments/8fc6f35a-885c-47d5-955f-919ad7ef2659/DFLASH_PREFIX_RANKING_EXECUTOR_READINESS_20261025.txt`.
- Readiness source inspection: `../dflash_prefix_ranking_readiness_20261025/READINESS_REPORT.md`.
- Completed ordinary baseline and retained-pool contract: `../dflash_osd_one_event_20261025/final_review/REPORT.md`.
- Latest explicit user handoff in the conversation: U/P implementation and CPU tests only, retained feedback/schedule, no GPU work yet.

Qualification scheduling clarification: the two presentations are the first two rows of the full retained 128-response pass, including its original 128-element shuffle and subsequent anchor RNG draws. Do not replace this with a newly shuffled two-response pool. Stop the disposable trial after the first complete accumulation window and restore state between trials.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_implementation_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
