# Independent implementation review

**No remaining launch blocker found.** This review covers the user-authorized single ordinary K step followed by at most one direct correction, not a new training event or serving campaign. Native reproduction and numerical behavior remain execution gates.

## Material Passport

- Workflow: ARS experiment-agent, implementation and reproducibility review.
- Sources: [handoff](HANDOFF.md), [authorization](AUTHORIZATION.json), [numerical contract](NUMERICS.json), new correction core/runner/tests, unchanged K training helpers and retained first-step reference.
- Validation: source review and an independent run of 21 focused CPU tests; no GPU work, model generation or remote operation by this reviewer.
- Stage: existing bounded research stage; no mechanism expansion or follow-up authorization.

## Contract checks

The [core](worktree/online_feedback/prefix_correction.py) records detached predictions during K's existing training forwards. It freezes the original strict matching-prefix membership and triangular weights, including existing seed, validity and EOS rules. Its recording callback returns detached zero diagnostics and does not change K's loss. The CPU integration test now checks exact pre-clipping K gradients, final weights, Adam state, RNG and counters against ordinary K.

Post-K evaluation uses both saved responses with fixed anchors and full-block conditioning. The hinge numerator is reduced over all chunks and both responses against one common protected-weight denominator. This is distinct from the unchanged response-averaged K loss. Rival selection uses native vocabulary logits with deterministic tie handling; gathered logits remain differentiable, and margin/reduction arithmetic and separately accumulated gradients use FP32. Wrong native argmax decisions at zero margin are reported separately and receive zero hinge derivative.

The correction uses one global FP32 gradient squared norm and exactly `-V / norm_squared`. It forms FP32 changes to the actual post-K BF16 weights, casts through BF16 once, and leaves Adam untouched. Empty protection, zero violation and positive violation with zero gradient norm have explicit outcomes. There is no epsilon, coefficient tuning, clipping, extra Adam step or line search. Proposed and rounded norm reporting uses FP64 and is explicitly distinct from FP32 decision arithmetic.

The [runner](worktree/experiments/prefix_correction/run.py) restores the original full initial state and samples the original first two presentations. It checks all eight retained prefix metrics separately for each response before applying a correction. It checks exact optimizer, RNG, mode, cache, gradient and counter preservation around post-K and corrected evaluation, and checks that evaluation itself leaves parameter values unchanged. Repair retention is matched by block identity rather than by subtracting aggregate repair counts.

The budget is one ordinary Adam step, two training presentations, six core response forwards and at most four backwards. Two existing disposable startup forward/backwards are separately accounted for, with zero Adam steps and exact restoration. No generations, checkpoint export, extra qualification step or automatic retry is present. The launcher admits storage before execution and stops after this diagnostic; the supervisor has a 30-minute limit.

## Corrections resolved before launch

1. Initial K observation was marked `no_grad` after a CPU integration test exposed activation-checkpoint saved-tensor interference. The original failed test log remains retained. Observation now has no autograd side effects.
2. Explicit nonfinite checks cover margins, hinge numerators and the final corrected aggregate, including finite BF16 logits whose FP32 subtraction overflows. A regression test exercises this case.
3. The reporting-only K displacement norm now detaches parameters, avoiding a large unnecessary FP64 autograd graph. This changes memory handling, not its numerical formula.
4. Attempted replay forwards and completed work are separately recorded. Chunk-only capture timing is labeled as partial; input copies and snapshots remain in the outer runtime. Nonfinite failure values are preserved explicitly in final JSON.

## Independent CPU validation

Command, from the worktree with CUDA disabled:

```text
python -m unittest test_prefix_correction experiments.prefix_correction.test_run -q
```

**21 tests passed** after the final detach and pre-clipping-gradient changes; CUDA remained uninitialized. Coverage includes global normalization, rival gradients and ties, zero/no-op/unresolved branches, BF16 correction and overflow rejection, block-identity repair accounting, fixed-prefix replay, unchanged K training, durable reproduction failures, state mutation rejection and the execution budget. Existing prefix helpers are reused rather than redefined.

CPU checks do not establish native reproduction, memory fit or a useful correction. The run must preserve unfavorable results. Local advancement requires both fewer lost old-prefix tokens and greater net prefix gain than ordinary K; lower hinge alone is insufficient, and no result from this diagnostic establishes held-out serving improvement.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_correction_step_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
