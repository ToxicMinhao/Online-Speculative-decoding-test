# Prefix-preservation guard: handoff understanding

## Material Passport

- Date: 2026-10-27. ARS experiment-agent, experiment-plan review.
- Status: handoff and retained source inspected; no implementation, tests, remote execution or training in this review.
- Scope: one new G event and the specified qualification, local probes and K/G development comparison. Stage 1/project None remains unchanged.
- Review: parent inspected the retained K objective, optimizer, reduction and qualification evidence; one delegated reviewer independently checked the handoff's mathematics, budgets and interpretation. No substantive protocol clarification is needed.

## Experimental question

Does correcting the component of K's native optimizer displacement that locally increases a current-prefix protection quantity improve held-out drafting, while retaining useful repair progress?

G starts from released DFlash with fresh Adam state and the qualified seed-17 initialization. It reuses the original 128 event-1 responses/features and exact presentation/anchor schedule: two passes, 256 presentations, 128 optimizer steps. The comparison checkpoint is retained **event-1 K**, not K after four events. No feedback collection or F/O/K retraining is included.

The original 64-request development panel is held out from training but has already been evaluated and used in development. This screen does not provide fresh-panel confirmation.

## Exact mechanism

K's ordinary forward-KL objective, selected-weight denominator, response averaging, clipping and BF16 AdamW arithmetic remain unchanged. K selects the current correct prefix **including its first error**. H protects only the preceding correct prefix; fully correct blocks also contribute.

For each block, number protected proposal positions from 1 through ell. Assign weights ell, ell-1, ..., 1. These triangular H weights do not replace or multiply K's positional decay. Exclude the known seed, stop at invalid positions and canonical termination, and use the native argmax rule for ties.

Compute H for each response as the sum of weighted softplus(-margin) across all sampled blocks divided by log(2) times the total triangular weight. An empty set contributes zero. Average the two response quantities equally; do not pool their denominators or renormalize away an empty response. Use native logits with FP32 margin subtraction/reduction. Freeze the protection membership and weights during the update and its probes.

Obtain H gradients separately and accumulate them in FP32 without changing K gradients, clipping, Adam moments or counters. No FP32 master model is introduced. Preserve pre-step parameters and execute the native K optimizer step first. Its actual parameter displacement is the proposal.

Apply the stated half-space projection once globally across all trainable parameters when h dot delta_K is positive. Form active corrected weights from the preserved pre-step parameters through the qualified BF16 cast. On disabled/inactive branches leave native optimizer output untouched. Keep Adam state from the K step.

Log both ideal and rounded displacement norms/directional derivatives. The exact-arithmetic projection is a first-order constraint; it does not guarantee finite-step H decrease, preserved token decisions, transfer, or the same constraint after BF16 rounding. Do not add rollback, backtracking, tuning or a penalty coefficient.

## Qualification and bounded measurements

- CPU checks: boundaries, EOS/padding, ties, triangular weights, response normalization, empty sets, zero gradients, global projection and untouched inactive branches.
- Three disposable GPU trials: ordinary K, guard-disabled G and enabled G, all with identical initial state, responses, anchors and RNG. Disabled G must match K; enabled G must match K before correction in gradients, native optimizer output, moments and counters.
- The budget is three temporary steps and six presentations. Count every forward/backward and any startup work. Do not silently import an additional historical optimizer warm-up; any inability to meet the specified qualification must remain visible rather than relaxing gates.
- At main steps 1, 64 and 128, evaluate three candidate states on the two existing responses: native K proposal, projected G proposal and uniformly scaled native proposal. This is 18 response-level draft-only evaluations, with no teacher target generation, optimizer steps, full-response generation or post-update KL evaluation.
- Interpret the scale as the projected proposal's FP32 norm divided by the native proposal's FP32 norm, before the final BF16 cast. Record realized norms too. A zero native displacement needs no rescaling.
- Keep the old H membership, triangular weights and denominator fixed; recompute each candidate's rival. Reuse pre-step measurements. Record H, useful first-error repairs, new earlier errors, matching-prefix changes and proposed/realized norms. Restore parameters, RNG, mode and caches; do not mutate Adam state.

| Work | Budget |
|---|---:|
| Main G response presentations | 256 |
| Main G optimizer steps | 128 |
| Disposable qualification presentations / steps | 6 / 3 |
| Local draft-only response evaluations | 18 |
| Fresh K/G complete-response evaluations | 384 |
| New feedback generations | 0 |

## Native evaluation and interpretation

Use retained K and final G on the original 64 requests with three balanced repeats, the pinned Qwen3-8B/DFlash-b16 and SGLang/FA3 setup, greedy decoding, block 16, 2,048 output tokens or EOS, context 4,096, concurrency 1, capture disabled and the existing cache policy. Timed serving has no concurrent training or transfers. Validate exact tokens and termination against retained canonical references.

Primary reduction is 1 - sum(G calls)/sum(K calls), using one validated trajectory per request. Serving time uses summed per-request means from the three fresh repeats. Preserve per-request gains, regressions and ties. Report guard activation, correction size, realized constraint violations and added update costs; separate loading, training, probes and publication.

Lower H alone is insufficient. Native improvement with retained repair progress supports further study; near-zero updates may indicate suppressed learning. The local scaled candidates are mechanism diagnostics, not trained shrinkage controls. No automatic tuning, additional arm or later event follows this screen.

## Inspected sources

- `../dflash_ok_continuation_20261027/worktree/online_feedback/first_error.py`
- `../dflash_ok_continuation_20261027/worktree/online_feedback/objective.py`
- `../dflash_ok_continuation_20261027/worktree/online_feedback/optimizer.py`
- `../dflash_ok_continuation_20261027/worktree/online_feedback/training.py`
- `../dflash_first_error_execution_20261026/REPORT.md`
- `../dflash_first_error_execution_20261026/QUALIFICATION_REPORT.md`
- `../dflash_prefix_ranking_execution_20261026/startup_review/REVIEW.md`


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_preservation_guard_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
