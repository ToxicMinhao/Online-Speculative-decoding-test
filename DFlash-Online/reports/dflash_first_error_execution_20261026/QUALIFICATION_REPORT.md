# O/C/K/A GPU qualification

All seven correctness gates passed on A800. Independent compact-evidence audit found no issues across 22 exact comparison receipts, the complete work log and 35 audited temporary-tensor retirements.

- All-selected K matched ordinary O at both backward boundaries, before the optimizer and in updated weights, Adam state, counters, RNG and training records.
- All-selected A matched full canonical CE under the same checks.
- Natural K/A masks, first-error inclusion, BF16 loss arithmetic, per-response normalization and logging matched independent calculations on current-forward logits. Sampled excluded positions had zero direct prediction-loss gradients.
- Fresh initialization, retained anchors/order, no-reference execution and memory fit passed. Original O first-window scalar metrics matched its retained receipt.

Work: 14 response presentations/current forwards/backwards; seven discarded optimizer steps; 30 bounded detached-logit gradient checks; zero generation, teacher-model replay or reference forwards. Internal objective chunks and activation-checkpoint recomputations are counted separately in RESULT.json. Driver time: 639.096 s; maximum allocated GPU memory: 14.464 GB. The enclosing qualification timer also includes setup and is retained separately for the final cost report.

Temporary states from this qualification were retired only after durable exact checks and hash receipts; original failure records and O/U/P/R artifacts remain intact. Direct tensor-level rerun of those retired temporary states is no longer available. Retained compact evidence is in raw/campaign_001/qualification.

This establishes implementation compatibility for the checked windows, not benefit from training or full 128-step equivalence to historical O. The original startup limitation remains: No equality claim for retained O's full historical 128-step trajectory

At 2026-10-26 23:05:40 UTC, the supervisor had moved to training_C. Main C's released-weight/empty-Adam/counter/RNG initialization matched the pinned state exactly (61 tensors). No qualification learning is carried into main training. C/K/A each run one independent 128-step event, followed by the planned 768 fresh O/C/K/A replays. Native evaluation output checks and efficacy remain pending.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
