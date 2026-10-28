# R one-event execution: independent integration review

Review completed October 26 under ARS experiment-agent execution/validation. No unresolved launch-blocking issue was found in the final inspected implementation. This is code/CPU evidence, not a GPU qualification result.

Reviewed [campaign.py](campaign.py), [preflight.py](preflight.py), [qualify_r.py](qualify_r.py), and the execution worktree's `experiments/prefix_ranking/{protocol,evaluate,report}.py`, with their retained trainer, presentation-schedule and controlled-startup dependencies.

One integration defect was found and fixed before the final CPU suite: `prepare_main` originally returned the tensor-bearing scientific snapshot from `validate_fresh`, while the campaign expected a JSON-compatible receipt containing `wall_s`. It now releases the returned snapshot and supplies a compact initialization receipt; the exact state comparison and persisted initial snapshot remain intact. The qualifier CPU test explicitly serializes this receipt.

The inspected flow bounds qualification to five discarded optimizer steps and ten presentations, including two R1 windows. R1 retains separate snapshots for both optimizer boundaries and all four backwards. Fresh main R validates released weights, empty Adam/counters and restored complete pinned RNG; no temporary weights or optimizer state are loaded. Qualification and main share the warmed process. Main training checks the original 128-feature, 256-presentation schedule and closes before native evaluation. Failures stop and retain evidence; valid unfavorable outcomes continue.

The evaluator accepts explicit F/O/P/R identities and comparison pairs while preserving old F/O/U/P defaults. Its first 96-generation correctness gate is inside the fixed 768-generation budget. Reports use R/O as primary, preserve all repeats and raw native records, and keep capture disabled. R diagnostic totals aggregate only completed accumulation boundaries. The outer storage allowance exceeds the independently counted 31 draft-weight copies of retained qualification snapshots plus failure/atomic overlap and main state/export needs.

[CPU_FULL.json](CPU_FULL.json) records **443 tests passed, zero failures/errors/skips, 57.545 seconds, CUDA uninitialized**: 423 retained tests, 15 campaign/evaluation tests and five qualifier tests. The earlier focused compatibility run passed all 31 campaign/evaluator tests, including 16 unchanged evaluator tests.

Remaining uncertainty: CPU eager fixtures do not establish full-size GPU/Flex numerical compatibility, peak device-memory feasibility, native serving correctness, latency or efficacy. Actual protected/active R counts must be reported separately from synthetic detached-logit gradient checks. New exact startup gates do not prove retained O's complete historical 128-step trajectory equivalent to warmed initialization. Stage 1/project None remains unchanged; the authorized GPU qualification must pass before the one-event campaign proceeds.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_margin_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
