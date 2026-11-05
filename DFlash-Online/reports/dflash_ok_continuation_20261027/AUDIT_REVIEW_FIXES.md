# Independent auditor review fixes

Two issues found by the separate reviewer were fixed before using the auditor on current-run evidence. They affect only local postrun validation, not sealed campaign code, training, inference, or any raw record.

1. A child evaluation with all384 valid measurement rows but `STATUS.status='failed'` previously returned an incomplete child summary without adding an issue. A root completion flag plus the full aggregate record count could therefore have passed. The global gate now requires all three pools, six own-arm resume/training summaries, and both evaluation summaries and child execution statuses to be complete. The regression materializes384 valid synthetic measurement/validation rows with a failed child status, verifies rejection despite root completion and exit zero, then verifies the positive gate with complete children.
2. The audit's `cap_hit` comparison used output length2048, while the qualified measurement helper defines it as `finish_reason=='length'`. It now uses the exact qualified definition. A synthetic EOS-at2048 fixture verifies `cap_hit=False`; changing that field to true is rejected. Native first-mismatch/token checks have separate tests and are mocked only for these accounting-focused fixtures.

All16 CPU tests pass in `AUDIT_CPU_TESTS_REVIEW_FIX.log`. Earlier14-test evidence remains in `AUDIT_CPU_TESTS.log`. No current-run data, GPU, model forward, optimizer step or remote operation was used for these fixes.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
