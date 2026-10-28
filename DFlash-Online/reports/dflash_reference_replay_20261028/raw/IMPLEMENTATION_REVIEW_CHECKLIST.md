# KR implementation and evidence checks

Material Passport: computational experiment; executor/monitor workflow; raw retained feedback and native measurements; user-authorized one-event practical replay control; Stage1/projectNone unchanged.

- Fresh released weights, fresh Adam state/counters and pinned seed17 state before first A.
- Compiler warm-up uses disposable runtime, two forward/backwards and no Adam steps; exact restoration, normal main trainer reconstruction.
- First A uses original first two presentations. Compare retained ordinary K metrics/anchors; no extra forward/protected capture or historical post-step tensor-equivalence claim.
- Next-pair cyclic R is disjoint from A, including last-window wrap inside each pass; own retained anchors; each response twice A and twice R.
- Real R trains in normal mode and advances optimizer, decay and RNG. No restoration around actual R; no correction machinery.
- Response normalization preserved independently; microbatch1 and accumulation2. Current first-error mask, known-seed/EOS rules unchanged.
- 128 outer windows, 256 Adam steps, 512 response forwards/backwards; one final state/export. Separate role timing and chunk/recomputation work.
- Close trainer and release GPU state before timed serving; no transfers/concurrent training.
- Two unchanged panels; 768 executions; included48-generation gate; balanced order; exact retained outputs and per-request retained KRC counts; repeat consistency.
- Complete request, native-generation and cache-maintenance timers separately, avoid nested cost double counting.
- Preserve invalid/unattempted records; no retry/tuning on failure.
- Primary comparisons per panel; historical K1 calls and KRC training cost contextual, not fresh timing control. No equal-compute or causal-isolation claim.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_replay_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
