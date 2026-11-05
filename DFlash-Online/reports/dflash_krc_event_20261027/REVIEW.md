# Independent implementation review: KRC

**No remaining algorithm or integration blocker in the reviewed source.** The tested correction formula, ordinary K objective and trainer are unchanged. The new code selects a separate reference pair and adds paired measurements around each ordinary K step. The first native learning-pair gate and full native output checks remain required during execution.

## Material Passport

- Workflow: ARS experiment-agent implementation and reproducibility validation.
- Inputs: HANDOFF.md, NUMERICS.json, current workspace decisions, tested KC/single-step sources and new KRC source/tests.
- Scope: local source review and CPU tests only; no remote or GPU work by this reviewer.
- Status: **31 new CPU tests passed independently**, with CUDA disabled. Full-event native validity and performance are pending.

The reference schedule is built before execution from the retained presentation receipt. Each pass has 64 two-response batches, with the reference taken from the next batch and wrapping within that pass. The original reference pair's own anchors and keep masks remain attached to it. Learning/reference IDs are disjoint; all identities bind the original feedback pool, and the retained development inputs remain outside both roles. Reference use is additional training exposure, not held-out evaluation.

Each reference prefix is captured under the current pre-step KRC parameters. Per-response RNG and cache state are saved before its no-grad forward and replayed for the post-K and corrected reference forwards, under the same evaluation mode. The surrounding ordinary K RNG, modes, cache and gradients are restored. A stochastic CPU fixture verifies exact unchanged-weight reference predictions after intervening RNG progression. The ordinary K training path retains its own original presentation/anchor sequence and optimizer state.

The unchanged tested correction receives only reference-response prefixes and their global two-response denominator. Canonical margins, triangular weights, FP32 gradient accumulation/norm decision and BF16 application are reused. Learning-batch probes obtain no correction gradient and never apply an independent update. Device-local Adam snapshots and runtime checks verify that auxiliary work leaves the ordinary optimizer state unchanged. Fresh per-window recorders prevent stale masks, gradients or feature buffers from carrying into the next window.

Only steps 1, 64 and 128 score both roles after correction. The first-window gate compares the complete learning-pair pre-step and ordinary-K post-step response records against the retained diagnostic before applying the reference correction and proceeding. Reference/corrected measurements are not required to match prior KC. The accounting is 256 ordinary K forwards, 256 pre-reference forwards, 256 post-K reference forwards, and 18 diagnostic forwards: **786 core response forwards**, at most 512 backwards, 128 Adam steps and at most 128 direct corrections. The separate qualified startup adds two forward/backwards and zero optimizer steps.

The genuine K/KRC evaluator preserves the original 64 development requests, exact retained F token/termination references and three balanced repeats, totaling 384 generations. It validates publication, native output, scalar measurements and within-arm repeat consistency. Training closes before serving. Null or harmful valid results do not prevent completion. Native serving measures, adaptation work, instrumentation and benchmark switching costs remain distinct; historical KC timings are not a fresh control.

One concrete integration bug was fixed during review: the learning probe computed its post-K V without storing the value needed by inherited post-correction scoring. The corrected full two-role fixture passes. Sparse learning-objective chunk counts are explicitly labeled as probe-step counts; the reference recorder alone counts actual module forwards to avoid duplicate hooks. First-gate timing is separately recorded. A launch-script test-module path typo was also identified for correction before sealing.

The independent CPU run covers cyclic wrap/pass anchors, disjointness and evaluation exclusion, active/inactive consecutive windows, stochastic replay, buffer cleanup, learning-gate failure before correction, real trainer two-pass schedule continuity, exact first ordinary Adam parameters/state/RNG, and bounded native-adapter/report validity. The unchanged correction, KC manager and trainer files match the tested sources byte-for-byte. CPU tests do not establish GPU output preservation or performance; execution must retain the prescribed native gates and stop on correctness failures.

Interpretation: improvement on the reference or learning probes is local evidence. The primary success criterion is fresh native KRC versus retained K1 on the reused development panel. Improvement over historical KC while still worse than K1 is reduced harm, not success. The extra reference exposure and work prevent a compute-matched causal claim against ordinary K.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_event_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
