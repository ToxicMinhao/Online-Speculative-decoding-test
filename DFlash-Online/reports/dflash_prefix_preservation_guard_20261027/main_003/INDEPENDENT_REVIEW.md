# Independent main-run review

**No launch-blocking implementation finding.** Review is limited to the authorized main event and its orchestration. No remote/GPU work was performed by this reviewer.

The main driver reuses the passed `repair_002` qualification, checking all three trial statuses, exact budget receipt, original feedback/RNG identities, byte-identical training configuration and the pinned compiler setting. It never calls `run_qualification`. Before new model compilation it pins `donated_buffer=False`, executes exactly two separately accounted startup forward/backward evaluations with zero optimizer steps, then constructs a fresh G trainer and checks its released parameters, empty optimizer/counters, modes and full RNG against the original pin.

The scientific path is unchanged: original 128 feedback features and matched presentation/anchor trajectory, two passes, 256 presentations and 128 steps; one enabled guard decision per step. Probe membership is exactly steps 1/64/128 × ordinary-K/projected-G/norm-scaled-K × two current responses = 18 draft-only response evaluations. Existing guard code restores parameters, RNG, modes and caches around probes. Training and guard objects close before loading the native backend, so timed evaluation has no concurrent training.

The retained K1 export and new G export are the only evaluation arms. The unchanged evaluator uses the original 64 development requests, three balanced repeats, exact retained-F tokens/termination, within-arm repeat-count consistency and one validated count trajectory per request. Its declared budget is 384 generations; the first 48 correctness-gate generations are included. Publication/loading costs remain separate from request latency. Failure paths preserve partial work and stop; there is no automatic retry or additional event.

Preparation verifies the passed repair's entire source/artifact seal, copies no new core implementation, and seals the successful qualification records with the new orchestration. Storage admission includes 23,069,850,144 bytes of tensor overlap allowance, 4 GiB transient/compiler/log allowance and 8 GiB reserve, totaling 35,954,752,032 bytes, plus 72 GiB available host RAM. The launcher checks GPU isolation, runs focused CPU orchestration tests, then executes the main driver once; postflight runs after success or failure. The supervisor bounds execution to three hours without automatic restart.

Seven orchestration CPU tests passed independently in 0.064 s, covering qualification reuse/configuration identity, no repeated qualification, startup-before-fresh-G ordering, one event, exact probe budget, trainer closure before evaluation and preserved original errors. The analyzer also passed two lightweight parser fixtures covering an interrupted startup and completed measurements, historical-versus-new step accounting, nested probe-time subtraction, BF16 directional violations and local-probe denominators. These fixtures are not native performance measurements.

`analyze.py` writes the compact report, guard-step/response CSVs and local-probe tables. It explicitly separates five historical temporary optimizer steps from zero new temporary steps and two new startup evaluations, excludes historical qualification from current wall time, and does not add nested probe/gradient/projection costs twice. Partial startup work is read from retained counters even when no successful startup receipt exists.

Remaining limits are unchanged: the qualification's enabled decision was inactive, so active GPU projection was not exercised there; main-run records must preserve any BF16 constraint deviations. Historical/default-compiler K1 full-tensor equivalence was not established, although old/repaired ordinary-K scalar and sampling records matched. The development panel is training-held-out but already used in development. Local shrinkage probes do not replace a trained shrinkage control, and no efficacy claim exists before the main results.

Reviewed files: `campaign.py`, `prepare_remote.py`, `preflight.py`, `run_campaign.sh`, `supervise.sh`, the retained guard/evaluator contracts, and `analyze.py`. Source remains frozen for launch; no algorithm or numerical-gate change was introduced by this review.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_preservation_guard_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
