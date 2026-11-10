# Remaining FSG qualification after startup repair

## Material Passport

- ARS experiment-agent, run and validate. Native execution followed by same-team source, numerical and work-record review; reviewers disclose implementation overlap. This is not independent GPU replication.
- Execution commit: `90666a76300da2200cb2062b7bc41ee4327f0dfb`, PR #33. New continuation `qualification_004`; original failures and ledgers remain immutable.
- Scope: remaining qualification only. Algorithms, native BF16 production eight-block head layout and partial final chunk, tolerances, inputs, strict norm/KL guards and original cases are unchanged. No feedback collection or pilot training.
- Raw private records remain outside Git at `/home/minhao2/EAGLE_SPEC/research/dflash_functional_step_geometry_20261108/qualification_resume_001/retained`; remote root `/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/qualification_004`.

## Result

**The remaining correctness checks passed, and the earlier startup mismatch is resolved. The pilot remains paused because both representative FSG proposals fell back to ordinary K+.** Exit status is 0. Supervisor wall time was 1,042 s (17 min 22 s); the qualification runner took 896.196 s. CPU tests, authentication and final validation are outside that runner timer.

This is a completed correctness qualification of the paths exercised. It is not evidence of FSG learning or serving benefit. Neither natural window retained a nontrivial correction. Active native scalar matching was also not reached: each SCALAR case rejected its own raw FSG proposal before the scalar search. CPU fixture coverage of that search does not substitute for an accepted native case.

## What changed, and what passed

The earlier ordinary-gradient comparison mixed a cold first backward with an already warmed backward. The preceding zero-Adam diagnostic reproduced that difference and found exact agreement when both branches were warmed. The repaired qualification now performs separately counted, discarded startup, restores the complete scientific state, and only then captures its ordinary reference. This fixes the qualification procedure; it does not change FSG, K+, AdamW, BF16 or tolerance values. It does not identify a defective low-level kernel.

A prelaunch packaging failure also exposed that compact retrieval intentionally omitted YAML files. The new sealer authenticates all 32 original sealed non-SEAL files and copies the complete original package. All six arm configurations match the old configuration except their disposable output directories. The incomplete sealing attempt is retained; it consumed no model work.

| Check | Outcome |
|---|---|
| GSM8K SCALAR ordinary gradients | All 58 tensors / 1,048,626,432 coordinates exactly equal |
| GSM8K SCALAR native Adam endpoint and state | All 239 tensors / 3,145,885,306 compared coordinates exactly equal |
| MathQA neutral, FSG and SCALAR ordinary controls | Exact ordinary gradients and native Adam endpoint/state |
| Independent same-layout formula oracle | GSM8K retained pass reused; MathQA fresh pass. Both 8+1 chunks, all 58 gradients exact, zero coordinate violations and zero relative L2 discrepancy |
| Masks, weights, sensitivity/energy and state isolation | Independent oracle and restored-state checks pass on their bounded fixtures |
| Zero-strength behavior and exports | Exact native endpoint/state; exported tensors verified against decoded state |
| Serving | Eight of eight executions preserve canonical tokens and termination; neutral cases also preserve native proposals and verification trajectories |
| Resources | Retained maximum-shape check reused; no duplicate resource forwards |

The oracle retains `atol=1e-5`, `rtol=0.016`, and global relative-L2 threshold `0.016`. The original cross-layout test **failed and remains a disclosed BF16 layout-sensitivity result**; it was not retroactively passed. Same-layout agreement qualifies the fixed numerical implementation on these fixtures, not arbitrary layouts or every future request.

The reused maximum-resource fixture is two 512-anchor responses at 2,272 positions with four simultaneous BF16 VJPs. Its peak allocation/reservation was 47.018 / 48.564 GB. It contains synthetic masked context and repeated anchors, so it establishes resource feasibility, not natural correction activity or efficacy. Fresh remaining cases peaked at 38.421 / 40.343 GB allocation/reservation.

## Why the pilot did not start

| Representative FSG case | Ordinary BF16 update norm | Continuous FSG norm | Rounded FSG norm | Rounded excess | Kept result |
|---|---:|---:|---:|---:|---|
| GSM8K, retained case | 1.4583258594 | 1.4559686184 | 1.4584726072 | +0.0100628% | Exact ordinary endpoint |
| MathQA, fresh case | 1.1423660015 | 1.1369677694 | 1.1424440155 | +0.0068292% | Exact ordinary endpoint |

Both continuous proposals shrink the update, but BF16 rounding makes their realized norms slightly larger. The unchanged strict norm guard rejects them before candidate KL evaluation. Thus candidate KL is unavailable for these rejected proposals; it was not measured and should not be inferred to pass or fail. Both SCALAR cases independently reached the same rejection using their own state, and retained exact native endpoints.

There are **0/2 nontrivial accepted representative FSG corrections**. The synthetic resource endpoint is not counted as a representative accepted write. This satisfies the reviewer's explicit stop-before-pilot condition. It demonstrates that this implementation and guard prevented application on these two windows; it does not prove universal inactivity or that functional geometry cannot help. No tolerance relaxation, norm allowance, precision change, tuning or new case was introduced.

On the two retained serving checks, all four methods used 19 calls for GSM8K and 20 for MathQA, with exact outputs. These are correctness controls after fallback, not a performance comparison or evidence of equivalence.

## Work, cost and preservation

| Work | New in this continuation | Cumulative qualification | Current cumulative ceiling |
|---|---:|---:|---:|
| Disposable Adam applications | 5 | 8 | 8 |
| Ordinary response forwards / backwards | 10 / 10 | 18 / 18 | 18 / 18 |
| Additional response forwards / backwards | 27 / 16 | 61 / 36 | 67 / 36 |
| Serving generations | 8 | 8 | 8 |

All attempted work completed; no unknown work or execution retry. The latest failed ledger is preserved byte-for-byte as the new ledger's prefix. Six candidate forwards were unused because norm rejection occurred first. Completed GSM8K ordinary, neutral and FSG optimizer work was not repeated. GSM8K oracle and maximum-resource results were reused with identity checks and zero new work.

The 27 new additional forwards include separate startup and ordinary-reference recapture, SCALAR/neutral/FSG helper work and the MathQA oracle. They are not all intrinsic FSG overhead. Earlier standalone debugging remains separate: 28 forwards / 20 backwards / zero Adam across its two attempts; earlier layout/precision/consequence diagnostics remain separately retained too. Hook timers contain nested component times and native Adam; do not sum them as disjoint costs or treat qualification wall time as deployment overhead.

All 222 selected local CPU tests passed; all 188 launcher CPU tests passed on A800. Sixteen focused tests passed after the input-packaging fix. Preflight and postflight authenticated 7,108 dependencies each. Admission required 32 GiB of peak additional space plus an 8 GiB reserve, against 55.339 GB free at the actual authenticated admission (55.525 GB before transfer). Postflight had 42.752 GB free. A separate final live-status receipt confirms both GPUs idle. This admits the bounded qualification, not the main pilot.

All 201 retrieved compact files (11,441,361 bytes) were independently rehashed. Raw inputs, responses, anchors, work ledgers and prior failure records remain private and retained. Original recovery dependencies and serving exports remain on A800. Disposable qualification state tensors were retired only after their exact decoded/export checks, with per-case receipts; they are not archived. No original scientific evidence was deleted. After validation, only the redundant 21,466,378-byte source-transfer archive was removed on each host; exact cleanup receipts and retained source locations are recorded. No scientific replay capability was lost through this archive cleanup.

## Interpretation and next boundary

The prior ordinary-gradient failure was a startup-comparison bug and is repaired. The remaining issue is a **mechanism-application limitation under the chosen BF16 norm rule**, not another observed gradient mismatch. Qualification does not authorize interpreting an inactive mechanism as a failed learning method.

The next reviewer decision is whether to change the BF16 acceptance design or defer this mechanism. This report makes neither change. A revised guard would be an explicit algorithm/protocol revision requiring new qualification; it must not be silently substituted. No main pilot was launched.

All 11 ARS interpretation categories were checked in the accompanying review: no population-rate, significance, equivalence, causal kernel, or serving-benefit claim is made; small-fixture scope, selected cases, historical failures, protocol revision and reused measurements are disclosed. Numerical correctness and mechanism efficacy remain separate.

Evidence links: [machine summary](STARTUP_REPAIRED_QUALIFICATION_SUMMARY.json), [exact ordinary controls](STARTUP_REPAIRED_EXACT_CONTROL_SUMMARY.json), [source admission review](STARTUP_REPAIRED_QUALIFICATION_SOURCE_REVIEW.md), [preceding startup diagnosis](ORDINARY_GRADIENT_DEBUG_REPORT.md), [historical revised-qualification failure](REVISED_QUALIFICATION_REPORT.md), and [BF16 consequences](BF16_CONSEQUENCES_REPORT.md). Private raw identities and outputs are retained at the local/remote locations above.

Final reviews: [numerical and interpretation review](STARTUP_REPAIRED_NUMERICAL_RESULT_REVIEW.md) and [work/cost review](STARTUP_REPAIRED_WORK_RESULT_REVIEW.md). They check retained evidence and are not independent native reruns.
