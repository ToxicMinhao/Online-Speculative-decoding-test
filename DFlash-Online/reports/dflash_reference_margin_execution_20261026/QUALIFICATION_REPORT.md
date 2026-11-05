# R GPU qualification — passed

## Material Passport

- Date: October 26, 2026; ARS experiment-agent run/validation.
- Scope: bounded GPU correctness and memory checks, followed by the user-authorized one-event R experiment.
- Status: qualification passed. The subsequent main experiment is completed; see [final report](REPORT.md). This document reports qualification only.
- Source: [qualification result](raw/campaign_001/qualification/RESULT.json), [R trial](raw/campaign_001/qualification/R1/TRIAL.json), [fresh main initialization](raw/campaign_001/R_INITIAL.json).

The run passed all 443 local CPU tests and 20 remote execution tests before GPU qualification. Source, model/runtime, feedback and retained-control checks passed; storage admission reserved 116.22 GB against 192.87 GB available. No cleanup or baseline retraining was needed.

## GPU checks

R with lambda zero matched fresh ordinary OSD exactly over the checked window: initial state, both backward boundaries, pre-optimizer gradients, updated weights, Adam state, RNG, counters and metrics. The final comparison covered 235 tensors and 3,145,885,050 coordinates. This checks the controlled startup; it does not claim exact equivalence to every step of historical O.

Active R passed four presentations across two temporary update windows. Checks covered fixed reference and RNG isolation, prefix masks, each model's own strongest noncanonical rival, KL and auxiliary reductions, logging denominators, and gradients. In the first window the auxiliary loss was zero because the current model matched the reference. In the second window it was nonzero, and actual-logit gradient checks included active positions. Separate detached fixtures covered equality, improvement, deterioration and changing rivals. These are implementation checks, not evidence that R improves subsequent drafting.

Qualification used exactly five temporary optimizer steps, ten response presentations/backwards, four reference forwards, four detached full-block comparison forwards and twenty bounded detached-logit autograd calls. It generated no answers and performed no teacher-backbone preparation. Activation-checkpoint recomputation and vocabulary-head work are separately counted in the raw ledger. All temporary states remain diagnostic evidence; none becomes the main checkpoint.

Qualification took **366.870 seconds**, with peak allocated GPU memory **24.693 GB** for the active-R trial. The main R constructor then passed released-weight, empty-optimizer, zero-counter and complete-RNG checks.

## Main experiment authorized after qualification

Only R is trained: lambda one, the original 128 retained responses/features and exact two-pass presentation/anchor schedule, 256 presentations and 128 optimizer steps. Existing F/O/U/P artifacts remain intact. Training closes before 768 fresh F/O/P/R evaluations on the same 64 requests, three rotating repeats each. The first eight requests qualify within that evaluation budget. Failures stop and preserve evidence; valid null or harmful results continue without tuning.

Primary comparison is R versus O; R versus P and F are also reported. Request latency excludes training, loading and activation and describes performance with an available checkpoint. Costs and all raw repeats will accompany the final report. No further event or mechanism change is authorized by this run.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_margin_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
