# Workload-conditioned replay with K

This fixed eight-event probe asks whether coverage-selected replay improves K
over both fresh-only training and uniform replay. It starts three new seed-17
trajectories from released Qwen3-8B-DFlash-b16. It is not a continuation of K58.
The scientific parent is publication `4d94f30c42701d0f86ac04e2829c834593c16d8e`
(PR23 execution `0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34`).

Each event first serves its 128 fresh requests under all three preceding
checkpoints. K alone captures feedback. All arms learn from those same complete,
target-verified packets after serving and validity checks finish. K receives two
fresh passes. From event 2 onward, U and R replace 32 of the second pass's 128
responses with past replay, keeping the same 96 fresh responses. U selects past
responses uniformly; R greedily maximizes prompt-feature coverage. Event 1 is
identical across all three arms and must pass exact decoded scientific-state
equality, including native sampling and optimizer/RNG state.

The K objective, native optimizer, model, attention, precision, loss normalization
and 7,424-step horizon stay unchanged. `training.py` adds explicit pass allocation
around the existing SpecForge operations. It does not weaken the historical
full-K no-replay guards. Every replay presentation samples native anchors again
and computes the current model's prefix-plus-first-error mask.

`selection.py` owns the independent PCG64 streams, 128-packet Algorithm R
reservoir and fixed FP64 descriptor/coverage arithmetic. Its API has no domain
labels, answers, losses or evaluation results. Reporting joins domains only after
selection. Memory insertion follows all three verified commits, so current
responses are never treated as old replay.

`campaign.py` separates serving, packet preparation, allocation, isolated native
training workers and deferred exports. `worker.py` makes no serving calls and
performs no disposable optimizer step. It preserves a hardlink to each verified
state before the next pending save can replace the latest pointer. Preparation
validates native target binding before retiring only this run's redundant bulk
features. Previous exports retire only after their serving checks finish;
previous recovery states retire only after an exact own successor is verified.

The storage plan includes full cap-length packets, shared initial tensors,
three model/Adam states, pending saves, exports/workspace, 5% tensor headroom,
4 GiB additional compiler growth, 3 GiB compact evidence, 2 GiB workspace and an
8 GiB free reserve. The compiler-growth allowance is an explicit operational
choice based on the already-populated qualified caches, not the previous run's
8 GiB allowance. It and compact growth are monitored; crossing a limit stops the
run. No historical deletion or unverified offload is part of admission.

The fixed math/code order is M1, M2, C1, C2, M3, M4, C3, C4. Final evaluation uses
the interleaved 128-math/128-coding reused development panel, with three balanced
capture-disabled repeats. Primary R/U and required R/K comparisons use pooled
calls; per-domain outcomes, all repeats and gross harms remain visible. Endpoint
intervals use the prescribed 10,000 paired stratified resamples. Future-event
counts are descriptive, without an independent-request trajectory bootstrap.

| Work | Total |
|---|---:|
| Pre-update complete generations, including K feedback | 3,072 |
| Additional capture parity generations | 4 |
| Final complete generations | 2,304 |
| All complete generations | 5,380 |
| Terminal-feature preparations | 1,024 |
| Native Adam steps | 3,072 |
| Ordinary response forward/backward presentations | 6,144 |
| Replay presentations, already included above | 448 |

Before launch, publish this code, run the focused CPU suite, seal the exact
retained inputs/configuration and actual paths with `seal.py`, then run
`run_campaign.sh <sealed-run-root>` on A800. The shell performs current source,
runtime, hardware and storage checks. GPU qualification is included in the
production event-1/event-2 paths and four capture-parity controls. There is no
extra preliminary GPU run, automatic retry or efficacy gate.

Outputs include `final_report/REPORT.md`, paired and all-repeat tables, event
counts/exposure, every descriptor/selection/memory receipt, exact source/input
bindings, native validity records, complete initial/final states and final
reservoir. The separate read-only review checks these outputs against the fixed
handoff. A completed negative result remains a completed experiment; no later
mechanism change or replication is automatically authorized.

## Completed fixed campaign

Execution `8a4cbcc8b78adfee845a0a1409db4a0d29c54058` completed all eight
events, 5,380 generations and 3,072 Adam steps. All planned outputs passed
validity checks. The balanced final panel did not establish an improvement
from coverage-selected replay over uniform replay or K. The complete
[report and paired records](../../reports/dflash_k_workload_replay_v1_20261102/REPORT.md)
retain the domain tradeoff, uncertainty, costs and separate CPU evidence reviews.
No follow-on experiment was launched.
