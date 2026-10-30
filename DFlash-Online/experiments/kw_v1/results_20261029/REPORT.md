# KW-v1: implementation complete; native qualification blocked

## Material Passport

ARS-Codex 3.24.0 experiment-agent execution and validation, October 29, 2026.
Scientific execution commit: **`9fd9fe6f7db9796a49e2a6879b5a79e7f01fbd22`**;
[PR #19](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/19).
Status: **Implementation invalid — first neutral-window state equality failed.**
Stage 1 / project None is unchanged. This is a qualification/blocker report,
not an evaluation of KW's drafting performance.

## Outcome

The bounded GPU qualification stopped after KW's first temporary update. Its
allocation was exactly zero, but the updated draft weights and Adam moments did
not exactly match the ordinary K control from the same retained K8 initialization.
This violates the required neutral-start contract. No production event or serving
evaluation was launched, and no tolerance, precision, optimizer or algorithm was
changed to obtain a pass.

The experiment therefore provides **no evidence that KW improves or harms
drafting**. The O/K/KW event-12 comparison is unavailable because its correctness
prerequisite failed.

## What was implemented and checked

KW adds the specified five-feature, five-parameter response allocator around the
existing K trainer, with raw half-loss gradient capture, exact neutral bypass,
two native BF16 AdamW previews, delayed FP64 credit, pass-boundary clearing and
complete-event controller restoration. O/K training code and scientific settings
remain unchanged. The campaign, input sealing, balanced three-arm evaluator,
work accounting and report are implemented for the fixed events 9–12 scope.

The combined 76-test CPU suite passed locally and on A800 with CUDA hidden.
The final 10-test trainer/qualification suite passed, including a complete tiny
KW event and state restore. After a separate review identified a completion-audit
gap, exact completed-work/zero-unknown gates were added; the focused nine campaign
and seven report tests passed locally and on A800. These suites overlap and must
not be added as independent test counts. CPU success did not establish full-model
GPU equivalence.

Inputs were sealed before outcomes: 512 eligible training requests from verified
source cursor 1024→1536; the three retained panels; and a new seed-20261030 final64.
Selection uses questions only. The final panel is current-study fresh; 61/64 have
known older unrelated-project exposure. A separate review reconstructed the
selection, all final rendered prompts, retained canonical references and all
4,032 evaluation slots. All executions planned for this campaign remain unattempted.

Preflight passed source/runtime/model pins, exact original O8/K8 state/export
checks, the four longest event-8 feature records and resource admission. Storage
was 218.0 GB free against the fixed 215.575 GB conservative requirement, including
maximum-length feedback copies, checkpoint overlap and the 8 GiB reserve.

## Native evidence

Both paths used the sealed first pair `train_00947`, `train_00916`. Their complete
retained response rows match: input identities, sampled anchors, every reported
loss/selection metric, optimizer-step flags and reported training counters.

| Recorded quantity | Ordinary K | KW |
|---|---:|---:|
| First response loss | 0.2223545313 | 0.2223545313 |
| Two-response mean loss at update boundary | 0.1569999009 | 0.1569999009 |
| Available supervised occurrences, pair | 14,320 | 14,320 |
| K-selected supervised occurrences, pair | 9,507 | 9,507 |
| Native pre-clip gradient norm | 0.2099609375 | 0.2099609375 |

KW logged `a=0`, weights `(0.5, 0.5)`, unchanged zero controller coefficients and
no delayed credit. Its ordinary and weighted FP64 gradient-norm summaries are
identical, at 0.2095905596. Equality of scalar metrics or norms does not establish
coordinatewise gradient equality.

The first full-state comparison nevertheless failed. The retained list contains
31 differing weight-tensor paths and 69 Adam-moment paths (42 `exp_avg`, 27
`exp_avg_sq`). **The list hit its 100-path cap:** it is not the total Adam-field
or coordinate violation count. The full weights traversal precedes the optimizer
traversal, so the 31 weight-tensor count is complete. Coordinate error magnitudes
were not retained. RNG, modes and all Adam counters cannot be declared equal
from their absence in this truncated list.

A read-only source review found no direct live-state mutation or scratch alias:
hooks clone contributions and return `None`; the neutral path skips gradient
replacement; preview parameters, moments and gradients are separate; the live
step uses the original adapter. This is source evidence, **not proof of GPU
preview isolation**. The baseline's coordinate gradients and the corresponding
KW pre/post-preview tensors were not persisted, so the records cannot distinguish
backward repeatability, hook/timing effects or an optimizer/integration issue.
No kernel or numerical root cause has been established.

## Work, cost and retention

| Qualification work | Completed | Limit |
|---|---:|---:|
| Response forwards | 6 | 8 |
| Response backwards | 6 | 8 |
| Disposable optimizer applications | 5 | 16 |
| Serving generations | 0 | 0 |
| Persistent updates | 0 | 0 |

The five applications comprise two ordinary K steps, two KW scratch previews and
one KW live temporary step. All 17 recorded forward/backward/optimizer attempts
completed. The second KW window and optimizer-only nonzero/isolation checks were
not attempted. No retry, warmup generation or additional model check followed.

Qualification took 106.590 s; preflight took 73.689 s; wrapper elapsed time was
186 s. Ordinary K peak allocation was 20.124 GiB. The partial KW path reached
39.816 GiB allocated and 40.930 GiB reserved without an out-of-memory error.
This does not complete the required resource/numerical qualification. Qualification
timers include host snapshots and equality checks and are not serving latency or
a representative estimate of production controller overhead.

All production work is unattempted: 512 feedback generations, 1,536 committed
Adam steps, 3,072 presentations, 1,024 previews, 504 credits and 4,032 evaluations.
There are no event-9–12 results, endpoint checkpoints, per-request serving gains
or timing repeats to report.

Postflight passed all 2,617 sealed-file checks and eight original artifact
identities, including exact O8/K8 state/export validation, with zero model work.
Both GPUs are idle. Raw qualification records, partial response/window records,
the failure traceback, source/input seals and original checkpoints remain intact.
Temporary full-model gradients and endpoint snapshots lived only in memory; they
were discarded on process exit, not archived.

Under the standing retention authorization, 77 obsolete paths representing 59
unique old qualification/initialization containers were retired after preserving
and verifying compact evidence on Fenrir. These containers represented 169.88 GB of unique allocated storage.
Original failures, small RNG/configuration metadata, source identities, tensor
summaries, useful final exports and active O8/K8 dependencies remain. Exact
reinspection of those retired old tensors and re-serving the three disposable
historical diagnostic exports are no longer available. Those tensors are retired,
not archived.

## Deliverables and limit

- [Work accounting](analysis/WORK_ACCOUNTING.json), [failure paths](analysis/FAILURE_PATHS.json),
  [matched first-window metrics](analysis/MATCHED_FIRST_WINDOW.csv),
  [costs](analysis/COSTS.json), [execution pin](analysis/SOURCE_PIN.json).
- [Separate qualification failure review](SEPARATE_QUALIFICATION_FAILURE_REVIEW.md)
  and [read-only source analysis](SOURCE_FAILURE_ANALYSIS.md).
- Private raw artifacts: `/data/sources/minhao/runs/dflash_kw_v1_20261029` on A800,
  with the compact retrieval under this report's `raw/` directory on Fenrir.

The concrete blocker is unresolved **native neutral-update equivalence**. Existing
evidence cannot locate the first coordinate-level divergence. No algorithm
revision, diagnostic rerun or production continuation was performed after failure.
