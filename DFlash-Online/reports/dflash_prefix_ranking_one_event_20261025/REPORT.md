# U/P one-event experiment: stopped during GPU qualification

## Material Passport

ARS experiment-agent, run and bounded failure analysis. Executed on A800 using
DFlash-Online on October 25, 2026 Edmonton time (04:12–04:14 UTC October 26).
**Status: qualification failed; no main U/P training or evaluation ran.**
This is an implementation qualification result, not an efficacy result or a
decision about the ranking method. Stage 1 / project None is unchanged.

## Outcome

The ordinary O qualification window reproduced the retained first window's
metrics exactly. U with **lambda = 0** then used the same two responses and
anchors, but its weights after one optimizer step failed exact equality to O.
The first reported mismatch was:

`draft.layers.0.self_attn.q_proj.weight`

The ranking term was disabled and no frozen-reference forward was run in either
trial. Execution stopped automatically at this failed check. P0, U1 and P1 were
not attempted. No tolerance, learning setting or algorithm was changed; no retry
or full campaign followed.

## Evidence that does and does not agree

Both trials started through the fresh-trainer path with the unchanged sealed
configuration and seed 17. The retained presentation/anchor guard passed for
both responses; all recorded scalar metrics agree exactly between O and U0.

| Measurement | O | U, lambda 0 |
|---|---:|---:|
| First response | train_00072 | train_00072 |
| First sampled anchors | 363 | 363 |
| First response loss | 1.0308470726013184 | 1.0308470726013184 |
| Second response | train_00121 | train_00121 |
| Second sampled anchors | 411 | 411 |
| Reported two-response accumulated loss | 1.1119227409362793 | 1.1119227409362793 |
| Accumulated supervised-position exposure | 11,400 | 11,400 |
| Reported gradient norm | 0.85546875 | 0.85546875 |
| Post-step draft-weight equality | Reference | **Failed** |

The second-row metrics follow the existing accumulation convention: they include
both response presentations. Do not add the first response's exposure again.
The report also retains the distinct author-weighted objective and accuracy in
[the presentation CSV](QUALIFICATION_PRESENTATIONS.csv).

Equal losses and a rounded scalar gradient norm do **not** imply coordinatewise
gradient or parameter equality. The evidence localizes the detected discrepancy
to the post-update state comparison, but does not establish whether it arose in
backward accumulation, optimizer execution, initialization/state isolation, or
another implementation detail. No numerical-cause claim is supported yet.

The state comparator stops at its first discrepancy. It therefore did not finish
the post-step Adam/RNG comparison. The failing U0 tensors were not serialized
before the exception, so the mismatch magnitude and coordinate count cannot be
recovered from this run. The disposable O state remains on A800; the exact
responses, features, anchors, scalar metrics, source and failure traceback are
preserved. This retention limitation is explicit rather than replaced by an
inferred error size.

## Work performed

| Work | Planned | Executed |
|---|---:|---:|
| Qualification current forward/backward presentations | 10 | 4 |
| Qualification reference forwards | 4 | 0 |
| Extra detached comparison forwards | 4 | 0 |
| Temporary optimizer steps | 5 | 2 |
| New main optimizer steps | 256 | **0** |
| New evaluation generations | 768 | **0** |
| New feedback generations / teacher-model forwards | 0 / 0 | 0 / 0 |

The four top-level current forwards involved 20 draft-layer forwards including
checkpoint recomputation and 784 vocabulary-head forwards including teacher
projection and recomputation. These are internal work counts, not extra response
generations. All temporary updates were qualification-only and were never
exported or published for serving. The O diagnostic state is retained as failure
evidence, not an available candidate checkpoint.

## Checks and costs

- All **396 installed A800 CPU tests passed**. The prior local full suite passed
  393 tests, and the final 27 focused tests include the three added runner tests.
- Independent source review found no remaining blocker after fixing exact model
  filename membership in preflight. CPU success did not establish native GPU
  compatibility; the GPU gate correctly stopped the run.
- Original configuration, model/runtime pins, 128 prepared features, existing O
  export and optimizer state passed preflight. Final content rechecking confirms
  **all 132 retained ordinary-run artifacts remain unchanged**.
- Qualification wall time: **79.513 s**. Main process including checks/cleanup:
  **80.551 s**. Supervisor including preflight: **114 s**.
- O loading: 22.447 s; two training presentations: 8.036 s. U0 loading: 18.722 s;
  two presentations: 3.793 s. These are qualification costs, not an efficacy or
  steady-state training-speed comparison.
- Recorded O peak CUDA allocated memory: **16,564,648,960 bytes** (15.43 GiB).
  U0's peak field was not written before the failed state check; active-ranking
  full-size memory feasibility remains unqualified.
- Peak storage admission required 42.56 GB including the 8 GiB reserve, against
  403.31 GB free. After preserving failure evidence, about **397.02 GB** remains.
  Both A800 GPUs are free. No cleanup, deletion, new feedback or O retraining.

## What this means for the requested comparison

There are **no new P-versus-O, P-versus-U or P-versus-F performance results**.
No per-request timing-repeat records exist for this attempt because evaluation
never started. The completed historical F/O results remain intact and are not
relabelled as new measurements.

Before resuming, the narrow unresolved question is why an objective-disabled U
trial differs from ordinary O after the same step. A future bounded diagnostic
could compare repeat O/O and O/U0 runs while retaining per-parameter discrepancy
statistics and the actual pre/post states. That is a proposed investigation,
**not an automatically authorized rerun**. No ranking change or broader campaign
is justified by this failure.

## Reproducible artifacts

- [GPU qualification result and traceback](raw/campaign_001/qualification/RESULT.json)
- [Every qualification work boundary](raw/campaign_001/qualification/WORK.jsonl)
- [Per-presentation measurements](QUALIFICATION_PRESENTATIONS.csv)
- [Main execution status](raw/campaign_001/STATUS.json) and [costs](raw/campaign_001/COSTS.json)
- [A800 CPU test log](CPU_ALL_A800.txt), [preflight](PREFLIGHT.json),
  [source/configuration seal](SEAL.json), [independent code review](INDEPENDENT_REVIEW.md)
- [Retained-artifact final audit](FINAL_RETENTION_AUDIT.json)
- [Independent compact-result audit](INDEPENDENT_AUDIT.json): incomplete evidence
  is consistent, with zero contradictions; this does not mark qualification as passed.
- Remote complete evidence: `/data/sources/minhao/runs/dflash_prefix_ranking_one_event_20261025`.
- Production implementation: `/data/sources/minhao/DFlash-Online/online_feedback/`;
  thin experiment driver: `experiments/prefix_ranking/`.

Compact raw records are copied to Fenrir. The roughly 6.3 GB disposable O state
remains only on A800 at `campaign_001/qualification/disposable_O_state.pt`.
Fenrir DFlash-Research was not modified. All unrelated campaigns remain paused.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_one_event_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
