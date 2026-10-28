# DFlash-Online experiment consolidation and audit

ARS experiment validation, October 22, 2026. Authorized scope: consolidate the
existing screen inside A800 DFlash-Online, simplify its orchestration and audit
the executed comparison. No new scientific generation, model training, tuning or
campaign restart occurred. Fenrir DFlash-Research remains unchanged.

## Maintained code

The canonical experiment package is on **A800**:
`/data/sources/minhao/DFlash-Online/experiments/projected_target/`.

| File | Responsibility |
|---|---|
| `run.py` | Frozen generation → shared A feedback → independent C/P updates → A/B comparisons. Begin reading at `run()` line 128 or `main()` line 236. |
| `checks.py` | Existing protocol checks, exposure accounting, reference qualification and screen-specific storage/retention policy. |
| `config.yaml` | Byte-identical pinned eight-step training recipe. |
| `report.py` | Read saved measurements and write tables; optional separate output directory preserves historical reports. |
| `README.md` | Procedure, entry points, boundaries and current failed/incomplete status. |

The existing `online_feedback` backend, teacher replay, objectives, optimizer,
publication and RNG handling remain the implementation. The screen calls those
components directly because it shares one frozen feedback pool between two fresh
learners; the persistent `OnlineController` buffer serves a different sequencing
need. There is no second inference or training implementation.

The main entry is 245 lines, versus 445 in the historical runner. This is a
reading-path improvement, not a claim that moving checks removes their complexity.
The reporter and most checks were relocated, not newly invented. The one-off
failed-preparation recovery option is absent from the maintained entry; its exact
historical source remains archived.

Only `online_feedback/training.py` changed within the shared core: it logs the
actual pre-step LR and permits an explicit qualification-only reference observer
before/after training. Normal calls leave the observer unset. This removes the
experiment's runtime replacement of `optimizer.step` and objective lifecycle
methods. Reference cleanup still runs if a check fails. No objective, sampling,
reduction, precision, optimizer update or publication setting changed.

The reporter additionally checks stop-token equality, matching the execution
gate. No previously retained result changes under that stricter check.

## Verification and evidence

- **133 A800 CPU tests pass** in the pinned environment, with CUDA hidden and
  offline libraries. Tests cover unchanged optimizer/weights/RNG with reference
  observation, cleanup on failed checks, actual pre-scheduler LR logging,
  protocol gates and output validity. The first test attempt exposed a new test
  stub that omitted the author's `loss_denominator` argument; accepting and
  forwarding that argument corrected the stub. No production update was changed
  to obtain the pass.
- Module entry `python -m experiments.projected_target.run --help` imports and
  exposes the expected arguments without constructing models.
- The moved reporter rebuilt `SUMMARY.json`, `REPORT.md`, `PER_REQUEST.csv`,
  `POOLED.csv` and `COSTS.json` in a separate directory. All five are byte-identical
  to the historical reports; original files remain unchanged.
- Independent code review found no production blocker. All nine relocated
  protocol/storage helpers preserve their original ASTs. All pre-edit core files
  match the exact executed archive; only the shared trainer differs afterward.
- [Independent protocol audit](../CONSOLIDATION_PROTOCOL_AUDIT.md) inspected all
  136 native traces, six training receipts, 24 compact mask records, C/P anchors,
  exposure, schedules, initial/final RNG and source bindings. No additional
  label/reference/matching/B-leakage discrepancy was identified.

These checks do **not** constitute a new GPU qualification of the consolidated
driver or resolve the historical native-output mismatch.

## What is and is not established

GSM8K and CodeAlpaca supply **96 qualified records**. Across their 16 B requests,
verification calls F/C/P are **901/1023/987**: P reduces harm versus C, while
remaining worse than frozen F. ShareGPT stops at P A07's last cap token, after
136/144 generations overall; its entire group is excluded from gain claims.
All six updates finished, totaling 96 response presentations and 48 optimizer
steps. The strict mismatch, eight unexecuted P-B requests and ShareGPT P restart
state remain preserved. The campaign remains stopped.

This is the current Qwen3/DFlash-b16 fixed-pool **hard-CE/projected-target** screen,
not the historical AdaFlash forward-KL recipe or a continual per-turn experiment.
Teacher replay checks every saved target argmax ID; the field named
`full_posterior_match` does not establish full probability-vector equality.
Single instrumented latencies do not establish serving speedup.

One pre-existing accounting limit remains explicit: if the native backend fails
before returning, completion of that execution may be unknown. Pairing failures
after return consume the recorded budget. Both stop the run, with no automatic
retry. This limit did not affect the audited 136 saved executions.

## Storage and retained source

A800 free space before source work was 13,477,670,912 bytes; after verification it
was 13,475,332,096 bytes. The source/test allowance was 64 MiB plus an 8 GiB reserve;
no deletion was needed. This does not admit another full GPU campaign.

Remote operation evidence and source backups:
`/data/sources/minhao/runs/projected_target_feasibility_20261022/consolidation_001/`.
Fenrir holds review archives and reports only; the old `runner.py` and execution
archives remain historical evidence, not the maintained entry point.

- [Final CPU results](TESTS_FINAL.log)
- [Report reproduction](REPORT_REPRODUCTION.json)
- [Source comparison](SOURCE_COMPARISON.json)
- [Shared-trainer diff](training.diff)
- [Final review-only source snapshot](final_source.zip)
- [Before-edit backup](before.zip)
- [Original result and limitations](../RESULTS.md)

The formal research stage and scientific settings are unchanged. No mechanism
claim or campaign restart follows from this code/evidence audit.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
