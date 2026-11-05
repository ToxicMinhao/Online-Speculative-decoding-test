# KRC events 3–4 trainer integration

Material Passport: ARS academic-research-suite 3.24.0 / experiment-agent run and reproducibility validation; bounded additive implementation and CPU qualification by the implementation agent. This is a same-team self-review, not an independent review or a native experiment. Research remains Stage 1 / project None.

The additive trainer is ready for the campaign's native restoration and correctness gates. Seven new CPU tests pass, together with eighteen relevant inherited tests. No inherited scientific implementation was edited, no Git commit was made by this agent, and no SSH, CUDA initialization, GPU work, feedback generation or serving generation was performed.

## Source and authorized scope

The user-adopted [handoff](HANDOFF.txt) fixes complete KRC2 → event 3 → event 4, with the existing K objective and full-frequency reference correction. Relevant source evidence was read in the retained KRC2 report/review and O/K continuation report; the actual event-2 external receipt is `research/dflash_krc_continuation_20261028/raw/campaign_001/KRC2/events/event_002/RESULT.json`. The inherited checkpoint serializer stores each event before its external `status="completed"` receipt is assembled, so saved history is validated using counters, objective, reduction, pass and record/feature counts rather than inventing a saved status key.

Repository base during the CPU qualification was `f4f0e7a23cfb8e50bd58648c5260924769594e14`. Two additive untracked files were tested under `research/dflash_online_publication_20261028/github_checkout/DFlash-Online/`:

| File | SHA-256 at qualification |
| --- | --- |
| `online_feedback/krc_late_continuation.py` | `76f3b4886567b557076e111a3dce145cc13a731b8fe4f214445a339308d442ea` |
| `tests/test_krc_late_continuation.py` | `01bf305d8aca1f1cac67a2031c9fa92839530c327ae5fa7f9bab09677c8cce62` |

The campaign must record its actual later execution commit and source/config seals. The publication base above is not an execution-time identity for historical experiments. An explicit `git diff --exit-code` check found no modifications to inherited `training.py`, `krc_continuation.py`, `reference_correction_event.py`, `prefix_correction_event.py`, `prefix_correction.py` or `reference_schedule.py`.

## Integration and API

```python
from online_feedback.krc_late_continuation import (
    KRCLateContinuationTrainer,
    KRCLateContinuationEvent,
)

trainer = KRCLateContinuationTrainer(config_path, fresh_output_root,
                                     resume_state=qualified_krc2_full_state)
# The campaign verifies exact live restoration against the bound KRC2 state/export here.
for number in (3, 4):
    with KRCLateContinuationEvent(
        trainer, schedule=reference_schedules[number],
        feature_paths=feature_maps[number], output_dir=window_output_dirs[number],
    ) as manager:
        receipt = trainer.train_event(
            f'event_{number:03d}', feature_paths[number],
            presentation_schedule=retained_ordinary_receipts[number],
        )
    correction_report = manager.report()
trainer.close()
```

`KRCLateContinuationTrainer` rebuilds the exact no-probe KRC2 contract, then calls the inherited strict state loader. It accepts only a fresh output directory, complete global step 256 / presentation 512 state, two saved completed-event boundaries, accumulation 2 and the inherited 7,424-step horizon. K2, KRC1's probe contract, correction-removal contracts, missing complete state, shortened configuration and malformed history are rejected. It then permits only event 3 at 256→384 and event 4 at 384→512, each with 128 responses and two passes, inside the active controller.

`KRCLateContinuationEvent` calls the qualified reference controller with an empty probe schedule. It inherits ordinary K backward/Adam handling, reference capture/replay, correction arithmetic and integrity isolation. Only event start/stop coordinates and the first-window counter offset differ from the old bounded event-2 controller; the old file and its guards remain intact. First-window checks inspect already completed work and add no model forward or optimizer step. The controller is closed between events; the trainer, optimizer and natural RNG remain alive.

The retained ordinary receipt remains an assertion on natural shuffle and anchor draws. Additional integration validation binds its cumulative counters to the current event and each learning/reference pair to the next disjoint pair in the same pass, including cyclic wrap. An anchor mismatch fails before that ordinary forward/backward; it is never repaired by forcing recorded draws or another arm's RNG. Optional `reference_check` work is rejected.

Reports retain the existing continuation keys: `counts`, `work`, `costs`, `steps`, `probes`, `first_window`, `diagnostic_migration`, `start_step` and `native_global_step`. The diagnostic migration now has `changed_fields=[]`, because probes are already disabled in KRC2. Each caller-specified window directory receives `CONTINUATION_CONTRACT.json`, `FIRST_WINDOW.json` and `window_001.json` through `window_128.json` in a full event. No new diagnostic forwards are introduced.

## Persistent KRC3 and KRC4 states

The inherited serializer writes a fresh pending file and atomically replaces `training_state.pt`; it does not mutate the previous inode. After each event succeeds, the additive wrapper creates a hard link at:

- `<fresh_output_root>/events/event_003/training_state.pt`
- `<fresh_output_root>/events/event_004/training_state.pt`

The original per-event serving exports remain at their existing `events/event_00N/export` paths. External `RESULT.json` and the returned receipt set `full_state_path` to the immutable event state, add `latest_state_path` for the shared latest slot, and record `costs.state_retention_s`. Saving/linking introduces no RNG draws or model work. Both complete states retain inherited config/history and BF16 optimizer state. The saved event payload is unchanged; only the later external receipt includes the immutable path.

Because latest aliases the newest event state, this retains two full-state inodes, not three. There is no extra full-state copy beyond the campaign's stated budget. A retention failure marks the trainer failed and writes `STATE_RETENTION_FAILED.json`; it cannot continue to event 4.

## CPU evidence

Final command, from the DFlash-Online source root:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/home/minhao2/EAGLE_SPEC/research/dflash_online_publication_20261028/github_checkout/DFlash-Online:/home/minhao2/EAGLE_SPEC/research/dflash_online_publication_20261028/github_checkout/DFlash-Online/training:/home/minhao2/EAGLE_SPEC/research/dflash_online_publication_20261028/github_checkout/DFlash-Online/inference:/home/minhao2/EAGLE_SPEC/research/dflash_online_publication_20261028/github_checkout/DFlash-Online/tests \
OMP_NUM_THREADS=1 /home/minhao2/.venvs/dflash-specforge/bin/python -m unittest \
  test_krc_late_continuation test_krc_continuation \
  test_reference_correction_event test_ranking_schedule -v
```

Result: **25 tests passed in 7.162 seconds**, exit 0. The existing environment warns that FlashAttention is absent locally; these tests intentionally use the existing tiny eager-attention CPU fixture.

The real-trainer fixture keeps BF16 Adam/moments, current K masks, natural shuffling/anchors, configuration, serialization and complete restart. Only model scale, distributed wrapping and export are replaced by inherited CPU fixtures. Four updates per event keep tests bounded. After exact KRC2 live restoration, both event 3 and event 4 exactly match uninterrupted execution in weights, Adam state, RNG, counters, anchors and loss metrics. Event-3 hard-link bytes remain hash-identical after event-4 publication; its saved counters remain 12/24 while event 4 reaches 16/32 in the shortened fixture. The production constants remain 128 updates per event and cumulative 256→384→512.

Controlled window tests cover applied corrections followed by empty protection, fresh consecutive controllers, optimizer identity, restored hooks, event-local/global coordinates, zero optional probes, and first-window failure without extra work. Negative tests cover contract/history/configuration/counter mismatch, missing schedule/controller, invalid reference binding and altered natural anchor observations. All six inherited event-2 continuation tests also pass.

Two preparation-only test-command issues were corrected and are disclosed: the first seven-test command had one overly specific expected horizon-error string (the inherited loader rejected the malformed configuration earlier); a subsequent command included a nonexistent `test_reference_schedule` module (the nineteen actual tests in that command passed). The final command above uses the existing `test_ranking_schedule` module and a consistent shortened-recipe fixture that reaches the new horizon guard. No production algorithm change was needed to remedy either issue.

## Limits and remaining execution gates

CPU exactness does not establish native FlexAttention or model-scale CUDA equivalence. Campaign orchestration must still bind the exact retained KRC2 state to its export, verify complete live restoration, validate all retained event-3/4 pools and exclusions, record native first-window evidence, audit both full states/exports, and enforce storage admission. Training/evaluation performance is unmeasured by this implementation task. The authorization stops after event-4 evaluation/report; this code permits no further event or alternative training arm.

Execution source commit: `11b7718a04416a89025a6df69c0c33e558e5fe1b` (committed before launch; no inherited scientific files changed).
