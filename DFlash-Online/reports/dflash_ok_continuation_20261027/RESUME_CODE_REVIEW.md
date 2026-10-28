## Material Passport

ARS experiment-agent, local source/restart feasibility review and narrow CPU implementation. Stage1/project None unchanged. Source analysis is **ANALYZED**; the new helper fixtures are **CPU VERIFIED**. No remote operations, CUDA initialization, scientific forward/backward, or optimizer updates were performed by this reviewer. The parent owns actual A800 snapshot verification and the authorized continuation launch.

## Outcome

The existing trainer supports genuine continuation from each arm's own full state. Preserve its exact saved configuration and7424-step horizon, restore into new output directories, and enforce the512-step stopping point in the new campaign. Do not use the prior fresh-event startup wrapper or repeat event1. Source review found no required warmup; the authorized zero-extra-step budget excludes the old disposable warmup/qualification windows.

The parent reports that both actual A800 event1 snapshots exist at about6.292GB each, with128 cumulative steps,256 presentations, event_001, accumulation2, horizon7424, identical config text and all four RNG components equal across O/K. That report is supporting operational evidence supplied by the parent, not an independent remote inspection by this reviewer. Exact production shape/optimizer/config compatibility must still pass the live gates below.

## Existing behavior and constraints

- [Trainer construction and restore](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/online_feedback/training.py:176) builds the selected runtime, then calls `_restore` after construction. `_restore` at396 rejects changed schema, exact config text, reduction, algorithm contract, horizon or accumulation before loading draft weights, backend Adam, full RNG, cumulative counters and event history. It restores `core._micro` to the presentation count. Therefore construction's seed17 call does not replace the saved continuation RNG.
- [Checkpoint payload](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/online_feedback/training.py:543) includes exporter-compatible draft weights, full optimizer payload, Python/NumPy/TorchCPU/CUDA RNG, steps, presentations, events, config and algorithm contract. [Direct BF16 AdamW loader](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/online_feedback/optimizer.py:61) validates recipe, parameter grouping and BF16 moment shape/dtype before loading. The backend's loader receives only the optimizer payload; the event adapter restores draft and full RNG separately.
- [Event admission](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/online_feedback/training.py:438) rejects duplicate event IDs, incomplete accumulation and work beyond the inherited horizon. It does not implement this experiment's512-step stop, so the campaign must bound event numbers to2/3/4 and independently check cumulative128→256→384→512 and presentations256→512→768→1024. Changing YAML horizon/max_steps/output_dir/run_id would fail exact config-text compatibility; the output-root constructor argument supplies the new destination without editing YAML.
- [Normal sampling](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/online_feedback/training.py:494) draws one TorchCPU permutation per128-response pass. [Anchor sampling](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/training/specforge/algorithms/common/dflash_family_model.py:344) uses `torch.rand` on the training device and does not depend on O/K predictions. Each arm must restore its own entire RNG state. Shared draws can be asserted because the actual parent-reported Torch states agree; never replace K's RNG with O's to force a match.
- [PresentationSchedule](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/online_feedback/presentation_schedule.py:7) asserts normal response order and anchors against a completed O result, including exact feature metadata/paths; it does not replace or resample anchors. Use the same immutable shared feature paths for both arms. Pass O event-N's RESULT to K event-N, not event1's receipt. Current O/K `begin_event` only resets cost counters and creates no reference, and [Export](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/online_feedback/training.py:564) preserves RNG.
- [Old first-error train_arm](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/worktree/experiments/first_error/campaign.py:72) is explicitly unsuitable: it creates fresh state, calls fresh-state preparation, requires zero counters and trains event_001. Its FirstErrorStartup inherits the discarded-warmup/fresh-pinned-RNG logic. None of that should enter continuation.
- The historical lambda-zero diagnostic associated a first-versus-later runtime difference with initial gradient differences but did not identify its cause ([source report](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_gpu_diagnostic_20261026/REPORT.md)). This does not establish that continuation needs extra warmup or that an exact restored state guarantees a bitwise GPU trajectory across process histories. Use a consistent declared runtime-construction policy for both arms; separate fresh training processes per arm/event avoid making only the later arm inherit an earlier arm's live initialization. Do not repair this historical limitation with unbudgeted updates, reseeding or changed tolerances.

## Minimal continuation API implemented

Only `worktree/experiments/ok_continuation/resume.py` and `test_resume.py` were added by this reviewer. The helper imports Torch only when called and never constructs a training runtime or performs scientific work.

```python
state, snapshot_receipt = load_snapshot(
    own_state_path, own_export_path, objective=objective,
    config_text=original_config_text, completed_event=event_number - 1,
)  # Defaults: horizon=7424, expected_cuda_rng_states=1.
trainer = SpecForgeEventTrainer(
    original_config_path, new_arm_output,
    resume_state=own_state_path, objective=objective,
    ranking_lambda=0.0, reduction='response_mean',
)
verify_live_resume(trainer, state, receipt_path=initial_gate_path)
del state
result = trainer.train_event(
    f'event_{event_number:03d}', shared_feature_paths, passes=2,
    presentation_schedule=ordinary_result if arm == 'K' else None,
)
validate_event_receipt(
    result, event_number=event_number, objective=objective,
    reference_schedule=ordinary_result if arm == 'K' else None,
    trainer_status=trainer.status(),
)
```

Persist the snapshot receipt separately, close the trainer on success/failure, and do not overwrite the historical state paths. Readonly startup snapshot loading must refer to the exact file passed to `resume_state`. Collect each128-response pool once with the previous O checkpoint before constructing/restoring training, so generation/setup cannot consume the next training RNG stream. Restore each next arm/event from its own last committed state, even if intervening serving or the other arm changed process-global RNG. Both arms finish an event before preparing the next O-generated pool.

`validate_snapshot` requires all58 expected BF16 draft tensors, all58 initialized Adam entries, matching parameter names/shapes/dtypes, finite moments, correct per-parameter Adam step, original optimizer options, complete event history and all RNG components. Python/NumPy/TorchCPU payloads are checked with isolated generators without touching global state; production requires one saved CUDA state. `load_snapshot` also compares every export tensor to the saved draft at native bit precision. Its CPU load does not mmap a file later replaced by training.

`verify_live_resume` compares live weights, Adam, complete RNG, counters, history, config, contract, horizon and reduction exactly, and requires zero residual gradients/accumulation and no reference/ranking state. It writes a compact failed receipt before raising if a receipt path is supplied; it never restores/reseeds to conceal a mismatch. `validate_event_receipt` checks128 distinct features, two complete permutations,256 presentations,128 correctly placed optimizer boundaries and one anchor observation per presentation; K optionally must match the new O schedule exactly. Event1 is accepted only for validating retained history, not as a continuation schedule target.

## CPU evidence and limits

Six helper tests pass in1.202s using `/home/minhao2/.venvs/dflash-specforge/bin/python`, CUDA hidden and the continuation worktree/tests plus local SpecForge/inference paths. Tiny fixtures retain all58 parameter identities and exercise the actual `_restore` function with real BF16 Adam objects for both O/K; no optimizer step is executed. Coverage includes missing/extra tensors, missing/incorrect moments or Adam steps, altered config/horizon/counters/RNG, nonfinite weights, export mismatch, exact live RNG/gradient failures with durable receipts, events2–4 schedules and native signed-zero bit differences. CPU fixtures explicitly request zero CUDA RNG entries; production defaults to one. The evaluator author independently reviewed these helpers and found no blocker.

I also independently reviewed the new retained-F evaluator/protocol/report: all14 CPU tests pass in2.632s. K-first outputs validate immediately against retained F, evolving O counts are permitted, reports use one complete common-valid cohort and ratios of pooled calls/summed per-request mean times, and historical F contributes no latency. These fixtures do not establish production checkpoint integrity or native execution correctness; those remain sealed/live gates owned by the parent.

## Helper source identities

- `worktree/experiments/ok_continuation/resume.py`: `4b5689444a423b189dbf2e2367cd5e25ddaf4e6874ae704dabfaa5e3161d70b6`
- `worktree/experiments/ok_continuation/test_resume.py`: `81851cd449e2d8822c57d2ad84df9b6a4d9226b77ab71b41ef3cb0eee0c5242e`

## Final parent integration review

No unresolved integration blocker found in `campaign.py`, `preflight.py`, `seal_remote.py` and `run_campaign.sh`. The copied YAML path can differ because the loader compares exact text; model/configuration references inside that text remain unchanged. The campaign uses own committed states, fresh runtime objects and exact live gates for every arm/event; it binds K to O's new normal schedule and checks shared feature hashes before/after each update. Both trainers close before the event2/event4 evaluation calls. The zero-extra-warmup budget is preserved. Per-phase storage admission is conditional on actual accumulated occupancy and stops before an unsafe next phase; it is not an unconditional all-maximum-length completion guarantee.

Two seal findings were corrected before launch: prepared windows are now bound to the completed selection audit's output hashes and ordered IDs, and both original parent RESULT receipts are checked against the previous seal before their paths are trusted. The O full restart digest additionally matches its historical retained receipt. K's full-state digest is explicitly newly captured from the original completed path; no earlier full-state digest is claimed. Its saved draft must match the previously sealed K export. Exact model/export file sets and immutable source hashes remain pre/postflight gates. Loading CUDA peaks are now recorded separately from training peaks, with explicit scopes.

The first independent parent test run passed14/15 and failed only because the new seal fixture omitted SELECTION_AUDIT.json after these stronger checks were added (`test_campaign.py` then413 -> `seal_remote.py`31). This was a fixture failure, not a scientific run or a relaxed production gate. After adding the required fixture evidence, the final independent15/15 parent tests pass in1.321s. Alongside6 helper and14 evaluator tests, this review independently observed35 passing CPU cases. A separate parent invocation used system Python3.13 without Torch; the correct pinned local interpreter is recorded above. No failed scientific run or retry is implied by either CPU invocation issue.

| Final orchestration/evaluator file | SHA256 |
|---|---|
| `campaign.py` | `59fd4067e14e8d23e164403a6bf2064c56989ecf0651de700fc13ddef01b9c66` |
| `preflight.py` | `15880ffffcbb7488340177c27ff8f72f862dd2f828eb33377ead1b96124dd0c2` |
| `seal_remote.py` | `9b12f07ac26fd8c765699b135f6fbe31bb6c5b7bafe46416b200c68d1ea7d1f3` |
| `run_campaign.sh` | `95e399b5e16d22370f81cb300fb9b8a1564766a7fe1a93dfe24152c177939792` |
| `supervise.sh` | `c2e88aab4b42a2e46638a4d95958640872b01453b595e76d04ded96511fdc566` |
| `test_campaign.py` | `a6f94cc6c1dd07faf44d9b5508c059761e77e2ee7e9c63f6fed60372ee1908b5` |
| `worktree/experiments/ok_continuation/protocol.py` | `71c3590fa83755fbfee1e4700bb091f9fd1a3e5f14514149e7eed26cd71a0a99` |
| `worktree/experiments/ok_continuation/evaluate.py` | `b5204f59b0a0f280b52c41b8108fd9a2a359d9c6c3faf008d8fc0411561cd9d2` |
| `worktree/experiments/ok_continuation/report.py` | `968cade13df3ba766ba5b459e3f0bc2b7a912f329c46bb7d325e99971cd25da7` |
| `worktree/experiments/ok_continuation/test_evaluate.py` | `b0083315118a263c56ca375f5487a7bc4ad05284382c6fb41bea80bfe5e9baa2` |

## Final parent integration correction before GPU execution

The parent found a concrete saved/live container mismatch missed by the initial fixture: upstream checkpoint_state_filter saves draft weights as a plain dict, while live state_dict returns OrderedDict. The gate compared Python types before checking tensor contents. Live draft mapping is now converted to dict before exact key/dtype/shape/bitwise tensor comparison. The fixture now uses the real save format, and a new regression proves equal content passes while a changed tensor fails durably. All7 resume tests pass. This changes only the validation representation, not learned weights, precision, tolerance or optimizer behavior. Previous helper hashes are superseded by resume.py297c1b5fffd9d3211b2fd49d0f2739209880e6ffac87f1d7e3e97d0b54215c80 and test_resume.py03570244dc2f73acbe35933e6b13d58799baab727762d19958d68b4845547728. No GPU run had started.

The installed CPU test initially exposed a fixture-only double mapping under A800’s approved TMPDIR. The path mapper now treats its own sandbox as already mapped. Original failure log is retained; no production path or scientific change was needed.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
