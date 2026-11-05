## Material Passport

ARS-Codex 3.24.0, experiment-agent implementation/validation within the authorized KRC events 3–4 continuation. This review covers CPU integration fixtures only. It does not report model generations or a scientific result. Stage 1/project None remains unchanged.

The additive O4/K4/KRC4 evaluator is ready for the campaign's native qualification. All **13 focused CPU tests pass**. The fixture includes the entire **1,152-generation schedule**, with the first eight development requests providing the **72-generation correctness gate inside that budget**. No SSH, GPU, training, checkpoint mutation, or Git commit was performed by this evaluator task.

Source at review: branch `codex/krc-events-3-4-20261028`, base Git revision `f4f0e7a23cfb8e50bd58648c5260924769594e14`, with the following new uncommitted files. The base revision alone does **not** contain this implementation. The campaign must record its actual execution commit and source bindings before launch.

| File under DFlash-Online | SHA-256 |
|---|---|
| `experiments/krc_events_3_4/__init__.py` | `e579252ec43588e261566f612069ddf44622ad215e5b23fc170ec27bf260ca4c` |
| `experiments/krc_events_3_4/evaluate.py` | `5ecf7f5ed66c5411d69b1d229c605b79120b95bb85a5fe10d907e93cf7b1b00d` |
| `experiments/krc_events_3_4/evaluation_protocol.py` | `68ae90097fee1298547d1301e9b64afaf97961fcac3d3d0e3bd26c3e4cb0aa39` |
| `experiments/krc_events_3_4/test_evaluate.py` | `9b9746c42f96439366ab92f8d0c2743d6a15b97c8306e8f72b4ce9e218f9d6f9` |

The qualified native backend, inference runtime, trainer, publication checks, exact tensor checks, canonical output/trace validation, scalar/repeat validation, scoring, atomic measurement storage and execution accounting are reused. No inherited scientific source was edited. The new code supplies the three-arm loop, panel contracts and reporting; campaign/training orchestration is owned separately.

`evaluate(backend, requests, checkpoints, root, *, expected_control_counts=None, qualification_requests=8, historical_counts=None)` takes exact `O4`, `K4`, `KRC4` checkpoint keys. Development supplies `expected_control_counts={'O4': {request_id: calls}, 'K4': {request_id: calls}}`; additional supplies `None`. The campaign-level protocol requires exactly 64 requests per panel, development followed by additional, F versus K1 canonical provenance respectively, distinct request identities/prompts, and exact historical maps: development O1/K1/KRC1/O2/K2/KRC2; additional K1/KRC1/K2/KRC2. O1/O2 remain absent on additional. This validator intentionally owns evaluation fields only; the campaign separately binds all continuation artifacts and training counters.

The arm order is `('O4', 'K4', 'KRC4')` shifted by `(panel_ordinal + repeat) % 3`. Every request starts with each arm once and places each arm in each position once across three repeats. With 64 requests, the per-repeat first-position counts are necessarily a rotation of **22/21/21**, while all 192 request/repeat combinations give exactly **64 starts per arm**. The protocol receipt discloses this; it does not claim exact per-repeat equality.

The native session identifier is `krc_events_3_4_evaluation`. Raw records retain labels such as `0000_r0_O4.json`; every returned record receives validation evidence. An interrupted call remains an unknown attempted generation. Invalid returned generations, unattempted slots, publication/weight failures, nonfinite values and malformed records remain visible rather than becoming successful measurements. A valid harmful KRC result completes the declared schedule.

Reports reopen native records and recheck their identities, canonical tokens/termination, verification trajectories, counts and timing fields. All three comparisons use one common cohort of complete valid requests. Calls sum repeat 0 only after all three repeats agree; timings sum each request's three-repeat mean. O4 is the primary baseline (`KRC4_vs_O4`); `K4_vs_O4` and `KRC4_vs_K4` are separately reported. Gross saved/added calls and request directions use absolute paired differences. Ratios use pooled totals. Historical values provide event-1/2/4 count curves and changes in relative gaps, never count gates for additional or historical latency comparisons.

Artifacts include `PROTOCOL.json`, `EXPECTED_CONTROL_COUNTS.json`, `HISTORICAL_COUNTS.json`, `REQUESTS.json`, `CANONICAL.json`, `STATUS.json`, `QUALIFICATION.json` for development, activation/weight receipts, raw records, `MEASUREMENTS.jsonl`, validation receipts, and `report/{SUMMARY.json,PER_REQUEST.csv,ALL_REPEATS.csv,REPEAT_TOTALS.csv,EVENT_COUNT_CURVE.csv,REPORT.md}`. Request latency includes native generation and cache maintenance as nested components. Activation and tensor-check cost are separate. Backend loading, training and export remain campaign costs.

Validation command, run from the local `DFlash-Online` checkout:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:training:inference:tests /home/minhao2/.venvs/dflash-specforge/bin/python -m unittest experiments.krc_events_3_4.test_evaluate -v
```

Result: **13 tests, 7.881 s, OK**. An earlier 11-test iteration also passed; two added tests then explicitly exercised publication/weight failures before generation and the reused cache/capture contracts. `git diff --check` passes. Import isolation confirms the evaluator imports neither Torch nor `online_feedback.training` and does not initialize CUDA.

Remaining execution gates are real artifact/source bindings, complete-state continuation validity, storage admission, the native included gate and the fresh campaign itself. The CPU fixtures do not replace those gates. KRC4 versus K4 measures the complete correction history, since their event-2 starting states already differ; neither this implementation nor its tests isolates only event-3/4 corrections or establishes training-inclusive deployment speedup.

Execution source commit: `11b7718a04416a89025a6df69c0c33e558e5fe1b` (committed before launch; no inherited scientific files changed).
