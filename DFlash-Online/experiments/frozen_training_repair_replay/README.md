# Frozen event-seven training-repair replay

Read-only diagnostic of the existing PR44 NATIVE7 and RECENT7 exports, both at Adam step896. No training, optimizer, backward, teacher-head projection or response generation occurs.

The original815 tags are retained across1629 eligible presentations (814 in pass0,815 in pass1). Each endpoint replays the complete original256 response forwards and8170 eight-block vocabulary chunks. The parent K6 RNG/configuration define natural sampling; only the strict-loaded draft endpoint changes. Endpoint tensors, mode and shared embedding/head are checked before and after scoring. Raw token-bearing observations remain private.

## Reproduction

In the qualified A800 training environment, source the existing environment helper and use the pinned execution checkout. Set `PYTHONPATH=.:training`, `CUDA_VISIBLE_DEVICES=0`, the qualified compiler and offline caches as recorded in `run.sh`. Then run:

```bash
python -m experiments.frozen_training_repair_replay.campaign --spec /path/to/private/RUN_SPEC.json
```

The private run specification contains `previous_root` (authenticated PR44 execution), `public_dir` (PR44 scalar identity tables), a new `output` directory, `execution_commit`, `source_manifest`, and `prior_pins` (hashes of retained original plan/seal/endpoint-result receipts). Preparation authenticates the archived source/features/assignments/parent rows and creates both endpoint plans and the exact row cohort before scoring. Existing output directories fail rather than being overwritten.

The scalar report pairs all1629 occurrence keys, averages each slot over its eligible presentations, then weights the815 slots equally. The main interval resamples128 complete requestIDs10000times using PCG64 seed1707 and a resampled-slot denominator. Integer occurrence counts and each pass are separate checks. No serving gain follows from fitted-anchor retention alone.

Run the focused CPU tests with `python -m unittest discover -s tests -p 'test_frozen_training_replay*.py'`. The tests do not instantiate a GPU model or optimizer.
