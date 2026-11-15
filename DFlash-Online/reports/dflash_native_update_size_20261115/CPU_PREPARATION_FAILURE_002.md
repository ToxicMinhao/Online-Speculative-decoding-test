# CPU preparation inventory failure and repair

Execution checkout: `7a784a7e9774a6802a6fabc70da7ba091f4e9058`.
CPU preparation began 2026-11-15 22:02:19 UTC and ended 22:03:35 UTC (76 seconds).
It stopped while producing the storage inventory because a retained file has
UID1013, which is absent from the container account-name database. Provenance
reconciliation had completed, but resource imports and serializer Gate0 had not.
No model forward, CUDA initialization, optimizer application or scientific probe
was run. The original log and partial preparation artifacts remain in
`/data/sources/minhao/runs/dflash_native_update_size_20261115/execution_002`.

The narrow repair always records the numeric filesystem UID and uses `uid:1013`
when there is no account name. Only the account lookup's KeyError is caught;
filesystem errors and scientific identity checks remain unchanged. No ownership,
algorithm, numerical tolerance, model, runtime or experimental setting changes.
68 focused CPU tests pass, including the unmapped-owner regression.
A separate same-model-family review confirmed this administrative fix and found
no second affected lookup in the resource inventory.

The next CPU preparation uses a new output directory and the published repair.
It is not a rerun of GPU qualification or training: both have zero work so far.

Retained prior staging/output roots are explicitly included in the same110GB
new-allocation accounting, preventing their bytes from being hidden by a new
attempt directory. This is storage accounting only; the cap is unchanged.
