The completion auditor is ready for the final retained full-run evidence. It is an independent stdlib-only receipt reducer, with shared implementation authorship and context disclosed. It imports no production validators, Torch, model, optimizer or GPU code.

Run after collection of the final compact evidence:

```sh
PYTHONDONTWRITEBYTECODE=1 python audit_completed_work.py \
  --root /absolute/path/to/final/raw \
  --source-manifest /absolute/path/to/executed/SOURCE.json \
  --tensor-inventory /absolute/path/to/FINAL_TENSOR_INVENTORY.json \
  --postflight /absolute/path/to/POSTFLIGHT.json \
  --output /absolute/path/to/new/audit_work_001
```

Add `--external-map FILE.json` only for explicitly remapped retained external locators. It is a mapping from original absolute path to local evidence path. Names inside the original run root map directly to the local raw root. Unknown external paths are reported missing; the auditor does not guess replacements. `--zstd /path/to/zstd` selects only the CPU decompressor.

Outputs are `AUDIT_WORK.json` (sanitized counters, conclusions and failed/missing section names) and `PRIVATE_INPUTS_AND_ERRORS.json` (private input paths, their fresh audit hashes, and diagnostic details). Existing output directories are never overwritten. Exit0 requires PASSED; missing or inconsistent evidence exits2. Missing final metadata is INCOMPLETE, never an inferred success. These audit hashes identify the current input bytes; they do not replace the separate authenticated retrieval/postflight chain.

The audit covers both actual 58-event trajectories, all 29,696 response presentations, current method flags, fixed recipe/accounting, response-wide FKL/PAIR logging, original local work-ledger ordering, global counts, qualification caps, the single shared retry, startup versus committed optimizer work, all nine shared TRAIN witness endpoints, original elapsed/card clock, and recorded storage checks. Frozen formula hashes are independently pinned to PR65 publication `ab7d8231a70387c7e02f0d9d1feeedd1e80dd258`. All event archives are independently hashed and streamed once, with every member identity checked; only requested compact metadata is retained in memory.

It distinguishes 72 logical witness response reads from at most72 physical forwards if an input has zero valid anchors. It distinguishes48 scoring activations from the separate canonical-F activation (normally49 total physical production activations). It counts startup and failed cases separately and does not infer efficacy from coefficient activity.

Required final tensor metadata, produced by a separate live read-only inventory:

```json
{
  "status": "passed",
  "checked_epoch": 0,
  "files": [
    {"path": "ORIGINAL_ABSOLUTE_PATH", "exists": true, "bytes": 0, "sha256": "ACTUAL_SHA256"}
  ],
  "absent_paths": ["ORIGINAL_ABSOLUTE_RETIRED_PATH"]
}
```

The present-file entries must include each arm's event57 and event58 complete recoveries plus event58 serving weights. The absent list must include that arm's retired event1–56 recoveries and event1–57 duplicate exports. Boundary and retirement receipts supply the expected identities, successor ordering and retained exact-export proof. This auditor reads no tensor payload and therefore cannot itself independently validate their numerical contents; the final inventory must remain bound to its authenticated producer/retrieval receipt.

`POSTFLIGHT.json` must have `status: passed` and a `storage` object with `charged_bytes`, `free_bytes`, and `projected_peak_bytes`; include `remaining_peak_bytes` and `required_free_bytes` when the original accounting includes them. The all-location publication/retrieval charge remains necessary; a remote-only free-space snapshot is not sufficient. Recorded storage maxima are labelled recorded checkpoints rather than a continuously observed physical peak.

The separate serving/statistical auditor still owns exact canonical token checks, repeat counts and paired bootstrap re-reduction. This completion audit is complementary; it must not be described as an independent GPU replication.

Synthetic validation:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest test_audit_completed_work -q
```

12 tests pass. They cover an absent run, event/flag/counter errors, valid neutral TAIL behavior, duplicate or unknown local work, canonical activation accounting, global pending work, qualification overrun, missing event57 retention, elapsed/card/storage failures, one shared qualification retry, startup-without-Adam reconciliation and explicit external-locator requirements. No actual full-run completion result has yet been audited.
