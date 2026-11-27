# Read-only final inventory and resource audit

`postflight_inventory.py` is an outside-checkout, standard-library-only audit tool. It does not import the experiment, load tensors, initialize CUDA, launch a model, transfer data or delete files. Its optional live GPU check is only an `nvidia-smi` management query. It has **not** been run against A800 or native experiment data during preparation.

The tool verifies the final BASE/TAIL event-57/58 complete recoveries, event-58 exports and current recovery aliases against boundary identities. It verifies exact retirement receipts for earlier states/exports, all captured/terminal/prepared feature files, and preserves the eight shared-witness feature packets. Archived metadata is streamed and hashed without extraction. Disposable qualification exports and state-retirement proofs are checked separately.

Both the original scientific source and any explicitly sealed infrastructure repair are verified. Only the four agreed production files may differ. Original failure/process/clock/work receipts remain authenticated; the packaging recovery preserves the failed receipt, sums failed and recovery time, and must add no model work. No missing or unfinished proof becomes a pass.

The final clock audit accounts for the original start and every retained physical-card interval. It checks terminated worker records and current owned-PID/card state. A still-existing PID is reported for inspection, including possible PID reuse; the tool does not kill processes. All-location storage combines measured execution-host allocations and pinned compiler-cache growth with a fresh explicit peer-location receipt and a declared remaining retrieval/publication allowance. Recorded storage admissions include nested per-arm event forecasts. This is point-in-time and retained-snapshot evidence, not continuous peak monitoring.

## Execution order after the campaign finishes

1. On Fenrir, measure every new study/publication/shared-Git allocation root. Include a conservative allowance for any remaining archive, retrieval, extraction and publication overlap. The caller must supply complete roots; the tool cannot discover unrecorded off-host copies.

```text
python postflight_inventory.py measure-external \
  --root ABSOLUTE_STUDY_ROOT --root ABSOLUTE_ADDITIONAL_GIT_OR_PUBLICATION_ROOT \
  --binding ORIGINAL_PLAN_BINDING --commit ef792694a52395a6c13ea5dabcf7ae64dd1d04ff \
  --remaining-bytes MEASURED_REMAINING_WRITE_BOUND \
  --reserve-bytes 20000000000 --output NEW_EXTERNAL_STORAGE_RECEIPT.json
```

2. Supply that small receipt to the execution host through the approved transfer procedure. With both cards quiet, run the read-only postflight using the retained original plan-byte hash. Keep its output in a new destination inside the charged run root. The receipt must be less than one hour old at the resource check.

```text
python postflight_inventory.py postflight \
  --plan ORIGINAL_RUN_ROOT/sealed/PLAN.json --plan-sha256 ORIGINAL_PLAN_FILE_SHA256 \
  --original-clock-epoch 1795738374.2656422 \
  --external-storage NEW_EXTERNAL_STORAGE_RECEIPT.json \
  --output ORIGINAL_RUN_ROOT/postflight_001
```

3. Retrieve and authenticate the compact evidence and postflight receipts; run `audit_completed_work.py` separately on the retained metadata. Map any external original source manifest explicitly. Its detailed `AUDIT_WORK.json` retains uppercase `PASSED`, `FAILED` or `INCOMPLETE`; `AUDIT_WORK_PUBLIC.json` and `AUDIT_RESOURCES_PUBLIC.json` provide lowercase, plan-bound wrappers for the report builder.

4. Remeasure storage after substantial retrieval/publication. Earlier postflight evidence must not be described as proving later actual usage; the declared remaining-write allowance is a bound, not a measured transfer cost.

## Outputs and limits

- `FINAL_TENSOR_INVENTORY.json`: private paths, exact present-file identities and verified absent paths.
- `POSTFLIGHT.json`: public-safe, lowercase status, plan/source bindings, verified source chain, resource totals and inventory/content identities.
- `PRIVATE_DETAILS.json`: errors, input identities, process observations and original path details. Keep out of public Git.

Read-only budgets are 1 TB of streamed input hashing, 500,000 filesystem entries and two CPU wall-hours, plus at most 64 MiB of receipt hashing. There is no GPU-card consumption. File hashing is repeated where needed rather than treating a same-size/same-time file as unchanged. Resource accounting reserves 64 MiB for these receipts. Budget exhaustion is `INCOMPLETE`, preserving partial evidence. A missing field or contradictory identity is rejected; a genuinely unfinished/missing artifact is not relabelled as a successful check.

Preparation validation: **18 focused postflight tests and 14 completion-auditor tests pass**, including a full synthetic source-amendment/resume metadata chain, changed scientific source rejection, preserved failed archive recovery, hash mutation, unknown/live work, missing evidence, and nested forecast/all-location storage checks. Shared implementation authorship and context are disclosed; this is not blind external review.
