# Final audit scope and reproduction

Actual execution: `6f7f3a5d1769a630e4ec852205eb06ca55047341`. ARS experiment-agent validate workflow. The auditors independently implement arithmetic and receipt checks; they do not import production analysis helpers. They are internal same-family agents, not blinded external reviewers.

## Published checks

- `WORK_AUDIT.json`: 783,262 checks of retained contracts, full event/state receipts, work accounting, scalar MAX/SUM branches, all native output/trace records and resource receipts.
- `MECHANISM_AUDIT.json`: 26,785 checks on common-origin TRAIN readouts and nested cost/work accounting.
- `INDEPENDENT_SERVING_AUDIT.json`: independent paired totals, all three contrasts, shared bootstrap and exact freeze rule.
- `PUBLIC_TABLE_RECHECK.json`: 3,492 scalar checks using public tables alone.
- `IDENTITIES.json`: source-file, recovery/export-file and independent safetensors tensor-content authentication in place on A800, with CUDA disabled and no model/framework instantiation.

Check counts overlap and are not independent data samples. Native full-state equality is supported by exact runtime receipts plus stored file authentication. Full model-sized gradients, optimizer Jacobians and contractions were not retained; these are not independently reconstructed by a scalar audit. Strongest-rival selection is a native implementation/receipt claim, while saved-margin arithmetic and resulting block metrics are independently checked. Repeated request timing does not replicate training seeds or supply fresh requests.

## Reproduce public statistics

From this report directory, with Python and NumPy:

```bash
python audit_code/verify_public_serving_tables.py .
```

See `audit_code/README.md` for the standalone serving auditor and tests. The rechecker uses the declared public ordinal ordering, PCG64 seed 20261125, 10,000 paired request draws, and linear 95% percentile endpoints. No original prompts, IDs or token data are required.

To reproduce token/state receipt auditing, obtain the private compact run directory and historical PR62/PR60 receipt files from the project owner, then run the published work and mechanism auditors with those explicit paths. On A800, source the project's approved environment helper and disable CUDA before the read-only `authenticate_retained_files.py --root PRIVATE_RUN` check. This hashes retained state files and native safetensors bytes without executing a model. It does not recreate missing tensors or train anything.

## Audit-only repairs, preserved evidence

1. The first work-audit invocation used a relative raw root; the checker prepended it twice for certain constructed paths. That output is preserved privately as `INDEPENDENT_WORK_AUDIT_ATTEMPT_001.json` (774,407 checks; three sections incomplete). Normalizing the auditor root and adding a regression test resolved this checker defect. No experimental file changed.
2. The first CPU source-authentication check treated a tracked directory symlink as a regular file. Its script and traceback remain privately retained. The corrected checker authenticates symlink targets explicitly and verifies regular-file bytes. No GPU state was loaded or modified.
3. A publication manifest included its own earlier bytes. The table publisher was corrected to exclude the manifest itself; the earlier manifest remains privately retained. Report data and scientific comparisons were unchanged.

These repairs are independent audit/publication work, not experimental retries. No new GPU forwards, Adam steps, data collection or serving generations occurred. Original qualification and historical failure records remain intact.

## Privacy and publication

Public tables replace original request IDs with ordinal labels; the bootstrap order is explicit. Prompts, original source IDs, tokens, anchors, teacher features, model tensors and recoveries stay outside Git. Every raw returned record, including all warmups, is retained privately; no unfavorable request was excluded. The public manifest binds this report bundle. Both final training recoveries and exports remain retained on A800.

Published CSV line endings are normalized to LF for Git review; parsed cell values are unchanged from the independently audited exports. A superseded duplicate mechanism-note filename was removed before publication. These are formatting-only changes.
