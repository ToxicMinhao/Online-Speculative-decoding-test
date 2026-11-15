# Approved retry accounting review

Scope: narrow accounting change for execution_002 after the user approved the
six-case qualification retry and unchanged scientific work. This is the
implementation author's review, not an independent GPU qualification.

The new plan authenticates the original STATUS.json, WORK_AND_COSTS.json and
training/PHYSICAL_WORK.jsonl before reading their accounting. The receipt source
must be the failed fdccdf5482f2125f212476547522068f953c564a qualification, with
exactly five completed and known disposable Adam applications. Cost, status and
physical-work records must reconcile. Original files are read only.

The scientific constants and local work ceiling remain unchanged: six
qualification plus 176 main Adam applications, at most 182 in the new attempt.
The explicitly approved cumulative ceiling is 187. Original occupied time is
deducted from the existing 10,800-second ceiling without rounding or resetting.
execution_002 requires this evidence; changed sealed accounting or removal of
the prior receipt fails before model work. No automatic retry path was added.

Validation used the existing local CPU environment with CUDA visibility disabled:

```
PYTHONPATH=.:training:tests /home/minhao2/.venvs/dflash-specforge/bin/python -m unittest -v test_native_prefix_retry test_native_prefix_campaign
```

All 14 tests passed. Coverage includes three-file identity checks, incorrect
counts, scientific work in the prior qualification, unknown work, changed source,
nonfinite/exhausted time, inconsistent receipt totals, missing carryover,
destination binding, seal identities, reduced campaign alarm and child deadline,
and separate current/cumulative reporting. Mocked orchestration performs no model
work.

The real retained inventory was also checked against its previously verified
inventory identity. All three retained receipt pins passed. The resulting values
are five prior Adam applications and 203.0700637307018 prior occupied GPU seconds,
leaving 10596.929936269298 seconds for the new attempt. These values are derived
from authenticated receipts, not inserted as synthetic prior work.

No GPU execution, SSH, scientific setting change, artifact deletion, or commit was
performed by this reviewer. H tensor parity still requires the approved native
qualification. The original failed attempt remains failed and preserved.
