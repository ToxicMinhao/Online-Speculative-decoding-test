Execution: `3b25b068f0caa0035aeb4c8d58ba498f02a814cc`. This is a CPU evidence-reader repair, not a scientific execution change.

# Compact audit reader repair

The original attempt is preserved in `training_audit_001/` and `training_audit_001.log`.
It ran 83,390 checks. Its reported failures came from two incorrect reader assumptions:

- The allocation table preserves physical block indices: the known seed is slot 0, and a correct prefix of length L occupies slots 1 through L. The reader incorrectly required 0 through L−1. The retained source and table agree; the reader was repaired and a boundary regression test added.
- Replay publication paths `export_path` and `full_state_path` were not excluded by the reader, although they were already declared publication-only exclusions before execution. The replay intentionally writes no replacement tensor checkpoint. The reader now uses those declared exclusions.

No executed source, model state, scientific tolerance or raw evidence changed. Seven focused CPU reader tests pass. The repaired complete audit in `training_audit_002/` passes all 83,386 checks, with no missing inputs or mismatches.

This is a separate standard-library reader with author overlap in the C/replay wrapper implementation. It is not blind independent replication. Full endpoint tensors were not reloaded locally; exact runtime comparison receipts and the 235 tensor fingerprints per replay remain that evidence.
