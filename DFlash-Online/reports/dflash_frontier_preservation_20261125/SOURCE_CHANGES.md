# Execution versus publication

The only A800 execution used `da4f4cfe28844141a5548401ca86e0ec08e73abf`.
It adds a new experiment package to PR59 publication `a780c94`; no inherited
objective, training, inference, model or environment source was modified.

Code follow-up `fd2bb9e56b32a63b5c62563d533512cb53e3d7d9` was CPU-tested only. After that immutable execution source was staged, CPU review reproduced a false rejection
in the allocation invariant for a large common score offset. The publication
adds a centered-dot check and a regression while retaining raw diagnostics;
allocation values, cap and tolerances are unchanged. The native prerequisite
never imports or invokes this mathematics helper. No GPU work was repeated.

Publication also adds `analyze.py`, compact evidence and this report. The
exporter reads retained records only and performs no model computation. The
scientific execution commit is recorded throughout; the publication commit is
bound by Git and recorded in the PR summary/operational record.

Read the README for exact prepare/qualification/evidence-export commands.
Production training and serving are intentionally not represented as executable
or qualified until the native derivative blocker is resolved.
