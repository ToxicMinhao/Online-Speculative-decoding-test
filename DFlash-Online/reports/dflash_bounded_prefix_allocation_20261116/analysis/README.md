# Completion analysis

Execution source: `c94b1bb9b32dbd9a1f523105cff4713f7b880ec3`.
These CPU-only readers consume the verified private delivery after model work finishes.

- `audit_primary.py` independently reconstructs native traces and paired effects without production metric helpers. Eight synthetic tests pass. Its initial implementation was published before endpoint effects were inspected.
- `analyze_training.py` validates original-forward allocation summaries and work accounting. Six tests pass. Its cancellation-sensitive bookkeeping comparison was repaired after the first actual audit failed; the original failure and analytic FP32 bound are disclosed in `TRAINING_AUDIT_REPAIR.json`. Scientific per-block tolerances and execution data were unchanged.
- `run_secondary_collection.py` authenticates retained N7/S7 provenance and C7's own feedback, then computes only the prespecified secondary count comparison. Six tests pass. The driver was prepared before reading C7 collection outcomes, but its portable packaging and final publication followed knowledge of the endpoint result. It does not change the primary decision.

Private token arrays and runtime artifacts are not included here. These are independently implemented checks, not a claim of blinded human review or independent experimental replication.
