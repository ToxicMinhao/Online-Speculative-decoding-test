# Retained-result audit

Scientific execution: `416cb7931028621c09b8fa983e53fc8a7dcebaab`. These scripts ran **after** production and did not change the experimental source or launch model work.

`audit_results.py` independently reconstructs native trace commits/stop clipping, frozen canonical output/termination agreement, the execution schedule and work ledgers, request-paired reductions, bootstrap intervals, and frozen-origin witness outcomes. It uses the standard library and NumPy, without importing the experiment's analysis helpers or model code. Full raw observations remain private.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python audit_results.py \
  --run-root /path/to/authenticated/execution_002 \
  --output-dir /path/to/new/private/audit_output
```

`summarize_training.py` reads each response observation once and preserves the distinction between equal-response loss averages and pooled weighted ratios. It retains separately scoped and overlapping costs, actual feature-preparation forwards, startup work and recorded allocator measurements.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python summarize_training.py \
  --root /path/to/authenticated/execution_002 \
  --output /path/to/new/aggregate_training_summary.json
```

The executed local interpreter was `/home/minhao2/.venvs/dflash-specforge/bin/python`. Inputs were the authenticated compact retrieval, not new outputs. Both scripts completed; source and result identities are recorded in `AUDIT_SUMMARY.json`. The raw auditor checked 792 generations and 47,925 verification blocks, and reproduced all serving intervals and witness aggregates. The training summarizer's three synthetic CPU fixtures also passed. Production-source fixtures (60 unique CPU tests) and reused native qualification are separate from this result audit.

Limits: no model replication, fresh vocabulary projection, tensor reloading or new GPU work occurred. Saved margin arithmetic/predicates are independently checked, but unretained full-vocabulary logits cannot be independently reconstructed. Exact model/optimizer restoration and publication are supported by the retained native receipts, not by loading those tensors in this offline audit. The all-11-category interpretation review is an analysis review, not a blinded external replication.

`AUDIT_PRIVATE.json`, `PRIVATE_RECOMPUTATION.json` and raw inputs must remain outside public Git. Public scalar tables and numeric ordinal ordering support independent serving reductions without prompt text, token arrays or source identifiers; see `TABLE_DEFINITIONS.md`. Original failure, blocked-preflight and qualification records remain unchanged.
