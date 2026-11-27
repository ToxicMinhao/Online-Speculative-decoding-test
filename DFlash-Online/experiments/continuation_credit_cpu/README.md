# Retained continuation-credit analysis

CPU-only analysis of PR64's native parent/child JSON records. No model libraries,
checkpoints, teacher features, training or serving are used. PR65 TAIL observations
are optional descriptive evidence; only native child outcomes enter fitting.

From `DFlash-Online`, run in an existing NumPy/SciPy environment:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
taskset -c 64,65 python -m experiments.continuation_credit_cpu \
  --primary-root /path/to/diagnostic_001 \
  --tail-root /path/to/pilot_001 \
  --output /path/to/new/cpu_001/results \
  --source-revision "$(git rev-parse HEAD)"
```

Use a new output directory to preserve previous evidence. On A800, source the
approved environment helper first and keep inputs/outputs under approved roots.
Pin affinity to two available host CPUs; do not overlap timing sweeps.

`records.py` defines old/shared scores, outcomes and follower features.
`association.py` adapts the retained PR64 response-effects solver.
`prediction.py` implements the fixed four-fold logistic comparison.
`allocation.py` replays response-wide coefficients and one-at-a-time sensitivity.
`__main__.py` writes derived tables, full bootstrap draws, model settings and costs.

Public block ordinals follow each response's sealed anchor order. Original record
IDs, prompts, tokens and anchor locations stay in the retained private inputs.
The coefficient tables describe 64-anchor diagnostic pools, not the historical
up-to-512-anchor training allocations. C alpha/cap sensitivity is unavailable
without exact retained canonical probabilities; margins cannot replace them.
