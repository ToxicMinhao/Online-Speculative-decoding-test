# PR49 event-seven collection: CPU evidence reduction

This analyzes the existing capture-enabled N7/E7/S7 collections. It imports only
Python's standard library and NumPy. It does not import Torch, serving backends,
trainers, or tensor loaders. It produces no latency comparison.

Keep the original authenticated deliveries in place. No raw run copy is needed.
The provenance checker resolves original paths through those deliveries, verifies
the retained transfer-manifest chain and source identities, and inherits checkpoint
identities from authenticated receipts without reading model files.

From `DFlash-Online`, using the qualified Python/NumPy environment:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD"
analysis_python=/home/minhao2/.venvs/dflash-specforge/bin/python
analysis_evidence=/home/minhao2/EAGLE_SPEC/research/dflash_persistent_prefix_withholding_20261115/delivery
analysis_output=/home/minhao2/EAGLE_SPEC/research/dflash_pr49_event7_collection_cpu_20261116/reproduction

"$analysis_python" -m experiments.pr49_collection_cpu.provenance \
  --delivery "$analysis_evidence/production_continuation_001" \
  --qualification-delivery "$analysis_evidence/qualification_retry_002" \
  --restart-delivery "$analysis_evidence/production_restart_001" \
  --output "$analysis_output/provenance"

"$analysis_python" -m experiments.pr49_collection_cpu.reducer \
  --delivery "$analysis_evidence/production_continuation_001" \
  --qualification-delivery "$analysis_evidence/qualification_retry_002" \
  --provenance "$analysis_output/provenance/PROVENANCE.json" \
  --analysis-commit "$(git rev-parse HEAD)" \
  --output "$analysis_output/reduction"

"$analysis_python" -m experiments.pr49_collection_cpu.independent \
  --delivery "$analysis_evidence/production_continuation_001" \
  --output "$analysis_output/independent"

"$analysis_python" -m experiments.pr49_collection_cpu.compare \
  --independent "$analysis_output/independent" \
  --primary "$analysis_output/reduction" \
  --output "$analysis_output/INDEPENDENT_REDUCTION.json"

"$analysis_python" -m unittest discover -s tests -p test_pr49_collection_cpu.py -v
"$analysis_python" -m unittest discover \
  -s reports/dflash_pr49_event7_collection_cpu_20261116 \
  -p test_independent_reader.py -v
```

The analysis directory must remain private: manifests contain local path mappings,
although tables contain only approved source/request identities and counts. Publish
sanitized receipts, count tables and summaries, never raw prompts/token arrays.

`PER_REQUEST.csv` has384 arm/request rows; `PAIRED_REQUESTS.csv` retains128 requests
and all three call differences. Undefined zero-call request ratios are blank. If
any authentication, pairing or validity check fails, the reducer saves its inventory
and marks the primary comparison incomplete; it does not compute a favorable subset.
Bootstrap draws use a common paired128-request sample across all three arms,
PCG64(1708),10,000 draws, linear percentile intervals. The draw hash is over
row-major little-endian int64 bytes. Native accepted proposals (`A`) are separate
from the audit-only matching count (`M`), particularly at terminal clipping.

This is retrospective evidence. The collection and final-panel populations differ;
changes between their contrasts are not a temporal learning effect. No result from
this command authorizes GPU work, a new algorithm or a learning-rate sweep.
