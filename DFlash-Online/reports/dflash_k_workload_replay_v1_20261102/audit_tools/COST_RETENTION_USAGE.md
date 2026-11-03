# Local cost/resource/retention extractor

This tool is prepared for retrieved Track2 compact evidence from execution
`8a4cbcc8b78adfee845a0a1409db4a0d29c54058`. No production result has been
extracted during preparation. Run only after the parent supplies the completed
retrieval location, preserving its `sealed/` and `campaign_001/` layout.

```sh
python summarize_cost_retention.py --run-root RETRIEVED_RUN_ROOT --output NEW_OUTPUT_DIRECTORY
```

The output directory must be new. The tool uses only the standard library,
opens evidence files inside the supplied local root, and does no network, model
or GPU work. Absolute production paths remain receipt values; it does not open
them. It refuses to treat preparation-only files as results.

Outputs:

- `COST_RESOURCE_RETENTION.json`: all extracted scopes, per-arm and per-event
  exposure/costs, source hashes, missing evidence and qualifications.
- `COST_RESOURCE_RETENTION.md`: readable accounting with nested-timer rules.
- `TRAINING_COST_ROWS.csv`: event costs and presentation-weighted exposure.
- `RETIREMENT_PATHS.csv`: exact retired paths, reasons, sizes, hardlink counts
  and retained-evidence references; sizes are not claimed as reclaimed space.
- `RETAINED_ARTIFACTS.csv`: latest producer-validated state per arm, remaining
  materialized exports, shared initial references and final reservoir packets.

CUDA peaks are allocator peaks reset after trainer construction and restoration.
Filesystem figures are discrete samples; selector memory is counted persistent
NumPy arrays, not process RSS. Compiler timers may nest. Different timing scopes
must not be summed into a training-inclusive speedup claim.

Retention claims are based on producer receipts and retirement exclusions,
not fresh bulk-file existence/hash checks. A completed eight-event delivery is
checked for the source-defined retirement category counts, but the tool does
not replace an authoritative transfer-inventory audit. Latest arm commits are
resolved independently, including partial next-event progress.

The extractor author also implemented the additive trainer and seal. This is
source-informed accounting, not an independent trainer audit or GPU rerun.
Endpoint reconstruction, count reconciliation and bootstrap audits belong to
the separate audit tools and are not repeated here.
