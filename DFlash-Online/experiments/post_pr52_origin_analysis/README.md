# Retrospective origin count analysis

CPU-only reconstruction for the original separate PR51/52 MathQA panels. This reads retained JSON and native trace evidence; it imports no model/training/serving package.

Run from DFlash-Online:

```sh
python -m experiments.post_pr52_origin_analysis.reducer \
  --pr51-root "$PR51_DELIVERY" --pr52-root "$PR52_DELIVERY" \
  --compatibility "$COUNT_COMPATIBILITY_RECEIPT" --output "$NEW_OUTPUT_DIRECTORY"
```

The compatibility receipt records authenticated origin/runtime/card evidence. Each delivery manifest must match every consumed source. All 256 scored requests, canonical tokens/termination, caps and all three endpoint repeats must validate. A missing or incompatible panel produces an unavailable contrast; it never produces a favorable subset.

Native accepted proposals equal committed progress minus calls, with one initial token excluded per request. The code reproduces original paired 10,000 draws with seeds 1711 and 1717, checking original draw hashes and source order. No origin latency comparison is computed.

[Report](../../reports/dflash_post_pr52_origin_analysis_20261117/REPORT.md) · [Focused tests](../../tests/post_pr52_origin_analysis/test_reducer.py).
