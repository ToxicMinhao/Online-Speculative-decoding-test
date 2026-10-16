# Independent first-update sensitivity check

This bounded follow-up reconstructs the first update from the six completed
per-turn pilot streams. It preserves their saved input/output token IDs and
original dataset wording. No new answer is generated for training or history.

For each stream, independent processes load released DFlash and construct fresh
seed-17 AdamW optimizers. One learns at `3e-4`, the other at `1e-4`. Both use the
qualified AdaFlash OSD training path under our session protocol: forward KL,
temperature 1, no hard-label mixture, full-weight update, two response passes,
two optimizer steps, unchanged author anchor sampling/position weighting and
normalization. Exact masks, anchors and supervised exposure must match the
original first update. Teacher replay reconstructs supervision from saved tokens.

Three checkpoints (released, `3e-4`, `1e-4`) replay two fixed inputs three times
in rotating order: request 2 for future transfer, request 1 for in-sample fit.
The first `3e-4` request-2 replay is the original output-ID/verification-count
reproduction gate and is included in the declared total. Mismatches stop the
campaign with raw evidence preserved; no automatic retry or gain interpretation.
All frozen request-2 counts must also reproduce. Historical timings are not
pooled with fresh measurements.

Total: **12 isolated updates, 24 optimizer steps, 108 timed executions**.
Greedy, block size 10, 1,024 tokens or EOS, pinned native runtime and cold-request
cache policy remain unchanged. Exact saved inputs are passed without rerendering;
every output must match its canonical reference. Replays disable feedback capture
and learning. Activation and cache maintenance stay outside request latency.

Outputs are `RESULT.json`, full event/trace records, all checkpoints and training
metadata, plus `report/raw_repeats.csv`, `per_request.csv`, `training_costs.json`,
`SUMMARY.json` and `REPORT.md`. Optimizer state stays in memory during each isolated
branch; no subsequent update or optimizer continuation is part of this test.
Exports and process startup/shutdown are timed separately. Source/data/model and
runtime hashes are checked before model execution. Six streams run sequentially
to avoid overlapping benchmark work. Peak storage includes retained checkpoints,
export overlap, compact evidence and an unchanged 8 GiB free-space reserve.

This tests first-update sensitivity, not full-conversation low-rate or warmup
efficacy. Better fit to response 1 does not establish better request-2 drafting.
The earlier beneficial update-2 diagnostic remains separate prior evidence that
smaller updates can sacrifice gains.
