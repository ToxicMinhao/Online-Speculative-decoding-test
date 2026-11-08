# Role-loss pilot: raw endpoint audit

## Material Passport

- Workflow: ARS-Codex 3.24.2 experiment-agent, validate.
- Status: ANALYZED; all recorded endpoint checks passed. No GPU rerun was performed.
- Source execution: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`.
- Method: a separately written local standard-library audit. It imports no production validator or scoring function. This is a same-team evidence audit, not independent GPU replication.

## Result

**Passed: 4,608 / 4,608 timed executions, 264,251 native verification rounds and 2,811,839 assertions.** All four stages completed with zero invalid, unknown or unattempted executions. The audit read and hashed 13,881 unique JSON/JSONL files. CPU audit time: 8.087 seconds; model work: zero.

| Dataset | Checkpoint | Executions | Repeats | KPLUS / MIX / ROLE calls | Cap-hit requests per method |
|---|---:|---:|---:|---:|---:|
| gsm8k | event_002 | 768 | 1 | 12307 / 12314 / 12315 | 1 |
| gsm8k | event_004 | 1536 | 2 | 12116 / 12014 / 12134 | 1 |
| mathqa | event_002 | 768 | 1 | 17243 / 17309 / 17317 | 9 |
| mathqa | event_004 | 1536 | 2 | 17074 / 17227 / 17158 | 9 |

Call totals use one validated trajectory per request after repeat equality checks. Event-4 repeat copies are not additional independent requests. All one GSM8K and nine MathQA cap-hit requests remain included in every arm and checkpoint.

## Checks performed

- Exact presealed panel, canonical and arm-order schedule identity; 256 requests per dataset; three actual checkpoint identities at each checkpoint stage.
- Every raw input and output token ID and termination matched its saved canonical. Native response geometry, output cap, stop IDs, full context allowance and no earlier stop tokens checked.
- Every block checked independently: proposal/target geometry, first mismatch, accepted prefix, committed tokens, anchor token, absolute positions, contiguous output progress, stop clipping and finished status. Summed counts reconstructed from these decisions match the saved measurements.
- All 4,608 native sidecars equal their corresponding embedded capture records. Captured features are disabled and zero bytes; runtime is pinned SGLang FA3, block 16, context 4,096 and full cache flush between requests/publication.
- Every per-request validation record is valid, and all endpoint repeats agree in calls, matched acceptance, native acceptance, committed progress, output length, termination and checkpoint identity.
- All 144 publication records show empty caches, healthy workers, refreshed helpers/fusion and correct checkpoint paths. The twelve per-stage arm weight checks report exact equality of 58 exported tensors mapped to 43 native tensors. Selected paths match training-boundary records.
- All 144 timing-boundary records pass training/transfer/foreign-GPU quiescence and storage admission. Timings are finite and positive, cache costs nonnegative, and saved scalar timings match raw metadata.
- All four stage status records reconcile attempted/returned/valid work exactly and report zero training, teacher preparation, feedback collection or extra warmup generations during evaluation.

## Limits

The audit validates retained evidence rather than independently measuring kernels or reloading native tensors. Exact loaded-tensor equality and quiescence are checked from recorded execution receipts; the parent audit separately verifies source/export and retrieval hashes. This audit cannot establish scientific efficacy by itself and does not independently verify the target model outside the qualified native verifier. The paired bootstrap and interpretation are separate analyses.

Artifacts: `ENDPOINT_AUDIT.json`, `ENDPOINT_AUDIT_FILES.json`, `ENDPOINT_AUDIT_EXECUTION.log`, `audit_endpoints.py`. Raw token records remain private.
