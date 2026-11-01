# O_HALF training, cost and retention audit

## Material Passport

- **Status: passed — independent compact-record analysis; no GPU rerun or tensor reload.**
- Execution `19ee22faf985e0576d287eaa8c8624b5a44b46f5`, [PR #22](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/22).
- Actual run: `/data/sources/minhao/runs/dflash_o_half_matched_lr_20261031/production_002`.
- Evidence: immutable `compact_evidence_001`, the pre-launch actual-state migration receipt, deployment-repair receipt and fresh `RETENTION_AFTER.json`.
- ARS 3.24.0 experiment-agent validation; no change to the experiment or published source.

## Finding

All **76,167 checks pass**, with zero discrepancies against the raw training records or reported costs. Rehashing independently verifies all **3,486 retrieved files / 280,550,166 bytes**. Both seeds completed four continuous O_HALF events with the intended ordinary forward-KL objective and LR1.5e-4. No missing, invalid, unknown or extra training work appears in the retained records.

The old `production_001` attempt stopped during deployment admission because 35 exact committed publication files were missing. Its retained repair receipt records zero generations, forwards or optimizer steps. `production_002` is the sole scientific execution; fixing file deployment did not change scientific settings. The prepared audit plan's old root and `ENDED_UTC` assumption are superseded here: completion is established by actual `EXIT_CODE=0` and `FINISHED_UTC`.

## State, sampling and supervision

- **Two initial migrations and six own-state restores pass.** The new initial states retain their original per-seed released weights, empty Adam and RNG, with the narrow LR/metadata change. Both authoritative original states also passed the separate CPU migration check before GPU execution. Runtime construction restores saved RNG afterward.
- All eight live-entry, decoded saved-endpoint, live-boundary and closed-state receipts pass exact-state/export checks and clean accumulation. Each seed ends with **512 Adam steps, 1,024 presentations and four completed events**. Adam moments/counters continue within that arm; no reseed, reference model or correction state is recorded.
- All **2,048 response presentations** match the original O response identity, pass and sampled-anchor records exactly. Feature metadata matches too. Each event has 128 distinct responses shown once in each pass; each seed's four pools are disjoint. These records and the audited natural-sampling code support schedule equality; the audit did not regenerate RNG draws.
- The actual sampler exposes **554,920 retained anchors** and **8,117,722 supervised proposal occurrences**. Reconstructing ordinary valid positions from saved anchors and prompt/sequence lengths reproduces every response's exposure, including positions after a first prediction error. No K selection field is introduced.
- All 2,048 durable response observations match RESULT exactly. Each response loss matches the FP32 division of its recorded numerator/denominator; all losses, exposures and 1,024 window gradient norms are finite. There are **201 clipped optimizer windows**. Per-response, per-pass and event aggregation matches the primary report; accumulated window observations are not double-counted.

These exact-state statements describe the execution-time validation receipts, corroborated by the implementation and compact evidence. Full recovery tensors were not decoded again on Fenrir by this audit.

## Work accounting

| Work | Reconstructed total |
|---|---:|
| New ordinary Adam steps | 1,024 |
| Original response forwards / backwards | 2,048 / 2,048 |
| Additional qualification optimizer steps | 0 |
| New feedback / target-feature preparation / reference operations | 0 / 0 / 0 |
| Draft forwards / layer forward calls | 2,048 / 10,240 |
| Original objective chunks | 70,140 |
| Checkpoint-recomputed objective chunks | 70,140 |
| Vocabulary-head calls, including recomputation | 280,560 |

Every ledger start has one matching completion. All eight workers have distinct processes and 48 distinct initially empty compiler/kernel cache paths; they use one physical A800 updater, with zero declared generation/training warmups. Chunk counts independently equal the sum of the retained block counts divided into groups of eight, with one checkpoint recomputation per chunk.

**Zero new target-feature preparation does not mean zero teacher-head work.** Ordinary KL projects the retained teacher features through the frozen vocabulary head; that intrinsic work remains included. The shared head hook counts both draft and teacher projections. No extra full-response forward is hidden in those chunk counts.

## Cost reconstruction

All elapsed values below come from the fresh O_HALF run. They match the primary report exactly.

| Seed | Training loop (s) | Event incl. save/export (s) | Complete worker (s) | Interpreter subprocess (s) |
|---|---:|---:|---:|---:|
| 17 | 1,423.212 | 1,580.863 | 1,821.032 | 1,852.989 |
| 29 | 1,419.227 | 1,574.747 | 1,800.671 | 1,841.029 |
| **Total** | **2,842.439** | **3,155.609** | **3,621.703** | **3,694.019** |

Loading/initialization/resume validation totals **311.388 s**; endpoint validation/hash work totals **148.383 s**. Checkpoint save totals **160.578 s**, and export totals **124.392 s**. Save/export are inside event time. Observations, record writes, optimizer host timers and teacher-head time are nested inside training, so none should be added again to the table totals. The **886.115 s** teacher-head timer includes checkpoint recomputation and does not provide a full kernel breakdown.

Peak post-load allocation is **20.653 GB** and reservation **27.856 GB** (decimal); these are maxima across workers. Their scope includes training, persistence and endpoint validation. Loading peaks are separately recorded. The supervisor elapsed time is **6,022 s (1 h 40 m 22 s)**, including CPU checks and reporting/postflight; the campaign itself reports **5,157.132 s**. Neither is just training time.

Retained O/K/K_HALF training costs are **historical unmatched context**. They do not establish a fresh causal overhead comparison. This count-focused experiment also supplies no new comparative serving-latency or training-inclusive speedup claim.

## Retention and limits

All four allowed retirement receipts match their prior tensor identities and passed successor proof hashes. Only the new run's event1/3 state/export tensors were retired: **eight files, 33,559,334,252 logical bytes**. They were **retired, not archived**; exact odd-event optimizer resume and serving export are unavailable.

The fresh parent retention check confirms both new initial states and the four event2/4 state/export pairs still exist with recorded sizes; final states have the expected two links for the latest alias. It reports idle GPUs and **45.356 GB** available. This current existence/size/link check is separate from the full tensor equality/hash checks performed at each saved endpoint; it is not a second tensor-content validation.

No source, remote file or GPU state was changed during this review. Model quality and uncertainty are analyzed separately by the statistics review. A valid negative or favorable result would not change this accounting verdict.

Evidence: `FINAL_TRAINING_COST_REVIEW.json`, `TRAINING_EVENT_AUDIT.csv`, `FINAL_TRAINING_COST_REVIEW.log`; reconstruction script `audit_training_records.py` is outside the execution tree. Raw retrieved files are unchanged.
