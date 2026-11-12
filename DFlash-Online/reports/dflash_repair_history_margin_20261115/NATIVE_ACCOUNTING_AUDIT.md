# Native execution and accounting audit

**PASS: 1,536,925 checks, zero discrepancies.** This CPU-only audit independently reduced the retained work ledger and native records; it did not run any model, contact A800, modify raw artifacts, or change scientific code.

Scientific execution commit: `dea3fe81dea957416a4f7c7188b60f44a7ced52c`.

| Work | Completed |
|---|---:|
| K+5 parent walks | 192 |
| Natural parent starts | 8,647 |
| Neutral K+5 replays | 8,647 |
| K+4 replays | 8,647 |
| Required unique K+6 replays | 1,343 |
| Measured draft blocks | 27,284 |
| Startup draft-block operations | 51 |
| Total draft-block operations | 27,335 |
| Measured target calls | 27,476 |
| Startup target calls | 48 |
| Total target calls | 27,524 |
| Native initialization/publication operations | 386 |
| Context forwards, including repeated cache reconstruction | 551,021 |
| Optimizer steps / backwards / timed endpoints | 0 / 0 / 0 |

All admitted work completed, with no failed or unknown tickets. The parent walks reproduce retained canonical outputs and archived native traces. All 8,647 neutral replays match every recorded proposal, acceptance, commit, position, clipping and termination field; target and neutral draft cache checks pass. All K+4/K+6 replays use reconstructed destination draft caches with preserved target state and original budgets. The required K+6 start set equals the unique completed set.

CPU cache compaction verified 709,054 prefix tensors. The final compact cache views account for 12,588,818,432 bytes; 16,028,971,836 cumulative bytes of snapshot metadata were offloaded from GPU storage. These are storage/accounting totals, not peak memory measurements.

The ledger establishes that all 192 parents, 8,647 neutral blocks and 8,647 K+4 blocks completed before the first K+6 outcome admission. That admission carries the pre-K+6 seal SHA256 `f7bd53131c9358d4bbab59a83be12a299744f8313fd86247590780ac0e4a7678`; exactly one subsequent publication occurs. The pinned runner binds it to K+6. Publication ledger rows do not separately timestamp the checkpoint path, so this chronology uses the sealed-hash ledger together with authenticated control flow, rather than claiming an independent timestamped load trace.

The checkpoint identities, source commit and successful postflight immutability receipts agree. One occupied-GPU interval was recorded on physical GPU0, UUID `GPU-9986735a-3e68-d1bd-5998-9634ef24341e`, for **4,963.502 seconds (82.725 minutes)**, below the 10,800-second ceiling.

Native execution completed successfully. The original wrapper's exit 1 arose during report rendering and remains preserved; this audit does not relabel that original exit as successful. The separately reviewed CPU reporting repair is distinct from the executed scientific revision.

**Independence disclosure:** the auditor implemented the native observer and cache-retention adapter. This is a separate read-only evidence reduction with authorship overlap, not an independent GPU replication. Statistical matching, native-margin QA and effect estimates are covered by the separate reduction review.
