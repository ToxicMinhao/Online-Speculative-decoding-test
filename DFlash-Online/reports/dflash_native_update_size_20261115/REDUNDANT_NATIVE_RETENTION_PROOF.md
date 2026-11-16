# Supplemental retention proof for the two new native payloads

The two framework-generated PR48 payloads are eligible for retirement after final report publication and the remote checks below. They contain the reproduced native endpoint; they are not new candidate mechanisms. This audit read local receipts/source only and deleted nothing.

| Newly generated file, relative to execution_003 | Logical bytes |
|---|---:|
| training/trainer/events/event_007/export/model.safetensors | 2,097,259,104 |
| training/trainer/training_state.pt | 6,299,188,123 |
| **Total** | **8,396,447,227** |

The exact absolute paths and protected canonical replacements are in [REDUNDANT_NATIVE_RETENTION_PROOF.json](REDUNDANT_NATIVE_RETENTION_PROOF.json).

The training result and standalone event-publication receipt agree exactly. They record completed publication at 896 Adam steps and 1,792 presentations. `FINAL_NATIVE_PARITY.json` reports no differences against historical NATIVE7 in model tensors, Adam state, RNG, counters, strategy, world size, reduction, accumulation and horizon, with event-seven request/order/anchor checks. The historical model and recovery paths are classified KEEP_INPUT and have matching identities in the sealed input manifest, live-input manifest and artifact inventory.

The inspected source matches all seven source-manifest identities at scientific commit `5d4e0ff65c5976318a50ebb181923447068169c8`. Publication writes `_checkpoint_state` atomically, exports its saved draft tensors while preserving RNG, and renames the pending recovery to `training_state.pt`. The final parity check then compares the published trajectory's live scientific state to historical NATIVE7. The serializer and live-state constructors select the same scientific fields. This establishes scientific redundancy through publication provenance and recorded final parity; **it does not claim identical whole-file bytes**. New run/output identity and additive diagnostic metadata may differ, and the newly generated files have not been reopened by this CPU-only audit.

Before deletion, authenticate the protected original export and recovery against their sealed size/SHA-256 values; confirm each new path is a regular, uniquely allocated file inside the new execution root, not a symlink or alias into KEEP assets; record its current size and hash. Preserve all compact receipts, source identities, manifests and original KEEP files. Delete only the two named new bulk payloads after the final report is published and both final audits pass. If any identity or path check fails, keep that file and report the discrepancy.

Retirement removes the ability to open these two particular PR48 bulk files. Original NATIVE7 remains available with the same scientific endpoint and continuation state, while PR48's additional metadata remains in compact results. These payloads would be retired, not archived.
