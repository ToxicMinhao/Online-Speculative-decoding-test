# Compact evidence retrieval completed

The pending Fenrir-copy statement in the sealed report is now resolved. All **97,485 original compact evidence files (5,764,699,468 bytes)** were copied from A800 and verified against the frozen per-file inventory, with **zero size or hash discrepancies**. Retrieval and verification completed at **2026-11-07 13:41:16 UTC**, taking **2,800.649 s**. The original A800 evidence remains intact.

This is an evidence-backup update only. The scientific execution remains `7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a`; the report published at `0d17534bd7e96f68cba15c782d40eda34594fbf0` is unchanged. No training, generation, tensor loading, new analysis or scientific retry was performed for retrieval.

Raw responses, token IDs and native traces remain private outside Git. This copy contains the complete original compact inventory, not model/optimizer tensors. Selected retained tensor states remain on A800; the previously retired intermediate tensors are not recoverable from this copy and are not described as archived. Audit outputs and public report tables were separately retrieved and verified.

Fenrir had **13,857,820,672 bytes** free after verification, above the preserved 8 GiB reserve. The last A800 check at 13:35:30 UTC found **149,772,161,024 bytes** free and no GPU compute processes. Retrieval wrote only to Fenrir.

See [RETRIEVAL_COMPLETION.json](RETRIEVAL_COMPLETION.json) for the inventory identity, exact work and timing. This sidecar supersedes only the old pending-copy status; every experimental result, caveat and historical failure remains unchanged. The authorized experiment ends here, with no automatic follow-up. Research remains Stage 1 / project None.
