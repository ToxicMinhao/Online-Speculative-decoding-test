# Threatened-repair replay execution metadata

Scientific source: fdbf319b2e68e166c53a8304ad26d0c4fa500e29.

This metadata-only staging envelope contains source/input identities, existing
dataset IDs and normalized-text hashes, and actual execution paths. It contains
no source/generated text, prompt/output token IDs, teacher packets, cache tensors,
weights, credentials or budget-service authentication. Private inputs remain at
the authenticated A800 paths. These files are outside DFlash-Online so publishing
the envelope does not change the sealed scientific source snapshot.

TRAINING_EXCLUSIONS contains exactly the 896 original rows consumed by
validate_disjoint, with every field/value unchanged. The source identity binds
the verified original inventory; unrelated evaluation rows with private token
IDs are deliberately omitted. This changes packaging only, not the fixed memory,
probe selection, exclusion checks or scientific code.

The source archive and these manifests can be downloaded directly on A800 when
the Fenrir upload route is impaired. Exact file identities must still pass before
model execution. Network staging never supplies learning feedback.
