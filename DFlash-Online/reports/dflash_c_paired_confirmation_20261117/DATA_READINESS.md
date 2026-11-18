# Sealed input readiness

Execution source: `31b36fd27976fb46bb6754d8b8b7e769c3d6e5f6` (PR53). This inventory describes preparation; it does not claim completed GPU qualification or an experimental benefit.

| Dataset | Pinned test rows | Excluded | Eligible | Scored | Warmups |
|---|---:|---:|---:|---:|---:|
| GSM8K | 1,319 | 130 | 1,189 | 256 | 2 |
| MathQA | 2,985 | 1,541 | 1,444 | 256 | 2 |

The fixed SHA-256 ranking selects the first 256 scored requests and next 2 warmups without consulting answers, generated lengths or performance. Normalization preserves case, punctuation and MathQA options. Original adaptation inputs, ordering and origins remain unchanged. Identities/hashes are in [DATA_INVENTORY.csv](DATA_INVENTORY.csv); revisions, source identities and sealed-file hashes are in [DATA_READINESS.json](DATA_READINESS.json).

GSM8K excludes the bounded-prefix-allocation C family's prior panel, warmups, probes and adaptation ancestry. Older unrelated-project exposure is allowed: 255 of 258 selected requests have positive prior-use evidence. The remaining 3 are not certified globally unused. MathQA excludes the complete enumerated project inventory:1,025 prior exclusions plus 258 PR51 requests and 258 PR52 requests. The subsequent origin analysis used existing records and added no model exposure.

The common GSM8K event-seven pool retains original producer receipts and historical full-file identities. The MathQA event-five pool retains the explicit PR52 functional qualification: both complete N5/S5 replays matched their retained scientific receipts and endpoints using the currently sealed 128-packet pool. **Historical MathQA feature byte identity remains unverified.** Current full-file/per-tensor identity and the typed replay receipt preserve this distinction; no new full-event replay is requested.

Both methods continue complete native origins (GSM8K KPLUS6; MathQA KPLUS4) with inherited initialization seed 17, optimizer state and horizon 7,424. One declared continuation fork changes Torch CPU and assigned logical-CUDA RNG to 29 after startup restoration. Native K+ and unchanged C share the first-event pool, then use their own newly collected second-event feedback. Evaluation inputs never enter either pool.

This readiness package contains no prompts, responses or token arrays. Exact private manifests remain sealed and authenticated by their hashes. [FINAL_PACKAGE_REVIEW.json](FINAL_PACKAGE_REVIEW.json) records the local source/package check; actual A800 preflight and qualification remain separate evidence. The first private review attempt misread an authenticated source symlink as a regular file; that reader failure is preserved, and no scientific source or input changed.
