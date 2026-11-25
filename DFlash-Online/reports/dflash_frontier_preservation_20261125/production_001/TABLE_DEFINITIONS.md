Execution: `225802657d630c0985abd68d6e7453f3e47c5b74`. These files contain scalar metrics and local ordinals only.

`ALL_REPEATS.csv` retains every observed scored/warmup row, including invalid rows; incomplete rows have blank missing values. Warmups are excluded from gain estimates. `PER_REQUEST.csv` contains one row per original panel ordinal only after all 792 generations and both endpoints validate; time columns average the three repeats, while deterministic counts use one checked repeat.

`accepted_draft_tokens` is the stop-clipped accepted-proposal count. `native_accepted_draft_tokens` is committed progress minus verification calls. `committed_progress` excludes the initial prefill token and equals output tokens minus one.

Primary gain is 100×(1−sum(FRONTIER_P request means)/sum(CONTROL request means)). The 95% interval uses10,000 paired whole-request PCG64 resamples, seed20261125, linear2.5/97.5percentiles. This reused development panel conditions on two fixed endpoints; it is not training-seed replication.

Witnesses are overlapping training-block diagnostics. Full vocabulary logits are not retained, so strongest-wrong identities and ties cannot be independently re-projected. Cost scopes overlap and must not be added. Missing or pending work stays unknown. Private audit input fingerprints and failure detail remain in `PRIVATE_AUDIT.json`; do not publish that file.
