## Material Passport

- Scope: independent review of the parent's new frozen-endpoint runtime, entry, preflight and launch code against the adopted handoff; CPU-only fixes and verification.
- Status: reviewed and repaired; ten focused CPU regression tests pass. The actual sealed 1,024-request panel passes the new preflight input gate.
- Authorship: review by the inventory agent, separate from the original four-file author. This reviewer made the listed repairs and wrote tests; it is not blind review or independent GPU replication.
- Evidence: [RUNTIME_REVIEW.json](RUNTIME_REVIEW.json) records the exact reviewed file hashes and actual input-gate result. [RUNTIME_CPU_TESTS.log](RUNTIME_CPU_TESTS.log) records the ten passing tests.

The original code had three immediate integration blockers: it expected a generic `passed` pool status instead of `passed_pool_availability`; it treated the sealed schedule container as a bare list; and freshly produced canonical records omitted the request ID, stop IDs and cap now required by the evaluator. All are repaired without changing native decoding or timing measurements.

The repaired preflight authenticates the original panel-seal files, the audit identity, actual counts (2,985/1,799/1,186/1,024), complete declared history coverage, fixed revision/split and exact seed-20261119 ranking. It checks raw/normalized/token hashes and unique identities, the exact schedule, the native session identifier and warmup disjointness. It rejects missing exposure evidence and non-O58 fresh canonical producers.

Canonical generation now binds request/source IDs, stop IDs, cap, physical-generation identity and all target/tokenizer/runtime/producer identities. Each shard and the complete merge run the strengthened reference validator. Merge requires all 1,024 canonical attempts and completions before any scored attempt; an existing sealed canonical population cannot be overwritten. Native invalid returns remain saved before the exception, and failed/partial attempts remain charged.

Failure cleanup now closes the native backend while its occupied-card lease remains active, fixing the prior undercount of shutdown work on exceptional exit. CPU imports no longer eagerly import the unrelated training qualification path merely to install signal handlers. Native generation/publication/weight-check functions remain inherited unchanged.

Storage boundaries preserve the projected remaining-write reserve rather than checking only current free space. They reject nonpositive or over-30GB projected peaks. Forecasts reject negative, nonfinite or Boolean durations; the original eight-hour/twelve-card-hour ceilings and engineering contingency remain unchanged. The tests exercise these gates, failed-generation accounting, fixed-generation caps, canonical integration with synthetic complete native traces, malformed generation identity, strict exposure failure and cleanup order.

All three input export weight files are now rehashed after the scored worker exits and before report generation, outside all scored latency. `POSTFLIGHT_EXPORT_IDENTITIES.json` retains expected/current identities for every arm, including failures; a changed or unavailable export stops final publication. A regression test proves that a changed AREPO export is rejected while all three observations are retained.

The execution commit is forthcoming. These are CPU readiness checks, with no GPU execution claim.

This review does not pass current native GPU qualification, actual full-run time/storage admission or PR54 archive/keeper reconciliation. Quiet timing still relies on the inherited process/GPU barrier plus the operational requirement to pause CPU reference tests, hashing, compression and transfers. Parent must preserve that operational pause; no CPU reference work is authorized concurrently with scored timing. No model work, network/SSH operations or deletion occurred in this review.
