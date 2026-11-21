# CODE production running

Execution commit: `1551e5e5d0dd7736d8441977aa3dea250535da7b`. Qualification commit: `658d734c422a1a200fe400dceb256492f529204e`. The report's publication revision is separate from the execution checkout.

**O is collecting its first 128-response feedback window on A800.** At this snapshot, 11 responses had completed and 1 was in progress. No production optimizer step or event had completed yet. AREPO and C are queued after O, as planned. Both assigned GPUs are reserved by the online trajectory; serving currently runs on card A, with the trainer on card B.

The fresh CPU preflight passed in 24.62 seconds. Exact released weights, fresh empty Adam state and the intended RNG initialization passed without overwriting the RNG. The continuation verified all 5,018 source entries and preserved the qualified source. All 222 CPU tests and independent review passed. The launcher is running under supervisor PID 4145120.

The original clock and consumed work are preserved. Limits are 96 elapsed hours and 120 occupied-card-hours, with the unchanged 90 GB study cap and 400 GB free reserve. The remaining forecast is 59.15 raw hours; admission estimates 81.87 total original-clock hours including contingency and 110.85 total card-hours. The planned 86.53 GB storage peak is admitted; free space at this snapshot is 491.121 GB. These are forecasts, not endpoint measurements.

Passed qualification and the full 2,048-generation reference precheck were reused: zero repeated model executions. All earlier failures, the one historical unknown generation, source identities and consumed costs remain retained. The administrative archive and 6,601 retained evidence files were authenticated. Algorithms, settings, panels, work counts and authorized output policy are unchanged.

Continue automatically through the fixed O→AREPO→C trajectories and final evaluation unless a concrete correctness/resource failure occurs. No comparative performance result exists yet. [Launch evidence](PRODUCTION_LAUNCH.json), [continuation contract](BUDGET_CONTINUATION.md), [historical budget stop](QUALIFIED_BUDGET_STOP.md).
