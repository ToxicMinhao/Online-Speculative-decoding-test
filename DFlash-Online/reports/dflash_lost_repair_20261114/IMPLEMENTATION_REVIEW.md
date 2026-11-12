# Prelaunch implementation review

ARS experiment-agent run/validate; no scientific results yet. Execution source: 29b6a15c5f5f16135663ea61e3d31f21b3f1d465, stacked on PR41 ab27eaa.

71 focused CPU tests pass in the existing deepspec_repro environment. Earlier 14 transition/headroom constructor tests also passed. One test command was accidentally invoked from the repository root, where the tests directory is absent; rerunning from DFlash-Online passed. No GPU work was consumed by that invocation.

Scoped parallel source and stub review found and repaired three prelaunch integration mistakes: opening capture must retain prompt-ingested draft KV; completion RPC needs the transition_ dispatcher prefix; teacher preparation must be exposed by the inherited worker dispatcher. Terminal teacher work is explicitly charged as preparation, not startup. Interrupted phase time and partial-presentation denominator evidence remain visible. Original algorithms, native inference settings and checkpoints are unchanged.

The exposure path strictly checks K+6 saved config bytes, step/presentation counters, horizon, accumulation and response reduction; it loads saved weights and uses a disposable saved RNG. It independently repeats native presentation and anchor draws, checks final RNG, and constructs no optimizer/backward path. Native ineligible-successor first errors retain their low coefficient and all denominators. Checkpoint and source identities are checked before/after execution.

Reviewers share the same team and some authorship context; this is not independent GPU replication. Neutral K+6 proposal/acceptance/commit parity at every natural start remains required. Full corpus/anchor coverage is required for the gate. Resource ceiling remains four aggregate occupied A800-hours; one card is planned. Expected post-launch wall time is provisionally 1–2 hours, not a guaranteed completion time. No cleanup is required at 624.3GB free versus 48GiB admission.
