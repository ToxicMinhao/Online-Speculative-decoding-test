# SURVIVAL confirmation implementation review

Reviewed 2026-11-11. Outcome: no remaining blocking finding in the scoped CPU/source review. This is implementation validation, not native GPU qualification or confirmation evidence.

ARS Material Passport: local working-tree source, retained handoff, CPU fixtures/test logs and prior pilot interfaces; intended use is serving-only confirmation implementation review. Same-team review with shared conversation context and adapter/test authorship overlap is disclosed. A delegated read-only reviewer checked the shared-engine guard boundary; this is not an independent external audit. Stage 1 / project None remains unchanged.

The confirmation adapter admits only a sealed integer N from 136 through 768 in multiples of eight, with three frozen event-five arms, two complete sweeps and exactly 6N scheduled executions. It reuses the pilot's native engine and raw validator while explicitly restoring every guard bypassed by the shared engine's non-pilot-size mode: fixed count, exact schedule, mandatory callback, pinned physical GPU and complete retained boundary receipts. The original pilot's production-128 contract remains intact. Failed boundaries, unknown attempts and backend returns that fail before measurement persistence have durable accounting; no automatic retry is available. Immediate seed EOS remains valid only through the existing exact one-token/zero-call/empty-trace native branch.

Final runtime/seal inspection verified all six review repairs: worker counters persist before and after cleanup without replacing the primary failure; canonical generation timers include failed calls and disclose returned-but-unvalidated work; each timing boundary queries the observed physical GPU UUID; SIGTERM raises through normal cleanup (SIGINT already raises KeyboardInterrupt); retained pilot RESULT/PLAN/model/runtime inputs are checked against the completed pilot INPUT_MANIFEST and its completed-summary execution commit; and storage uses the qualified 5,327,030-byte cap per generation plus 3 GiB staging/cache/report allowance and the unchanged 8 GiB reserve. Exact checkpoint export identities are checked before serving and afterward, with full source/model/runtime checks in preflight and postflight.

The capacity check deliberately repeats the entire initial peak requirement at each boundary. For N=768 this requires 36,358,114,304 bytes with no missing references, or 40,449,273,344 bytes with 768 missing references. It does not credit already-written artifacts, so it may stop conservatively before a remaining-growth calculation would require a stop. It does not lower or consume the required reserve. Fresh actual-path capacity admission remains the executor's responsibility.

Validation: `preparation/CPU_TESTS_FINAL.log` records 42/42 passing tests (21 adapter, 10 report, 7 data and 4 runtime). The four runtime tests were also independently rerun during this final review and passed. The seven original pilot evaluation regression tests passed separately. The adapter fixture executes and independently audits all 816 native-schema rows for N=136, including 102 authenticated boundaries and the in-budget first-eight-request checks; tampered/missing native evidence, schedules, GPU identities and accounting fail closed. `git diff --check` passed. These are CPU fixtures, not real target generations.

The first retained combined CPU log records a missing-pydantic import failure through eager training/barrier imports; the corrected runtime imports the barrier only when needed. The later 35-test and final 42-test logs preserve that progression. No GPU, optimizer, teacher, feedback or scientific generation work was performed by this review. No remote command or commit was made by this reviewer. No source/input pin, actual GPU/runtime behavior or statistical result is claimed qualified by these CPU tests.

Reviewed working tree is based on HEAD `9239dc559d47362c6fedda1bab7ad8a09904583c`; source edits had not yet been committed at review time. File fingerprints below identify the reviewed bytes; publication and actual-path sealing remain separate executor steps.

| Reviewed file (relative to DFlash-Online) | SHA-256 |
| --- | --- |
| `experiments/survival_confirmation/evaluation.py` | `1b3f1f90cebcc7625c1e4f2284b2c9c7be77d681b3fe62d1faf6c0e74895cb0e` |
| `tests/test_survival_confirmation_evaluation.py` | `3b3e3c3b35f54522234e99600056331e59d664e608b618f00495ed5d829e3723` |
| `experiments/survival_confirmation/runtime.py` | `e3d4d65a4eaf49766392a552491659cb19c49599342a549ea505675ab8980a6d` |
| `experiments/survival_confirmation/seal.py` | `9771fcbc663f7cd656a752f8e768770e0734020e5393e925f59b2d6eb70abcba` |
