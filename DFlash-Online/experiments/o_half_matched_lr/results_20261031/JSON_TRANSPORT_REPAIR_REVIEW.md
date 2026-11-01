# O_HALF contract JSON transport repair

## Material Passport

- ARS 3.24.0 experiment-agent; implementation correctness review before GPU work.
- Finding confirmed against publication `e2629d1df56a1ccfa7dbc057c96a8251c810ce4e`; repaired source must be published/resealed by the parent.
- **14 focused CPU tests pass in 16.748 s.** No scientific model work, remote operation, objective/optimizer change or tolerance relaxation.

## Confirmed issue

The pinned DFlash provider constructs `dflash_target_layer_ids` as a tuple. SpecForge also constructs `specforge_step_options` and `specforge_model_provenance` as tuples; provenance contains nested tuples. Saving REUSE_AUDIT/PLAN through JSON converts them to lists. The original O_HALF wrapper passed that transport object to the historical exact, type-sensitive comparator. Consequently, a valid source/native contract would be rejected during runtime construction or migration. This was a metadata-transport launch blocker, not a discrepancy in gradients or an observed adaptation result. It was caught before the new GPU experiment.

## Narrow fix

`native_algorithm_contract()` decodes exactly those three pinned tuple-valued fields, recursively within their tuple payloads. Other JSON lists, including original coding event bindings, remain lists. Source states, runtime contracts and new scientific snapshots retain their native types; they are not normalized or rewritten. Exact `resume.differences` remains unchanged. Different parameter values, changed bindings, unknown mechanism state and wrong saved types still fail.

The existing tests now JSON-roundtrip `original_algorithm_contract` instead of handing the wrapper an already typed Python object. New checks cover all three fields and nested provenance, source immutability, preserved unrelated lists, wrong saved list-versus-tuple rejection, and scientific-value drift. The native tiny five-layer O_HALF event, complete restore and exact plain-O parity test still passes after JSON roundtrip. This fixes the test blind spot without changing model mathematics.

## Actual retained-state check

`check_actual_initial_migration.py` is prepared outside the execution tree for the parent to run with CUDA hidden. It reads both seeds' authoritative original O initial states and the real JSON REUSE_AUDIT, applies the in-memory LR-only migration, validates complete state and released-export equality, and checks unchanged process RNG/no CUDA initialization. It writes only a compact receipt; no model is constructed, no scientific state is restored into a runtime, and no training or serving is executed.

This subagent has not run that actual-state check remotely. The parent should record its receipt before the fixed compatibility bridge, then use the repaired published source/seal. The historical original O/K/K_HALF/KP validators and artifacts remain unchanged.

Evidence: `TRAINING_JSON_TRANSPORT_TESTS.log`, `JSON_TRANSPORT_REPAIR_REVIEW.json`, and the two-file source diff. Earlier CPU evidence and the unlaunched publication remain preserved.
