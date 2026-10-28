# Working on the online loop

Follow `/data/sources/minhao/EAGLE_SPEC/AGENTS.md` and `A800_RULES.md` for research
scope and server operations.

Write this code for step-by-step inspection by the researcher. Use descriptive
names, ordinary multiline Python, and short functions with an explicit purpose.
Keep inference, feedback preparation, training, and evaluation easy to follow.
Do not compress statements to reduce line count or add generic abstractions for
hypothetical future services.

Validate external inputs and scientific boundaries once. Preserve masks,
conditioning, optimizer/RNG continuity, exact replay comparisons and publication
ordering. Avoid repeated SHA-256 checks, whole-model scans, broad fallback logic,
concurrency machinery and historical one-off recovery paths in the normal loop.
Put expensive correctness checks in tests or the explicit qualification runner.
Keep upstream sources and historical experiment artifacts intact.

## Source provenance for reports

Before any future experimental launch, record the Git commit, working-tree status and exact source/configuration bindings. Every report must state that revision; publish uncommitted patches before claiming the commit alone describes execution. For historical runs, cite the archival publication commit plus the per-run source manifest, explicitly distinguishing it from an execution-time commit. Preserve sealed reports and attach provenance sidecars rather than rewriting their bytes.
