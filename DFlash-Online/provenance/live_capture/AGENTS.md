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
