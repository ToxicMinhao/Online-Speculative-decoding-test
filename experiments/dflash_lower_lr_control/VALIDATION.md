# Implementation validation

The user authorized this bounded control after approving the handoff, including
correctness-only first-group qualification and continuation of all five remaining
groups even for unfavorable results. Both S8 and L8 originally start from the same
released weights and fresh AdamW. S8 is now reused as a completed artifact.

Local CPU validation: **244 tests passed** across the lower-LR control and the
unchanged baseline-learning-curve and KL-to-native diagnostic packages. Compilation
and whitespace checks passed. Tests include BF16/FP32 toy AdamW equivalence, exact
training exposure matching, separate source/held-out anchor contracts, response-only
masks, RNG restoration, honest L8 identities, trainer shutdown before serving,
durable partial accounting, native token/counter validation and report revalidation.
A real-evidence CPU plan also binds all six original groups, 72 baseline trajectories,
24 subsequent source trajectories and 48 retained F/S8 KL/native pairs.

The new files do not modify the historical training guards, recipes or implementations.
Source hashes are recorded at execution. Only code, tests, documentation and compact
artifact hashes belong in Git; private prompts, responses and runtime artifacts do not.

**Native qualification has not run.** Local tests do not establish GPU memory fit,
installed-runtime compatibility or real output preservation. The A800 SSH-agent socket
was unavailable during implementation; execution requires restored approved access,
current GPU/storage admission, installed-runtime checks and the planned first group.
No gradient-filter or correction-bank campaign resumes.
