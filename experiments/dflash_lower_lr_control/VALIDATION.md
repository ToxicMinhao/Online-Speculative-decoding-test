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

**Native execution completed:** the first complete group passed correctness, and all
five remaining groups continued unchanged. The A800 run of implementation commit
`673e75efb72595f718b41c03fb2c7aa0e71fc40d` completed in1,186.57s with exit0:
48 optimizer steps,96 presentations,48 natural generations and48 KL evaluations.
All48 outputs and termination conditions match the retained references. No invalid
comparisons, zero-supervision cases, retries or extra warmups occurred.

An independent audit passed1,416,556 checks; all332 transferred compact files passed
size/SHA256 verification. All six final L8 checkpoints remain on A800; existing
comparators were preserved. Both GPUs are idle. No scientific settings changed;
no agreement-filter or correction-bank campaign resumed.

The executed source digest is
`0fe9edc78043453fdd716b7994cf99e926a6206609ced984f9497a2da630e2f3`.
Subsequent changes to this validation note do not change the executed code revision.
Private raw measurements, per-request tables, configurations and audit artifacts
remain outside Git. Historical/fresh timings are not a serving-speedup comparison.
