# Graduate development rules

This independent project lives at `/home/minhao2/Graduate` on Fenrir. It is a
staged refactor, separate from EAGLE_SPEC and the active A800 campaign.

For research context, read the existing project records before work:
`/home/minhao2/EAGLE_SPEC/AGENTS.md`, `RESEARCH_WORKFLOW.md` and
`RESEARCH_EXPERIENCE.md` in that same directory. These are historical/context
references, not runtime dependencies. Use the user's ARS skill for research
implementation and review. Read `/home/minhao2/EAGLE_SPEC/A800_RULES.md` before
any A800 operation. Current user instructions take precedence over older notes.

## Current scope

Step 4 is reviewed. Step 5A is implemented and locally qualified: live generation, native
teacher-feature capture and saving the same feedback format used by Step4,
with training/evaluation separation. Stop for review before continuous online
updates, publication/recovery integration or experiment campaigns (5B/5C).

- Build our own K+ method with bounded correct-prefix allocation. Retain the
  forward-KL calculation, first-error selection, DFlash/SpecForge and native
  verification components it actually uses.
- Omit standalone OSD/AdaFlash baseline modes, frozen/target-only comparison
  runners and historical ablation branches. Do not add a method registry for them.
- This scope applies to Graduate. Preserve historical baseline source and results
  and the active experiment's dependencies in their existing locations.

## Readable, purposeful code

- Every active file must support a retained workflow, its tests, configuration
  or documentation. Do not add speculative extension points or placeholder code.
- Use descriptive variable and function names, normal multiline Python and
  explicit control flow. Explain meaningful tensor dimensions and reductions.
- Keep functions focused without fragmenting a simple operation across many
  wrappers. Prefer direct code over a framework for hypothetical future needs.
- Each step extends the preceding working entry. Remove unused paths rather
  than shipping future-stage stubs. Let ordinary Python errors reach their
  source in the debugger; do not wrap the whole workflow in catch-all handlers.
- Validate external inputs and meaningful scientific/state boundaries. Avoid
  repeated checks, routine full-model scans, broad fallback behavior and repeated
  hashing in the normal learning loop. Preserve useful error messages and the
  original failure when recording evidence.
- Extract genuinely shared code before removing historical experiment imports.
  A module may be used through a configuration string or subprocess entry point.
- Preserve upstream attribution and licenses. Keep compatibility changes explicit.

## Debugging and functional equivalence

- F5 and CLI must invoke the same real application and configuration path.
  Keep `justMyCode: true`; support inspecting the required worker when computation
  runs in a separate process/environment.
- At every refactor step, update `.vscode/launch.json` and the reading guide
  together. Keep one integrated F5 entry for the current normal workflow, check
  its CLI arguments and required inputs, and identify the first useful
  breakpoint. Keep automated tests separate from the normal review path;
  overlapping test scenarios do not require repeated manual code review.
- Keep launch configuration simple: one module and the shared TOML path. Bind
  real data in TOML rather than asking the user for internal artifact paths in
  F5 input prompts. Verify actual debugger startup and a useful source breakpoint;
  a headless command with substituted test inputs is not an interactive F5 check.
  Disclose missing real inputs rather than claiming the workflow is ready.
- Debugging must expose model preparation, feedback, batches, losses, backward,
  optimizer updates and checkpoint publication. Do not substitute a toy loop.
- Preserve data roles, conditioning, masks, reductions, sampling policy, optimizer
  continuity and causal update/publication order.
- Hard numerical equality across hardware, kernels and debug runs is not required.
  Report differences and investigate whether functionality or learning semantics
  changed. Do not erase existing run-specific scientific validity requirements.
- Breakpoint-inclusive wall time is not a serving or training benchmark.
- Do not change numerical precision, batch geometry, kernels, model placement or
  scientific settings silently in the name of cleanup or optimization.

## Review and execution

- Follow the seven steps in README.md. Complete one step and stop for the user's
  review before moving to the next. Split an oversized diff before presenting it.
- Explain changed and removed files, behavior preserved, validation performed,
  remaining limitations and useful breakpoints.
- Keep the running experiment's source, configuration, environments and artifacts
  unchanged. Local preparation does not authorize a new GPU experiment.
- Profile before optimizing; measure time and memory alongside correctness.
  Treat changes to numerical execution separately from structural cleanup.
- Keep models, feedback, checkpoints, logs and historical archives outside the
  active source tree. Preserve necessary source provenance and recovery evidence.
