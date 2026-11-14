# Threatened-repair replay implementation review

## Material Passport

- Stage: authorized bounded implementation and execution; native qualification pending.
- Source review: MIR author implementation `35eda78bcdd35025b16b0b1039a926d34aad4851`.
- Prerequisite: user-approved scoped reproduction exception for this pilot only.
- New model work before publication: zero.

The implementation preserves one ordinary KPLUS6→7 trajectory and sixteen
same-state disposable comparisons. Selection uses only original memory tags and
the actual native pending update. U/T receive the same positional-weight budget
and padded full-response layouts. Replay gradients enter after the two native
response losses have been averaged and before their one global clip/Adam step.
Every branch restores parameters, Adam state, counters, RNG, sampler, pending
gradients and mutable runtime state. The physical-work ledger is not restored.

A separate reviewer inspected selection/normalization and the controller. A
second reviewed native cache reconstruction and campaign accounting. CPU tests
cover actual TrainerCore and BF16 AdamW fixtures, state restoration, failure
persistence, matched control selection, partial work, exact-state aliases and
independent request-level aggregation. Details are in CPU_REVIEW.json.

Review fixes bind the child interpreter and physical GPU explicitly, preserve an
original failure if reporting also fails, and forbid a completed campaign status
when final validity fails. Original PR44 training and later evaluation-continuation
artifacts have separate authenticated paths; no later checkpoint substitutes for
the required origin.

Native qualification remains required: all retained proposal/canonical checks,
independent same-layout memory observation, exact disabled/zero/padded-zero
neutrality, nonzero sparse-gradient activation, complete state restoration and
final historical NATIVE7 parity. No passing CPU fixture is described as GPU
qualification or scientific evidence.

Evaluation is delayed until training releases the card and uses exact saved
per-window weights. The full 64-request/four-start grid is sealed before training;
its outcomes cannot affect branch selection. Three occupied card-hours remain a
hard limit, including startup and qualification. Stop incomplete rather than
shrink the panel. Provisional execution estimate is one to two GPU hours once
staging and native admission succeed.
