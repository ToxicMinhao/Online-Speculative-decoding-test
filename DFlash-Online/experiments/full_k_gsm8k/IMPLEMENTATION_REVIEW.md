# Full K/K_HALF implementation review

## Material Passport

- Workflow: ARS 3.24.0 experiment-agent, implementation and execution validation.
- Authorization: fixed October 31 full GSM8K K/K_HALF handoff.
- Source parent: `7276d3df6bffb3206c86f30f45b9a867fcad5120`.
- Exact execution revision: recorded in each run's sealed `PLAN.json` and report.
- Review: primary implementer plus scoped reviews by separate collaborating
  agents. Some reviewers authored components; this is disclosed same-team review,
  not an independent GPU reproduction.
- Status at publication: CPU tests and original-artifact inventory passed;
  native correctness and resource admission remain execution gates.

All 58 focused CPU tests passed in 38.922 seconds. Coverage includes unchanged
native-K numerical behavior with observations enabled, independent initialization,
uninterrupted versus own-state continuation, true event-58 contracts, exact
budgets and four inherited correctness generations, unknown-work preservation,
rolling retention, final schedule balance, native record validation, cumulative
response accounting and fixed request-level bootstrap. Tiny fixture training is
CPU-only and is not full-model GPU qualification.

The authoritative read-only A800 inventory checked 15,379 file identities,
6,144 original F/O executions and 285,942 native verification blocks. It
reproduced 50,584 F calls and 44,730 O58 calls on the original 1,024-request panel.
No model forward, feature reconstruction or optimizer operation was performed.
All original input bytes remain unchanged.

The original full-O runtime/model sources match current source except the two
reviewed `online_feedback/objective.py` and `online_feedback/training.py` files.
Their historical bytes remain in the provenance object store. Those existing
changes add first-error selection, associated normalization and experimental
interfaces; this campaign changes neither file. Ordinary O's numerical path and
teacher/runtime behavior remain available unchanged.

Resolved before launch: the boundary-file inventory initially matched auxiliary
validation/retirement receipts as well as event boundaries; it now selects only
`event_NNN.json`, with regression coverage. A tiny-test double-close issue was
fixed by preserving idempotent cleanup. A combined test attempt used the shell's
Torch-free Python and was rerun with the existing qualified CPU environment;
this was an environment selection error, not a model qualification failure.
Original failed logs are retained privately.

A final partial-failure review also found that a hard-killed worker could leave
missing terminal clocks that prevented its audit from being written. Missing
clocks now produce explicit errors and unavailable values while preserving the
audit. Two regression tests cover this path. No scientific execution preceded
this repair, and complete-run accounting is unchanged.

The enclosing resource deadline now unwinds Python cleanup instead of abruptly
leaving the worker's separately launched native process running. A CPU test
checks the termination path; post-run retrieval separately requires idle GPUs
and no remaining experiment process. Neither adds scientific work.

The peak storage model includes atomic overlap and both completed/active arms,
with separate evidence, compiler, workspace and reserve allowances. A
conservative metadata estimate fits its 16 GiB compact allowance, but actual
capacity is rechecked before each event. New artifacts alone use rolling
retirement after verified successors. Historical offload requires complete
byte verification before any A800 copy is removed.

No performance result is implied by these checks. The run continues through
valid harmful or null outcomes, stops on concrete correctness/resource failures,
and retains all attempted/invalid/unknown/unattempted work. There is no automatic
retry, tuning, extra probe, O retraining or follow-on experiment.

## November 1 operational early-launch update

The user authorized retiring additional completed intermediate optimizer states so
training can begin while the existing historical archive copy continues. The
scientific algorithms, data, optimizer settings, work budget and endpoint schedule
are unchanged. An opt-in, run-bound transfer-clearance barrier now prevents the
final timed backend from starting before the coordinator verifies transfer
quiescence. Waiting is recorded separately within campaign wall time. The
coordinator must keep transfers stopped until campaign exit; the marker is an
operational attestation, not automatic discovery of arbitrary external jobs.

All 65 focused CPU tests passed in 38.857 seconds, including seven new barrier
checks. This is not a GPU qualification result. No model work was executed under
the preceding preparation seal. Historical archival candidates remain protected;
additional retirement has its own exact-path, hash, dependency and loss-of-resume
receipts outside the scientific source tree.
