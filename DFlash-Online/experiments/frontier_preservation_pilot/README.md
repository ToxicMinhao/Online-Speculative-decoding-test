# Frontier preservation v1

This package is the bounded implementation and derivative-feasibility gate for
one warm-state CONTROL/FRONTIER_P continuation from PR59 PAIR1. Both intended
arms retain native K+PAIR; only the candidate replaces C's prefix allocation.

The first executable milestone is the native derivative prerequisite. No
production event is admitted by passing CPU tests or this prerequisite alone.
`math.py` implements the frozen reachable endpoint, assessment, signed
mass-conserving allocation and optimizer-shadow VJP. `fork.py` validates and
imports the original complete parent without rewriting its history. These are
separate from the ordinary native optimizer. The parent remains read-only.

`native.py` samples one fixed pair of retained **training** responses, restores
PAIR1 under its original validator, verifies ordinary first-order backward,
then attempts parameter-to-hidden JVP by forward AD and reverse-over-reverse.
It performs zero optimizer steps, zero vocabulary-head calls, and no generation.
This tests a necessary derivative primitive, not the full sensitivity formula.
A missing native derivative is a feasibility blocker, never a failed efficacy
result. No alternative attention kernel or finite-difference score is substituted.

## Commands

Use the qualified environment and source the A800 helper. Put this package,
`training/` and `inference/` on `PYTHONPATH`. All paths below are explicit arguments;
`SOURCE.json` binds every staged file to its committed identity.

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s experiments/frontier_preservation_pilot -t . -v
python -m experiments.frontier_preservation_pilot.prepare \
  --parent-plan "$PARENT_PLAN" --root "$NEW_RUN" \
  --source-manifest "$SOURCE_MANIFEST" --execution-commit "$EXECUTION_COMMIT"
python -m experiments.frontier_preservation_pilot.qualify --plan "$NEW_RUN/PLAN.json"
python -m experiments.frontier_preservation_pilot.analyze --root "$NEW_RUN" --output "$REPORT_DIR"
```

Preparation admits only this derivative check with a 6 GB additional storage
allowance, the 400 GB reserve, an idle training card and a 15-minute child limit.
The cumulative handoff limits remain 4 elapsed hours, 8 occupied card-hours and
40 GB additional storage. Parent tensors are reused read-only. Every attempted
primitive is recorded before execution; failures retain traceback and pending
work. Final restoration checks weights, Adam state, RNG and counters. No
automatic retry is implemented. Production collection/training/evaluation needs
the full new qualification and a separate phase admission; it is deliberately
not launched by this prerequisite runner.

The shadow promotes native weights, gradients and Adam moments to FP32, includes
global clipping and the gradient-dependent Adam denominator, then uses native
BF16 forward realizations. This is mixed-precision shadow sensitivity, not an
exact derivative of discrete BF16 updates. Allocation centers scores by first
subtracting one pool score, then their weighted mean difference, an algebraically
identical FP32 form avoiding cancellation for nearly constant pools. It introduces
no threshold, epsilon or tunable coefficient. Native clipping's existing 1e-6
and the existing response denominator remain unchanged.

If the native derivative gate blocks, retain the work, source and restoration
receipts and publish the exact limitation. Do not claim that the production
trainer, full native coefficient sensitivity, disabled-path equivalence or
serving comparison has been qualified.

Current native outcome: [bounded repair report](../../reports/dflash_frontier_preservation_20261125/repair/REPORT.md).
The original unsupported-operation [failure report](../../reports/dflash_frontier_preservation_20261125/REPORT.md) remains historical evidence.
The scoped attention/SiLU derivative now executes with exact native output and
first gradients, but its full-model numerical consistency check still fails.
Neither production nor a numerical-tolerance change is admitted.

The repair entry point is `python -m experiments.frontier_preservation_pilot.repair_native --help`.
It exposes fixed-input analytic, localization and cotangent-transpose diagnostics;
`prepare`/`qualify` bind their prior receipts and cumulative costs. These are
zero-update prerequisites, not alternate production trainers. Run only against
the recorded source, state and plan; the retained failure does not authorize an
automatic retry.
