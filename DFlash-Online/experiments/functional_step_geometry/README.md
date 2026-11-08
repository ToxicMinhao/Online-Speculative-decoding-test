# K+ functional-step geometry: implementation and qualification

This additive path keeps the native K+ response objective, accumulation, clipping
and BF16 AdamW step. FSG filters the actual displacement through a two-column
functional-sensitivity sketch. SCALAR computes its own hypothetical FSG norm,
then uses the specified 24-step BF16 norm-matching search. Both check the actual
rounded candidate's norm and fixed-reference old-draft KL before committing.
Adam moments remain those of ordinary K+; no master FP32 model is introduced.

The current executable milestone is **bounded qualification only**. It does not
launch the proposed events-5–6 pilot, its 16 counterfactual diagnostic windows,
or its endpoint evaluation. The continuation/data contracts are prepared for
that later scope. Historical PR32 guards and reports remain unchanged.

## Code map

- `online_feedback/functional_geometry_math.py`: prefix weights, detached
  cotangents, streamed vector algebra, Gram solve, BF16 endpoints and acceptance.
- `online_feedback/functional_geometry.py`: capture the two native responses;
  advance Adam once; isolate auxiliary runtime; restore/commit with `copy_`.
- `training.py`: authenticate each dataset's complete KPLUS4 state, retain its
  original history, then append a separate continuation contract.
- `data.py` / `seal.py`: the prescribed next requests and byte-identical shared
  feature identities. Selection does not inspect generated outcomes.
- `qualify.py`: ordinary KPLUS, traced lambda-zero FSG, FSG, and SCALAR from the
  identical recovered state on each dataset; exact scientific-state checks.
- `native_oracles.py`: small-anchor native chunked/un-chunked VJP comparison,
  same-logit FP64 risk reductions, and a 512-anchor/four-vector resource fixture.
- `work.py`: persistent attempted/completed/failed counters and hard ceilings.

`run_qualification.sh ROOT` performs CPU checks, pinned-source/model/runtime and
storage admission, four shared feedback generations, and bounded qualification.
Those four answers are the first two prescribed event-5 requests per dataset;
they count toward the future shared-feedback budget and must not be regenerated.
Every temporary optimizer state is disposable. No qualification update becomes a
serving checkpoint for the pilot.

## Fixed numerical and work contracts

Strength 1; margin floor .05; one direction per response; FP32 geometry
probabilities/cotangents and BF16 native VJPs; streamed FP64 scalar/vector
reductions; vector chunks 262144; symmetric FP64 direct solve; no damping.
Gram negative-eigenvalue arithmetic allowance is
`64 * eps64 * max(1, column_count) * max(abs(Gram))`.
The risk deadband is the handoff's `16 * gamma * (1 + A)`, using independently
recomputed FP64 probabilities/log-softmax. Expanded-oracle covariance uses the
fixed `1e-6` relative cutoff. It is a qualification fixture, not a new arm.

The native small-anchor VJP check is pinned before execution to coordinate
`atol=1e-5, rtol=.016` and global relative L2 discrepancy at most `.016`.
This follows the existing BF16 qualification convention and does not change the
native objective or its precision. A failed check is recorded, not relaxed.

Hard cumulative limits: 8 disposable Adam steps, 16 ordinary presentations and
backwards, 96 additional response forwards, 32 additional backwards, and 8
serving generations. Startup/oracle work is included. A failed attempt still
consumes budget. Physical head/chunk/recomputation counts are separate.

The resource oracle repeats already selected anchor bindings deterministically
to 512 occurrences. This checks maximum anchor-count memory, not independent
feedback or useful learning. It retains four native gradient vectors for the
expanded working set. Original production anchor sampling is untouched.

## Interpretation

Qualification establishes execution correctness and resource feasibility, not
performance benefit. Lower protected KL is partly enforced by the mechanism.
If both FSG and SCALAR improve over KPLUS but FSG has no demonstrated advantage
over SCALAR, the necessity of directional geometry remains unestablished; this
does not establish equivalence. If SCALAR/KPLUS is also unresolved, the result is
inconclusive. Off-fit buffered-response diagnostics are not held-out evaluation.
