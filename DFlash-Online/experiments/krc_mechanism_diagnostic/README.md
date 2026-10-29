# KRC mechanism diagnostic

Replay historical events 2 and 4 independently. The original full R correction
always continues; at ten fixed windows per event, measure temporary R_FULL,
R_HALF, D_FULL and K_ONLY candidates from the same post-K weights. No serving,
new feedback, candidate export, tuning or extra Adam update is permitted.

The complete authorized protocol is retained in the private run's HANDOFF.txt.
Model, optimizer, objective, sampling, startup and runtime settings are inherited
unchanged. New files instrument the existing qualified KRC implementation.

## Work and validity

- Original replay: 256 Adam steps, 512 ordinary presentations, 1,536 forwards.
- Twenty probes: 600 additional forwards, at most 40 additional backwards.
- Total: 2,136 response forwards and at most 1,064 backward equivalents.
- Separate startup: at most four disposable forwards/backwards, zero Adam.
- Four candidate slots per sampled window; fixed order ends with a physical
  K_ONLY restoration witness. Invalid temporary candidates remain visible.
- R and D each use their own pre-K masks, anchors and global pair denominator.
  R_HALF scales the FP32 displacement before BF16 rounding.
- Exact restoration checks cover weights, buffers, Adam, gradients, accumulation,
  RNG, modes, sampler and cache bindings. Sampled no-op windows check parameter
  values instead of unchanged version counters, because temporary copy_ advances
  those counters without changing the restored values.
- Final decoded scientific states must match original KRC2 and KRC4. Paths,
  timing and diagnostic receipts are compared as metadata separately.

D is buffered training data, not independent held-out data. Token oracles are
hypothetical substitutions, not realizable update or serving-speed claims.
Prediction residuals can reflect finite precision and nonlinear/nonsmooth changes;
this diagnostic alone cannot uniquely separate those causes.

## Execution and evidence

`prepare.py` verifies retained source/input bindings, adds the new files, and
seals the fixed windows. `preflight.py` checks runtime and peak storage admission.
`run_campaign.sh` runs only the two episodes. No automatic retry is configured.
Each execution records its exact Git commit; private tokens, responses, features
and runtime snapshots stay outside Git. `build_report.py` produces compact tables
and a report from retained records without GPU execution.

The replay uses exact transient CPU snapshots for diagnostic isolation. These
copies and hashes are instrumentation overhead, not proposed adaptation work.
Retiring an independently verified duplicate replay endpoint does not retire its
historical original; metadata and the exact removal/recovery receipt are retained.
