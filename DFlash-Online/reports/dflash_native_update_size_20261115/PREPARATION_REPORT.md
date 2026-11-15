# Native update-size attribution: implementation and admission

Status on November15: implementation and focused CPU checks are complete; actual
CPU serializer Gate0 and GPU qualification are not yet complete. Source staging
is blocked by intermittent A800 SSH connections. No new Adam step, GPU model
forward, parent reconstruction or scalar acceptance score has run.

## Question and fixed comparison

PR47's E intervention omitted current correct-prefix gradients and improved
independent local acceptance relative to the ordinary native endpoint N. Its
realized BF16 update was also smaller (89.57–97.00% of N). This diagnostic tests
that measured alternative explanation using two ordinary native AdamW learning-
rate controls whose realized norms bracket E within0.25% on each side.

The new implementation does not modify the production K+ objective or optimizer.
Each disposable control restores the same P state, installs fresh immutable
native preclip gradients, changes only the actual optimizer-group learning rate,
and applies the original finite check, clipping and BF16 AdamW step. Native Adam
moments/counters and clipped gradients must reproduce N exactly. Only N advances
the single KPLUS6→7 trajectory.

At every fixed window8,16,…128, norm-only bisection permits at most10 interior
trials. All32 logical control bindings must be sealed before outcome scoring.
Historical E measurements are authenticated and reused; the14 missing E models
are not regenerated. Missing P/N files must reproduce their original SHA256 and
size using the original exporter, after CPU serialization qualification on all
six surviving P/N/E boundary exports. Failed copies remain preserved.

## Implemented checks

All **67 focused CPU tests pass**. Coverage includes bracket/exact-match/plateau/exhaustion behavior,
nonfinite rejection, fresh and None gradient handling, zero-LR neutrality,
effective-LR checks, Adam/clipping parity, complete restoration, interrupted-work
accounting, export aliases, original-file failure preservation, source/runtime
admission, exact control joins, fixed sentinels and request-level analysis.

Separate implementation reviews found and repaired orchestration errors involving
a one-process GPU inventory, runtime/source validation, interrupted-record recovery,
and report completion status. These are CPU implementation findings, not failed
scientific runs or changed experimental settings. Native integration remains
unqualified until the bounded six-case test passes.

An independent CPU re-reduction of the12,288 retained P/N/E rows reproduces PR47's
historical E−N estimate exactly: **+0.015625 accepted proposals per fixed block**,
95% request-bootstrap interval **[+0.00244140625,+0.02880859375]**. This is reused
historical evidence, not a new result for either size control.

## Budget and expected duration

- One unchanged native event:128 steps,256 response presentations.
- At most160 cached-gradient calibration steps and6 disposable qualification steps.
- At most260 response forwards/backwards each, including qualified startup.
- At most8192 main endpoint blocks,24 fixed P/N/E sentinel blocks and8 neutral
  blocks. Reconstruct64 saved parents once; no complete-response generation.
- One A800,180 cumulative occupied-card minutes, including failed work.
- New allocated storage ceiling110GB; retain400GB free after maximum writes.

Planning estimate: **1.5–2.5 GPU hours after preparation**, with a hard3-hour cap.
CPU provenance/serialization, implementation and connection delays are separate.
The estimate must be reassessed after qualification and the first calibration
window; historical native kernels alone are not a valid duration estimate.

## Storage and access

The successful live check at2026-11-15 10:42:08UTC found705,720,819,712bytes free,
149,777,620,992bytes available host RAM and both A800 cards idle. This supports
admission in principle, not a later live guarantee. Subsequent staging/read-only
connections stalled or timed out before login; the transfer state must be
inspected before reuse. No experiment was launched.

This handoff does not require an arbitrary extra deletion total. No further old
artifact was positively proven disposable by the bounded dependency review, so
no new historical cleanup has been performed. The earlier200-file cleanup is
not counted again. Selected paper endpoints, native recovery states, current
features and compact evidence remain protected; ambiguous assets remain REVIEW.
The additional compiler environment used by the native launcher is also retained.

## Decision boundary

Primary advancement requires E to beat **both** qualified size controls, with
both paired95% lower bounds above zero and a complete valid grid. Otherwise the
result is unresolved or supports a size explanation as specified. No equivalence
claim follows from an interval crossing zero. This local reused-panel experiment
cannot establish full-request acceptance, speedup or an error-only trajectory.
No additional continuation, tuning or missing-E regeneration is authorized by
this implementation.

Actual source and any later execution/report commits are recorded separately in
PUBLICATION.json. Private raw records, responses, features and checkpoint tensors
remain outside Git. Reviews use separate agents in the same model family and do
not constitute independent native replication.
