# VFD second-pass selection: fixed four-event screen

This experiment keeps PR25 K+ (Verification Frontier Distillation) unchanged and
compares FULL, ONCE, RANDOM64, STATIC64 and PROGRESS64. It restores the authenticated
PR25 event-zero weights, empty BF16 AdamW and seed17 RNG, not K+58. Scientific
reference execution is `e0a9d3c235e1efd15404c549282d21c99243a21c`.

Each of four events collects the next128 original GSM8K responses using the
preceding FULL export and prepares their teacher features once. Event1 executes
pass1 once and forks its complete state. Subsequent events restore each policy's
own complete state. No policy borrows FULL's weights or optimizer thereafter.

The existing loss, full-block attention, canonical target conditioning, training
sampler, positional weights, response normalization, clipping and BF16 AdamW
are reused directly. FULL's native response permutations control all policies'
ordering. Each executed pass consumes its own native permutation draw; non-FULL
policies replace only its order with FULL's recorded order. A subset filters the
order, never sorts by score. ONCE does not execute a second pass or its RNG draw.
Each policy's subsequent anchor draws remain native and are recorded, not forced.
The inherited7424-step horizon is retained; the screen ends after four events.

`online_feedback/vfd_scoring.py` implements the exact detached16-anchor scoring
rule in the handoff. `training.py` exposes a resumable pass boundary around the
existing trainer. `worker.py` runs one own-state job. `campaign.py` controls the
shared feedback barrier, isolated workers, retirement and final endpoint.

The input-only selection pipeline prefers unused questions in available retained
inventories, then orders by the declared hash. It does not use generated answers
or measurements. The200-request panel is train-disjoint; prior project overlap
is disclosed. Canonical references require exact prompt/model/runtime provenance;
missing references are generated once, outside the timed endpoint.

## Fixed work

| Work | Main campaign | Disposable qualification |
|---|---:|---:|
| Feedback generations / preparations |512 /512|0|
| Training presentations / Adam steps |3328 /1664|4 /2|
| Scoring response forwards |at most1408|at most8|
| Timed generations |2000|0|
| Missing canonical generations |at most200|0|

The logical standalone totals are3840presentations/1920steps: the shared first
pass counts once physically but is charged to each policy logically. Scoring is
charged only to policies that need it. Logical updater time includes the pass
and scorer worker wall time, including input loading; initialization, restoration,
preparation, persistence and export remain separate. Score forwards may contain
projection chunks; those are additionally counted, never additional samples.

Qualification uses the first four immutable feedback packets on disposable state:
four before and four after score calls, one native four-presentation pass and two
Adam steps. It checks exact restoration, score bounds/gates, RNG isolation and
complete-state fork equality. It then restores event-zero state and discards all
temporary learning. CPU integration tests additionally compare the adapter with
unchanged native K+ weights, optimizer, RNG, anchors and counters exactly.

## Endpoint and decision

Only event4 is evaluated. All five checkpoints use two sweeps on the same200
requests, arranged in20-request blocks with rotating method order and reversed
order in sweep2. Serving uses the pinned SGLang/FA3 verifier, BF16 greedy block16,
2048tokens orEOS, context4096, concurrency1, captureoff and full native cache
flushes. Training workers exit before timing. Every generation is saved and
validated for exact canonical tokens/termination and repeat-consistent counts.

Primary: PROGRESS64 complete-request time reduction versus FULL, from summed
per-request two-repeat means. The paired request bootstrap uses10000PCG64draws,
seed20261104. Control comparisons are declared in `statistics.ANALYSIS_RULES`.
The three-way decision follows the handoff. For an operational definition of
"clearly dominates", a simpler RANDOM64/ONCE policy must have lower measured
updater time and a PROGRESS-versus-that-policy latency interval wholly below0.
For equivalent declared efficiency criteria, ONCE is preferred to RANDOM64 and
then PROGRESS64. This is a fixed decision convention, not an extra experiment.

## Resources and stopping

Peak admission includes five rolling model/Adam states, one atomic replacement,
the shared pass1 state, five final exports, worst-case128 prepared packets plus
one replacement,5%tensor margin,16GiBcompact evidence,4GiBcompiler allowance,
2GiBworkspace and8GiBreserve: approximately104.6GB free at initial admission.
Retire duplicate raw features only after exact prepared-row validation. Retire
an event pool only after all five own successor states pass. Keep compact tokens,
anchors, scores, measurements, failures and final recovery states. Deleted
intermediate tensors are retired, not archived.

Expected native execution is approximately2.5–4hours based on historical measured
K+ training/serving, before observing new scorer/runtime costs. Initialization,
compilation, qualification and all work are reported. A12-hour operational ceiling
prevents an unattended stuck process; it is not a performance-based stopping rule.
Correctness/resource failures stop with evidence. Null or harmful valid results
continue. No automatic retries, score tuning, additional arms or longer run.
