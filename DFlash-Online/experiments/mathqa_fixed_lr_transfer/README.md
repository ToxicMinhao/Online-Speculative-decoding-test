# MathQA fixed learning-rate transfer

This is the fixed November 16 N/S comparison, not a new loss or controller.
Both arms restore the original complete PR32 MathQA KPLUS4 at 512 Adam steps
and 1,024 presentations. N keeps LR `3e-4`; S uses literal `2.88e-4` in the
configuration, optimizer recipe and live Adam groups. K+ arithmetic, BF16,
sampling, clipping and the inherited 7,424-step horizon are unchanged.

The retained historical native feature pools and event-five state were retired,
so this run uses the handoff's fresh two-arm fallback. One common origin-produced
128-response pool trains both event-five arms. N5 and S5 then produce their own
128-response event-six pools. Both continuations finish at 768 cumulative Adam
steps. No event-seven continuation is included.

The new 256 scored and two warmup MathQA inputs are selected and sealed before
generation using the handoff's input-only hash ranking. A separate frozen KPLUS4
run supplies all 258 canonical outputs. They are validity references under the
native target verifier, not a target-only speed baseline.

## Entry points

- `prepare.py` authenticates historical seals and new request manifests, writes
  arm configurations and prices peak storage.
- `preflight.py` checks the assembled source, actual runtimes and complete
  recovery/export on CPU before any model work.
- `qualification.py` checks native/factor-one identity, S's actual rate and
  exact own-state replay with five disposable Adam steps per physical card.
- `launch.py` owns the cumulative clock, worker schedule and measured admission.
- `task_runner.py` executes common collection, canonical references, one arm's
  training or collection task, or final evaluation.
- `training.py` and `state.py` adapt the existing native trainer's state guards;
  they do not implement a different training objective.
- `evaluation.py` and `analysis.py` preserve all raw returns, validate both arms,
  and reduce the prescribed paired outcomes.

An assembled server package is launched with the pinned training interpreter:

```text
python -m experiments.mathqa_fixed_lr_transfer.launch --plan /approved/run/sealed/PLAN.json
```

The actual launcher sources the server's approved environment first. Source
revision, physical GPU UUIDs, original state identities and exact runtime paths
are sealed in the plan. Private inputs, generated answers, feature tensors and
optimizer states remain outside Git.

## Fixed budget and validity

The planned work is 512 production plus ten disposable Adam applications;
1,056 response forwards and backwards each including qualified startup; six
trainer initializations; 384 feedback generations and teacher preparations;
258 canonical generations; eight qualification generations; 48 warmups and
1,536 scored generations. The handoff caps remain 528 Adam, 1,088 forwards and
backwards each, eight initializations, four elapsed hours and four summed
occupied-card hours. Failed attempts remain charged; no automatic retry.

Independent untimed workers may share the two A800s only when host/GPU memory
allows. Cross-card numerical mismatch may trigger the explicitly allowed
same-card sequential fallback; provenance or same-card correctness failure
stops. Final timing always uses one fixed GPU with the other idle and no
training or transfers.

Peak storage includes all four new persistent boundaries, both disposable
qualification payloads, three unique feature pools, atomic publication,
compiler allowance and evidence. Only new redundant capture tensors are retired
after the existing exact decoded-feature equivalence check. Older assets remain
untouched. The new-allocation limit is 140 GB with 400 GB free after peak writes.

Final serving is greedy BF16, block16, 2,048 tokens or EOS, context4,096,
concurrency1, capture disabled and the existing cache/publication policy.
Every output and termination must match its origin reference, and each arm's
native counts must agree across repeats. Invalid/incomplete comparisons remain
visible and do not enter benefit claims.

## Interpretation

Primary outcomes are native accepted proposals per verification and summed
per-request mean latency. Ten thousand common whole-request bootstrap draws use
NumPy PCG64 seed1711, source-ID order and linear quantiles. Native acceptance
and stop-clipped acceptance remain separate. Calls and acceptance are linked
under identical outputs and are not independent mechanism evidence.

The prescribed decision precedence is practical transfer, clear harm, practical
time gain unsupported, then unresolved/mixed. Practical transfer requires both
primary lower bounds above zero, at least0.5% timing reduction and favorable
timing in all three repeats. This is one origin/seed and dataset comparison;
it cannot establish optimal learning rates, an adaptive controller, or a new
paper contribution. The earlier allocation candidate C remains closed.
