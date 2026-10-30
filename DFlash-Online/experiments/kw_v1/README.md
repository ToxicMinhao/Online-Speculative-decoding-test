# KW-v1: learned response allocation on the retained K trajectory

KW tests whether weighting the two responses in K's existing Adam window from
next-window training loss transfer improves adaptation. O and K continue their own
complete event8 states. KW starts as an exact complete K8 copy, including BF16
weights/moments/counters, RNG and history, with a zero five-parameter controller.
The fixed target, native DFlash architecture, K token selection and loss,
response-mean accumulation, anchors, optimizer and serving configuration remain
qualified. No result is assumed to improve.

Each response's ordinary forward supplies five detached FP32 features: selected
position-weight fraction; normalized first-error location (16 for no error);
no-error-block fraction; nonnegative loss divided by one plus that loss; and full
response length divided by2048. Empty blocks do not enter block fractions. Their
difference is `d`; CPU FP64 controller arithmetic computes
`a = .25*tanh(phi_entry·d)` and weights `(.5+a,.5-a)`.

The response hooks observe native half-loss backward contributions `u1,u2`
without altering BF16 accumulation. At zero allocation the native gradient `G`
remains unchanged. Otherwise the stored gradient is
`BF16(FP32(G) + float32(2*a)*(FP32(u1)-FP32(u2)))`, using distinct subtract,
multiply and add operations. Missing response gradients contribute zero while the
native active/None parameter set remains unchanged.

Two isolated actual native BF16 optimizer previews, including global clipping,
start from the same pre-step model/Adam state at clipped `a±1/16`. Their FP32
parameter difference divided by the actual allocation difference is eligibility.
The next ordinary unweighted, unclipped training gradient supplies its dot product
using fixed-order FP64 chunks. The producing window's cached derivative updates
`phi` by normalized-credit descent: RMS decay.95, epsilon1e-12, clip±1, learning
rate.05 and coefficient L2 cap2. Zero increments leave `phi` bitwise unchanged.

The current allocation freezes before the previous window's credit updates the
controller. Therefore the first two zero-initialized windows exactly match K and
the first learned allocation can appear in window3. Pending eligibility clears at
every pass boundary; controller/RMS state persists. Each event has128 committed
steps,256 presentations,256 previews and126 credits. Quantized optimizer secants
are an approximate first-order training-loss signal, not an exact derivative
through BF16/future Adam or an acceptance objective. Controller credit comes from
shuffled training responses, not held-out data or necessarily the next user.

## Fixed execution

Events9–12 each collect128 new feedback responses from preceding O8/O9/O10/O11.
O, K and KW share immutable feedback/features and retain their natural RNG draws;
K/KW verify response order and anchors against O. Selection recovers the preceding
campaign's consumed source cursor, accounting for skipped candidates. The new final
panel uses isolated seed20261030 after source-order exclusions and deduplication.
Prior training and all earlier evaluation/probe content, including seed20261027,
are excluded. Only question columns are read. Known older unrelated-project
exposure is annotated after selection.

Retained development/additional/fresh panels retain F/K1/O8 canonical outputs and
run at events10/12. The new final64 runs only at12, taking the first O12 output as
canonical and deferring earlier K/KW checks. Each request has three balanced-order
repeats. The first eight development10 requests supply the included72-generation
correctness gate. No generation warmups, target-only references or extra probes.

Production totals are512 feedback generations,1536 live Adam steps,3072 response
presentations,1024 optimizer previews,504 credits and4032 evaluations. Every arm
ends at cumulative step1536/presentation3072, retaining horizon7424. Bounded
qualification is separate: at most8 response forwards/backwards and16 disposable
optimizer applications, zero serving generations and persistent updates. The four
longest retained event8 responses are sealed before qualification. Both ordinary
K and zero-initialized KW execute the same two windows from identical K8 clones;
full scientific states must match exactly. Captured gradients support nonzero
optimizer-only tests without another model pass.

## Commands and recovery

From the repository, prepare inputs locally without model execution:

```bash
PYTHONPATH=.:tests /home/minhao2/.venvs/dflash-specforge/bin/python -m experiments.kw_v1.prepare --output /home/minhao2/EAGLE_SPEC/research/dflash_kw_v1_20261029
PYTHONPATH=.:tests /home/minhao2/.venvs/dflash-specforge/bin/python -m unittest experiments.kw_v1.test_campaign experiments.kw_v1.test_resume experiments.kw_v1.test_evaluate experiments.kw_v1.test_report test_response_weighting -v
```

Pin the execution commit, exact commands and source/input identities in the sealed
run before transferring. Read the workspace A800 rules before any remote action.
On the qualified remote environment, `python -m experiments.kw_v1.preflight ROOT`
checks source/runtime/model pins, original O8/K8, the four retained features, idle
hardware, host memory and the complete maximum-length disk budget. Production
requires `ROOT/IMPLEMENTATION_QUALIFICATION.json` with passed exact-neutral,
isolation and full-buffer memory checks plus bounded work counters.

After preflight, in the same pinned training environment and on the admitted
GPU, run the one disposable qualification before the campaign supervisor:

```bash
python -m experiments.kw_v1.qualify --plan /data/sources/minhao/runs/dflash_kw_v1_20261029/sealed/PLAN.json --root /data/sources/minhao/runs/dflash_kw_v1_20261029
```

The planned qualification uses 14 optimizer applications: two ordinary K steps,
two KW live steps, four KW previews, and six optimizer-only native/preview checks
at allocations −0.25, 0.137 and 0.25. It uses eight response forwards/backwards
and no warmup, generation or persistent update. Attempt ledgers and completed
response records survive a failure. Complete decoded model, Adam, RNG and counter
equality is required; controller state is additional and has no K counterpart.

```bash
bash /data/sources/minhao/DFlash-Online/experiments/kw_v1/supervise.sh /data/sources/minhao/runs/dflash_kw_v1_20261029 28800
```

The supervisor has one explicit wall-time bound, never retries and preserves its
exit/failure records. After an explicitly diagnosed interruption at a complete
idle boundary, the same pinned environment can run:

```bash
python -m experiments.kw_v1.campaign /data/sources/minhao/runs/dflash_kw_v1_20261029 --resume-after 10
```

Only complete9/10/11 boundaries are admitted. Restore checks exact state/export,
counters, controller and historical identities; any later feedback attempt or
partial-event artifact blocks this entrypoint. There is no mid-pass recovery,
credit reset, repeated pool or automatic budget extension. An interruption inside
scientific work needs an explicit recorded resolution before any further work.

## Retention, costs and decision

Original O8/K8 remain untouched. Every new boundary is retained completely;
event10/12 states and exports are permanent. Rolling9/11 states and exports may
retire only after all successor states are verified and required evaluations close.
Every exact retired path, identity, retained successor and lost resume capability
is logged. All new feedback/features and compact verification/failure records stay.
The disk budget includes every maximum-length capture/prepared copy, pending state,
export workspace and8GiB reserve; no transfer-throughput assumption is used. Each
phase also rechecks actual space. Qualification checks full gradient/eligibility/
preview memory, rather than inferring fit from historical training peaks.

Window logs retain compact features, gradients' scalar summaries, allocation,
credit, controller state, preview differences, timing, memory and counters. Full
gradients/eligibilities are not archived as diagnostics. Cost accounting separates
updating, preview/controller work, saving/exporting/loading and serving; nested
timers are not added twice. No training or transfers overlap measured serving.

Primary is KW12 versus K12 call reduction on new final64. Report all seven fixed
panel/endpoints, the pooled four final panels, exact output/count validity,
per-request three-repeat mean timings and complete costs. The primary interval is
10,000 paired request bootstrap resamples, NumPy PCG64 seed20261030, ratio of summed
calls; repeats are not independent requests. Nonpositive primary reduction is
unsupported in this run; positive primary with nonpositive pooled reduction is
mixed. Correctness failure is invalid. A gain over O alone does not establish the
new component. One trajectory/panel does not establish seed robustness, causal
mechanism, deployed queue/lag benefits or a completed publication contribution.
No tuning, fourth arm, event13, merge or deployment is authorized.

## Execution outcome

[The October29 qualification report](results_20261029/REPORT.md) records a concrete
correctness blocker at the first neutral KW window: recorded forward metrics and
anchors match K, but decoded weights and Adam state do not. Execution commit is
`9fd9fe6f7db9796a49e2a6879b5a79e7f01fbd22`. Qualification consumed six response
forwards/backwards and five disposable optimizer applications; no production or
serving work ran. Original O8/K8 and input/source integrity passed postflight.
No cause, efficacy result or automatic retry is claimed.

## Neutral-update qualification repair

The original failed qualification remains in `results_20261029/`. The repaired
qualification runs ordinary K and KW in two fresh interpreter processes, so KW
does not inherit K's process-wide compiled attention history or retained optimizer
storage. Both execute the same four sealed responses; there are no added warmup
passes. Initial state, all backward boundaries, normalized gradients, preview
isolation and two neutral endpoints are checked exactly. Mismatches retain every
parameter path and bounded coordinate statistics. Optimizer-only restores clone
the saved state so temporary steps cannot mutate its CPU Adam counters.

This changes the test harness, not the KW algorithm or tolerances. The new run is
limited to eight response forwards/backwards and sixteen disposable optimizer
applications (fourteen expected), with zero serving generations and persistent
updates. It ends at requalification; production is not resumed automatically.
The plan and exact operational entrypoints are in `neutral_repair_20261029/`.

The repaired native qualification passed on execution commit
`708e974e06f5d9f4b186c32607c16b37cbd4b14f`: 31 exact checks, eight response
forwards/backwards, fourteen disposable optimizer applications, zero generations
or persistent updates. See [the repair report](neutral_repair_20261029/REPORT.md).
The original failed qualification remains preserved; no production event followed.
