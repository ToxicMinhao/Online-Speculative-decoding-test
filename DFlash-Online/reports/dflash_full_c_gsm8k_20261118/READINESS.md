# Full C58 readiness

**Updated status:** executed `2897d483241e0c24b64deab31db09cc10669d9c7` stopped at the exact initial RNG check before any training or serving. See [qualification report](QUALIFICATION_REPORT.md). The original prospective plan below remains preserved.

## Material Passport

Mode: ARS experiment execution and evidence validation. Scope: the user-approved November 18 full-C and classified cleanup handoff. Stage remains research; native K+ is the primary method and C’s incremental utility is unresolved. This file reports readiness, not GPU performance.

## Fixed comparison

One new unchanged C trajectory starts at released DFlash and the original authenticated seed-17 empty-Adam state. It serves the original ordered 7,473 GSM8K requests and completes 58 causal update events, 7,424 Adam steps and 14,848 response presentations. The final 49 requests receive no update. Four inherited capture/publication checks remain inside production accounting. Retained O58 and KPLUS58 are not retrained.

C retains smoothing 0.5 and relative cap 0.25. The objective and allocator source files remain unchanged from PR53. K+ support, first-error redistribution, response normalization, BF16 Adam, clipping, constant 3e-4 LR, two shuffled passes, 512-anchor cap, positional decay and horizon remain pinned. This is own-feedback online training, not a feature-matched replay across methods. There is no seed-29 fork, intermediate efficacy evaluation or hyperparameter search.

Final evaluation freshly measures all three checkpoints on the original 1,024 GSM8K and 1,024 MathQA inputs: 18,432 scored executions and 576 fixed warmups. Six lexical arm orders occur eight times per dataset; every logical activation receives both warmups. MathQA is serving transfer of GSM-trained models. Both panels have been used in prior research.

## Authentication and CPU checks

All 71 protected dependency rows freshly authenticate, including released models, original step-zero witness, O58/KPLUS58, serving runtime overlays and the four retained qualification feature packets. The original O recipe and ordered stream match the intended control; its full-position KL objective is the intended algorithm contrast. The two original panels and four out-of-panel warmups are sealed from existing records; zero new canonical-reference generations are needed. MathQA’s original canonical source is O58’s native target verifier, not target-only decoding.

The current combined suite passes **38 CPU tests**. These cover real small BF16 training continuity, exact own-state resume, native/zero-cap qualification, active C invariants, failure accounting, storage retirement, all 9,504 logical evaluation executions per dataset, pending-only evaluation resume, native trace reconstruction and both 10,000-draw bootstrap seeds. A separate integration review checks the assembled plan and original source/runtime identities. Review authorship overlap is disclosed in its receipt; this is not an independent GPU replication.

## Bounded native qualification

The planned sequence uses one trainer initialization, six disposable Adam steps and 14 response forwards/backwards, including two discarded startup forwards/backwards. It checks bare versus wrapped native K+, zero-cap neutrality, active allocation, exact own-state next-step restoration and CPU export reconstruction. Four authenticated old feature packets provide conditioning only; they never replace the new production feedback.

Serving qualification uses two fixed original panel sentinels per dataset, O58/KPLUS58/disposable active C and two repeats: 24 generations, no generation warmups. All outputs/termination must match canonical records, both controls must reproduce historical calls, and every checkpoint must be repeat-consistent. GPU qualification is pending at this publication. Any substantive failure is preserved; production does not start without all gates.

## Resource admission and cleanup

The measured-size storage plan reserves approximately 71.04 GB new peak, including current-window tensors, initial/latest/predecessor/atomic-state overlap, publication overlap, source/cache growth and a 20 GB compact-evidence sizing allowance. The hard ceiling remains 100 GB, with 400 GB shared reserve after projected writes. Compact evidence is never deleted merely for crossing its sizing allowance.

Verified retirement freed **62,743,060,480 allocated bytes** on A800 from exactly eight completed PR53 recoveries and 512 prepared packets. All four final PR53 exports remain and match their source recovery tensors exactly. Their compact records still reproduce the CPU tables. Per-tensor identities were saved before deletion; these are identities, not backups. Exact historical optimizer continuation and feature replay are deliberately lost. Separately, 1,903,190,016 bytes of redundant newly generated proof copies were removed. A800 then had 517,850,693,632 bytes free. Twelve verified duplicate published worktrees freed 2,599,972,864 allocated bytes on Fenrir; unique private evidence and Git histories remain.

Production uses physical A800 A for serving/teacher capture and B for its persistent trainer, with synchronous publication. Both cards are conservatively charged while allocated. Final latency uses only A after training ends and checks both cards plus host transfer activity before every group. The clock is never reset. Complete-state or complete-evaluation-activation boundaries may resume only after exact evidence checks; ambiguous partial optimizer or generation work is not automatically repeated.

The working estimate is **15–18 hours after launch**, plus CPU preparation/cleanup. Qualification supplies a measured forecast before production: consumed time + 1.3 × remaining active work + 30 minutes must fit the 24-hour cap, and occupied time must fit 40 card-hours. A negative outcome does not stop the fixed campaign. Resource or correctness failures do.

## Required interpretation

Report C/K+, C/O and K+/O separately for each dataset. Use summed request means for latency and one validated trajectory per request for counts. Native acceptance is progress minus calls; stop-clipped acceptance is separate. Use paired whole-request PCG64 bootstrap draws (seeds 20261118 and 20261119) in lexical request order and common draws across all contrasts. Confidence intervals are conditional on these fixed fitted endpoints and reused panels, not seed robustness. C beating O does not attribute the whole recipe gain to C; an interval spanning zero does not prove preservation or equivalence. No automatic follow-up is authorized.
