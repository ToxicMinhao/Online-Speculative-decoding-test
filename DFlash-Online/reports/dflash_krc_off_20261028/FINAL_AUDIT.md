# Independent audit of KRC2_OFF

**Audit passed with zero issues. The result is mixed: correction helps on the additional panel but hurts slightly on development.** The first audit invocation passed; no audit-parser repair, scientific rerun or result modification was needed.

Material Passport: ARS academic-research-suite, experiment-agent validation. Verification status **ANALYZED**: independent same-team source and retained-data audit, with shared project context. No model forward, training step, serving generation, SSH operation or external review was performed by the auditor. Full tensors retained only on A800 are bound through exact native restoration/publication receipts and remote pre/postflight; they were not independently rerun on the reviewer’s machine.

## Evidence checked

- **1,111 sealed inputs/dependencies** passed remote preflight and postflight. The auditor independently rehashed all **764 available local counterparts**; 347 dependencies remained remote-only. The two explicitly archived older/current objective/trainer source mappings remain intact.
- All **2,418 retrieved files** matched their transfer manifest, covering 115,510,051 bytes.
- All **768 native executions** were reconstructed from their raw traces, totaling **36,543 verification blocks**. Raw outputs, canonical token IDs, termination, counters, accepted/committed progress, per-request metadata and native worker captures agree.
- Execution order matches the balanced two-arm schedule. The first eight development requests supply exactly 48 included correctness executions. The additional panel has no duplicate qualification gate. Every request has three repeat-consistent trajectories for each arm; retained per-request KRC2 counts reproduce on both panels. There are zero invalid, unknown or unattempted executions and no extra worker captures.
- Fresh timings were recalculated as summed per-request means across three repeats. Counts use repeat 0 once per request after checking consistency. Native-generation and cache-maintenance timers fit inside complete-request latency; activation and weight checks remain separate. Each panel has 193 publications and two native weight checks.

## Restoration and training

The exact KRC1 snapshot/export check and live-state check passed at global step 128, with zero weight/Adam/configuration/RNG mismatches. The contract migration changes only the reference-correction entry after restoration; reseeding and optimizer reset are false. Configuration, event history and empty accumulation are preserved.

The first actual A update reproduces both retained KRC2 pre-correction records exactly, including anchors, loss/selection metrics, learning rate, scopes and counters. Its native Adam counter is 129. The check adds no forwards or optimizer steps and makes no later-trajectory equality claim.

The complete OFF event has **128 new Adam steps, 256 ordinary response presentations, 256 response forwards and 256 backwards**, ending at global 256. All 256 naturally sampled presentation/anchor records match the retained event-2 schedule. The capped `train_00187` is preserved at 2,048 response tokens and 2,133 total training tokens. Both evaluation panels are excluded by request ID and tokenized prompt.

Exposure is **66,938 sampled anchors, 977,434 available proposal occurrences and 556,454 selected supervised occurrences**. Selection totals were independently rebuilt using completed accumulation-boundary rows once; individual presentation rows were not added again. The event receipt and durable presentation journal agree. Reference forwards/backwards, protected-prefix capture, direct corrections, replay updates and diagnostic forwards are all zero. Source review confirms the unused reference controller was never installed.

Separate disposable startup used exactly two forwards/backwards and zero Adam steps. The trainer closed before native evaluation. Startup and main-work counters, module/projection invocation counts and recomputation remain separate. The supervisor exited 0.

## Recalculated outcome

Positive reduction favors keeping correction: `1 − KRC2 calls / OFF calls`.

| Panel | OFF calls | KRC2 calls | KRC2 call reduction | KRC2 request-time reduction | KRC2 wins / losses / ties | Gross saved / added |
|---|---:|---:|---:|---:|---:|---:|
| Development | 2,960 | 2,967 | −0.2365% | −0.2531% | 22 / 17 / 25 | 26 / 33 |
| Additional | 3,138 | 3,116 | +0.7011% | +0.5715% | 23 / 14 / 27 | 40 / 18 |

All three development timing repeats favor OFF; all three additional repeats favor KRC2. More request wins do not override the pooled development regression: KRC2’s 33 added calls exceed its 26 saved calls there. No panels are pooled.

Relative to retained KRC1, OFF improves development calls **3,005→2,960 (+1.4975%)** but worsens additional calls **3,125→3,138 (−0.4160%)**. KRC2’s corresponding count changes are +1.2646% and +0.2880%. This is count-only learning context, not a comparison of historical and fresh latency. Retained K2 remains descriptive context at 2,973/3,136 calls.

## Costs and limits

OFF’s complete event took **375.987 s**: 336.097 s recorded training windows, 21.219 s persistence and 15.409 s export, plus other nested event overhead. Peak allocated/reserved memory was **27.347/29.931 GB**. Disposable startup took 44.332 s; live construction/restoration 27.223 s; snapshot validation 6.190 s; native backend loading 21.758 s. The campaign function took 1,892.305 s. These separate costs reconcile with receipts and are not added twice. The appended historical cost context also matches retained source receipts: KRC2 event 696.634 s, including 297.366 s of reference/correction work. These earlier timings are not a fresh matched overhead measurement. Final report links identify the sealed plan, configuration/runtime pins, exact first-A receipt and saved export/recovery identities. The stated training-exposure counts agree with the independent reconstruction.

The report’s mixed interpretation is supported. Event-2 correction has a small endpoint benefit on the additional panel and a small disadvantage on development from the same KRC1 initial state. This does not establish equivalence, a consistent across-panel benefit, seed robustness, a universal “correct only in event 1” policy or training-inclusive acceleration. The no-tuning/no-further-event stopping boundary remains in force.

Machine-readable evidence: `FINAL_AUDIT.json`, `AUDIT_INVOCATION_001.log`, panel-specific reports and raw validity records. Implementation and report-builder reviews are retained separately.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_off_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
