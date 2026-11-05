# Independent KRC event-2 results audit

**Passed: no execution-validity, accounting or report discrepancy found.** The auditor reconstructed the retained measurements and checked the source/receipts; it did not run a model, generate a response or perform an optimizer step. ARS experiment-agent validation status: **ANALYZED**. This is independent review within the same project team, not a blinded external replication.

## Scope and validity

The complete KRC1 restart payload was bound to its corrected export. The exact native live-resume receipt covers weights, Adam moments/counters, Python/NumPy/Torch/CUDA RNG, configuration, event history and empty accumulation. Global step 128 and 256 presentations were restored before the first event-2 window. The only recorded contract change disables the old optional probes. All 256 ordinary presentation/anchor draws match the retained O2 schedule; none was overwritten or reseeded. All 128 cyclic R bindings use their own same-pass anchors and remain disjoint from the current A pair. Both evaluation panels are excluded from the feedback by request identity and prompt tokens. The capped train_00187 response remains2048 tokens/2133 total tokens.

Exactly 128 new Adam steps and 256 ordinary K presentations completed, reaching global 256. Core work is 768 response forwards and 482 backwards (256 K plus 226 reference backwards). All 128 reference-window integrity records pass. There are 124 applied corrections, all nonzero after BF16 casting, and four zero-violation no-ops. No extra Adam step, new feedback, optional A capture or post-correction replay occurred. Separate startup has two forward/backwards and zero Adam steps; the first actual window's checks add no model work.

The raw training records independently recover66,938 anchors and 977,434 available proposal occurrences. Ordinary K selected 556,686 occurrences. R uses 256 presentations of 128 distinct responses, with the same 66,938 anchors/977,434 available occurrences and 505,338 protected occurrences. Pre/post reference phases are additional forwards, not new examples. Per-window prefix membership, triangular weights, global denominator, hinge reduction and prescribed FP32 coefficient reconcile with retained records. There is no post-correction prefix-benefit measurement in this event.

All 768 native executions are valid, with exact retained tokens/termination and repeat-consistent counts. The included 48-execution development gate adds no work. Retained K2 counts are enforced only on development; additional-panel K1 counts are descriptive history, never a K2 gate. The audit reconstructs 36,576 native verification blocks and all raw worker captures. Every repeat starts 32 requests per arm; each panel contains 384 executions, with zero invalid, unknown or unattempted executions. Counts use one trajectory per request and times use the three-repeat request mean.

## Independently reconstructed outcome

| Panel | K2 → KRC2 calls | KRC2/K2 reduction | Improved / regressed / tied | Gross saved / added | Fresh request-time reduction |
|---|---:|---:|---:|---:|---:|
| Development | 2973 →2967 |0.2018%|25 /19 /20|43 /37|0.1521%|
| Additional |3136 →3116|0.6378%|31 /20 /13|56 /36|0.6228%|

The relative call advantage shrinks from 0.7268% to 0.2018% on development and from 0.7937% to 0.6378% on additional requests. KRC itself improves from 3005→2967 calls (1.2646%) and 3125→3116 (0.2880%). K improves more: 3027→2973 (1.7839%) and 3150→3136 (0.4444%). These are count-only learning comparisons; historical-versus-fresh timings are not used as learning progress. Both panels remain separate. Fresh timing directions favor KRC2 in all three repeats per panel, but the development difference is particularly small.

Thus the observed small advantage persists through the second shared-feedback event but does not grow. It is not evidence of seed robustness, an independent online deployment, a general preservation guarantee or adaptation-inclusive acceleration. The fixed reused panels and two tested feedback windows limit the conclusion.

## Costs, provenance and report review

The event takes 696.634s, including 656.807s of training windows, 20.454s checkpoint persistence and 15.728s export. Intrinsic reference/correction work is 297.366s and integrity checks 6.870s, nested within event/window timing. Peak GPU allocated/reserved memory is 36.635/37.845GB. Startup 42.603s, exact snapshot validation 6.365s, resume loading/checks 27.228s and native backend loading 22.008s remain separate. Each panel has 193 checkpoint publications, taking 382.007s/380.329s; these are excluded from request latency. Complete request, native generation and cache-maintenance timers reconcile and are not added twice. No historical K training cost is treated as a fresh matched overhead control.

All 1068 sealed artifacts pass remote pre/postflight. The auditor rehashed 722 available local counterparts, including unchanged author training code; 346 nonlocal tensors/artifacts rely on the retained remote checks. Every one of 2547 retrieved files (463,904,140 bytes) was independently rehashed. The exact two archived historical O/K source copies and live KRC1 source mappings also pass. The peak admission retains the 8 GiB disk reserve and 72 GiB host-memory requirement.

One inherited descriptive label in the compiler-flag receipt still mentions the earlier K/G qualification constructions. This is stale helper text, not additional work: actual receipts show only the declared two zero-Adam startup evaluations and resumed KRC2 event. Raw evidence is preserved unchanged. The report's headline, tables, reference exposure, timer nesting and interpretation match the independent reconstruction. No auditor parser correction was needed; the first audit passed. Four focused audit fixtures also pass. A later repeat initially used the workspace root as its working directory and could not import the local test module; that invocation log is preserved as `INITIAL_AUDIT_TEST_INVOCATION.log`. Running from the experiment directory passes all four tests in `AUDIT_TESTS.log`; no test or source change was needed.

Evidence: `FINAL_AUDIT.json`, `AUDIT_EXECUTION_001.log`, `audit_results.py`, `test_audit_results.py`, raw resume/window/evaluation records and sealed inputs. No additional experiment is authorized or performed by this audit.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_continuation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
