# Native qualification and production launch

Execution commit: `ef792694a52395a6c13ea5dabcf7ae64dd1d04ff`. Plan binding: `0997159a261799b3ba99b31cda0f7f699aa7daaca307c1b503b64cf7deab94c4`. Original model-load clock: `1795738374.2656422`.

**Qualification passed on its first attempt.** Four disposable optimizer applications and eight serving generations completed. No retry, production training data or production checkpoint was used by qualification.

| Check | Result |
| --- | --- |
| New BASE versus frozen PR65 BASE | Exact checked losses, gradients, native updated state, counters and RNG |
| New TAIL versus frozen PR65 TAIL | Exact checked losses, gradients, native updated state, counters and RNG |
| Fresh independent initialization | Released weights, empty Adam, seed/RNG and common contracts match |
| Fixed coefficient formula | Existing FP64 oracle and tolerance pass; shared LOW/preservation quantities unchanged |
| Typed disposable serialization | Exact endpoint readback, export and original-state restoration |
| F / BASE / TAIL / AREPO serving | All eight executions pass native-F token/termination agreement and repeat-consistent counts |

The training qualification phase took 651.760 seconds. Its four case bodies total 649.457 seconds. Eight original-response forwards/backwards and four Adam applications were checked; eight additional startup forwards/backwards are separately recorded. Native recomputation is retained in the per-case physical work receipts. The disposable endpoint exports were removed only after both gates passed; compact parity, restoration and export identities remain.

The 51 new-study CPU tests pass on Fenrir and A800. The broader retained-plus-new regression suite passes 105 tests on Fenrir. Two independent statistical tests, 12 completion-auditor tests and 13 report-builder tests also pass. The latter audit/report tooling is prepared for final evidence; it does not constitute a completed-run audit.

**Production is running.** Fresh BASE has started its first 128-response feedback window. TAIL will start independently after BASE completes event 58. No production event has finished at this snapshot. The runner continues through the fixed final evaluation if correctness and resource checks remain valid; null or harmful results do not stop it.

The current forecast remains about 40 hours raw, or 51 hours including contingency. It is refreshed at each complete event. There is no serving-benefit result yet. Final acceptance still requires both complete trajectories, all four endpoints, the full fixed evaluation and independent audits.

The old PR65 result remains NO_DEMONSTRATED_BENEFIT. This run follows the newly declared practical 0.1% latency-degradation rule and reports statistical support separately.
