# Ordinary-gradient mismatch: startup comparison repaired

**The retained SCALAR failure is reproduced by comparing cold ordinary K+ backwards with warmed ordinary K+ backwards. With matched startup, ordinary K+ and SCALAR's pre-Adam gradients agree exactly, including after the auxiliary/resource workload. The qualification startup procedure has been repaired; no loss, optimizer, precision, tolerance or guard was changed. Full qualification and the pilot have not resumed.**

Native diagnostic execution: [`6f44f57425cf2bfefdc0085dd8112832966d2264`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/6f44f57425cf2bfefdc0085dd8112832966d2264). Startup repair: [`c9da97310b701e31690baa8b687871f774850ccf`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/c9da97310b701e31690baa8b687871f774850ccf). Both are in [PR #33](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/33). See the [comparison table](ORDINARY_GRADIENT_DEBUG_COMPARISONS.csv) and [summary](ORDINARY_GRADIENT_DEBUG_SUMMARY.json). The [earlier qualification report](REVISED_QUALIFICATION_REPORT.md) remains preserved as the outcome of that attempt.

## What was wrong

The original qualification discarded two startup backwards and restored the saved scientific state before measuring its ordinary reference. The resumed qualification incorrectly combined those two roles: its first, cold backwards became the reference, while SCALAR ran after compilation and other work. Restoring weights, optimizer, RNG and model buffers does not reset process-global compiler state.

The bounded diagnostic used the same retained KPLUS4 origin, two saved response/features and their original sampled anchors. Every branch restored the exact origin. We intercepted each ordinary update before clipping or Adam, so this comparison contains **zero optimizer updates**. SCALAR used its actual capture hook; its auxiliary update had not yet begun.

| Comparison | Exactly differing coordinates | Global relative L2 difference | Result |
|---|---:|---:|---|
| Cold ordinary vs warmed ordinary | 155,364,420 | 0.2888553% | Reproduces the old failure |
| Warmed ordinary vs repeated warmed ordinary | 0 | 0 | Exact |
| Warmed SCALAR vs warmed ordinary | 0 | 0 | Exact |
| Ordinary after auxiliary work vs before it | 0 | 0 | Exact |
| SCALAR after auxiliary work vs ordinary after it | 0 | 0 | Exact |

Each comparison covers **58 parameter tensors and 1,048,626,432 gradient coordinates**. There are no nonfinite coordinates. Cold versus warm differs in 49 tensors, with maximum absolute error `0.00048828125` and discrepancy L2 `0.001644496259488957`. Its warmed reference norm is `0.569314870183913`.

The reproduction is stronger than matching aggregate totals: all 49 previously differing tensor names, coordinate counts and maximum errors match the retained failure. Its nine exact tensors remain exact. All retained bounded coordinate samples match after reversing the comparison orientation; discrepancy norms agree up to diagnostic reduction rounding. Both saved response observation rows, including losses and anchors, match in every branch.

The actual runtime receipts show a FlexAttention compiled-cache transition during the cold pair: the first response has entry `1/0`; the second adds `1/1`. Warmed repetitions retain those entries. The auxiliary workload adds another entry but leaves subsequent ordinary gradients exactly unchanged. This supports a startup-dependent numerical explanation. **It does not isolate a particular compiler transformation or kernel, demonstrate a defective kernel, or establish universal numerical repeatability.** The confirmed implementation error was comparing different startup conditions.

## Repair and verification

The resumed GSM8K qualification now explicitly discards its two startup forwards/backwards, restores the exact saved state, and only then captures ordinary-reference gradients. Those startup operations are separately charged. A stale sealed plan that omits their declaration is rejected before model construction or ledger work. The old plan, ceilings and failed work have not been silently overwritten.

The first debugging attempt also exposed a cleanup-order bug. A deliberate pre-Adam stop leaves TrainerCore metric totals pending. `restore_origin` previously cleared those totals after invoking the native clean-boundary validator, so restoration raised `unclean_live_boundary`. The totals are now cleared before validation. Diagnostic receipts are also preserved if cleanup fails. This fixes disposable-state restoration; it does not change the training objective or optimizer.

- [Discarded startup and reference recovery](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/c9da97310b701e31690baa8b687871f774850ccf/DFlash-Online/experiments/functional_step_geometry/qualify.py#L161).
- [Resumed qualification contract and call](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/c9da97310b701e31690baa8b687871f774850ccf/DFlash-Online/experiments/functional_step_geometry/resume_qualification.py#L174).
- [Interrupted-boundary cleanup](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/c9da97310b701e31690baa8b687871f774850ccf/DFlash-Online/experiments/functional_step_geometry/qualify.py#L91).

All **39 execution tests passed on A800**, and all **188 local FSG/functional-geometry CPU tests passed** after the final startup repair. New regressions cover real interrupted TrainerCore state, receipt preservation, separate startup work, exact restoration before reference capture, startup failure, and rejecting a stale plan before model work. The native diagnostic validates the matched-startup control; it is not a fresh execution of the complete repaired qualification entry point.

All six completed diagnostic windows have exact final scientific-state restoration receipts. Both attempts' **6,921 sealed dependencies** match before and after execution. The same-layout oracle and maximum-size resource intervention also passed again within the declared conditional diagnostic budget. Historical cross-layout failures remain failed layout-sensitivity results; this repair does not relabel them.

## Bounded work and costs

| Work | Failed debugging attempt | Completed debugging attempt | Total |
|---|---:|---:|---:|
| Response forwards | 2 | 26 | 28 |
| Response backwards | 2 | 18 | 20 |
| Adam steps | 0 | 0 | 0 |
| Feedback/teacher preparation/serving generations | 0 | 0 | 0 |

The completed attempt contains six ordinary two-response windows (12 forwards/12 backwards), plus the retained auxiliary/resource intervention (14 forwards/6 backwards). Each ordinary window records 94 objective chunks and 188 vocabulary-head calls. The completed attempt totals 26 draft forwards, 130 layer forwards, 2,556 vocabulary-head calls, 564 ordinary objective chunks and 388 proxy chunks; the underlying helper receipts preserve their roles. The failed attempt did not write a complete final physical-work receipt, so no cumulative physical count is invented for it. No new trained checkpoint was produced.

The failed attempt, execution `a281bfe7b33fb7c7711dbcc215160a760b1aa04d`, remains failed and charged. It had exact initial-state, anchor/loss and unchanged-weight/optimizer receipts, but no successful final-restoration receipt or completed gradient comparison. It is not treated as a usable resumed state. A separate archive-admission check rejected an incomplete transfer before execution; it consumed no model work and passed only after transfer completion.

The completed supervisor ran **561 seconds**; its runner used **545.667 seconds**. The failed supervisor/runner used **119 / 104.156 seconds**. Combined supervisor time was **680 seconds (11 min 20 s)**. Peak GPU allocated/reserved memory was **47.021 / 48.771 GB**. These times include extensive state restoration and CPU comparison of a billion coordinates; they are diagnostic costs, not FSG adaptation-overhead estimates.

All **99 compact evidence files / 20,791,340 bytes** across both attempts were copied and independently checked against remote hashes. Only two verified redundant source-transfer archives, totaling 42,914,819 bytes per host, were removed; published/extracted source, raw failures, origin states and replay dependencies remain. A800 had **55.525 GB free**, and both GPUs were idle at postflight. Private response records and runtime artifacts remain outside Git.

## What this resolves—and what remains

**This resolves the tested ordinary-gradient mismatch under matched startup. It provides no evidence that FSG improves drafting.** The implemented remedy restores the intended qualification procedure without weakening equality or changing the algorithm.

The remaining SCALAR optimizer/own-state checks, MathQA checks and serving checks still need to finish. Before execution, a fresh continuation plan must include the latest failed qualification ledger and the separately counted discarded startup; reusing the old plan would omit work. This debugging task did not reseal or launch that continuation. It also did not launch the pilot or change the strict BF16-realized norm/KL guards.

## Material Passport

- Workflow: ARS experiment-agent run/validate; scoped source, numerical and work reviews by the same project team, with implementation authorship overlap disclosed.
- Verification: native matched-startup gradient control **VERIFIED on the retained fixture**; overall qualification **INCOMPLETE**.
- Evidence: exact-state and anchor receipts, per-parameter/bounded-coordinate comparisons, actual imported compiler metadata, work ledgers, native tests, final CPU regression log, dependency and transfer inventories.
- Interpretation: all 11 ARS fallacy categories checked. A fixed numerical fixture is not a serving sample, failed work is retained, no significance or generalization claim is made, and low-level causality is narrower than a kernel-defect claim.
- Scope: unchanged BF16, K+/FSG/SCALAR mathematics, optimizer and tolerances; zero Adam updates, new feedback or serving; no mechanism-performance conclusion.
