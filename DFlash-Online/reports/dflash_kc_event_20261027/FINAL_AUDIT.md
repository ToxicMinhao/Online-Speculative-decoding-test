# Independent final audit: one KC event

**Audit passed with no discrepancies or missing evidence. KC did not improve the primary native drafting outcome: verification calls increase from 3,027 to 3,035 (+0.2643%).** Sixteen requests improve, 23 regress and 25 tie. The correction improves some measured training prefixes, but those local changes did not produce a benefit on this development panel.

## Material Passport

- Workflow: ARS experiment-agent evidence validation.
- Inputs: sealed source/input identities, training/window receipts, retained first-step reference and every fresh native record.
- Checks: [auditor](audit.py), [machine-readable audit](FINAL_AUDIT.json), [correction steps](analysis/CORRECTION_STEPS.csv), [probe blocks](analysis/PROBE_BLOCKS.csv).
- Scope: local compact-record analysis only; no model execution, source revision or additional experiment.

All **1,358 retrieved files** verify against the retrieval manifest. Preflight and postflight both verify the **593-file seal**. The audit checks the original feedback/presentation/anchor identities, all 128 window records, exact state-preservation receipts, canonical training token identities, EOS boundaries, triangular prefix weights, global normalization and prescribed FP32 correction coefficient. It also reconstructs all **18,186 native verification blocks** across the 384 executions, including outputs, committed progress, accepted tokens, publication identities and repeat consistency. No invalid, incomplete or unattempted evaluation remains.

| Fresh native result | Retained K1 | KC |
|---|---:|---:|
| Verification calls, one trajectory/request | 3,027 | 3,035 |
| Accepted draft tokens | 16,513 | 16,506 |
| Identical output tokens | 19,547 | 19,547 |
| Sum of per-request mean latency | 52.101776 s | 52.076363 s |

The latency difference is **0.0488%**, or 0.0254 seconds across the panel. Its sign changes across repeat-level totals: KC is about 0.0008 s and 0.1794 s faster in repeats 0/1, and 0.1040 s slower in repeat 2. This tiny aggregate timing difference does not support claiming a serving-performance improvement. Counts are stable across every repeat; the small worsening in verification calls is the prespecified primary result. The panel was already used in development and is not a fresh confirmation set.

The event completes exactly **128 native Adam steps and 256 presentations** with **65,096 sampled anchors**, 949,560 available proposal occurrences and 534,329 selected K-loss occurrences. Prefixes refresh from KC's current state each window. All exact optimizer/gradient/RNG/runtime checks pass. The first window reproduces the complete retained diagnostic evidence exactly before window 2 is allowed to proceed.

There are **84 applied corrections**, all 84 nonzero after BF16 rounding, and **44 zero-violation no-ops**. This is not the earlier inactive-guard outcome. Correction protection covers 483,389 prefix occurrences. The execution uses **518 core response forwards and 371 backwards** (256 K plus 115 violation backwards), within the declared limits. The separate startup performs two forward/backwards and zero Adam steps. There is no new feedback, baseline retraining or extra qualification step.

| Probe step | Lost old-prefix tokens, K → KC | Net prefix gain, K → KC | K repairs retained / lost / newly obtained |
|---:|---:|---:|---:|
| 1 | 153 → 113 | +1,740 → +1,807 | 527 / 0 / 6 |
| 64 | 2 → 1 | +179 → +197 | 63 / 2 / 4 |
| 128 | 10 → 10 | +99 → +100 | 51 / 0 / 1 |

The block-level comparisons retain regressions: step 1 has 17 improvements, 8 regressions and 749 ties; step 64 has 5/2/741; step 128 has 1/0/395. Step 64 loses two useful repairs while gaining four. V decreases at steps 1 and 64 but is unchanged at step 128. Thus neither local repair preservation nor nonlinear loss reduction is guaranteed. These are overlapping training-block observations. The later ordinary K candidates begin at the same current KC pre-step state; they are not checkpoints from an independently trained K trajectory.

The complete KC event costs **534.715 s**, including **501.377 s** in training windows. Nested intrinsic post-K gradient/correction work costs **151.595 s**; corrected-state probes cost **3.719 s**; integrity instrumentation costs **4.209 s**. Checkpoint persistence/export costs **17.151/13.508 s**. Peak allocated/reserved GPU memory is **40.898/43.734 GB**. These components must not be added again to the event total. All 193 benchmark checkpoint publications cost 268.647 s and are excluded from request latency. The driver takes 1,205.141 s; the enclosing supervisor runs 19:46:16–20:07:30 UTC (21m14s), exit 0. Historical K training time is not a fresh matched overhead control.

The supported conclusion is narrow: the tested correction can change and sometimes improve local training prefixes while preserving native optimizer state, but **this one-event screen does not show improved later-request drafting over K1**. It does not identify the underlying reason or establish that every preservation mechanism fails. No tuning, new arm, additional event or automatic follow-up is justified by the execution authorization; the run has stopped at the agreed report.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_kc_event_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
