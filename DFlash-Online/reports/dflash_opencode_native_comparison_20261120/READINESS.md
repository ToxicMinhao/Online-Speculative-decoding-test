**Current authorization supersedes historical holds below:** the 90 GB cap was approved, and the user now permits AR/speculative output differences. Exact within-path and F/O/AREPO/C agreement remain required. The original failed qualification and earlier readiness record below are preserved as historical evidence. See [the current recovery contract](OUTPUT_POLICY_RECOVERY.md).

# Native online CODE comparison: readiness

**Diagnostic update (execution `7ba747702d5b86628cfed95ac064db4d54469586`):** Two read-only native replays reproduce the mismatch and identify an exact AR logit tie versus a0.125 DFlash margin at the first divergent token. No token-commit bug was found; the underlying operation causing the numerical perturbation remains unisolated. The strict equality gate and main campaign remain blocked. See [the diagnostic report](MISMATCH_DIAGNOSTIC.md).

**Current native status: blocked.** The AR startup repair passed, but same-card AR/F outputs differ after279matching tokens. Production remains unstarted. See [qualification report](QUALIFICATION_REPORT.md); earlier readiness statuses below are historical.

**Retry authorized:** the user approved the repaired bounded qualification and unchanged campaign if all gates pass. The new explicit recovery path preserves the original clock, work, inputs, compiler baseline and failure archive. 81 focused CPU tests pass; native retry is pending.

**Native status update:** CPU preflight passed; AR startup failed on both GPUs before scientific work. The narrow piecewise-graph configuration repair passes44CPUtests but has not been retried. See [failure and repair report](STARTUP_FAILURE.md). The campaign remains stopped.

**November20 authorization update:** user approved90GBadditional storage;400GBreserve and all other limits/settings unchanged. Prior80GBblocked assessment below remains historical. Actual admission and GPU qualification are pending; no model work yet.

Implementation source: `c0a0794e8a7a42a3e2e70833a73992a6201ea6d6`. Native execution: **not started**.

## Material Passport

Mode: ARS experiment run with CPU implementation validation. Status: **resource-blocked before GPU qualification**. Scientific execution commit: none yet. Source base: PR55 `18d60135e6a482bf2f0f92454c572ccadc2adf1e`. This document reports preparation, not an experiment result or a native qualification pass.

## Scientific contract

Three independent released-start trajectories run in fixed order O_CODE58 → AREPO_CODE58 → C_CODE58. O uses ordinary OSD forward KL; AREPO reuses PR31's literal released AdaFlash mixture/clipping implementation; C reuses PR54's unchanged K+ plus C rule. Existing objective implementations, BF16 arithmetic, native AdamW, constant3e-4 LR, two passes/event and natural anchor sampling remain unchanged. The dedicated adapter changes the shared training context from2,272 to4,096 and uses exactly7,424 CODE requests without the old49-request tail.

Each arm collects its own128-response feedback window from its current published checkpoint, prepares only that completed window, performs128 native optimizer steps, validates and publishes, then collects the next window.58events yield7,424optimizer steps and14,848response presentations per arm. There is no cross-arm feature reuse, feedback generation from held-out requests, future collection, learned checkpoint initialization or eventwise reseeding. Each starts with released weights, fresh optimizer and seed17. Qualification's temporary states never initialize production.

Serving uses the pinned Qwen3-8B target and five-layer DFlash drafter, BF16/FA3, greedy no-thinking prompts, block16,2,048tokens/EOS, logical context4,096, physical engine capacity4,112, concurrency1, cold caches, disabled feedback capture for evaluation and enabled native CUDA graphs. Genuine AR disables speculation through the supported engine configuration; it is not block-size-one DFlash. AR has its own native token/termination and cache validation.

## Sealed inputs

Pinned OpenCodeInstruct parquet revision: `9819cc0464bf7312883763f93751749007b5bbcf`. Exactly50,000 `input` rows were read; reference solutions were not decoded. Single-split branch uses raw[0,30000) for training and[30000,50000) for evaluation.204 source occurrences were excluded for duplication;29,892/19,904 eligible inputs remain. SHA256 role ranking with seed20261120 selects7,424training,1,024evaluation,two warmups and16qualification requests. All four sets are disjoint and were sealed before model outputs. The historical audit consumed150,766files; its limitations and two preserved parse gaps assessed as non-input gaps remain disclosed. This is the author's dataset/extraction with a custom panel, not AdaFlash's exact test slice.

Private complete IDs, raw prompts, rendered tokens, exclusion records and order are sealed onA800 under `/data/sources/minhao/runs/dflash_opencode_native_comparison_20261120/sealed/`. [Seal](readiness/DATA_SEAL.json), [counts and all58 input-length bounds](readiness/DATA_SUMMARY.json), [source/history review](readiness/DATA_SOURCE_REVIEW.md). The selected training prompts are79–1,349tokens; the largest128-request full-output-cap window contains293,039sequence rows. Longest qualification prompt1,102 does not establish full-context memory safety; the synthetic4,096-position fixture is still required.

## Exact remaining qualification

| Work | Planned | Ceiling |
|---|---:|---:|
| Native generations |64|64|
| Disposable Adam steps |17|48|
| Response forwards / backwards |40 /40|128 /128|
| Teacher preparations |16|32|
| Trainer constructors |3|8|

The initial60generations cover16reserved AR/F pairs, both warmups, cross-card parity, repeated cold-cache outputs and caps1/17. Four later generations check active-C export/reload. Five O and five AREPO one-step cases test original/adapted equivalence, uninterrupted versus reloaded continuation and full-context execution. Seven C cases add bare/native/cap-zero and active-C controls. Six disposable startup forwards/backwards and zero startup Adam steps are included in the40/40 count. Graph-initialization forwards are recorded separately. All temporary data are discarded only after complete checks and preserved identities.

After small qualification, the entire1,024-request panel receives genuine AR and F execution and must agree exactly before long training. A mismatch blocks the study; no trimming, replacement, recanonicalization or tolerance change is authorized.

## Fixed production and measurement

Production totals are39,932complete generations,174events,22,272Adam steps,44,544response forwards/backwards each and22,272teacher preparations. The ceiling including qualification is39,996generations. The five-arm final evaluation has15,360scores,240warmups and120logical activations. Ten rotate/reverse method orders give each method pair12precedence occurrences each way; this is aggregate balance, not per-request position balance.

All final outputs/termination must match fresh target-only AR canonicals; repeat counts and immutable endpoint weights are checked. Complete worker-request latency and native engine time remain distinct; no trace-overhead subtraction. Native accepted tokens, stop-clipped matching tokens and committed progress have separate denominators, with legitimate zero-call requests retained. AR draft statistics are N/A. The primary contrast is C/AREPO complete-request time reduction.10,000paired whole-request bootstrap draws use PCG64 seed20261120, lexicographic IDs and shared draws across contrasts. Secondary intervals are descriptive and unadjusted.

## Storage admission and current blocker

The complete conservative forecast is **86.57GB**, above the unchanged **80GB** new-study cap. The exact sealed replacement-row calculation gives86.53GB; the86.57GB review intentionally reserves extra overlap. It already uses verified lossless completed-event packaging and keeps two complete recoveries but only the current necessary serving export. Final retention remains complete events57/58 plus event58export for every arm. No historical checkpoint deletion is proposed.

The most recent remote check found493.55GB free and both80GB A800s idle. The400GB reserve remains mandatory. A90GB cap was proposed to the user but is **not approved or applied**. The48-hour elapsed and90-card-hour ceilings also remain unchanged. The working compact forecast is not a guaranteed worst-case bound: unusually long low-acceptance traces can exceed it, requiring a safe resource stop rather than a smaller experiment.

The user-approved Fenrir cleanup retired only four clean, published duplicate source checkouts, freeing1,178,304,512allocated bytes. Reports, raw results, checkpoints and shared Git history remain. Fenrir has less than1GB free; large private archives stay onA800 until verified delivery is possible. [Cleanup receipt](readiness/FENRIR_DUPLICATE_RETIREMENT.json).

## Duration and remaining evidence

Historical three-arm training alone suggests27–30hours. A provisional full-work estimate is35–45hours, but CODE outputs and genuine AR costs are unmeasured. This range is not an admission: the fixed rule requires consumed time +1.25×remaining forecast +1hour to fit48hours. At the upper end it would fail. The native qualification and full-panel precheck must calibrate the complete forecast before any58-event trajectory. Recheck after each arm and at safe event boundaries; no automatic time or work extension.

The combined Torch CPU suite passed18tests in245.178seconds, including isolated three-arm continuous training/resume and the bounded C qualification fixture with a real synthetic4,096-position pass. The separate pure-CPU suite passed136tests in2.91seconds:154distinct CPU tests in total. [Machine-readable receipt](readiness/CPU_TEST_RESULTS.json) and [integrated log](readiness/CPU_INTEGRATED.log). Native correctness, GPU memory fit, AR/F agreement and actual performance remain unqualified. No scientific generation, teacher preparation or Adam update has run. A valid negative result would complete the fixed study if all gates and resources permit it; no result-driven tuning or new direction follows automatically.

## Review evidence

- [Independent code/native-API review](readiness/reviews/INDEPENDENT_CODE_REVIEW.md)
- [Storage calculation and source-backed sensitivity](readiness/reviews/STORAGE_ADMISSION_REVIEW.md)
- [Training adapter and exact qualification review](readiness/TRAINING_ADAPTER_REVIEW.md)
- [Launcher/disposal review](readiness/LAUNCHER_WIRING_REVIEW.md)
- [Combined Torch CPU tests](readiness/CPU_TRAINING_QUALIFICATION.log)

Review agents shared the same implementation context and some author roles. These are source/CPU checks, not independent GPU replication. Historical fixture sizes and CPU serialization stress are used only for resource planning, not as CODE learning or serving results.
