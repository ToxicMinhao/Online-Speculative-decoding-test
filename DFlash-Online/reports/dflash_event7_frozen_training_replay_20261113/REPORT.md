# Frozen event-seven training-repair replay

## Material Passport

- Status: completed frozen-checkpoint replay; statistical interpretation is conditional on these endpoints.
- Executed source: `10d89bf8ba657cd905d7ad4fb31bf65a0a851656`.
- Population: 128 fitted requests, 815 original slots and 1,629 paired eligible presentations.
- No optimizer steps, backward calls, generations, coefficient changes or new training.

**The direction of the fitted retention difference remains unresolved.** RECENT minus NATIVE acceptance through the tagged position is **+0.613497 percentage points**, with request-cluster 95% interval **[-0.494453, +1.668574] pp**. PR #44's measured serving result remains unchanged.

## Primary equal-slot result

Each slot first averages its one or two eligible presentations. All 815 slots then receive equal weight. The singleton is retained at full slot weight; neither pass nor reachable/firing cases are selected after seeing outcomes.

| Measure | NATIVE7 | RECENT7 |
|---|---:|---:|
| Accepted-through slot equivalents / 815 | 740 | 745 |
| Acceptance through tag | 90.797546% | 91.411043% |
| Earlier-break equivalents | 19 | 14 |
| First-mismatch-at-tag equivalents | 56 | 56 |
| Direct correctness at tag | 92.269939% | 92.392638% |
| Mean canonical-minus-rival margin | 2.638957 | 2.651764 |
| Below original K+6 floor | 10.552147% | 9.815951% |
| Wrong-at-tie | 0.490798% | 0.490798% |

RECENT rescues **13 slot equivalents (1.595092%)** and loses **8 (0.981595%)**. Net = rescues − losses passes exactly. Native headroom is **9.202454 percentage points**, the absolute maximum net increase available on these tags. The observed net does not exceed it.

## Occurrences, disagreements and pass checks

Raw counts on 1,629 paired occurrences: NATIVE accepts through 1480; RECENT 1489; 25 rescues and 16 losses. These integer counts are separate from the primary equal-slot result.

Requests with a retention disagreement: 19/128; with any three-category outcome disagreement: 20/128; with a direct-correctness disagreement: 16/128.

| Pass | Paired occurrences | Native accepts through | Recent accepts through | Rescues | Losses | Net pp |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 814 | 740 | 744 | 12 | 8 | +0.491400 |
| 1 | 815 | 740 | 745 | 13 | 8 | +0.613497 |

The complete 3×3 outcomes are in [outcome_table.csv](outcome_table.csv), with separate integer occurrence and equal-slot-equivalent cells. [paired_occurrences.csv](paired_occurrences.csv) retains every failure, including earlier breaks; [slots.csv](slots.csv) and [per_request.csv](per_request.csv) expose every denominator. Direct correctness and margins are scored even after an earlier failure. A correct later token is not usable prefix progress if an earlier token blocks it. Margin zero does not determine the native argmax winner.

## What changed at the repaired position?

The paired changes below distinguish a changed decision at the tag from a change earlier in the block. Counts are the same equal-slot equivalents used by the primary result; raw occurrence counts and the independent descriptive calculation are in [RESCUE_BREAKDOWN.json](RESCUE_BREAKDOWN.json).

| Paired endpoint change | Rescues | Losses |
|---|---:|---:|
| First mismatch exactly at the tag in the failing endpoint | 7 | 6 |
| Earlier break, while the tag itself was already/still correct | 4 | 1 |
| Earlier break and the tag itself also wrong | 2 | 1 |
| Total | 13 | 8 |

In four rescued slot equivalents, RECENT removes an earlier blocker while the tagged token was already directly correct under NATIVE. Seven rescues correct NATIVE's exact first error at the tag while preceding matches survive; the other two require both the earlier blocker and tag decision to change. Across all original slots, direct tag decisions have nine gains and eight losses, only **one net slot equivalent (+0.122699 pp)**, compared with five net accepted-through gains. This describes final endpoint predictions; it does not identify the update responsible for any change.

The singleton `train_00786 / start 91 / slot 13` is rescued and contributes one full slot weight. Hence 25 raw occurrence rescues and 16 losses become 13 versus 8 slot-equivalents, rather than simply halving both counts. Both eligible passes of the other 814 slots have identical recorded prefix/direct-correctness outcomes; repeated presentations are not independent request samples.

## Uncertainty and interpretation

The primary interval uses 10,000 paired whole-request resamples with PCG64 seed 1707 and linear percentile limits. Each draw divides the summed slot-level differences by its resampled slot count. It measures sensitivity to which fitted requests are represented, not variation across training seeds. Exact rational cohort results and bootstrap ordering are retained in [SUMMARY.json](SUMMARY.json). Only this primary contrast has an inferential interval; pass, margin and outcome summaries are descriptive.

The held-out PR #44 repair comparison remains **8 rescues versus 11 losses** (RECENT accepts through 275/431 versus NATIVE 278/431). This fitted-input diagnostic is not another serving test. It cannot convert local retention into time saved, locate the update that first caused damage, establish that earlier intervention would help, justify a coefficient change, or identify the cause of any fitted-versus-held-out difference. The +0.50% serving target is not a threshold for this percentage-point diagnostic.

## Work and reproduction

The two endpoints run sequentially on one A800 using fresh scoring models. K+6 supplies the retained sampler RNG and validated configuration; each endpoint export is strict-loaded before any draft forward. The full original frozen training geometry is used: zero dropout, gradients disabled, BF16, Flex Attention, 16-token blocks including seed, and eight-block vocabulary projections with original partial chunks. Natural order, anchors and keep masks must match every retained binding. Installed endpoint tensors are checked exactly before and after replay.

| Work | NATIVE7 | RECENT7 | Total |
|---|---:|---:|---:|
| Response forwards | 256 | 256 | 512 |
| Vocabulary projection chunks | 8,170 | 8,170 | 16,340 |
| Backward calls / optimizer steps / generations | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Exact endpoint/input identities and measured device/worker costs are in the run receipts. No prompt text, token IDs, model tensors or private raw paths appear in the scalar tables.

| Measured interval (seconds) | NATIVE7 | RECENT7 |
|---|---:|---:|
| Synchronized scoring forwards | 81.445874 | 80.252544 |
| Model loading and initial tensor checks | 26.619107 | 26.467217 |
| Final tensor integrity check | 5.752227 | 5.706855 |
| Worker total | 127.471636 | 126.067765 |

Synchronized scoring time waits for the selected device around each response and includes host dispatch and scalar extraction. Worker totals additionally include initialization, integrity checks and persistence. These are nested wall-clock intervals, not additive costs or measured GPU-kernel busy time.

Run from the `DFlash-Online` directory at the executed commit, in the retained qualified A800 environment with the private immutable inputs and a fresh output destination. The private run specification binds all paths and identities. Reproduction command:

```bash
python -m experiments.frozen_training_repair_replay.campaign --spec /path/to/private/RUN_SPEC.json
```

## Completion, provenance and audit

The campaign completed on 2026-11-13 at 22:50:11 UTC, with exit 0 and no failed or unattempted scientific work. Its elapsed wall time was **275.951 seconds (4 minutes 36 seconds)**, including **12.423 seconds** of initial preparation. The two synchronized scoring intervals sum to **161.698 seconds**. Peak allocated/reserved GPU memory was **5.531/6.874 GB**. These are diagnostic costs, not serving-speed measurements. Network staging, retrieval and report review are outside campaign wall time. No extra model warmups, backward calls, training or generation occurred.

[WORK_AND_VALIDITY.json](WORK_AND_VALIDITY.json) preserves the before/after equality checks for all 58 draft tensors (1,048,626,432 BF16 coordinates) per endpoint, unchanged embedding/head, original configuration, exact input identities, natural schedule/mask checks, modes and restored RNG. [ENDPOINT_IDENTITIES.csv](ENDPOINT_IDENTITIES.csv) records both fixed exports; [INPUT_IDENTITIES.csv](INPUT_IDENTITIES.csv) lists all 410 authenticated input files under public artifact labels. Actual private paths and token-bearing records remain outside Git. [SOURCE_IDENTITY.json](SOURCE_IDENTITY.json) binds the executed source and sealed population; [EXECUTION_RECEIPT.json](EXECUTION_RECEIPT.json) gives nested costs and postflight resources. All 1,056 private delivery files were hash verified after retrieval; the delivery inventory itself is additional.

Pre-execution network failures caused no scientific execution. The earlier [readiness report](PRE_EXECUTION_REPORT.md) and [pending status](PRE_EXECUTION_STATUS.json) remain historical records and are superseded by this completed report. Both cards were idle after completion; no checkpoint, recovery state or original evidence was deleted.

The independently authored scalar auditor passes **51,521 checks** and exactly reproduces the primary fraction and interval without importing the production reduction: [INDEPENDENT_SCALAR_AUDIT.json](INDEPENDENT_SCALAR_AUDIT.json). A separate raw-to-public audit passes **58,525 checks**, including every sealed occurrence, natural presentation order, anchor/mask binding and work receipt: [INDEPENDENT_RAW_RECEIPT_AUDIT.json](INDEPENDENT_RAW_RECEIPT_AUDIT.json). The [interpretation audit](FINAL_INTERPRETATION_AUDIT.md) checks all 11 ARS statistical-fallacy categories. The separate [publication review](FINAL_PUBLICATION_REVIEW.md) discloses its analysis-code authorship overlap. These are separate same-family reviews, not independent GPU replications.

## Decision boundary

Native headroom is **9.202454 pp**; RECENT rescues **13** slot equivalents and loses **8**, for **+0.613497 pp**. The direction of the fitted retention difference remains unresolved.
The interval remains compatible with changes from -0.494453 to +1.668574 pp. No fitted retention advantage is established. Native leaves 75 slot-equivalent misses (19 earlier breaks and 56 exact-tag failures), so a lack of available headroom does not explain the unresolved result. The source-fitting advantage itself remains unresolved; this comparison therefore cannot establish that a proven fitted benefit failed specifically at transfer.
Stop after this report. Any future training is a separate research decision.
