# Point3 final assessment comparison

**No demonstrated improvement from MAX over SUM, or from either sensitivity arm over C. Keep native C + K+PAIR under the prespecified freeze rule.** Both new arms finished the one-event continuation and the complete three-arm evaluation. This report closes the assessment choice; it does not authorize another mechanism search or experiment.

Actual GPU execution commit: `6f7f3a5d1769a630e4ec852205eb06ca55047341` ([source](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/tree/6f7f3a5d1769a630e4ec852205eb06ca55047341/DFlash-Online)). Report/audit publication is a later commit, identified by Git history and the publication receipt. The execution checkout was clean and all 5,346 staged source entries authenticated. No new GPU work, training or generation was performed for the final audit. ARS experiment-agent validation was used; separate same-family agents independently audited serving arithmetic, state/work receipts, and mechanism readouts. This is an internal independent implementation audit, not blinded external peer review.

## Serving outcome

The panel is the original first 128 OpenCodeInstruct development requests. C is the retained **native C + K+PAIR event-2 checkpoint**, not ordinary OSD or frozen DFlash. SUM and MAX independently continue the same complete PAIR1 state on the same 128 feedback responses, presentation order and anchors. Both commit 128 updates / 256 presentations, ending at cumulative step 256 / presentation 512 / event 2. C is reused without retraining.

| Endpoint | Verification calls | Mean-request latency summed over 128 requests (s) | Native-generation time (s) | Cache maintenance (s) | Native accepted tokens/call | Committed progress/call |
|---|---:|---:|---:|---:|---:|---:|
| C | 7,639 | 141.610171 | 132.961496 | 4.660006 | 5.558188 | 6.558188 |
| SUM | 7,634 | 142.663959 | 133.376249 | 5.181386 | 5.562484 | 6.562484 |
| MAX | 7,636 | 142.108876 | 133.549724 | 4.370157 | 5.560765 | 6.560765 |

Positive reduction favors the candidate. Counts use one validated trajectory per request; each latency first averages that request's three repeats. The primary component comparison is MAX versus SUM. The other two are required whole-method comparisons.

| Candidate / reference | Complete-request time reduction, % [95% CI] | Call reduction, % [95% CI] | Request call wins / losses / ties | Gross calls saved / added |
|---|---:|---:|---:|---:|
| MAX / SUM | +0.3891 [−0.5207, +1.2986] | −0.0262 [−0.4259, +0.3605] | 42 / 42 / 44 | 62 / 64 |
| SUM / C | −0.7441 [−2.3723, +0.7756] | +0.0655 [−0.4356, +0.5547] | 51 / 49 / 28 | 86 / 81 |
| MAX / C | −0.3522 [−1.7655, +0.9857] | +0.0393 [−0.4369, +0.4969] | 49 / 44 / 35 | 77 / 74 |

MAX uses two **more** calls than SUM. Its 0.3891% favorable complete-request timing estimate is unresolved and accompanies a 0.1301% unfavorable native-generation estimate. SUM and MAX save only five and three calls against C, respectively, while both have slower complete-request point estimates. The data establish neither an advantage nor equivalence. No claim that either variant is reliably slower follows from these intervals either.

All arms generate the same 50,226 output tokens: mean 392.390625, range 144–1,314; all 128 requests terminate at EOS. Coupled native accepted counts are C/SUM/MAX = 42,459 / 42,464 / 42,462. Separately recorded stop-clipped acceptance counts are 42,580 / 42,586 / 42,584; these counters have different end-of-request semantics and must not be substituted for one another. Committed progress totals 50,098, excluding one initial prefill token per request. Complete-request time includes other request overhead, so native-generation plus cache-maintenance time need not equal it; no detailed kernel breakdown is inferred.

Intervals use the prespecified 10,000 paired whole-request PCG64 bootstrap draws, seed 20261125, shared resampled indices across all arms, linear 2.5/97.5 percentiles. Timing repeats are not extra independent requests. Secondary intervals are descriptive and unadjusted. These are fixed endpoints on a reused development panel, not fresh confirmation or training-seed replication. [Independent serving audit](INDEPENDENT_SERVING_AUDIT.json), [contrasts](CONTRASTS.csv), [per request](PER_REQUEST.csv), [all repeats](ALL_REPEATS.csv), [warmups](WARMUPS.csv), [repeat totals](REPEAT_TOTALS.csv), [statistical interpretation](INDEPENDENT_SERVING_REVIEW.md).

## What changed, and what the comparison tests

Only the outer assessment differs between the two new arms. For each current-origin block, K ends after the first error and its consecutive matching followers. Canonical EOS is valid; seeds, padding, gaps and positions after EOS are excluded. With native canonical-minus-strongest-wrong margin m and phi(m) = softplus(−m)/log(2):

```text
R_SUM = sum_{j=1..K} (K−j+1) phi(m_j)
R_MAX = sum_{k=1..K} max_{j=1..k} phi(m_j)
```

MAX selects the earliest lowest margin in each prefix with detached branch indices, then gathers differentiable margins. There is no second triangular multiplier. Assessment averages nonempty blocks within each response, then the two responses; an empty response contributes zero. It has no training-loss denominator or K normalization.

For greedy agreement with fixed origin K and current matching-prefix length A, `D = K − min(A,K) <= R_MAX <= R_SUM`. This is a fixed-K greedy-block shortfall bound, not a proof of better gradients, stochastic verification behavior, preserved predictions or serving speedup. A numerically smaller MAX than SUM at the same endpoint is structural and cannot demonstrate benefit.

Both arms obtain actual native C preclip gradients without an Adam step, evaluate the existing FP32 optimizer shadow and native-BF16 endpoint sensitivity, then use the unchanged detached beta-centered allocator (cap 0.25). Ordinary inner K+PAIR teaching, C settings, LR 3e-4, weight decay 0.01, clipping 1, microbatch 1 / accumulation 2, two passes, up to 512 anchors, gamma 7, eight-block head chunks, FlexAttention and constant-LR horizon 7,424 are unchanged. MAX is not a new direct teaching loss.

The allocation changed and survived into distinct trained parameters; this was not an inactive mechanism. At windows 1/64/128, MAX's half-L1 coefficient movement from current C is 175.40 / 158.56 / 174.32. Its cumulative bottleneck multiplicity assigned directly to correct followers is only 15/6,665, 44/5,848 and 15/8,716. Followers can still affect how many prefix terms exist; sparse direct outer-gradient support is not proof of the reason serving failed. SUM records the same diagnostic definition, but its actual outer gradient remains triangular and dense.

Matched parent-origin witnesses give the following **TRAIN** observations (each set has eight responses / 64 blocks). “Useful repair” requires repairing the old first error while preceding matches survive. It is distinct from a raw correct decision at that old position.

| Witness set / endpoint | Matching-prefix occurrences | Fixed-K shortfall | Useful repairs | Damaged old-prefix positions |
|---|---:|---:|---:|---:|
| New buffer / parent | 488 | 104 | 0 | 0 |
| New buffer / C | 656 | 35 | 33 | 3 |
| New buffer / SUM | 670 | 31 | 35 | 2 |
| New buffer / MAX | 674 | 24 | 34 | 4 |
| Parent training / parent | 676 | 64 | 0 | 0 |
| Parent training / C | 675 | 75 | 9 | 14 |
| Parent training / SUM | 681 | 69 | 9 | 12 |
| Parent training / MAX | 689 | 62 | 9 | 13 |

MAX gains four and eight prefix occurrences versus SUM, respectively, but has one fewer useful repair / two more damaged positions on the new buffer, and equal useful repairs / one more damaged position on parent-training witnesses. Its two surrogate assessments are worse than SUM on the new buffer (+0.105725 SUM assessment, +0.048653 MAX assessment) and slightly better on parent-training witnesses. The readouts are mixed. Both sets were training data, and the new-buffer set's exclusion from the two qualification responses does not make it held-out data. Their gains cannot establish later-request usefulness or explain a percentage of serving time.

[Mechanism review](MECHANISM_REVIEW.md), [endpoint totals](WITNESS_SUMMARY.csv), [per-response readouts](WITNESS_RESPONSES.csv), [per-block readouts](WITNESS_BLOCKS.csv), [paired block changes](WITNESS_PAIRED_BLOCK_CHANGES.csv), [sparse allocation/bottleneck measurements](SPARSE_DIAGNOSTICS.csv).

## Correctness, qualification and audit

All declared gates passed on the first GPU attempt: actual no-step C preclip gradient equals the planned ordinary C gradient bitwise; SUM reproduces the full historical PR62 F_C scientific receipt; MAX passes the unchanged coefficient/alignment/support gates against 28 fixed serial native token-gradient backwards (worst coefficient error 0.3424% of beta, minimum reference-centered alignment 100.2093%; required alignment at least 90%); repeated scores and allocations agree; complete parent restoration passes. Three disposable updates remain separate from the two production events. The old hidden-state discrepancy of 2.047197% against 1.6% remains a **failed historical diagnostic**, not a retrospectively passed test. The numerical receipt's inherited `production_qualified=false` field describes its limited numerical-proof scope; the full-event and serving evidence is established separately here.

The final independent work audit passed **783,262 checks**, including all 3,255 retrieved file hashes, historical control parity, both 128-window trajectories, all 1,188 exact canonical outputs/terminations and native traces, and the resource/work ledgers. There are zero invalid, unknown or unattempted generations and no scientific retries. All 1,152 scored records plus 36 warmups / 18 activations remain included. The independent mechanism audit passed 26,785 checks; serving totals and all intervals reproduce with zero discrepancies. Public-only tables additionally reproduce 3,492 scalar checks, including all point/interval endpoints. Nine work-auditor and eight serving-auditor CPU tests pass. These counts describe overlapping checks, not independent experimental samples.

In-place CPU hashing authenticated all 5,346 source entries, eight parent/control/new endpoint files and three distinct serving tensor-content identities. Each export has 58 BF16 tensors / 1,048,626,432 elements. No model was instantiated for this check. Runtime receipts establish complete weights/moments/options/counters/RNG/history restoration and recovery/export equality; the independent audit verifies those receipts, not an unsaved full-vector reconstruction.

The audit itself exposed two checker issues: a relative root was prepended twice, and a tracked source-directory symlink was treated as a regular file. The failed/incomplete audit records are preserved; only the CPU auditors were corrected. A public-table manifest's self-inclusion was also corrected before publication. None changed the experimental implementation, measurements, settings, tolerances or GPU work. [Work audit](WORK_AUDIT.json), [mechanism audit](MECHANISM_AUDIT.json), [endpoint/source authentication](IDENTITIES.json), [audit scope and repairs](AUDIT_NOTES.md).

Serving used the pinned Qwen3-8B/DFlash BF16 SGLang/FA3 runtime, greedy block 16, max 2,048 tokens or EOS, logical context 4,096 / physical 4,112, concurrency 1, disabled capture and the qualified cache/graph policy. Evaluation ran on physical GPU A with B quiet, after training closed. Two 64-request groups rotate `(C,SUM,MAX)` by `(repeat+group)%3`; all requests see each arm in every order position. Nineteen boundary receipts check the quiet window. No training, preparation or artifact transfer overlapped timing.

## Work, costs and storage

| New work | Completed |
|---|---:|
| Disposable qualification Adam calls | 3 |
| Qualification native token-gradient oracle backwards | 28 |
| Production presentations / Adam calls | 512 / 256 |
| Total Adam applications | 259 |
| Isolated no-step C-reference forwards / backwards | 512 / 512 |
| Endpoint witness response forwards | 64 across parent/C/SUM/MAX |
| Scored generations / included warmups | 1,152 / 36 |
| New complete recoveries / exports | 2 / 2 |
| New feedback / canonical generations | 0 / 0 |

Qualified disposable startup contributes two response forwards/backwards and zero Adam steps in each of qualification, SUM and MAX (six / six total), separately from committed presentations. Qualification also includes its bounded readouts and no-step reference/scoring work. Serving initialization records 51 graph-capture forward calls across 17 capture entries, separate from the 1,188 complete generations. [Physical work](PHYSICAL_WORK.json), [chunk/recomputation accounting](PHYSICAL_ACCOUNTING.json), [qualification work](QUALIFICATION_WORK.json), [qualification report](../qualification_001/REPORT.md), [complete work audit](WORK_AUDIT.json).

| Cost (seconds) | SUM | MAX |
|---|---:|---:|
| Training, including scoring | 2,397.967 | 2,466.660 |
| Scoring, **nested in training** | 1,830.708 | 1,901.680 |
| Native C reference, **nested in scoring** | 564.577 | 563.488 |
| Checkpoint persistence | 19.535 | 20.619 |
| Serving export | 14.950 | 15.224 |
| Disposable startup | 17.714 | 20.390 |
| Whole occupied training phase, including process overhead | 2,529.215 | 2,602.555 |

MAX adds 68.692 seconds (2.8646%) to SUM's measured training time in this run. Nested timers must not be added again to training totals. Retained C training was 564.545 seconds, so the observed ratios are 4.248× for SUM and 4.369× for MAX; **C's cost is historical context, not a fresh matched overhead measurement**. No training-inclusive speedup is demonstrated. [Detailed costs](TRAINING_COSTS.csv), [cost contrasts](TRAINING_COST_COMPARISONS.csv), [historical C](HISTORICAL_C_COST.json).

Qualification occupied 261.792 card-seconds; evaluation occupied 1,485.208. All four serial phases total **6,878.770 occupied card-seconds (1h54m39s)** and 6,878.970 elapsed seconds, within four card-hours and six elapsed hours. The previously forecast 3–4 hours was conservative. Adding the explicitly scoped earlier preservation-development chain gives 12,669.702 card-seconds; this is not the entire project's lifetime compute. Model loading (51.517s), activation/switching (22.060s) and warmup request time (50.235s) are excluded from the primary serving totals. The 1,152 scored complete-request latencies sum to 1,279.149 seconds before averaging repeats. Full phase times already include these operations; they are not additive extra charges.

Both training processes peak at 39.681GB allocated / 51.831GB reserved, including startup. Qualification's post-startup peak is 35.560 / 56.617GB. Serving's recorded process peak is 57.321 / 57.634GB. These refer to different measurement scopes, not isolated algorithm allocations.

Postflight A800 allocation for this source/run plus incremental compiler growth is 17.965GB (compiler growth zero). Actual Fenrir delivery exceeds the original 600MB local forecast; it is explicitly charged at about 1.179GB, with new shared Git objects and 10MB publication headroom added. The conservative postflight total is about **19.154GB, below the unchanged 25GB cap**. A800 retains **405.989GB free**, above the 400GB reserve, and Fenrir has 73.717GB free above its 20GB allowance. Nineteen runtime boundaries and postflight measurements pass; these are sampled observations, not a continuous disk trace. Both A800 cards are idle.

The earlier authorized cleanup removed four authenticated disposable PR58 qualification tensors, freeing 16.778GB; exact reload of those two disposable endpoints was lost, not archived. Production dependencies and compact evidence were preserved. No further retirement was needed for this audit. Compact retrieval copied 814,912,280 bytes in 507.532 seconds, with no remote archive or model-tensor copy. Audit/retrieval CPU time is separate from the stopped GPU campaign clock. [Storage/accounting](RESOURCE_FINAL.json), [retrieval receipt](RETRIEVAL.json), [prior cleanup](../admission_002/REPORT.md).

## Decision and retained evidence

All three exported models have distinct tensor contents and the same inference configuration. Under the predeclared unrounded rule, an eligible candidate must have complete-request latency below C and calls no worse than C. Both satisfy only the call condition; **neither qualifies for retention. Freeze C + K+PAIR.** This is the agreed development decision, not proof that C is universally optimal. Point2B remains unexecuted; no new cap, reference, precision, smoothing, horizon, panel or endpoint was selected.

Private prompts, source IDs, token IDs, anchors, features and model/recovery tensors remain outside Git. Public tables use ordinal identifiers. Both new complete recoveries and exports remain on A800; the compact raw evidence is retained on A800 and authenticated on Fenrir. Full model-sized gradient contractions were not saved and cannot be independently reconstructed by the scalar audit. Final tensor hashes authenticate stored endpoints, not an independent re-execution of training.

The public `audit_code/` contains the independent serving/work/mechanism auditors and the public-table rechecker. Private raw receipt access is required for token-level and full-state-receipt checks; the sanitized serving statistics and bootstrap can be reproduced entirely from the published CSV/JSON files. [Audit commands and scope](AUDIT_NOTES.md). Original launch commands remain in [readiness](../readiness_001/COMMANDS.md); no command was rerun for this report.

The [method sequence](../../acl2027_method_development/SEQUENCE.md) records this completed decision. Historical failed checks and earlier park decisions remain intact. Graduate and the broader research stage are unchanged.
