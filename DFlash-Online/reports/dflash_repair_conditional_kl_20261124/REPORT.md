# Repair-conditional KL: implementation and bounded qualification

**Native qualification passed. The production pilot was not admitted because its conservative storage peak and measured offload time exceed the handoff's limits. PAIR's effect on drafting remains unassessed.** This is the handoff §6 qualification-only fallback, selected before GPU work—not a performance-based early stop.

Execution commit: **`79b8c076610e9b4d310169b3891cb76ac6d0f9bb`**, clean published checkout, based directly on PR56's `1551e5e5d0dd7736d8441977aa3dea250535da7b`. [PR #58](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/58). The later report-publication commit is not the execution revision. Graduate and all historical experiments were unchanged.

## Implementation and identities

BASE is the complete native **C + K+** objective. PAIR adds `(1-mu) KL(p_pair || q_pair)` only at each current first error, using the aligned canonical token and detached native winning rival. The auxiliary receives the same final response-wide redistributed K+ coefficient and the same response denominator as BASE. There is no new tunable coefficient, model forward, vocabulary projection, or per-role backward.

The native full-vocabulary BF16 FKL, C allocation, selection, full-block attention and response accumulation remain unchanged. Only selected first-error rows enter the new FP32 calculation; no full-vocabulary FP32 softmax is created. Teacher inputs, roles, rival indices and coefficients are detached; gathered draft logits remain differentiable. Small positive `log_mu` roundoff permits eight FP32 epsilons scaled by the log partitions, with every clamp counted. Larger violations and nonfinite values fail. The BF16-plus-FP32 computation is not claimed to be an exact floating-point KL decomposition.

Resolved recipe: released Qwen3-8B/DFlash-b16, five draft layers, seed 17 and fresh AdamW, constant LR `3e-4`, weight decay `0.01`, clipping `1`, up to 512 anchors, eight-block head chunks, microbatch 1/accumulation 2. Native horizon 7,424 steps remains intact; the pilot limit is separate. Gamma 7, repair interpolation 0.5, prefix smoothing 0.5 and relative bound 0.25 remain native. Training uses FlexAttention; serving uses pinned SGLang/FA3, BF16, greedy block 16, 2,048 tokens or EOS, context 4,096/physical allocation 4,112, concurrency 1, native graph/cache policy.

The first 128 PR56 training requests, first 128 held-out requests, two warmups and two reserved qualification requests were authenticated and sealed with their original identities. All 260 are disjoint. Existing frozen-F canonicals were retained; no target-only AR benchmark or replacement prompts were generated. CPU preflight checked 20 model artifacts, the native runtime, training environment and source/input identities. [IDENTITIES.json](IDENTITIES.json) records the exact hashes/configuration. Training uses Torch 2.13.0+cu130/Transformers 5.12.1; serving uses Torch 2.9.1+cu130/Transformers 4.57.1.

Core code: [repair_conditional_kl.py](../../online_feedback/repair_conditional_kl.py), its integration in [objective.py](../../online_feedback/objective.py), and the [pilot package](../../experiments/repair_conditional_kl_pilot/README.md).

## Qualification result

63 unique focused/inherited CPU tests passed: 10 new objective checks, 33 inherited objective checks and 20 pilot tests. Coverage includes FP64 formula/gradient references, disabled-mode neutrality, support/EOS boundaries, mixed successor groups across chunks, separate response normalization, checkpoint recomputation, flag propagation, state export/reload, bounded accounting and canonical validation. Tiny CPU fixtures exercise full 128-step event accounting; they do not establish full-event GPU feasibility. A later formatting-only change preserved AST/literals/comments and passed 13 repeated checks.

| Native requirement | BASE | PAIR |
|---|---:|---:|
| Real learning presentations / Adam steps | 2 / 1 | 2 / 1 |
| Disposable startup forwards/backwards / Adam steps | 2 / 0 | 2 / 0 |
| Finite draft-gradient tensors; head/embedding frozen | 58; pass | 58; pass |
| Exact exported weights and complete state reload | Pass | Pass |
| Exact startup restoration | Pass | Pass |
| Native generations valid / attempted | 4 / 4 | 4 / 4 |
| Export → released → export activation and cache refresh | Pass | Pass |

A separate CPU audit established exact equality of both initial released-weight witnesses, empty Adam state and RNG without overwriting RNG. Both original-forward anchor schedules, ordinary FKL numerators, denominators and allocation coefficients also match. Each arm collected its own two disposable responses. Serving used physical GPU A for BASE and B for PAIR, with training on the opposite card; both cards were exercised.

All eight generations reproduce retained canonical tokens and termination. Feedback counts reproduce 30 and 121 calls. The one reload-check request takes 191 calls with BASE and 190 with PAIR, each stable across two reload checks. These are qualification observations from one disposable step, **not the declared endpoint, a serving-speedup comparison or evidence of generalization**. See all eight rows in [QUALIFICATION_REQUESTS.csv](QUALIFICATION_REQUESTS.csv).

No GPU failure, nonfinite result, unknown operation or retry occurred. Preparatory CPU package/import and author-document symlink checks exposed issues that were fixed before the final published execution; those earlier logs remain preserved. Each training receipt deliberately leaves its later serving check pending; the final native status, reload receipts and CPU audit establish the combined pass.

## Was the new term active?

These statistics come only from the two original training forwards, excluding startup and checkpoint recomputation.

| Measurement | BASE | PAIR |
|---|---:|---:|
| Sampled blocks / valid proposal occurrences | 803 / 11,897 | 803 / 11,897 |
| Selected proposal occurrences / first-error occurrences | 7,422 / 557 | 7,422 / 557 |
| Positive auxiliary occurrences | Disabled | 467 / 557 (83.84%) |
| Mean ordinary response-normalized loss | 0.15587918 | 0.15587918 |
| Mean auxiliary loss | 0 | 0.01127588 |
| Mean total loss | 0.15587918 | 0.16715506 |
| Native pre-clipping gradient norm | 0.23046875 | 0.24218750 |
| Clipped optimizer windows | 0 / 1 | 0 / 1 |

The auxiliary is 7.2337% of the mean ordinary loss on this fixture. Mean pair mass is 0.896315; mean complement is 0.103685; mean conditional KL is 1.327442 across all 557 error rows. The teacher prefers the canonical token on 550 rows, ties on seven and strictly prefers the rival on zero. No `log_mu` clamp occurred. This establishes an active numerical intervention, not useful learning. Loss means average separately normalized responses; they are not ratios of pooled response numerators. [MECHANISM.json](MECHANISM.json) retains every numerator, denominator and count.

## Costs and complete accounting

The native launch completed in **288.524 s (4.81 minutes)** and **554.446 occupied card-seconds (0.1540 card-hours)**, including both-card reservations. The conservative prelaunch forecast was 20 minutes, with 30 minutes allowed in the forecast. The original cumulative clock was retained. A later delivery audit records 727.267 s elapsed and the same occupied GPU time, with both cards idle; CPU reporting adds elapsed time only. See [DELIVERY_RESOURCE.json](DELIVERY_RESOURCE.json).

| Measurement | BASE | PAIR |
|---|---:|---:|
| Native backend loading | 20.507 s | 20.257 s |
| Disposable startup | 21.177 s | 12.743 s |
| Recorded two-response training timer | 4.502 s | 4.936 s |
| Entire training qualification, including state/export checks | 94.781 s | 86.716 s |
| Peak training allocated memory | 18.461 GB | 18.403 GB |
| Peak serving allocated memory | 57.321 GB | 57.321 GB |

The observed training-timer difference is +0.434 s for PAIR. One step per arm on different cards and in serial order is **not a reliable production overhead estimate**. Startup, loading, publication and training timers are nested; do not add them together. Export/serialization were included in training-qualification wall time but not separately timed. Native activation/cache timings are retained in [COSTS.json](COSTS.json).

Completed work: four disposable feedback generations, four reload-check generations, four terminal teacher forwards (2,388 input rows; 0.198 s), four learning response forwards/backwards and two Adam steps. Separate startup adds four response forwards/backwards and zero Adam steps. Learning and startup each record 404 physical objective-chunk executions and 808 vocabulary-head calls across the arms, including recomputation. Initial native graph setup adds 102 instrumented model forwards. Request traces contain 1,064 verification rounds and eight prefills. No additional complete-response warmup, witness forward, production event or scored generation ran.

`accepted_draft_tokens` in the CSV preserves the native stop-clipped matching-prefix counter. `native_accepted_draft_tokens` separately satisfies `committed_progress - verification_calls`; the CSV gives both per-call ratios. These counters are not interchangeable, especially at EOS.

## Why the production pilot did not run

Before GPU work, A800 had approximately **442.062 GB free**. The conservative full-cap one-event peak requires **52.791 GB additional**, hence **452.791 GB free** to retain the mandated 400 GB reserve: a **10.729 GB shortfall** at final sealing. Two events also fail admission. A 26.37 MB transfer took 128.89 s; at that measured rate, offloading even the historically expected 3.83 GB first-arm feature pool would exceed five hours, before other work, against the three-hour elapsed ceiling. This is a conservative capacity forecast, not a claim that all responses would reach the cap.

The qualification-only envelope was 25 GB. Its final audit recorded 17.235 GB new allocation plus 0.013 GB compiler growth and **425.147 GB free**, remaining above the reserve. The 90 GB additional-allocation, three-hour elapsed and five-card-hour limits were unchanged. Only 117.501 MB of newly created redundant capture/terminal tensors were retired after exact prepared-packet validation; all pre-existing artifacts were preserved. Exact paths, retained replacements and deletion limitations are in [ARTIFACTS.json](ARTIFACTS.json).

**Unexecuted one-event work:** 256 production feedback generations and teacher preparations, 512 learning presentations/forwards/backwards, 256 Adam steps, 768 scored generations and 24 declared generation warmups. Production witness measurements were not attempted. Event-1/2 serving comparisons, confidence intervals, repaired boundaries, damaged prefixes and net prefix gain are **not measured**, not zero. [ENDPOINTS.json](ENDPOINTS.json) keeps them explicitly unavailable. The initial two-event maximum and candidate one-event version were both rejected before launch; no interrupted two-event run was relabelled.

## Decision and retained evidence

**Retain the scientific question as unassessed and stop at the authorized qualification-only result.** The tested code computes an active auxiliary and passes the bounded native checks. It does not yet justify expansion, establish a performance benefit or show that conditional structure is better than stronger repair gradients. A future production run first needs a feasible storage/transfer plan under the limits, or an explicit resource decision; no algorithm change or tuning follows from this result.

Private run root: `/data/sources/minhao/runs/dflash_repair_conditional_kl_20261124/execution_001`. Both qualification exports, full disposable states, prepared feedback, original-forward anchors, raw traces and all checks remain there. The states have one step/two presentations and no committed production event; they are not completed-event resume checkpoints. Public files contain scalar measurements and identities, not prompts, token arrays, traces or tensors. All 79 compact retrieved receipts were hash-verified. [AUDIT.json](AUDIT.json) records the independent CPU recheck; no new model work was used for it.
