# Released AdaFlash loss under our matched online protocol: completed comparison

## Material Passport

ARS experiment-agent run → validate. The prescribed campaign completed; this report analyzes retained native evidence, not an independent GPU replication. Scientific execution commit: **`7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a`** ([code](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a), [PR #31](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/31)). No algorithm, tolerance, panel or endpoint changed after sealing.

**Result:** the A_repo online recipe produces a stronger endpoint than both ordinary OSD and the earlier paper-defined AdaFlash arm. **K+ still leads it by 1.04% fewer verification calls and 1.21% lower complete-request time** on this panel. The primary paired 95% interval for request-time reduction is **[0.57%, 1.86%]**. Among the two AdaFlash-style recipes measured in this session, A_repo is the stronger control.

Separate same-team audits passed **4,310,189 endpoint checks** and **402,173 training/accounting checks**, with no discrepancies. They use separate calculations on retained records; implementation authorship overlaps are disclosed, and no independent GPU replication is claimed.

## What was compared

Only **A_repo** was newly trained. Its label is **released AdaFlash loss under our matched online protocol**. O58, K+58 and paper-A58 were retained unchanged and measured freshly alongside A_repo58.

The new loss ports [AdaFlash `478bf0f`, `pipeline/training/rkl_model.py:168–207`](https://github.com/ZinYY/AdaFlash/blob/478bf0ff7a3a5232b13d0f14e752d5fe1da0b718/pipeline/training/rkl_model.py#L168): 0.8 teacher-argmax cross-entropy plus 0.2 reverse KL at temperature one. It preserves the released probability operations, intermediate dtypes and reduction order: vocabulary sum, division by the full response weight sum plus 1e-6, clipping each normalized position at 0.01, position weighting, then the late mixture of separately reduced components. All valid proposals receive supervision, with the seed excluded and gamma 7. It is neither the earlier paper-entry clipping arm nor the PR #30 common-arithmetic proxy. No adaptive-length head is used.

The Qwen3-8B/DFlash pair, released initialization, fresh seed-17 BF16 AdamW, constant learning rate 3e-4, two shuffled passes, microbatch one and accumulation two remain qualified settings. A_repo generated its own feedback from the original 7,473 ordered GSM8K training requests. Each 128-response window triggered one 128-step event; 58 events produced 7,424 Adam steps and 14,848 presentations. The last 49 requests were served without another update. Earlier arms also used their own feedback, so this compares complete online recipes rather than isolating the loss on identical hidden features.

The final 1,024-request panel is train-disjoint but previously used. BF16/FA3, greedy five-layer drafting, block size 16, 2,048 tokens or EOS, context limit 4,096, concurrency one, disabled capture and the established cache policy remain fixed. Three balanced repeats give **12,288 timed completions**. There was no concurrent training or artifact transfer during measured serving.

## Fresh endpoint results

All counts below use one validated trajectory per request after checking repeat consistency. Time totals sum each request's mean over three repeats; they do not treat repeats as independent requests.

| Method | Verification calls | Complete-request time (s) | Native-generation time (s) | Cache-maintenance time (s) | Runtime accepted / call | Matched accepted / call | Committed progress / call |
|---|---:|---:|---:|---:|---:|---:|---:|
| O58 | 44,730 | 808.861 | 757.941 | 36.558 | 6.0159 | 6.0370 | 7.0159 |
| K+58 | **43,267** | **785.697** | **735.782** | 36.566 | **6.2532** | **6.2755** | **7.2532** |
| Paper-A58 | 45,083 | 816.072 | 764.865 | 36.726 | 5.9610 | 5.9818 | 6.9610 |
| A_repo58 | 43,721 | 795.352 | 742.192 | 39.509 | 6.1779 | 6.1990 | 7.1779 |

Every method produced the same **314,847 output tokens**. Committed progress totals **313,823**, excluding one initial target-prefill token per request. The runtime accepted counter counts committed progress minus one per verification call; the matched counter counts canonical-matching draft proposals clipped at the stopping boundary. EOS/cap handling can make them differ. The native and cache components do not exhaust complete-request latency; no detailed kernel-time attribution follows from these timers.

| Prespecified comparison; positive favors first method | Calls reduced | Request-time reduction [paired 95% CI] | Request call wins / regressions / ties | Gross calls saved / added |
|---|---:|---:|---:|---:|
| **K+ / A_repo — primary** | **1.0384%** | **1.2140% [0.5722%, 1.8611%]** | **472 / 370 / 182** | **1,256 / 802** |
| K+ / O | 3.2707% | 2.8639% [2.1752%, 3.5639%] | 633 / 250 / 141 | 1,917 / 454 |
| A_repo / O | 2.2558% | 1.6701% [0.9716%, 2.3544%] | 593 / 258 / 173 | 1,481 / 472 |
| A_repo / paper-A | 3.0211% | 2.5390% [1.8371%, 3.2419%] | 663 / 202 / 159 | 1,714 / 352 |

The primary call difference is **454 calls** and the complete-request time difference is **9.656 s** over 1,024 requests. K+'s native-generation time is also lower by **0.8636% [0.3497%, 1.3631%]**. The primary complete-request advantage favors K+ in all three repeats. The cache-only interval crosses zero (−2.38% to +16.59%); it does not establish a stable cache-cost advantage.

| Timing repeat | O (s) | K+ (s) | Paper-A (s) | A_repo (s) |
|---|---:|---:|---:|---:|
| 1 | 808.901 | 784.693 | 814.649 | 794.127 |
| 2 | 809.152 | 785.832 | 816.557 | 795.811 |
| 3 | 808.532 | 786.565 | 817.011 | 796.119 |

The sealed bootstrap resamples paired requests 10,000 times with PCG64 seed 20261102 and linear 2.5/97.5 percentiles. The primary interval addresses request variation for these fixed checkpoints and this timing session. The other intervals are unadjusted descriptive comparisons, not additional independently confirmed primary outcomes.

## Adaptation work and costs

| A_repo work | Completed |
|---|---:|
| Own-feedback generations | 7,473 |
| Responses used for training / untrained final tail | 7,424 / 49 |
| Training events / native Adam steps | 58 / 7,424 |
| Response presentations / core response forward-backwards | 14,848 / 14,848 |
| Sampled anchor occurrences | 3,728,050 |
| Supervised position occurrences | 54,368,391 |
| Clipped normalized-position occurrences | 11,559,628 / 54,368,391 (**21.2617%**) |
| Recorded zero draft probabilities / projected probability entries | 0 / 9,062,800,076,800 |
| Recorded nonfinite reverse-KL entries | 0 |

Exposure counts sum original response observations once, excluding autograd checkpoint recomputation and duplicate accumulation records. They concern overlapping sampled occurrences, not independent served tokens. The zero-probability denominator includes all projected vocabulary entries, including masked positions; the clipping denominator includes only eligible supervised occurrences. Scalar training diagnostics also passed their finiteness gates. Teacher argmax differs from the saved canonical token in **616** supervised occurrences; the prescribed CE uses the teacher argmax. This does not alter the serving-token equality requirement.

| Fresh A_repo cost | Seconds | Scope |
|---|---:|---|
| Ordinary training | 19,164.221 | Includes native training arithmetic and compilation |
| Whole update calls | 22,170.554 | Encloses teacher preparation, persistence, training/export, activation and commit |
| Terminal teacher preparation | 288.257 | Nested within update calls |
| Feature persistence | 338.837 | Nested within update calls |
| Checkpoint persistence | 1,094.266 | Nested within update calls |
| Serving export | 808.707 | Nested within update calls |
| Training-time checkpoint activation | 89.722 | Host activation wrapper; nested within update calls |
| Publication validation | 1,401.258 | Diagnostic correctness work |
| Retention work | 536.867 | Validated bulk retirement |
| Complete trajectory worker | 31,668.327 | Includes own-feedback serving, adaptation and overhead |
| Four-arm evaluation wall | 10,431.850 | Includes 12,288 timed completions and evaluation overhead |
| Endpoint loading | 21.007 | Outside request latency |
| Endpoint checkpoint activation | 321.754 | Outside request latency |
| Endpoint weight validation | 8.207 | Outside request latency |

These timers have overlapping scopes and must not be added. Teacher-head computation during training is 5,935.429 s, already included in training; it is distinct from terminal feedback preparation. The training-time native activation subcomponent is 19.286 s, already included in the 89.722 s host wrapper. Training peak allocated/reserved memory was **20.674 / 23.977 GB**. Whole campaign wall time was **42,153.573 s (11 h 42 m 34 s)** before final report/postflight overhead. Qualification cost is separate: **242.276 s**, six disposable Adam steps, 12 core response forward-backwards, two startup forward-backwards with zero Adam steps, and six tiny numerical fixture forward-backwards. Its state was discarded before main training. Four inherited first-event correctness completions are separate from the 12,288 timed endpoint completions.

O/K+/paper-A training costs remain historical measurements. Fresh serving results do not establish a training-inclusive speedup, asynchronous feasibility or equal-compute adaptation. The 19,765 complete generations comprise 7,473 own-feedback, four correctness and 12,288 endpoint generations; there were no additional generation warmups or automatic retries.

## Validity, retention and interpretation

All **12,288/12,288 executions** passed exact canonical token/termination checks and within-checkpoint repeat consistency, with zero invalid, unknown or unattempted cases. Each method has 1,021 EOS responses and three cap hits; all remain included. The endpoint audit reconstructed **530,403 native verification rounds** from **24,597 consumed compact files**, verified all 186 checkpoint publications and 192 timing boundaries, and independently reproduced the fixed bootstrap and every reported contrast. No production analysis function was imported.

The separate training audit reconciled all 58 events, response/anchor identities, two-pass exposure, native Adam accounting, scalar loss reductions, four inherited correctness generations, the 49-response untrained tail, costs and retention. It passed 402,173 checks; 29 additional local sums and inventory checks passed. Past full-state/export equality is supported by saved receipts; this analysis did not reload full optimizer/model tensors.

All **97,485 original compact files (5,764,699,468 bytes)** were independently rehashed on A800 after completion with zero differences from the frozen inventory. The complete Fenrir mirror is being copied and will receive a separate retrieval receipt; the nine original generated report/table files are already locally hash-verified. Raw tokens, responses and tensor states stay outside Git. All eleven ARS statistical-interpretation risks were reviewed; the largest limits are one seed, the reused mathematics panel, and own-feedback rather than identical-feature training.

The campaign retained its initial state, final A_repo58 complete recovery state and export. During the 58 validated event boundaries it retired 7,424 prepared feature files, 57 superseded recovery states and 57 superseded exports. Their cumulative file size is **599,331,995,491 bytes across the run**, not space reclaimed simultaneously. Separately, 7,424 preparation-retirement receipts preserve the checked conversion from native capture tensors to prepared features; their retired bytes are not added to that event-retirement total. Exact original feature-byte reinspection and intermediate optimizer/checkpoint continuation are no longer available; raw outputs, compact traces, anchors, metrics and validation receipts remain. Those tensors were retired, not archived. Source/model/retained O58/K+58/paper-A58 export checks passed after execution.

Before main training, the separately documented storage operation retired six older completed recovery-state inodes through 12 paths, reclaiming 37.826 GB and preserving their serving exports/evidence. The original storage-admission stop and the first retirement's pre-unlink permission stop remain preserved. Neither repeated scientific work. The successful main supervisor ran **01:02:22–12:46:33 UTC on November 7 (11 h 44 m 11 s), exit 0**. Its status is under `operational_continuation_001`; the older root-level exit 1 remains the prior admission record. No experiment was restarted, tuned or retried.

The supported conclusion is narrow: **K+ retains an endpoint advantage over the literal released AdaFlash loss in this matched one-seed mathematics protocol; the A_repo online recipe outperforms both OSD and the earlier paper-defined recipe.** This does not establish performance of AdaFlash's complete system, cross-domain or seed robustness, or the isolated causal benefit of clipping. Each arm's own feedback and retained history remain part of the recipe.

The experiment is complete. No tuning, further event or automatic follow-up was performed or is selected by this report. Prior failures, qualification records, sealed reports and checkpoints remain preserved according to the recorded retention policy. Research remains at Stage 1 / project None.

Pinned configuration, loss contract, model/runtime identities and endpoint export receipts are in `configuration/`. Machine-readable evidence: `SUMMARY.json`, `VALIDITY.json`, `PAIRED_REQUESTS.csv`, `ALL_REPEATS.csv`, `REPEAT_TOTALS.json`, `TRAINING.json`, `TRAINING_EXPOSURE.csv`, and `EVENT_COSTS.csv`. Original generated prose is preserved separately as `GENERATED_REPORT.md`; raw answers/token traces and tensor artifacts remain private outside Git.
