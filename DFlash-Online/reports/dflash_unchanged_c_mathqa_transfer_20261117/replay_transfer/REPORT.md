# Unchanged C MathQA transfer after exact E5 control replay

Execution: `3b25b068f0caa0035aeb4c8d58ba498f02a814cc`. **PRACTICAL_TIME_GAIN_NOT_SUPPORTED**.

C/N complete-request time reduction is **+0.0255% [-0.3028%, +0.3485%]**, and native acceptance change is **+0.0620% [-0.3180%, +0.4323%]**. C/S time reduction is **+0.1207% [-0.1895%, +0.4429%]**, with native acceptance change **+0.2417% [-0.1172%, +0.6177%]**. The C/N timing interval lies below the prespecified 0.5% practical target. That target is not supported in this fixed comparison; this is not a claim of equivalence or universal ineffectiveness.

## Material Passport

- Origin Skill: ARS experiment-agent.
- Origin Mode: run and validate; report assembly performs CPU evidence validation only.
- Origin Date: 2026-11-17.
- Verification Status: ANALYZED; native qualification and exact N5/S5 replay receipts passed, and retained scalar/raw evidence was independently checked. No independent rerun of C training is claimed.
- Version Label: pr52_replay_transfer_result_v1.
- Scope: `UNCHANGED_C_MATHQA_TRANSFER_AUDITED_UNEXPOSED_PANEL`; one fixed origin, inherited seed and two-event candidate continuation.
- Scientific execution source: `3b25b068f0caa0035aeb4c8d58ba498f02a814cc`. Publication is the later Git revision containing this report; it is not the execution source.

## Experiment and prerequisite

C is the unchanged PR50 bounded correct-prefix allocation: smoothing 0.5, relative cap 0.25, detached FP32 coefficients and unchanged native BF16 K+ learning, first-error treatment, response denominator and actual learning rate 0.000300. It resumes complete MathQA KPLUS4 at 512 Adam steps / 1,024 presentations, reaches C5 at 640 / 1,280, collects C5's own event-six feedback, and reaches C6 at 768 / 1,536. Complete optimizer, RNG and event history continue with inherited seed 17 and horizon 7,424; neither is reset. Each event uses its current 128-response pool for two passes, microbatch one and accumulation two, with response-only supervision; prompts supply context. Retained PR51 N6/S6 endpoints use learning rates 0.000300/0.000288 and receive fresh timings here. No endpoint was selected using evaluation outcomes.

The original [readiness report](../REPORT.md) is preserved. Its missing historical feature fingerprints were addressed prospectively by exact N5 and S5 functional replays, not by a byte-identity claim. Both replays match all 256 response observations, 256 presentation receipts and 128 windows, plus complete draft, Adam/RNG/configuration/ancestry/boundary categories. `historical_feature_identity_verified=false` remains explicit. [N5 receipt](REPLAY_N5.json), [S5 receipt](REPLAY_S5.json) and [replay provenance](REPLAY_PROVENANCE.json) retain the exact comparison exclusions and original artifact identities. This qualifies the retained pool's functional reproduction, not C's efficacy.

The fixed input-only panel has 256 scored requests and two warmups, selected from 1,702 eligible MathQA test rows after 1,283 enumerated exclusions. Freshness is relative to those inventories. Frozen KPLUS4 supplies 258 new native greedy target-verifier reference outputs; this is not a target-only autoregressive timing baseline. Serving uses pinned Qwen3-8B / DFlash-b16 with SGLang/FA3 ([bindings](BINDINGS.json)), BF16 greedy block 16, 2,048 tokens or EOS, context 4,096, concurrency one and capture disabled. All 2,304 scored executions and 72 warmups match exact canonical output and termination, with repeat-consistent counts on one quiet physical GPU.

## Final serving comparison

Counts use one validated trajectory per request; time uses each request's mean across three repeats. Pooled native acceptance is `Σ_i (output_tokens_i − 1 − verification_calls_i) / Σ_i verification_calls_i`: one separately generated first token is excluded for **each** request. Stop-clipped matches are reported separately.

| Arm | Calls | Native accepted/call | Stop-clipped matches/call | Progress/call | Request means sum (s) |
|---|---:|---:|---:|---:|---:|
| N6 | 18,664 | 6.414220 | 6.425579 | 7.414220 | 326.620268 |
| C6 | 18,654 | 6.418194 | 6.429291 | 7.418194 | 326.537047 |
| S6 | 18,693 | 6.402718 | 6.414005 | 7.402718 | 326.931650 |

| Comparison | Time reduction, 95% interval | Native acceptance change, 95% interval | Calls saved / added / net | Request wins / regressions / ties |
|---|---:|---:|---:|---:|
| C_vs_N | +0.0255% [-0.3028%, +0.3485%] | +0.0620% [-0.3180%, +0.4323%] | 166 / 156 / 10 | 91 / 81 / 84 |
| C_vs_S | +0.1207% [-0.1895%, +0.4429%] | +0.2417% [-0.1172%, +0.6177%] | 177 / 138 / 39 | 97 / 78 / 81 |
| S_vs_N | -0.0953% [-0.3953%, +0.1945%] | -0.1793% [-0.5232%, +0.1511%] | 145 / 174 / -29 | 84 / 98 / 74 |

Each endpoint returns 138,635 output tokens, including 13 capped requests and 243 EOS terminations; zero-call requests: 0. Invalid, unknown and unattempted final executions are all zero.

C/N time reductions by repeat: +0.0311%, +0.0404%, +0.0049%. C/S: +0.0803%, +0.1330%, +0.1490%. Full repeat totals and native/cache times are retained separately.

Uncertainty uses 10,000 common whole-paired-request bootstrap draws, PCG64 seed 1717, lexicographic source/request order and linear quantiles. These intervals condition on the fixed endpoints; they do not establish training-seed robustness. Identical-output call and acceptance changes are linked, not independent mechanism evidence.

## Fixed decision

| Prespecified check | Passed |
|---|---|
| C_vs_S_acceptance_point_positive | True |
| C_vs_S_time_point_positive | True |
| acceptance_lower_bound_positive | False |
| every_repeat_faster | True |
| time_lower_bound_positive | False |
| time_point_at_least_half_percent | False |

The C/N timing interval lies below the prespecified 0.5% practical target. That target is not supported in this fixed comparison; this is not a claim of equivalence or universal ineffectiveness. Unchanged C closes under the agreed rule for this comparison; this is not a universal algorithm-failure claim. No automatic tuning, further events or additional panel is authorized. Native K+ remains the primary baseline. Passing the C/S point-estimate gate does not establish superiority when its confidence interval crosses zero.

## Execution and costs

Independent final audit: 1,991,303 checks. Separate training/replay reader: 83,386 checks. Both pass; the training reader overlaps with implementation authorship and is not blind replication. Runtime tensor-equality receipts, rather than independent full-tensor reloads, support exact state comparisons.

The first private training-reader attempt reported 7,684 mismatches after 83,390 checks. It incorrectly treated physical proposal slots 1–15 as zero-based and compared already-excluded publication paths. The original failed reader and repair note are retained. Its correction changed no experiment code, model state, raw evidence or scientific tolerance; this was not a GPU failure. [Reader repair](AUDIT_READER_REPAIR.md).

A separate coefficient-table audit passes 63,756 checks over both events' 512 original response observations, including all prefix-length and physical-slot aggregates.

The coefficient summaries cover exactly 512 original response forwards. [Per-event totals](COEFFICIENTS.json), [prefix lengths](COEFFICIENT_PREFIX_LENGTH.csv) and [length/slot sums](COEFFICIENT_LENGTH_SLOT.csv) retain their occurrence denominators. All recorded mass, bounds, support and nonfinite violation totals are zero. These measurements confirm application of the intervention; they do not establish its causal benefit.

| Work | Completed | Limit |
|---|---:|---:|
| adam | 524 | 528 |
| canonical_generation | 258 | 258 |
| feedback_generation | 128 | 128 |
| qualification_generation | 8 | 8 |
| response_backward | 1060 | 1088 |
| response_forward | 1060 | 1088 |
| scored_generation | 2304 | 2304 |
| teacher_preparation | 128 | 128 |
| training_initialization | 6 | 8 |
| warmup_generation | 72 | 72 |

| Stage, startup included | Adam steps | Response forwards | Response backwards | Trainer initializations |
|---|---:|---:|---:|---:|
| functional_replays | 256 | 516 | 516 | 2 |
| qualification | 12 | 28 | 28 | 2 |
| C_production | 256 | 516 | 516 | 2 |
| totals | 524 | 1060 | 1060 | 6 |

Response counts include each stage's disposable startup; C production core is 512 forwards/backwards, plus four startup forwards/backwards.

| C production event | Intrinsic training (s) | Load (s) | Disposable startup (s) | State save (s) | Export (s) | Event wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| C5 | 556.083 | 67.194 | 41.421 | 19.956 | 16.484 | 604.378 |
| C6 | 574.292 | 67.784 | 43.616 | 22.480 | 16.816 | 634.802 |

| Other required production work | Recorded cost (s) |
|---|---:|
| Own event-six feedback generation | 194.287 |
| Own event-six teacher preparation | 23.071 |
| Own event-six collection loading | 24.011 |
| Frozen reference task wall | 388.126 |
| Final evaluation loading | 20.007 |
| Final evaluation wrapper wall | 3230.701 |
| Final complete worker wall | 3281.505 |

Event and wrapper walls include nested operations; table columns are not disjoint and must not be added into a speedup claim.

Original estimate: **97.0 minutes raw / 155.5 minutes conservative**. Actual original continuous clock: 134.427 elapsed minutes and 135.053 occupied-card minutes, each below 240 minutes. N/S prerequisite replay costs are additional diagnostic overhead; no duplicate replay state/export was published. C5/C6 intrinsic training, disposable startup, loading, collection, teacher preparation, reference generation, publication, warmups, native serving and cache maintenance are separated in [COSTS.json](COSTS.json). Nested timers overlap and must not be summed as disjoint costs. No training-inclusive speedup is claimed, and historical PR51 training timings are not a matched overhead control.

Admitted peak allocation: 81.586 GB. Final charged allocation: 43.178 GB; final free space: 529.365 GB. The 90 GB allocation cap and 400 GB free-space reserve remain explicit. [STORAGE.json](STORAGE.json) retains actual retirement/deferred receipts and sampling limits. No historical cleanup was part of this experiment.

Both physical GPUs had no active compute processes or card leases in the post-completion checks before and after packing compact evidence. These are recorded boundary checks, not a claim of continuous device monitoring.

## Historical context and interpretation limits

These are separate historical panels and fresh-versus-historical latency is not treated as a learning curve.

| Retained study | Contrast | Native acceptance change, 95% interval | Request-time reduction, 95% interval | Original decision |
|---|---|---:|---:|---|
| [GSM8K, 128 requests](../../dflash_bounded_prefix_allocation_20261116/REPORT.md) | C_vs_N | +0.8498% [+0.0721%, +1.6292%] | +0.6860% [+0.0470%, +1.3098%] | `CLOSE_FIXED_CANDIDATE_AT_THIS_SCREEN` |
| [GSM8K, 128 requests](../../dflash_bounded_prefix_allocation_20261116/REPORT.md) | C_vs_S | -0.5250% [-1.2600%, +0.1948%] | -0.4607% [-1.0675%, +0.1264%] | `CLOSE_FIXED_CANDIDATE_AT_THIS_SCREEN` |
| [MathQA, 256 requests](../../dflash_mathqa_fixed_lr_transfer_20261116/REPORT.md) | S_vs_N | +0.1638% [-0.2059%, +0.5126%] | +0.1112% [-0.2141%, +0.4153%] | `PRACTICAL_TIME_GAIN_NOT_SUPPORTED` |

The original PR50 GSM8K C/S advancement gate remains failed; this prospective MathQA test does not revise it. Its negative C/S points with intervals crossing zero did not establish inferiority or equivalence. PR51 S/N evidence is contextual and its practical-gain criterion remained unsupported. [Pinned historical numbers and identities](HISTORICAL_CONTEXT.json) preserve the exact source values. No N/S training replication beyond the exact E5 prerequisite was claimed. No general seed robustness, adaptive-controller benefit or gradient-level causal explanation follows.

[SUMMARY.json](SUMMARY.json), [PER_REQUEST.csv](PER_REQUEST.csv), [ALL_REPEATS.csv](ALL_REPEATS.csv), [WARMUPS.csv](WARMUPS.csv), [REPEAT_TOTALS.csv](REPEAT_TOTALS.csv), [TRACE_COUNTS.csv](TRACE_COUNTS.csv), [VALIDITY.json](VALIDITY.json), [QUALIFICATION.json](QUALIFICATION.json), [BINDINGS.json](BINDINGS.json), [INPUT_SELECTION.json](INPUT_SELECTION.json), [INPUT_IDENTITIES.json](INPUT_IDENTITIES.json), [AUDITS.json](AUDITS.json), [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json). Private prompts, generated text, token arrays, raw proposal traces and tensor payloads remain outside Git. Public checkpoint locators are role aliases; numerical CSV fields are unchanged.
