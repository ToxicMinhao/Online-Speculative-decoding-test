# K+ role-dependent loss pilot: reviewed result

## Material Passport

- ARS-Codex 3.24.2 experiment-agent, validate; **ANALYZED**.
- Scientific execution commit: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599` ([PR #32](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/32)).
- Six completed trajectories, both prescribed checkpoints, no retuning or additional GPU work.
- Separate same-team source, raw-endpoint, training and numerical audits; no independent GPU replication.

## Finding

**ROLE did not demonstrate the intended benefit over K+ in this four-event pilot.** Its endpoint point estimates used more verification calls and more request time on both datasets, while all primary intervals include zero. ROLE also added 8–9% measured training time. This supports keeping K+ for now; it does not prove ROLE is universally harmful or equivalent.

MIX produced a small GSM8K call-count improvement but regressed MathQA. Assigning the mixture only to the first error did not produce a consistent advantage over applying it to every selected position.

## What was tested

KPLUS keeps forward KL on the current matching prefix and first error. MIX uses the 0.8 teacher-argmax CE / 0.2 released-repository clipped reverse-KL mixture on both roles. ROLE keeps forward KL on the prefix and uses the mixture only on the first error. All retain the native K+ priorities, selected-position normalization, full-response denominator inside the reverse-KL cap, BF16 training and full-block attention.

Each dataset/arm starts independently from released Qwen3-8B-DFlash-b16 and a fresh seed-17 AdamW. Four events each collect 128 own responses and train two passes, giving 512 Adam steps per trajectory. LR is constant 3e-4, weight decay 0.01, clipping 1, microbatch 1/accumulation 2, gamma 7 and up to 512 anchors per response presentation. Serving is pinned Qwen3-8B/DFlash, BF16/FA3/five layers, greedy block 16, 2,048 tokens or EOS, context 4,096 and concurrency one.

The fixed 256-request development panel per dataset is train-disjoint but previously used. Event 2 has one timed repeat; event 4 has two, averaged within each request. Timing repeats are not independent samples. All arms are benchmarked after training has closed, with capture disabled, cold-cache policy and separately charged checkpoint activation.

## Endpoint: event 4

Positive percentages mean fewer calls or less time for the named candidate. Calls use one validated trajectory per request. Times are sums of within-request repeat means.

| Dataset | Comparison | Call reduction % [95% CI] | Request-time reduction % [95% CI] | Call wins / regressions / ties |
|---|---|---:|---:|---:|
| gsm8k | ROLE versus KPLUS | -0.149 [-0.753, +0.430] | -0.595 [-1.633, +0.405] | 103 / 100 / 53 |
| gsm8k | MIX versus KPLUS | +0.842 [+0.139, +1.537] | +0.634 [-0.531, +1.780] | 122 / 83 / 51 |
| gsm8k | ROLE versus MIX | -0.999 [-1.681, -0.332] | -1.236 [-2.288, -0.178] | 84 / 133 / 39 |
| mathqa | ROLE versus KPLUS | -0.492 [-1.000, +0.006] | -0.390 [-1.242, +0.469] | 99 / 110 / 47 |
| mathqa | MIX versus KPLUS | -0.896 [-1.442, -0.376] | -1.028 [-1.930, -0.192] | 95 / 118 / 43 |
| mathqa | ROLE versus MIX | +0.401 [-0.146, +0.940] | +0.631 [-0.310, +1.593] | 114 / 100 / 42 |

The ROLE/K+ comparisons are primary, separately for each dataset. All intervals are the prespecified 10,000 paired-request PCG64 bootstrap draws, seed 20261107, ratio of summed measurements and linear percentiles. Secondary and event-2 intervals are descriptive and unadjusted; no pooled cross-dataset or familywise-confirmation claim is made.

## Both checkpoints and accepted progress

| Dataset | Event | Arm | Calls | Request seconds | Native-generation seconds | Cache-maintenance seconds | Matched accepted proposals/call | Committed progress/call |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 2 | KPLUS | 12307 | 222.951 | 209.146 | 9.583 | 5.5394 | 6.5196 |
| gsm8k | 2 | MIX | 12314 | 223.773 | 208.582 | 10.852 | 5.5355 | 6.5159 |
| gsm8k | 2 | ROLE | 12315 | 221.876 | 208.384 | 9.103 | 5.5351 | 6.5154 |
| gsm8k | 4 | KPLUS | 12116 | 220.051 | 205.424 | 10.288 | 5.6422 | 6.6224 |
| gsm8k | 4 | MIX | 12014 | 218.656 | 204.178 | 10.344 | 5.6989 | 6.6786 |
| gsm8k | 4 | ROLE | 12134 | 221.360 | 205.840 | 11.463 | 5.6326 | 6.6126 |
| mathqa | 2 | KPLUS | 17243 | 312.578 | 297.560 | 9.352 | 6.2465 | 7.2322 |
| mathqa | 2 | MIX | 17309 | 314.248 | 297.986 | 10.502 | 6.2187 | 7.2046 |
| mathqa | 2 | ROLE | 17317 | 314.045 | 298.071 | 10.092 | 6.2153 | 7.2013 |
| mathqa | 4 | KPLUS | 17074 | 308.483 | 294.518 | 8.397 | 6.3169 | 7.3038 |
| mathqa | 4 | MIX | 17227 | 311.654 | 296.184 | 9.810 | 6.2516 | 7.2389 |
| mathqa | 4 | ROLE | 17158 | 309.686 | 294.766 | 9.284 | 6.2803 | 7.2680 |

Matched acceptance is the stop-clipped consecutive proposal match count. The separate native counter is retained in the machine-readable tables; it can differ at terminal clipping. Committed progress excludes the initial prefill seed. Native-generation and cache timing are components of complete-request time, not additive extra costs. Call changes across event 2/4 describe learning on the same panel; historical-versus-fresh latency is not used as a learning curve.

## Local explanation and limits

Fixed diagnostic contexts contain 16 requests × 8 anchors per dataset, shared across all checkpoints. ROLE gained 20 matching-prefix occurrences versus K+'s 50 on GSM8K, and 53 versus 82 on MathQA. It therefore showed no superior net-prefix gain in this small diagnostic sample. On MathQA, MIX repaired 36 original first errors versus K+'s 34, but broke 26 old prefixes versus 20 and gained only 29 prefix occurrences versus 82. Counting repaired first errors alone misses the lost progress.

MIX triggered gradient clipping in 128/512 GSM8K windows and 22/512 MathQA windows; KPLUS and ROLE triggered none. The reverse-KL cap affected about 26.6–27.0% of first-error occurrences on GSM8K and 9.3–9.8% on MathQA. MIX prefix cap fractions were about 0.344% and 0.060%, respectively. ROLE's prefix-cap fraction is undefined because that branch is not evaluated. These are observed differences, not proof that clipping caused the workload-dependent result. Cross-objective raw training losses are not quality rankings.

All 512 feedback output/termination identities and all 1,024 presentation/anchor bindings per dataset matched across arms in this run. Nevertheless, feedback was collected independently by each arm; equality of answers does not independently establish equality of all native target-feature tensors. The records do not identify whether the remaining difference is due to role assignment, the mixture's effective gradient scale, numerical conditioning, or their interaction. The pilot does not justify adding a preservation mechanism or tuning coefficients.

## Measured adaptation costs

| Dataset / arm | Training seconds | Extra training vs K+ | Whole update-call seconds | Peak allocated GB |
|---|---:|---:|---:|---:|
| gsm8k/KPLUS | 1367.688 | +0.00% | 1569.307 | 20.674 |
| gsm8k/MIX | 1548.740 | +13.24% | 1756.801 | 20.719 |
| gsm8k/ROLE | 1491.915 | +9.08% | 1696.041 | 20.674 |
| mathqa/KPLUS | 1963.830 | +0.00% | 2218.906 | 20.733 |
| mathqa/MIX | 2229.812 | +13.54% | 2487.517 | 20.816 |
| mathqa/ROLE | 2128.170 | +8.37% | 2386.585 | 20.771 |

Training is nested inside whole update-call time. Whole update calls include preparation, training/export, activation and commit; publication validation/retention are separately recorded. TRAJECTORY_COSTS.csv separates feedback request/native/capture costs, teacher preparation, feature persistence, checkpoint/export, activation, loading and observed trajectory wall time. Feedback capture-observer time is nested inside feedback native-generation time; native generation and capture feature persistence are nested inside the complete feedback-request time. Do not add these overlapping columns. Compiler sub-timers are nested in training and are not an independently measured kernel breakdown. Own-trajectory timings are not equal-compute microbenchmarks or a training-inclusive speedup claim.

## Completion and correctness

- Supervisor: 2026-11-07 22:42:28 to 2026-11-08 05:25:18 UTC: **6 h 42 m 50 s**, exit 0; main campaign 23,635.184 s. Qualification took 377.465 s separately.
- Completed main work: **24 events, 3,072 feedback generations, 6,144 presentations/forward-backwards and 3,072 Adam steps**. No algorithm, panel or endpoint was selected from the intermediate results.
- Completed evaluation: **4,608/4,608 valid executions**, zero invalid, unknown or unattempted. Exact canonical tokens and termination, repeat-consistent counts, and all cap cases pass. One GSM8K and nine MathQA cap-hit requests per method remain included.
- Additional bounded work: eight first-event control generations; zero missing canonical generations; six disposable qualification Adam steps/twelve response forwards-backwards; 224 fixed-context draft response forwards and 32 target-feature preparations. The latter use 32 prefill, 1,838 native target-verification replay and 32 terminal forwards; preparation count is not physical-forward count.
- CPU qualification passed 77 new tests plus 21 distinct retained regressions; the source review additionally reran 36 focused CPU tests. GPU qualification states were discarded and never used as production checkpoints.
- Postflight rechecked all 3,847 sealed source/input identities and pinned runtimes. The raw endpoint audit checked 4,608 executions and 264,251 native rounds; numerical and work audits reconcile the original records. Exact tensor equality is supported by original execution receipts, not a new GPU reload.
- All 39,260 compact evidence files / 3,361,688,538 bytes were copied privately and rehashed with zero differences. Event-2/4 exports and complete event-4 recovery states remain on A800. Original historical evidence is unchanged. Rolling retirement removed validated intermediate feature/recovery copies; retired tensors are not claimed archived.

## Decision

**Recommendation: keep K+ as the current method; this pilot does not support advancing ROLE.** MIX's small GSM8K call improvement does not provide a consistent replacement because it regresses MathQA and its GSM8K latency interval includes zero. If the branch is reopened, it needs a separately agreed question supported by these measurements; no sweep, extra event or mechanism change was launched.

This is one training seed, four early events and reused development panels. It neither establishes the 58-event ranking nor rules out a later benefit. The fixed-context diagnostics establish local prediction changes only. All 11 ARS statistical fallacy checks are documented in NUMERICAL_AUDIT.md. Stage 1 / project None remains unchanged.

## Evidence

- [All-repeat measurements](ALL_REPEATS.csv), [paired request table](PAIRED_REQUESTS.csv), [repeat totals](REPEAT_TOTALS.csv).
- [Training exposure](TRAINING_EXPOSURE.csv), [separated trajectory costs](TRAJECTORY_COSTS.csv), [reviewed summary](REVIEWED_SUMMARY.json).
- [Code/protocol review](CODE_PROTOCOL_AUDIT.md), [raw endpoint audit](ENDPOINT_AUDIT.md), [work audit](WORK_AUDIT.md), [numerical/interpretation audit](NUMERICAL_AUDIT.md).
- [Original generated report](ORIGINAL_GENERATED_REPORT.md) is preserved unchanged; this reviewed report adds explicit costs and a more direct primary conclusion.
- Raw private request/token/trace records and model tensors are excluded from Git. The execution commit above is the scientific code identity; report-only publication does not alter it.
