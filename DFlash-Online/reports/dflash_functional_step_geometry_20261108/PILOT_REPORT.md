# Final FSG R1 pilot: events 5–6 on GSM8K and MathQA

**The completed pilot is a valid negative for the tested FSG mechanism.** FSG6 requires more verification calls and more complete-request time than both KPLUS6 and SCALAR6 on both panels. All four primary paired latency intervals exclude zero in the unfavorable direction. SCALAR's small differences from KPLUS remain unresolved; this is not equivalence or evidence that scalar shrinkage improves K+.

FSG was applied in **512/512 production windows**. It reduced its protected-distribution KL while keeping the realized update norm almost unchanged, but generally reduced useful repair/progress in the fixed training diagnostics. The outcome is consistent with suppressing valuable learning through harmful local redirection. The records do not identify a unique cause or prove that every functional-geometry method would fail.

**Recommendation under the approved handoff: retain ordinary K+ and stop this FSG version after the bounded pilot.** Do not extend it to 58 events or select a new weighting/sketch variant from these diagnostics. No further experiment or tuning has been launched.

## Completed experiment and validity

Scientific execution: `6440e8ff5af1a845f830d65c884f6711d7e4a687`, binding `8ec5fed0d4ed9b40dab7c26d5089cd42b01f357ab0379193da42baa93f0b3dc5`; A800 run `dflash_functional_step_geometry_20261108/pilot_r1_001`. The repaired launch ran from **2026-11-09 05:02:43 to 23:02:04 UTC**: **17 h 59 min 21 s**, finishing **November 9, 4:02 p.m. America/Cambridge_Bay**. Controller time was 64,567.09 s; these are different envelopes. Exit code 0, all 18 tasks completed, none unattempted, and the final 15,122-file integrity check passed. Both GPUs were idle when results were checked.

Each dataset's three arms restored its own original complete KPLUS4 state, then ran events 5–6 from 512 to 768 cumulative Adam steps. Shared feedback came from evolving KPLUS, with identical prepared features, presentation orders and native anchors across arms. This is a controlled shared-feedback continuation, not three independent online deployments. K+'s loss/optimizer and the fixed R1 geometry, BF16, five-layer draft, greedy block-16 decoding, 2,048-token/EOS cap, 4,096 context, concurrency-one and cache policies remained unchanged.

All **512 fixed development requests / 3,072 endpoint executions** passed raw canonical token/termination, checkpoint, schedule, native trace, cap and repeat checks. No invalid, unknown or unattempted endpoint work remains. Twelve exact continuation/export receipts and all 1,024 filtered-window decisions also validate. Old qualification failures remain disclosed; the approved replacement checks and R1 supplementary qualification are separately documented in the [qualification report](BF16_REALIZATION_R1_REPORT.md). Disposable qualification/startup work did not initialize these trajectories. This CPU review did not reload models or independently regenerate predictions.

## Endpoint serving performance

Times are sums of the two timing sweeps averaged **within each request**. Verification calls and accepted/progress counters use one validated deterministic trajectory per method/request.

| Dataset | Method | Calls | Total request time, s | Mean/request, s | Matched accepted/call | Committed progress/call |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | KPLUS6 | 12,018 | 205.469 | 0.802612 | 5.6962 | 6.6764 |
| GSM8K | FSG6 | 12,752 | 218.321 | 0.852817 | 5.3107 | 6.2921 |
| GSM8K | SCALAR6 | 11,975 | 205.037 | 0.800928 | 5.7201 | 6.7004 |
| MathQA | KPLUS6 | 16,782 | 292.831 | 1.143871 | 6.4432 | 7.4309 |
| MathQA | FSG6 | 17,667 | 307.872 | 1.202623 | 6.0709 | 7.0586 |
| MathQA | SCALAR6 | 16,776 | 292.951 | 1.144340 | 6.4465 | 7.4335 |

Positive **increase** below means worse performance; interval bounds are expressed as percentages of the reference metric. Complete reduction-sign results (`100*(1−candidate/reference)`) and correctly sign-reversed intervals are retained in the linked comparison tables.

| Dataset / candidate versus reference | Calls increase %, 95% interval | Request-time increase %, 95% interval | Call wins / regressions / ties | Time wins / regressions / ties |
|---|---|---|---|---|
| GSM8K: FSG / KPLUS | +6.11 [+4.38, +8.53] | +6.26 [+4.44, +8.84] | 51/175/30 | 65/191/0 |
| GSM8K: FSG / SCALAR | +6.49 [+4.78, +8.92] | +6.48 [+4.68, +9.09] | 39/187/30 | 53/203/0 |
| GSM8K: SCALAR / KPLUS | -0.36 [-0.81, +0.08] | -0.21 [-0.69, +0.28] | 105/88/63 | 136/120/0 |
| MathQA: FSG / KPLUS | +5.27 [+4.61, +5.95] | +5.14 [+4.49, +5.81] | 33/206/17 | 42/214/0 |
| MathQA: FSG / SCALAR | +5.31 [+4.59, +6.04] | +5.09 [+4.39, +5.80] | 36/200/20 | 50/206/0 |
| MathQA: SCALAR / KPLUS | -0.04 [-0.37, +0.29] | +0.04 [-0.30, +0.38] | 90/94/72 | 125/131/0 |

FSG saves/adds **106/840 calls on GSM8K** and **66/951 on MathQA** versus KPLUS. The unfavorable effect is distributed across 175/256 and 206/256 requests, respectively; it is not claimed to affect every request. All methods produce the same **80,493 GSM8K output tokens (one capped request)** and **124,961 MathQA tokens (nine capped requests)**. Committed progress is 80,237 and 124,705, excluding each request's initial seed. Matched accepted proposals and the runtime's `progress−1` accepted counter remain separate, including stopping-boundary differences. This is a drafting-efficiency regression, not degraded target answer quality.

The unchanged prespecified analysis uses 10,000 paired-request bootstrap samples, PCG64 seed 20261108, linear percentiles and shared request indices across contrasts. A separately coded arithmetic review reproduces the saved points and intervals. Repeats are timing measurements, not additional requests. Intervals are unadjusted and conditional on one continuing training seed and these reused panels; no familywise, training-seed-robustness or population-generalization claim is made.

The complete-request stopwatch includes generation, cache maintenance and its existing wrapper, excluding checkpoint loading/activation. Native/cache components are already inside this time and do not constitute a full kernel decomposition. Both sweep totals and every request/repeat are retained: [GSM8K endpoint](pilot_final/endpoint_gsm8k/SUMMARY.json), [MathQA endpoint](pilot_final/endpoint_mathqa/SUMMARY.json), [table index](pilot_final/README.md).

Authenticated historical KPLUS4 records match the exact panels, canonical outputs/termination, runtime and origin exports. Calls were **12,116 GSM8K / 17,074 MathQA**. Relative to that origin, FSG6 uses **5.25% / 3.47% more calls**, whereas KPLUS6 uses **0.81% / 1.71% fewer**. FSG therefore deteriorates from the common origin, not merely improves less. Historical latency is not compared.

## Effective corrections and numerical behavior

| Dataset / arm | Windows | Genuine corrections | Native fallbacks | Numerical ambiguity | Measured KL increase | Feasibility searches |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K FSG | 256 | 256 | 0 | 0 | 0 | 254 |
| GSM8K SCALAR | 256 | 34 | 222 | 188 | 34 | 252 |
| MathQA FSG | 256 | 256 | 0 | 0 | 0 | 255 |
| MathQA SCALAR | 256 | 40 | 216 | 182 | 34 | 254 |

A genuine correction differs from both the pre-step and native Adam endpoint; an accepted flag alone is insufficient. All 1,024 windows had nonzero native displacement, protected positions, a nonempty sketch and an actual candidate replay. No candidate KL is replaced with an imputed zero. All recorded strict norm/KL decisions independently reproduce.

FSG's median committed/native norm ratios are **0.99999999958 GSM8K / 0.99999999957 MathQA**; median direction cosines are **0.999881 / 0.999887**. Median protected candidate/native KL ratios are **0.711684 / 0.711372**. Thus, inactivity and broad realized-norm shrinkage do not explain the regression. Lower protected KL is partly imposed by acceptance and does not establish useful preservation. These are descriptions of divergent trajectories, not same-state causal contrasts.

SCALAR performs its own hypothetical geometry and norm matching in every window, without borrowing FSG's state or scales. Maximum relative target/realized norm mismatch is **3.70×10⁻⁷ GSM8K / 1.77×10⁻⁸ MathQA**. It commits only 74/512 candidates; most windows correctly retain native. Its endpoint comparison therefore does not establish how a substantially smaller-step recipe would behave. Continuous/raw/feasible/committed norms, per-response risks, margin-floor/tie weights, numerical sketches and scan costs are retained in [production tables](pilot_final/README.md#production-and-costs). Optional ordinary-gradient directional products were not recorded and are not inferred from auxiliary products.

## All sixteen fixed diagnostic windows

Events 5/6 × windows 1/32/65/128 on both datasets are complete, with exact recorded restoration. **A** is the fitting pair; **D** is a disjoint pair of already buffered training responses, not held-out evaluation. Every raw/feasible/committed/scalar/uniform/expanded candidate is retained in the 320 window/pair/candidate rows, with repair identities, role-specific teacher KL, protected KL, norms, guard outcomes and denominators.

Below, each entry is native → committed FSG, summed over four fixed windows. Counts are overlapping sampled-block occurrences, not independent tokens or requests.

| Dataset/event/role | Blocks / first-error blocks / old-prefix occurrences | Useful repairs | Old-prefix occurrences lost | New earlier errors | Net prefix gain |
|---|---|---|---|---|---|
| GSM8K / 5 / A | 2,329 / 2,013 / 15,496 | 415 → 381 | 64 → 58 | 16 → 16 | +1,154 → +1,016 |
| GSM8K / 5 / D | 1,854 / 1,359 / 15,158 | 73 → 39 | 147 → 166 | 45 → 50 | +44 → -59 |
| GSM8K / 6 / A | 1,945 / 1,475 / 15,661 | 357 → 355 | 22 → 24 | 9 → 11 | +1,194 → +1,174 |
| GSM8K / 6 / D | 1,873 / 1,406 / 15,004 | 44 → 40 | 164 → 162 | 54 → 56 | -10 → -33 |
| MathQA / 5 / A | 3,309 / 2,606 / 25,244 | 459 → 456 | 102 → 100 | 29 → 29 | +1,369 → +1,322 |
| MathQA / 5 / D | 3,011 / 2,142 / 26,702 | 118 → 81 | 201 → 242 | 57 → 72 | +123 → +0 |
| MathQA / 6 / A | 2,675 / 1,929 / 21,961 | 434 → 415 | 77 → 66 | 22 → 24 | +1,338 → +1,243 |
| MathQA / 6 / D | 3,679 / 2,651 / 31,737 | 79 → 78 | 216 → 218 | 72 → 87 | -28 → -37 |

FSG lowers net prefix progress versus the same-state native update in **7/8 A windows on each dataset**, and **6/8 GSM8K D / 5/8 MathQA D windows**. Its same-state scalar counterfactual closely matches the realized norm (maximum relative mismatch 2.43×10⁻¹² GSM8K / 1.77×10⁻¹² MathQA) while largely retaining native progress. This supports harmful redirection locally, beyond the measured norm difference. These scalar scores are **pre-guard counterfactuals**, not the independently trained SCALAR arm; candidate guard outcomes are disclosed separately.

Useful-prefix preservation is inconsistent, so the stronger observation is **loss of useful repair/progress despite lower distribution drift**. FSG usually damages A progress too; a clean “better fitting but worse transfer” account is not supported. BF16 feasibility changes local outcomes, with mixed effects; raw FSG already underperforms native in aggregate, so realization alone is not established as the cause. Uniform weights and the expanded sketch produce mixed local gains, not a selected successor. One favorable exception is retained: uniform weighting on MathQA event-6 D reaches net zero versus native −28, but repairs fall from 79 to 72 and A remains below native. All other windows and unfavorable outcomes remain included. See the [full diagnostic review](pilot_final/diagnostics/FINDINGS.md).

These sparse same-state comparisons cannot allocate a percentage of endpoint time/call regression to a cause, or isolate weighting, sketch rank, finite-step effects and BF16 rounding. A mislabeled historical free-text `raw_fsg` description is documented in the interim review; its executable candidate identity and measurements are correct and remain unchanged.

## Cost and complete work accounting

| Dataset / arm | Intrinsic training, s | Relative to KPLUS | Separate diagnostic observer, s |
|---|---:|---:|---:|
| GSM8K / KPLUS | 635.52 | 1.00× | 0.00 |
| GSM8K / FSG | 9,933.52 | 15.63× | 1,182.47 |
| GSM8K / SCALAR | 13,661.10 | 21.50× | 0.00 |
| MathQA / KPLUS | 1,027.01 | 1.00× | 0.00 |
| MathQA / FSG | 11,216.98 | 10.92× | 1,353.06 |
| MathQA / SCALAR | 14,851.92 | 14.46× | 0.00 |

Intrinsic training subtracts the explicitly measured diagnostic observer from the training envelope. It still includes the production geometry, parameter scans, matching and bookkeeping. Nested probe/realization/hook timers are **not added** to training. FSG provides no serving gain to offset this additional cost. These measured implementation costs do not establish a fundamental complexity lower bound or an equal-compute comparison. Peak event allocation was at most **35.60 GB FSG**, **27.99 GB SCALAR** and **20.81 GB KPLUS**; measured memory can include diagnostics/publication activity and is not an isolated algorithm-only peak. Loading, startup, teacher preparation, shared-feedback capture, checkpoint persistence/export and activation are separately tabulated.

| Work | Completed | Failed / unknown |
|---|---:|---:|
| Shared feedback | 512: 508 new + 4 retained qualification packets | 0 / 0 |
| New teacher preparations | 508 | 0 / 0 |
| Main ordinary presentations / forwards / backwards | 3,072 / 3,072 / 3,072 | 0 / 0 |
| Main native Adam steps | 1,536 | 0 / 0 |
| Additional main forwards / backwards | 6,144 / 2,048 | 0 / 0 |
| Fixed diagnostics | 16 windows; 640 forwards / 64 backwards; zero Adam | 0 / 0 |
| Disposable startup | 24 forwards / 24 backwards; zero Adam | 0 / 0 |
| Endpoint complete-response generations | 3,072 | 0 / 0 |

Total main response forwards/backwards are **9,216 / 5,120**, separate from startup, diagnostics, physical head chunks/recomputation and feedback/teacher work. All six trajectories finish at **768 cumulative Adam steps** with no seed reset. Shared-feature identities and native schedules agree across arms. Exact audit records and physical counters are available in the [work and state audit](pilot_final/production/work/WORK_SUMMARY.json). The initial preparation-only transfer-conflict failure and all historical qualification/debugging costs remain separate; the completed main run did not retry scientific cases.

## Reviewed conclusion and delivery

The original automated `recommend()` returns its generic `inconclusive_or_mixed_pilot` fallback because it has no dedicated negative-result branch; its [unchanged output](pilot_final/AUTOMATED_RECOMMENDATION.json) is preserved. The reviewed conclusion applies the handoff's explicit **active corrections plus clear serving regression → valid negative; retain K+** criterion. It does not confuse unresolved SCALAR/KPLUS differences with uncertainty about the demonstrated FSG regressions.

The [evidence index](pilot_final/README.md) links all per-request/repeat, production, diagnostic, cost and provenance tables. The [input/source manifest](pilot_final/INPUT_SOURCE_MANIFEST.json) binds both stable snapshots, all six final checkpoint identities and the execution commit. Raw private prompts/token traces, features and recovery tensors remain outside Git; their retention/retirement limits are disclosed in the work review. Prior GSM8K interim and qualification reports are preserved.

ARS experiment-agent status: **ANALYZED**, with independently coded CPU arithmetic and same-team source/work/numerical review; implementation-author overlap is disclosed. All 11 statistical-fallacy checks were covered. Neither repeated measurements nor overlapping anchor occurrences are treated as independent training seeds. No new model loading, teacher work, feedback, optimization, model forward/backward or generation was performed for this final review. The completed pilot remains stopped; this report authorizes no tuning or further campaign.
