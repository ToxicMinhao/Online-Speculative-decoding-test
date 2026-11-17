# Bounded prefix allocation: completed C7–C8 development screen

## Material passport

Mode: ARS experiment execution and evidence validation. Status: **completed, valid; fixed candidate closed at this screen**. Scope: `DEVELOPMENT_FEASIBILITY_REUSED_DATA`. Execution source: `c94b1bb9b32dbd9a1f523105cff4713f7b880ec3`; scientific base: `9b2d8d8752c5b48e34a3537743ac349ab159b284`; independent-analysis source: `bef13cb893156a0ec8b468356956185df77b33f3`. [PR #50](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/50). No further model work is authorized by this report.

**C8 modestly improves over native K+, but misses the advancement criterion against the fixed scalar control.** Against N8, it saves 42 calls and reduces request time by +0.686%; against S8, it adds 26 calls and is 0.461% slower by the point estimate. The C/S intervals include zero: this does not demonstrate inferiority or equivalence. The prespecified rule required favorable C/S points on both primary measures, so the fixed candidate does not advance.

## What ran

The only training change is detached allocation within each currently correct prefix. Probabilities come from the existing BF16 log-softmax; full-valid-block scores use smoothing 0.5, FP32 products/reductions and a ±25% relative coefficient bound. Each block retains its original prefix mass, native support and direct first-error rule. The additive correction follows the unchanged eleven-term K+ numerator before its original denominator. Full KL, response averaging, anchors, BF16 path, clipping, AdamW and LR `3e-4` remain native. No extra model forward, projection, learned parameter or direct post-error loss is added. Conserving scalar weight mass does not conserve gradient direction or optimizer movement; this is a constrained heuristic, not exact accepted-length optimization or an unchanged D-PACE reproduction.

Complete K6 (768 Adam steps) → C7 (896) → C7's own 128 event-eight responses/features → C8 (1,024). Each event uses 128 responses, two passes, 256 presentations and 128 Adam steps. C8 retains C7's adapted weights, moments, counters, RNG and candidate contract. Retained N8 is native K+; retained S8 uses a fixed 0.96× LR (`2.88e-4`). Neither control was retrained. Complete source/state/data bindings were authenticated; the completed S8 restart was used.

Fresh final evaluation uses the unchanged 128-request PR49 panel: C/N/S, three rotating repeats, four fixed blocks, 1,152 scored generations and 72 separate warmups. Qwen3-8B/DFlash, BF16 greedy, block size 16, 2,048 tokens or EOS, context limit 4,096, concurrency 1, capture disabled, same physical GPU `GPU-9986735a-3e68-d1bd-5998-9634ef24341e`. The other card was idle at every required quiet boundary. No concurrent training or artifact transfers occurred during timing.

## Validity and qualification

All 1,224 final returns passed exact canonical tokens/termination and repeat-consistent native counts; zero invalid, unknown or unattempted. There were no cap hits or zero-call requests; each arm produced 39,869 output tokens. Independent raw reconstruction passed 634,767 checks. Controls reproduce the retained count totals. All 4,859 compact files (251.76 MB) were transferred and byte-verified; private prompts, token arrays and native runtime records remain in the private delivery.

Qualification used five disposable Adam steps, ten response presentations plus two separately charged startup forwards/backwards, and eight native generations. Native versus zero-cap matched loss, gradients, parameters, Adam state, counters and RNG exactly. Active allocation moved 2,902 coefficient occurrences and changed 39 parameter tensors. Original inputs/support/first-error terms stayed unchanged on that same forward. Candidate save/reload reproduced its next update exactly and origin restoration passed. This proves a functioning intervention, not efficacy. Final CPU execution suites passed 80 integration, 13 objective/property and 54 inherited tests; independent analysis adds 8 primary, 6 training-reader and 6 secondary fixtures.

Administrative issues were repaired before GPU work: missing remote rsync/slow full-source transfer, a potentially nondeterministic diagnostic scatter, and a CPU-preflight validator rejecting explicitly charged earlier staging directories. The last failed CPU preflight took 66.40 s and used no model/CUDA work; the repaired check passed in 59.63 s. Exact source reuse was verified, not assumed. These did not alter loss, tolerances, data or budgets.

## Final endpoint results

Counts use one validated trajectory per request; latency sums each request's three-repeat mean. All ratios are pooled from summed measurements.

| Endpoint | Calls | Native accepted / call | Progress / call | Complete request time (s) | Native generation (s) | Cache maintenance (s) |
|---|---:|---:|---:|---:|---:|---:|
| N8 — native K+ | 5,835 | 5.810797 | 6.810797 | 100.908707 | 98.717836 | 0.051972 |
| C8 — allocation | 5,793 | 5.860176 | 6.860176 | 100.216471 | 98.052551 | 0.051506 |
| S8 — fixed 0.96× LR | 5,767 | 5.891105 | 6.891105 | 99.756854 | 97.596403 | 0.051746 |

| Contrast | Time reduction [95% CI] | Acceptance change [95% CI] | Call reduction [95% CI] | Fewer/more/tied calls | Gross saved/added |
|---|---:|---:|---:|---:|---:|
| C/N | +0.686% [+0.047%, +1.310%] | +0.850% [+0.072%, +1.629%] | +0.720% [+0.061%, +1.370%] | 53/40/35 | 106/64 |
| C/S | -0.461% [-1.067%, +0.126%] | -0.525% [-1.260%, +0.195%] | -0.451% [-1.092%, +0.167%] | 35/51/42 | 60/86 |
| S/N | +1.141% [+0.524%, +1.762%] | +1.382% [+0.608%, +2.164%] | +1.165% [+0.515%, +1.814%] | 62/33/33 | 119/51 |

Positive reduction favors the first arm. Native accepted proposals equal committed progress minus verifier calls; progress excludes the separately generated first token. Identical outputs make calls and acceptance linked views, not independent evidence. Matching-EOS acceptance is retained separately in machine tables. Complete request time contains more than native-generation/cache time; no detailed kernel decomposition is inferred.

| Repeat | N time (s) | C time (s) | S time (s) | C/N reduction | C/S reduction |
|---|---:|---:|---:|---:|---:|
| 1 | 100.832550 | 100.104418 | 99.701886 | +0.722% | -0.404% |
| 2 | 100.832007 | 100.106857 | 99.685331 | +0.719% | -0.423% |
| 3 | 101.061562 | 100.438138 | 99.883345 | +0.617% | -0.555% |

Intervals use 10,000 common whole-request paired draws, PCG64 seed 1708, source-ID order and linear quantiles, keeping all repeats together; zero undefined draws. These are conditional intervals on reused development data, not independent confirmation or training-seed uncertainty.

## Allocation and secondary chronology

| Event | Sampled blocks | Correct-prefix occurrences | Changed blocks | Changed coefficient occurrences | Prefix mass moved (half-L1 share) |
|---|---:|---:|---:|---:|---:|
| C7 | 64,564 | 507,638 | 56,062 | 502,722 | 3.3084% |
| C8 | 60,910 | 487,136 | 53,343 | 482,683 | 3.2745% |

There are zero recorded per-block mass, bound, support or nonfinite violations. Per-length/slot movements are in `PREFIX_LENGTH_SLOT.csv`; overlapping anchors and repeated presentations are occurrences, not independent tokens. Qualification directly verifies unchanged native first-error terms; production summaries are implementation receipts, not an independent counterfactual on every response. No extra source-fitting model replay was run.

The first post-analysis training reader incorrectly compared separately rounded FP32 sums using a cancellation-sensitive criterion: 307 bookkeeping identities failed, with maximum discrepancy 5.05447e-05. An analytic FP32 reduction bound resolves all of them (largest error/bound 0.235913); the corrected reader passes 255,636 checks. Original failure, reader and repair diagnosis are retained. **The original scientific per-block tolerances were not widened and the executed model/data were not changed.**

The already required C7 collection has calls N7/C7/S7 = 4,402/4,407/4,393 on 128 other fixed requests. C/N call reduction is -0.114% [-0.764%, +0.535%]; 36 requests improve, 41 regress and 51 tie, with 54 calls saved and 59 added. There is no demonstrated C7 collection advantage. The 5,177 retained provenance checks, 1,623 new source/chronology checks and 174,950 native three-arm validation checks pass. No collection latency ranking is made. These requests differ from the final panel, so subtracting effects cannot establish temporal improvement or deterioration; this secondary result does not alter the endpoint decision.

## Work and cost

Observed cumulative work: **261 Adam applications, 528 response forwards and 528 backwards, three trainer initializations, 128 feedback generations, 128 teacher preparations, eight qualification generations, 72 warmups and 1,152 scored generations**. No pending or unknown work. Main learning uses 256 Adam steps and 512 forwards/backwards each; four production startup forwards/backwards and 12 qualification forwards/backwards explain the remaining work. Total complete responses = 1,360. This is below the limits of 272 Adam steps and 544 forwards/backwards each and the declared generation limits. Recomputations, projection chunks and physical call receipts are separate in `COSTS.json`.

| Task | Complete task (s) | Training compute (s) | Load (s) | Startup (s) | Save (s) | Export (s) |
|---|---:|---:|---:|---:|---:|---:|
| C7 | 466.248 | 365.546 | 28.939 | 19.537 | 18.060 | 15.311 |
| C8 | 448.459 | 344.179 | 28.993 | 18.739 | 20.253 | 15.142 |

These timers are nested, not additive. The candidate chain used 18.38 minutes including its own feedback. The collection task took 188.327 s, including 87.664 s generation, 10.743 s teacher preparation, 41.373 s verified-duplicate retirement and 20.509 s loading. The final evaluation task took 1,067.909 s, including 21.759 s loading; its evaluation wrapper includes 37.747 s checkpoint activation and 11.498 s weight validation. The qualification training wrapper took 218.302 s; the two independent serving workers took 36.103/35.176 s and are charged separately. Historical N/S training costs are not fresh matched overhead controls; no training-inclusive speedup is established.

From first model-load clock to completion: **40.71 minutes elapsed; 40.99 occupied-card minutes**, below both four-hour ceilings. CPU preflight/staging and 141.76 s compact retrieval are outside that clock and disclosed. The initial estimate was 1–2 hours; measured post-qualification admission estimated 54.70 minutes raw/92.05 minutes conservative remaining. The final evaluation task took 17.80 minutes.

Main training peak allocated/reserved GPU memory: C7 = 20.515/25.812 GB, C8 = 20.535/32.133 GB; sampled process-tree RAM peaks were 28.146/28.574 GB. Planned conservative new peak was 58.337 GB; final retained allocation was 28.189 GB, charged allocation 32.484 GB including a 4 GiB reserved compiler allowance, against an 80 GB cap. Final free space was 632.580 GB, against a 400 GB reserve; compiler-cache growth was zero. This reports admitted peak and measured boundary/final usage, not a continuously sampled disk high-water mark. Exact newly created duplicate retirement receipts are retained; no older project assets were deleted. C7/C8 complete states, exports, own features and compact evidence remain available. Fenrir removed 0.773 GB of verified redundant local source unpackings while retaining their exact archives.

## Prespecified decision and learning record

C/N passes the +0.5% time-point threshold, positive timing/acceptance lower bounds and all-repeat timing condition. **C/S fails the required favorable point estimates on both primary measures.** Decision: `CLOSE_FIXED_CANDIDATE_AT_THIS_SCREEN`. No tuning, endpoint selection, extra generation or automatic confirmation follows.

Supported: this bounded redistribution is an active, valid intervention with a small native-K+ endpoint benefit on this reused panel. Weakened: the argument that this structural extension is preferable here to the tested simpler scalar adjustment. Unresolved: confidence versus position-shape versus update-magnitude causes, learning-seed robustness and generalization. A negative C/S point with overlapping intervals does not prove equivalence or universal inferiority. One fixed scalar control does not exhaust LR alternatives. Native K+ remains primary; S remains a development control, not a validated replacement. Research stage unchanged.

Evidence: [summary](SUMMARY.json), [per-request](PER_REQUEST.csv), [all repeats](ALL_REPEATS.csv), [warmups](WARMUPS.csv), [native trace counts](TRACE_COUNTS.csv), [validity](VALIDITY.json), [independent audit](INDEPENDENT_PRIMARY_AUDIT.json), [final report/privacy review](FINAL_REVIEW.json), [allocation audit](TRAINING_AUDIT.json), [per-slot coefficients](PREFIX_LENGTH_SLOT.csv), [secondary collection](SECONDARY_COLLECTION.json), [configuration/ancestry](CONFIGURATION.json), [costs](COSTS.json), [qualification](QUALIFICATION.json), [analysis code](analysis/README.md). Private raw token/response artifacts are retained outside Git and bound by the verified retrieval manifest.
