# Frozen K+58 block-depth: native qualification and skip pricing

**Native qualification passed. A real three-layer exit is implemented, and the prespecified same-progress pricing screen clears its early 2% budget threshold.** No shallow-readout or gate fitting and no FINAL evaluation occurred. This result justifies considering the next fitting stage; it does not demonstrate an adaptive serving speedup.

Execution: [`9150cc3b20714153b8e4fb3d70e8a32861c5921f`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/9150cc3b20714153b8e4fb3d70e8a32861c5921f) in [PR #28](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/28). Raw private evidence: `/data/sources/minhao/runs/dflash_block_depth_20261105/qualification_002`; verified Fenrir copy: `/home/minhao2/EAGLE_SPEC/research/dflash_block_depth_20261105/retrieved_002/qualification_002`.

## Result and interpretation

| Measured route | Mean matched-start cost, ms |
|---|---:|
| Original FIVE, no gate | 2.504396 |
| Static THREE, no gate | 1.928122 |
| ADAPTIVE forced-three | 2.043350 |
| ADAPTIVE forced-five | 2.560951 |

Means cover 128 matched native starts, with four balanced repeats per route. **Continuing costs 0.517601 ms**, measured as ADAPTIVE forced-five minus ADAPTIVE forced-three. Gate/segmented-dispatch surcharge versus the corresponding static path is 0.115228 ms for shallow and 0.056556 ms for deep. These are route-cost differences, not complete-request speedups.

The optimistic rule chooses the cheaper adaptive action at each start and permits zero savings when neither beats original FIVE. Expanding sampled starts to each prompt's observed round count gives **245.297 ms** potential savings over **9,295.292 ms** of original complete-request latency: **2.6389%, paired prompt-bootstrap 95% interval [2.5450%, 2.7035%]** (10,000 draws, seed 20261105). The cohort has 530 original rounds. A 2% target requires **0.350766 ms per existing round**, or **0.701531 ms per early exit if half exit**, assuming unchanged round counts.

The upper limit is above 2%, so the early stop does not fire. This is an optimistic same-progress budget calculation. The untrained shallow path can change acceptance and round counts; this calculation does not establish realized savings or bound outcomes when proposal quality improves. A fitted ADAPTIVE endpoint would still need to beat **both** fixed paths, with both paired 95% lower bounds above zero. Report 2% over FIVE separately as the practical target.

## Native correctness

- All **32/32** complete requests valid: 16 original FIVE, four forced-five, four static THREE and eight alternating 3→5→3 qualifications. Zero invalid, unknown or unattempted generations.
- Forced-five reproduces the original proposals, calls, final token IDs and termination on its four complete requests. At all **1,007 qualification live starts**, shadow original-five and forced-five reproduce all 15 proposals; the shadow comparisons include starts arising under shallow/mixed histories.
- THREE and ALTERNATE preserve canonical final tokens and termination; different proposals and round counts are permitted. Eight alternating traces execute the exact 3,5,3,5 sequence.
- All **2,048/2,048** pricing measurements valid, including all-15 original/forced-five reconstruction gates and exact restoration of request mapping, allocator, graph buffers, attention metadata and overwritten KV bytes.
- Captured layer invocations are `[1,1,1,0,0]` for static THREE and the shared prefix, `[0,0,0,0,0]` for shallow finish, and `[0,0,0,1,1]` for deep continuation. Deep work is actually skipped. Qualification records count **3,021 vocabulary projections = 1,007 live + 2,014 shadow**; one live projection per round.
- The original FIVE model/forward/head remain unwrapped during baseline timing. All five committed-prefix KV layers remain managed by the unchanged native worker. Canonical outputs come from fresh original FIVE through the qualified target verifier, not an additional target-only generation. Frozen model weights pass exact before/after checks; source/model/runtime pins pass preflight and postflight.

All 64 focused CPU tests pass locally and on A800. Separate CPU reconstruction passes **11,955 checks with zero discrepancies**, including graph/layer/head counts and bootstrap arithmetic. The auditing agent also authored some data/pricing/qualification code: this is same-team review with separate calculations, not blinded review or independent GPU replication.

## Timing and evidence boundary

Route clocks run from native graph-input/attention preparation through dispatch, any gate CPU synchronization, the chosen suffix and one vocabulary projection ready for verification. Original FIVE includes no gate or shallow tap. The common committed-prefix KV append and temporary allocation occur before this measured boundary; their unchanged cost cancels in route differences and remains included in the original complete-request denominator. Offline KV cloning/restoration is separately charged diagnostic work, not deployment cost. No detailed kernel-time attribution is inferred.

Pricing covers the first 16 sealed READOUT requests and 128 starts, all below 512 prefix tokens (96 starts below 256; 32 in [256,512)). It does not establish skip economics at every possible 4,096-token context. Per-prompt, per-start, context-binned and all-repeat tables retain the observed variation. Qualification timing includes parity shadows and is not a deployment comparison.

## Setup and work

Frozen Qwen3-8B target plus authenticated five-layer K+58 b16; original BF16/greedy/FA3/C1 runtime, 2,048 output tokens or EOS, 4,096 serving context and original flush policy. The sidecar norm is copied from the frozen final norm, rank-32 A is initialized with seed 17 and B is zero. Gate shapes are real (97 features); coefficients are initialized, not fitted. Existing checkpoint weights never change.

Before any generation, the study sealed 384 train READOUT, 128 train GATE and 384 test FINAL requests, disjoint by identity/normalized text/tokens using the specified hash ordering. No dataset solution text or model outcomes were used for selection. All 384 FINAL questions have historical project exposure; FINAL is fit-disjoint, not globally new. No FINAL answers were opened or generated in this milestone. Private manifests and pins remain retained outside Git.

| Work/cost | Observed |
|---|---:|
| Complete generations | 32 |
| Timed single-round route measurements | 2,048 |
| Untimed route warmups | 512 |
| Live qualification shadow proposal forwards | 2,014 |
| New segment-graph setup | 4 graphs; 2 warmups + 1 capture each |
| Optimizer steps / fitting / FINAL generations | 0 / 0 / 0 |
| Successful wrapper wall time | 97 s |
| Preflight | 19.515 s |
| Native loading | 19.756 s |
| Original FIVE complete requests | 9.295 s |
| Qualification requests, including shadows | 23.029 s |
| Entire matched-start profiling, nested warmups/timing/restoration | 13.431 s |
| Postflight | 19.308 s |

These time categories overlap; do not add them as independent costs. Native engine initialization graph work is retained in logs separately from the four new segment graphs. Exact total internal startup-forward counts were not instrumented. Storage admission required 14 GiB including reserve; 61.830 GB was free before launch and 61.836 GB after. No cleanup, model offload, training or concurrent GPU job was needed. Both GPUs are idle after the bounded run.

## Preserved startup failure and repair

Attempt 001 (`4fe52546`) failed during target graph initialization because the new launcher omitted the qualified C/C++ compiler environment. Its log reports `c++: not found`. It completed **zero requests and zero route timings**, with all 32 requests unattempted. Internal partial initialization work remains in the failure log; it is not counted as zero GPU work. Attempt 002 restores the same explicit compiler paths used by the qualified K+ launcher, and pins the launcher in the execution source. No model, algorithm, precision or tolerance changed. Original failed records remain intact.

## Deliverables and stopping point

`GENERATIONS.csv`, `ROUTE_REPEATS.csv`, `PER_PROMPT.csv`, `PER_START.csv`, `CONTEXT_BINS.csv`, `PRICING.json`, `WORK_ACCOUNTING.json`, `COST_ACCOUNTING.json`, `PROVENANCE.json`, admission receipts and the independent arithmetic audit accompany this report. Raw token/proposal/cache-parity records remain private in the retained paths above. The archive SHA-256 is recorded in provenance and matched after retrieval.

**Stopped after native qualification and pricing.** No readout fit, gate fit, FINAL campaign, checkpoint update, tuning or automatic next stage was performed.

## Material passport

ARS experiment-agent run/validate; study `vfd_depth35_gsm8k_v1`; source-bound native implementation qualification and prespecified pricing analysis; one checkpoint/cohort/device run; same-team review; no research-stage advancement. Stage 1 / project None unchanged.
