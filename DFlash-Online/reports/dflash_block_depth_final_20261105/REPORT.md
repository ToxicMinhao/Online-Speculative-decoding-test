# Frozen K+58 block-depth FINAL

**Decision: retain original FIVE. Do not advance this fitted readout/gate from this screen.** All 384 questions and 2,304 timed completions passed. ADAPTIVE was **0.8084% slower than FIVE**, with a paired 95% slowdown interval of **0.6418–0.9915%**. It beat fixed THREE, but failed the required superiority to both fixed paths and the separate practical target of 2% over FIVE.

Execution: [`0f704167b268672d980fabe704898abf9b134c39`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/0f704167b268672d980fabe704898abf9b134c39), [PR #28](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/28). Successful wrapper: **2026-11-05 13:16:36–14:11:44 UTC, 55 min 8 s**. No tuning or further experiment followed the result.

## Fixed setup

Qwen3-8B target and the authenticated five-layer K+58 DFlash-b16 export remained frozen. Serving used the qualified SGLang/FA3 BF16 runtime, greedy decoding, concurrency 1, 2,048 new tokens or EOS, 4,096-token context and the original cache-flush policy. FIVE used the original path without a gate or layer-3 tap. THREE and ADAPTIVE shallow exits shared the fitted rank-32 residual readout and copied RMSNorm; a shallow exit omitted layers 4–5 while preserving all-layer committed-prefix KV.

The prespecified GSM8K partitions were READOUT 384 train questions, GATE 128 other train questions and FINAL 384 test questions. Selection, rendered inputs and order were sealed before fitting. All FINAL questions had prior project exposure; they were disjoint from this fit, not globally new. Dataset solutions were unused. One FINAL question hit the valid output cap and remained in all six executions.

Readout fitting used exactly three groups of 128 responses, two shuffled passes per group, seed 17, microbatch 1 and accumulation 2: **768 presentations, 384 auxiliary AdamW steps, zero backbone steps**. Only the norm and two rank-32 factors trained. Native K+ KL, current shallow predictions, decay, response normalization, BF16 optimizer/probability arithmetic and FlexAttention were preserved; LR 3e-4, clip 1, weight decay 0.01, at most 64 anchors per presentation. The readout was frozen before GATE. The fixed weighted ridge gate and its threshold were frozen before FINAL.

## FINAL results

Time is the **sum of per-question means over two sweeps**, not the wall time of the entire campaign. Counts use one trajectory per question after repeat consistency checks. The request timer includes native generation, cache maintenance and required RPC overhead.

| Path | Request time (s) | Verification calls | Committed progress/call | Native accepted counter/call | Shallow rounds |
|---|---:|---:|---:|---:|---:|
| Original FIVE | 265.6633 | 15,948 | 7.3947 | 6.3947 | 0 / 15,948 |
| Fitted THREE | 606.5688 | 38,448 | 3.0673 | 2.0673 | 38,448 / 38,448 |
| ADAPTIVE | 267.8109 | 16,031 | 7.3564 | 6.3564 | 335 / 16,031 (**2.0897%**) |

| Comparison | Request-time reduction | Paired 95% interval | Verification-call reduction |
|---|---:|---:|---:|
| ADAPTIVE vs FIVE | **−0.8084%** | **[−0.9915%, −0.6418%]** | −0.5204% (83 more calls) |
| ADAPTIVE vs THREE | +55.8482% | [+54.8672%, +57.0551%] | +58.3047% |
| THREE vs FIVE | −128.3224% | [−134.7445%, −123.3720%] | −141.0835% |

Positive reduction favors the first named path. Both timings for each question travel together with all three arms in 10,000 paired question bootstrap draws, seed 20261105. ADAPTIVE had lower mean request time on 29 questions and higher time on 355 versus FIVE; it was faster than THREE on all 384. For verification calls versus FIVE, 17 requests improved, 64 regressed and 303 tied: 20 calls saved and 103 added. Both repeats, every regression and every request remain in the tables.

| Path | Sweep 1 request seconds | Sweep 2 request seconds |
|---|---:|---:|
| FIVE | 265.9074 | 265.4191 |
| THREE | 607.0840 | 606.0537 |
| ADAPTIVE | 268.0338 | 267.5879 |

All paths produced the same 118,315 output tokens over the 384 requests. Native accepted counters use committed progress minus one per verification; they are not a separately reconstructed count of all logically accepted proposals at a clipped EOS. The native generation and cache-maintenance components, the host decode-round subcounter, raw repeats and per-request measurements are preserved. The host subcounter adds no synchronization and may omit trailing GPU completion; **complete-request time is primary**.

## What this establishes

The fitted THREE path produced much less committed progress per verification than FIVE and required 22,500 additional calls. The adaptive rule mostly selected FIVE, avoiding most of fixed THREE's slowdown, but its rare shallow exits did not yield an endpoint benefit over original FIVE. The specified integration therefore fails its feasibility screen.

This does not identify whether more shallow capacity, another fitting objective/budget or another selection rule would succeed. It does not prove all early-exit methods fail, establish an online gate-refresh benefit, or isolate an exact kernel-level cause from aggregate latency. None of those alternatives was tested or automatically launched.

The earlier optimistic pricing result of 2.6389% cleared a **budget gate**; it was never a measured deployment speedup. The final scientific criterion required both ADAPTIVE/static paired confidence lower limits above zero. That criterion and the practical 2% target over FIVE both failed.

## Calibration and validity evidence

- All **2,304 / 2,304** executions are valid: zero invalid, unknown, unattempted or excess completions; all 384 paired questions complete. Token IDs, termination, per-arm repeat counts and sealed balanced order pass.
- GATE used 128 alternating FIVE/THREE behaviors and **5,452** deterministic matched starts, with **10,904** paired proposal replays. All behavior-proposal/progress and cache-restoration checks passed. Signed labels ranged from −6 to +15; all 221 negative labels were retained.
- The fixed continuing price was **0.517600885 ms** from the prior matched-route cohort. κ was **1.882085087 ms/token**, from 31.018644 s of synchronized native decode rounds and 16,481 committed tokens on the 64 FIVE calibration behaviors. The deployed rule continued when κ × predicted progress gain met the continuing price.
- All four route graphs were built before FINAL: eight component warmup executions and four captures, counted separately from generations. Timing had no feedback capture, detailed traces, shadow proposals, concurrent training or artifact transfer. Fitted tensors kept their graph-bound storage.
- All three readout boundaries passed decoded sidecar/export, Adam, counters and RNG checks; both resumptions were exact. Every step had finite gradients/parameters and nonzero BF16 updates. All original native draft weights and source/model/runtime identities remained exact.
- **119 CPU tests passed.** A separate same-team CPU audit independently recomputed the results and confidence intervals: **107,718 checks, zero errors**, reading 4,909 evidence files. A further readout/cost/retention review passed 7,715 checks with zero discrepancies. Authorship/shared-context overlap is disclosed; these are not independent GPU replications. The audit initially selected native sidecar JSONs as execution rows; its file-selection bug was repaired, eight audit tests passed, and the original audit failure was retained. No production measurements changed.

Training uses SpecForge's residual-complete h3 and ordinary RMSNorm; serving uses the qualified fused residual RMSNorm. Their mathematical input is aligned, but bitwise training/serving normalization equivalence is not claimed. Native THREE and ADAPTIVE shallow inference use the exact same implementation.

## Work, costs and retention

The run collected 368 new READOUT responses and reused the earlier 16 exact saved responses, then collected 128 GATE responses. Reconstruction used 16 target prefills and 530 target verification forwards, checking all 8,480 retained target predictions. All 384 responses received native teacher preparation. Canonical references reused 306 authenticated answers and separately generated the remaining 78; these never trained or calibrated the models. New complete-response generations totaled **2,878**, including the **2,304** timed FINAL executions. Earlier qualification's 16 additional completions and 2,048 route prices remain separate.

| Stage | Seconds | Scope |
|---|---:|---|
| Native loading | 19.757 | Includes normal native initialization; no extra generation warmup |
| READOUT collection/reconstruction/preparation | 287.639 | Three groups; includes persistence and redundant-feature retirement |
| Readout worker lifetimes | 334.486 | Three isolated workers; initialization and fitting included |
| Native readout training timers | 252.139 | Subset of worker lifetimes; not additive |
| GATE collection, matched replay and fit | 219.436 | Includes first route-graph construction |
| Missing canonical references | 55.270 | 78 separately counted, untimed references |
| FINAL stage wall time | 2,304.147 | All 2,304 timed completions plus orchestration |
| Campaign function wall time | 3,267.322 | Includes admission and orchestration; overlapping subtimers above |
| Successful wrapper wall time | 3,308 | CPU checks, admission, run, postflight and report |

Peak readout allocation was 6.204 GB; no full backbone checkpoint was created. Detailed preparation, exposure, training, persistence and retirement accounting accompanies this report. Decreasing pass-mean training losses use changing anchors and current prediction masks; they are descriptive traces, not a fixed-anchor source-fit proof. Upfront fitting/calibration cost is not hidden in serving latency; no training-inclusive speedup is claimed. Training sampled 49,152 anchors and evaluated 714,583 valid proposal occurrences, with 169,351 selected direct-KL occurrences. Among 48,947 first errors, 21,763 received priority 2 because the immediate valid canonical successor matched, and 27,184 received priority 1; 205 blocks had no error. The detached successor cue changes a weight, not the number of supervised positions, so it is not added to direct-KL exposure. Priority 1 includes both mismatching and unavailable successors; their separate denominator is not retained.

The first full attempt stopped after one successful target prefill because the reconstruction request omitted the native vocabulary-size field. The repair also restored normal empty stop-string initialization. There were **zero complete generations and zero optimizer steps** in that attempt. Its code, logs and failure receipt remain preserved; the repaired run used a separate directory. This extra prefill is accounted separately. An optional real-runtime CPU-only request import was unavailable because the pinned GPU library cannot load with GPUs hidden; this was not recorded as a passed test. The actual repaired GPU reconstruction subsequently passed all native checks.

The prelaunch peak-storage budget was 36.248 GB including an 8 GiB reserve; The successful-run admission measured 61.683 GB free. Consumed per-group teacher tensors retired only after successor-state validation, with exact deletion receipts and compact token/anchor/validity evidence retained. The receipts identify **1,152 retired paths / 12,606,855,808 bytes**, including redundant capture/terminal copies and consumed prepared pools. Deleted tensors are **retired, not archived**; direct replay from those deleted tensors is lost. No old experiment was deleted to run this screen. Postrun A800 free space was 61.315 GB after evidence packaging.

All **8,452 files / 214,461,182 bytes** of retained full-run evidence were copied to Fenrir and verified against their inventory. The archive SHA-256 is `186f15c796ecdc49836b98c255e455476140df31a2976ca888eb0c0b6302b3db`. Both GPUs are idle. The research stage remains unchanged; this work stops at the report.

## Deliverables

- `PER_REQUEST.csv`, `ALL_REPEATS.csv`, `SUMMARY.json`: complete endpoint measurements and analysis.
- `PLAN.json`, `READOUT_CONFIG.yaml`, `MODEL_PIN.json`, `RUNTIME_PIN.json`, `FINAL_FREEZE.json`, `FITTED_GATE.json`: scientific/configuration/artifact identities.
- `MANIFEST_IDS.csv`, `MANIFEST_SUMMARY.json`: selected source identities, order, prompt-token digests and exposure disclosure. Exact private request tokens remain in the original sealed manifest.
- `AUDIT.md`, `RETRIEVAL_VERIFIED.json`, supplementary cost/exposure/retention review: verification evidence.
- Private raw records, canonical references, GATE proposals/features, readout recovery states and sidecars: `/home/minhao2/EAGLE_SPEC/research/dflash_block_depth_20261105/full_retrieved_002/full_production_002/` and the original A800 run directory. Private prompts/responses and raw runtime artifacts are not uploaded to Git.
