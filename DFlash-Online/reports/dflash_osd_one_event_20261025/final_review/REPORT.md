# One-event ordinary OSD baseline — completed

## Material Passport

ARS experiment-agent, run and validate modes. October 25, 2026. Status: **completed and raw evidence audited**, not a separately rerun reproducibility study. User-authorized baseline-only test; Stage 1 / project None unchanged. Independent arithmetic/trajectory audit and all 11 statistical-fallacy categories checked. No ranking algorithm or additional event implemented or executed.

## Answer

**One ordinary OSD update modestly improves subsequent drafting on this 64-request GSM8K development panel.** Verification calls decrease **2.07%** and request time decreases **2.23%**, with identical output tokens. The benefit is uneven:40 requests use fewer verification calls, 19 use more, 5 tie.

The update costs **363.63s**, while one pass through the 64-request panel saves **1.21s** of serving time. These observations establish a small benefit once the checkpoint is available; they do **not** establish a net runtime benefit after charging this update over this evaluation window. They do not identify a new algorithm or prove that a proposed ranking objective will help.

## Exact experiment and implementation

- **F:** released Qwen3-8B-DFlash-b16 with the pinned frozen Qwen3-8B target.
- **O:** released draft, fresh qualified BF16 AdamW optimizer, seed 17. One event after 128 fresh F responses; two shuffled passes, 256 response presentations, 128 optimizer steps. One post-training checkpoint publication.
- **Data:** the first 128 chronological GSM8K training requests from the existing qualified manifest, with newly generated F answers. Evaluation uses the exact existing64-request development panel selected previously with seed 20261024. No prompt-ID overlap; evaluation feedback never enters training. These are reused development requests, not a newly sealed unseen test set.
- **Learning:** unchanged forward KL, temperature 1, no hard-label mixture; all 58 draft tensors train, target-owned embedding/projection remain fixed. Constant LR3e-4, AdamW betas(.9,.999), epsilon1e-8, weight decay.01, clipping1, BF16 parameters/moments/KL, Flex training attention, effective response batch 2, up to512 sampled anchors per response presentation, 8-block objective chunks, response-mean normalization, position decay gamma 7. Only answer positions receive supervision; prompts supply context. Training cap 2272; no truncation.
- **Serving:** pinned SGLang/FA3 and tokenizer, BF16 greedy, block 16, concurrency 1, 2048 output tokens or EOS, 4096 canonical context capacity, qualified cold-cache/publication policy. Three fresh capture-disabled repeats with rotating F/O order. Prefill, drafting, verification and request cache maintenance are inside request latency; training and checkpoint activation are outside it.
- **Code:** only `experiments/osd_one_event` and focused tests were added to A800 DFlash-Online. Existing `online_feedback`, upstream training and inference implementations stayed unchanged. Fenrir DFlash-Research and its user debugger were untouched.

The configuration retains the qualified7424-step constant-LR horizon and 58-event maximum as provenance. The experiment runner invokes **exactly one** event and closes the trainer after step 128. It neither resumes the old7424-step checkpoint nor executes subsequent events. Explicit fresh output paths prevent writes into historical runs.

## Paired serving results

Verification metrics count one validated trajectory per request, designated repeat0. All three repeats reproduce these counts. Time sums all 64 request latencies and averages the three repeats. Reductions use pooled sums, not the average of per-request percentages.

| Metric | Frozen F | One-event O | Reduction versus F |
|---|---:|---:|---:|
| Verification calls | 3,140 | 3,075 | **2.0701%** |
| Calls /1,000 output tokens | 160.6385 | 157.3131 | 2.0701% |
| Sum of mean request times | 54.0079s | 52.8010s | **2.2346%** |
| Accepted draft tokens | 16,399 | 16,468 | — |
| Accepted draft tokens /call | 5.2226 | 5.3554 | — |
| Committed progress /call | 6.2048 | 6.3359 | — |
| Output tokens | 19,547 | 19,547 | identical |

Committed progress excludes the first token produced by prefill. Accepted-token accounting preserves stop-clipped matching tokens and the qualified native trace conventions; it is not inferred by subtracting calls from output length.

| Request-level outcome | Improved | Regressed | Tied |
|---|---:|---:|---:|
| Verification calls | 40 | 19 | 5 |
| Mean request latency | 41 | 23 | 0 |

The call reductions save113 calls across improving requests and add48 across regressions:65 net saved. Latency improvements save1.9873s and regressions add0.7804s:1.2069s net saved. Latency signs are measured differences, not per-request significance tests or proof of changed agreement.

| Timing repeat | F total | O total | Time reduction |
|---|---:|---:|---:|
| 0 | 54.0012s | 52.8236s | 2.1807% |
| 1 | 54.0134s | 52.7599s | 2.3207% |
| 2 | 54.0091s | 52.8195s | 2.2025% |

Three repeats are repeated timings of64 requests, not192 independent examples or three training seeds. Every repeat, gain and regression is retained.

## Adaptation and execution costs

| Measured component | Seconds | Accounting scope |
|---|---:|---|
| Frozen feedback collection, controller boundary | 138.824 | Includes capture, record validation and buffer persistence |
| Native captured-feedback request latency | 100.973 | Nested within collection |
| Teacher preparation | 4.891 | Includes128 terminal-feature target forwards; no new generations |
| Prepared feature persistence | 5.546 | Part of blocking update |
| Training | 313.492 | Includes teacher-head projection during KL computation |
| Full checkpoint persistence | 18.177 | Part of blocking update |
| Inference checkpoint export | 15.225 | Part of blocking update |
| Activation, complete worker RPC | 1.877 | Includes checkpoint validation/publication; native internal timer0.399s |
| **Whole blocking update** | **363.633** | Includes the preceding preparation/training/save/export/activation components once |
| Backend + trainer loading | 42.051 | One-time initialization |
| All benchmark checkpoint switches | 361.134 | Evaluation-only overhead; excluded from serving gains and deployment reconstruction |
| Entire paired evaluation wall time | 691.742 | Includes384 generations, switching, validation and recording |
| Actual campaign wall time | 1,247.771 | About20m48s, including loading and experiment overhead |
| Outer execution, including CPU preflight | 1,272 | **21m12s**, 08:27:53–08:49:05UTC |

Capture host work76.275s, capture persistence5.739s and capture cache maintenance0.140s are nested within collection, not additional charges. The teacher-head timer103.741s is nested within training 313.492s; it is not a separate full-target replay cost. Aggregate request latency does not identify a detailed GPU-kernel breakdown.

Collection plus the blocking update totals502.457s. Adding one mean O panel gives a **reconstructed sequential cost of555.258s**, excluding loading/diagnostic benchmarks. This is a reconstruction of the instrumented sequence, not a separately measured deployment episode. The64-request serving saving of1.207s does not offset the 363.633s update; no long-run amortization or asynchronous-serving claim is made.

## Qualification, exposure and retention

- **337 A800 CPU tests passed.** Source/config/model/runtime checks passed before launch.
- First8 evaluation requests/48 generations passed native qualification within the 384-evaluation budget. Both checkpoints passed exact native tensor/helper checks. A null or harmful result was never a stop condition.
- **512/512 generations:**128 feedback +384 evaluations. No extra generation warmups, U/P checks, extra learning events, retries, invalid comparisons, unknown work or unattempted work.
- Every evaluation input/output/termination matches the saved reference; F historical counts and each method's repeated counts pass. Independent audit checks384 raw records and 18,645 native verification blocks.
- Training accounts for128 responses, 256 presentations, 128 steps, 65,096 kept anchors and**949,560 supervised position occurrences**. Source answers contain32,679 tokens; complete training sequences total42,154 tokens, maximum 619. No feedback or evaluation cap hits.
- Exposure was reconstructed from anchors and response masks, then reconciled with optimizer-boundary metrics. Summing all metric rows would give1,424,190 by counting intermediate responses again inside their two-response accumulation-window totals. This is a reporting convention, not extra training.
- The saved checkpoint contains 58 BF16 Adam states, each at step 128, with 128 global steps, 256 presentations, one event and saved RNG. CPU-only final inspection initialized no CUDA context.
- Fresh feedback output IDs and all 256 presentation orders/anchor draws exactly match the archived first event of the successful reference. Parent audit also finds all 256 recorded training metric dictionaries equal. This supports preservation of the qualified first-event recipe; it is not comparison with the old58-event endpoint.
- Training closed before timing. No status/artifact retrieval occurred between08:36:54UTC, when training was last observed active, and the verified process exit. All retrieval occurred afterward.
- No files deleted. A800 retains native/prepared features, compact traces, the new export and full optimizer/RNG state at `/data/sources/minhao/runs/dflash_osd_one_event_20261025/campaign_001`. Both GPUs are free; approximately35.7GB A800 and 83.4GB Fenrir remain free.

## Files and interpretation limits

[Per-request paired results](PER_REQUEST.csv), [all 384 timing records](ALL_REPEATS.csv), [repeat totals](REPEAT_TOTALS.json), [training exposure](TRAINING_EXPOSURE.csv), [independent audit](INDEPENDENT_AUDIT.json), [validity audit](VALIDITY_AUDIT.json), [final optimizer-state audit](FINAL_STATE_AUDIT.json), [raw receipts](raw/campaign_001/), [pinned configuration](../sealed/config.yaml), [manifest](../sealed/MANIFEST.json), [runner source](../worktree/experiments/osd_one_event/run.py).

The supported conclusion is limited to one event, one training seed and this reused GSM8K development panel: **ordinary OSD gives a small positive serving benefit, with substantial request-level regressions and update cost not recovered within 64 requests.** This neither establishes a new mechanism nor tests the ranking proposal. The run stops here.

ARS fallacy scan11/11: preserve individual outcomes alongside aggregate gains; no subgroup/universal inference; disclose reused panel and fixed selection; no collider adjustment; report denominators; no extreme-score selection or unsupported regression-to-mean explanation; no attrition or favorable-only filtering; no hidden multiple-testing/significance claim; no outcome-based retuning; do not attribute observed gains to an untested mechanism; feedback temporally precedes evaluation and never contains its answers.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_osd_one_event_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
