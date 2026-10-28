# K1 versus KRC: additional-request confirmation

**KRC has a small measured drafting and request-time advantage on this additional panel for these fixed checkpoints.**

Exactly 384/384 generations completed; 64/64 requests are valid. Independent audit: passed. No training, new feedback, corrections, KL evaluation, local probes or new checkpoints were performed.

| Metric | Retained K1 | Retained KRC |
|---|---:|---:|
| Verification calls | 3150 | 3125 |
| Accepted draft tokens | 16769 | 16795 |
| Output tokens | 19928 | 19928 |
| Committed progress / call | 6.306032 | 6.356480 |
| Calls / 1,000 output tokens | 158.069049 | 156.814532 |
| Sum of per-request mean latency (s) | 55.833496780 | 55.088521150 |

Primary pooled call reduction: **+0.793651%**, computed as `1 − ΣKRC/ΣK1`. Requests improved/regressed/tied: **25/14/25**. Gross calls saved/added: **42/17**; net saved: **25**.

Request-time reduction from summed three-repeat request means: **+1.334281%**. Counts use one validated repeat-0 trajectory per request; timing repeats are not extra request samples.

| Repeat | K1 request time (s) | KRC request time (s) | Time reduction |
|---:|---:|---:|---:|
| 0 | 56.134092480 | 55.578913737 | +0.989022% |
| 1 | 55.349528149 | 54.642227497 | +1.277880% |
| 2 | 56.016869711 | 55.044422215 | +1.735990% |

Leave-one-request-out call reduction ranges **+0.678295% to +0.869845%**. 0/64 omissions produce a nonpositive reduction. This is descriptive sensitivity only: **all 64 requests remain in the primary result**. All omission-level count and timing results are retained.

Output lengths are 122–872 tokens (mean 311.375000), identical across checkpoints. EOS/cap terminations: 64/0 requests. Accepted tokens and committed progress are reported separately; committed progress excludes the initial target-generated token.

## Inputs and serving contract

Seed **20261028** selected 64 requests from 1191 eligible pinned GSM8K test questions. Current-study exclusions cover 512 feedback questions, the 64-request development panel, the 64-request O/K confirmation panel, and all recorded qualification/probe inputs. Source IDs and NFKC/whitespace-normalized questions were checked; test prompts were deduplicated. All 640 inherited prompt sequences and all 64 new prompt sequences were reconstructed exactly. Selection used no answers, generated lengths or performance. All 64 selected questions have disclosed older unrelated-project exposure; the panel is new to this study, not globally unused.

Dataset revision: `740312add88f781978c0658806c59bc2815b9866`. Target/tokenizer: Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`; released DFlash family: `9b41424b7109f9c5413454f481b09a82b85333f4`. Exact K1/KRC export identities and rendered token IDs were sealed before generation.

Qualified SGLang/FA3 runtime, BF16, greedy decoding, block 16, 2,048 tokens or EOS, context 4,096, concurrency 1, capture disabled and established cache flushing/checkpoint refresh. The order alternates by `(request_ordinal + repeat) % 2`, with 32 requests starting each checkpoint in every repeat. No concurrent training or artifact transfers occurred during timed serving. Normal native initialization was measured separately; no generation warmup was added.

## Correctness and reproducibility

The first 8 requests supplied the included 48-execution correctness gate; remaining requests continued unchanged. Each request's first fresh K1 execution supplied canonical tokens and termination. KRC-first records were kept pending and validated once K1 was available. This verifies agreement under the qualified native verifier, not an independent target-only reference. Every returned record was preserved before validation; verification counts must match across repeats. Native checkpoint tensors were checked before/after, and file identities/source/runtime were rechecked by postflight.

Valid/invalid/pending returned generations: 384/0/0. Unknown/unattempted: 0/0. Audit issues: 0; reconstructed native blocks: 18825.

18 evaluator/driver CPU tests and 3 selection tests passed; the evaluator/driver checks were independently repeated. Three focused independent audit-accounting tests also passed. These are implementation checks, separate from the included native gate.

## Costs

| Cost | Seconds |
|---|---:|
| Backend loading | 20.257213 |
| Checkpoint publications (193) | 282.785738 |
| Native before/after weight validation | 8.544533 |
| All 384 measured request latencies | 332.766054 |
| Evaluation controller wall | 628.486646 |
| Campaign process wall | 654.172813 |
| Supervisor wall including pre/postflight and CPU checks | 703 |

Supervisor: 2026-10-27T23:11:50Z–2026-10-27T23:23:33Z; exit 0. Timers are nested and must not be added again. Loading, checkpoint switching, checks and data retrieval are not included in serving latency. New training/correction cost is zero; historical preparation costs are not erased and no training-inclusive speedup is claimed.

## Interpretation and deliverables

This panel is reported separately from development. The test concerns these fixed checkpoints and additional requests. It does not establish training-seed or feedback-window robustness, isolate preservation from extra reference-response exposure, or establish a training-inclusive speedup. No statistical-significance claim is made. No algorithm change, new panel, retraining or further event follows automatically.

- [Sealed request/checkpoint manifest](raw/sealed/PLAN.json)
- [Exclusions and selection](inventory_audit/SELECTION_AUDIT.json)
- [Per-request table](raw/campaign_001/evaluation/report/PER_REQUEST.csv)
- [All 384 repeats](raw/campaign_001/evaluation/report/ALL_REPEATS.csv)
- [Repeat totals](raw/campaign_001/evaluation/report/REPEAT_TOTALS.csv)
- [Leave-one-out table](raw/campaign_001/evaluation/report/LEAVE_ONE_OUT.csv)
- [Raw validity events](raw/campaign_001/evaluation/VALIDATION_EVENTS.jsonl)
- [Independent audit](FINAL_AUDIT.md)
- [Source review](REVIEW.md)

ARS experiment-agent run/validate; separate same-team reviewers, not blinded external review. Stage 1 / project None unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_confirmation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
