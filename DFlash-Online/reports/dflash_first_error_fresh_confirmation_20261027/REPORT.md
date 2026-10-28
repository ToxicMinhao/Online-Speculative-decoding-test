# O/K confirmation on a new current-study panel

## Material Passport

- ARS experiment-agent run and validate; status **completed and audited**, interpretation **ANALYZED**.
- Native source: `raw/campaign_001`; independent audit: `FINAL_AUDIT.json` / `FINAL_AUDIT.md`.
- Supervisor: 2026-10-27 00:41:16–00:54:01 UTC, exit 0, **12m45s**. Stage 1 / project None remains unchanged.

## Answer

**K's modest advantage extends to this second request panel for these fixed checkpoints.** It used **60 fewer verification calls (1.880%)** and **1.386% less complete request time** than ordinary OSD. This is a fixed-checkpoint, one-training-seed result; it does not establish robustness across training seeds, feedback windows, workloads or future adaptation events. No revised objective, training or tuning was performed.

## Panel and qualification

The user approved 64 requests **new to the current one-event O/K study**, after retained executions showed that 64 globally unused GSM8K test questions were unavailable. All 64 selected questions have older unrelated-project exposure: 54 in an older serving reference and 10 in an earlier objective check. This exposure is disclosed per request. The panel is not described as globally unseen.

Selection used pinned `openai/gsm8k` main/test revision `740312add88f781978c0658806c59bc2815b9866`, seed `20261027`, source order followed by deterministic shuffle, and no output/performance filtering. The current 128 feedback, 64 development and contained qualification/probe inputs were excluded by source identity and normalized question matching. There were 1,255 eligible test rows. Source wording was preserved; all 128 training and 64 development rendered sequences were reproduced exactly before preparing the new prompts. New prompt lengths are 64–151 tokens. Input IDs, order and O/K checkpoint identities were sealed before generation.

The unchanged Qwen3-8B / DFlash-b16 pair ran through DFlash-Online's qualified SGLang/FA3 backend: greedy, block size 16, 2,048 tokens or EOS, context limit 4,096, concurrency 1, capture disabled, full cache flush before requests/publications. Only the two existing O/K exports were evaluated. No model/optimizer update, feedback, extra reference generation or generation warmup occurred. No concurrent training or artifact transfer occurred during timed serving.

All 14 focused CPU tests passed locally and on A800; independent code review resolved malformed-return and cleanup-receipt handling before sealing. The first 8 requests were the native correctness gate: 48 executions **included in 384**, with automatic continuation after correctness passed. The complete run contains 384/384 valid generations, 64/64 complete requests, and no invalid, pending, unknown or unattempted records. Every output and termination matches that request's **first new O execution**; K-first records were retained and checked once O became available. This verifies agreement under the native verifier, **not an independent target-only reference**. Both arms have identical verification counts across their three repeats; no historical O call-count gate was applied.

## Prespecified comparison

Counts use one validated repeat 0 trajectory per request; timing is the sum of each request's mean over three alternating-order repeats. Positive reduction means improvement. All output lengths are identical across methods.

| Metric | O: ordinary OSD | K: first-error-selected KL |
|---|---:|---:|
| Verification calls | 3,191 | 3,131 |
| Calls per 1,000 output tokens | 155.242 | 152.323 |
| Sum of mean complete request times | 56.101814 s | 55.324202 s |
| Output tokens | 20,555 | 20,555 |
| Accepted draft tokens | 17,363 | 17,418 |
| Accepted draft tokens per verification | 5.4412 | 5.5631 |
| Committed progress per verification | 6.4215 | 6.5446 |

Primary pooled call reduction is `1 − 3131/3191 = 1.880288%`. K reduced calls on **31 requests**, increased them on **20**, and tied on **13**. The improving requests saved 93 calls while regressions added 33, yielding 60 net saved. All regressions remain in the totals.

The pooled request-time reduction is **1.386073%**, saving **0.777612 s** across the 64 per-request means. 35 requests have lower mean latency with K and 29 have higher; these individual timing directions are descriptive. The three repeat-level pooled reductions are **2.9045%, 0.7005%, 0.5292%**, all favorable but variable. Three repeats are timing observations, not three independent request panels or training seeds. No p-value or significance claim is made.

Output lengths range 188–901 tokens; all 64 requests terminated at EOS, with zero cap hits. Committed progress totals 20,491 tokens per arm, excluding each request's first target-generated token. Accepted-draft tokens and committed progress therefore use different numerators.

This panel remains separate from the earlier development panel. Its positive direction agrees with the earlier K/O result, but no counts or timings are pooled across panels. The evidence supports a small additional serving benefit for **these fixed K/O checkpoints**, not universal superiority of first-error supervision or a mathematical guarantee. K's mask and selected-weight normalization are jointly embodied in its trained checkpoint.

## Costs and integrity

| Cost | Measured value | Scope |
|---|---:|---|
| Backend loading | 21.257 s | Outside request timing |
| 193 checkpoint publications/switches | 367.024 s | Outside request timing; experiment overhead |
| Native weight validation | 5.985 s | Outside request timing |
| All 384 measured request latencies | 334.278 s | Sum of three repeats, both arms |
| Evaluation controller wall time | 711.577 s | Includes requests, switches, checks and recording |
| Inner campaign wall time | 738.449 s | Includes loading, evaluation and cleanup |
| Supervisor wall time | 765 s | Also includes preflight and remote CPU tests |
| New training / feedback / optimizer steps | 0 | Existing checkpoints reused |

The nested timers are not added again to wall time. Dataset preparation, source review, postflight, transfer and this final audit are outside the supervisor duration. Serving results assume checkpoints are available; their original preparation cost is not erased by this reuse, and no adaptation-inclusive deployment speedup is established here.

Independent audit reconstructed **18,966 native verification blocks** and checked all 384 returned records and 384 worker captures. All 168 sealed file digests and exact checkpoint/model file sets pass postflight; the pinned external native runtime also passes its separate postflight check. Both GPUs are idle. All 1,187 retrieved evidence files (55,136,723 bytes) were verified on Fenrir. Only redundant transferred archives were retired after verified copies were retained; original source, model/checkpoint files, inputs, raw results and validity evidence remain intact.

## Deliverables

- [Per-request table](raw/campaign_001/evaluation/report/PER_REQUEST.csv)
- [All 384 repeat measurements](raw/campaign_001/evaluation/report/ALL_REPEATS.csv)
- [Repeat totals](raw/campaign_001/evaluation/report/REPEAT_TOTALS.csv)
- [Machine-readable result](raw/campaign_001/evaluation/report/SUMMARY.json)
- [Sealed inputs/checkpoints](raw/sealed/PLAN.json)
- [Selection and exposure audit](inventory_audit/INVENTORY_REPORT.md)
- [Code review](CODE_REVIEW.md)
- [Independent result audit](FINAL_AUDIT.md)

The authorized experiment ends at this report. No new update event, retuning, objective change or follow-on experiment is started.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_fresh_confirmation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
