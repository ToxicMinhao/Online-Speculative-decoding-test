# One-event R reference-margin experiment — completed

## Material Passport

- Date: October 26, 2026. ARS experiment-agent run/validation.
- Status: completed on A800, exit 0. The bounded experiment is stopped; no additional event, tuning or baseline retraining was launched.
- Scope: one training event, seed 17, the existing 64-request GSM8K development panel; Stage 1/project None unchanged.
- Evidence: [raw summary](raw/campaign_001/evaluation/report/SUMMARY.json), [per-request table](raw/campaign_001/evaluation/report/PER_REQUEST.csv), [all repeats](raw/campaign_001/evaluation/report/ALL_REPEATS.csv), [independent audit](FINAL_AUDIT.md), [qualification](QUALIFICATION_REPORT.md).

## Result

**R did not improve on ordinary OSD in this screen.** Both required 3,075 verification calls. R saved 40 calls on 25 requests, added 40 calls on another 25, and tied on 14. The equal total does not mean that the checkpoints or their request-level behavior were identical.

R's summed mean request time was 53.087305 seconds versus O's 53.059998 seconds: **0.027307 seconds slower across the complete panel**, a 0.0515% increase. Compared with P, R required six more calls and 0.042803 seconds more request time. These differences do not demonstrate a useful advantage from the revised penalty.

R beat frozen by 65 calls (2.0701%) and 1.222594 seconds (2.2511%). Ordinary OSD already delivered the same aggregate call reduction. **The improvement over frozen is not evidence of an additional benefit from R.**

## What ran

The target was frozen Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`; the released draft was z-lab/Qwen3-8B-DFlash-b16, revision `9b41424b7109f9c5413454f481b09a82b85333f4`.

Only R was trained. It began from released weights with a fresh BF16 AdamW optimizer and the qualified seed-17 full RNG state. It reused the retained 128 frozen-feedback responses/features and exact original presentation/anchor schedule: two passes, 256 response presentations, 128 optimizer steps. Held-out evaluation requests never supplied training feedback. F/O/P checkpoints were reused unchanged, and U artifacts were preserved.

R retains ordinary weighted forward KL and P's event-fixed reference-prefix mask. Its auxiliary penalty is the per-position positive part of current strongest-rival softplus minus detached reference strongest-rival softplus, with each model choosing its own noncanonical rival. Lambda remains one. Equal or improved margins receive zero auxiliary penalty. Weights, denominator, response averaging, accumulation and all learning settings are unchanged.

Settings: constant learning rate 3e-4, weight decay 0.01, betas 0.9/0.999, epsilon 1e-8, gradient clipping 1, BF16 parameters/Adam/KL, FP32 margin/softplus, FlexAttention, response batch size one with accumulation two, two shuffled passes, up to 512 anchors per response, gamma seven, eight-block objective chunks and 2,272-token training sequences. The inherited 7,424-step horizon is a preset ceiling; **only the declared 128-step event ran**.

Serving used pinned SGLang/FA3, greedy decoding, block size 16, 2,048 output tokens or EOS, context limit 4,096, prefill chunk 8,192, concurrency one and valid cold caches. All training closed before timing. F/O/P/R were evaluated freshly together in three rotating-order repeats, capture disabled and without concurrent training or transfers.

## Fresh paired serving measurements

| Checkpoint | Verification calls | Sum of mean request times (s) | Calls / 1,000 output tokens | Accepted draft tokens / call | Committed progress / call |
|---|---:|---:|---:|---:|---:|
| F — frozen | 3,140 | 54.309900 | 160.6385 | 5.2226 | 6.2048 |
| O — ordinary OSD | 3,075 | 53.059998 | 157.3131 | 5.3554 | 6.3359 |
| P — prefix ranking | 3,069 | 53.044502 | 157.0062 | 5.3682 | 6.3483 |
| R — reference-margin preservation | 3,075 | 53.087305 | 157.3131 | 5.3554 | 6.3359 |

Request time includes prefill, drafting, verification and request cache maintenance. It excludes adaptation, loading and checkpoint activation: these are measurements **with the checkpoint available**. Each request contributes its three-repeat mean; counts use one validated repeat-zero trajectory. All checkpoints produced the same 19,547 tokens; committed verification progress excludes the 64 initial seed tokens.

| Comparison | Call reduction | Request-time reduction | Requests with fewer / more / equal calls |
|---|---:|---:|---:|
| **R versus O — primary** | **0.0000%** | **−0.0515%** | **25 / 25 / 14** |
| R versus P | −0.1955% | −0.0807% | 15 / 17 / 32 |
| R versus F | +2.0701% | +2.2511% | 38 / 18 / 8 |
| O versus F | +2.0701% | +2.3014% | 40 / 19 / 5 |
| P versus O | +0.1951% | +0.0292% | 24 / 25 / 15 |

Positive reductions mean improvement. Ratios use pooled sums, not averages of per-request percentages. R/O request-time reductions for the three repeats were −0.063319%, −0.012148% and −0.078958%. No significance or population-generalization claim is made. Three repeats are repeated measurements of 64 requests, not 768 independent examples.

## Correctness and mechanism activity

All 768 planned evaluations completed and passed exact input/output-token, termination, capture, publication and cache checks. All four checkpoints passed native weight equality. The first eight requests qualified within the existing evaluation budget. No request was invalid, incomplete, unattempted or of unknown completion. Independent auditing reconstructed 37,077 native verification blocks. Fresh F/O/P counts, accepted tokens, committed progress and output lengths match their prior retained measurements in all 576 corresponding rows; historical timings are not pooled.

GPU qualification passed before main training: five discarded optimizer steps, ten presentations/backwards, four reference forwards, four detached full-block comparison forwards, twenty bounded detached-logit gradient calls and no generations. R with lambda zero matched fresh O exactly at the checked gradient/state boundaries. Active R passed masks, own-rival calculations, loss/logging algebra, gradients and reference/RNG isolation. Fresh main R started at zero counters with no inherited temporary learning. All 443 local CPU tests and 20 remote execution tests passed; eight auditor fixtures also passed.

Main exposure matched the existing schedule: **65,096 sampled anchors, 949,560 supervised-position occurrences and 429,510 protected-prefix occurrences**. At optimizer boundaries, the records show:

| Training diagnostic | Numerator / denominator | Result |
|---|---:|---:|
| Protected positions with active R penalty | 241,673 / 429,510 | 56.2671% |
| Reference-correct positions predicted incorrectly by the current model | 38,062 / 545,042 | 6.9833% |
| Pooled weighted forward KL | 377,333.257141 / 423,956.893433 | 0.890027 |
| Pooled weighted auxiliary penalty | 11,175.402288 / 423,956.893433 | 0.026360 |

The auxiliary term was active during training; the null serving result is not explained by it never activating. These diagnostics aggregate sampled training occurrences across the trajectory, using only completed accumulation boundaries to avoid double-counting. They are not final-checkpoint held-out losses or unique-token rates. The correctness-regression denominator includes all supervised reference-correct positions, including those outside the protected consecutive prefix. There is no matched O/P diagnostic for these new rates, so they do not establish that R preserved margins better than either control. A soft penalty is not a hard preservation guarantee.

## Costs

| Cost | R / campaign measurement |
|---|---:|
| R loading | 19.246 s |
| Fresh-state/RNG validation | 8.079 s |
| R training | 383.806 s |
| R checkpoint save | 17.652 s |
| R export | 15.475 s |
| **Complete R event** | **419.580 s** |
| Reference setup/forward/head, nested in event | 66.828 s |
| Teacher-head work, nested in training | 103.769 s |
| R peak allocated / reserved GPU memory | 29.254 / 34.356 GB |
| GPU qualification | 366.870 s |
| Native backend loading | 54.017 s |
| Evaluation wall time | 1,836.385 s |
| Benchmark-only activation / weight validation | 1,173.958 / 12.428 s |
| Python campaign wall time | 2,710.199 s |
| Complete supervisor time, including preflight/CPU checks | **46 min 26 s** |

Reference and teacher-head timings are nested costs, not additions to the event total. New feedback generation and teacher-backbone preparation were zero because retained artifacts were reused; their historical cost is not zero. The 705 benchmark publications serve the rotating comparison and do not represent a deployment update schedule. The request-time result does not establish an adaptation-inclusive speedup; this screen spends about seven minutes preparing R for roughly 1.22 seconds of measured serving savings over frozen on this panel.

## Limits and decision boundary

The supported conclusion is narrow and clear: **this one-event R revision provides no useful improvement over ordinary OSD on the tested panel.** It does not justify a positive mechanism claim or automatic tuning. It also does not establish that every margin-based objective would fail, or identify which element—mask, objective strength, fitting/transfer relationship or optimization—explains the result. This run contains no new matched-prefix attribution or final-checkpoint source-fit diagnostic.

The retained startup caveat remains: O is a valid practical comparator with the same learning settings, but historical O's complete 128-step numerical equivalence to warmed initialization was not established. New R/lambda-zero gates check the controlled window; they do not relabel the original failed gate as passed. R and P share the qualified warmed-startup design. No baseline retraining follows from the absence of full-trajectory bitwise equivalence. See the [earlier evidence review](../dflash_prefix_ranking_execution_20261026/startup_review/REVIEW.md).

All 11 ARS statistical/methodological fallacy categories were considered. This is descriptive, paired development evidence: no p-values or confidence claims; no repeat inflation, outcome-dependent exclusions, subgroup causality, selective reporting or extrapolation beyond the panel. Training diagnostics are separated from held-out serving results. Aggregate ties retain the underlying gains and regressions.

## Retention and operational status

Both A800 GPUs are idle. Postflight rehashed 110 source files, nine execution files, 32 dependencies and 140 retained artifacts without changes. All 2,386 retrieved compact files (176,302,541 bytes) match the server manifest. Full qualification snapshots, R weights/optimizer state and previous controls remain on A800; compact evidence is on Fenrir. No files were deleted. A800 had 117.185 GB free after the run; Fenrir has sufficient space for the compact retrieval.

Key artifacts: [independent audit](FINAL_AUDIT.json), [qualification audit](QUALIFICATION_AUDIT.md), [postflight](raw/POSTFLIGHT.json), [transfer audit](RETRIEVAL_AUDIT.json), [training receipt](raw/campaign_001/R/events/event_001/RESULT.json), [costs](raw/campaign_001/COSTS.json), [all timing repeats](raw/campaign_001/evaluation/report/ALL_REPEATS.csv), [per-request results](raw/campaign_001/evaluation/report/PER_REQUEST.csv).


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_margin_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
