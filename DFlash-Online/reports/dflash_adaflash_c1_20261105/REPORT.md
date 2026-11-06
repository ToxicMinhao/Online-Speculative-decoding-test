# K+ retains its advantage over paper-defined AdaFlash OPD at concurrency one

Execution: `681336c5e5c5caa0aa67560e46ffe83ce81e9351`, [PR #29](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/29). Verification status: **ANALYZED — complete campaign and retained-record audits passed**. The separate same-team reviews reconstruct records and arithmetic; they are not an independent GPU replication. Reviewer authorship overlap is disclosed in the linked audits.

## Result

The prespecified primary comparison favors **K+58**: complete-request time is **4.02845% lower than A58**, with a **95% paired-request bootstrap interval of 3.32888%–4.73465%**. Verification calls are **4.02813% lower**. K+ also has lower request time than retained ordinary OSD; A58 is slightly slower than OSD in this experiment.

| Checkpoint | Verification calls | Complete-request total, s | Native-generation total, s | Cache-maintenance total, s |
|---|---:|---:|---:|---:|
| O58 — ordinary OSD | 44,730 | 810.069 | 757.819 | 37.876 |
| K+58 | 43,267 | 783.742 | 734.924 | 35.453 |
| A58 — paper-defined AdaFlash OPD | 45,083 | 816.640 | 766.151 | 36.026 |

Each time total sums the per-request mean of three repeats over 1,024 requests. Calls use one validated trajectory per request after checking repeat consistency. Every method generated the same **314,847 output tokens**. Native generation and cache clocks are components; they do not provide a detailed kernel-time decomposition.

| Comparison: candidate / reference | Request-time reduction | 95% paired interval | Call reduction | Request call wins / regressions / ties |
|---|---:|---:|---:|---:|
| K+ / A — primary | +4.02845% | [3.32888%, 4.73465%] | +4.02813% | 687 / 195 / 142 |
| K+ / O — secondary | +3.25001% | [2.56777%, 3.94162%] | +3.27074% | 633 / 250 / 141 |
| A / O — secondary | −0.81112% | [−1.38820%, −0.24470%] | −0.78918% | 298 / 468 / 258 |

Reduction is `100 × (1 − candidate total / reference total)`; positive values favor the candidate. K+/A saves 2,181 calls and adds 365 across individual requests, for **1,816 net calls saved**. The primary time reduction is positive in all three repeats: 4.1873%, 4.2078%, and 3.6887%. All requests, including three capped requests, remain in the primary analysis.

The bootstrap uses 10,000 paired-request draws, PCG64 seed 20261102, and linear 2.5/97.5 percentiles. Repeats are not independent requests. Secondary intervals are unadjusted descriptive comparisons; they do not replace the prespecified primary decision.

## Acceptance and progress: explicit counter definitions

| Checkpoint | Runtime accepted counter / verification | Matched accepted draft tokens / verification | Native committed progress / verification |
|---|---:|---:|---:|
| O58 | 6.015940 | 6.037044 | 7.015940 |
| K+58 | 6.253172 | 6.275499 | 7.253172 |
| A58 | 5.961005 | 5.981811 | 6.961005 |

The runtime counter is `max(committed count − 1, 0)` per round. The matched-accepted counter clips the canonical-matching draft prefix at the actual stopping boundary. They can differ when EOS or the output cap removes the normal bonus-token slot; these are distinct counters, not an accounting error. Native committed progress totals **313,823 tokens** per method and excludes the first target-prefill token of each request. Full output includes those 1,024 initial tokens. All ratios use summed counts over summed verification calls. [Per-request counters](ACCEPTANCE_COUNTERS.csv) and [instrumentation audit](reviews/endpoint/AUDIT.md) preserve both definitions.

## Fixed comparison and completed work

One new A trajectory started from released Qwen3-8B-DFlash-b16, a fresh native BF16 AdamW optimizer and seed 17, with frozen Qwen3-8B target. The A objective is **0.8 teacher-argmax CE + 0.2 reverse KL**, temperature 1. Each vocabulary contribution `q*(log q − log p)` is capped from above at 0.01 before vocabulary summation; negative entries remain. It uses all valid response proposals, excluding the known seed, with the qualified sampler, positional decay and response normalization. There is no K/K+ selection mask or adaptive-length head.

The [paper formula](https://arxiv.org/html/2607.19223v3) differs from the [pinned released trainer](https://github.com/ZinYY/AdaFlash/blob/478bf0ff7a3a5232b13d0f14e752d5fe1da0b718/pipeline/training/rkl_model.py#L168-L207), which sums and normalizes before its position-level cap. The handoff explicitly chose the paper formula. This is a matched-protocol loss comparison, not a full author-system reproduction.

A served its own fixed 7,473-request GSM8K stream, trained after each 128 completed responses for **58 events**, and served the final 49 without updating. It completed **14,848 response presentations and 7,424 AdamW steps**: two shuffled passes, microbatch 1, accumulation 2, constant LR 3e-4, gamma 7, up to 512 anchors per response and the qualified 2,272-token training preparation limit. Existing O58 and K+58 exports were retained without retraining. All 14,848 natural presentation/anchor bindings match the historical comparator; no schedule was forced. This does not establish equal hidden features between arms.

Final serving used the original train-disjoint, previously used 1,024-request panel; three balanced repeats for O58/K+58/A58; greedy BF16/FA3, concurrency 1, block 16, context 4,096 and 2,048 tokens or EOS, with capture disabled and the established cache policy. Training ended before timed evaluation. All **9,216 timed executions** passed exact canonical tokens/termination, repeat counts, source/checkpoint identity and sealed-order checks. Invalid, unknown, excess and unattempted executions are all zero.

Total complete generations were **16,693**: 7,473 online + 9,216 timed + 4 inherited correctness checks. Tiny numerical GPU qualification used 12 synthetic loss forwards/backwards, zero model/Adam steps. No efficacy tuning, extra arm or follow-up run occurred.

## Costs and exposure

| A cost | Measured seconds | Scope |
|---|---:|---|
| Training | 18,398.732 | 5 h 06 min 39 s; includes nested teacher-head and compiler work |
| Whole update calls | 21,517.472 | Includes training, preparation, saving/export and activation; do not add to training |
| Terminal teacher preparation | 290.858 | Separately recorded component |
| Feature persistence | 336.574 | Separately recorded component |
| Checkpoint saving | 1,106.264 | Separately recorded component |
| Export | 896.716 | Separately recorded component |
| Publication/state validation | 1,465.548 | Separate validation work |
| Rolling retention | 540.700 | Separate storage work |
| A worker, including feedback collection | 31,265.087 | 8 h 41 min 05 s |
| Three-arm final evaluation | 7,794.139 | 2 h 09 min 54 s, including switching and evaluation bookkeeping |
| Entire wrapper | 39,212 | **10 h 53 min 32 s**, 00:14:09–11:07:41 UTC on 2026-11-06 |

A training peak was **19.254 GiB allocated / 22.338 GiB reserved** on its training device. Endpoint checkpoint switching took 209.693 s outside request latency. Costs above overlap where stated and must not be summed indiscriminately. Retained O/K+ training costs are historical; this run supplies no fresh matched training-overhead comparison or training-inclusive speedup claim.

Actual exposure was **3,728,050 anchors and 54,368,391 supervised position occurrences**. Entry clipping affected 897,525,502 / 8,260,515,854,976 vocabulary entries (**0.01086525%**), spanning 44,662,694 / 54,368,391 positions (**82.14827%**). These occurrence counts include overlapping sampled blocks. Each original forward contributes once; accumulation rows and checkpoint recomputation do not inflate exposure.

The recomputed teacher argmax differed from saved canonical tokens at 616 sampled occurrences across 46 presentations (0.00113301%). The prescribed CE uses teacher argmax. Native preparation bindings and final output validity pass; this diagnostic does not establish a numerical cause. Training loss or clipping activity alone is not evidence of better drafting.

## Evidence review, retention and limits

Separate same-team reviews passed **3,016,692 raw-endpoint checks**, **8,321,740 training/accounting checks**, and **204,139 statistics checks**, with no unresolved discrepancy. All 11 ARS fallacy categories were assessed. The raw audit covers 9,216 request records, 9,216 native sidecars and 399,240 verification rounds; 139 checkpoint switches and 144 quiescence boundaries also pass. Full model/optimizer/RNG equality is supported by retained execution receipts, not a second tensor-level reproduction by the auditors.

The prelaunch archive extraction error and a post-run auditor-only list-versus-count schema error remain preserved. Both were repaired without altering scientific measurements, tolerances or adding model work. All runtime scientific work completed under execution 681336c. [Endpoint audit](reviews/endpoint/AUDIT.md), [training audit](reviews/training/REPORT.md), and [statistics/interpretation review](reviews/statistics/REVIEW.md) describe their exact scopes and authorship overlap.

Rolling retention retired 22,386 tensor paths after checks; 59,864 compact evidence files still match their recorded identities. Initial/final A states and final export remain. Intermediate event 1–57 tensor replay/resume and retired feedback-feature reconstruction are unavailable; these tensors were retired, not archived. Compact raw token/native records remain on A800. The retrieved report bundle has 22 byte-verified files; this is not a claim that all raw artifacts were copied to Fenrir.

**Supported conclusion:** K+58 retains its endpoint advantage over this paper-defined A58 implementation under the fixed C1 setup. A58 did not improve over ordinary OSD here. The comparison does not isolate CE, clipping or feedback-feature effects, test full AdaFlash/adaptive length, establish other seeds/domains/concurrency, or prove adaptation-inclusive speedup. It does not by itself identify why A58 is worse. No further experiment is launched by this result.

## Deliverables

- [Per-request paired measurements](PAIRED_REQUESTS.csv), [every timing repeat](ALL_REPEATS.csv), [repeat totals](REPEAT_TOTALS.json), [acceptance counters](ACCEPTANCE_COUNTERS.csv).
- [Summary](SUMMARY.json), [validity](VALIDITY.json), [training/exposure](TRAINING.json), [exposure table](TRAINING_EXPOSURE.csv), [event costs](EVENT_COSTS.csv).
- [Checkpoint/configuration identities](CHECKPOINTS.json), [exact A configuration](config_A.yaml), [loss contract](LOSS_CONTRACT.json), [postflight](POSTFLIGHT.json), [retrieval verification](RETRIEVAL_VERIFIED.json).
- [Unmodified generated report](GENERATED_REPORT.md); original raw records remain under `/data/sources/minhao/runs/dflash_adaflash_c1_20261105/production_001`.
