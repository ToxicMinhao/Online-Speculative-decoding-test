# KPLUS4 to event-six MathQA: retrospective count analysis

**COMPLETED_EXISTING_RECORD_ANALYSIS. All five requested contrasts are available.**

The same native N6 checkpoint improves accepted proposals per verification over KPLUS4 on both panels: **+1.561% [95% CI +0.729%, +2.406%]** on PR51 and **+1.738% [+1.070%, +2.436%]** on PR52. These two native adaptation events added transferable acceptance on these fixed inputs. This does not establish a serving-time improvement relative to KPLUS4: its reference-generation latencies were not compared.

C6 also improves over KPLUS4, but **C6/N6 remains the relevant incremental contrast**. The already published MathQA C/N estimates do not establish an allocation benefit or non-degradation.

## Material passport and scope

ARS experiment-agent run/validate; 2026-11-17; verification status **ANALYZED** by authenticated retained-record reconstruction and a separate arithmetic reader. No new model execution or independent training replication.

Analysis execution commit: `c6e12678bb3989e088564b6de4239075be62cf3a`. Original PR51 model execution: `15284b323a307d01118e2630e665ed0191ba7c33`; PR52: `3b25b068f0caa0035aeb4c8d58ba498f02a814cc`. Publication is the later Git revision containing this report. Native K+ remains primary; C is **candidate retained for contribution assessment; performance preservation unresolved**. This is the user's explicit prospective policy revision, not a rewritten experimental decision. Stage 1 / project None remains unchanged.

## PR51 panel — 256 original scored requests

| Checkpoint | Calls | Native accepted/call | Acceptance gain vs KPLUS4 [95% CI] | Call reduction [95% CI] | Calls saved / added / net | Wins / regressions / ties |
|---|---:|---:|---:|---:|---:|---:|
| KPLUS4 | 17,910 | 6.294472 | — | — | — | — |
| N6 | 17,672 | 6.392712 | +1.561% [+0.729%, +2.406%] | +1.329% [+0.625%, +2.034%] | 500 / 262 / 238 | 133 / 84 / 39 |
| S6 | 17,647 | 6.403185 | +1.727% [+0.902%, +2.569%] | +1.468% [+0.772%, +2.168%] | 512 / 249 / 263 | 139 / 76 / 41 |

Each arm returns 130,900 output tokens: 130,644 committed-progress tokens after excluding one initial token per request. All 10 capped requests and 246 EOS requests remain; zero-call requests: 0. Native accepted totals are KPLUS4 112,734, N6 112,972 and S6 112,997. The two separate warmup inputs and all final timing warmups are excluded by the sealed scored-input list.

## PR52 panel — 256 original scored requests

| Checkpoint | Calls | Native accepted/call | Acceptance gain vs KPLUS4 [95% CI] | Call reduction [95% CI] | Calls saved / added / net | Wins / regressions / ties |
|---|---:|---:|---:|---:|---:|---:|
| KPLUS4 | 18,944 | 6.304635 | — | — | — | — |
| N6 | 18,664 | 6.414220 | +1.738% [+1.070%, +2.436%] | +1.478% [+0.914%, +2.059%] | 484 / 204 / 280 | 136 / 77 / 43 |
| C6 | 18,654 | 6.418194 | +1.801% [+1.116%, +2.517%] | +1.531% [+0.954%, +2.124%] | 511 / 221 / 290 | 146 / 74 / 36 |
| S6 | 18,693 | 6.402718 | +1.556% [+0.870%, +2.275%] | +1.325% [+0.747%, +1.923%] | 476 / 225 / 251 | 146 / 85 / 25 |

Each arm returns 138,635 output tokens: 138,379 committed-progress tokens. All 13 capped requests and 243 EOS requests remain; zero-call requests: 0. Native accepted totals are KPLUS4 119,435, N6 119,715, C6 119,725 and S6 119,686. The two separate warmup inputs and all final timing warmups are excluded. No favorable subset replaces either complete panel.

Both tables compare each endpoint with KPLUS4. Positive percentages favor the endpoint. Wins/regressions/ties and gross saved/added calls also use KPLUS4 as the comparator. CSVs retain unrounded relative gains, absolute accepted-length differences, interval endpoints and undefined-draw counts.

## Validity and reduction

Every source record is rehashed against its retained delivery manifest and joined by source/request identity and exact rendered input tokens. Canonical output tokens, stop tokens, termination, cap handling and all three endpoint repeats agree. There are no unavailable requested contrasts, excluded scored requests or undefined bootstrap draws.

The origin export identity is 2,097,259,104 bytes, SHA256 `931e46620d8fc930ea0c22fcdf6af30dd16e16445f8141474ecfa4e48746dff7`. Original seals, prelaunch authentication and before/after loaded-weight checks bind it; this review did not reload weights. Recorded tokenizer/model pins and runtime source identities agree. Serving remains BF16 greedy SGLang/FA3, block 16, at most 2,048 tokens or EOS, request context 4,096, concurrency one, capture off and the original cache policy.

Origin references ran on physical card B and final evaluation on A. Retained qualification has exact native count/output signatures across both cards and identical disposable training endpoints. This bounded qualification supports the count contrast; it is not an exhaustive hardware-invariance theorem. Different phases, cards and concurrent work make **origin latency unsuitable for a speedup comparison**.

For each request, native accepted proposals = committed progress − verification calls, and committed progress = output length − 1. The reducer reconstructs every native block, including first rejection, committed tokens and EOS/cap clipping. The legacy top-level `accepted_draft_tokens` field instead counts stop-clipped matches; it differs from native acceptance on 218/219 origin requests in PR51/52 and is retained separately. It is not substituted into the primary ratio. A hypothetical zero-call request is retained; undefined pooled ratios/draws are explicitly reported rather than silently dropped.

Counts use one validated trajectory per request; timing repeats are not independent samples. Pooled ratios are computed from summed counts. The original 10,000 paired whole-request draws are reproduced exactly: PCG64 seeds 1711/1717, lexical source/request order, linear percentile endpoints, original draw-matrix hashes matched. Every sampled request retains origin and all endpoints. Panels remain separate: they share N6/S6 checkpoints and are **not independent training replications**. These are retrospective, pointwise conditional intervals, not simultaneous multiplicity-adjusted intervals or training-seed uncertainty. Acceptance gains and call reductions are mathematically linked under identical outputs, not independent corroboration.

## What the evidence supports

**Added operation and useful role.** C redistributes bounded scalar supervision within the correct prefix while retaining native first-error treatment and total prefix mass. Its qualified implementation applied that change. PR50 found a positive native-K+ result on reused GSM8K development data; this MathQA origin analysis shows that C's complete continuation improves over its starting draft. Neither activation nor C/KPLUS4 identifies allocation as the cause of that improvement. Conserved scalar weight mass does not conserve gradients, Adam displacement or correctness.

**Performance relative to native K+.** The original [PR52 C/N comparison](../dflash_unchanged_c_mathqa_transfer_20261117/replay_transfer/REPORT.md) remains +0.02548% request-time reduction [−0.30275%, +0.34851%] and +0.06197% native acceptance [−0.31798%, +0.43226%]. Its 166 calls saved and 156 added leave 10 net, with 91 wins / 81 regressions / 84 ties. The sign remains unresolved; an interval crossing zero does not prove preservation or equivalence. No origin-to-endpoint timing was calculated here.

**Simpler controls and necessity.** [PR50](../dflash_bounded_prefix_allocation_20261116/REPORT.md) C/N was positive, while C/S points were negative with intervals crossing zero. PR52 C/S points are positive but both intervals cross zero. This leaves directional allocation's advantage over the fixed smaller-LR comparator unestablished; it does not erase the earlier native-K+ benefit. S remains informative, not a universal advancement gate. The user's prospective standard no longer discards a candidate merely below 0.5%; PR50/51/52 original gates and decisions remain historically intact.

The newly visible native N6/KPLUS4 benefit shows that these updates retained useful transfer. It does not show saturation, exhausted headroom, a solution to regressions, or a causal explanation for C. C remains under contribution assessment with performance preservation unresolved. This record analysis is measurement context, not a new substantive mechanism. Any later complete-method comparison with OSD or other GPU work needs its own fixed question and budget; none follows automatically.

## Validation, cost and evidence

Eleven focused reducer tests pass. The executed reducer passes 3,218,804 checks; a separate raw arithmetic reader passes 2,878,387, and reconciliation passes 15,520. The final provenance reader passes 6,948 checks. These are deterministic audit assertions, not statistical samples. Readers share the project/model family and some historical implementation authorship; this is not blinded independent experimental replication.

The initial independent reader encountered PR51's absent newer source-order metadata field after validating its raw records. Its original output is preserved; the reader now checks PR51's actual saved row order and original draw hash. An initial resource-timer wrapper stopped before Python because its output directory was absent; creating that directory changed no scientific work. The earlier compatibility receipt consumed by the reducer is preserved byte-exactly; a later supplementary provenance pass adds target/tokenizer checks without changing any raw evidence, count or conclusion. No model retry occurred.

Initial estimate: **30–60 minutes**, with a one-hour executor-work ceiling and 100 MB new-output cap. Core reduction: **10.153 seconds** (resource wrapper 10.25 s; 11.15 user-CPU + 0.54 system-CPU seconds), peak RSS 135,668 KiB. Preparation/review duration through the completion receipt and output size are in [COMPLETION.json](COMPLETION.json). Model loads, forwards, backwards, optimizer steps, teacher preparation, generations, A800 operations and cleanup are all **zero**. Assets were read in place; no checkpoint or feature copy was made. No new timing or coefficient audit was requested or performed.

[PR51 table](PR51_TABLE.csv) · [PR52 table](PR52_TABLE.csv) · [PR51 summary](PR51_SUMMARY.json) · [PR52 summary](PR52_SUMMARY.json) · [PR51 paired counts](PR51_PAIRED_REQUESTS.json) · [PR52 paired counts](PR52_PAIRED_REQUESTS.json) · [PR51 source bindings](PR51_BINDINGS.json) · [PR52 source bindings](PR52_BINDINGS.json) · [consumed compatibility receipt](PROVENANCE.json) · [supplementary provenance](PROVENANCE_SUPPLEMENT.json) · [receipt versions](PROVENANCE_VERSION_NOTE.json) · [independent audit](INDEPENDENT_AUDIT.json) · [interpretation review](INTERPRETATION_REVIEW.md) · [reduction code](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/c6e12678bb3989e088564b6de4239075be62cf3a/DFlash-Online/experiments/post_pr52_origin_analysis/reducer.py) · [tests](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/c6e12678bb3989e088564b6de4239075be62cf3a/DFlash-Online/tests/post_pr52_origin_analysis/test_reducer.py).

Public source bindings preserve relative evidence locators and original byte identities; absolute private filesystem locators are consistently aliased. Original private token/text/native records and the alias map remain outside Git. No research-stage advancement, candidate tuning or new experiment is authorized by this completion label.
