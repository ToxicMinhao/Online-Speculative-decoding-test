# O58 / K58 / K+58 MathQA serving transfer

## Material Passport

ARS experiment-agent run → validate. Frozen GSM8K-trained checkpoints on new MathQA requests. Saved-record audit; not an independent model reproduction.

Execution commit: `12583640baf2210f767e108e3632c0e5f9b8caaa`.

Status: **valid_complete**. Zero training, updates, feedback capture or teacher preparation.

**Completed result:** K+58 reduces complete request time by **2.0522% versus ordinary OSD**, but **does not establish an incremental improvement over K58**. The K+/K estimate is −0.1003%, with a 95% interval crossing zero. Its small earlier GSM8K advantage over K did not reproduce on this MathQA panel. This is not evidence of equivalence or definite harm.

## Complete-request time

| Checkpoint | Total request seconds¹ | Mean ms/request | K+ reduction versus this checkpoint [95% CI] | Native-generation seconds¹ | Cache-maintenance seconds¹ | Verification calls² | Output tokens² | Cap hits² |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| O58 | 1358.1274 | 1326.2963 | +2.0522% [+1.6520%, +2.4525%] | 1293.3374 | 54.5015 | 74786 | 519261 | 45 |
| K58 | 1328.9236 | 1297.7770 | -0.1003% [-0.4399%, +0.2427%] | 1266.9123 | 51.2900 | 73117 | 519261 | 45 |
| KPLUS58 | 1330.2565 | 1299.0786 | — | 1266.2131 | 53.2152 | 73132 | 519261 | 45 |

¹ Sum of per-request means over three fresh repeats. Complete-request time includes cache maintenance; loading and checkpoint switching are outside it. The two listed components need not exhaust the complete timer.

² One validated trajectory per request after repeat consistency checks, not three times the same output.

KPLUS58 has lower complete request time than O58 on this panel.

An incremental complete-request-time improvement over K58 is not established. Secondary role: **confirmatory**.

The prespecified primary is K+ versus OSD (O). K+ versus K is confirmatory only if the primary complete-time interval has lower endpoint above zero. Both estimates remain visible. Call and component-time intervals are descriptive.

## Three timing repeats

| Repeat | O complete / native / cache seconds | K complete / native / cache seconds | K+ complete / native / cache seconds |
|---|---:|---:|---:|
| 1 | 1357.0353 / 1296.2342 / 51.7586 | 1329.1207 / 1265.3741 / 52.2439 | 1331.8143 / 1266.6127 / 53.2859 |
| 2 | 1359.4598 / 1290.1292 / 58.2371 | 1330.2712 / 1268.9946 / 50.5355 | 1328.9231 / 1263.9278 / 54.3359 |
| 3 | 1357.8870 / 1293.6488 / 53.5087 | 1327.3791 / 1266.3681 / 51.0905 | 1330.0320 / 1268.0988 / 52.0238 |

## Drafting counts

K+ versus O: +2.2116% pooled call reduction; 635 request improvements, 249 regressions, 140 ties. Gross calls saved/added: 2228/574.

K+ versus K: -0.0205% pooled call reduction; 407 request improvements, 396 regressions, 221 ties. Gross calls saved/added: 859/874.

## Analysis and interpretation limits

All 1,024 paired request identities are included. The same 10,000 PCG64 seed-20261103 resamples are used for both contrasts; each draw resamples 1,024 request identities carrying every method. Intervals use linear 2.5/97.5 percentiles. Timing repeats are measurements of a request, not independent samples.

Canonical outputs are each request’s fresh O58 repeat-0 tokens and termination under the native verifier. This is not a separate target-only reference. Valid cap hits and slow requests remain included.

This tests cross-dataset serving transfer for these fixed GSM8K-trained checkpoints. It does not establish training-seed robustness, MathQA adaptation, training-inclusive speedup, or a causal explanation for a difference. Historical GSM8K times are not pooled with these measurements.

## Work, costs and retained evidence

Budget: 9,216 timed generations plus three untimed historical controls = 9,219 total. The controls check each export on historical panel ordinal 0; they do not enter endpoint statistics.

Actual work and campaign/loading/switching costs are preserved in SUMMARY.json and COSTS.json. They are separate from request latency; nested clocks must not be added.

Companion files: PAIRED_REQUESTS.csv, ALL_REPEATS.csv, RAW_REPEAT_ROWS.csv, REPEAT_TOTALS.csv, VALIDITY.json, SUMMARY.json and COSTS.json. Native raw token/trace files remain in the run directory.

No tuning, replacement requests, selective retiming, extra arms or automatic subsequent experiment.

## Acceptance and output accounting

| Metric (one validated trajectory per request) | O58 | K58 | K+58 |
|---|---:|---:|---:|
| Accepted draft tokens | 444,325 | 445,987 | 445,974 |
| Committed output progress | 518,237 | 518,237 | 518,237 |
| Calls per 1,000 output tokens | 144.0239 | 140.8097 | 140.8386 |
| Accepted draft tokens per verification | 5.9413 | 6.0996 | 6.0982 |
| Committed progress per verification | 6.9296 | 7.0878 | 7.0863 |
| EOS / cap-terminated requests | 979 / 45 | 979 / 45 | 979 / 45 |

Output lengths range from 74 to 2,048 tokens (mean 507.0908). Progress counts exclude the initial output token outside the speculative loop, so 519,261 output tokens minus 1,024 initial tokens equals 518,237 committed progress. Accepted-token totals count accepted draft proposals actually committed, clipped at EOS/cap, using each trace’s `accepted_draft_tokens` field; they are not inferred as output length minus verification calls. All ratios above pool summed counts; repeats are not additional samples.

## Completion and cost

All **9,219 planned generations** completed and passed: 9,216 timed executions and three historical controls. Invalid, pending, unknown, unattempted and extra generation counts are all zero. Training, optimizer steps, feedback collection and teacher preparation are zero. Controls reproduced O/K/K+ verification counts **23/23/22**, with exact canonical output and termination. All checkpoints remained unchanged.

The wrapper ran from **2026-11-04 01:48:38 to 05:30:25 UTC**, taking **3 h 41 m 47 s**. Campaign wall time was 13,160.476 s; timed-evaluation wall time was 13,050.616 s. Complete serving across all timed executions summed to 12,051.922 s. These are nested clocks, not additive costs.

Initial loading took 21.508 s and closing 1.215 s. Timed checkpoint publication took 235.168 s across 139 publications, with a separate 9.863 s weight-validation timer. The three controls took 19.210 s wall time, including 6.316 s publication and 11.031 s weight validation. CPU preflight/postflight took 49.625/48.931 s. Component timers do not justify a detailed kernel-time attribution. See [COSTS.json](COSTS.json) for exact fields.

## Reproducibility and review

The execution source is [`12583640baf2210f767e108e3632c0e5f9b8caaa`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/tree/12583640baf2210f767e108e3632c0e5f9b8caaa/DFlash-Online/experiments/mathqa_transfer), with a clean scientific source tree, in [PR #26](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/26). This result publication changes documentation and tables only.

A separate same-team scalar reconstruction reproduced the endpoint, bootstrap, repeats and costs with **450,996 checks**. A second raw-record reconstruction checked **all 663,173 native verification blocks** and 18,466 distinct source files with **4,216,173 checks**, finding zero failures. Both calculations use separate helpers; implementation authorship overlaps are disclosed in their reviews. They validate saved evidence, not an independent GPU replication. The ARS statistical interpretation review covers all 11 fallacy categories. [Endpoint review](reviews/FINAL_REVIEW_VERDICT.md); [interpretation scan](reviews/ENDPOINT_REVIEW.md); [raw validity review](reviews/RAW_RECORD_REVIEW_001.md).

The full private evidence copy contains **27,767 files / 2,029,327,893 bytes**, with exact relative paths, sizes and every SHA-256 verified. It includes all original native/token records, sealed inputs, logs, successful run receipts, failed initial CPU startup and deployment sources. Private responses and raw traces remain outside Git. Public tables retain all paired measurements, regressions and repeats. [Retrieval receipt](reviews/RETRIEVAL_VERIFIED.json); [execution details](EXECUTION_AND_PROVENANCE.md).

This test is finished. No tuning, additional panel, training or subsequent experiment follows from this result.
