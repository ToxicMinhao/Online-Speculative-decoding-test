# O58 / K58 / K+58 MathQA serving transfer

## Material Passport

ARS experiment-agent run → validate. Frozen GSM8K-trained checkpoints on new MathQA requests. Saved-record audit; not an independent model reproduction.

Execution commit: `12583640baf2210f767e108e3632c0e5f9b8caaa`.

Status: **valid_complete**. Zero training, updates, feedback capture or teacher preparation.

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
