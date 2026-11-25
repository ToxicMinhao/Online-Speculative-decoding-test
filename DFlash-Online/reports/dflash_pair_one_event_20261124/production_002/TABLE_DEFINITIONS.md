# Public serving table definitions

Execution source: `416cb7931028621c09b8fa983e53fc8a7dcebaab`. These tables describe the completed BASE1/PAIR1 one-event comparison. They contain scalar measurements and zero-based public ordinals only; private request/source identifiers, prompt/response tokens, checkpoint paths and GPU identifiers are excluded.

`PER_REQUEST.csv` has 128 rows in held-out manifest order. Each arm contributes one validated count trajectory and the arithmetic mean of its three timing repeats. `*_valid=1` means all three saved generations passed validation; `*_timing_repeats=3`. Counts are not summed across repeats. The `BASE_` and `PAIR_` prefixes identify the completed event-one endpoints.

`ALL_REPEATS.csv` preserves all 792 generations in execution order: 768 scored and 24 warmups. `phase` and `is_warmup` identify their role. Scored rows use `request_ordinal` from 0 to 127 and leave `warmup_ordinal` blank; warmups use `warmup_ordinal` 0 or 1 and leave `request_ordinal` blank. `activation` ranges from 0 to 11, `repeat` from 0 to 2, and `group` from 0 to 1. Each activation has two warmups followed by 64 scored requests. The two arms alternate their first position across repeat/group combinations. Warmups are excluded from the serving contrasts.

| Column or suffix | Definition |
|---|---|
| `prompt_tokens`, `output_tokens` | Prompt and generated-response lengths; output includes the first prefill token. |
| `verification_calls` | Native verification-block count. |
| `stop_clipped_accepted_draft_tokens` | Sum of the matching draft prefix clipped to the tokens actually committed in each block; the raw field is `accepted_draft_tokens`. |
| `native_accepted_draft_tokens` | Sum of native committed draft proposals, excluding the one target/bonus token per verification call; equals `committed_progress - verification_calls`. |
| `committed_progress` | New tokens committed by verification; excludes the first prefill token and equals `output_tokens - 1`. |
| `finish_reason`, `cap_hit` | Native `eos` or `length` termination and the output-cap indicator. |
| `request_latency_s` | Measured complete-request latency, in seconds. |
| `native_generate_s` | Native generation time within the request, in seconds. |
| `cache_maintenance_s` | Separately recorded cache-maintenance time, in seconds. These timing scopes are not additive. |
| `*_mean` | Arithmetic mean of that timing field across the three repeats. |
| `valid` | 1 when the saved generation passed the canonical-output and native-measurement checks. All 792 rows are valid. |

`SERVING_SUMMARY.json` contains aggregate serving totals, contrasts, confidence intervals and the bootstrap recipe. Totals use the 128 request means for timing and one count trajectory per request. `repeat_totals` provides each scored timing repeat separately. Native accepted proposals/call, stop-clipped accepted tokens/call and committed progress/call remain distinct measures.

The primary reduction is `100 * (1 - sum(PAIR request means) / sum(BASE request means))`. Confidence intervals use 10,000 paired whole-request draws from PCG64 seed 20261124 and linear 2.5/97.5 percentiles. `bootstrap.input_order_ordinals` gives the exact public-ordinal order needed to reproduce the audited draws; it differs from manifest order because the original analysis sorted private request IDs. Reorder the CSV rows by this numeric list before indexing the draws. `draw_sha256` hashes the resulting 10,000-by-128 little-endian int64 draw matrix. No private ordered IDs are published.

The exported CSVs were read back independently. Their row coverage, count identities, repeat means, per-arm totals and all four bootstrap intervals reproduce the audited summary within 1e-12 absolute/relative floating tolerance. Call wins/regressions/ties are 50/43/35, with 19 net calls saved. The primary latency reduction is 0.14997143%, with 95% interval −0.41100502% to 0.69737390%.

These are serving-only measurements with each endpoint already available. Adaptation costs are reported separately. Timing repeats do not create independent training replications; the small primary contrast and interval spanning zero do not establish a held-out latency benefit.
