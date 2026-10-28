# Persistent Qwen/GSM8K OSD rerun — completed results

## Material Passport

ARS experiment-agent monitoring and retained-record validation; October 24, 2026.
The authorized campaign completed with exit 0 at 09:51 Edmonton time. No new
training, generations, tuning or campaign restart was performed for this review.
Stage 1 / project None remains unchanged.

**Final OSD reduces held-out request time by 11.33% and verification calls by
11.57% versus released DFlash, with identical output tokens.** This is performance
with the checkpoint already available; adaptation costs are reported separately.

## Completed protocol

Qwen3-8B target and released Qwen3-8B-DFlash-b16; A800 DFlash-Online,
SGLang/FA3, greedy block 16, concurrency one. Output cap 2,048 tokens or EOS;
training sequences retain the full prompt and answer up to 2,272 tokens.

The run served 7,473 fresh GSM8K training-stream answers in the retained request
order. The first 7,424 answers supplied 58 updates of 128 responses; the last 49
were served without another update. Weights, optimizer and RNG persisted across
events. Two shuffled passes, batch one and accumulation two produced exactly
7,424 optimizer steps and 14,848 response presentations. Forward KL, constant
3e-4, direct BF16 AdamW with decay .01, current anchor eligibility/cap512, Flex
attention and response-only supervision remained fixed. These are independent
questions with persistent learning, not concatenated multi-turn conversations.

After training stopped, released F and final OSD O were freshly benchmarked on
the original 1,024 held-out questions, with three capture-disabled repeats and
rotating method order. Held-out answers never entered training. The final paired
comparison does not assess every intermediate update individually.

## Held-out performance

| Metric | Released DFlash | Final OSD | Change |
|---|---:|---:|---:|
| Complete request time for 1,024 questions, mean across repeats | 860.21 s | 762.76 s | **11.33% less** |
| Verification calls per repeat | 50,584 | 44,730 | **11.57% fewer** |
| Calls per 1,000 output tokens | 160.66 | 142.07 | **11.57% fewer** |
| Committed progress per verification | 6.204 | 7.016 | +0.812 tokens/call |
| Accepted draft tokens per repeat | 264,170 | 270,037 | +5,867 |

Each repeat produced the same 314,847 output tokens per method. Progress excludes
the initial prefill-produced token; it is not the same as accepted draft tokens.
All ratios pool summed measurements. Three repeats are measurements of 1,024
questions, not 3,072 independent questions.

| Timing repeat | Frozen time | OSD time | Time reduction |
|---|---:|---:|---:|
| 1 | 860.34 s | 763.04 s | 11.31% |
| 2 | 860.27 s | 762.47 s | 11.37% |
| 3 | 860.02 s | 762.76 s | 11.31% |

Verification counts repeat exactly. Of 1,024 questions, **952 improve, 44 worsen
and 28 tie** in verification calls. Mean request latency improves on 963 and
worsens on 61; these are descriptive signs, not separate significance tests.
Three questions hit the output cap in both methods. No invalid pair was removed:
all 3,072 request-repeat pairs passed canonical input/output equality.

## Costs

| Measured span | Seconds | Hours |
|---|---:|---:|
| Production serving, capture and buffer persistence | 7,490.21 | 2.08 |
| All 58 complete blocking updates | 20,421.37 | 5.67 |
| Serving plus updates, counted once | **27,911.58** | **7.75** |
| Actual online phase, including loading and diagnostic overhead | 28,209.72 | 7.84 |
| Final benchmark wall time, including checkpoint switches/persistence | 10,356.95 | 2.88 |
| Actual complete campaign | **38,567.30** | **10.71** |

The following components are already inside the blocking-update span; do not add
them again: teacher preparation 287.25 s; feature persistence 326.17 s; training
17,732.05 s; trainer-state persistence 1,025.47 s; export 692.08 s; complete
activation 99.40 s. Internal native activation (18.24 s) is nested inside complete
activation. Capture is already charged in serving. Loading, qualification,
publication validation and retention remain separate in the raw status.

The serving table excludes these preparation costs. No frozen counterpart was
run over the entire 7,473-question training stream, so this experiment does not
establish a complete online-episode speedup or a measured amortization horizon.
Aggregate timers do not establish a kernel-level breakdown.

## Correctness and retained evidence

Independent CSV aggregation reproduces the saved totals. The raw-record audit
checks all 6,144 timed records against the sealed prompt IDs and canonical fresh
F answers, including EOS/cap termination, capture-disabled status, final/released
checkpoint identities, rotating order and all 285,942 verification blocks.
Committed blocks reconstruct every answer. All 58 publication receipts report
exact equality for 58 exported tensors in their native fused representation.

Training records confirm two passes over each event's exact 128 request IDs,
constant learning rate, the optimizer-step counts and finite metrics. They record
3,728,050 anchor presentations and 54,368,391 supervised-position exposures.
Exposure sums use optimizer-boundary rows, which already include both responses
in the accumulation window. Repeated positions are exposures, not unique tokens.
The audit did not reload optimizer tensors or rerun model execution.

All 16,941 selected evidence files were copied to Fenrir in a verified archive.
Full raw runtime evidence, the final export and current restart state remain on
A800. Qualified rolling retention retired completed-event feature tensors and
superseded exports; its receipts describe lost direct replay capability. Deleted
tensors are not archived. No additional deletion occurred during this review.
At inspection both A800 GPUs were idle and approximately 53.3 GB remained free.

## What this establishes

The implemented OSD recipe can learn useful held-out drafting improvements under
this sustained GSM8K setup. The gain is broad across these questions and stable
across three timing repeats. Earlier short-buffer negative results are still
valid for their own protocols: multiple settings and learning exposure differ,
so this comparison cannot assign the recovery to learning rate, buffer size,
precision or any other individual change. It does not establish improvements on
CodeAlpaca, ShareGPT or a new mechanism, nor exact reproduction of the historical
author run. No mechanism or follow-up experiment is automatically selected.

## Files

- [Every timed measurement](status_check_20261024/evaluation/PER_REQUEST.csv)
- [Pooled summary](status_check_20261024/evaluation/SUMMARY.json)
- [Per-event accounting](status_check_20261024/EVENTS.csv)
- [Cost summary](status_check_20261024/COST_SUMMARY.json)
- [Independent metric audit](status_check_20261024/INDEPENDENT_AUDIT.md)
- [Raw-record audit](status_check_20261024/RAW_AUDIT.json)
- [Verified evidence transfer](status_check_20261024/TRANSFER_VERIFIED.json)
- [Sealed configuration](sealed/config.yaml)
- [Original scope and preparation](REPORT.md)


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_rerun_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
