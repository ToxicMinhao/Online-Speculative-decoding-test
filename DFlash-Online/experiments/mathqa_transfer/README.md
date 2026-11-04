# O58 / K58 / K+58 MathQA serving transfer

This experiment evaluates three unchanged GSM8K-trained exports on 1,024 MathQA
test requests. It uses the existing DFlash-Online SGLang/FA3 backend and native
trace validator. It constructs no optimizer and performs no training, feedback
capture, target-feature preparation or generation warm-up.

## Fixed protocol

- Authenticate O58 from PR23 and K58/K+58 from PR25, including historical export
  byte hashes, tokenizer/model pins and the installed native runtime.
- Verify the specified MathQA archive Git blob; render only `Problem` and
  `options` with the pinned reasoning-off chat wrapper.
- Filter malformed/long inputs, duplicates and available prior input-inventory
  overlaps in source order. Select the first 1,024 eligible rows ordered by
  SHA256(`mathqa-transfer-v1:` + decimal zero-based row ID). Preserve every
  exclusion, scanned-inventory identity and exact rendered token sequence.
- Run one historical ordinal-zero control per arm, then 9,216 timed generations:
  16 request blocks, 64 requests per block, three arms and three repeats. The
  parity/left-rotation schedule is fixed in `contract.py`.
- Keep BF16 greedy block16, output2,048/EOS, context4,096, C1 and full per-request
  cache flushing. Loading and publication remain outside complete-request time;
  cache maintenance remains inside it. No concurrent training or transfers.
- Bind each new request to O repeat0. Earlier K/K+ executions remain pending
  until that reference is available. Every native trace, token sequence,
  termination, repeated count and timer must validate. No row dropping or retry.

## Entry points

`data.py` prepares the input-only panel and historical-overlap inventory.
`seal.py` rechecks rendering on the deployed pinned tokenizer and seals the clean
published source, exact inputs, authenticated exports, schedule and analysis.
`preflight.py` checks identities, runtime, idle hardware and the full worst-case
storage budget before native work; it verifies unchanged artifacts afterward.
`campaign.py` runs three controls and the exact timed schedule using one native
session. `report.py` reconstructs controls and endpoint validity from original
raw/native records before producing any gain claim. `run_campaign.sh` supplies
the qualified environment and preserves exit receipts; it never retries.

All private prompts, canonical answers, native traces and source archive bytes
remain outside Git. The public report records the execution commit and aggregate
preparation/audit evidence. Serving failure records and unattempted work remain
visible. The 12-hour runner ceiling is an operational safety limit, not an
efficacy gate or expected duration.

## Analysis

Complete-request latency is primary. Average each request's three measurements,
then compare summed paired means. K+ versus O is tested first; K+ versus K is
confirmatory only when the primary paired 95% interval lies above zero. Both
estimates are always reported. The same 10,000 paired request resamples use
PCG64 seed20261103 and linear percentile intervals. Timing repeats are not
independent requests. Verification counts use one repeat-consistent trajectory
per request; component times and call intervals are descriptive.

The result establishes only serving transfer for these particular fixed
checkpoints. It does not establish MathQA adaptation, training-seed robustness,
training-inclusive speedup, or a causal explanation. Historical GSM8K latency
does not enter the new comparison.

## CPU checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:training:inference:tests \
  python -m unittest discover -s tests -p 'test_mathqa*.py' -v
```

Tests cover source authentication, input selection, history matching, balanced
order, deferred canonical validation, native traces, controls, failure/work
accounting, reserve admission, unchanged exports, shared paired resampling,
ordered interpretation, and rejection of incomplete or invalid reports.
