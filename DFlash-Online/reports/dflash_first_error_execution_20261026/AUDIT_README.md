# Local result audit

Prepared October 26, 2026 under the ARS experiment-agent validation workflow.
Status: **auditor tested; campaign result not assessed by this preparation step**.
This is compact-evidence analysis, not a separate GPU reproduction.

`audit_results.py` uses only the Python standard library and the separately
hashed retained native-trace auditor. It does not load tensors or models, access
remote paths, launch jobs, or generate responses. It is outside the campaign's
sealed execution source. All 10 sealed execution files remained byte-identical
after preparation.

Run only after retrieving the completed campaign's compact files and the
supervisor/preflight/postflight receipts. For the expected retrieval layout:

```bash
python3 /home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/audit_results.py \
  --campaign /home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/raw/campaign_001 \
  --manifest /home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/sealed/MANIFEST.json \
  --ordinary-receipt /home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/final_review/raw/campaign_001/trainer/events/event_001/RESULT.json \
  --ordinary-measurements /home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/final_review/raw/campaign_001/evaluation/MEASUREMENTS.jsonl \
  --seal /home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/SEAL.json \
  --postflight /home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/raw/POSTFLIGHT.json \
  --output /home/minhao2/EAGLE_SPEC/research/dflash_first_error_execution_20261026/FINAL_AUDIT.json
```

Missing, partial or malformed evidence produces a failed receipt and exit code
1. Passed preservation alone is insufficient: the supervisor must also exit 0,
all three training events must complete, and all 768 evaluation slots must pass.
Valid null and harmful results pass. No performance win is a correctness gate.

The audit checks:

- Exactly seven discarded qualification windows, 14 presentations/backwards,
  seven temporary steps, 30 bounded logit VJPs, zero reference forwards and zero
  generations. It binds durable work and oracle records to each trial, reads 22
  exact comparison receipts, and verifies the compact evidence hashes for all
  35 retired qualification tensors.
- Fresh C/K/A initialization, 128 retained features and the exact two-pass,
  256-presentation, 128-step schedule per arm. It compares every anchor and
  learning-rate row against O, reconstructs available proposal counts/decay
  weights, checks first-error histogram/count bounds and response denominators,
  and sums diagnostics only at optimizer boundaries.
- Exactly 768 ordered O/C/K/A slots over 64 requests and three rotating repeats.
  It reconstructs native longest-prefix acceptance, commits and termination;
  requires canonical tokens and retained O counts; validates publications,
  weights and empty caches; independently recalculates all four comparisons;
  and checks JSON/CSV agreement and historical F counts on the same valid cohort.
- Separate qualification, loading, initialization, training, checkpoint/export,
  evaluation and total costs, without adding nested timers twice. It checks
  retained-input/source bindings and preflight/postflight preservation coverage.

The compact audit cannot reconstruct unsaved main-training logits. It checks
their logged accounting and uses the separately retained qualification oracle
for current-mask, arithmetic and gradient correctness. It reads exact tensor
comparison receipts without rerunning those tensor comparisons. It makes no
claim that retained O's entire historical 128-step trajectory was reproduced.
F supplies historical counts and canonical outputs, never fresh latency.

Tests: **11 passed**, 18.020 seconds, CUDA uninitialized. See
`AUDIT_CPU_TESTS.json` and `AUDIT_CPU_TESTS.txt`. Temporary test fixtures include
a synthetic 768-slot panel made from retained native traces, valid negative/null
outcomes, canonical corruption, missing/duplicate/fresh-F slots, altered anchors,
double-counted diagnostics, pooled ratios, nested costs, nonzero supervisor exit,
missing input, and actual tiny CPU seven-window qualification receipts. Those
fixtures are not campaign measurements and are removed after testing.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
