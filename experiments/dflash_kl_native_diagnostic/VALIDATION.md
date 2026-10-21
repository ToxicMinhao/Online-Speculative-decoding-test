# Implementation validation — 2026-10-20

Status: code ready for review; native GPU execution has not run.

## Local checks

The diagnostic and inherited baseline package pass **169 CPU tests**:

```bash
python -m pytest -q \
  experiments/dflash_kl_native_diagnostic \
  experiments/dflash_baseline_learning_curve
```

Local environment: Python 3.10.18, Torch 2.11.0+cu130. Tests use small CPU
fixtures, including BF16 loss-parity fixtures. Torch emits a host CUDA detection
warning (error 804); no GPU/model experiment was performed. The local Torch
version does not replace the pinned execution environment.

Coverage includes exact qualified loss parity on synthetic inputs; read-only
evaluation and RNG restoration; fixed paired anchors and response-only masks;
short/zero-supervision handling; checkpoint and evidence binding; source-output
and termination validation; durable partial failure records; controller counts
and continuation after unfavorable results; worker shutdown; and report
revalidation, exclusions and aggregation. Syntax and whitespace checks pass.

A CPU-only plan also validates the actual retained compact evidence: all six
groups, the sealed manifest, 72 original native trajectory bindings and all 48
F/S8 source-loss records. It does not load or qualify GPU checkpoint tensors.
Private responses, tokens, trajectories and plan outputs remain outside Git.

## Remaining execution checks

After review, the first complete group must qualify the installed native
integration, physical checkpoint bindings, GPU memory fit, natural source-output
preservation and paired held-out loss evidence. No performance improvement is
required. If correctness passes, the remaining five groups continue unchanged.
Identity/output/anchor mismatches and nonfinite losses stop execution while
preserving evidence.

The planned scope remains 24 new S8 source generations and 48 response-level
held-out KL evaluations, with zero optimizer steps and no extra timing campaign.
Better native drafting means fewer calls on identical output tokens, not an
established serving-time gain. Diagnostic patterns guide further investigation;
they do not establish a cause or general unsuitability of forward KL.
