# DFlash lower-learning-rate control

This implements one prespecified control, **L8: eight AdamW steps at `3e-5`**, on
all six original four-response groups. It tests whether a smaller update retains
source learning while improving held-out native drafting. It is not a learning-rate
sweep, an additional mechanism or a restart of agreement-filter/correction-bank work.
The user has authorized this bounded execution: validate the first complete group,
then continue the remaining five on correctness, irrespective of benefit or harm.

Both **S8 and L8 originally initialize from released DFlash with a fresh optimizer**
and seed 17. S8 is now a retained eight-step `3e-4` comparator; only L8 is trained
and evaluated anew. F is the released draft.

## Fixed data, training and execution scope

The original baseline is PR9 commit
`165e25b7a85cabfe16f7b9d642fb575b23d82ad6`; the completed KL/native diagnostic is
PR10 commit `7b1f920bce2c8566b6315606bb3ffd36209aba3f`. The sealed sample manifest is
`1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d`.
The new commit, complete source digest, configuration, original checkpoint bindings,
exact inputs/outputs and training presentation identities are recorded separately.
Missing evidence is not regenerated or replaced.

Each group uses its four original frozen-generated responses, in the original order,
for four passes. Microbatch 1 and accumulation 2 average two independently normalized
response losses before clipping and each optimizer step. **Only the current response
receives supervision. Its own prompt supplies context. Held-out responses never enter
training.** No other response or group enters that example's context.

Only learning rate changes. Preserve the qualified frozen Llama-3.1-8B-Instruct
target, UltraChat DFlash, tokenizer, BF16 parameters, frozen embedding/vocabulary
head, trainable backbone, forward KL at temperature 1, target conditioning,
position weighting and normalization. AdamW remains betas `(0.9, 0.999)`, epsilon
`1e-8`, weight decay `0.01`, gradient clipping `1.0`. Record the effective learning
rate at every step. Do not resume S8 or load old optimizer state.

The new `0→8` control path does not fabricate completed F/S2 diagnostics or weaken
the historical trainer's guards. Export one final, genuinely labeled **L8**
checkpoint per group in a new directory. Do not alias it as S8/U2. Intermediate
checkpoints, optimizer-state exports and FP32 master-weight changes are outside scope.

| New work | Per group | Six groups |
|---|---:|---:|
| Optimizer steps | 8 | 48 |
| Response presentations | 16 | 96 |
| Natural native generations | 8 | 48 |
| Response-level KL evaluations | 8 | 48 |
| Explicit generation warmups | 0 | 0 |

No F/S8 reruns, new feedback, S2 evaluation, additional prompts, tuning or timing
campaign. Teacher preparation and projection chunks are additional operations,
not additional response-level measurements. Training terminates before native evaluation.

## Three distinct anchor contracts

1. **Training:** match the original 16 presentation identities/order, normal
   per-presentation sampling trajectory, masks, weights and exposure.
2. **Source KL:** use PR9's original fixed source-diagnostic anchors.
3. **Held-out KL:** use PR10's retained deterministic diagnostic anchors.

Diagnostic anchors do not replace the training sampling trajectory. F/S8/L8 KL
comparisons must match anchors, weights, masks, supervised positions and denominators.
Read-only KL uses frozen evaluation mode and no gradients, preserves training RNG
and leaves held-out roles unchanged. Zero-supervision cases are retained and excluded
from loss-improvement claims, without resampling.

## Native evaluation and correctness

L8 generates naturally from each exact saved prompt: concurrency 1, greedy decoding,
block size 10, 1,024 tokens or EOS, with the pinned runtime, full context and cache
policy. The saved canonical answer is only a reference for exact output-token and
termination equality; it is never forced into generation. Disable feedback capture.
Record verification calls, accepted draft tokens, committed/emitted progress,
lengths, EOS/cap status and per-round offsets. Ensure caches match each checkpoint.

Validate the first complete original group, then continue automatically through the
remaining five if correctness passes. Null or harmful adaptation results continue
unchanged. Identity, output, training/diagnostic-anchor or nonfinite failures stop
the affected run and preserve completed/partial evidence. GPU execution, memory fit
and exact output equality require actual runtime qualification; CPU tests alone do
not establish these properties.

Preflight all six final checkpoints, temporary writes, compact evidence and the
existing reserve. Keep all original F/S8 artifacts intact. No automatic offload or
deletion is implied by this package.

## Reporting and interpretation

`write_reports` writes `REPORT.md`, `SUMMARY.json`, `PAIRED_REQUESTS.csv`,
`POOLED.csv`, `COSTS.csv` and `TRAINING.csv`. Show F/S8/L8 together with L8-versus-F
and L8-versus-S8 contrasts; retain S8-versus-F context. Report source and held-out
populations separately, at request/group/domain/overall levels. Mean KL equally
weights each already-normalized response loss. Call reductions use summed paired
counts, retaining better/worse/tied counts and every invalid/zero-supervision case.
The report reruns the shared validator and never re-admits an execution-rejected
comparison.

Costs preserve teacher preparation, training, evaluation, export/loading, activation,
cache work and peak memory separately. Some timers are nested; they must not be
blindly summed. Fresh L8 times and historical F/S8 times are **not a speedup
comparison**. Better native drafting here means fewer verification calls for identical
output tokens, not an established serving-time benefit.

- Source gains plus fewer held-out calls than F support transfer on these requests.
- Beating S8 while remaining worse than F means reduced harm, not successful transfer.
- Source learning without held-out benefit leaves transfer unresolved. Feedback
  coverage is a possible next question, not an identified cause.
- Negligible source learning makes the control inconclusive. No success or
  negligibility threshold is invented after inspecting the results.

This reused diagnostic set and single seed cannot establish broad generalization;
a later selected method needs fresh held-out confirmation. Private responses, raw
runtime artifacts and model/optimizer tensors stay outside Git.

## Commands

From the repository root, test and inspect the pinned evidence without model execution:

```bash
python -m pytest -q experiments/dflash_lower_lr_control
python -m experiments.dflash_lower_lr_control.run plan \
  --baseline-root /absolute/private/baseline/attempt_001 \
  --diagnostic-root /absolute/private/kl_native/attempt_001 \
  --out /absolute/private/lower_lr_plan
```

Use a new output directory for the authorized bounded execution, with the code
digest recorded by the implementation preflight:

```bash
python -m experiments.dflash_lower_lr_control.run run \
  --baseline-root /absolute/private/baseline/attempt_001 \
  --diagnostic-root /absolute/private/kl_native/attempt_001 \
  --out /absolute/private/lower_lr_run \
  --allow-gpu-execution \
  --approved-code-sha256 PINNED_IMPLEMENTATION_SHA256
```

These are path placeholders. Runtime/model/source bindings, storage admission and
owned-worker cleanup checks apply before execution; the entry point does not
authorize a broader campaign or silently overwrite earlier evidence.
