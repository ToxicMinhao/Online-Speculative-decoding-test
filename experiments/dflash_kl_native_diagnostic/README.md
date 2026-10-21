# DFlash KL-to-native diagnostic

This bounded diagnostic fills the two missing measurements from the completed
six-group fixed-feedback learning-curve screen. It asks where lower measured
KL stops translating into fewer native verification calls. It implements no
new adaptation mechanism and performs **zero optimizer steps**.

Implementation is submitted for review. **GPU execution remains on review
hold.** PR #8/agreement filtering and the correction-bank campaigns stay paused.
CPU tests qualify the controller and report logic, not native output equality,
GPU memory fit or diagnostic efficacy.

## Fixed scope and provenance

| Population | F/S8 fixed-anchor KL | F/S8 native decoding |
|---|---|---|
| 24 source responses | Reuse original records | Retained F; one new natural S8 generation per source prompt |
| 24 held-out responses | Two new loss evaluations per saved response | Reuse original F/S8 records |

The budget is **24 new source generations, 48 response-level held-out KL
evaluations, and no training**. Teacher preparation, projection chunks,
checkpoint loading, initialization and any explicitly recorded warmups are
additional operations. Forty-eight response loss evaluations are not forty-eight
total model forwards. S2, new prompts, resampling, tuning, extra timing repeats
and replacement mechanisms are outside scope.

The original baseline is pinned to commit
`165e25b7a85cabfe16f7b9d642fb575b23d82ad6`. The sealed manifest file SHA256 is
`1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d`.
Record the new diagnostic commit and source digest separately. Each group uses
its **own** retained S8 checkpoint, physically named `checkpoints/U2`, from:

```text
/data/sources/minhao/EAGLE_SPEC/research/dflash_baseline_learning_curve_20261020/execution_001/attempt_001/native/<group>/checkpoints/U2
```

Identity checks use retained manifest/runtime bindings, checkpoint metadata,
exact prompt/output IDs, source-loss records, frozen source trajectories and
held-out native records. A summary table is insufficient evidence. Missing
artifacts are reported; checkpoints, source answers and original measurements
are never replaced through training or regeneration. Diagnostic outputs live in
a new directory, with existing artifacts preserved. No optimizer is created or
loaded, no checkpoint is exported, and no bulk checkpoint copying is needed.

## Native source replay

Use the existing frozen Llama-3.1-8B-Instruct target and DFlash model architecture,
tokenizer, pinned native runtime, BF16 precision, full context limit and cache
policy. For each original source prompt, S8 generates naturally with concurrency
1, greedy decoding, block size 10 and 1,024 output tokens or EOS. The saved
answer is a validation reference, never a forced decoding sequence.

Require exact output-token equality with the retained frozen answer, including
termination and cap status. Preserve raw mismatches. Record calls, accepted
draft tokens, verification-committed progress, output length and round offsets.
Accepted draft tokens exclude correction/bonus tokens; verification-committed
progress includes them but excludes the initial target-prefill token. Ensure
valid caches after each checkpoint switch and independent request.

**Historical F timing and fresh S8 timing do not support a speedup comparison.**
Fresh elapsed times are retained only as diagnostic cost evidence. No additional
timing campaign is requested.

## Paired held-out loss evaluation

Both checkpoints evaluate the same saved canonical answer. Dataset answers and
freshly generated held-out responses are unused. The read-only loss path retains
the qualified author implementation's forward KL at temperature 1, teacher
conditioning, anchor sampling, position weighting, padding and per-response
normalization. Prompt positions remain context; only response positions receive
supervision. Run in evaluation mode under `torch.no_grad()` with frozen weights.
Held-out records are not relabeled as training feedback, and existing training
guards remain unchanged.

Use a separate deterministic anchor RNG derived from seed 17 and immutable
request identity, independent of checkpoint. F/S8 receive identical anchors,
masks, weights, positions, exposure and denominators. Save and check them,
without perturbing training/global RNG or resampling an inconvenient result.
Zero-supervision cases remain explicit and are excluded from loss-improvement
claims; their valid native measurements remain reportable.

## Correctness, execution and reporting

The first complete group is one of the original six. Its gate checks evidence
identity, checkpoint binding, exact native outputs, paired anchor/exposure
equality, read-only execution and finite loss. If correctness passes, the runner
continues through all five remaining groups automatically. Null or harmful
adaptation is valid evidence, never a reason to retune or halt. Identity/output/
anchor mismatches or nonfinite losses stop the affected run and preserve all
completed evidence. There is no automatic repair, substitution or restart.

From the repository root, inspect and test without model execution:

```bash
python -m pytest -q experiments/dflash_kl_native_diagnostic
python -m experiments.dflash_kl_native_diagnostic.run plan \
  --baseline-root /absolute/private/baseline/attempt_001 \
  --out /absolute/private/diagnostic_plan
```

After review and explicit execution authorization, use the qualified environment
and reviewed source digest:

```bash
python -m experiments.dflash_kl_native_diagnostic.run run \
  --baseline-root /absolute/private/baseline/attempt_001 \
  --reviewed-code-sha256 REVIEWED_DIAGNOSTIC_SOURCE_SHA256 \
  --allow-gpu-execution \
  --out /absolute/private/fresh_diagnostic_run
```

Reports can be rebuilt from retained results without model execution:

```bash
python -m experiments.dflash_kl_native_diagnostic.run report \
  --result /absolute/private/diagnostic/RESULT.json \
  --out /absolute/private/diagnostic_report
```

The private output contains the raw paired records, exact tokens/trajectories,
loss and sampling evidence, identity bindings, configuration, costs and failures.
`REPORT.md`, `SUMMARY.json`, `PAIRED_REQUESTS.csv`, `POOLED.csv` and `COSTS.csv`
separate source from held-out results at request, group, domain and overall
levels. Report code reruns the shared validation checks; a rejected execution
row cannot reenter gain calculations through a stale `passed` flag. Every
request, regression, zero-supervision case and incomplete group stays visible.

Each already-normalized response loss gets equal weight. Call reductions are
ratios of summed paired counts, with better/worse/tied counts. Source and
held-out responses are never combined into a single pooled result. Teacher
preparation, loss forwards, loading, native diagnostic generation and other
costs remain separate; nested timers are not blindly summed.

**Better native drafting means fewer verification calls for identical output
tokens in this diagnostic. It does not yet establish a serving-time improvement.**
Source decoding gains with poor held-out loss/decoding point toward transfer;
lower held-out KL with worse decoding points toward a measured KL/native
mismatch; source KL gains without source decoding gains leave useful drafting
unproven even on source responses. These patterns identify where to investigate
next, **not the underlying cause**. They do not establish that forward KL is
generally unsuitable, nor justify a broader mechanism or campaign.

Keep private responses, raw runtime artifacts, model/optimizer files and
credentials outside Git. Only task code, tests and documentation are uploaded.
