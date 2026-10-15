# Retained author-path A1/A2 comparison

Status: code review before GPU execution. This extension adds an evaluation
entry point, not a new training or speculative-decoding mechanism.

The completed author-path experiment compared A2 with the earlier updater and
frozen DFlash. It did not compare A2 with its own A1 on the same future inputs.
This replay measures that incremental effect using the retained model files.
The retired optimizer states and training tensors are not needed.

## Fixed experiment

- Ten completed R1/R2 episodes, original saved contexts for turns 25–32 only.
- Frozen DFlash (`F`), retained author-path first update (`A1`) and second update
  (`A2`), preserving their identities rather than relabeling them as `C1/C2`.
- Three fresh repeats with rotating F/A1/A2 order. All three methods are timed
  together again: 10 × 8 × 3 × 3 = 720 executions over 80 distinct contexts.
- Original models, full saved input IDs, greedy decoding, block size 10,
  256 output tokens or EOS, 31,744 input / 32,768 total context limits and
  8,192-token prefill chunks. No truncation or regenerated conversation history.
- No encoding of a new conversation, support collection, teacher replay,
  optimizer creation, training or export. Existing serving/cache/timing code is
  reused; cache validity is restored when weights change and each scored request
  uses the same cold-prefix policy as the preceding campaign.
- R3/R4 remain reserved. The subsequent WildChat audit is a separate task.

## Evidence bindings

Execution requires explicit hashes for the completed campaign's `MANIFEST.json`
and terminal `STATE.json`, as well as the original result file. The loader checks
the manifest/state relationship, completed episode status, retained result receipt,
dataset and runtime bindings, and the author's connection to the original input
reference. It admits A1/A2 from that completed episode only. The native checkpoint
guard checks the retained model/config bytes before worker use and publication.
No optimizer file is opened or reconstructed.

The original saved request inputs are authoritative. The author result supplies
checkpoint provenance, not a replacement set of input histories. Validation fails
on a missing, altered or foreign binding before native workers start.

## Execution after review

First inspect the metadata-only plan; it launches no GPU worker:

```bash
python experiments/dflash_two_update_followup/run.py plan \
  --comparison author-replay \
  --selection /external/completed-campaign/inputs/selected_prompts.jsonl \
  --streams /external/completed-campaign/inputs/streams.jsonl --out /external/replay-plan
```

After review, use the pinned original runtime configuration and actual recorded
hashes. Runtime artifacts must stay outside this public code checkout:

```bash
python experiments/dflash_two_update_followup/run.py run \
  --comparison author-replay --episode R1_STABLE_CODE \
  --selection /external/completed-campaign/inputs/selected_prompts.jsonl \
  --streams /external/completed-campaign/inputs/streams.jsonl \
  --reference /external/completed-campaign/inputs/references/R1_STABLE_CODE.json.gz \
  --reference-sha256 ORIGINAL_FILE_SHA256 \
  --completed-campaign /external/completed-campaign \
  --campaign-manifest-sha256 CAMPAIGN_MANIFEST_SHA256 \
  --campaign-state-sha256 TERMINAL_STATE_SHA256 \
  --runtime-config /external/completed-campaign/RUNTIME_CONFIG.json \
  --out /external/fresh-replay/R1_STABLE_CODE --allow-gpu-execution
```

Use the same command for each admitted episode with its own bound reference.
The placeholders denote the actual paths already recorded in the completed
campaign manifest; moving those inputs without updating provenance is rejected.
Keep timing runs isolated from other GPU work. Check peak storage and the existing
reserve before launch. This extension retains the existing conservative storage
gate; it does not create checkpoints or relax the reserve to force admission.

## Measurements and delivery

`A2/A1` is primary; `A1/F` and `A2/F` distinguish incremental effects from absolute
benefit versus frozen. Compare generated token IDs with saved references. An
output mismatch is recorded, stops that episode and suppresses the affected
same-generation percentage; it is never silently excluded or repaired by
changing verification. Preserve partial results, failures and all regressions.

`request_s` remains inference with an available checkpoint: prefill, drafting and
target verification within the same observer boundary. Publication, checkpoint
validation, startup and cache maintenance are separate. The worker's internal
activation timer remains nested inside full activation; these are not additive.
No complete-session cost saving is inferred from a checkpoint replay.

Each episode writes exact inputs/output IDs, checkpoint/source bindings, method
order, raw scores, `RESULT.json`, `REPORT.json` and `COMPARISONS.csv`. CSV rows
include each repeat and pooled metrics; aggregate reductions use summed costs
and counts, not the average of per-request or per-episode percentages. Accepted
draft tokens exclude correction/bonus tokens. Verification progress counts
tokens committed during verification; all-output progress also includes the
initial target token. Keep these measures distinct.

Aggregate the ten fresh replay results with the CPU-only reporting command:

```bash
python experiments/dflash_two_update_followup/report.py \
  /external/fresh-replay/R*/RESULT.json --out /external/replay-summary
```

That produces `AGGREGATE_REPORT.json`, `RESULT_BINDINGS.json` and the combined
comparison CSV. The short interpretation must retain the two SHORT_CODE_FIRST
episodes even if unfavorable. A2 can be worse than A1 while still outperforming
frozen. Timing repeats are not independent prompts or new training seeds.

## Validation boundary

CPU tests cover chronology, no-learning replay, real author labels, fresh rotating
orders, immutable inputs, output mismatch retention, checkpoint/file bindings,
separate cost timers and ratio-of-sums reporting. They do not certify native GPU
compatibility of the new retained-checkpoint admission path. No GPU execution is
authorized until the user reviews this change.
