# DFlash feedback-allocation control (P8)

This bounded control asks whether allocating the same **per-model** presentation
and optimizer-step budget to more distinct saved feedback improves held-out
drafting. It changes feedback allocation; it introduces no new loss or mechanism.
This PR contains implementation and CPU validation for review. GPU qualification
and the campaign have not run. Earlier paused campaigns remain paused.

| Arm | Source responses per model | Passes | Presentations | Steps | LR |
|---|---:|---:|---:|---:|---:|
| Retained S8 | 4 from one original group | 4 | 16 | 8 | `3e-4` |
| New P8 | 8 from the two groups of one dataset | 2 | 16 | 8 | `3e-4` |

There are three new P8 trajectories (GSM8K, CodeAlpaca, ShareGPT), compared with
six original-group S8 trajectories. This is not six new independent replicates,
a pure diversity test, or an equal-compute comparison: repetition, pairing/order,
and potentially supervised token work also change. Selection on this reused
held-out set and single seed still requires fresh confirmation.

## Evidence, ownership and ordering

Reuse the original manifest
`1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d`.
The unchanged PR10/PR11 evidence readers admit the exact original baseline and
diagnostic commits and file hashes. These bindings, together with the new source
digest and implementation commit, are recorded in the plan's source manifest.

A and B are the two groups in original manifest order. Pool the four original
frozen-generated source responses from each group in
`A0, B0, A1, B1, A2, B2, A3, B3` order, twice, without shuffling. Each accumulation
pair therefore contains one response from each original group. Never combine
datasets, use held-out training feedback, or regenerate source answers.

`pool_id` owns the new model; original dataset/group/request/role IDs remain
intact. JSON-encoded composite identities prevent collisions from reused local
turn numbers. Pool seals bind exact tokens, original metadata, checkpoint hashes,
diagnostic anchors and original F/S8 measurements. Each request retains its own
original-group S8 comparator. Missing or modified evidence blocks execution.

## Training and the three anchor contracts

Each P8 starts independently from released UltraChat DFlash, fresh AdamW and
seed 17. Keep the qualified Llama-3.1-8B-Instruct target, tokenizer, BF16 backbone
training path, frozen target/embedding/vocabulary head, forward KL at temperature
1, target conditioning, position weighting and loss normalization. AdamW uses
`lr=3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, clipping `1`.

Microbatch 1 and accumulation 2 **average independently normalized response
losses**, then clip the aggregate gradient once and step. Every response uses
its own original prompt as context; only its own answer receives supervision.
Held-out answers never enter training.

1. Training uses a fresh normal seed-17 sampling trajectory for the new response
   order. It does not match S8's training anchors or substitute diagnostic anchors.
2. Source KL reuses each response's original PR9 fixed diagnostic anchors.
3. Held-out KL reuses each response's original PR10 diagnostic anchors.

Diagnostics accept no optimizer, run under `eval`/`no_grad`, and restore RNG,
parameter flags and module modes. Compare masks, weights, supervised positions,
denominators and RNG/anchor hashes with the retained reference. Record and exclude
zero-supervision losses without resampling. Native comparisons remain separate.

`PooledFeedbackTrainer.train_eight` is an explicit path. The old four-response
and single-group guards, source diagnostics, and qualified loss are unchanged.
Only the final checkpoint is exported as `checkpoints/P8`, with a real P8 identity
and dataset ownership. No F/S2 diagnostic fabrication, S8/L8/U2 alias, old
optimizer loading, intermediate checkpoint or optimizer-state export is allowed.

## Bounded evaluation and serving contract

| New work | Per dataset | Total |
|---|---:|---:|
| Optimizer steps | 8 | 24 |
| Response presentations | 16 | 48 |
| Natural P8 generations | 16 | 48 |
| Response-level P8 KL evaluations | 16 | 48 |
| Explicit generation warmups | 0 | 0 |

The eight source and eight held-out requests per dataset each receive one new P8
native generation and one fixed-response KL evaluation. No new F/S8/L8 generation,
extra block probe, repeated timing campaign or regenerated feedback is included.
Teacher preparation, projection chunks, loading and export are additional recorded
operations, not additional response-level observations.

Training and its child processes must terminate before timed native evaluation.
Use concurrency 1, greedy sampling, block 10, 1,024 output tokens or EOS, exact
pinned tokenizer/runtime, and the qualified cold-cache policy at each request and
checkpoint switch. Context is 32,768 and prefill chunks are 8,192. The inherited
nominal input limit is 31,744; the unchanged runtime also reserves nine proposal
positions, so its effective prompt bound is 31,735. Never truncate history.
Feedback capture is disabled. Generate naturally from saved prompts; canonical
answers are equality references, not forced generation tokens.

Require exact output-token and termination equality against retained F. Preserve
per-round offsets, accepted draft tokens (excluding correction/bonus tokens),
committed/emitted progress, lengths, EOS and cap hits. Raw receipts precede
validation; invalid rows cannot re-enter gain calculations. Unresolved worker
work remains unknown rather than being classified as unattempted.

The first complete dataset pool is the correctness qualification **inside this
budget**. On correctness, finish the other two unchanged even for null or harmful
results. Identity, output, anchor, accounting, nonfinite or cleanup failures stop
and retain partial evidence. There are no automatic retries.

## Storage and review-gated execution

Preserve original evidence. Before training, verify the actual artifact destination
and peak final/staging/evidence writes plus the existing 8 GiB reserve. The
18.05 GB handoff estimate omitted repeated full-pool configurations and controller
evidence copies. Sizing the actual retained artifacts gives a revised conservative
requirement of **25.83 GB free on one filesystem**, including the 8 GiB reserve.
Separate destinations require **19.51 GB for native artifacts and 14.91 GB for
controller evidence**, each including its own reserve. This includes one controller
retrieval-staging copy and at most one checkpoint export/retrieval overlap.
The implementation records the derived budget and actual destination free space;
destinations on the same filesystem share one summed budget, not duplicated free
space. It refuses insufficient capacity before training. Additional simultaneous
archives need a new budget. This package does not delete or relocate historical files.
A separate fresh `--native-root` can select an adequately provisioned filesystem.

CPU-only admission and sealing:

```bash
python -m experiments.dflash_feedback_allocation_control.run plan \
  --baseline-root /private/original_pr9_attempt \
  --diagnostic-root /private/original_pr10_attempt \
  --out /private/p8_plan
```

After review, use the clean approved source digest and exact sealed plan:

```bash
python -m experiments.dflash_feedback_allocation_control.run run \
  --baseline-root /private/original_pr9_attempt \
  --diagnostic-root /private/original_pr10_attempt \
  --sealed-plan /private/p8_plan/PLAN.json \
  --approved-code-sha256 REVIEWED_SOURCE_DIGEST --allow-gpu-execution \
  --out /private/p8_attempt --native-root /adequate/private/p8_native
```

A plan made from local evidence paths must be resealed at its actual destination
with verified artifact rebinding; path changes must not silently change content.
Both actions require fresh output directories outside this public checkout.
`report` reads an existing result and cannot execute models.

## Deliverables and interpretation

`RESULT.json` and append-only `EVENTS.jsonl` preserve raw receipts, native traces,
exposure, partial failures and costs. Reports provide `PAIRED_REQUESTS.csv`,
`POOLED.csv`, `COSTS.csv`, `TRAINING.csv`, `SUMMARY.json` and `REPORT.md`.
Separate source and held-out populations, with original-group, pool/dataset and
overall summaries. Call changes are ratios of summed paired counts; normalized
KL is averaged equally across eligible responses. Retain better/worse/tied and
invalid counts and all regressions. The primary result is held-out calls versus
F, with P8 versus each request's S8 also shown.

Source fitting with fewer held-out calls than F supports transfer under this
allocation. Beating S8 but remaining worse than F is reduced harm. Source fitting
without held-out benefit means this allocation was insufficient; it does not
prove coverage was or was not the cause. Negligible source fitting makes that
transfer test inconclusive.

Serving, teacher preparation, training, read-only evaluation, loading/export,
activation, cache work and peak memory remain separate. Nested timers are not
summed twice. **Fresh P8 versus historical F/S8 latency cannot establish speedup.**
CPU tests do not establish native GPU compatibility, memory fit or output equality.
