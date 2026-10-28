# Locate the useful changes in the successful OSD checkpoint

## Material Passport

October 24, 2026. ARS experiment-agent, plan mode. **Proposal for user and reviewer
discussion only:** no implementation, sampling seal, remote operation, model run,
training or new mechanism. Stage 1 / project None remains unchanged. The proposed
experiment uses the already-successful current DFlash-Online OSD checkpoint and
its released initializer. Independent internal reviews checked prior overlap and
design limitations; they are not external peer review.

## 1. Question and reason

**Can either of two predeclared groups of OSD-learned parameters preserve most of
the measured drafting benefit when the other group is restored to released
weights?**

We already know ordinary OSD can learn, additional passes sometimes help, and
feedback scheduling does not automatically beat competent simple controls. We
also have unsuccessful correction-bank experiments and positive conventional
adapter results in other settings. Those do not tell us where the current
successful Qwen OSD changes are sufficient to improve native drafting.

Before proposing a new correction head, update rule or loss, locate that useful
change at a coarse architectural boundary. This is a no-training intervention
on an existing successful endpoint. It is not a new adaptation method, a test of
training only a subset of weights, or proof that a subset is necessary.

The current [completed run](../dflash_online_osd_rerun_20261023/RESULTS.md) reports
11.57% fewer calls and 11.33% less complete request time on 1,024 held-out GSM8K
requests, excluding adaptation costs. Its endpoint is after 58 events / 7,424
optimizer steps; none of those steps is repeated by this diagnostic.

## 2. Architectural partition, fixed before outcomes

The inspected SpecForge DFlash implementation first computes
`target_hidden = hidden_norm(fc(target_hidden))`, then runs the draft decoder
layers and final normalization. Use this existing boundary, not a partition
chosen by weight magnitude or evaluation results.

| Group | Exact exported parameter membership | Role |
|---|---|---|
| C: target-feature conditioning | `fc.weight`, `hidden_norm.weight` | Map the target's selected hidden features into the representation supplied to the draft layers. |
| B: draft body | Every `layers.*` tensor and `norm.weight` | Attention/MLP processing and final normalization of draft predictions. |

The retained final-publication receipt reports 58 source tensors and 43 fused
native tensors: five draft layers plus the three top-level tensors. C contains
two source tensors; B contains the remaining 56. Exact tensor shapes, parameter
counts, changed-element counts, dtypes and total bytes must be inventoried before
building the two hybrids. C is not assumed small or inexpensive to train merely
because it has two tensors.

Frozen target embeddings and vocabulary projection, tokenizer, model config,
rotary constants and all other nonlearned state remain fixed. Require the exported
parameter partition to be complete and disjoint. If actual artifacts contain a
different key set or unexpectedly changed target-owned state, stop preparation
for review rather than silently assign extra keys to a group.

Inspected source: `/home/minhao2/DFlash-Research/training/specforge/modeling/draft/dflash.py`
(constructor around lines 704–725; forward around 872–897). This local source
inspection is not live A800 source qualification. The run's sealed runtime and
actual artifact identities remain authoritative before execution.

## 3. Four fixed checkpoints

Let F be released DFlash and O be the final successful OSD export.

| Arm | Conditioning C | Draft body B | Purpose |
|---|---|---|---|
| F | Released | Released | Frozen baseline |
| O | Final OSD | Final OSD | Competent trained reference |
| C-only | Final OSD | Released | Test whether the retained conditioning changes suffice on these requests |
| B-only | Released | Final OSD | Test the complementary retained changes |

Copy the selected BF16 tensors exactly. Do not interpolate deltas, recompute an
optimizer update, rescale changes, choose layers after looking at results, or add
rank truncation. Both hybrids use the same architecture and inference code.

No training is performed. The optimizer, learning rate, objective and data used to
produce O remain the already-executed successful recipe: pure forward KL, constant
3e-4, BF16 AdamW with .01 decay, 128 responses/event, two shuffled passes and up to
512 anchors per response presentation. Hybrid construction introduces no new
learning parameters. It does deliberately change which trained tensors are used.

## 4. Data and exact execution budget

Select 64 requests without replacement from the successful run's original 1,024
held-out manifest using a private `random.Random(20261024)`. Sort their original
manifest indices to preserve order. Seal IDs and exact input token IDs before
constructing or evaluating hybrids. Selection must not use existing improvements,
losses, lengths or failure labels. These are previously inspected development
requests, not a fresh confirmation set. Their responses were never training data
for O.

Use each request's retained canonical output token IDs and termination condition.
Evaluate all four checkpoints three times, with cyclic method order
`(3 * request_index + repeat_index) % 4`. Across 64 requests this balances method
positions exactly. Every replay uses the same input IDs and canonical reference;
there is no history regeneration or feedback collection.

**Total: 64 distinct requests x 4 checkpoints x 3 repeats = 768 generations.**
There are zero optimizer steps, zero new target-feedback collections, zero source
loss probes and zero extra generation warmups. The first eight selected requests
are the correctness qualification and are included in the 768 budget (96 runs).
Complete the remaining 56 if correctness passes, including null or harmful results.

Keep the pinned Qwen3-8B/DFlash-b16 revisions, native SGLang/FA3, BF16, greedy
decoding, block 16, concurrency one, 2,048 output tokens or EOS, exact prompts,
4,096 canonical context capacity plus the original internal proposal allowance,
and the established cold-cache/publication policy. No concurrent/background
training or transfers during timing. No shorter cap, altered verifier or fallback
precision is permitted to make a hybrid look better.

## 5. Correctness and artifact gates

Before GPU execution, verify both retained endpoint identities and actual live
source/runtime against the successful run's seal. The latest receipt places O at
`/data/sources/minhao/runs/dflash_online_osd_rerun_20261023/campaign_001/trainer/events/event_058/export`.
F is the pinned released model under the existing Qwen foundation root. These
paths are recorded historical locations, not a new availability check. Missing
O does not authorize reconstructing it by training.

CPU construction checks must show exact per-key origin, same shapes/dtypes, fixed
nonparameter state, and exact reconstruction of F/O when both groups come from
the corresponding endpoint. Retain both original exports untouched.

Use the current publication API for every checkpoint; refresh the fused QKV/KV
helpers and normalization-dependent state, and flush affected caches. Validate
all 58 exported tensors against their native fused representation for each arm.
Matching only the unfused checkpoint file does not qualify a native hybrid.

Qualification checks F and O against their retained output IDs and verification
counts, checks repeated native counts, and checks both hybrids against canonical
output IDs. Investigate discrepancies before interpreting the affected comparison.
Do not replace requests or use a later checkpoint. Preserve invalid rows; exclude
them from gain claims. A structural publication/cache failure stops execution.
Later isolated output discrepancies remain explicit; full-panel claims require
the complete declared valid panel, with common-valid-cohort results labeled partial.

## 6. Measurements and interpretation

Primary measurements: total verification calls and calls per 1,000 output tokens
on identical outputs. Secondary: complete request latency, accepted draft tokens,
committed progress per call, per-request regressions, output lengths and cap hits.
Keep every repeat. Use one validated deterministic trajectory per checkpoint and
request for call statistics; timing repeats are not new request samples.

For group X, report retained full-OSD call savings:

    R_X = [sum calls(F) - sum calls(X)] / [sum calls(F) - sum calls(O)].

Do not clip negative values or values above one. If O has no positive pooled
saving on this panel, do not interpret the ratio. If its pooled reduction is less
than 2%, report raw differences and mark retention ratios too unstable for a
directional screen. Do not choose another panel in response.

Prespecify 80% retained call savings as a descriptive screening threshold, not a
theorem or proof of noninferiority. Report absolute counts and group-versus-O gaps
alongside it. A group reaching this point estimate is only a candidate for a later
restricted-training test. Use 10,000 paired request bootstrap resamples, analysis
seed 20261024, keeping all arms/repeats together; report interval stability and
state that it conditions on a single trained seed and reused development panel.
If denominator instability occurs in bootstrap samples, report that frequency;
do not drop those samples to manufacture a tight ratio interval.

An optional report calculation requires no extra runs: with call savings
`v(X) = calls(F) - calls(X)`, record
`I = v(O) - v(C-only) - v(B-only)` as the measured two-group interaction. This
is arithmetic on this intervention table, not a universal mathematical law,
an additive decomposition assumed in advance, or proof of a training mechanism.

| Observed result | Supported next design implication | Unsupported conclusion |
|---|---|---|
| C-only retains most gains | Target-feature conditioning is a plausible adaptation site to test next. | Training C alone will learn equally well, cheaply or quickly. |
| B-only retains most gains | The retained body changes suffice on this panel; evaluate a mechanism that acts there if independently motivated. | All body weights must be trained, or conditioning never helps. |
| Both hybrids retain gains | Multiple sufficient routes or redundancy may exist. | There is a unique location responsible for learning. |
| Neither retains gains but O does | These trained parts may rely on one another; avoid assuming a separable head/patch can inherit full learning. | Restricted training is impossible or full-backbone training is mathematically necessary. |
| O fails to reproduce its advantage | This panel/runtime does not qualify the intended positive reference. | A new loss is required; replace the panel until positive. |

This intervention concerns the final trained model. It does not reveal the full
learning trajectory, locate the first useful event, or directly explain the old
Llama correction-bank failures. Those have different models and mechanisms.

## 7. Costs, implementation and stopping

Reuse DFlash-Online's existing native backend, publication validation, generation
and paired-scoring code. A small experiment driver builds the two exports,
validates their parameter provenance, and schedules the bounded replays. No
changes to `online_feedback` training, its objectives or the upstream models are
needed. Fenrir DFlash-Research remains untouched.

Report request-ready serving performance separately from checkpoint construction,
loading/publication, cache maintenance and actual diagnostic wall time. The hybrids
inherit O's historical full-training cost; they are not cheaply learned models.
Parameter counts and native inference timing cannot establish cheaper backward
computation. Do not claim training savings or net online acceleration from this
experiment, and do not add a profiling/training campaign to fill that gap.

Budget two new draft exports, one atomic-write overlap, compact raw records and
the required reserve. With the retained approximately 2.10 GB export size, a
conservative initial admission target is 18 GB free for new diagnostic work:
three export-size slots + 2 GB records/work + at least 8 GiB reserve, rounded up.
Actual sizes and destination capacity must be checked before execution. No large
optimizer or feedback-feature artifacts are needed; no existing evidence is deleted
by this proposal.

Estimated wall time is roughly 30–60 minutes, based on the current run's measured
benchmark throughput including publication overhead; startup/compilation may add
time. This is not a measured duration. Proposed wall bound two hours and existing
per-operation timeout. Stop with partial records rather than add requests,
interpolation factors, layers, training seeds, new update rules or automatic retries.

Return one four-arm pooled table, per-request rows, all timing repeats, parameter
origin/count manifest, construction/publication qualification and a short decision
report. Any next restricted-training or mechanism experiment requires a separate
review. This proposal ends at evidence about the usefulness of retained groups.

## 8. Historical overlap check

- D97/D108/D142 tested exposure/scheduling and cost controls; do not repeat them
  here. D142 completed the previously missing cost-matched comparator and closed
  the special scheduling claim. See [gap review](GAP_REVIEW.md).
- Earlier shared-adapter studies trained many attention/MLP/conditioning modules
  together; they did not perform this two-group retained-weight intervention.
- Correction banks changed the final hidden representation through added modules;
  they did not copy the successful OSD conditioning/body changes.
- Earlier matched-prefix attribution compared whole checkpoints and token errors;
  it did not establish which original parameter group was sufficient.

No completed equivalent was found in the inspected source reports and ledger.
This is a scoped history search, not a claim to have inspected every artifact.
Novelty of a research method is not claimed for this standard diagnostic.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_next_experiment_review_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
