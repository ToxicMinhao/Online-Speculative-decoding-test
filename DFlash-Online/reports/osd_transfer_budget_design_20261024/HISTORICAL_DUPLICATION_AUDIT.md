# Prior experiments and duplication audit

## Material Passport

October 24, 2026. ARS experiment-agent, validate mode; **ANALYZED** from retained
protocols, reports, configurations and machine-readable summaries. No model
reexecution, remote operation, production-code change or campaign launch. Two
scoped agents reviewed earlier allocation and replay studies; this is internal
review, not external peer review. Stage 1 / project None remains unchanged.

## Conclusion and correction to the preceding recommendation

**We have already tested the central new-versus-repeated-feedback idea in closely
related, controlled experiments.** The proposed four-event run is not an exact
execution duplicate, but its broad scientific motivation substantially overlaps
completed work. The preceding proposal did not recover the October 4 paid-learning
grid (D97) and October 5 common-state schedule comparison (D108) before recommending
another broad screen. That omission matters: these are more relevant than only
the recent eight-response failures or the latest long-run endpoint.

**Recommendation:** do not launch the proposed 2,176-generation comparison
unchanged merely to rediscover whether more training or different feedback can
help. Keep its configuration and manifest as an unexecuted design. First reuse
the findings below. Reopening should require a specific decision that depends on
the remaining current-runner/pure-forward-KL/live-feedback difference, rather
than treating that difference alone as a new bottleneck. This is a recommendation,
not a user-approved replacement experiment or a research-stage change.

## 1. Direct predecessor: pure forward-KL repetition versus coverage

[D97 protocol](../dflash_paid_learning_20261004/PROTOCOL.md) and
[completed report](../dflash_paid_learning_20261004/RESULT_REVIEW.md) use the same
released Qwen3-8B/DFlash-b16 model revisions as the current proposal. Every arm
uses native OSD forward KL, temperature 1, LR 3e-4, direct BF16 AdamW/decay .01,
anchor cap 512, gamma 7, block 16 and effective response batch 2. It tests nested
128/512/2,048-response pools at one/two/four passes, with ordinary buffered control.

All checkpoints face the same 256 development requests and produce identical
output token IDs. The counts below come from the
[machine-readable curve](../dflash_paid_learning_20261004/analysis_001/SUMMARY.json).

| Comparison | Steps per model | Verification calls | Interpretation |
|---|---:|---:|---|
| 128 responses x 4 passes versus 512 x 1 | 256 | 12,048 versus 12,086 | More distinct feedback did not provide a clear advantage at this budget. |
| 512 responses x 4 passes versus 2,048 x 1 | 1,024 | 11,681 versus 11,689 | Another near-tie; no formal equivalence claim. |
| Frozen versus 128 responses x 2 passes | 0 versus 128 | 12,607 versus 12,116 | A 3.89% call reduction already existed after 128 responses under pure forward KL. |

The accepted-draft-token metric also gives near-ties in the two matched-step
contrasts, with paired prompt intervals spanning zero. Accepted-token means and
pooled verification calls are different metrics; their tiny rankings need not
agree. Matched steps/presentations are not exact matched supervised tokens or cost.

This is internally controlled and already uses the competent core recipe. It
does not suffer from the recent small Qwen screen's 64-anchor/cosine-LR defaults.
Differences from the current proposal are precollected frozen feedback, seed 0,
the author runner, global/pass ordering, a 2,048-token training limit (69 output
tokens truncated in one response), and a development panel. The current proposal
uses causal S-generated feedback, seed 17, current DFlash-Online, and exact tokens
up to 2,272. These limit transport of exact numbers; they do not invalidate D97.

The older small-budget audit's 512-response earliest-positive statement described
D79's saved checkpoints only. It must not be read as the earliest project-wide
positive result: D97 contains positive 128-response endpoints, including the
two-pass endpoint above. It still does not establish a minimum useful buffer.

## 2. Closest sequential predecessor: repeat versus unused feedback after a fork

[D108 protocol](../dflash_schedule_value_20261005/PROTOCOL.md) and
[completed D114 report](../dflash_schedule_value_20261005/final_review_001/RESULT_REVIEW.md)
fork a common complete model/Adam/RNG state after 512 responses used four times.
Each subsequent phase uses 1,024 response presentations and 512 optimizer steps.
The menu includes repeat/repeat, unused/unused, repeat/unused, unused/repeat,
fixed mixing, ordinary continuation and ordinary updating from release.

| Endpoint on the identical 256-request panel | Verification calls |
|---|---:|
| Shared warm-up, no further training | 11,529 |
| Repeat old feedback in both phases | 11,531 |
| Use unused feedback in both phases | 11,181 |
| Ordinary native updating from release | 11,219 |

Unused/unused needs 3.04% fewer calls than repeat/repeat, but only **0.339% fewer
than ordinary updating**. No tested full-budget schedule passes the predeclared
2% advantage over the strongest ordinary alternative. Repetition almost doubles
preparation relative to stopping after warm-up without improving calls. The
recorded forks do not exhibit a reversal that requires adaptive action selection.

The principal difference is objective: D108 uses AdaFlash's 0.8 teacher-argmax CE
+ 0.2 clipped reverse-KL bundle, without its length head, rather than pure forward
KL. It keeps the Qwen/DFlash pair and competent native optimizer/sampling recipe.
All feedback is already available, and the schedule budget is larger than the
new proposal. This is a substantial conceptual predecessor, not an exact OSD
replication. It was not unfair simply because it used a different objective.

The original D108 report identified an incomplete cost comparison, but the later
[D142 completed experiment](../dflash_efficiency_decision_20261006/final_review_001/RESULT_REVIEW.md)
**resolved it**. At approximately 20.87 minutes, the candidate used 11,310 calls,
ordinary early stopping 11,419, and the stronger random control 11,350. The
candidate's 0.352% advantage over random did not pass the 2% screen; the tested
special scheduling claim was closed. The first version of this audit stopped at
the older D108 report and incorrectly described that gap as still open. This
correction supersedes that statement; it is not a reason for another rerun.

## 3. Other overlapping evidence

| Completed study | Existing answer | Scope and difference |
|---|---|---|
| [D79 ordinary OSD curve](../dflash_information_value_20261002/STATUS_REVIEW_20261002.md) | Native Qwen OSD already improves at 512/1,024/2,048 responses; selection does not beat competent native learning at roughly equal updater time. | Common frozen collector, seed 0; exact total-cost matching was not established. |
| [PR9 fixed-pool curve](../dflash_baseline_learning_curve_20261020/execution_001/final_review/REPORT.md) | Four responses, 0/2/8 steps: source KL improves by 59–82% at S8, while held-out calls are 1,857/1,918/1,901. | Qualified Llama author-path OSD, tiny source pools; cannot establish large-budget Qwen failure. |
| [PR13 allocation control](../dflash_feedback_allocation_control_20261021/execution_001/final_review/REPORT.md) | Four responses x 4 passes versus eight x 2, both eight steps per model: held-out calls 1,901 versus 1,984; frozen 1,857. | Broader feedback did not help this small Llama setting. One pooled model is compared against each request's own original-group model; model-level budgets match, not total fleet preparation. |
| [A1/A2 checkpoint replay](../dflash_author_replay_20261015/campaign_001/completion_review_001/REPORT.md) | On the same final inputs, the second OSD update saves 5.01% calls and 3.07% request time over the first; three of ten episodes regress. | Eight-response Llama session updates. This is checkpoint replay, not replay-training. |
| [Per-turn OSD](../dflash_per_turn_20261015/campaign_001/completion_review_001/REPORT.md) | Accumulated gains coexist with 39/110 latest updates increasing next-request calls. | One-response/two-step Llama updates and full histories; already establishes heterogeneous marginal utility in that setting. |

The one-versus-two-pass asynchronous comparison changes optimization exposure and
publication timing, so it is not a clean equal-step new-feedback control. The
connected-eight versus delayed split-four comparison isolates conditioning under
matched publication, not distinct feedback. Neither should be relabeled as the
exact proposed experiment; neither warrants erasing its original result.

## 4. Fairness is distinct from applicability

There are three different questions:

1. **Were comparison arms treated consistently within the study?** D97, D108 and
   PR13 retain matched initialization, declared step/exposure contracts and common
   output-validated evaluation. Their source records support scoped comparisons.
2. **Was the baseline competent for the intended regime?** D97/D108 use successful
   native recipes and have positive ordinary controls. PR13 is a legitimate tiny
   feedback experiment, not a qualification of persistent large-budget OSD.
3. **Does the result transfer unchanged to our current implementation/scenario?**
   Different objectives, model families, data volume, feedback provenance and
   numeric execution prevent that claim. This is an external-validity limit,
   not by itself an experimental-control failure.

The [October 23 short Qwen screen](../osd_historical_recipe_audit_20261024/REPORT.md)
is a different case: it used nominal 6e-4 cosine LR, FP32-master AdamW/decay 0,
fixed response order, 64 anchors and only eight steps. Its arms were matched,
but OSD was not established as a strong baseline there. Its negative result must
not be promoted into a general OSD-failure claim. This does not retroactively
disqualify the older native studies above.

## 5. What remains new, and what is already known

The exact combination of a current DFlash-Online pure-FKL learner, four causal
128-response events, a common B1 full-state fork, repeated B1, and fresh rotating
timing panels was not found among inspected completed studies. It would be a
**replication/extension in the current serving protocol**, not the first test of
freshness, repetition, source fitting versus transfer, or marginal update value.

Already supported: ordinary updating can learn; more passes can help with
diminishing returns; more distinct feedback is not automatically superior at
matched steps; unused feedback can beat excessive repetition; ordinary schedules
can capture essentially all of that gain; source fitting and next-request benefit
can disagree. None of these observations by itself identifies a new algorithm.

Consequently, a new mechanism needs a residual limitation and its strongest
ordinary control. The present proposal's broad discovery rationale is insufficient
after recovering this history. No replacement mechanism, smaller rerun, fairness
sweep, baseline campaign or parameter change is introduced by this audit.

## Limits and interpretation checks

This is a targeted audit of the closest experiment lineages, not proof that every
historical file has been inspected. It rechecks original protocols/reports and
D97's numeric summary; it does not reproduce missing model tensors or benchmark
latency. Historical retained results are preserved, including negative cases.
Question/panel reuse, one-seed development results and different inference/cost
boundaries remain explicit. No percentages are pooled across studies.

ARS fallacy scan **11/11 checked**: retain dataset/episode heterogeneity
(aggregation), avoid universal per-request claims (ecological), disclose reused
development panels (selection/Berkson), do not condition on successful updates
(collider), use explicit per-model and request denominators (base rates), compare
fixed controls rather than selected extremes (regression to mean), preserve all
arms and failures (survivorship), retain complete menus and original interval
qualifications (look-elsewhere), do not launch outcome-driven replacements
(forking paths), do not attribute cross-study differences to one setting
(causation), and keep feedback-before-evaluation chronology explicit (reverse
causality). Near-ties are not equivalence proofs; call savings are not automatically
serving-time or adaptation-inclusive savings.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_transfer_budget_design_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
