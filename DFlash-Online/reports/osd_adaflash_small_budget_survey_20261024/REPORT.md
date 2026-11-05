# OSD and AdaFlash: what training budget is actually supported?

**October 24 historical-evidence correction:** the later
[duplication audit](../osd_transfer_budget_design_20261024/HISTORICAL_DUPLICATION_AUDIT.md)
recovers an omitted pure-forward-KL Qwen study: 128 responses x two passes already
reduced calls by 3.89% on a fixed 256-request panel. D97 also tested repetition
versus broader feedback at matched steps; D108 tested repeat/unused schedules
from a common complete state under the AdaFlash mixed objective. These correct
any project-wide reading of 512 as the earliest positive budget and weaken the
recommendation to repeat a broad learning/allocation screen. The current runner's
exact causal early-event curve remains unmeasured; a new run needs a specific
decision beyond rediscovering the earlier results. The paper/code findings and
original study-specific numbers below remain unchanged.

October 24, 2026. Focused ARS deep-research source verification and retained-result
audit. Three independent reviews cover original OSD, AdaFlash, and local smaller
budgets. Papers, pinned author code, and local execution receipts are distinguished.
No models were run, no A800 operation occurred, and no recipe or research stage
changed. Fenrir had approximately 78 GiB free; only small research artifacts were
added. This is a targeted survey, not an exhaustive systematic review.

## Answer

**The recent eight-response screen was much smaller than the successful persistent
Qwen run, and insufficient feedback/training is a plausible explanation for its
negative transfer. It is not an isolated causal finding.** The optimizer, schedule,
sampling coverage, runtime and evaluation panel also changed.

**128 responses is not a universal OSD requirement.** Original OSD explicitly uses
an eight-request interval in its online experiments. AdaFlash uses a different
128-response DFlash recipe. Neither inspected paper establishes a smallest budget
that reliably improves a new model/workload. Running a small batch successfully,
fitting its source responses, improving future drafting, and recovering adaptation
cost are four different standards.

## 1. Correct units for our comparison

| Quantity | Recent Qwen eight-response screen | Successful persistent Qwen rerun |
|---|---:|---:|
| New feedback responses per event | 8 | 128 |
| Events | 1 per independent dataset group | 58 |
| Distinct responses used in training | 8 per group | 7,424 |
| Passes over each response | 2 | 2 |
| Effective response batch | 2 | 2 |
| Optimizer steps | 8 per group | 7,424 |
| Response presentations | 16 | 14,848 |
| Anchor cap per response presentation | 64 | 512 |
| Actual anchor presentations | 1,024 for GSM8K | 3,728,050 |

An event collects feedback, trains for the declared passes, then publishes an
updated draft. It contains many optimizer steps. For our one-training-replica
recipe, 128 responses × 2 passes / effective batch 2 = 128 steps per event;
58 × 128 = 7,424. The full run has **928 times** the responses and optimizer steps
of one old eight-response group. Even one new event has 16 times the responses
and steps. The three old datasets did not pool their feedback into one model.

512 is a maximum number of sampled block starts **per response presentation**,
not a run total or a count of independent examples. Eligibility and response
length can lower it. Overlapping blocks and repeated passes reuse tokens; actual
supervised exposure must be reported separately. The current run records
54,368,391 supervised-position exposures. The 58-event count follows our
7,473-request stream and 128-response schedule, with 49 leftover responses;
it is not a universal paper constant or minimum convergence requirement.

Sources: [old executed configuration](../dflash_online_osd_20261023/execution_001/final/campaign/gsm8k/O/config.yaml),
[old training receipt](../dflash_online_osd_20261023/execution_001/final/campaign/gsm8k/O/trainer/events/update/RESULT.json),
[new sealed configuration](../dflash_online_osd_rerun_20261023/sealed/config.yaml),
[new results and raw audit](../dflash_online_osd_rerun_20261023/RESULTS.md).

## 2. What the primary sources establish

### Original Online Speculative Decoding

The [paper, Algorithm 1 and §5](https://arxiv.org/html/2310.07177v4#S5) sets the
online update interval to eight requests. It studies autoregressive LLaMA-160M /
Vicuna-7B and T5-Small / FLAN-T5-XL, rather than DFlash. Online acceptance curves
use a rolling window of 50 requests over continuing streams; they do not isolate
one eight-request update as a sufficient budget. Its two-epoch statement in §5.1
describes the offline experiment and should not automatically be assigned to the
online loop.

At official code commit `788a403d5495896b4fc5b7f56cfd41de5ae61967`, the common
LLaMA GSM8K/Spider/code scripts use interval 8, constant LR 1e-4 and AdamW with
zero weight decay; the online path computes a gradient-bearing update from the
buffer and clears it. This is not our DFlash two-pass/512-anchor recipe. Other
scripts differ: Arena uses interval 1 and T5 interval 2. These code configurations
are not a published controlled interval ablation.

The [official README](https://github.com/LiuXiaoxuanPKU/OSD/blob/788a403d5495896b4fc5b7f56cfd41de5ae61967/README.md#llama)
shows the first 100 Spider records with rising but noisy acceptance and falling
loss. This supplies an early-learning illustration, not a controlled minimum,
held-out result or adaptation-inclusive speedup. See [original-source audit](ORIGINAL_OSD.md)
for exact scripts, implementation qualifications and the paper/code discrepancy.

### AdaFlash and its OSD-on-DFlash control

[AdaFlash Appendix B.1](https://arxiv.org/html/2607.19223v2#A2.SS1) specifies a
128-response trigger, two epochs, microbatch 1, accumulation 2, LR 3e-4,
AdamW/BF16 and maximum training length 2,048. Its own method adds mixed
distillation and a length head; its OSD baseline applies forward KL to DFlash.
Figure 2 plots improvement across adaptation rounds. The examined main text and
appendix do not provide a smaller-buffer success sweep or a reliable minimum
feedback count. Their hyperparameter ablations concern loss mixture, clipping
and length scaling rather than establishing an 8/16/32/64-response learning
threshold.

The [pinned OSD launcher](https://github.com/ZinYY/AdaFlash/blob/507b7aba5fbccaa99a400f161e4f2794ba6a388c/scripts/pipeline/run_async_pipeline_osd.sh)
exposes threshold, pass count, batch size, accumulation and LR. Setting a smaller
threshold is supported by configuration; that is not evidence it will improve
drafting. The [DFlash objective](https://github.com/ZinYY/AdaFlash/blob/507b7aba5fbccaa99a400f161e4f2794ba6a388c/specforge/core/dflash.py)
defaults to 512 sampled anchors, subject to eligible positions. Neither a shell
argument nor a default provides a data-efficiency guarantee.

There is an important replica-count distinction. The paper calls the effective
batch 2, while the [Qwen OSD shell recipe](https://github.com/ZinYY/AdaFlash/blob/507b7aba5fbccaa99a400f161e4f2794ba6a388c/recipes/qwen3_osd.sh)
uses two training GPUs with TP1: microbatch 1 × accumulation 2 × DP2 gives global
batch 4 and, for a full valid buffer, 64 optimizer steps per event. Our successful
reference and current run use DP1, global batch 2 and 128 steps. Thus even an
identical response buffer does not uniquely determine the optimization budget.
See [AdaFlash audit](ADAFLASH.md) for sampling, reduction and small-buffer edge cases.

One additional code path uses the MATH-500 split of 450 training questions and
50 evaluation questions, with two offline adaptation epochs. This is a concrete
smaller-total-budget recipe, but the paper does not explicitly bind its reported
table to that exact file/checkpoint. It is neither a sub-128 streaming-buffer
experiment nor a measured minimum. Full AdaFlash also has a separate example
that initializes its length head from 4,096 regenerated responses; that preparation
must not be erased from its data budget. These are documented code-path facts,
not proof that every published run followed those examples exactly. The detailed
audit links the preprocessing, offline recipes and initializer separately.

## 3. Existing smaller-budget evidence from this project

These are **our observations**, not author-paper claims and not one common
learning curve. They use different model families, histories and panels.

| Setting | Training budget at the measured checkpoint | Observation | What it does not establish |
|---|---|---|---|
| Qwen/GSM8K native author-path reference, D79 | 512 responses; four 128-response events; 512 steps | On the same 128 development prompts, frozen 6,027 calls versus updated 5,555: **7.83% fewer**, identical outputs | 512 is necessary; benefit at 8/128/256 responses; a cost-inclusive online gain |
| Llama multi-turn author-path, first update | Eight newest responses; two passes; eight steps per episode | Across ten episodes, **7.14% fewer calls and 5.30% less request time**; nine episodes improve and one regresses | Every eight-response update works; the same outcome holds for Qwen standalone questions |
| Llama per-turn author path | One response and two steps per event, accumulated over each stream | Overall WildChat gains, while **all six first updates** worsened the next request | One response from released weights suffices; overall gains make every update useful |
| Recent Qwen short screen | Eight responses; eight steps, older optimizer/sampling settings | Held-out GSM8K calls 303 → 361 | Competent persistent OSD cannot learn |
| Current persistent Qwen/GSM8K | 7,424 responses; 7,424 steps | Held-out calls 50,584 → 44,730: **11.57% fewer** | All 58 events are required; training cost is repaid |

The D79 feedback was collected by a common frozen collector. It is a useful
smaller-budget native training reference, not an identical persistent online
trajectory to the latest run. Its earliest retained measured checkpoint is
buffer 4, not proof that earlier checkpoints were ineffective. The familiar
“buffer 8” label in that study means **1,024 responses**, not an eight-response
training event. Its 128-prompt development panel must not be mixed with its
separate 256-prompt final panel.

The Llama eight-response arm sees full causal history as context; only the newest
eight responses receive loss. Its timing measures serving with the checkpoint
available. Model and workload differences prevent using that positive result as
a threshold for our current Qwen setup. The complete source pairing and metrics
are in [local evidence audit](LOCAL_EVIDENCE.md).

## 4. Consequence for our next comparison

The old short screen supports a limited-budget negative observation under its
specific recipe. It does not justify “OSD fails,” “forward KL fails,” or “a new
mechanism is needed because ordinary learning cannot work.” Conversely, source
fitting without next-request benefit is a real observation; merely increasing
training is not guaranteed to repair it. Feedback diversity, optimization amount
and update frequency are separate axes.

**Recommendation, not a newly authorized experiment:** retain the now-working
recipe and establish a compact learning curve before the next objective comparison.
Keep 128 responses/event and all learning settings fixed; assess checkpoints
after 128, 256, 512 and, if needed, 1,024 responses on one common future-request
panel. Our retained 512-response result gives a concrete starting reference;
there is no evidence-based requirement to repeat all 58 events for every idea.
First use existing checkpoints where valid and available rather than recreating
missing ones without authorization. Treat the panel as development data if used
to select a budget, retaining separate evaluation for subsequent claims.

This curve would answer when benefits emerge under the working recipe. It would
not by itself identify the smallest event size or prove that budget alone caused
the old/new reversal. To isolate event size later, equalize total feedback and
training exposure; to isolate coverage versus extra passes, vary those separately.
Those controls are potential later work, not additions to this survey.

For a practical result, distinguish fewer held-out verification calls, lower
complete request time with a ready checkpoint, and net serving-plus-adaptation
cost. The current run's 11.33% request-time reduction excludes 5.67 hours of
blocking updates. A method that learns with less feedback or less computation
could be valuable, but that contribution requires competent matched controls
and measured costs rather than an assumed minimum buffer.

## Deliverables and limits

- [Original OSD paper/code audit](ORIGINAL_OSD.md).
- [AdaFlash paper/code audit](ADAFLASH.md).
- [Local smaller-budget evidence](LOCAL_EVIDENCE.md).
- [Previous recipe/fairness audit](../osd_historical_recipe_audit_20261024/REPORT.md).

Paper curves do not provide raw records, repeated-seed uncertainty or calibrated
smallest-useful endpoints. This review does not establish a universal threshold,
exact reproduction of the papers, or causation of earlier regressions. No
hyperparameters were changed; no campaign was resumed or launched. Stage 1 /
project None and unrelated pauses remain unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_adaflash_small_budget_survey_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
