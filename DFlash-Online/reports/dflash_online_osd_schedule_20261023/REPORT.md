# OSD schedule update and feedback-collection explanation

## Material Passport

October 23, 2026. Scoped implementation and CPU validation, completed. The user
selects 128 responses per event, 58 events and 7,424 optimizer steps, then asks
about feedback collection. No GPU qualification, serving run or scientific campaign
was launched. Collection policy, dataset selection, optimizer settings, anchors,
teacher preparation and serving runtime remain unchanged.

## Implemented schedule

A800 `/data/sources/minhao/DFlash-Online/configs/osd.yaml` now sets:

- `online_feedback.responses_per_event: 128`
- `online_feedback.passes_per_event: 2`
- `online_feedback.max_events: 58`
- `training.max_steps: 7424`

With accumulation two: 128 responses × two passes / two responses per step =
128 steps/event. Across 58 events: 7,424 distinct feedback responses, 14,848
response presentations and 7,424 optimizer steps.

`recipe.py` parses the schedule and checks consistency with the optimizer horizon.
`controller.py` uses it for omitted buffer size, update count and passes, and rejects
event 59 before sealing or preparing feedback. One trainer continues weights,
optimizer and RNG across events. Explicit count/capacity/pass arguments remain for
bounded qualification and historical callers; the arithmetic above assumes the
configured defaults, not arbitrary overrides. Existing trainer horizon enforcement
is unchanged. No automatic dataset loop or new launcher was introduced.

Only six listed source/config/documentation/test files were installed, with old
contents retained in `before.tar`. Author training/inference code and Fenrir
DFlash-Research were untouched. Historical experiment drivers/configs remain
unchanged and do not silently become launchers for this revised recipe.

## Feedback collection: what differs

The historical Qwen/GSM8K run collected each completed target-verified serving
answer from the draft checkpoint currently active in its inference worker. A new
buffer trained the same persistent draft and optimizer. Training and serving were
asynchronous, so later requests could still use an older checkpoint while a new
update was being computed; the run was not a strict 128-serve/stop/train cycle.

The recent controlled comparison instead generated its A responses once with
released frozen DFlash, then reused those exact records for each objective arm.
That controls the training data while comparing objectives. It was a choice of
experiment driver, not a limitation of `online_feedback`.

The existing controller already generates through its currently published draft,
stores that actual response, and publishes the trained checkpoint for subsequent
requests. Its synchronous loop can therefore collect successive buffers from its
own evolving serving state. The present change only sets the schedule; it does not
select or launch that data-collection campaign.

For the first buffer, both approaches start from released DFlash. The distinction
principally matters after updates accumulate. An updated draft does not automatically
mean a different correct answer: exact greedy speculative verification should
preserve the target answer on identical inputs, while proposal/rejection trajectories
can change. Preserve actual output IDs rather than assuming equality or difference.
This provenance difference alone is not an established cause of the earlier result.

Separately, the old native buffer retained prompt/answer text, retokenized it and
recomputed teacher features on each training presentation. Our buffer retains exact
prompt/answer IDs, the serving checkpoint identity and the verification trace;
`prepare_feedback()` reconstructs causally aligned frozen-target features once,
then training reuses them to obtain teacher distributions for sampled blocks. It
does not use dataset answer keys or unverified draft proposals as the committed
response. These preparation/precision differences remain under review.

## Checks and limits

All 181 CPU tests pass locally and on A800. New checks cover two consecutive,
nonoverlapping 128-record buffers, persistent trainer counters, event-59 stopping,
schedule/horizon mismatch rejection and explicit two-response qualification.
Existing tests cover actual tiny-model optimizer/RNG restart and learning arithmetic.
The two 128-record lifecycle events use test doubles; they are not 7,424 neural
training steps or a GPU performance result. Independent scoped review found no
blocking issue. An initial local test invocation used an older Conda interpreter
without Pydantic; its failed import log is retained. Correct-environment reruns pass.

Source-only installation was admitted for 256 MiB plus an 8 GiB reserve. A800 free
space was 28,112,982,016 bytes before installation and 28,112,711,680 bytes after
CPU checks. Both GPUs were idle. Fenrir had about 80 GiB free. Nothing was deleted.
These checks do not admit the eventual campaign's checkpoint/feature storage peak.
Full-model Flex qualification and scientific efficacy remain pending.

## Evidence

- [Authorization](AUTHORIZATION.md).
- [Configuration snapshot](worktree/configs/osd.yaml).
- [Local CPU results](fenrir_cpu_tests.log), [A800 CPU results](cpu_tests.log).
- [Installed files and prior hashes](INSTALL_FILES.json), [installation receipt](INSTALL_RECEIPT.json).
- [Original settings comparison](../dflash_online_osd_recipe_20261023/QWEN_REFERENCE_SETTINGS_CHECK.md).

## Feedback-loop clarification

The user correctly notes that both systems implement the same intended causal
feedback cycle. The recent one-event screen learned from released-draft A feedback,
then evaluated its updated checkpoint on B and stopped. B did not train a second
checkpoint. The original reference continued learning across later buffers. Thus
“frozen-generated versus evolving feedback” should not be presented as a separate
algorithmic defect or an independent cause of the first-update result: both begin
with the released draft, and the distinction arises principally because the small
screen stopped after one training event. The current controller already supports
serving, buffering, updating and publishing repeatedly with persistent state.
The 128-response/58-event schedule is implemented but has not executed. No new
collection policy, experiment or scientific setting is selected in this explanation.

Checked sources: `experiments/projected_target/osd_followup.py` reuses retained
F/A features; its shared `run_arm` trains one event and evaluates A/B. The completed
follow-up report explicitly states that B never trains. Historical asynchronous
publication and current synchronous execution are separate scheduling differences.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_schedule_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
