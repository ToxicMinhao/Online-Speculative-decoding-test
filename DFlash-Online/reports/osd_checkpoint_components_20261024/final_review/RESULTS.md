# Completed OSD checkpoint-component comparison

## Material Passport

October 24, 2026 local / October 25 UTC. ARS experiment-agent, **validate** mode.
Verification status: **ANALYZED**. Completed execution receipts, raw native traces,
all timing repeats and arithmetic were audited; no new model run or training was
performed for this review. Independent internal arithmetic and interpretation
reviews are retained. Statistical fallacy scan: **11/11 checked**.

**Status:** 768/768 generations completed, exit 0, in **33 min 41 s**. All 64
declared requests pass the native common-cohort checks. The no-transfer timing
condition was violated by monitoring transfers; see the explicit disclosure below.
Stage 1 / project None and other paused work remain unchanged. This report
supersedes earlier “qualification passed; campaign running” status summaries.

## Practical conclusion after user review

**This experiment did not identify an actionable mechanism or a concrete way to
improve on competent OSD.** The checkpoint differences below are descriptive
results, not sufficient algorithmic guidance. The earlier body-only-training
follow-up suggestion is withdrawn as a recommendation based on this result alone.
The measurements and limitations remain unchanged. No further component test is
authorized; mechanism brainstorming remains open.

## Main finding

**Keeping the jointly learned draft body preserves substantial drafting benefit;
keeping the learned target-feature conditioning alone preserves little. The
conditioning changes become more useful when combined with the learned body.**

The body-only hybrid retains **70.72%** of full OSD's net verification-call savings;
the conditioning-only hybrid retains **9.28%**. Neither meets the predeclared 80%
point-estimate screen. This is an endpoint-substitution result, not evidence that
training only the retained group will reproduce those gains.

## What was compared

The target is the pinned Qwen3-8B; the drafter is the pinned Qwen3-8B-DFlash-b16.
The two parameter groups were fixed before evaluation:

- **C — target-feature conditioning:** `fc.weight` and `hidden_norm.weight`,
  mapping target features into draft-layer inputs. Two tensors, 83,890,176
  parameters, approximately 8% of the exported draft parameters.
- **B — draft body:** all `layers.*` and `norm.weight`. Fifty-six tensors,
  964,736,256 parameters, approximately 92%.

| Arm | Conditioning C | Draft body B |
|---|---|---|
| F | Released | Released |
| O: full OSD | Jointly learned | Jointly learned |
| C-only | Jointly learned | Released |
| B-only | Released | Jointly learned |

Both hybrids are exact BF16 tensor substitutions from the same released and
successful final OSD endpoints. Target-owned embeddings/projection and model
architecture stay fixed. This experiment introduces **zero optimizer steps,
feedback collections or extra generation warmups**.

O is the previously trained event-58 endpoint after 7,424 optimizer steps. Its
recipe was pure forward KL, constant LR 3e-4, BF16 AdamW with weight decay .01,
128 responses/event, two shuffled passes, effective response batch size two,
up to 512 sampled anchors per response presentation, BF16 KL arithmetic and
Flex training attention; maximum training sequence length 2,272. Those training
settings were not changed or rerun here. See the copied
[source configuration](../a800_prepared/SOURCE_TRAINING_CONFIG.yaml).

The evaluation panel contains 64 requests sampled uniformly with seed 20261024
from the original 1,024 held-out GSM8K development requests. These are previously
inspected development requests; their responses did not train O. All four arms
use identical saved prompt IDs and generate normally. Serving is pinned native
SGLang/FA3, BF16, greedy, block 16, concurrency one, 2,048 output tokens or EOS,
4,096 canonical context capacity plus the original internal proposal allowance,
and cold caches. Three rotating repeats give 768 generations, including the
first eight requests' 96-run correctness qualification.

## Results on the full valid panel

Calls use designated repeat 0; all three repeats reproduce those counts. Each
arm generates the same **19,547 output tokens**, with **zero cap hits**. The
three repeats are not additional independent requests or training seeds.

| Arm | Verification calls | Calls per 1,000 output tokens | Calls saved versus F | Call reduction | Fraction of O's saved calls | Mean summed request time, s | Observed time reduction* |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 3,140 | 160.64 | 0 | 0% | — | 55.041 | 0% |
| O | 2,795 | 142.99 | 345 | **10.99%** | 100% | 48.811 | 11.32% |
| C-only | 3,108 | 159.00 | 32 | **1.02%** | 9.28% | 54.271 | 1.40% |
| B-only | 2,896 | 148.16 | 244 | **7.77%** | 70.72% | 50.987 | 7.36% |

*Time totals sum the 64 requests and average the three repeats. These are
available-checkpoint request times, including prefill, drafting, verification
and nested request cache maintenance, excluding historical training and
checkpoint publication. **Monitoring transfers overlapped execution**, so the
timings are observations with possible I/O interference, not isolated-timing
confirmation. Verification-count conclusions are supported separately.

| Arm | Requests improved / tied / regressed versus F | Gross calls saved | Gross extra calls | Net saved | Extra calls versus O |
|---|---:|---:|---:|---:|---:|
| O | 56 / 3 / 5 | 359 | 14 | 345 | 0 |
| C-only | 29 / 11 / 24 | 78 | 46 | 32 | 313 |
| B-only | 55 / 1 / 8 | 266 | 22 | 244 | 101 |

Both hybrids have less gross saving and more gross harm than O. C-only's small
net benefit is not simply a uniformly weaker improvement: it regresses on 24
of 64 requests, including 21 of the 56 improved by O. B-only improves 51 of
those 56. These signs describe outcomes; they do not define a prospective router.

| Arm | Accepted draft tokens | Accepted draft tokens per verification | Committed progress per verification |
|---|---:|---:|---:|
| F | 16,399 | 5.223 | 6.205 |
| O | 16,748 | 5.992 | 6.971 |
| C-only | 16,434 | 5.288 | 6.269 |
| B-only | 16,643 | 5.747 | 6.728 |

Committed progress totals 19,483 per arm: the output total less the initial
prefill-generated token for each of 64 requests. Stop/EOS clipping is preserved
when counting accepted tokens; accepted tokens are not inferred by subtracting
one from each block's progress.

## The groups are complementary

| Substitution | Calls saved | Pointwise 95% paired-request bootstrap interval |
|---|---:|---:|
| Add learned C to released B | 32 | [-7, 72] |
| Add learned C to learned B | 101 | [56.98, 145] |
| Add learned B to released C | 244 | [185, 303] |
| Add learned B to learned C | 313 | [253, 373] |
| Interaction: joint saving minus the two isolated savings | **69** | **[17, 118]** |

The interaction is `345 - 32 - 244 = 69` calls. In plain terms, conditioning
changes save 32 calls with the released body but 101 with the learned body.
It would be incorrect to conclude that C is useless or that 70.72% of all
learning occurred inside B. These are conditional endpoint effects on native
verification trajectories. They are compatible with co-adaptation, but do not
identify its optimization mechanism or assign the interaction uniquely to a group.

The prescribed 10,000 paired-request resamples reproduce independently. O's net
saving interval is [277, 411] calls. Retained-saving intervals are C-only
[-2.12%, 19.73%] and B-only [59.42%, 82.02%]. Every resample has O savings of at
least 2% of F calls; no denominator cases were filtered. C-only's absolute
interval includes zero. B-only's interval crosses 80%, so missing the 80% point
screen is not proof that 80% retention is impossible, nor a noninferiority test.

These exploratory, pointwise intervals describe request-composition sensitivity
on one reused development panel and one training endpoint. They are not training-
seed uncertainty, simultaneous guarantees or fresh generalization confirmation.

## Validity and execution audit

- All 768 planned records exist; none invalid, missing, unknown or unattempted
  under the native measurement checks. All exact prompt and output IDs agree.
- Fresh F and O calls reproduce their retained historical values on every
  request/repeat. The four arms' repeated calls and token statistics agree.
- All **35,817 verification blocks** were checked against recorded proposals,
  target choices, first rejection, committed tokens, absolute output offsets
  and stop handling. No discrepancy was found.
- All 768 publications invalidate caches and refresh helpers. Each arm's full
  weight qualification checks all 58 source / 43 fused native tensors.
- Method positions are balanced: each arm occupies each position 48 times.
  Capture remains disabled. No trainer or model-update process runs in timing.
- **Timing protocol deviation:** monitoring rule/status transfers and a
  13.49 MB first-eight qualification retrieval occurred while later requests
  ran. Exact request overlap and latency bias were not measured. This prevents
  declaring full timing-protocol compliance. See the
  [transfer disclosure](TIMING_PROTOCOL_DEVIATION.md). No rows were removed or
  corrected, and no additional measurement was launched.

The historical O export had no contemporaneous content digest available from
its earlier training run. Its event-58 publication receipt, retained location,
new prelaunch seal and exact historical native output/count reproduction support
the identity used here; do not describe the old missing digest as recovered.

## Cost and storage accounting

| Cost | Seconds | Scope |
|---|---:|---|
| CPU preparation | 49.664 | Before campaign; includes hybrid preparation |
| Initial worker/model loading | 20.257 | Outside request timing |
| All 768 complete request latencies | 627.328 | Four arms × three repeats |
| Checkpoint activation | 1,327.738 | Full publication call, outside request timing |
| Full tensor-equality qualification | 8.265 | Outside request timing |
| Other campaign wall time | 36.562 | Unattributed remainder; no kernel breakdown inferred |
| Measured campaign wall time | 2,020.150 | Loading + requests + activation + validation + remainder |
| Supervisor wall time | 2,021 | 33 min 41 s, exit 0 |
| New training / feedback collection | 0 | None in this diagnostic |

Request cache maintenance (4.418 s) is already inside request latency. The native
worker's internal activation timer (305.868 s) is already inside the full
activation total. Neither is added again. Transfer costs were not separately
instrumented. Frequent publication is a benchmark cost of testing fixed endpoints;
it does not measure a proposed deployment schedule. Historical O training cost
is not charged by this no-training comparison, so no adaptation-inclusive speedup
or cheaper-training claim follows.

At the post-run check, both A800 GPUs were idle with 0 MiB allocated. Available
storage was **48.40 GB on A800** and **83.25 GB on Fenrir**. The complete remote
run is about 168 MB; raw results were copied to Fenrir and retained remotely.
No cleanup/deletion was needed or performed. Existing Fenrir F5 work was left
untouched. The [storage receipt](STORAGE_REVIEW.json) records exact quantities.

## What this changes, and what remains open

1. Competent full forward-KL OSD remains the working baseline. This diagnostic
   does not establish a need to replace its loss.
2. Retaining learned conditioning alone does not recover most of this endpoint's
   gain. A small component at that boundary should not be assumed sufficient.
3. The learned body contains a useful inherited solution, but it includes about
   92% of parameters. This is not yet a compact or cheaper online mechanism.
4. Full OSD still wins, and the 69-call interaction cautions against treating
   the groups as independent contributions.

If a follow-up is selected, a directly related question is whether B can learn
useful changes **while C is held fixed during training**, rather than inheriting
B from joint training. That requires a separately reviewed matched-feedback,
matched-exposure comparison with full OSD and measured costs. Freezing only 8%
of parameters does not itself promise worthwhile savings. This is a discussion
candidate, not a selected mechanism or an automatically launched experiment.

The present experiment does not reveal which body layers/directions matter,
whether restricted training succeeds, how much feedback is needed, when online
learning first becomes useful, or whether future serving savings repay training.
It neither reopens failed mechanisms nor proves full-backbone training necessary.

## Evidence package

- [Every timing repeat](raw/campaign_001/evaluation/report/MEASUREMENTS.csv)
- [Per-request results](raw/campaign_001/evaluation/report/PER_REQUEST.csv)
- [Pooled totals](raw/campaign_001/evaluation/report/POOLED.csv),
  [conditional effects](raw/campaign_001/evaluation/report/INTERACTION.csv),
  [cross-tabs](raw/campaign_001/evaluation/report/CROSS_TABS.csv)
- [Machine-readable summary](raw/campaign_001/evaluation/report/SUMMARY.json)
- [Native trace audit](NATIVE_VALIDITY_AUDIT.json) and
  [CPU audit script](audit_native_records.py)
- [Independent arithmetic audit](ARITHMETIC_REVIEW.md),
  [interpretation and 11-point fallacy scan](INTERPRETATION_REVIEW.md)
- [Approved protocol](../../osd_next_experiment_review_20261024/PROPOSAL.md),
  [implementation report](../IMPLEMENTATION_REPORT.md),
  [source/artifact identities](../a800_prepared/ARTIFACT_IDENTITIES.json)

Native traces, exact token IDs, all repeats, qualification/publication receipts
and supervisor records are under `final_review/raw/`. The runner-generated raw
report predates this audit and does not include the monitoring-transfer disclosure;
read it together with this final report. No raw evidence has been rewritten.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_checkpoint_components_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
