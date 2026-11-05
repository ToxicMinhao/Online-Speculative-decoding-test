# Persistent OSD rerun: settings audit and preparation

**October 24 completion update:** the campaign has finished and both GPUs are idle.
See [completed results and audit](RESULTS.md):11.33% held-out request-time reduction,
11.57% fewer verification calls,0invalid pairs; adaptation costs remain separate.
The dated preparation and first-event notes below are historical.

October 23, 2026. ARS experiment-agent run workflow. **Running on A800; first full-event qualification passed:**
the user selected 2,048 output tokens or EOS and a 2,272-token training limit. No new learning or
performance result is claimed. Stage 1 / project None remains unchanged.

## Verified settings

| Item | Current implementation for the rerun |
|---|---|
| Code | A800 `/data/sources/minhao/DFlash-Online`; existing `online_feedback` backend, controller and trainer |
| Models | Released Qwen3-8B + Qwen3-8B-DFlash-b16; frozen target, target embedding and vocabulary head |
| Objective | Forward KL, temperature 1, no hard-label mixture; gamma-7 position weighting; response averaging |
| Optimizer | Direct BF16 AdamW; constant LR 3e-4, no warmup; decay .01; betas (.9,.999); epsilon 1e-8; gradient clipping 1 |
| Precision | BF16 parameters, optimizer moments and probability/KL rows; FP32 weighting/reduction |
| Sampling | Up to 512 eligible anchors per response presentation; current eligibility unchanged; new samples each pass |
| Training | Two shuffled passes, one response per microbatch, accumulation two; persistent weights/optimizer/RNG; seed 17 |
| Attention/loss execution | Flex training attention; eight-block checkpointed vocabulary loss; no optimizer step between loss chunks |
| Event budget | 128 newly completed responses per event; 58 events; 7,424 optimizer steps; 14,848 response presentations |
| Serving | Qualified SGLang/FA3 backend, BF16 greedy block 16, concurrency one; full cache flush before requests and publication |
| Publication | Synchronous, after a complete event; serving waits for the update |
| Training inputs | Exact generated token IDs; only response positions supervised; prompt context retained; no silent truncation |

The 19 installed core/config files match the qualified source snapshot. Only a
small campaign package and its tests were added. Upstream training/inference and
Fenrir's clean `/home/minhao2/DFlash-Research` were not modified.

## Data and evaluation

The preparation code verified and copied the original 7,473 GSM8K training prompt
IDs, identities and ordering. Historical responses are not copied into training.
The first 7,424 fresh responses fill 58 events; the final 49 requests are served
without another update. Each question remains an independent prompt; weights,
optimizer state and RNG persist across questions. This is not a multi-turn
conversation concatenation experiment.

The original 1,024 held-out prompt IDs and order are retained. They do not train
the draft. Final OSD and released DFlash will be measured freshly on those inputs,
with three capture-disabled repeats in rotating order and no concurrent training.
The historical extra generation warmup is excluded. Request timings exclude
checkpoint loading/publication; these measure performance with an available
checkpoint. Invalid output pairs remain visible and are excluded from gains.
Ratios use summed measurements. Three repeats do not triple the number of
independent questions.

The first complete event checks training counts, exact native/export weight
equality and capture-disabled frozen/updated/reloaded output preservation. A
performance loss does not block continuation; a correctness failure does.

## Selected length policy

Training prompt lengths are 24–224 tokens. Both prepared candidates use a 4,096
serving context limit; the runtime separately reserves 16 proposal slots.
Held-out generation keeps the original 2,048-output cap.

| Candidate | Training total-length limit | Training output cap |
|---|---:|---:|
| Keep the current training limit | 2,048 | Remaining space: 1,824–2,024 tokens, or EOS |
| Preserve the old output cap | 2,272 | 2,048 tokens, or EOS |

The user selected the second row: 2,048 output tokens with a 2,272-token training
limit. The first row is retained only as preparation history. The old reference
retokenized and truncated training sequences to 2,048; the current implementation
must not silently reproduce that truncation. Six old training responses reached
2,048 output tokens, so this distinction is reachable. The selected manifest and training configuration are sealed together before
launch; no other learning setting changes.

## Validation and cost accounting

All **276 CPU tests passed on Fenrir and A800**, including 37 campaign tests for
request identity, event accounting, capture isolation, invalid-pair exclusion,
publication checks and safe rolling retention. An independent integration review
found no remaining blocker. This does not substitute for native first-event
qualification of the longer run.

Serving and blocking update spans will be counted once in the reconstructed
episode total. Capture, teacher preparation, training, state persistence, export
and activation are reported as explanatory components, without adding nested
timers again. Loading, qualification, publication validation, artifact retirement,
benchmark evaluation and total campaign wall time remain separately identifiable.

Storage is admitted before launch and each event, including capture/prepared
features, overlapping atomic trainer states and exports, compact evidence, work
allowance and an 8 GiB reserve. Completed events retain compact traces, tokens,
anchor/exposure metadata and receipts; only then may bulk tensors and the prior
export be retired. The current restart state/export remain available.

Storage cleanup completed under the standing retention authorization. A800 free
space increased from about **19.59 GB to 67.80 GB**. The first retirement removed
12 obsolete export paths and two completed-run trainer states after verifying
3,652 compact files and both retained model exports. Hardlinks were accounted for:
46.14 GB of selected path sizes freed about 35.65 GB of actual storage.

A second retirement removed three completed per-turn diagnostic Adam states
(D2/D5/D6), freeing about 12.58 GB. Their model exports remain, and every
nonoptimizer field—including counters and RNG state—was preserved and checked.
Exact optimizer continuation for these completed snapshots is lost; they were
not archived as full optimizer states. Paused-run recovery and active models were
preserved. See the [additional plan](storage/ADDITIONAL_PLAN.json) and
[executed receipt](storage/ADDITIONAL_RETIREMENT.json).

Both candidate budgets now pass: **64.01 GB** for the 2,048-token training limit,
or **66.97 GB** for 2,272, each already including the 8 GiB reserve. Admission
will be repeated at actual launch and each event. Both A800 GPUs were idle at admission; the run now uses GPU 0 for serving
and GPU 1 for training.

## Interpretation

This run asks whether the selected persistent OSD recipe improves held-out
drafting in DFlash-Online. It does not isolate one setting as the cause of an
improvement. Exact-token preparation, anchor eligibility, teacher execution,
loss chunking, synchronous publication and RNG realization still differ from
the historical author implementation. Matching major recipe settings does not
establish bitwise equality or guarantee the historical gain.

Evidence: [preparation receipt](preparation/PREPARATION.json),
[local tests](CPU_TESTS.log), [A800 tests](CPU_TESTS_A800.log),
[retirement receipt](storage/RETIREMENT.json), [authorized scope](AUTHORIZATION.md).

## Launch

Detached PID 3068001 started at 2026-10-24T05:08:57.774200+00:00. The exact
manifest, config and source identities are in [SEAL.json](sealed/SEAL.json).
Live results: `/data/sources/minhao/runs/dflash_online_osd_rerun_20261023/campaign_001`.
The first complete event passed correctness checks and the second update is
running. Full learning results and final held-out comparisons remain pending.

## First complete event

The first event completed 128 optimizer steps over 256 response presentations
(two passes over 128 fresh responses). Native publication matched all 58 export
tensors through the 43 fused native tensors. Updated/frozen/reloaded replays
received identical inputs, produced identical outputs, and updated/reloaded
verification counts matched. No replay supplied learning feedback.

| Cost | Seconds |
|---|---:|
| Feedback collection, controller span | 139.20 |
| Blocking update, complete controller span | 375.47 |
| Training computation, inside update | 330.20 |
| Teacher preparation, inside update | 4.95 |
| Prepared feature persistence, inside update | 5.90 |
| Trainer-state persistence, inside update | 17.20 |
| Export, inside update | 11.58 |
| Complete activation, inside update | 1.40 |

These nested update components are not added again to the 375.47-second span.
The production collection-plus-update total for event 1 is 514.67 seconds; loading,
qualification, publication validation and retirement are additional experimental
wall-time components reported separately.

The single post-update qualification request produced 172 identical output tokens:
frozen 32 verification calls, updated 33, and reloaded 33. This local regression is
retained; it is not an efficacy gate or an estimate of the full held-out outcome.

Verified rolling retirement removed 4.15 GB of event tensors after preserving compact
evidence. The current restart state and export remain. At the current measured
rate the 58-event learning phase is roughly 8–10 hours; fresh final evaluation adds
further time. This is a first-event estimate, not a measured campaign duration.

[First-event summary](FIRST_EVENT_SUMMARY.json),
[raw compact evidence](FIRST_EVENT_EVIDENCE.tar.gz).

[Independent first-event audit](FIRST_EVENT_REVIEW.md) verifies the two shuffled passes,
128 step learning rates, anchor eligibility and 949,560 supervised-position exposures.
Exposure totals use optimizer-boundary metrics because those rows contain accumulated
two-response windows; summing all presentation metric rows would double-count.
All 653 compact retirement-evidence files were copied and verified locally.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_rerun_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
