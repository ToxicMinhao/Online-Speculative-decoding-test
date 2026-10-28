# KRC: one-event reference-response correction

**KRC improved drafting on this panel: 3,027 → 3,005 verification calls (+0.7268% call reduction).** Requests improve/regress/tie: **31/15/18**.

KRC starts from released DFlash with the fresh seed-17 optimizer and uses the original 128 responses, two passes and unchanged K learning objective. The correction protects the next disjoint response pair in each pass, using its original anchors and freshly captured prefixes. The reference responses are **additional training exposure**, not held-out evaluation. The correction formula, BF16 path, optimizer and serving settings remain unchanged. No new feedback or F/O/K retraining was performed.

First learning-window ordinary-K reproduction: **True**. Completed update windows: **128/128**; event presentations/Adam steps: **256/128**.

## Fresh native K1/KRC comparison

Validated paired requests: **64/64**; valid timed executions: **384/384**. Tokens and termination are checked against retained F; each checkpoint must have repeat-consistent counts. Count totals use one trajectory per request. Latency pools the three-repeat mean for each request.

| Arm | Verification calls | Accepted draft tokens | Calls / 1,000 tokens | Progress / call | Summed mean time (s) |
|---|---:|---:|---:|---:|---:|
| K | 3027 | 16513 | 154.8575 | 6.4364 | 52.275888 |
| KRC | 3005 | 16535 | 153.7320 | 6.4835 | 51.829177 |

Pooled request-time reduction: **0.8545%**. Gross calls saved/added: **44/22**; net saved: **22**. Serving latency excludes training, reference passes, loading and checkpoint switching. This is the reused development panel, not a fresh-input confirmation.

| Repeat | K time (s) | KRC time (s) | Time reduction |
|---:|---:|---:|---:|
| 0 | 52.319884 | 51.817682 | 0.9599% |
| 1 | 52.265882 | 51.823922 | 0.8456% |
| 2 | 52.241899 | 51.845928 | 0.7580% |

Earlier KC used **3035** calls on this panel; that is historical count context only. The primary comparison is the fresh K1/KRC pair above. Historical KC timings are not combined with this run.

## Local behavior on learning A and reference R

Correction branches: `{'applied': 126, 'zero_violation': 2}`. Applied/nonzero-after-BF16 corrections: **126/126** out of 128; applied but rounded to zero: **0**. Median realized correction/K-step norm ratio: **0.002167**, or **0.002196** among applied corrections. Every window's V, norms, no-op branch and costs remain in STEPS.csv.

| Step | Role | Candidate | V | Useful repairs | Earlier errors / blocks | Lost old prefix | Net prefix gain | Negative margins | Wrong ties |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | A_learning | ordinary_K_candidate | 0.003567589 | 527/632 | 47/774 | 153/5236 | +1740 | 50 | 1 |
| 1 | A_learning | corrected_KRC | 0.003624605 | 530/632 | 48/774 | 155/5236 | +1746 | 47 | 4 |
| 64 | A_learning | ordinary_K_candidate | 0.000005220 | 84/655 | 2/748 | 3/4478 | +258 | 1 | 1 |
| 64 | A_learning | corrected_KRC | 0.000005220 | 80/655 | 1/748 | 1/4478 | +254 | 1 | 0 |
| 128 | A_learning | ordinary_K_candidate | 0.000019503 | 64/292 | 3/396 | 13/3155 | +163 | 1 | 2 |
| 128 | A_learning | corrected_KRC | 0.000019503 | 63/292 | 3/396 | 13/3155 | +165 | 1 | 2 |
| 1 | R_reference | ordinary_K_candidate | 0.014633836 | 57/411 | 48/456 | 151/2692 | +43 | 59 | 6 |
| 1 | R_reference | corrected_KRC | 0.009598635 | 59/411 | 41/456 | 121/2692 | +88 | 43 | 6 |
| 64 | R_reference | ordinary_K_candidate | 0.000011174 | 5/522 | 6/774 | 6/6943 | +9 | 3 | 3 |
| 64 | R_reference | corrected_KRC | 0.000016761 | 6/522 | 7/774 | 7/6943 | +14 | 5 | 2 |
| 128 | R_reference | ordinary_K_candidate | 0.000065968 | 11/418 | 8/651 | 26/6047 | +0 | 5 | 3 |
| 128 | R_reference | corrected_KRC | 0.000053403 | 15/418 | 6/651 | 21/6047 | +12 | 4 | 2 |

| Step | Role | Repairs retained / lost / new | Blocks improved / regressed / tied | Prefix gain / loss vs K candidate |
|---:|---|---:|---:|---:|
| 1 | A_learning | 526/1/4 | 14/12/748 | 36/30 |
| 64 | A_learning | 77/7/3 | 5/7/736 | 10/14 |
| 128 | A_learning | 62/2/1 | 1/2/393 | 5/3 |
| 1 | R_reference | 54/3/5 | 16/6/434 | 59/14 |
| 64 | R_reference | 4/1/2 | 4/4/766 | 9/4 |
| 128 | R_reference | 11/0/4 | 7/1/643 | 14/2 |

A is the current learning pair; R is the separate correction pair. Each ordinary K candidate starts from that window's current KRC weights. The later candidates are not historical K64/K128 checkpoints. All local counts concern overlapping sampled-block occurrences, not independent requests or realized serving gains. Lower reference V alone does not establish successful transfer.

At step 64, reference V rises; lost reference-prefix occurrences change from 6 to 7. On learning A at that step, the correction loses 7 useful repairs and adds 3. These local measurements do not establish a monotonic preservation guarantee; the regressions remain visible separately from the final native result.

## Exposure and costs

K sampled anchors: **65096**; reference sampled anchors: **65096** across **256** additional reference-response presentations, each evaluated once before and once after K (512 intrinsic reference forwards in the complete run). Reference protection covers **484656** prefix occurrences, with triangular weight sum **2915155**. Ordinary K exposure is counted once per optimizer-boundary row. Reference exposure is recorded separately; this is not an equal-compute or equal-exposure comparison with K.

| Cost | Seconds | Scope |
|---|---:|---|
| Disposable startup | 37.667 | Separate; two forward/backwards, zero Adam |
| Fresh initialization | 19.772 | Before main event |
| Full KRC event | 674.201 | Training, references, correction, probes, checks and publication |
| Training windows | 640.685 | Nested in event |
| Reference capture/gradients and correction | 287.090 | Intrinsic added work; nested in event |
| Pre-step reference capture | 73.360 | Includes retained feature loading/capture |
| Post-K reference gradients | 193.682 | Nested intrinsic work |
| Direct parameter correction | 20.048 | Nested intrinsic work |
| Eighteen A/R diagnostic replays | 6.192 | Diagnostic only |
| State-integrity and reproduction checks | 6.621 | Instrumentation |
| Checkpoint persistence | 17.132 | Nested in event |
| Final export | 13.611 | Nested in event |
| Serving backend loading | 19.506 | Benchmark setup |
| All benchmark checkpoint activations | 280.493 | Benchmark-only switching; excluded from request latency |
| Benchmark weight-identity checks | 5.589 | Separate from activation and serving latency |
| Campaign process wall | 1358.423 | All execution including timing repeats |

Core work: **786/786 response forwards**, **482/≤512 backward equivalents**, **128/128 Adam steps**, and **126/≤128 corrections**. The **18/18** local diagnostic forwards are included. New qualification optimizer steps: **0**. Startup and actual head/layer recomputation counts are retained separately in SUMMARY.json.

The benchmark made **193** checkpoint publications (`{'K': 97, 'KRC': 96}`), not one per generation: consecutive executions can reuse the active checkpoint. Its activation sum is benchmark overhead, not a deployment charge per request. The first measured KRC activation took **1.627s**. Every publication is retained in ACTIVATIONS.csv.

Event time excluding diagnostic replays: **668.009s**; this still includes integrity checks and sparse A prefix capture. Component timers are nested and must not be added to the event total again. No fresh matched K training-cost measurement exists. Reusing feedback/features incurs no new collection or teacher-preparation work here; it does not erase their historical cost.

The ready-checkpoint serving gain on this panel is **0.447s**, while measured intrinsic reference/correction work is **287.090s**. This result does not establish an adaptation-inclusive speedup. There is no fresh matched K training-cost control, and no benefit is extrapolated to future requests.

Peak allocated/reserved GPU memory: **38.722 / 40.613 GB**.

## Scope and retained tables

This one-seed, one-event screen separates local preservation from final serving performance. It does not establish feedback-window robustness, generalization to a fresh panel, or an adaptation-inclusive online speedup. No tuning, subsequent event or replacement mechanism is implied.

- `analysis/SUMMARY.json`, `STEPS.csv`, `PROBE_SUMMARY.csv`, `PROBE_RESPONSES.csv`, `PROBE_EFFECTS.csv`, and `REPAIR_IDENTITIES.csv`.
- `analysis/LEARNING_EXPOSURE.csv`, `REFERENCE_EXPOSURE.csv`, `PER_REQUEST.csv`, `REPEAT_COMPARISON.csv`, and `ACTIVATIONS.csv`.
- Every timing repeat and native record remains under `raw/campaign_001/evaluation/`.
- Source/result audit, first-step reproduction and storage records are reported separately.

## Material Passport and reproducibility

- Status: completed and independently audited. The user authorized the ARS experiment-agent/run workflow, with separate code and result reviews.
- Evidence: 627 sealed source/input files, 1,367 verified retrieved files and 18,096 native blocks; no audit discrepancies. All 74 focused CPU tests passed. The 31 new tests were independently and remotely repeated; these counts overlap.
- Models: Qwen/Qwen3-8B target at `b968826d9c46dd6066d109eabc6255188de91218`; z-lab/Qwen3-8B-DFlash-b16 draft at `9b41424b7109f9c5413454f481b09a82b85333f4`.
- Learning: qualified BF16 AdamW/KL, constant LR 3e-4, weight decay 0.01, clipping at 1, microbatch 1/accumulation 2, Flex attention, up to 512 anchors per response, projection chunks of 8 and positional decay 7. The inherited horizon is unchanged; this runner stops after one 128-step event.
- Serving: pinned native runtime, greedy block size 16, 2,048 tokens or EOS, context limit 4,096, concurrency 1 and capture disabled. Full pins and effective configuration are in `raw/sealed/`.
- Supervisor: 2026-10-27 21:17:46–21:41:34 UTC, 23m48s, exit 0. Both A800 GPUs are idle; final KRC export/restart state and source/raw evidence remain.
- Code: 11 additive files in A800 `/data/sources/minhao/DFlash-Online`; 161 prior source files unchanged. Local implementation is in `worktree/`; see `IMPLEMENTATION.diff`.
- Retention: the completed inactive G optimizer state was retired under the standing policy; its exact optimizer continuation is unavailable. G export/results and KC state remain. Only verified redundant transfer archives were removed after this run. See `STORAGE_MAINTENANCE.json`, `POSTRUN_STORAGE.json` and `LOCAL_POSTRUN_STORAGE.json`.

This report closes the authorized screen. No subsequent run, tuning or project-stage change was made.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_event_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
