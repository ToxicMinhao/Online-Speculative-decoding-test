# One-event U/P prefix-ranking experiment — completed

## Material Passport

- Date: October 26, 2026. ARS experiment run/validation workflow.
- Status: completed, exit 0; independent artifact audit passed with no issues.
- Scope: one event per new arm, one seed, the existing 64-request GSM8K development panel. No tuning or additional event.
- Execution: A800 `/data/sources/minhao/DFlash-Online`.
- Evidence: [raw summary](raw/campaign_001/evaluation/report/SUMMARY.json), [independent audit](FINAL_AUDIT.md), [audit JSON](FINAL_AUDIT.json).
- Project stage remains Stage 1; this result does not select a mechanism.

## Finding

**The tested prefix-ranking modification did not provide a useful improvement over ordinary OSD in this screen.** P saved only six verification calls relative to O (0.1951%), while complete request time was effectively unchanged (0.0145% lower). P improved calls on 24 requests, worsened them on 25 and tied on 15. Its 45 calls saved on improving requests were largely offset by 39 extra calls on regressing requests.

P's total mean-time advantage over O was 0.007667 seconds across all 64 requests. P/O time reductions by repeat were −0.017648%, +0.028324% and +0.032866%. These measurements do not establish a practical serving improvement. Most of P's benefit relative to frozen was already obtained by ordinary OSD.

## What ran

The frozen target was Qwen/Qwen3-8B (`b968826d9c46dd6066d109eabc6255188de91218`); released draft was z-lab/Qwen3-8B-DFlash-b16 (`9b41424b7109f9c5413454f481b09a82b85333f4`). F and O reused their existing checkpoints. U and P each began with released weights, a fresh BF16 AdamW optimizer and the qualified seed-17 RNG state.

All learning arms use ordinary weighted forward KL. U/P add the strongest-noncanonical-rival softplus ranking term with lambda 1. P applies it only within each block's consecutive canonical-matching prefix predicted by an independent event-fixed reference. U spreads the same weighted coefficient mass uniformly across that response presentation. The existing loss denominator, response averaging and accumulation remain unchanged.

Both new arms reused the same 128 retained frozen-feedback responses/features and exactly the original presentation/anchor schedule: two shuffled passes, 256 presentations and 128 optimizer steps **per arm**. Response batch size 1 and accumulation 2 give **256 new main steps and 512 presentations in total**. Neither evaluation requests nor replays supplied training feedback.

Settings remained constant LR 3e-4, weight decay 0.01, Adam betas 0.9/0.999, epsilon 1e-8, gradient clipping 1, BF16 parameters/Adam/KL arithmetic, FlexAttention, up to 512 anchors per response, gamma 7 and eight-block objective chunks. The complete training sequence limit was 2,272 tokens. The inherited preset still contains the historical 7,424-step/58-event horizon and old default path labels; the one-event controller overrides execution paths and stops at exactly 128 steps per new arm. Those preset fields do not describe additional executed work.

Serving used pinned SGLang/FA3, greedy decoding, block size 16, 2,048 output tokens or EOS, context limit 4,096, prefill chunk 8,192 and concurrency 1. Each replay used exact saved inputs and valid cold caches. Three rotating-order repeats were run freshly for F/O/U/P, with capture disabled and no concurrent training or transfers. Request timing includes prefill, drafting, target verification and request cache work; it excludes training, loading and checkpoint activation.

## Paired serving results

| Method | Verification calls | Sum of mean request times (s) | Calls / 1,000 output tokens | Accepted draft tokens / call | Committed progress / call |
|---|---:|---:|---:|---:|---:|
| F: Frozen released DFlash | 3,140 | 54.069934 | 160.6385 | 5.2226 | 6.2048 |
| O: Ordinary OSD | 3,075 | 52.814925 | 157.3131 | 5.3554 | 6.3359 |
| U: OSD + uniform ranking | 3,082 | 53.012440 | 157.6713 | 5.3410 | 6.3215 |
| P: OSD + reference-prefix ranking | 3,069 | 52.807258 | 157.0062 | 5.3682 | 6.3483 |

Each request contributes one validated verification trajectory and the mean of its three timings. All methods generated the same 19,547 output tokens. Committed verification progress is 19,483 tokens, excluding the 64 initial seed tokens. Timings are repeated measurements of 64 distinct requests, not 768 independent examples. Aggregate reductions are ratios of summed measurements.

| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |
|---|---:|---:|---:|
| P versus O | +0.1951% | +0.0145% | 24 / 25 / 15 |
| P versus U | +0.4218% | +0.3870% | 28 / 22 / 14 |
| P versus F | +2.2611% | +2.3353% | 36 / 19 / 9 |
| O versus F | +2.0701% | +2.3211% | 40 / 19 / 5 |

Positive reductions mean fewer calls or less time. **P versus O is primary.** P versus U supplies the uniform-ranking control, but its small difference does not establish a useful selective-protection mechanism. All valid regressions remain included.

## Correctness and actual exposure

All 768 planned native generations completed; all 64 requests passed exact input/output token, termination, publication and cache checks. The first eight requests qualified within the existing budget. F/O counts reproduced retained controls; all repeats had consistent calls. Independent auditing reconstructed 37,098 verification blocks. No comparison was invalid, incomplete, unattempted or of unknown completion.

The completed controlled qualification's 27 checks were reused. Startup performed one separately counted ordinary warm-up (two presentations, two backwards, one temporary step), discarded its state, then constructed and validated fresh U/P. All three initial-state gates passed. No warm-up update became a serving checkpoint. The integration passed 25 focused CPU tests on both Fenrir and A800; the independent auditor has five passing fixture tests. A fixture-only NameError during local test development was fixed and its log retained; there was no GPU correctness failure in this campaign.

Each arm used 65,096 sampled anchors, 949,560 supervised-position occurrences and 429,510 protected-prefix occurrences. Reference-prefix weight was 251,420.569641 in both arms. Ranking coefficient mass was 251,420.567932 for U and 251,420.569641 for P, within the declared numerical check. Exposure sums use optimizer-boundary rows; intermediate accumulation metrics are not double-counted. Full presentation/anchor/keep-mask schedules match O exactly.

**Startup clarification (October 26 evidence review):** retained O remains a valid ordinary-OSD performance comparator: source and records show the same released initialization, fresh BF16 AdamW, training mode and learning settings, with no inherited warm-up learning. U/P used an additional discarded process warm-up and restored complete pre-sampling RNG from a fresh diagnostic O2 state. Controlled O/O/U0/P0 then matched exactly over the checked window; a prior ordinary O/O diagnostic also showed the discrepancy, with no output or verification-count effect on four tested requests. The precise internal cause and any effect on historical O's final endpoint remain unresolved. This limits attribution of the six-call P/O difference solely to ranking; it does not invalidate the measured comparison or require retraining because a full 128-step equivalence test is absent. The earlier failed gate remains failed and preserved. See the [source-backed startup review](startup_review/REVIEW.md) for magnitudes, code locations and the optional smallest additional diagnostic; no new experiment was run.

## Adaptation and experiment cost

| New cost | U | P |
|---|---:|---:|
| Loading | 19.081 s | 18.409 s |
| Initial-state control | 7.977 s | 7.895 s |
| Training | 382.763 s | 381.627 s |
| Checkpoint save | 19.501 s | 19.196 s |
| Export | 14.319 s | 14.445 s |
| Complete event | **419.419 s** | **417.917 s** |
| Reference setup/forward/head, nested within event | 67.446 s | 65.698 s |
| Teacher-head calculation, nested within training | 103.733 s | 103.796 s |
| Peak allocated GPU memory | 29.329 GB | 29.254 GB |
| Peak reserved GPU memory | 34.809 GB | 34.353 GB |

Training, save and export are parts of the event; reference and teacher-head work are nested costs and must not be added again. New feedback generation and teacher-backbone feature preparation were zero because existing artifacts were reused, not because those operations are intrinsically free. O was not retrained, so its historical training cost is not a freshly matched cost control.

The discarded startup warm-up took 47.756 s, including 22.280 s loading. Native backend loading took 20.009 s. Evaluation wall time was 1,896.182 s, including 1,234.589 s of benchmark checkpoint activation and 12.961 s native weight validation. The 705 activation receipts reflect repeated benchmark switching, not recurring deployment adaptation. Activation is excluded from the request-time table; no deployment-speedup claim is made from the diagnostic switching schedule.

Python campaign wall time was 2,860.061 s. The complete supervisor run, including preflight and CPU checks, took **48m37s**, 09:21:23–10:10:00 UTC. This is separate from the summed serving times. P's roughly 418-second update is not repaid by its 1.263-second serving saving against F over this panel, even before adding other adaptation costs. No adaptation-inclusive acceleration or future break-even claim follows.

## Evidence retained and stopping point

All 107 deployed source files, 132 retained O artifacts and 32 qualification dependencies passed the final unchanged-artifact check. Full U/P export/restart states, initial-state snapshots, feedback and prior qualification evidence remain on A800 under the approved roots. A 12.9 MB compact package containing 2,369 files was retrieved to Fenrir after timing ended. No cleanup or retirement occurred. Post-exit A800 free space was 193.222 GB; both GPUs were idle. Fenrir has about 82.9 GB free. The clean Fenrir DFlash-Research folder was unchanged.

- [Per-request results](raw/campaign_001/evaluation/report/PER_REQUEST.csv)
- [Every timing repeat](raw/campaign_001/evaluation/report/ALL_REPEATS.csv)
- [Repeat totals](raw/campaign_001/evaluation/report/REPEAT_TOTALS.csv)
- [Method totals](METHOD_TOTALS.csv), [paired comparisons](COMPARISONS.csv), [adaptation costs](ADAPTATION_COSTS.csv)
- [Raw measurements](raw/campaign_001/evaluation/MEASUREMENTS.jsonl), [activation receipts](raw/campaign_001/evaluation/ACTIVATIONS.jsonl)
- [Complete cost record](raw/campaign_001/COSTS.json), [configuration](raw/campaign_001/config.yaml), [source seal](SEAL.json)
- [Code changes](IMPLEMENTATION.patch), [implementation review](CODE_REVIEW.md), [independent final audit](FINAL_AUDIT.md)

This one-event, one-seed development result does not establish behavior over a persistent stream, other datasets or other ranking objectives. It does establish that the specified P addition did not yield a useful improvement over ordinary OSD here. No demonstrated new mechanism follows from this result. The authorized experiment ends with this report; no further update, tuning or campaign was launched. Revisiting this branch would require a concrete revised hypothesis and a separately agreed test, rather than treating the tiny observed difference as success.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
