# KC: one-event K-plus-correction experiment

**KC increased verification calls by 0.2643%: 3,027 calls for K1 versus 3,035 for KC.** The experiment did not demonstrate better drafting than ordinary K. The signed call-reduction metric is -0.2643%.

Summed mean request times are 52.101776s for K1 and 52.076363s for KC (signed reduction +0.0488%). This descriptive timing difference is not evidence of a deployment speedup. These are fresh comparisons with available checkpoints on the reused development panel.

KC starts from released DFlash with the qualified fresh seed-17 optimizer, uses the original 128 responses/features and exact two-pass schedule, and applies the tested correction after each ordinary K step. K1 is the unchanged retained one-event checkpoint. No new feedback or F/O/K retraining is included. Models, BF16 arithmetic, native AdamW, constant LR 3e-4, weight decay 0.01, clipping 1, response accumulation 2, Flex attention, full history and greedy block-size-16/2,048-token-or-EOS serving remain fixed.

First-window exact reproduction: **True**. Completed windows: **128/128**; actual native Adam calls: **128**; committed training presentations: **256**.

## Fresh native comparison

Valid requests: 64/64; valid timed generations: 384/384. Outputs and termination are checked against retained F; counts must be consistent across repeats. Training closes before timed evaluation.

| Arm | Verification calls | Sum of mean request time (s) | Accepted draft tokens | Calls / 1,000 tokens | Committed progress / call |
|---|---:|---:|---:|---:|---:|
| K | 3027 | 52.101776 | 16513 | 154.8575 | 6.4364 |
| KC | 3035 | 52.076363 | 16506 | 155.2668 | 6.4194 |

Requests improved/regressed/tied on verification calls: **16/23/25**. Counts use one validated trajectory per request; latency uses three per-request repeats and pooled sums. Publication/loading and training are excluded from request latency. Historical F is the output reference, not a fresh latency comparator.

Canonical outputs: **19547 tokens**; EOS requests: **64**; output-cap hits: **0**. Per-request lengths, all repeats and invalid/incomplete slots remain in the raw evaluation tables.

## Correction behavior

Applied correction decisions: **84/128** (65.62%). Actual nonzero BF16 corrections: **84/128**; applied but rounded entirely to zero: **0**. Positive-V/zero-V windows: **84/44**. Branches: `{'applied': 84, 'zero_violation': 44}`. Median realized correction/K-step norm ratio across all windows: **0.001313** (range 0.000000–0.008120); among applied corrections: **0.002273**.

V is measured at the actual post-K candidate on the fixed pre-step prefix. It is not a held-out loss. Post-correction V and prefix outcomes are measured only at steps 1, 64 and 128. Every step, no-op, zero-gradient outcome, BF16-realized norm and component cost remains in STEPS.csv.

| Step | Candidate | V | Useful repairs | Earlier errors / blocks | Lost old-prefix tokens / original prefix | Net prefix gain | Wrong ties |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ordinary_K_candidate | 0.003567589 | 527 / 632 | 47 / 774 | 153 / 5236 | +1740 | 1 |
| 1 | corrected_KC | 0.002353957 | 533 / 632 | 41 / 774 | 113 / 5236 | +1807 | 3 |
| 64 | ordinary_K_candidate | 0.000015346 | 65 / 650 | 2 / 748 | 2 / 4522 | +179 | 1 |
| 64 | corrected_KC | 0.000005115 | 67 / 650 | 1 / 748 | 1 / 4522 | +197 | 0 |
| 128 | ordinary_K_candidate | 0.000006298 | 51 / 287 | 2 / 396 | 10 / 3215 | +99 | 1 |
| 128 | corrected_KC | 0.000006298 | 52 / 287 | 2 / 396 | 10 / 3215 | +100 | 1 |

| Step | Useful repairs retained / lost / new | Blocks improve / regress / tie versus K | Prefix gain / loss versus K |
|---:|---:|---:|---:|
| 1 | 527 / 0 / 6 | 17 / 8 / 749 | 77 / 10 |
| 64 | 63 / 2 / 4 | 5 / 2 / 741 | 20 / 2 |
| 128 | 51 / 0 / 1 | 1 / 0 / 395 | 1 / 0 |

Later probe K candidates start from the same current KC pre-step weights; they are not historical K64/K128 checkpoints. These are overlapping sampled-block occurrences on training responses, not independent examples or native serving results. REPAIR_IDENTITIES.csv identifies each retained/lost/new repair by step, response, block and anchor. A lower V does not by itself demonstrate useful learning. The repair table also exposes lost useful repairs; net improvement does not imply that every repair survives.

## Exposure and cost

K supervision: **65096 sampled valid anchors**; available/selected occurrences **949560/534329**. Correction protection covers **483389 old-prefix occurrences**, with total triangular weight 2905424. Exposure totals use optimizer-boundary rows once; intermediate accumulation rows are not double-counted.

| Cost | Seconds | Scope |
|---|---:|---|
| Disposable startup | 37.495 | Two forward/backwards, zero Adam; separate from core budget |
| Fresh model and exact initialization | 19.718 | Separate setup |
| Complete KC event | 534.715 | Training, correction, probes, checks, persistence and export |
| Training windows | 501.377 | Nested inside event |
| Post-K gradients and direct corrections | 151.595 | Intrinsic added work; nested inside training |
| Six corrected-state rechecks | 3.719 | Diagnostic only; nested inside training |
| State integrity and step-1 reproduction | 4.209 | Instrumentation nested inside training |
| Checkpoint persistence | 17.151 | Nested inside event |
| Final checkpoint export | 13.508 | Nested inside event |
| Native backend loading | 19.507 | Benchmark setup |
| All benchmark checkpoint switches | 268.647 | Benchmark only, excluded from serving |
| Actual campaign wall | 1205.141 | All current execution work, including repetitions |

For explicit cost accounting, the measured KC event excluding corrected-state rechecks costs **530.996s**. Adding one measured KC activation (1.391s) and one reconstructed 64-request KC serving pass (52.076s) gives **584.464s**. This arithmetic retains integrity/persistence/export overhead, excludes runtime setup and historical feedback preparation, and excludes all benchmark-only extra repetitions/switches. It is not a matched end-to-end online speedup estimate: retained K has no fresh matched training-time measurement.

The original feedback and teacher features are reused: zero new collection/preparation in this run does not mean those artifacts had zero historical cost. Intrinsic correction timers and capture timers are nested; do not add them to event time again. Historical K training time must not be presented as a contemporaneous measure of KC overhead.

Training-process peak allocated/reserved GPU memory: **40.898 / 43.734GB**. Actual core counters: `{'K_backward_attempted': 256, 'K_backward_completed': 256, 'K_backward_failed': 0, 'Adam_attempted': 128, 'Adam_completed': 128, 'Adam_failed': 0, 'presentations_attempted': 256, 'presentations_completed': 256, 'presentations_failed': 0, 'windows_completed': 128}`; response/backward work: `{'original_response_forwards': 256, 'current_objective_chunks': 16500, 'post_k_response_forwards': 256, 'post_c_response_forwards': 6, 'post_k_response_attempted': 256, 'post_c_response_attempted': 6, 'violation_backward_attempted': 115, 'violation_backward_completed': 115, 'violation_backward_failed': 0, 'direct_corrections': 84, 'post_k_objective_chunks': 12267, 'post_c_objective_chunks': 243}`. New qualification optimizer steps: 0. Startup/chunk/recomputation counters are retained separately in SUMMARY.json.

## Limits and artifacts

This is one prespecified event with one seed and a previously used development panel. Local repair preservation and final native gains remain distinct measurements. The sampled local effects do not establish transfer or robustness, and this screen does not identify the cause of any transfer limitation. No tuning or follow-up run is implied by this report.

- `analysis/STEPS.csv`, `PROBE_SUMMARY.csv`, `PROBE_RESPONSES.csv`, `PROBE_EFFECTS.csv`, `REPAIR_IDENTITIES.csv`, `EXPOSURE.csv` and `SUMMARY.json`.
- `raw/campaign_001/evaluation/report/PER_REQUEST.csv`, `ALL_REPEATS.csv` and `REPEAT_TOTALS.csv`.
- `raw/campaign_001/FIRST_WINDOW_REPRODUCTION.json`, `CORRECTION_EVIDENCE.json`, training receipts and raw native records.
- Independent source/result audit and integrity/storage records are reported separately.

## Verification, storage and code

The independent [final audit](FINAL_AUDIT.md) passes with zero discrepancies or missing evidence: all 1,358 retrieved files verify, all 593 sealed source/input files pass both preflight and postflight, and 18,186 native verification blocks are reconstructed. The complete budget is 128 Adam steps, 256 presentations, 518 core forwards, 371 core backwards and 384 native generations. Startup adds two forwards, two backwards and zero Adam steps. The supervisor ran for 21m14s (2026-10-27 19:46:16–20:07:30 UTC), exiting with code 0.

Seventy focused local CPU tests and 30 remote integration tests passed; these suites overlap. The 30 new tests also passed independent review. The [implementation diff](IMPLEMENTATION.diff) contains nine additive source/test files. The tested correction, ordinary K objective and upstream model/training/inference files are unchanged. Fenrir DFlash-Research remains untouched.

The final KC export and complete recovery state remain under A800 `/data/sources/minhao/runs/dflash_kc_event_20261027/campaign_001/KC/`. Existing feedback, K1 and all prior artifacts remain retained. Both GPUs are idle. After verified retrieval, only redundant upload/evidence archives were removed; no replay or resume capability was lost. At the final storage check, A800 had 27.51 GB free and Fenrir 82.13 GB free. Exact paths, reasons and before/after readings are in [STORAGE_MAINTENANCE.json](STORAGE_MAINTENANCE.json).

- [Per-request native results](raw/campaign_001/evaluation/report/PER_REQUEST.csv) and [every timing repeat](raw/campaign_001/evaluation/report/ALL_REPEATS.csv).
- [Per-step corrections](analysis/STEPS.csv), [probe metrics](analysis/PROBE_SUMMARY.csv), [repair identities](analysis/REPAIR_IDENTITIES.csv), and [training exposure](analysis/EXPOSURE.csv).
- [Full analysis](analysis/SUMMARY.json), [independent machine-readable audit](FINAL_AUDIT.json), and [source review](REVIEW.md).

The event is complete and stopped. No tuning, additional event, fresh-panel run or mechanism change was performed.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_kc_event_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
