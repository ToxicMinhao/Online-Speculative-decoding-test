# KRC event-2 continuation versus retained K2

Execution: **completed**. This report keeps the two fixed panels separate.

**Answer: KRC’s small advantage persists after the second event, and KRC itself improves on both panels. However, the advantage over K shrinks: ordinary K gains more from this additional feedback window.** This supports persistence through the two tested events, not a growing advantage or a generally established online-adaptation method.

KRC2 resumes its own corrected KRC1 weights, Adam state/counters and RNG at step 128. It uses the retained shared event-2 feedback and original ordinary order/anchors, then stops at step 256. No new feedback or K2 training is performed. Reference correction is unchanged; optional learning-prefix and post-correction probes are disabled.

| Panel | K2 calls | KRC2 calls | KRC2/K2 reduction | Improved / regressed / tied | Gross calls saved / added |
|---|---:|---:|---:|---:|---:|
| development | 2973 | 2967 | 0.2018% | 25 / 19 / 20 | 43 / 37 |
| additional | 3136 | 3116 | 0.6378% | 31 / 20 / 13 | 56 / 36 |

| Panel | Historical KRC1/K1 advantage | KRC2/K2 advantage | Change (percentage points) | KRC2/KRC1 call reduction | K2/K1 call reduction |
|---|---:|---:|---:|---:|---:|
| development | 0.7268% | 0.2018% | -0.5250 | 1.2646% | 1.7839% |
| additional | 0.7937% | 0.6378% | -0.1559 | 0.2880% | 0.4444% |

The relative advantage over K and the change in KRC’s own call count are distinct. The historical entries above use counts on the same valid request cohort; historical-versus-fresh latency is not interpreted as learning progress.

| Panel | Arm | Complete-request time (s) | Native generation (s) | Cache maintenance (s) | Accepted draft tokens |
|---|---|---:|---:|---:|---:|
| development | K2 | 51.523534 | 50.349077 | 0.037779 | 16568 |
| development | KRC2 | 51.445192 | 50.225593 | 0.041815 | 16575 |
| additional | K2 | 54.191128 | 53.111198 | 0.051856 | 16789 |
| additional | KRC2 | 53.853610 | 52.779670 | 0.045618 | 16808 |

Request totals sum each request’s three-repeat mean. Native generation and cache maintenance are nested components of complete request latency, not additional costs.

| Panel | Repeat | K2 complete time (s) | KRC2 complete time (s) | Reduction |
|---|---:|---:|---:|---:|
| development | 0 | 51.581909 | 51.451855 | 0.2521% |
| development | 1 | 51.495703 | 51.442952 | 0.1024% |
| development | 2 | 51.492990 | 51.440769 | 0.1014% |
| additional | 0 | 54.182707 | 53.868619 | 0.5797% |
| additional | 1 | 54.170733 | 53.858538 | 0.5763% |
| additional | 2 | 54.219945 | 53.833673 | 0.7124% |

## Restoration, training and correction

Exact live resume: **True**. Training: **128 new Adam steps / 256 ordinary presentations**, global steps 128→256. Actual core work: 768 response forwards / 482 backward equivalents; optional diagnostic forwards: 0.

Reference exposure: **256 presentations of 128 distinct saved responses**, evaluated with 256 pre-update and 256 post-K forwards (512 reference phase forwards total). The two phases do not double the number of examples. These reference occurrences are additional training exposure.

Correction decisions: 124 applied; 124 have a nonzero BF16-realized displacement; 0 applied corrections round to zero. Branch counts: `{"applied": 124, "zero_violation": 4}`. Median active correction-to-K norm ratio: 0.002141.

V and reference-prefix statistics are measured after K and before correction. No post-correction replay is run; they do not measure the correction’s actual final prefix effect.

Response-forward totals are separate from actual chunk and module invocation counts in `WORK_COUNTERS.csv`. Post-K reference chunk entries include checkpoint recomputation. The learning-phase vocabulary-head counter includes both draft and teacher projections plus recomputation; it is not a count of unique training blocks. Zero sparse learning-probe chunk counts mean that optional instrumentation was disabled, not that K had no loss work.

Ordinary supervised exposure uses boundary accumulation metrics once. Reference examples are additional training exposure, not held-out evaluation. Detailed counts, every correction decision, original anchors and reference bindings remain in the machine-readable tables.

| Cost | Seconds |
|---|---:|
| event_wall_s | 696.634 |
| training_windows_s | 656.807 |
| intrinsic_reference_correction_s | 297.366 |
| pre_reference_s | 77.806 |
| post_K_reference_s | 199.875 |
| correction_s | 19.686 |
| teacher_head_s | 106.586 |
| first_error_selection_s | 14.576 |
| integrity_s | 6.870 |
| checkpoint_save_s | 20.454 |
| export_s | 15.728 |
| startup_s | 42.603 |
| snapshot_validation_s | 6.365 |
| loading_and_live_resume_s | 27.228 |
| backend_loading_s | 22.008 |
| actual_campaign_wall_s | 2218.705 |

Peak allocated/reserved GPU memory: 36.635 / 37.845 GB.

Training, persistence, intrinsic correction and integrity timers are nested. Startup is separately counted as two disposable forward/backwards and zero Adam steps. Feedback/teacher features are reused; their historical preparation cost is not erased. No historical K training time is presented as a fresh matched overhead control.

| Panel | Valid / invalid / unknown / unattempted | Publications | Publication time (s) | Weight checks (s) |
|---|---:|---:|---:|---:|
| development | 384 / 0 / 0 / 0 | 193 | 382.007 | 8.348 |
| additional | 384 / 0 / 0 / 0 | 193 | 380.329 | 7.690 |

Development output validity uses retained F tokens/termination and retained K2 counts. Additional-panel validity uses retained first-K1 tokens/termination, with no historical K2 count gate. Both arms require repeat-consistent counts. The first eight development requests provide the included 48-generation gate. Loading/publication are outside request latency; benchmark switching is not charged once per deployed request.

This is a shared-feedback continuation, not two independently measured online deployments. The measurements concern these checkpoints, two feedback windows and fixed panels; they do not establish seed robustness or training-inclusive acceleration. No later event, retuning or new comparison is authorized by this report.

One inherited runtime-compatibility metadata label still names the earlier K/G-disabled/G-enabled qualification trials. That label is stale: the actual startup and campaign receipts for this run record two disposable response forward/backwards, zero startup Adam steps, and the single resumed KRC2 trajectory. The original metadata is preserved; it is not counted as additional execution.

Evidence: `analysis/SUMMARY.json`, `TRAINING_SUMMARY.json`, `PRESENTATIONS.csv`, `CORRECTION_STEPS.csv`, `REFERENCE_EXPOSURE.csv`, `WORK_COUNTERS.csv`, panel-specific request/repeat tables; raw resume checks, schedule checks, per-window records and native outputs remain under `raw/campaign_001`.

Execution retains the qualified KRC1 source versions. The earlier O/K run used pre-guard versions of `objective.py` and `training.py`; their original bytes and the current-version mapping are retained in `HISTORICAL_SOURCE_BINDINGS.json` and the source diff. The first preparation stopped at this version check before GPU work; resolving the provenance mapping did not change the K loss or KRC correction. Exact KRC1 restoration is checked, without claiming full-trajectory bitwise equality with historical K2. K2/KRC2 serving timings above were measured freshly together.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_continuation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
