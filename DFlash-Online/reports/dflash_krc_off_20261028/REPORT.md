# KRC2 correction-removal control

Execution: **completed**. Development and additional panels remain separate.

KRC2_OFF restores the exact corrected KRC1 weights, AdamW moments/counters, configuration/history and training RNG at step 128. Only then are reference processing, correction and optional probes disabled. Ordinary K continues on the unchanged event-2 pool to step 256. Retained KRC2 is the contemporaneously evaluated control.

**The call-count comparison is mixed across panels.** This test does not establish a consistent endpoint benefit from event-2 correction.
Differences are descriptive for these fixed checkpoints and requests. Small or mixed differences do not establish equivalence or a universally optimal event-1-only correction policy.

| Panel | OFF calls | KRC2 calls | Call reduction favoring KRC2 | Request-time reduction | KRC2 wins / losses / ties | Gross saved / added |
|---|---:|---:|---:|---:|---:|---:|
| development | 2960 | 2967 | -0.2365% | -0.2531% | 22 / 17 / 25 | 26 / 33 |
| additional | 3138 | 3116 | 0.7011% | 0.5715% | 23 / 14 / 27 | 40 / 18 |

Primary reduction is `1 − ΣKRC2_calls / ΣOFF_calls`; positive values favor retaining correction. Counts use one validated trajectory per request after repeat-consistency checks. Ratios use summed counts, not averages of request-level percentages.

| Panel | Retained KRC1 calls | OFF calls | OFF reduction versus KRC1 | Retained K2 calls (context) |
|---|---:|---:|---:|---:|
| development | 3005 | 2960 | 1.4975% | 2973 |
| additional | 3125 | 3138 | -0.4160% | 3136 |

Historical context uses count measurements on the same valid cohort. K2 and KRC1 are not freshly generated or timed here.

| Panel | Arm | Complete request (s) | Native generation (s) | Cache maintenance (s) | Accepted draft tokens | Output tokens |
|---|---|---:|---:|---:|---:|---:|
| development | KRC2_OFF | 51.221521 | 50.042341 | 0.041649 | 16582 | 19547 |
| development | KRC2 | 51.351186 | 50.133773 | 0.042040 | 16575 | 19547 |
| additional | KRC2_OFF | 54.049624 | 53.025694 | 0.043219 | 16787 | 19928 |
| additional | KRC2 | 53.740707 | 52.666200 | 0.051941 | 16808 | 19928 |

Request totals sum three-repeat means. Native-generation and cache-maintenance times are nested components of complete request latency. Loading and checkpoint switching are separate.

| Panel | Repeat | OFF request time (s) | KRC2 request time (s) | KRC2 reduction |
|---|---:|---:|---:|---:|
| development | 0 | 51.278328 | 51.365215 | -0.1694% |
| development | 1 | 51.185846 | 51.351339 | -0.3233% |
| development | 2 | 51.200389 | 51.337003 | -0.2668% |
| additional | 0 | 54.047137 | 53.730048 | 0.5867% |
| additional | 1 | 54.060662 | 53.723694 | 0.6233% |
| additional | 2 | 54.041072 | 53.768379 | 0.5046% |

Per-repeat native-generation and cache-maintenance totals are also retained in `REPEAT_TOTALS.csv`.

## Restoration, training and cost

Exact KRC1 live restoration: **True**. First actual A-update reproduction: **passed**. This first update is part of the main event; only its pre-correction K metrics, anchors and counters are matched. No later trajectory equality is required.

Actual new work: **128 AdamW steps, 256 presentations, 256 core response forwards and 256 backwards**. Global steps: 128→256. Reference/correction work: `{"diagnostic_response_forwards": 0, "direct_corrections": 0, "post_k_reference_response_forwards": 0, "pre_reference_response_forwards": 0, "protected_prefix_captures": 0, "reference_backwards": 0, "replay_updates": 0}`.

Ordinary sampled-anchor occurrences: 66938. Exposure statistics sum 128 complete accumulation-boundary rows once. They do not add the individual presentation metrics again. No reference-response exposure is introduced.

| Cost | Seconds |
|---|---:|
| event_wall_s | 375.987 |
| training_windows_s | 336.097 |
| checkpoint_save_s | 21.219 |
| export_s | 15.409 |
| journal_s | 0.072 |
| first_A_check_s | 0.004 |
| startup_s | 44.332 |
| snapshot_validation_s | 6.190 |
| loading_and_live_resume_s | 27.223 |
| backend_loading_s | 21.758 |
| actual_campaign_wall_s | 1892.305 |

Peak allocated/reserved memory: 27.347 / 29.931 GB.

Qualified disposable startup is separately counted: two response forwards/backwards and zero Adam steps. Core response counts are separate from projection/module invocations and recomputation. Startup, first-A journal checks, persistence and export are recorded separately; nested costs must not be added twice.

Historical KRC2 training costs are context, not fresh matched overhead measurements. Serving gains exclude adaptation cost; historical-versus-fresh serving latency is not a learning curve.

| Panel | Valid / invalid / unknown / unattempted executions | Publications | Switching time (s) | Weight checks (s) |
|---|---:|---:|---:|---:|
| development | 384 / 0 / 0 / 0 | 193 | 377.403 | 8.124 |
| additional | 384 / 0 / 0 / 0 | 193 | 380.477 | 7.616 |

Both panels require exact retained canonical output tokens and termination, repeat-consistent counts, and reproduction of retained per-request KRC2 verification counts. Only the first eight development requests provide the included 48-execution gate. Training closes before timed evaluation; no additional generation warmups are used.

This control isolates removal of event-2 reference processing and correction from the same KRC1 state. It does not establish seed robustness, equivalence, training-inclusive acceleration or a general optimal correction schedule. No tuning or further events follow from this report.

Evidence: `analysis/SUMMARY.json`, `TRAINING_SUMMARY.json`, `PRESENTATIONS.csv`, `WORK_COUNTERS.csv`, panel-specific request/repeat tables, and raw resume/migration/first-A receipts and native outputs under `raw/campaign_001`.

Historical cost context: the retained KRC2 event took 696.634 s, including 297.366 s of reference/correction work. These earlier measurements are not a contemporaneous training-cost control.

Implementation and reproducibility: [independent implementation review](REVIEW.md), [implementation diff](IMPLEMENTATION.diff), [sealed actual-path plan](raw/sealed/PLAN.json), [configuration](raw/sealed/config.yaml), [model pins](raw/sealed/MODEL_PIN.json), [runtime pins](raw/sealed/RUNTIME_PIN.json), [full cost receipt](raw/campaign_001/COSTS.json), [first-update check](raw/campaign_001/off/FIRST_A.json), [retained final checkpoint/recovery identities](raw/FINAL_ARTIFACTS.json). Nineteen focused CPU tests passed locally and on A800. Inherited training and inference source remained unchanged.

Both checkpoints use Qwen3-8B with the qualified DFlash implementation, BF16, greedy block-16 decoding, 2,048 output tokens or EOS, context limit 4,096, concurrency one and capture disabled. The original capped feedback response was preserved. Ordinary training sampled 66,938 anchor occurrences; selection admitted 556454 of 977434 available proposal occurrences. These are overlapping training occurrences, not independent examples.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_off_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
