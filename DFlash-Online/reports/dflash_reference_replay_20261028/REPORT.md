# KR: ordinary reference replay versus retained KRC

**KR used more verification calls than retained KRC on both fixed panels.**

Completed training: **512/512 presentations, 256/256 AdamW steps**. Native execution: **768/768 valid generations**. Independent audit: **passed**.

KR starts from released DFlash with fresh seed-17 AdamW and alternates ordinary K updates on each original A pair and its same-pass next-pair R. Both updates advance the same optimizer and training RNG. The response-level K normalization, current first-error selection, BF16 path, constant learning rate 3e-4, clipping and model/runtime pins are unchanged. There is no hinge, protected-prefix capture or direct correction.

This is a practical replay control: KR has twice the Adam steps and twice the ordinary K presentations of KRC. The comparison changes optimizer-state evolution and training exposure; it does not isolate correction direction or match computation exactly.

| Panel | KRC calls | KR calls | KR call reduction | Requests improved / regressed / tied | Gross calls saved / added |
|---|---:|---:|---:|---:|---:|
| development | 3005 | 3109 | -3.460899% | 14 / 44 / 6 | 44 / 148 |
| additional | 3125 | 3255 | -4.160000% | 9 / 44 / 11 | 23 / 153 |

Primary call reduction is `1 − ΣKR_calls / ΣKRC_calls`, with one validated trajectory per request after repeat-consistency checks. Every request and regression remains included. The two panels are reported separately.

| Panel | KRC complete-request time (s) | KR complete-request time (s) | Reduction |
|---|---:|---:|---:|
| development | 51.864162 | 53.764256 | -3.663597% |
| additional | 53.695815 | 55.876409 | -4.061012% |

Times above sum each request’s three-repeat mean. These are contemporaneous KR/KRC measurements with available checkpoints, after training closed. Native generation and cache maintenance below are components of complete-request latency; they must not be added again.

| Panel | Arm | Native-generation time (sum of means, s) | Cache-maintenance time (sum of means, s) | Accepted draft tokens | Progress / call |
|---|---|---:|---:|---:|---:|
| development | KRC | 50.649791 | 0.042829 | 16535 | 6.483527 |
| development | KR | 52.370573 | 0.041660 | 16426 | 6.266645 |
| additional | KRC | 52.688909 | 0.033433 | 16795 | 6.356480 |
| additional | KR | 54.823019 | 0.034509 | 16658 | 6.102611 |

| Panel | Repeat | KRC request time (s) | KR request time (s) | Reduction |
|---|---:|---:|---:|---:|
| development | 0 | 51.873996 | 53.955984 | -4.013548% |
| development | 1 | 51.864660 | 53.678800 | -3.497834% |
| development | 2 | 51.853830 | 53.657983 | -3.479306% |
| additional | 0 | 53.695579 | 55.894132 | -4.094477% |
| additional | 1 | 53.690128 | 55.865913 | -4.052487% |
| additional | 2 | 53.701740 | 55.869181 | -4.036074% |

## Training exposure, correctness and cost

The sealed schedule contains **130,192 sampled-anchor occurrences**. Each of the 128 responses appears twice as A and twice as R, using its own original per-pass anchors. R is training exposure, not held-out data. No evaluation request enters training.

The first A update exactly matched retained ordinary K training records (loss/exposure metrics, anchors, learning rate and counters). Fresh released model/Adam/RNG state was checked separately. This does not claim historical post-update tensor equality: that first-step export was not retained. The first A/R window used four actual presentations and two Adam steps, with no duplicate qualification work.

| KR cost | Seconds | Scope |
|---|---:|---|
| A updates | 327.862 | 128 ordinary updates; includes feature loading and bookkeeping |
| R updates | 328.094 | 128 ordinary updates with continuous Adam/RNG |
| Full KR event | 691.503 | Training, checks, persistence and export |
| Checkpoint persistence | 18.619 | Nested in event |
| Final export | 14.737 | Nested in event |
| Disposable startup | 38.677 | 2 forwards/backwards, 0 Adam steps |
| Fresh initialization | 20.703 | Before main event |
| Native backend loading | 21.507 | Benchmark setup |
| KRC full event, historical | 674.201 | Prior measurement, not fresh matched training time |
| Campaign process wall | 2160.595 | Includes benchmarks and switching; excludes supervisor pre/postflight |

Peak KR allocated/reserved GPU memory: **27.157/30.906 GB**. Historical KRC peaks were **38.722/40.613 GB**. All recorded training head/layer invocations, including recomputation, remain in TRAINING_SUMMARY.json; recomputation is not counted as another response presentation.

The historical KRC event includes 6.192 s of 18 diagnostic forwards; excluding those measured forwards gives 668.009 s. KR has no corresponding local probe cost. These are historical training-cost observations, not fresh paired timing or evidence of equal compute.

Component costs are nested, so do not add them to the event total again. Feedback collection and teacher preparation are reused from retained artifacts; there is no new cost for those phases here, but their historical cost is not erased. No fresh KRC training run was performed.

| Panel | Checkpoint publications | Publication time (s) | Native weight validation (s) | Evaluation wall (s) |
|---|---:|---:|---:|---:|
| development | 193 | 359.263 | 8.197 | 689.992 |
| additional | 193 | 355.293 | 6.743 | 694.895 |

Publication is benchmark switching overhead, not a charge once per deployed request. All activations are retained. The first eight development requests supplied the included 48-generation correctness gate; no extra generation warmups or optimizer qualifications were added. The supervisor ran from 2026-10-28 00:42:52 to 01:20:45 UTC (2,273 seconds), including pre/postflight, and exited 0.

## Validity and interpretation

| Panel | Valid / invalid / unknown / unattempted executions | Canonical output tokens | EOS / cap |
|---|---|---:|---:|
| development | 384 / 0 / 0 / 0 | 19547 | 64 / 0 |
| additional | 384 / 0 / 0 / 0 | 19928 | 64 / 0 |

Development outputs are checked against retained F; additional-panel outputs are checked against the retained fresh K1 canonicals from the seed20261028 confirmation. Both panels reproduce their retained per-request KRC counts and require repeat-consistent counts for both methods. The additional panel has permitted, disclosed older-project exposure. Neither panel is newly selected for this test.

Historical K1 call totals are 3,027 on development and 3,150 on the additional panel; they are context only, with no fresh K1 latency claim.

**This ordinary replay recipe did not replace KRC successfully.** KR requires more verification calls and more serving time on both panels; all six panel-repeat timing comparisons favor KRC. KR uses less peak training memory, but shows no event-time saving against the historical KRC measurement. KRC therefore gains support against this specific replay control. This does not isolate the correction’s causal role: optimizer-step count, optimizer-state evolution and training exposure also change.

The supported conclusion concerns these fixed feedback responses, checkpoints and two panels. It does not establish training-seed/window robustness, training-inclusive acceleration, overfitting as the cause of the regression, or a general advantage of correction over ordinary replay. All results stop at this prespecified comparison; no tuning, extra arms or subsequent event was launched.

## Evidence

- `IMPLEMENTATION.diff`, `REVIEW.md`, `CPU_TESTS.log`: additive code and independent review; 23 focused CPU tests (independent rerun overlaps).
- `raw/sealed/PLAN.json`, `REPLAY_SCHEDULE.json`, `REFERENCE_SCHEDULE.json`, `SEAL.json`: fixed sources, checkpoint identities, panels and bindings.
- `analysis/TRAINING_SUMMARY.json`, `PRESENTATIONS.csv`, `UPDATES.csv`, `ROLE_EXPOSURE_COSTS.csv`, `RESPONSE_EXPOSURE.csv`: all training exposure/cost accounting.
- `analysis/development/` and `analysis/additional/`: separate per-request, all-repeat and repeat-total tables; activations remain in each raw evaluation directory.
- `raw/campaign_001/evaluation/`: every raw native output, trace and validity record.
- `FINAL_AUDIT.md`, `RETRIEVAL_VERIFIED.json`, `COMPLETION.json`: independent audit, byte-verified retrieval and final status.

Final KR export: `/data/sources/minhao/runs/dflash_reference_replay_20261028/campaign_001/KR/events/event_001/export`. Recovery state: `/data/sources/minhao/runs/dflash_reference_replay_20261028/campaign_001/KR/training_state.pt`. All existing checkpoints and evidence remain retained.

Material Passport: computational experiment; ARS executor/monitor and independent audit; retained private requests/outputs; no stage advancement.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_replay_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
