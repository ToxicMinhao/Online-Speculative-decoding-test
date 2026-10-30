# KW-v1 production cost and controller review

Review date: 2026-10-30. Verification status: **ANALYZED**. This is a same-team retained-evidence review by the numerical-core author, not independent implementation review or a new experimental reproduction. No remote operation, model execution, optimizer application, or GPU test was performed for this review.

The recorded KW training time was **1,802.709 s versus K's 1,277.774 s: +524.935 s (+41.082%)**, or **8.749 extra minutes** across events 9–12. Both arms performed 512 committed optimizer updates and 1,024 response presentations. KW additionally performed 1,024 disposable optimizer previews and 504 delayed credits. These extra operations use captured gradients and optimizer state; the implementation adds no model forward/backward passes. This is a measured cost of this implementation and campaign, not a demonstrated deployment amortization result.

## Material passport and audit method

Execution source is commit `708e974e06f5d9f4b186c32607c16b37cbd4b14f`, as bound in [campaign COSTS](raw/campaign_001/COSTS.json). The source-level timing review used `online_feedback/kw_training.py`, `online_feedback/training.py`, `online_feedback/response_weighting.py`, and `experiments/kw_v1/campaign.py` in the dedicated DFlash-Online checkout. The parent review separately verifies source/input seals and retrieval identities; this report does not relabel that work as an independent hash audit by this reviewer.

Inputs were all four `KW/events/event_009` through `event_012/KW_WINDOWS.jsonl` and `KW_WORK.jsonl` files, all 12 arm/event `RESULT.json` files, all 12 `resume_*/COSTS.json` receipts, and [the generated cost report](raw/report/COSTS.json). Local Python standard-library parsing reconstructed cost sums, boundaries, identities, controller state evolution, and recoverable feature coordinates. [Machine-readable checks and values](COST_CONTROLLER_CHECKS.json) retain 6,660 structural/numerical checks and 4,608 feature-coordinate checks, with zero discrepancies. These are retained-record checks, not tests of the original model/optimizer arithmetic.

## Training and persistence costs

| Event | K training, s | KW training, s | Extra, s | Extra relative to K |
|---|---:|---:|---:|---:|
| 9 | 326.158 | 457.610 | 131.451 | 40.303% |
| 10 | 327.502 | 458.418 | 130.917 | 39.974% |
| 11 | 313.071 | 443.957 | 130.886 | 41.807% |
| 12 | 311.044 | 442.724 | 131.681 | 42.335% |
| Total | **1,277.774** | **1,802.709** | **524.935** | **41.082%** |

Mean measured training cost was 2.496 s per K optimizer window and 3.521 s per KW window, an extra 1.025 s per window. Each window has two response presentations. The total O training time was 1,224.930 s; K is the appropriate objective-matched cost comparator here.

`training_s` sums each response timer around batch preparation and `core.train_step`, through synchronized metric materialization. It includes native optimizer work, objective work, KW controller work, and associated instrumentation. It excludes each response's preceding feature-file load, checkpoint persistence/export, and initial loading/restoration. All 12 totals reproduce the response-level timing sums exactly.

| Recorded scope, summed over four events | K, s | KW, s | Difference, s |
|---|---:|---:|---:|
| Event wall time, including training/save/export | 1,410.207 | 1,937.879 | +527.671 |
| Loading and resume validation, before event | 147.038 | 176.908 | +29.871 |
| Checkpoint save, **inside event wall time** | 67.405 | 67.996 | +0.591 |
| Export, **inside event wall time** | 54.376 | 56.278 | +1.902 |
| Recorded saved-state load/validation component, after event | 20.884 | 22.003 | +1.118 |

The event-wall comparison is +37.418%, a different denominator from training time. Do not add training, save, and export again to the event total. Loading is separately measured. The saved-state validation timer is one recorded post-event component; it does not cover every later boundary comparison and hash operation, so these rows are not an exhaustive whole-process runtime decomposition. Shared feedback generation, serving evaluation, and prior qualification are outside the K/KW training comparison.

## What the KW timers measure

| KW timer | Four-event total, s | Mean per window, s | Scope |
|---|---:|---:|---|
| `ordinary_training_s` | 1,265.873 | 2.472408 | Window entry to pre-optimizer boundary; includes observed ordinary forwards/backwards, feature observations and gradient capture, plus inter-response work |
| `previews_s` | 295.507 | 0.577161 | Two scratch native optimizer previews, copying/checking state, secant construction and statistics |
| `credit_and_controller_s` | 103.798 | 0.202730 | Allocation, ordinary-gradient norm, delayed dot credit/RMS/controller update, and work logging |
| `weighting_s` | 96.714 | 0.188894 | Weighted live-gradient construction and weighted-gradient norm bookkeeping |
| `live_update_s` | 34.047 | 0.066498 | Native clipping/AdamW step, work logging, full parameter/Adam-state finite checks, and final synchronization |
| `capture_host_s` | 1.019 | 0.001991 | Host time submitting hook clones/additions; **nested inside ordinary training** |
| `window_observed_s` | 1,804.373 | 3.524165 | Inclusive observed window wall time; **overlaps all preceding components** |

The two-preview timer accounts for 16.392% of KW's recorded training time. Preview, credit/controller, and weighting timers together total 496.018 s. They are measured implementation scopes, not isolated kernel benchmarks or an exact causal decomposition of the 524.935 s K/KW difference. In particular, the credit and weighting timers include full-gradient norms; the live-update timer includes validation beyond the native optimizer alone.

`capture_host_s` does not synchronize each hook and does not isolate GPU copy execution time. It must not be added to ordinary training a second time. The five sequential main window scopes sum to 1,795.938 s, leaving 8.435 s within the inclusive window timers for bookkeeping outside those scopes. Inclusive window time exceeds summed response `training_s` by 1.663 s because the window spans inter-response loading/bookkeeping while the response timers have different boundaries. Objective `teacher_head_s` and `selection_s` are also inside training; adding them again would double-count work.

## Memory and its scope

| Maximum recorded post-load allocation across events | K | KW |
|---|---:|---:|
| Allocated, decimal GB | 29.251 | 38.577 |
| Reserved, decimal GB | 34.874 | 42.490 |
| Allocated, GiB | 27.241905 | 35.927340 |
| Reserved, GiB | 32.478516 | 39.572266 |

KW's maximum allocated memory is 9.326 decimal GB above K's maximum; reserved memory is 7.617 GB higher. These are per-arm maxima, potentially at different events and phases, not an isolated measurement of one controller buffer.

**Scope correction:** the receipts label these peaks “Training-only peak after loading.” Source resets peak statistics after loading, but samples the maxima only after `train_event`, checkpoint/export, saved-state loading, and live-boundary validation. The accurate description is **post-load training/persistence/validation peak**. Initial loading peaks are separately recorded. The KW window-end snapshots range from 25.973 to 25.989 GB allocated and 40.429 to 42.490 GB reserved; they are snapshots, not transient peaks. No CPU peak-memory measurement or isolated hook/preview memory peak is available here.

## Controller accounting and numerical history

All four events contain exactly 128 KW windows, 256 optimizer previews, 126 delayed credits, and 128 committed live updates. Started/completed work receipts agree with these counts. The complete history has 512 windows, 1,024 previews, 504 credits, and eight pass-final eligibility discards. Each 64-window pass has no incoming credit at its first window, then 63 credits. Every credit names the immediately preceding window in the same pass. All pass-final windows still compute both previews before discarding pending eligibility.

The controller starts at zero. Exactly the first two production windows have `a=0` and weights `(0.5, 0.5)`; the first learned allocation is global step 1027. Window two uses its entry state before consuming window one's credit. Phi/RMS/count state remains continuous across pass and event boundaries, while pending credit does not cross those boundaries. Final credit counts are 126, 252, 378, and 504. All event-final states agree with the corresponding last window.

| Event | Minimum a | Maximum a | Final controller norm |
|---|---:|---:|---:|
| 9 | −0.002815771 | +0.001974728 | 0.023788453 |
| 10 | −0.006280615 | +0.003528484 | 0.018298018 |
| 11 | −0.003000887 | +0.003122005 | 0.019955184 |
| 12 | −0.003065231 | +0.001504137 | 0.011821917 |

Across all windows, response weights stay in **[0.4937193848, 0.5062806152]**. Median `abs(a)` is 0.0003831624; there are 248 positive, 262 negative, and two zero allocations. There is no allocation saturation, clipped secant endpoint, controller norm projection, zero secant, zero credit, or live-gradient clipping in the records. Normalized credit is clipped in 98/504 opportunities. Credits are positive 230 times and negative 274 times. Preview endpoints differ on 1.2304%–4.1149% of parameter coordinates; this endpoint comparison must not be interpreted as the fraction changed by the much smaller actual learned allocation or as a beneficial-change fraction.

All 1,024 feature vectors have five finite FP32-representable values in [0,1]. First-response selected-weight fractions (512) and the other four feature coordinates for both responses (4,096) reconstruct exactly from retained response metrics, individual losses and response lengths, totaling 4,608 exact feature checks. The second response's selected-weight fraction cannot be exactly reconstructed from the retained rounded, merged weight totals; no exact claim is made for those 512 coordinates. The allocation formula recomputed with Python scalar summation differs by at most 4.34e−19 from the recorded CPU tensor-dot result; cached derivatives match exactly. RMS normalization and all unprojected phi updates reconstruct exactly from recorded credits and the preceding cached derivatives. This does not recompute the original billion-coordinate credit dot product, for which the production gradient/eligibility tensors were not loaded in this review.

Small allocation changes and nonzero secants are observations about execution. They do not establish why any serving result changed, whether the credit is a useful causal signal, or whether the numerical mechanism generalizes.

## Interpretation limits and fallacy coverage

All 11 ARS statistical fallacy types were checked for this bounded cost/controller analysis: Simpson's paradox (all four event overheads have the same direction); ecological fallacy (no per-request or future-run inference from aggregate costs); Berkson's paradox and collider bias (no selected cost subgroup or conditional regression used); base-rate neglect (not a diagnostic-classification analysis); regression to the mean (no extreme-selected pre/post improvement claim); survivorship bias (all four completed events and all 512 windows included, prior failed qualification kept outside this production cost total); look-elsewhere effect (all events and timer scopes disclosed); garden of forking paths (no window exclusions, endpoint change or tuning); correlation versus causation and reverse causality (no claim that recorded controller signs, features or magnitudes explain serving outcomes). **Coverage: 11/11.**

These are descriptive timings from one completed fixed campaign. There are no timing confidence intervals, independent training repetitions, measured deployment overlap/amortization, or training-inclusive serving-speedup estimates. Prior qualification and debugging expenses are separate retained historical costs, not production training overhead. Efficacy, correctness seals, evaluation accounting, and the final KW-versus-K serving judgment belong to the campaign's separate result audit.
