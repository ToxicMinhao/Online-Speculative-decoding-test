# Recent-repair retention: completed event-seven pilot

## Material Passport

- Status: **completed; analyzed with separate raw and statistical reductions**.
- Training execution: `562f63fdff6955d5ffcf6086bf9b2f8f948dba8b`.
- Evaluation execution: `345b1321856c40caab7ee3af7a9905dece3d0229`.
- Counter repair: `8fc2f68afc28b6f6891814f180eb257dca7153cf`.
- Audited result publication: `0bc814027c3537c3cab19dde52d6d576f833e2bb`; this provenance annotation follows that publication. See `PUBLICATION_RECEIPT.json`.
- Same-team review includes implementation authorship overlap; this is not an independent GPU replication.


**Decision: this version does not qualify as an upgrade to native K+.** The RECENT arm did not demonstrate better drafting, speed, or held-out repair retention. Its accepted-length confidence interval against NATIVE ends at +0.1044%, below the prespecified +0.50% useful-gain target on this fork. The matched CONTROL also provides no demonstrated history-placement advantage. Keep native K+ as the working method; no further event or tuning was launched.

All three training trajectories, all 64 fixed-start requests and all 1,536 timed executions completed. The earlier evaluation-startup accounting failure was repaired with explicit user authorization; evaluation alone resumed from the unchanged endpoints. Independent raw and statistical reductions passed.

## Primary results

Counts use one validated trajectory per request. Accepted length is **clipped accepted draft proposals / verification calls**, not output progress. Complete-request time is the sum of the 256 per-request means over two sweeps.

| Arm | Accepted proposals | Verification calls | Accepted proposals/call | Complete-request time (s) |
|---|---:|---:|---:|---:|
| NATIVE | 62,402 | 11,026 | 5.659532 | 189.156668 |
| RECENT | 62,367 | 11,055 | 5.641520 | 189.568193 |
| CONTROL | 62,375 | 11,049 | 5.645307 | 189.563942 |

| RECENT versus | Accepted-length gain % [95% CI] | Call reduction % [95% CI] | Time reduction % [95% CI] | Speedup |
|---|---:|---:|---:|---:|
| NATIVE | -0.3183 [-0.7478, +0.1044] | -0.2630 [-0.6294, +0.0975] | -0.2176 [-0.6016, +0.1615] | 0.997829× |
| CONTROL | -0.0671 [-0.5067, +0.3726] | -0.0543 [-0.4293, +0.3189] | -0.0022 [-0.3839, +0.3764] | 0.999978× |

| RECENT versus | Requests improved / regressed / tied | Gross calls saved / added | Net calls saved |
|---|---:|---:|---:|
| NATIVE | 80 / 94 / 82 | 107 / 136 | -29 |
| CONTROL | 74 / 74 / 108 | 110 / 116 | -6 |

Positive reductions favor RECENT. The small negative point estimates do not by themselves establish degradation: the primary serving intervals include zero. They nevertheless do **not** support the prespecified useful improvement; the +0.50% target lies above the native accepted-length interval. Unresolved differences are not equivalence.

## What happened to held-out repairs?

At 2,775 shared K6 block starts from 64 requests, 431 slots were recent K5→K6 repairs. Earlier breaks remain in that 431-slot denominator. There are 2,410 genuine K6 first blockers. These are fixed-prefix diagnostic predictions, not extra independent serving requests.

| Arm | Exact-slot relapse / 431 | Earlier break / 431 | Accepts through repair / 431 | Direct agreement at repair / 431 | Passes old first blocker / 2,410 | Acquired / lost / net prefix occurrences |
|---|---:|---:|---:|---:|---:|---:|
| NATIVE | 121 | 32 | 278 | 291 | 442 | 1327 / 1270 / +57 |
| RECENT | 123 | 33 | 275 | 292 | 435 | 1356 / 1361 / -5 |
| CONTROL | 118 | 32 | 281 | 294 | 439 | 1383 / 1274 / +109 |

RECENT shows no favorable retention direction: it has 123 exact-slot relapses versus 121 for NATIVE and 118 for CONTROL, and accepts through fewer repair slots than either. Across all starts it acquires 29 more prefix occurrences than NATIVE but loses 91 more, leaving 62 fewer net occurrences. These counts describe the observed local changes; they do not prove why the serving endpoint differed. Direct agreement at a later position does not count as useful prefix progress when an earlier error blocks it.

## Prespecified decision and limits

| Requirement | Observed disposition |
|---|---|
| RECENT accepted-length point at least +0.50% vs NATIVE | Fails: −0.3183% |
| Native contrast lower bound above zero | Fails: −0.7478% |
| Placement support / balance | Passes: 114 slots, 70 requests, 16.4569% switched mass, absolute SMD 0.012456 |
| CONTROL contrast lower bound above zero | Fails: −0.5067% |
| Fewer exact relapses and more accepts-through than both controls | Fails on both directions |
| No demonstrated slowdown vs NATIVE | Passes: time-reduction upper bound +0.1615%; no demonstrated time benefit either |

The auxiliary term was active: RECENT fired in 107/128 optimizer windows, and its first firing changed 297,041 BF16 parameter entries relative to the same-state penalty-disabled update. Its final weights differ from NATIVE in 296,587,104 entries. Thus this is not an unexercised mechanism. Frozen assigned exposure is small (0.335772% of selected weight before the 0.10 coefficient), but that fraction is not a measure of gradient influence and does not establish an insufficient-dose explanation.

Observed main-training time is 339.372 s for RECENT versus 333.391 s for NATIVE: +5.981 s (+1.794%). Frozen preparation, qualification, first-firing capture/replays, loading and publication are separate below. The timings compare diverging instrumented trajectories, not an isolated kernel-cost experiment. Reused teacher/feedback/history costs were not freshly measured, so training-inclusive deployment speedup is not established.

This is one seed and one shared-feedback continuation. All 256 evaluation requests were held out of this fork’s training, but have disclosed older-project endpoint exposure; they are not globally unseen. Paired request-bootstrap intervals are conditional on these trained checkpoints and fixed placements. All 22 marginal intervals are reported without claiming simultaneous family-wise coverage. The matched placement contrast tests the 16.4569% switched weight; shared residual recent tags remain in both active arms.

**Stop at this milestone.** The prognostic observation that recent repairs can be fragile remains separate from the failure of this particular preservation intervention to deliver the specified benefit. Reopening would require a separately agreed, discriminating design addressing transfer or the treatment’s effective influence; this run does not authorize a coefficient search, another seed, or more events.

## Live weighted exposure

Fractions below pool raw position weights, not gradients. Assignment equality is a pretraining control; live reach and firing can diverge.

| Arm | Assigned weight | Reachable / assigned | Firing / assigned | Firing / reachable |
|---|---:|---:|---:|---:|
| RECENT | 967.535860 | 96.3697% | 22.3649% | 23.2074% |
| CONTROL | 967.535858 | 96.4602% | 20.5667% | 21.3214% |

Per-pass quantities use the same explicit denominators; raw numerators are retained in `weighted_exposure.csv`.

| Arm / pass | Reachable / assigned | Firing / assigned | Firing / reachable |
|---|---:|---:|---:|
| CONTROL / 0 | 95.6174% | 28.0876% | 29.3749% |
| CONTROL / 1 | 97.3027% | 13.0485% | 13.4103% |
| RECENT / 0 | 95.3726% | 31.0956% | 32.6043% |
| RECENT / 1 | 97.3664% | 13.6375% | 14.0064% |


## Method and qualification

All arms resume the same complete KPLUS6 state at Adam step 768, consume the same 128 feedback responses and teacher packets, and finish at step 896 after 256 presentations / 128 steps. Native K+ sampling, loss redistribution, optimizer, BF16 arithmetic and seed-17 RNG continuation are unchanged. RECENT adds a reachable-slot hinge at coefficient **0.10**, with frozen floor **0.50 × min(K6 margin, 1)**, to prospective K5→K6 repairs. CONTROL moves matched tags to established successes and retains unmatched tags. The reference margins and assignments are fixed before training; only the newest response supplies supervision.

The Qwen3-8B / five-layer DFlash serving path uses greedy BF16/FA3, block 16, concurrency 1, full prepared inputs, context 4,096 and cap 2,048/EOS. Timing uses the same physical A800, two counterbalanced sweeps, capture disabled and valid cache/weight switching. Both cards were quiet; training and transfers occurred outside timed requests.

All four qualification cases passed. Disabled and zero-coefficient paths match native gradients, weights, optimizer state, counters, selections and RNG exactly. Active and longest-response fixtures pass with restoration. Both first-natural-firing ACTIVE/OFF checks reproduce the production active endpoint and restore the final state exactly. Recovery performed **zero additional training, feedback collection or canonical preparation**. All 1,536 serving executions match canonical tokens/termination and repeat-consistent counts; zero invalid, unknown or unattempted executions remain.

Scientific training is 384 Adam steps / 768 presentations. Three disposable qualification steps and four first-firing replay steps are separate, never serving checkpoints. Frozen preparation uses 256 forwards / 8,170 head chunks, no backward or Adam work. The fixed-start phase completes 64 parent walks, 2,775 neutral replays, 2,775 K5 replays and 8,325 endpoint blocks. Canonical preparation reused 254 references and generated two separately before evaluation recovery.

## Cost, failure and recovery accounting

| Arm | Main training s | Loading s | Disposable startup s | Persistence s | Export s | Diagnostic state capture s | First-firing replay s | Worker wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NATIVE | 333.391 | 32.273 | 19.199 | 18.603 | 14.505 | 0 | 0 | 442.578 |
| RECENT | 339.372 | 31.229 | 19.576 | 19.301 | 14.505 | 55.720 | 107.531 | 627.868 |
| CONTROL | 338.977 | 32.965 | 18.431 | 20.818 | 15.707 | 54.989 | 104.772 | 612.502 |

The main-training clock includes the bounded first-firing preclip-gradient CPU copy and instrumentation. Other snapshot capture and diagnostic replays are separate. These nested clocks are not additive; worker wall includes loading and lifecycle work. `costs.csv` retains all measured components, and `sweep_totals.csv` separates complete-request, native-generation and cache-maintenance times for each sweep. Actual serving weight-loading/activation costs remain outside request latency.

Frozen preparation takes 103.909 s. Qualification worker walls are 93.180 / 95.714 / 97.177 / 68.616 s. Canonical preparation takes 49.965 s, including 42.012 s loading and 1.678 s generation. Successful fixed-start work occupies 2,233.125 GPU-phase seconds; the timed-evaluation driver takes 1,479.969 s including switching and lifecycle overhead.

The original attempt completed training but failed before the first parent generation because a load counter budgeted engine construction without subsequent checkpoint publications. Its one failed parent-walk wrapper and **51 startup draft blocks / 48 startup target calls / 24.235622 GPU-phase seconds** remain preserved. No scientific parent generation or timed completion was repeated. With explicit user authorization, the counter was fixed and the unexecuted evaluation ran in a separate directory. This is an infrastructure repair, not a loss, sample or tolerance change.

The original and recovery measured execution walls sum to **6,258.991 s (104.317 min)**; this excludes repair/user-wait gaps. Calendar time from the first launch (03:24:48 UTC) to final completion (05:31:22 UTC, November 13) is 2 h 6 min 34 s. Both are within the original duration allowance. Allocated phase windows total **1.6660 card-hours**, including CPU loading/cleanup; this is not kernel-busy time. The instrumented training/qualification peak is 20.515 GB allocated / 25.812 GB reserved, not a measured serving-wide peak. Final A800 free disk is 504.193 GB; both GPUs are idle. No scientific artifact was deleted.

Feedback collection, teacher preparation and historical label trajectories were reused, not freshly timed here. Full online deployment cost and training-inclusive speedup remain unestablished. See [cost scope](ADAPTATION_COST_SCOPE.md) and the preserved [original incomplete report](INCOMPLETE_REPORT.md).

## Evidence and reproducibility

- Independent raw-block and placement audit: **814,215 checks**, zero errors or missing files. Counts are reconstructed from saved proposals, verifier predictions and terminal clipping.
- Independent scalar/bootstrap reduction: **27,486 checks**, including all **22 prespecified intervals**, zero failures; 10,000 valid and zero undefined draws for each interval. No model work was used by either audit.
- CPU validation: 92 retention tests on A800 before recovery; 30 focused local recovery/runtime/report tests; separate helper fixtures and seven report tests pass. These suites overlap and are not added into a fictitious unique total.
- All **48,066 compact files / 489,773,862 bytes** retrieved and matched their recorded hashes. Private raw traces, token IDs and full audit paths remain outside Git; model/recovery tensors stay on A800, not archived on Fenrir.
- Candidate accounting: 1,319 inputs; 256 excluded (192 PR43, 64 PR34/40), 1,063 eligible, 256 selected and 807 eligible unselected. See `PANEL_SELECTION.json` / `panel_exclusions.csv`. The panel and assignments preceded training.
- Full raw results and exact reproduction commands are linked through the experiment README and private execution receipts. Training and evaluation source identities are distinct; report-only wording and publication helpers do not change the scientific implementation.

Tables: [per request](per_request.csv), [every repeat](all_repeats.csv), [sweep totals](sweep_totals.csv), [fixed-start origins](fixed_start_origins.csv), [fixed-start outcomes](fixed_start_outcomes.csv), [all local intervals](LOCAL_UNCERTAINTY.md), [activation](activation.csv), [weighted exposure](weighted_exposure.csv), [matching](matching.csv), [gradient receipts](gradient_receipts.csv), [costs](costs.csv), [raw audit summary](RAW_AUDIT_SUMMARY.json), [statistics audit](STATISTICS_AUDIT.json), [interpretation audit](FINAL_INTERPRETATION_AUDIT.md), and [11-category review](ARS_FALLACY_CHECKLIST.md).
