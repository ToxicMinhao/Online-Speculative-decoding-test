# Independent execution and retained-data audit

## Material Passport

ARS academic-research-suite 3.24.0, experiment-agent validation. **Audit passed, with zero issues.** Verification status is **ANALYZED**: this separate same-team reviewer reconstructed retained evidence without rerunning training or generation. The review is not blinded external validation. Stage 1 / project None remains unchanged.

The first raw-audit invocation succeeded without parser or evidence repair. It reconstructed **54,435 native verification blocks across all 1,152 unique completed generations**, verified all **3,845 retrieved files** (867,097,921 bytes), and rehashed **1,053 locally available counterparts** of the **1,659 remotely sealed source/input files**. The remaining 606 identities rely on passed remote preflight/postflight checks. Full state/model tensors remain on A800; this audit does not claim to have downloaded or locally rehashed them.

The full `FINAL_AUDIT.json`, `AUDIT_EXECUTION_001.log`, `AUDIT_SOURCE_CHECK.json` and `raw/campaign_001/` native, correction, boundary and retention receipts are retained private evidence. Their filenames and A800 paths below identify the audit basis; they are not public download links. This public audit contains compact operational metadata and aggregate results, without prompts or token arrays.

## Source and execution identities

Execution commit: [`11b7718a04416a89025a6df69c0c33e558e5fe1b`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/11b7718a04416a89025a6df69c0c33e558e5fe1b). All 18 addition identities match that commit's exact blobs and the execution seal. Analysis commit: [`ea4f289c6e18383b7560c8c4103a9d91ca3943bf`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/ea4f289c6e18383b7560c8c4103a9d91ca3943bf). The auditor, its fixtures and report builder exactly match that commit.

The analysis auditor differs from the execution archive only in its previously documented O1 count-source pointer: it reads the original OSD one-event evidence instead of the later first-error comparison's retained O1 measurements. Both complete 64-request maps are identical, totaling 3,075 calls. This change was recorded before the raw audit and changes no scientific result. No auditor code was modified during this audit. An initial source-check sidecar used an overbroad unchanged-since-execution label; it is preserved as `AUDIT_SOURCE_CHECK_INITIAL.json`, while the final sidecar states the exact difference.

Supervisor execution was 2026-10-28 18:28:24–19:46:15 UTC, **4,671 s**, exit 0; the campaign driver recorded 4,428.999 s. All 48 remote CPU tests passed. Actual preflight admitted 49.231 GB free against 33.648 GB required, including the 8 GiB reserve; available host memory was 136.252 GB against 72 GiB required. All 1,659 sealed checks passed before and after execution.

## Training lineage and correction work

Exact KRC2 complete-state/export and live restoration passed at global step 256 / presentation 512. All 512 newly sampled ordinary presentations and anchors matched their retained O/K contracts naturally. Event-3 and event-4 feedback, prepared features, canonical responses, evaluation exclusion and capped `train_00399` were preserved. The horizon remained 7,424. Optional probes, new feedback and extra qualification Adam steps were zero.

| Event | End step / cumulative presentations | Core forwards / backwards | Applied, BF16-nonzero corrections | Zero-violation no-ops | Event wall (s) |
|---|---:|---:|---:|---:|---:|
| 3 | 384 / 768 | 768 / 480 | 127 | 1 | 693.529 |
| 4 | 512 / 1024 | 768 / 476 | 125 | 3 | 681.568 |

The new total is **256 Adam updates, 512 ordinary presentations, 1,536 core forwards and 956 backwards**. Separate startup was two disposable forward/backwards and zero Adam updates. Each event has 256 reference presentation occurrences over 128 distinct retained responses, evaluated in two reference phases; the 512 phase-forwards are not 512 independent examples. Reconstructed reference available/protected occurrences are 975,476/516,445 for event 3 and 963,417/504,525 for event 4. The independent parser verifies triangular weights, pooled denominator, native prefix/margin decisions, correction branches and per-window work sums. These are pre-correction statistics; disabled post-correction probes provide no measured final local-prefix benefit.

Reference capture/replay preserves ordinary RNG, mode and cache; correction preserves Adam state/counters. Complete KRC3 and KRC4 states and exports passed snapshot/export and live-boundary checks, then identical full-file hashes after event 4. Their immutable full-state paths are:

- `/data/sources/minhao/runs/dflash_krc_events_3_4_20261028/campaign_001/KRC/events/event_003/training_state.pt`
- `/data/sources/minhao/runs/dflash_krc_events_3_4_20261028/campaign_001/KRC/events/event_004/training_state.pt`

Their serving exports are the sibling `export/` directories. Full-state SHA-256 values are `66033b346717c66df8a78c8206beb8f5278b507153ad5adb3e3d0f234c453808` and `1934f9b7b162a77ef4b2333a10cbb8e92a471a9347e78b49317ee1c90becba95`. Training closed before fresh evaluation.

Intrinsic reference/correction work was 296.045 s and 288.806 s, nested within the respective training totals of 653.578 s and 642.801 s. Persistence and export, state checks, loading, startup and final retention validation are separately recorded; nested timers must not be added twice. Peak allocated/reserved memory was 38.231/40.624 GB and 39.855/42.488 GB. Historical O/K costs remain historical controls.

The inherited runtime-compatibility metadata still mentions “all three fresh trainer constructions” and “K, G_disabled and G_enabled.” These descriptive labels are stale. Actual work records show the two-response disposable startup and one continuous KRC training trajectory through events 3 and 4. The labels do not represent additional G trials; original metadata is preserved.

## Fresh serving results and interpretation

All 1,152 planned generations returned valid canonical outputs and termination, with no invalid, unknown or unattempted work. Repeats agree on deterministic counts. The first 72 development generations provide the included gate. Development reproduces every retained O4/K4 per-request count; additional O4/K4 values are fresh measurements with no historical count gate. Publication order, three weight-equality checks per panel, cache policy, capture-disabled traces and unique worker-capture identities pass.

| Panel | O4 / K4 / KRC4 calls | KRC4 versus O4 | K4 versus O4 | KRC4 versus K4 |
|---|---:|---:|---:|---:|
| Development | 2994 / 2942 / 2950 | +1.4696% | +1.7368% | −0.2719% |
| Additional | 3130 / 3058 / 3071 | +1.8850% | +2.3003% | −0.4251% |

Positive percentages mean pooled call reduction. KRC4 saves 44/59 calls against O4 and adds 8/13 against K4 on development/additional. Against O4, request wins/losses/ties are 35/21/8 and 36/20/8; against K4 they are 24/28/12 and 22/26/16. All three timing repeats on each panel agree: KRC4 is faster than O4 and slower than K4. Summed mean request-time reductions are +1.5959%/+1.8135% versus O4 and −0.3369%/−0.5623% versus K4. Native generation and cache maintenance remain nested inside complete request time; publication/loading are excluded.

On development, KRC's benefit versus O persists but shrinks from +3.0075% at event 2 to +1.4696%; its advantage versus K reverses from +0.2018% to −0.2719%. On additional, the K advantage reverses from +0.6378% to −0.4251%. Additional O1/O2 counts are absent, so persistence versus O cannot be assessed there. KRC's own improvement does not establish relative improvement: KRC2→KRC4 saves 17/45 calls, while K2→K4 saves 31/78.

All 11 ARS statistical-fallacy categories were reviewed within this descriptive scope. Panels remain separate, request-level regressions remain included, all declared executions are accounted for, and historical gaps stay absent. No significance or independent-seed claim is made. This is one trajectory with shared O-generated feedback and three timing repeats. KRC4/K4 compares complete correction histories, not only corrections in events 3/4; neither this audit nor these measurements establish training-inclusive deployment acceleration. No tuning or continuation is authorized by the result.
