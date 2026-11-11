# Frozen SURVIVAL study-held-out GSM8K confirmation

**The pilot’s tentative benefit did not replicate.** SURVIVAL needed 174 more verification calls than KPLUS and 182 more than LOCAL on identical output tokens. Both prespecified call-reduction intervals exclude zero in the unfavorable direction. Complete-request latency points and both sweeps also favor the controls, but the timing intervals cross zero: a latency slowdown is not established. The prespecified outcome is **mixed or unfavorable**, with a small demonstrated call-count regression for these fixed checkpoints and this panel.

## Material Passport

ARS experiment-agent validate; ANALYZED. CPU reconstruction of native serving records; one continuing training seed17, no training in this confirmation.

Status: **complete**. Decision: **mixed_or_unfavorable**.

Execution commit: `e0129ee217f3f5b4bf6bb62bc44652e7d47b737d`.

Sealed N=768; 4608 timed completions. Planning half-width: ±0.5001 percentage points (1.225×sqrt(128/N), an assumption-based projection, not measured precision).

The panel is held out from SURVIVAL design, development and checkpoint-lineage feedback. It is not claimed project-unused. Exact exclusions, deduplication and prior exposure are retained in the private delivery’s SEALED_PANEL.json and EXPOSURE_INVENTORY.json; public identifiers and artifact hashes are in PANEL_AND_PROVENANCE.json. Older serving records supply only identities or authenticated canonical outputs; all three arms are timed anew.

| Contrast | Calls candidate/reference | Gross saved/added | Call reduction % [95% CI] | Complete-time reduction % [95% CI] |
|---|---:|---:|---:|---:|
| SURVIVAL_vs_KPLUS | 34540/34366 | 595/769 | -0.5063 [-0.8728, -0.1348] | -0.3388 [-1.4446, 0.7880] |
| SURVIVAL_vs_LOCAL | 34540/34358 | 564/746 | -0.5297 [-0.8867, -0.1821] | -0.8141 [-1.8802, 0.2418] |
| LOCAL_vs_KPLUS | 34358/34366 | 677/669 | 0.0233 [-0.3264, 0.3741] | 0.4715 [-0.6386, 1.6247] |

| Arm | Calls (one repeat) | Summed mean request seconds | Native seconds | Cache seconds | Cap-hit requests |
|---|---:|---:|---:|---:|---:|
| KPLUS | 34366 | 631.726160 | 588.083791 | 24.038165 | 2 |
| LOCAL | 34358 | 628.747594 | 585.759620 | 24.121126 | 2 |
| SURVIVAL | 34540 | 633.866153 | 589.464074 | 26.717951 | 2 |

| Contrast | Sweep 1 complete-time reduction % | Sweep 2 complete-time reduction % |
|---|---:|---:|
| SURVIVAL_vs_KPLUS | -0.189293 | -0.486230 |
| SURVIVAL_vs_LOCAL | -1.133798 | -0.501519 |
| LOCAL_vs_KPLUS | 0.933917 | 0.015213 |

Failed directional conditions: ['SURVIVAL_vs_KPLUS:call_reduction:nonpositive_point', 'SURVIVAL_vs_KPLUS:request_latency_s_reduction:nonpositive_point', 'SURVIVAL_vs_KPLUS:sweep_0:latency_not_improved', 'SURVIVAL_vs_KPLUS:sweep_1:latency_not_improved', 'SURVIVAL_vs_LOCAL:call_reduction:nonpositive_point', 'SURVIVAL_vs_LOCAL:request_latency_s_reduction:nonpositive_point', 'SURVIVAL_vs_LOCAL:sweep_0:latency_not_improved', 'SURVIVAL_vs_LOCAL:sweep_1:latency_not_improved']
Primary lower bounds not above zero: ['SURVIVAL_vs_KPLUS:call_reduction:lower_bound_not_positive', 'SURVIVAL_vs_KPLUS:request_latency_s_reduction:lower_bound_not_positive', 'SURVIVAL_vs_LOCAL:call_reduction:lower_bound_not_positive', 'SURVIVAL_vs_LOCAL:request_latency_s_reduction:lower_bound_not_positive']

| Shared first-round diagnostic | Wins/losses/ties | Prefix tokens gained/lost |
|---|---:|---:|
| SURVIVAL_vs_KPLUS | 88/74/606 | 219/214 |
| SURVIVAL_vs_LOCAL | 88/67/613 | 146/162 |
| LOCAL_vs_KPLUS | 98/97/573 | 261/240 |

| A800 phase/component | Recorded value |
|---|---:|
| references/wall_s | 139.90297977626324 |
| references/loading_s | 20.757918702438474 |
| references/generation_wall_s | 88.11112010478973 |
| references/valid_generations | 117 |
| references/reused | 651 |
| worker/wall_s | 4990.915245095268 |
| worker/loading_s | 21.00672951154411 |
| timing/evaluation_wall_s | 4947.476776659489 |
| timing/activation_s | 830.1025041155517 |
| timing/weight_validation_s | 6.267053307965398 |
| run/wall_s | 5180.357660789043 |
| run/optimizer_steps | 0 |

Timing averages the two repeats inside each request. Calls use one repeat after exact agreement. Bootstrap resamples whole requests jointly across arms, 10,000 PCG64 draws, seed20261110, linear 2.5/97.5 percentiles; intervals are unadjusted. Pilot and confirmation estimates are separate. Only the first native round shares a conditioning prefix; later rounds are not aligned mechanism evidence.

Reference preparation, startup and timed serving receipts are separate in RESOURCES.json and RESOURCE_TIMERS.csv. Native generation/cache components sit inside complete-request time; activation and checks sit inside evaluation wall time; loading/evaluation sit inside worker wall time. Missing timers remain unavailable. No training-inclusive speedup is claimed.

Confirmation requires positive calls and latency against both controls, all four lower interval bounds strictly above zero, and both latency sweeps favorable against both controls. Favorable points/sweeps with an interval including zero are directionally replicated but unresolved. Any failed point or sweep is mixed or unfavorable. A KPLUS-only advantage leaves the prefix-specific benefit unresolved. This is a decision for these checkpoints, not seed-robustness or unique mediation.

All 11 fallacy checks are retained. Stop after this panel; no additional training seed, tuning or follow-up experiment is automatically authorized.

## Executor audit and delivery

Execution source: `e0129ee217f3f5b4bf6bb62bc44652e7d47b737d`; original PR37 source: `7f89719956f0256c0916a583d7ff615954cc66f9`. All 4,608 timed executions were valid; invalid, unknown, extra and unattempted executions are zero. The included first-eight gate passed 48/48. All three frozen exports, 3,762 source files, the recorded symlink, target files and 14 native runtime file identities passed pre/postflight. There were zero optimizer steps, feedback collections or extra complete-response warmups. The 117 missing references completed separately and all passed; 651 authenticated references were reused.

The independent scalar implementation recomputed the comparison from `evaluation/MEASUREMENTS.jsonl`, without importing the production reducers. All 75 numerical summary checks, including all 12 bootstrap intervals, match exactly. All 4,608 private, unsanitized report CSV rows and their 156,672 fields match the raw rows in order. The public scalar table has 33 columns after replacing machine paths with a checkpoint hash; its 152,064 fields are separately checked against the private source and pinned exports. The 481 checkpoint publications match the presealed schedule. Native token/trace validity is supported by the separate passed native audit; the scalar audit does not claim an independent model rerun. This is a separate same-team audit with shared context and disclosed implementation authorship, not a blinded external review.

| SURVIVAL comparison | Requests with fewer / more / tied calls | Requests faster / slower / tied by mean time |
|---|---:|---:|
| versus KPLUS | 277 / 353 / 138 | 353 / 415 / 0 |
| versus LOCAL | 289 / 338 / 141 | 360 / 408 / 0 |

All arms produced 231,018 output tokens across the 768 requests; the same two cap-hit requests remain included. There were no immediate seed-EOS cases. The shared first-round comparison nets only +5 prefix tokens versus KPLUS and −16 versus LOCAL. These local observations do not explain or overturn the complete-request call regression.

### Pilot and confirmation remain separate

| Panel | N | SURVIVAL/KPLUS call / complete-time reduction | SURVIVAL/LOCAL call / complete-time reduction |
|---|---:|---:|---:|
| Original pilot | 128 | +0.3073% / +0.8602% | +0.5968% / +0.3868% |
| Study-held-out confirmation | 768 | −0.5063% / −0.3388% | −0.5297% / −0.8141% |

All four original pilot intervals crossed zero. The panels are not pooled. The projected ±0.5001-point latency half-width was an assumption-based planning estimate: achieved half-widths are 1.1163 points versus KPLUS and 1.0610 points versus LOCAL. The projected precision was not attained; N and the analysis remained fixed.

### Costs and preserved startup failure

Successful execution ran 07:06:34–08:33:06 UTC on 2026-11-11. Its controller wall was 5,180.358 s (86.34 min); the wrapper timestamps span 86 min 32 s including report creation. Timed evaluation’s enclosing wall was 4,947.477 s, including 830.103 s of checkpoint publication. Reference preparation’s 139.903 s includes 20.758 s loading and 88.111 s generation. These are nested clocks, not quantities to add together.

Before that run, execution001 at `cf03c95e8f7e268537d0e2eb81a622419ba1b4ca` failed during worker initialization because the launcher omitted the qualified compiler path. It performed zero response generations and zero optimizer steps. Its 46.688 s of startup/preflight wall and failure records remain separate. Restoring the exact pilot compiler, Conda, thread and offline environment was a launcher repair; no scientific setting changed. Four launcher regression cases were added, giving 46/46 passing CPU tests locally and on A800. See [STARTUP_REPAIR.md](STARTUP_REPAIR.md).

The two recorded execution walls total **5,227.045 s (87.12 min)**; this excludes staging and later analysis/publication gaps and is not active GPU-kernel time. Both A800s were idle after completion. No result or model artifact was deleted by this experiment. Global disk availability increased from about 45 GB at admission to 714 GB at the final observation; its external cause is unknown and no timing-causality claim is made from it.

### Reproducible artifacts

- `COMPARISONS.csv`: all three contrasts and intervals.
- `PAIRED_REQUESTS.csv`: 768 paired requests, preserving every gain, regression and tie.
- `TIMING_REPEATS.csv`: all 4,608 fresh scalar rows; only machine paths are omitted and checkpoints are bound by SHA256. Private raw rows remain unchanged.
- `RESOURCE_TIMERS.csv`, `REPEAT_TOTALS.json`, `AUDIT_SUMMARY.json`, `PINNED_CONFIGURATION.json`, `PANEL_AND_PROVENANCE.json`: accounting, validity, source/runtime/export identities, ordered source IDs, exposure counts and private-delivery hashes. The portable public configuration omits three local environment/cache paths; the exact original pins remain in the verified private seal.
- Full sealed inputs, canonical tokens, raw measurements, native audit, receipts and independent arithmetic are retained in the Fenrir private delivery. Per-request native trace bodies remain at `/data/sources/minhao/runs/dflash_survival_confirmation_20261111/execution_002` on A800. All 50 compact delivery file identities were verified after retrieval.

Selection admitted 1,063 eligible rows and sealed the first 768 by the handoff’s input-only hash order. The entire 256-question development source panel is excluded; 640 lineage/event-five feedback identities and all documented qualification/probe inputs are excluded or shown disjoint. Independent CPU review reproduced all 768 rendered/tokenized prompts and 128 pilot prompts. Prior unrelated-project exposure is documented for 765 selected requests; 3 are unknown, not asserted unused.

**The authorized confirmation is complete and stopped.** This result does not support a prefix-specific endpoint advantage for the retained SURVIVAL checkpoint. It neither establishes a universal failure of survival-head training nor identifies the cause of the regression. No tuning, new seed, additional panel or training has been started. Stage 1 / project None is unchanged.
