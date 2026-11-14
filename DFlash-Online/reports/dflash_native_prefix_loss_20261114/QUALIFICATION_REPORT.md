# Native prefix-loss investigation: qualification stopped

## Material Passport

ARS-Codex 3.24.2 experiment-agent, execution and evidence review. Scientific execution commit: `fdccdf5482f2125f212476547522068f953c564a`; PR47. Execution001 finished 2026-11-14 22:28:39 UTC, exit1. **Incomplete; no efficacy result.** Original failure records remain unchanged on A800 and in the verified private delivery.

## What this experiment asks

At sixteen fixed windows of the native KPLUS6→7 event, compare ordinary native learning N with E, which suppresses only current correct-prefix KL gradients. Prefix-only C and zero-current-gradient Adam H are explanatory controls; P is the static pre-update state. Only N would continue the trajectory. The primary endpoint is E−N accepted proposals on 64 reused development requests × four starts × sixteen windows, with whole-request uncertainty. This is a local one-step screen, not a serving-speed test or a new trained trajectory.

## Completed checks and failure

| Check | Result |
|---|---|
| Focused CPU tests on Fenrir and staged A800 source | 57 passed on each |
| Exact staged source | 4,214 entries matched published execution source |
| Retained-input, runtime, RAM and peak-storage admission | Passed |
| Native saved-path reconstruction and four neutral starts | Passed; all retained proposals/commits checked |
| N ordinary update | Completed |
| ALL wrapper versus N | Exact loss, gradient, weights, Adam, RNG and counters passed |
| E role gate | Passed: 291 permitted error occurrences carry gradient; excluded roles have zero row gradient |
| C role gate | Passed: 2,937 permitted prefix occurrences carry gradient; excluded roles have zero row gradient |
| H versus direct zero-gradient Adam reference | **Stopped at mapping type comparison** |
| Restored native repeat | Unattempted |
| Restore original weights, Adam, RNG, gradients and runtime | Passed at step768 / 1,536 presentations |

The H failure path was exactly `zero_reference.draft_state_dict`. Production `live_state()` obtains a **plain dict** from `DFlashTrainStrategy.checkpoint_state_filter()`. The independent H oracle instead copied `draft_model.state_dict()`, an **OrderedDict**. The shared `differences()` function checks exact container type before comparing keys or tensor values. Therefore the check could fail even if every weight were identical. The CPU fixture used an identity checkpoint filter and did not exercise the production mapping conversion.

This identifies a checker implementation defect. It is not evidence of a BF16 numerical discrepancy or a failed research mechanism. **It also does not prove H tensor equality:** the original check never reached those values, and the temporary H tensor pair was not retained after process exit. Native requalification remains necessary after the narrow checker repair. Adam-state comparison, H RNG/counter checks and exact origin restoration passed; these do not substitute for the missing weight check.

The H endpoint was distinct from static P: actual parameter displacement norm0.9326663058820043, with95,357,380 changed entries among1,048,626,432. This is an optimizer-history observation only, not evidence of independent drafting benefit.

## Narrow checker repair and review

Repair commit `5e9952315b749ad5edc3cd661d33e36cb6128a6e` builds the oracle weight mapping through the existing production checkpoint filter, just as the live endpoint does. The BF16 zero-gradient recurrence, native optimizer, algorithm and strict bitwise tensor comparison are unchanged. Failure receipts now distinguish mapping schema from tensor mismatch paths.

The committed source passes **59 focused CPU tests**, including the actual production filter, single/foreach Adam with historical moments and native None gradients, full six-case restoration, and a deliberately perturbed H tensor that still fails and saves its exact path after five applications. The earlier execution used57tests; the two new tests exercise the checker defect and failure-saving behavior. A same-team reviewer independently confirmed the source cause and reviewed the patch/report; its nine-test rerun preceded the final additional failure-saving fixture. No independent GPU replication or cross-model review is claimed.

The repair has **not** been rerun on GPU. Original GPU failure records and the actual execution commit remain unchanged. The five consumed steps are not reset by this fix.

## Work, duration and unfinished scope

The monotonic physical ledger is authoritative, including any partial case. Case summaries alone must not be used to erase failed work.

| Work | Completed | Remaining scientific work |
|---|---:|---:|
| Disposable qualification Adam applications | 5 of6 | Qualification incomplete |
| Qualification response forwards/backwards | 8 /8 | Final native repeat unattempted |
| Separate startup response forwards/backwards | 2 /2 | No startup Adam step |
| Main native/disposable Adam applications | 0 of176 | All176 |
| Additional fitted-response observation forwards | 0 of160 | All160 |
| Independent conceptual native outcomes | 0 of20,480 | All20,480 |
| New complete-response generations / teacher preparation | 0 /0 | Neither is part of this screen |

Separate native qualification work:119 draft blocks,117 target-verification calls,178 context forwards and2 model-load operations; the four neutral starts are included. No unknown or failed physical operations were recorded; the completed H operation was subsequently rejected by its checker. Training instrumentation recorded10 draft forwards,50 layer forwards,504 objective chunks and1,008 vocabulary-head calls including startup. Chunk/head counts are nested operations, not extra response presentations.

Occupied-card time was **203.070064s (3.38min)**; controller wall time278.156522s (4.64min), excluding earlier source transfer and sealing. Nested clocks must not be added. Maximum recorded H-case allocated/reserved memory was21,384,539,648 /23,412,604,928bytes (19.92 /21.80GiB). Both A800 cards were idle after exit; approximately348.970GB disk remained free. No historical artifacts were deleted.

All95 compact evidence files (3,585,286bytes) were retrieved and hash-verified. Private request text, tokens, anchors and raw runtime evidence remain outside Git. The public receipt is a compact extract, explicitly preserving the original failed status.

The earlier1–2GPU-hour estimate is still provisional. The observed maximum qualification pair cost7.452527s gives a conservative remaining-work estimate of roughly103min under the original probe/IO allowances; this is an extrapolation, **not completed resource qualification**. No trustworthy finish time can be given until the checker is repaired and qualification completes.

An unchanged six-case retry plus the full main screen would total **187 physical Adam applications**:5 already spent +6 retry +176 main. The failed203.070064GPU seconds must also remain charged to the original180-minute cumulative limit. No retry, changed allowance or scientific intervention has been launched.

## Decision and interpretation

There is **no result yet on whether withholding correct-prefix KL helps independent drafting**. No primary comparison, confidence interval, source-versus-transfer conclusion or new mechanism recommendation is justified. C/H are not substitute winners. The original six-case qualification and entire independent panel remain required; CPU repair alone cannot qualify the GPU experiment.

This remains a reused development panel, one native trajectory and local disposable interventions. Even a later positive E−N result would leave update scale, clipping and persistent Adam interactions unresolved, and would authorize only a separately designed next step.
