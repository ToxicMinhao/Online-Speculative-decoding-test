# PR36: original report location and result

## Material Passport

- Origin: ARS-Codex 3.24.2, experiment-agent / validate.
- Verification status: **ANALYZED** — retained evidence review, not model reproduction.
- Original retrieval scope: CPU/filesystem work only; no CUDA imports, model loads, server access, training, generation, cleanup, source changes or external publication.
- Date: 2026-11-17, America/Edmonton. This newly written location/result note is not the original experimental report.
- Decision: native K+ remains primary. Neither SAM nor Lookahead is selected or authorized for execution. Research stage unchanged.

## Publication note

Published on the user’s subsequent explicit request. This is a public edition of the location note: private host paths are omitted and links point to the included evidence. [ORIGINAL_REPORT.md](ORIGINAL_REPORT.md) is byte-identical to the retained final report. [SUMMARY.json](SUMMARY.json) contains unchanged endpoint/comparison fields from the original summary, and [EVIDENCE_IDENTITIES.json](EVIDENCE_IDENTITIES.json) preserves source identities, completion and qualification excerpts. Original per-request CSVs are not included. No model execution or algorithm change accompanies publication.

## Located original evidence

The completed report was already retained on Fenrir: [original REPORT.md](ORIGINAL_REPORT.md); [original SUMMARY.json endpoint excerpt](SUMMARY.json). The latter includes absolute method totals, both timing-repeat totals, six paired comparisons and native validity receipts. Search stopped after locating this original report; no A800 retrieval or reconstruction was needed.

Execution/source revision: **`c64dfe3bf50d8bce84f2cca3576826c8d12a0087`**, recorded clean. Original report version: `kplus_abt_report_v1`; its source digest matches the retained `report.py` (38,356 bytes, SHA256 `b61a5602b51a15c32b330fba20298614d386c320900cd5a3ad06785099b986d0`). The original receipts do not identify a historical report-publication commit. This documentation commit publishes the recovered report; it is separate from the experimental execution revision above.

Execution binding: `5afd12aa4873b4c1c36e1b9cff2b787866ae474887da6f6f6e440f468f6b0669`. Original A800 execution root: `@execution_002`. Start **2026-11-10 19:00:53 UTC**, finish **2026-11-11 02:58:13 UTC**, exit **0**, elapsed **7 h 57 min 20 s**. The original controller reports 26,323.782 s for its campaign phase; that is nested within the enclosing elapsed time.

Supporting originals: [retained evidence metadata](EVIDENCE_IDENTITIES.json), [retained evidence metadata](EVIDENCE_IDENTITIES.json), [retained evidence metadata](EVIDENCE_IDENTITIES.json), [retained evidence metadata](EVIDENCE_IDENTITIES.json).

## Completion, protocol and validity

All **16/16 arm-events** completed: KPLUS, CLEAN2, RANDOM and ADV, on GSM8K and MathQA, events 5 and 6. Each dataset restores its own complete PR32 KPLUS4 state at Adam step 512 and each arm ends at 768. The recorded postflight checks the original Adam/RNG state and export agreement. No checkpoint was loaded or rehashed during this retrieval.

This is **shared evolving-native feedback**: KPLUS4 produces event-5 responses; this campaign’s KPLUS5 produces event-6 responses. All arms share the packets, ordinary response order and anchors. It is not four independent online feedback trajectories. Rho remains 0.01; CLEAN2/RANDOM/ADV average two views, followed by one native clipping/Adam operation per ordinary two-response window.

At event 6, each dataset uses **256 requests × four arms × two repeats = 2,048 executions**. Both datasets pass their retained native output-token/termination audit; **4,096 valid**, zero invalid, unknown, pending-canonical or unattempted executions. This note verifies the retained audit receipts, not every raw token trajectory. All arms have 80,493 GSM8K output tokens (one capped request) and 124,961 MathQA output tokens (nine capped requests).

## Original endpoint measurements

Call totals use one validated deterministic trajectory per request. Complete-request times sum the per-request mean of the two repeats. Native acceptance is `native_accepted_draft_tokens / verification_calls`; the separate stop-clipped matching counter is slightly larger and must not be substituted. Committed progress excludes the initial seed token.

| Dataset | Arm | Calls | Native accepted/call | Progress/call | Complete-request time (s) |
|---|---|---:|---:|---:|---:|
| gsm8k | KPLUS | 12,018 | 5.676402 | 6.676402 | 206.608839 |
| gsm8k | CLEAN2 | 11,994 | 5.689762 | 6.689762 | 206.414827 |
| gsm8k | RANDOM | 11,975 | 5.700376 | 6.700376 | 205.680488 |
| gsm8k | ADV | 11,969 | 5.703735 | 6.703735 | 205.556988 |
| mathqa | KPLUS | 16,782 | 6.430878 | 7.430878 | 295.389076 |
| mathqa | CLEAN2 | 16,739 | 6.449967 | 7.449967 | 294.591265 |
| mathqa | RANDOM | 16,704 | 6.465577 | 7.465577 | 293.804169 |
| mathqa | ADV | 16,753 | 6.443741 | 7.443741 | 294.597781 |

Positive reduction favors ADV. All six comparisons use **256 paired requests** each. These are original saved intervals, not newly fitted intervals.

| Dataset / ADV versus | Call reduction % [95% CI] | Complete-time reduction % [95% CI] | Wins/losses/ties | Gross calls saved/added |
|---|---:|---:|---:|---:|
| gsm8k / KPLUS | +0.4077 [-0.0992, +0.9025] | +0.5091 [-0.0292, +1.0508] | 109/92/55 | 209/160 |
| gsm8k / CLEAN2 | +0.2084 [-0.2854, +0.6797] | +0.4156 [-0.1139, +0.9321] | 113/89/54 | 193/168 |
| gsm8k / RANDOM | +0.0501 [-0.3978, +0.4992] | +0.0600 [-0.4111, +0.5288] | 91/90/75 | 163/157 |
| mathqa / KPLUS | +0.1728 [-0.1571, +0.4958] | +0.2679 [-0.3592, +0.8919] | 101/89/66 | 184/155 |
| mathqa / CLEAN2 | -0.0836 [-0.4370, +0.2737] | -0.0022 [-0.6443, +0.6401] | 93/97/66 | 170/184 |
| mathqa / RANDOM | -0.2933 [-0.6457, +0.0458] | -0.2701 [-0.8911, +0.3371] | 90/106/60 | 147/196 |

Bootstrap: 10,000 PCG64 paired-request draws, seed 20261110, linear 2.5/97.5 percentiles and shared draws across the three contrasts. Intervals are unadjusted, conditional on one continuing training seed and reused development panels. Repeats are not additional requests or seeds. **Native-acceptance intervals are not present** in the retained summary; call-reduction intervals are not acceptance intervals.

## Cost and prior failure

| Dataset | KPLUS intrinsic training (s) | CLEAN2 (s) | RANDOM (s) | ADV (s) | ADV extra versus KPLUS |
|---|---:|---:|---:|---:|---:|
| gsm8k | 637.704 | 1258.303 | 1262.459 | 1257.843 | +620.139 s / +97.246% |
| mathqa | 1029.466 | 2036.728 | 2043.474 | 2038.886 | +1009.420 s / +98.053% |

These two-event intrinsic times come from the original report and exclude only its measured diagnostic observer interval. The extra-cost column is this retrieval’s arithmetic on those rounded original values. ADV uses roughly **1.97–1.98×** KPLUS intrinsic training time. Observer time is separately 12.497 s GSM8K /16.122 s MathQA for ADV. Loading, startup, save/export and feedback preparation are separate costs; no adaptation-inclusive speedup follows.

Endpoint activation was 2,826.877 s for GSM8K and 3455.024 s for MathQA, within evaluation phases of 4,972.087 s and 6,454.666 s respectively. These are not serving latency and must not be added to their enclosing timers. Detailed persistence/initialization ledger rows are referenced in the original report but were not copied into this local status package.

Completed main work implies **2,048 Adam updates** (16 completed arm-events ×128), with 512 unique shared feedback packets: four reused from the earlier qualification and 508 newly generated. This is a count derived from completed tasks and the sealed schedule, not a fresh per-step audit. The successful qualification records 12 disposable Adam steps, 68 response forwards/backwards and eight serving checks.

The separately retained first attempt (`8862bc3161c281d11eb21d619bd272486558a80f`) failed its nested finiteness checker after one disposable Adam step, four forwards/backwards including startup, zero main updates and zero serving checks; its endpoint was not saved. Its recorded qualification wall time was 161.684 s. This failure remains an implementation-checker failure, not negative ABT efficacy. No new failure or retry occurred in this retrieval.

## Evidence availability and identities

The nine small original files below still exist locally and match the hashes/byte counts captured by the successful **2026-11-11 05:07:39 UTC** retrieval. No multi-GB model was copied or hashed. The report-source hash also matches its original analysis identity.

| Relative to the original PR36 evidence directory | Bytes | SHA256 |
|---|---:|---|
| START_UTC | 21 | `deb534c88e363f4e308c186e2be81418641bdcdf98550f6b77bddb0c4e95d9f8` |
| END_UTC | 21 | `4a724dbb5889063f76cf7439942a91924badde9e666cc838ef3f6e860497dff5` |
| EXIT_CODE | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| campaign_001/STATUS.json | 13,864 | `89092b5ce86b0c948644447628276ba62ff0ea20a9953603db2fee2509702f20` |
| qualification/ADMISSION.json | 9,662 | `d7d7da2990cb9ff81f48e8cc6323b1fb9d2ac127d178a6b3b7c14b62ad66a23a` |
| qualification/STATUS.json | 8,224 | `6e03b4b8e86526c113f16b40b7d0af80031f5e848aa982c1eccc85eb2cb082b7` |
| final_report/REPORT.md | 8,068 | `0102bb84468c4b4d70b09724a5c100bfcf4ac5fe54a723308fba2bf29ecabcc4` |
| final_report/SUMMARY.json | 201,951 | `b3f5c1e52561652d0fb1161b90a2b994a7583fee430b2d69e5dc52ff34939dd7` |
| POSTFLIGHT_PILOT.json | 3,706 | `b9b7f1e66dc1ffae1b4a5ed6d509ec50c50fb57353238bdc55ad1e2e2a6ab587` |

Historical retrieval receipt: [retained evidence metadata](EVIDENCE_IDENTITIES.json). It also inventories the original tables at `@execution_002/final_report`: `COMPARISONS.csv` (1,886 bytes), `PAIRED_REQUESTS.csv` (542,711), `TIMING_REPEATS.csv` (1,813,266), `TRAINING_COSTS.csv` (18,499), `OPTIMIZER_WINDOWS.csv` (267,457), `AUGMENTATION.csv` (1,490,444), `DIAGNOSTICS.csv` (14,813), `RUN_STATUS.csv` (2,087,782), plus qualification/input/fallacy manifests. These are **historical remote locations and sizes**, not fresh existence checks. They are absent from this particular local PR36 final-report folder. Because the original report was immediately available, the handoff’s stop condition was honored; no expanded remote search was made.

The compact original report/summary, completion, qualification and postflight evidence demonstrably survive. Current survival of all remote tables, raw token files, final-arm recovery tensors and exports after cleanup is **not established by this retrieval**. No deleted artifact is described as archived. Initial-state identities and original ancestry checks are excerpted in EVIDENCE_IDENTITIES.json; no later checkpoint was guessed or substituted.

## Interpretation and review scope

**PR36 supplies no demonstrated adversarial-direction benefit.** ADV slightly improves point estimates over native KPLUS on both datasets, but every requested call/time interval includes zero. GSM8K ADV is nearly tied with RANDOM; MathQA ADV is worse than CLEAN2/RANDOM on the point estimates. Those unresolved contrasts do not establish equivalence. The approximate doubling of intrinsic training cost is established by the recorded times; worthwhile serving benefit is not.

The attack changed its intended training quantities, but this is not a substitute for serving improvement. These results concern mask-embedding perturbations with shared feedback. They do not establish that parameter-space SAM fails, that sharpness caused prior failures, or that Lookahead would help. Neither candidate is promoted.

This retrieval recalculates only ratios from saved totals, gross/net consistency, repeat-total means and the displayed training-overhead percentage. It does **not** recompute bootstrap intervals or regenerate a scientific report. A separate same-team reader independently checked the original report and summary and found no substantive inconsistency. Shared source evidence is not an independent experimental replication.

ARS interpretation coverage **11/11**: datasets separated (Simpson); no individual mechanism inferred from aggregate gains (ecological); reused selected panels disclosed (selection/Berkson); no outcome-conditioned adjustment (collider); denominators explicit (base rates); no selection of extreme improvements (regression to mean); all arms and failed qualification retained (survivorship); all six contrasts shown (look-elsewhere); no analysis/tolerance tuning (forking paths); no robustness-to-serving causal attribution (causation); chronological shared-feedback contract retained without reverse-causal inference. Multiple-comparison and single-seed limitations remain.

## This retrieval’s work and stopping point

Initial estimate: 10–20 minutes, hard 30-minute active-search/reduction limit. The original report was found within the first few minutes; remaining work was bounded reading, small-file identity checks and this note. New GPU time, optimizer steps, generations, environment changes, deletions, external publications and remote transfers: **zero**. Local starting free space: 9,588,801,536 bytes; new output is this small report plus concise workflow/experience entries, well below the 250 MB cap. No cleanup was performed.

One administrative receipt reader initially encountered a list-valued inventory entry while iterating file records; restricting identity checks to actual file-record dictionaries resolved it. No source artifact or experimental measurement changed.

**Stop here.** No new mechanism, third paper family, model reproduction, tuning or repetition is authorized by this handoff.
