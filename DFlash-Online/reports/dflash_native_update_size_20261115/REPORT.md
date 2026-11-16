# Native K+ update-size attribution — completed

Scientific execution: `5d4e0ff65c5976318a50ebb181923447068169c8`, PR48. The report/publication commit is separate from the code that ran. ARS-Codex 3.24.2 experiment-agent; Stage 1 / project None unchanged.

**The experiment completed correctly, but its prespecified advancement criterion was not met.** Withholding the current correct-prefix contribution (E) demonstrated an advantage over the upper size-matched ordinary-update control, but not over the lower control. It therefore does not establish a prefix-specific advantage beyond both nearby ordinary update sizes. Neither scalar control established an improvement over native K+ (N), so this result also does not establish that smaller updates explain the earlier E−N gain. No equivalence or mechanism-failure conclusion follows.

## Question and controlled comparison

PR47 found that E gained +0.015625 accepted proposals per fixed block over N, while its BF16 displacement was smaller. This experiment holds the native pre-update states, feedback, optimizer and saved independent evaluation prefixes fixed, and supplies the missing local size controls.

- P: native state immediately before the selected update.
- N: ordinary K+ update from P, with its established BF16 AdamW path.
- E: PR47's update withholding the current correct-prefix contribution. Its authenticated outcomes/norms are reused; no missing E model was regenerated.
- S_low / S_high: native AdamW learning-rate alternatives whose realized BF16 norms bracket E within 0.25% on each side. Only these new outcomes were scored.

Only the unchanged native KPLUS6→7 trajectory advances. Each control is a disposable one-step alternative; controls are not accumulated into another trained model. All 16 windows/32 controls were sealed before scoring. Historical E results were already known. This is not blinded independent confirmation and alpha is a per-window experimental control, not a deployable learning-rate schedule.

The fixed panel comprises 64 reused development requests, four exact native starts each, across 16 windows. Qwen3-8B/DFlash identities, BF16, greedy block 16, FA3, full saved history/cache reconstruction, context 4,096 and original 2,048-token/EOS/censoring rules are unchanged. No complete response, teacher feature or source-fitting observation was newly generated.

## Primary result

Units below are **accepted draft proposals per fixed block**, not percentages, verification-call reductions or serving speedups.

| Contrast | Mean difference | Conditional paired 95% interval | Status |
|---|---:|---:|---|
| E − S_low | +0.008789 | [−0.002930, +0.019287] | Unresolved |
| E − S_high | +0.016602 | [+0.005371, +0.028809] | Positive |
| E − N | +0.015625 | [+0.002441, +0.028809] | Historical PR47, reproduced |
| S_low − N | +0.006836 | [−0.008301, +0.022949] | Descriptive, unresolved |
| S_high − N | −0.000977 | [−0.018799, +0.016357] | Descriptive, unresolved |
| S_high − S_low | −0.007812 | [−0.020996, +0.004395] | Descriptive, unresolved |

Advancement required both primary means and both lower bounds strictly above zero. The fixed decision is **UNRESOLVED_STOP_THIS_SCREEN**. We retain native K+ and stop; no additional run, coefficient sweep or training extension is initiated.

Every estimate averages the four starts within request/window, then sixteen windows within request, then 64 requests equally. Intervals use 10,000 common paired whole-request bootstrap draws, PCG64 seed 1707 and linear 2.5/97.5 percentiles. All states/windows of a sampled request stay together. This captures conditional request variation, not independent training seeds or domains. The joint primary is mandatory; descriptive contrasts and pass/window views do not replace it.

## Acquisition, loss and actual repairs

| State relative to P | Acquired prefix occurrences | Lost prefix occurrences | Net | Verified old-first-blocker repairs | Earlier breaks |
|---|---:|---:|---:|---:|---:|
| N | 177 | 136 | +41 | 52 | 43 |
| E | 194 | 89 | +105 | 57 | 40 |
| S_low | 154 | 85 | +69 | 45 | 39 |
| S_high | 143 | 106 | +37 | 44 | 38 |

Across 4,096 paired block/windows, E−S_low gains 36 net occurrences: 40 more acquired and 4 more lost. E−S_high gains 68: 51 more acquired and 17 fewer lost. E loses 8 of N's 52 useful repairs; S_low loses 12 and S_high 19. These are repeated block/window occurrences, not independent token examples. There are 59,952 valid proposal occurrences, 3,577 old-first-blocker repair opportunities and 3,511 old-prefix break opportunities. The repair and advance fields remain distinct; a longer matching prefix is not mislabeled as a demonstrated repair without the token check.

Complete request/window/decomposition tables accompany the report; all regressions remain. No historical-versus-fresh timing comparison is used as a learning effect.

## Correctness and execution

- All 68 focused CPU tests passed before execution. CPU Gate0 reproduced all six surviving P/N/E files at 8/128 exactly, including original file hashes/sizes; both E norms reproduced exactly. No CUDA/Adam operation occurred in CPU preparation.
- All six disposable GPU cases passed, including native/cached alpha 1 equivalence, repeatability, alpha 0 weight neutrality, half-rate repeat and exact origin restoration.
- All 32 P/N endpoint checks passed: four direct comparisons at retained boundaries and28 exact reconstructed-original-file comparisons. Both retained full-state boundary witnesses passed. These new reconstruction checks are not historical pre-cleanup tensor fingerprints.
- All 16 numerical brackets qualified. Maximum selected-control norm gap was0.244533%, below the fixed 0.25% side limit. 149 interior trials were needed; no window exceeded ten. No exact endpoint aliases occurred.
- Final native weights, Adam state, counters, RNG and scientific runtime fields match retained NATIVE7 exactly at 896 steps / 1,792 presentations. Calibration restores the authoritative native state after each trial.
- All 64 saved parent reconstructions, 24 historical-reuse sentinels andeight neutral blocks passed. All 8,192 new control outcomes are valid, with zero invalid, unknown, missing, duplicate or unattempted rows. Together with 12,288 reused P/N/E rows, the analysis grid has 20,480 records.
- Independent CPU analysis rescored all 20,480 native records against saved canonical tokens and reproduced all six estimates/intervals and every decomposition:524,512 checks passed. All11 ARS interpretation categories were assessed. This separate same-model-family audit is not an independent experimental replication.

An independent execution audit passed **621,322 assertions**, reconciling raw work ledgers, identity/restoration receipts, control sealing, validity and resource admission. These are receipt-level assertions, not additional optimizer steps or model evaluations.

## Actual work and cost

| Work | Actual | Prespecified limit |
|---|---:|---:|
| Native Adam applications | 128 | 128 |
| Calibration Adam applications | 149 | 160 |
| Disposable GPU qualification applications | 6 | 6 |
| Total Adam applications | 283 | 294 |
| Response forwards / backwards | 260 / 260 | 260 / 260 |
| New control endpoint blocks | 8,192 | 8,192 |
| Extra neutral/sentinel blocks | 32 | 32 |
| New complete-response generations / source-fitting forwards | 0 / 0 | 0 / 0 |

The 260 response forwards/backwards comprise 256 native, 2 qualification and 2 startup. Calibration uses cached preclip gradients and adds no forward/backward. Actual projection accounting: 16,556 objective chunks, 33,112 vocabulary-head calls and 1,300 layer forwards.

The instrumented native ledger additionally records 11,050 draft blocks, 11,111 target calls, 182,295 context forwards and 37 model loads. These totals include reconstruction; the new endpoint count does not hide that work. No failed or unknown work appears in this GPU execution.

Execution ran 22:16:17 UTC November 15 to 00:16:30 UTC November 16. Campaign wall time was 7,211.20 s (120.19 min); occupied-card accounting 7,147.99 s (119.13 min), below 180 min. The qualification/native/calibration phase took 5,819.08 s; frozen scoring 1,181.11 s. These are nested clocks, not additive extra costs. Native trajectory plus calibration accounted for 5,522.87 s. Training-process peak allocated/reserved memory was 20.487/26.552 GB; it is not a measured whole-campaign GPU peak.

CPU preparation took 499 s, including 121.65 s Gate0. The earlier 76 s CPU administrative failure is preserved separately with zero GPU/Adam work. Transport/staging records are also retained. Compact retrieval/verification took 281.39 s, including a transient transfer stall; no scientific retry occurred. Both GPUs were idle afterward; only physical GPU0 was used throughout.

## Storage and retained evidence

The authenticated private delivery contains 25,384 files / 288,073,661 bytes, all hash-verified, with no bulk checkpoints copied to Fenrir. Raw tokens, teacher features, private reconstruction traces and optimizer tensors remain outside Git. Public tables expose only identities, counts, geometry and costs.

Postflight before temporary-payload retirement: project 968.047 GB plus 31.698 GB environments = 999.745 GB allocated; filesystem free 629.314 GB. A separate baseline-relative scan measured 76.360 GB of new allocation across all sealed roots, including prior attempts/caches, against the110GBcap and 400GBfree reserve. This is a measured postflight value, not a continuous peak measurement; admission receipts and sampled high-water accounting are reported separately.

Storage closeout completed on November 16 after report publication and exact remote identity checks. Removed only the 32 temporary scalar exports plus two scientifically redundant new native payloads: **34 files, 75.509 GB allocated**, with filesystem free space **629.334 → 704.843 GB**. Original NATIVE7 export/recovery hashes match, and all original keepers, source, raw outcomes and compact evidence remain. The native redundancy is exact scientific-state parity, not assumed whole-file duplication. Direct replay of the retired scalar exports is no longer available; regeneration would require separately authorized model work. Files are retired, not archived. The initial maintenance audit stopped before deletion because host PID1 descriptors were outside the project container; the corrected check explicitly inspected the 17 in-container processes and excluded 1,574 foreign host processes. Its scoped check, failed log and full 34-path retirement journal are preserved. No GPU/model work occurred during cleanup.

## Limits and conclusion

Equal global norm does not match layer-wise changes, update direction or output-distribution changes. S_low/S_high retain N's updated Adam moments, whereas E's may differ; this is a control of the complete local intervention, not isolation of one harmful gradient component. Two bracketing sizes do not exclude an unobserved nonlinear optimum or every learning-rate alternative.

E still inherits earlier prefix learning and optimizer history. Sixteen one-step alternatives cannot be summed into a valid E training trajectory. E tensor geometry is available only at 8/128; it remains unavailable elsewhere. This reused development panel, one seed and one native trajectory do not establish held-out serving benefit, training-seed robustness or speedup.

**This bounded comparison ends unresolved.** It narrows the claim we can make: the earlier local E−N improvement survives as historical evidence, but a contribution beyond ordinary update size is not established by the prespecified joint test. Smaller updates are also not demonstrated to be the explanation. No mechanism revision or follow-up experiment is launched.
