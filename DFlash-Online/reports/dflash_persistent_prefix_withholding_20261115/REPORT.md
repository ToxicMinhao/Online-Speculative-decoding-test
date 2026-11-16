# Persistent prefix withholding: CPU readiness result

## Material Passport

ARS-Codex 3.24.2, experiment-agent, source and data validation. Status: **INCOMPLETE_DATA_GATE — stopped before model work**. The source baseline is `a21f9d47e8e05779ada99b713f764c27411ba899`. The publication archives the CPU audit scripts and their source-bound receipts; it is not a GPU execution commit. No new model forward, Adam step, feedback response or evaluation generation has run. Stage 1 / project None remain unchanged.

## What the handoff would test

This is a persistent two-event feasibility test, distinct from PR48's disposable one-step comparisons. It does not retrospectively change PR48's unresolved conclusion.

- **N:** native K+, reusing the authenticated complete N7 state and training only N8.
- **E:** start from complete K6 and carry prefix-gradient withholding through E7 and E8, preserving its own weights, Adam history, RNG and changing current error roles.
- **S:** start from complete K6, change only the declared current LR contract to the exact inherited LR multiplied by 0.96, and continue through S7 and S8.

E keeps native forward loss values, selection, denominator, redistribution and attention. The existing hook detaches direct prefix-token loss rows before both native numerators; first-error gradients still propagate through context. It is not error-only renormalization or detachment of prefix hidden states. The disposable PR47 trainer cannot be used unchanged because its default pair role and restoration behavior return to N. A persistent arm/event adapter is therefore required.

Event 7 uses the existing feedback and schedule. Each arm would collect its own event-8 responses on the same next 128 input requests and train from its own complete state. Final evaluation would compare only N8/E8/S8 on 128 fresh same-domain requests, with three repeats and the prescribed balanced four-block order. Untimed work may run on two independent single-GPU workers; final timing uses one fixed physical GPU with the other idle and no heavy host/transfer work.

The practical E/N gate requires at least 0.5% time reduction, positive lower 95% bounds for time and pooled native acceptance, and positive aggregate time reduction in every repeat. E/S and S/N remain mandatory comparisons. No new result for these comparisons exists.

## Blocking data result

**The required 128-request globally unused evaluation panel cannot be formed from the pinned GSM8K test split.** The original native manifest binds `openai/gsm8k`, `main`, revision `740312add88f781978c0658806c59bc2815b9866`, with 1,319 test questions.

| Proven completed prior use | Distinct test questions | Newly covered questions |
|---|---:|---:|
| Retained AdaFlash AR serving reference | 1,024 | 1,024 |
| Retained buffered GSM8K objective evaluation | 276 | 276 |
| October 22 frozen projected-target evaluation | 8 | 8 |
| Retained async final-one-pass evaluation | 256 | 3 |
| **Proven union** | **1,311** | **1,311** |

That leaves **at most eight**, before any further exclusions: a shortfall of **at least 120**. Eight is an upper bound, not a certification that all eight are fresh. Additional historical exposures, duplicates or context exclusions cannot increase availability.

The audit checks actual completed rows and completion receipts, rather than merely planned manifests. It authenticates both Parquet files, maps opaque IDs bijectively to original questions, and checks all 1,024 inherited native evaluation prompt-token arrays against the completed historical serving inputs. The exclusion table publishes source IDs and hashes without private questions or token arrays.

A separate worker, without reading the first audit, independently checked the first two completed runs. Its 6,354 assertions establish 1,300 distinct prior-use questions and an upper bound of 19, already sufficient to fail the 128-request gate. The broader audit tightens the bound to eight. These are separate same-model-family CPU audits, not independent GPU replications.

The handoff explicitly states: “If128 genuinely eligible unused same-domain inputs are unavailable, stop during CPU preparation; do not silently reuse the old64/256/1024 panel or change domain.” No previous study's scoped freshness exception was transferred to this campaign. No replacement panel was selected.

## Checks that did pass, and work not attempted

The next 128 training inputs, native stream positions 896–1,023, are disjoint from the first 896 by request ID, normalized problem text and saved prompt-token hash. Event-7 feature-manifest and two-pass schedule identities match their retained input receipts. This is input-only validation, not a new live checkpoint/optimizer/RNG qualification.

The existing role hook implements the requested E arithmetic. Source review identified the required persistent adapter, own-state resume contract, historical-versus-current LR validation and precise S migration. The mandated Python product is `0.00028799999999999995` (`0x1.2dfd694ccab3ep-12`), distinct from independently writing the literal `2.88e-4`.

The persistent training adapter, two-card qualification, runtime/canonical generation integration, production resource admission and full campaign remain **unimplemented or unqualified**. No CPU training integration tests are claimed. Work stopped at the prior data-availability gate rather than implementing and launching an experiment that could not satisfy its evaluation contract. Source review notes are preserved for a later explicitly revised handoff.

## Resources and duration

The live read-only A800 check found both physical A800 80 GB cards available, each with 81,153 MiB free and 0% utilization; host available memory was approximately 148.45 GB, and disk free space was 629.334 GB. These observations are not a full peak-memory or storage admission. Fenrir had approximately 34.18 GB free before the small preparation checkout.

The fixed budget would be five new training events, 640 production plus eight qualification Adam steps, at most 1,310 response forwards/backwards, and at most 1,744 complete generations. The four-hour GPU limit is **summed across cards**, with a separate four-hour elapsed cap; it is not four hours per GPU.

| Historical planning component | Approximate GPU-card minutes |
|---|---:|
| Five events at PR48's actual native training rate | 40.48 |
| 1,152 scored generations plus 72 warmups at PR32's GSM8K rate | 17.54 |
| 384 feedback generations at PR32's rate | 5.07 |
| 384 terminal feature preparations at PR32's rate | 0.25 |
| **Partial total** | **63.3** |

This is **not a full completion estimate**. It excludes target-only autoregressive references, qualification, startup/graph capture, activations, publication and resident waits; it also does not establish parallel contention costs or new request lengths. Pricing training at PR48's slower instrumented qualification-pair rate instead gives a partial total of about 123.7 card-minutes. The 5,522.87-second PR48 trajectory clock includes calibration and must not be mislabeled ordinary training time.

Before production, the contract requires `spent + 1.5 × estimated_remaining + 1800 <= 14400` card-seconds, using actual qualification and metadata-based autoregressive costs. With zero time spent, the remaining unbuffered estimate must fit 140 card-minutes. No defensible launch/finish estimate is available until an eligible panel is authorized and the remaining integration/resource checks pass. No GPU campaign time has been consumed by this preparation.

PR48 temporary-checkpoint retirement completed separately: 34 exact files removed, 75.509 GB allocated freed, and 704.843 GB free afterward. Original K6/N7, other paper keepers and compact raw results remain. Deleted scalar checkpoints are retired, not archived; their compact outcomes and regeneration inputs remain. The first read-only cleanup check stopped at an inaccessible host process outside the project container, before deletion; the corrected scoped check and final journal are preserved. This maintenance added no GPU model work.

## Decision and retained evidence

**Stop before GPU qualification and production.** This is a data-availability failure, not a negative result for persistent prefix withholding. The existing K+, PR47 and PR48 results remain unchanged. Proceeding requires an explicit revision to the evaluation-data requirement; no dataset switch, freshness relaxation, smaller panel or extra experiment was applied.

- `data_audit/`: reproducible source audit, 1,311-row exclusion lower bound, event-8 input identities and receipts.
- `data_crosscheck/`: independent source/ID/token checks and the sufficient 1,300-question bound.
- `implementation_review/`: narrow implementation requirements and source-backed cost calculations.
- `resources/`: live read-only device/storage observation and maintenance status.
- `HANDOFF.txt`, `STATUS.json`: adopted specification and explicit zero model-work accounting.

Private source questions, answers, token arrays, teacher features and checkpoint tensors remain outside public Git. The new draft PR is a readiness report and CPU evidence package, not a claim that the requested training algorithm has been implemented or tested on GPU.
