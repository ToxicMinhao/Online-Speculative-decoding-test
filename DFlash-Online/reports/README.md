# Experiment report guide

Read each report with its own comparison, dataset panel and accounting. Ratios are pooled from summed measurements; repeats measure timing and are not additional requests. Historical and fresh timings are not pooled into a learning curve.

The table below groups the recent DFlash-Online work by question rather than by whichever outcome was favorable. Each report should link to its exact source snapshot, sealed settings, implementation review and retained validity evidence.

| Question | Report/run identifier | Interpretation boundary |
|---|---|---|
| What does ordinary OSD do under the selected setup? | [dflash_online_osd_20261023](dflash_online_osd_20261023/), [dflash_online_osd_rerun_20261023](dflash_online_osd_rerun_20261023/), [dflash_osd_one_event_20261025](dflash_osd_one_event_20261025/) | These use different event budgets/settings; do not treat them as interchangeable baselines. |
| Do reference-prefix ranking or reference-margin penalties help? | [dflash_prefix_ranking_one_event_20261025](dflash_prefix_ranking_one_event_20261025/), [dflash_reference_margin_execution_20261026](dflash_reference_margin_execution_20261026/) | Preserve the earlier numerical qualification failures and the startup analysis alongside final results. |
| Does selecting through the current first error help? | [dflash_first_error_execution_20261026](dflash_first_error_execution_20261026/) | Four-arm O/C/K/A comparison separates position selection from KL-versus-CE choice. |
| Does fixed-checkpoint K extend beyond development? | [dflash_first_error_fresh_confirmation_20261027](dflash_first_error_fresh_confirmation_20261027/) | Requests are new to the current study; earlier unrelated-project exposure is disclosed. |
| How does K change with additional shared feedback? | [dflash_ok_continuation_20261027](dflash_ok_continuation_20261027/) | Exact own-state continuation; count learning curves differ from fresh paired timing comparisons. |
| Does protecting old matching prefixes help locally or natively? | [dflash_prefix_preservation_guard_20261027](dflash_prefix_preservation_guard_20261027/), [dflash_prefix_correction_step_20261027](dflash_prefix_correction_step_20261027/), [dflash_kc_event_20261027](dflash_kc_event_20261027/) | Local prefix repair is not held-out drafting improvement. |
| Does protection of other buffered responses help? | [dflash_krc_event_20261027](dflash_krc_event_20261027/), [dflash_krc_confirmation_20261028](dflash_krc_confirmation_20261028/) | Reference responses add training exposure; fixed-checkpoint confirmation does not establish seed robustness. |
| Is ordinary replay a better use of the extra work? | [dflash_reference_replay_20261028](dflash_reference_replay_20261028/) | KR changes steps, optimizer evolution and exposure; not an equal-compute or isolated-direction comparison. |
| Does KRC's advantage persist through event 2? | [dflash_krc_continuation_20261028](dflash_krc_continuation_20261028/) | Relative advantage and improvement over its own previous checkpoint are separate questions. |
| Does event 2 correction add benefit from exactly the same initial state? | [dflash_krc_off_20261028](dflash_krc_off_20261028/) | Small, mixed differences do not establish equivalence or a general event 1-only policy. |

Earlier projected-target, infrastructure and checkpoint-component studies belong in the full report inventory as well. Implementation-only or failed qualification records must be labeled as such; code completion does not imply a successful GPU experiment.

## Latest comparison

The correction-removal control restored KRC1 exactly, then trained ordinary K without reference processing for the next event. All 768 paired executions passed the specified validity checks. Positive reductions below favor keeping correction:

| Panel | KRC2_OFF calls | KRC2 calls | Correction's call reduction |
|---|---:|---:|---:|
| Development | 2,960 | 2,967 | −0.2365% |
| Additional | 3,138 | 3,116 | +0.7011% |

This is mixed evidence for event 2 correction's endpoint value. Both panels remain in the result. The source is [KRC2_OFF report](dflash_krc_off_20261028/REPORT.md); the table is not a cross-study average or a new analysis.

## What every report's provenance note should say

- **Published source commit:** immutable Git revision containing the relevant code or its archived exact version.
- **Run/source mapping:** where each historically executed file is preserved and how it matches retained source identities.
- **Configuration/runtime/model pins:** the actual run settings and dependencies, including separate native-serving and training environments.
- **Evidence scope:** which public compact tables are included, which private raw records remain external, and whether anything is unavailable.
- **Qualification versus main work:** temporary startup/checks and main training/generation budgets are not conflated.

The publication commit is created after these experiments. It should not be described as their original execution-time commit. Adding the provenance note must not rewrite immutable raw seals or erase original failure records.

## Complete report inventory

[All 177 report copies](INDEX.md) record source commit [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a) and their per-run historical source manifest. This is retrospective archival publication, not a backdated execution commit. Original body text and original evidence hashes are retained; links to private raw inputs/outputs may point to artifacts deliberately absent from Git. Publication provenance is the only addition to the reports.

There are nine unresolved historical source pins: seven initial foundation versions and two pre-reseal test revisions. These remain explicit; current A800 source and the latest KRC2_OFF implementation are fully captured. Survey-only upstream OSD source is linked rather than redistributed when its license could not be verified.
