# Prior repeat-versus-new-feedback audit

**Subsequent historical correction, October 24:** the section below describing
D108/D141's ordinary early-stopping control as still missing is superseded. D142
completed it and D146 reviewed the closure: candidate 11,310 calls versus ordinary
early 11,419 and stronger random 11,350 at approximately 20.87 minutes. The 0.352%
increment did not pass the 2% screen. See
[D142 results](../../dflash_efficiency_decision_20261006/final_review_001/RESULT_REVIEW.md)
and [expanded gap review](../../osd_next_experiment_review_20261024/GAP_REVIEW.md).
The earlier repetition/allocation comparisons and numbers below remain valid;
this correction prevents reopening the already-completed cost control.

## Material Passport

- ARS experiment-agent, validate/design review; October 24, 2026.
- Verification status: ANALYZED from retained protocols, source reports and machine-readable results. No GPU rerun, remote access or experiment execution.
- Scope: determine whether the proposed four-event current-runner comparison repeats an established project question, and distinguish internal controls from external generalization.

## Conclusion

**The substantive question has already been tested.** D97 directly compared more distinct responses with repeated use at matched optimizer-step counts under pure forward KL. D108 then compared repeated versus previously unused feedback from a common complete model/optimizer/RNG state, with ordinary buffered and order controls. These were not merely weak, tiny-feedback failure studies.

The proposed exact four-event, 128-response, live-causal DFlash-Online version has not been found among completed records. It is an implementation/operating-protocol replication or extension, not a new scientific bottleneck. Changing the runner, seed, feedback availability and small exposure horizon does not erase the previous evidence. The earlier proposal should explicitly acknowledge these antecedents before any execution decision.

## 1. D97: direct matched-exposure forward-KL precedent

Sources: [locked protocol](../../dflash_paid_learning_20261004/PROTOCOL.md), [reviewed results](../../dflash_paid_learning_20261004/RESULT_REVIEW.md), [machine-readable measurements](../../dflash_paid_learning_20261004/analysis_001/SUMMARY.json).

Released Qwen3-8B/DFlash-b16 revisions are exactly the current pair. All arms use pure forward KL, temperature 1, hard-label weight 0, AdamW LR 3e-4, weight decay .01, betas .9/.999, epsilon 1e-8, gradient clipping 1, response batch 1/accumulation 2, anchor cap 512, position decay 7 and native block 16. Same source pool and same 256 evaluation requests; all 11 arms produce identical 77,143 output tokens. Native loss/gradient/update bridges and release/optimizer restoration were qualified.

| Matched work | Repeated smaller pool | Broader pool | Verification calls, repeated / broader | Mean accepted draft tokens/check, repeated / broader |
|---|---|---|---:|---:|
| 256 optimizer steps | 128 responses × 4 passes | 512 responses × 1 pass | 12,048 / 12,086 | 5.6678 / 5.6718 |
| 1,024 optimizer steps | 512 responses × 4 passes | 2,048 responses × 1 pass | 11,681 / 11,689 | 5.9307 / 5.9243 |

There was no demonstrated useful broad-coverage advantage at these matched budgets. The acceptance differences were 0.0040 (95% paired interval -0.0527 to 0.0605) and -0.0064 (-0.0759 to 0.0628), not formal equivalence results. Counts slightly favor repetition and show why mean acceptance must not replace total verification work. All ratios are conditional on one trained seed and one development panel.

The same raw table also corrects an incomplete historical small-budget summary: useful pure-FKL results existed below 512 responses. From released weights, 128 responses × 1 pass gives 12,374 calls versus frozen 12,607 (1.85% fewer); 128 × 2 gives 12,116 (3.89% fewer). These are one-seed, fixed-panel source-compatible results, not a guaranteed minimum feedback budget or a live first-buffer prediction.

Differences from the new proposal: seed 0, precollected 2,048 frozen-serving responses, nested selected pools rather than newly arriving blocks, first global pass chronological then shuffled, 2,048 training limit, text/prepared author sequences, and no per-buffer future-panel audit. One response loses 69 supervised output tokens under the declared common limit. The current design uses seed 17, exact token retention with limit 2,272, persistent four-event state and S-generated feedback. These differences limit transportability; they do not make D97 internally unfair. Matched response presentations/steps were explicitly not exact token or compute matching, just as in the current proposal.

## 2. D108: common-state repeat/advance schedules already completed

Sources: [locked protocol](../../dflash_schedule_value_20261005/PROTOCOL.md), [reviewed results](../../dflash_schedule_value_20261005/final_review_001/RESULT_REVIEW.md), [original proposal](../../online_learning_proposal_20261003/WORK_ORDER_REVISION_D107.md).

This is the closest structural antecedent: warm on old pool O (512 responses × 4 passes), then fork exact model, AdamW, counters and RNG. Each of two continuation phases has 1,024 presentations / 512 steps. RR repeats O in both phases; NN uses unused U then V; RN/NR/mixed and ordinary native continuation supply controls. Full logical schedules have 4,096 presentations / 2,048 steps. Matched-order controls use the same multiset and per-occurrence anchors. All math endpoints have identical output tokens.

| Endpoint | Verification calls on same 256 requests | Preparation minutes |
|---|---:|---:|
| Frozen | 12,607 | 0 |
| Warm-up, stop | 11,529 | 13.812 |
| First unused phase, stop | 11,310 | 20.834 |
| Repeat / repeat | 11,531 | 27.123 |
| Unused / further unused | 11,181 | 27.760 |
| Ordinary native from release | 11,219 | 27.821 |

Unused feedback beats continued repetition by 350 calls (3.04%). Repetition adds essentially nothing beyond the warmed checkpoint. But the best schedule beats competent native updating by only 38 calls (0.339%); the multiplicity-adjusted paired interval was -0.934% to +1.658%, and no schedule passed the predeclared 2% increment screen. Unused feedback ranked ahead at all three observed forks: no observed ranking reversal justified a changing selector. Thus, finding new feedback better than repeated old feedback does not itself identify a missing algorithmic capability: ordinary OSD already advances to new data.

D108 uses the author AdaFlash draft objective, 0.8 teacher-argmax CE + 0.2 clipped reverse KL, not pure OSD forward KL. Optimizer/anchor/model settings otherwise track the competent recipe (3e-4, decay .01, 512 anchors, block 16). Its objective was shared across schedule arms; this is a deliberate objective-conditioned comparison, not an unfair loss mismatch. The current proposal would test transportability to pure forward KL and the newer online architecture, not overturn an invalid result.

### The genuinely missing control in D108

The shorter warm-up + unused phase costs 25.11% less than full ordinary training and needs 0.811% more calls. **Ordinary native training stopped at the same shorter preparation cost was not evaluated.** [D141](../../online_learning_proposal_20261003/NEXT_DECISION_D141.md) explicitly identifies this missing comparator. The current proposal does not supply it: four-event S versus repeated B1 is a different comparison and does not establish an efficiency advantage over ordinary early stopping. No additional efficiency experiment is authorized by this audit; it only records the unresolved original control.

## 3. Related studies that answer different questions

| Study | What it actually compared and found | Relationship/fairness limit |
|---|---|---|
| [D156 one-pass/two-pass](../../dflash_async_passes_20261006/recovery_001/completed_review_001/RESULT_REVIEW.md) | Qwen/GSM8K asynchronous 7,473-request serving; 128-response buffers. One pass consumed 58 buffers/7,424 distinct responses/3,712 steps; two passes 40/5,120/5,120 steps. Final identical-output 256-request checks: 9,720 vs 9,703 calls. | Competent comparison of complete schedules, not matched-step repeat/new control. Concurrent publication changes availability, waiting and data consumed. It does not isolate repetition, and it already cautions against a backlog-only algorithmic story. |
| [D199 connected8/split4](../../multiturn_matched_d199/analysis_001/RESULT_INTERPRETATION.md) | Llama/DFlash-UltraChat, same eight responses, two passes, order/anchors/104 steps and publication timing; only connected8 vs two four-response training contexts differ. 15,216 vs 15,527 calls (2.00% benefit); 131 equal-output requests. | Strong within-study context control, not repeated-data allocation. It changes both teacher and draft conditioning; historical frozen timing is not a contemporaneous repeated benchmark. |
| [D208 window/backlog](../../cosql_window_d206/reviewed_delivery_002/RESULT_REPORT.md) | Released/fresh-optimizer W8/40/128/500, two passes, W optimizer steps, 48 learned checkpoints. Calls versus frozen +4.57%, +9.21%, -1.83%, -12.67%. | Feedback amount, age/mix and steps change together by design. Cannot isolate diversity or cumulative online learning; not internally unfair simply because budgets differ for a budget-scaling question. |
| [A1/A2 retained-checkpoint replay](../../dflash_author_replay_20261015/campaign_001/completion_review_001/REPORT.md) | Llama, full-history, eight-response author OSD updates: A1 learns9–16; A2 continues on17–24; F/A1/A2 on identical25–32 saved contexts, three repeats. A2/A1 saves3.07% time and5.01% calls pooled; three episodes regress. | Already directly measures marginal effect of new feedback on common future requests. “Replay” here means inference checkpoint evaluation; it does not mean training again on A1 feedback. No repeated-old-buffer branch. |

## Implication for the proposed run

1. Do not describe the proposed study as the first test of fresh feedback versus more training, or use it to rediscover that ordinary updating works.
2. Do not dismiss D97/D108 as unfair: they used competent matched within-study settings and record the intended changing factors. Separate limited external generalization from a defective comparator.
3. The planned 2,176-generation campaign repeats substantial established work. It should not launch unchanged merely because the exact current-runner configuration is new.
4. A precise remaining current-code question could justify a deliberately bounded replication, but that decision must state what earlier result is uncertain and what mechanism decision could change. Fresh feedback outperforming R alone does not motivate selection beyond ordinary S.
5. If the goal is preparation efficiency, preserve the unresolved ordinary-early-stopping comparator from D108 rather than claiming the new S/R matrix resolves it.

No settings, manifests, source code or active run were changed. This document is a local retained-evidence audit, not authorization to execute a revised experiment.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_transfer_budget_design_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
