# Review of the alleged missing early-stopping control

## Material Passport

October 24, 2026. ARS experiment-agent, plan/retained-evidence review. Local source reports only. No remote operation, model execution, code change, or stage change. This is an independent internal review of one candidate gap, not a new authorization.

## Finding

**D141's ordinary early-stopping comparator is not missing. It was completed in D142 and explicitly closed in D146.** The preceding duplication audit recovered D108's original open question but did not follow its execution chain far enough. Its sentence describing the cost comparator as incomplete must be superseded.

[Completed D142 report](../dflash_efficiency_decision_20261006/final_review_001/RESULT_REVIEW.md) records:

| Endpoint | Resident preparation minutes | Verification calls |
|---|---:|---:|
| Frozen | 0 | 12,607 |
| D108 shorter schedule | 20.868 | 11,310 |
| Ordinary early stopping | 20.877 | 11,419 |
| Random feedback seed 1 | 20.876 | 11,366 |
| Random feedback seed 2 | 20.878 | 11,350 |
| Full ordinary updating | 27.855 | 11,219 |

All endpoints used the same 256 prompts and identical 77,143 output tokens. Short controls matched preparation within 0.046%. The proposed schedule saved 0.955% calls against early stopping and 0.352% against the stronger random control. The latter simultaneous 95% interval was [-0.610%, 1.314%], below the predeclared 2% screen. The report explicitly closes this tested efficiency claim and disallows an automatic subset/schedule/loss rescue.

The objective was the native AdaFlash backbone bundle (0.8 teacher-argmax CE + 0.2 clipped reverse KL), not the current pure forward-KL learner. This limits transport of exact outcomes to the current implementation; it does not turn a completed control into an absent one. One training seed, reused development data and conditional intervals remain qualifications.

## One-versus-two-pass work is also already completed

[Completed D156/D161 report](../dflash_async_passes_20261006/recovery_001/completed_review_001/RESULT_REVIEW.md) compares ordinary one-/two-pass asynchronous updating on the complete 7,473-request Qwen stream. One pass consumed 58 buffers versus 40, cost 20.09% less training time and left zero rather than 18 complete buffers unstarted. On common final inputs its 9,720 calls versus 9,703 for two passes differ by 0.175%, with identical outputs. One pass nevertheless had a 1.62% longer complete request window because publication/locking cost differed. The report explicitly rejects backlog alone as a novel bottleneck.

This was also the AdaFlash mixed objective. Its separate online histories and differing completed steps mean it is not an equal-step repetition experiment. It remains a competent ordinary scheduling precedent.

## Selector controls were already strong

The D136/D139 menu, reviewed in D140, included ordinary, random, recent, high-loss and GRAD-MATCH choices. Best simple controls were also best in the tested full menu. The later D142 efficiency comparison was the deliberate remaining closure test, not an unexplored opportunity.

## Recommendation for the next proposal

**Do not propose another cost-matched old-feedback/early-stop/random trial, or a generic one-/two-pass campaign, as the next discovery experiment.** No new GPU work is needed to resolve D141: its answer is retained and negative for the special schedule claim.

A pure-forward-KL replication could be scientifically valid, but it should only be run if a concrete mechanism decision depends on the objective/runtime difference. Merely using DFlash-Online instead of the historical author runner is not such a decision. We should not advertise a new control gap or promise that a loss/mechanism will follow automatically.

Any next bounded proposal must name a residual question not answered by: (a) ordinary learning works; (b) more passes/feedback can help; (c) ordinary early stopping or random feedback nearly matches the special schedule; (d) one pass resolves most learning backlog; and (e) source fitting and native transfer can disagree. A useful positive must survive the relevant ordinary alternative rather than only frozen or a deliberately over-repeated first buffer. This review deliberately does not invent a replacement GPU experiment from the disproved gap.

## Scope of this review

Read AGENTS.md, current workflow/experience entries, ARS experiment-agent workflow, D141 proposal, D142 terminal result review, D146 ledger interpretation, D156/D161 completed report, and the October 24 duplication audit. No claim that all historical reports have been exhaustively audited. Source facts are distinguished from the recommendation not to reopen this branch.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_next_experiment_review_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
