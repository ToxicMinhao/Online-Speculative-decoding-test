# Prior allocation and transfer experiments: bounded duplication audit

## Material Passport

ARS experiment-agent, validate/plan evidence audit; October 24, 2026. Read-only historical source/config/result inspection; no model reproduction, remote operation, code modification or experiment launch. This note audits the October 20–21 Llama learning-curve/allocation/transfer family against the proposed Qwen fresh-versus-repeated-buffer experiment. It is not an exhaustive project-history search.

## Main finding

**Closely related matched-step allocation experiments already exist, including an even stronger Qwen full-state-fork predecessor.** The proposed experiment is not a new research question at the broad level of “more different feedback versus more repetition.” The older studies were controlled for their bounded questions; they should not be dismissed as generally unfair or as using the recent weaker Qwen optimizer configuration. The complete new campaign should not be launched unchanged merely to rediscover this broad relationship.

The parallel audit identified D108/D114 (October 5); its protocol and final report were then inspected directly for this note. It uses the same Qwen3-8B/DFlash-b16 family, an identical full model/Adam/RNG warm-state fork, repeat versus unused feedback, and ordinary fixed controls. Warm-up is 512 responses×four passes; each continuation phase has 1,024 presentations/512 optimizer steps. Complete paths have 2,048 steps. The objective is AdaFlash's 0.8 teacher-argmax CE +0.2 clipped reverse KL, not pure OSD forward KL. NN uses 11,181 calls versus RR11,531, common warm11,529 and ordinary native updating11,219, on the same 256-request panel. Unused feedback beats repetition, while NN's gain over competent ordinary updating is only 0.339% and does not pass the predeclared 2% screen. This already directly answers the broad repeat-versus-new-feedback question in a related competent regime. Its missing narrow control was ordinary training stopped equally early, not another full-budget allocation menu.

## Closest prior evidence

| Study | Existing contrast and scope | Result | What need not be rediscovered |
|---|---|---|---|
| PR9 fixed-feedback learning curve, October 20 | Six independent Llama groups, four source responses each; same continuous F→S2→S8 trajectory; 0/2/8 optimizer steps, one/four passes; four held-out requests per group | Held-out calls F=1,857, S2=1,918, S8=1,901. S8 source KL falls 59–82% across groups. S8 partly recovers S2 but remains 2.37% worse than frozen in pooled calls | More passes can improve fit without improving ordinary held-out drafting; fixed-feedback checkpoint curves already exist |
| PR10 KL/native completion | Reuses exact own-group S8 checkpoints; source native replay plus fixed held-out KL; no training | Source KL 1.630581→0.448866 and calls 1,646→1,001 (39.19% fewer); held-out KL 1.341451→1.429618 and calls 1,857→1,901 | Better source fitting can have a substantial source native benefit while failing to transfer |
| PR11 lower-LR control | Same four responses, four passes, eight steps, paired original training anchors; fresh released model/Adam/seed17; only LR 3e-4→3e-5 | Source calls 1,574; held-out calls 1,858 versus F1,857/S81,901 | Smaller updates nearly remove this small-budget harm but did not add pooled held-out benefit |
| **PR13 P8 allocation control, October 21** | Per model: **four responses×four passes** (S8) versus **eight pooled responses×two passes** (P8); both 16 presentations/eight steps; three new dataset pools versus six original independent S8s | Held-out F1,857/S81,901/P81,984; P8 is 6.84% worse than F and 4.37% worse than S8. Source P8 calls1,278 vs F1,646/S81,001 | A matched-step increase in distinct feedback has already been tested, and did not help under this allocation |
| PR14 related-instance transfer | Same S8s, twelve presealed related children; 24 generations, no training | F820→S8781 calls (4.76% fewer); GSM8K28.86% fewer, CodeAlpaca8.49% more, ShareGPT4.37% fewer | Some benefit transfers beyond exact answer repetition, but unevenly; generic “transfer is impossible” is unsupported |

PR13's GSM8K held-out calls were F298/S8319/P8352; CodeAlpaca633/637/678; ShareGPT926/945/954. All six original-group P8 aggregates worsened versus both F and S8. P8's source fit still improved versus F, although less than S8. These are descriptive reused-set outcomes, not a proof that additional coverage is harmful generally.

## Were these old controls unfair?

The inspected retained recipe and source show constant LR3e-4, native BF16 draft parameters and ordinary AdamW, betas(.9,.999), epsilon1e-8, weight decay.01, clip1, pure forward KL at T1, gamma7 weights, up to512 random anchors per response, Flex attention and response-normalized loss averaging. BF16 vocabulary/softmax/KL arithmetic precedes FP32 position weighting. This is **not** the later eight-response Qwen screen's cosine6e-4/FP32-master/decay0/64-anchor recipe.

PR9/PR13 used Llama-3.1-8B-Instruct plus DFlash-UltraChat, greedy native SGLang/FA3, block10, 1,024 tokens or EOS. Training requests were independent first requests, not conversations. Source prompts supply context; only their own responses receive loss. Exact output equality and inactive trainers during native scoring were checked.

P8 versus S8 is fair for the declared **per-model fixed-presentation/eight-step allocation** question. It does not match total computation across three P8 versus six S8 models, token exposure, or isolated diversity: ordering, pairing, repetition and available feedback all change. Reports already state those limits. Historical F/S8 timings cannot be pooled with fresh P8 timings for a speed claim; the comparison correctly uses output-preserving verification counts.

None of these controls establishes an optimized persistent OSD baseline for thousands of requests. Tiny feedback and an eight-step horizon limit applicability; that does not invalidate their within-protocol results. Conversely, a newer positive long run does not prove the earlier settings were unfair or identify which changed factor caused success.

## What the proposed experiment genuinely adds

The proposed S/R experiment uses the now-qualified **Qwen3-8B/DFlash-b16** implementation, 128 responses per event, shuffled two-pass events, four events/512 steps per branch, block16 and 2,048-token cap. It shares the first 128-step update and forks complete weights/Adam/RNG/counters. Thereafter S learns newly completed buffers while R repeatedly learns the same first buffer. Both checkpoints are evaluated on the same future inputs before those inputs enter training.

That common-state sequential intervention, working Qwen regime and marginal next-buffer evaluation were absent from PR13. **D108 already supplies the Qwen/common-state intervention**, however, so those properties do not make the new design conceptually new. The remaining differences are pure forward KL, 128-response causal arriving-buffer events, current DFlash-Online implementation and contemporaneous next-buffer timings/checkpoint panels. PR13 trained pooled examples from release and did not ask whether replacing B1 by later buffers after an identical initial adaptation helps. The larger budget relative to PR13 is meaningful: 512 steps per new branch versus eight per old model, not a tiny implementation-only variation. It is nevertheless smaller than D108's 2,048-step paths. Cross-study outcome changes cannot be assigned solely to feedback volume.

The narrow unanswered question is therefore: **under the current successful persistent-Qwen recipe, is continuing to replace the feedback buffer more useful than spending the same steps repeatedly on B1?** The existing studies already answer the broad feasibility of source fitting, some harmful marginal updates and the existence of uneven transfer.

## Implication for avoiding redundant work

Retain the old experiments as evidence and explicitly frame any later run as a replication/extension of allocation control. **Recommend pausing the proposed full 2,176-generation campaign for this evidence reconciliation rather than launching it unchanged.** Do not pitch a new generic source-fit-versus-transfer screen or a new “does more data help?” study as if D108 or PR9–PR14 did not occur. The proposed source-fit probes can document current-branch behavior, but discovering source improvement without transfer would not itself be a new mechanism insight. A new run needs a specific decision not already settled by the earlier competent controls; changing to the exact current runner alone does not automatically provide that decision value. Any scope reduction or different experiment remains a proposal for the user, not an authorized action.

## Sources inspected

- [PR9 final report](../../dflash_baseline_learning_curve_20261020/execution_001/final_review/REPORT.md), [actual author runtime](../../dflash_baseline_learning_curve_20261020/execution_001/final_review/evidence/attempt_001/native/gsm8k_00/training_worker/AUTHOR_RUNTIME.json).
- [PR10 final report](../../dflash_kl_native_diagnostic_20261020/execution_001/final_review/REPORT.md).
- [PR11 final report](../../dflash_lower_lr_control_20261020/execution_001/final_review/REPORT.md).
- [PR13 final report](../../dflash_feedback_allocation_control_20261021/execution_001/final_review/REPORT.md), [handoff contract](../../dflash_feedback_allocation_control_20261021/HANDOFF_REVIEW.md), [P8 runtime](../../dflash_feedback_allocation_control_20261021/execution_001/retrieved_final/evidence/attempt_001/RUNTIME_CONFIG.json).
- [PR14 final report](../../dflash_related_instance_transfer_20261021/execution_001/final_review/REPORT.md).
- [Author optimizer/loss source](../../dflash_two_update_followup_20261015/github_checkout/experiments/dflash_two_update_followup/author_training.py), [fixed-pool training source](../../dflash_two_update_followup_20261015/github_checkout/experiments/dflash_baseline_learning_curve/training.py), [pooled training source](../../dflash_two_update_followup_20261015/github_checkout/experiments/dflash_feedback_allocation_control/training.py).
- [Current proposed protocol](../EXPERIMENT_PLAN.md).
- [D108 protocol](../../dflash_schedule_value_20261005/PROTOCOL.md), [D114 completed schedule comparison](../../dflash_schedule_value_20261005/final_review_001/RESULT_REVIEW.md).

Statistical interpretation: no new pooled cross-study estimate, significance test, speedup or causal-mechanism claim is made. All observed regressions, selected/reused request limitations, source/held-out distinctions, and one-seed limits remain explicit. Review uses retained receipts/source, not a new GPU reproduction or rehash of all historical model tensors.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_transfer_budget_design_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
