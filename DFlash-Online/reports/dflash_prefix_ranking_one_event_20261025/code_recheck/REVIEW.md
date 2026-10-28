# OSD infrastructure and U/P implementation recheck

## Material Passport

ARS experiment-agent, code/reproducibility review. User asks whether U/P still use
the existing OSD infrastructure and requests a code double-check. Scope: source
review, retained failure evidence and small CPU correctness fixtures. No A800
operation, GPU qualification restart, main training, generation, production edit
or tolerance change. Two delegated internal reviewers contributed; this is not
external peer review. Reviewed source matches all105 files in the failed run seal.

## Answer

**Yes: U/P extend our qualified OSD implementation in DFlash-Online.** The full
forward-KL objective remains on all supervised positions. The addition is a
ranking penalty, with an event-fixed reference determining where it is applied.
It is not a replacement training infrastructure or a new optimizer. U/P with
lambda1 are modified OSD objectives; they should not be called unchanged OSD or
exact reproductions of published OSD/AdaFlash.

| Component | Ordinary O | U/P |
|---|---|---|
| Model/target | Released DFlash; frozen target | Same |
| Trainer | SpecForgeEventTrainer / TrainerCore | Same |
| Training strategy | OSDTrainStrategy / response mean | Same |
| Core loss | Weighted forward KL, temperature1 | Same KL plus ranking penalty |
| Optimizer | Direct BF16 AdamW | Same class and parameters |
| LR / weight decay | Constant3e-4 / .01 | Same |
| Data | Retained128 responses and teacher features | Same |
| Schedule | Two passes,256 presentations,128 steps | Same per independent arm |
| Sampling and attention | Author sampler, Flex, complete blocks | Same; retained schedule checked |
| Response loss denominator | sum(weights)+1e-6 | Same |
| Serving and verification | Qualified SGLang DFlash | Unchanged |

Only objective.py and training.py changed among the pre-existing online modules;
ranking.py and presentation_schedule.py were added. Fifteen other retained online
Python files, including optimizer, recipe, RNG, controller and native/SGLang
paths, remain byte-identical. Upstream training/inference implementations were
not modified by the ranking extension.

## Actual algorithm addition

The new event reference is an independent frozen draft copy. It evaluates the
same sampled full blocks before the trainable forward, without retaining its
autograd graph or changing sampling RNG. It marks the consecutive canonical-
matching prefix, excluding the known seed and positions after a mismatch/EOS.

The existing KL is still calculated by objective.py120–136. Lines158–164 add
weighted softplus of the strongest rival minus canonical logit. P applies this
to the selected prefix. U spreads the same total coefficient weight over all
supervised positions using one response-level rho. Neither normalizes by the
selected-prefix count. Both use current logits for the rival and differentiate
the gathered logits. Additional reference and schedule code supports this loss;
it does not replace the optimizer, serving model or feedback collection.

## Lambda-zero path and new CPU evidence

objective.py94 sets ranking_enabled false when lambda0. begin_event then creates
no reference, forward254–268 enters the ordinary path, and _chunk_loss receives
no ranking coefficients. O/U/P all select OSDTrainStrategy and DirectBF16AdamW
in training.py246–277. U/P still read EOS metadata under preserved RNG and carry
separate checkpoint-contract identities; those are the remaining setup differences.

The added audit script loads the actual archived pre-ranking objective module,
then compares archived O, current O, U0 and P0 on the same tiny BF16 DFlash fixture.
Each uses two synthetic response presentations and one AdamW step. All four
exactly match in losses, anchors, all14 draft gradient tensors, weights, optimizer
state, RNG and gradient norm. All use the same ordinary branch, with no reference.
CUDA is never initialized. This extends the prior CPU checks by comparing to the
archived objective, rather than only comparing branches of the new implementation.

This is a small eager-attention CPU correctness check. It does not reproduce the
full-size GPU/Flex backward or overturn the failed native qualification.

## What the GPU failure establishes

The saved O and U0 trials have identical recorded scalar losses, accuracy,
supervised-position exposure, norm, response order and sampled anchors. Their
post-step draft weights failed exact equality. No active ranking/reference work
occurred. Source review found no confirmed arithmetic-path difference explaining
that discrepancy, and qualification observers were symmetric for these trials.

The qualification implementation has two diagnostic limitations:

1. qualify.py377–388 constructs U0's CPU state, then asserts equality before
   saving the failing state or computing discrepancy statistics. Only the O
   state survives. Thus the magnitude and affected-coordinate count are unavailable.
2. State comparison checks draft weights before optimizer/RNG. The assertion
   stopped in draft weights, so post-step Adam/RNG equality was never checked.
   The per-trial setup checks are not an O-versus-U0 initial-state comparison.

There is also no repeated ordinary-O GPU control. Therefore the run cannot
attribute the mismatch to the U/P label, numerical backward variability, initial
state, or optimizer behavior. No confirmed production bug was identified by this
review; that is not a claim that GPU integration is qualified.

The narrow next investigation would retain initial states, gradients and both
post-step states, report all comparison results before raising, and compare an
unchanged O repeat as well as U0. That is a proposed diagnostic, not an executed
run or permission to relax the exact gate. The campaign remains stopped.

## Evidence

- [CPU audit result](CPU_SHARED_PATH.json)
- [Executable small-fixture audit](check_shared_path.py)
- [CPU log](CPU_SHARED_PATH.txt)
- [Original GPU failure report](../REPORT.md)
- [Failed qualification result](../raw/campaign_001/qualification/RESULT.json)

The reviewed production remains on A800 under `/data/sources/minhao/DFlash-Online`.
This recheck did not alter it or Fenrir's clean DFlash-Research folder.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_one_event_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
