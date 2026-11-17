# K+-ABT implementation and qualification plan

The fixed four-arm GSM8K/MathQA events-five/six pilot is implemented in DFlash-Online. This report contains **implementation evidence only**: native GPU qualification and scientific results are pending. The exact execution commit will be recorded by the sealed plan and all execution reports.

96 CPU tests pass on the stable source snapshot in `CPU_TEST_RECEIPT.json`. Coverage includes the actual tiny BF16 DFlash two-view/Adam path, selection and priority freezing, native aggregate reduction, RNG isolation, sequential graph lifetimes, neutral dispatch, native-input audit, resource fixture construction and state restoration, complete-state continuation, ledger failures, four-arm evaluation and report validity. Existing K+, first-error, objective, BF16 Adam and historical ROLE-state tests also pass. An additional same-team tiny-model diagnostic check verifies unchanged weights, Adam, RNG and observations with/without the fixed probes in all four arms. CPU checks do not establish native GPU compatibility or efficacy.

Separate same-team reviews examined the scientific contract, implementation and campaign/storage/accounting. Author overlap is disclosed; these are not independent external reviews. Review fixes enforce a separate exact sampling/RNG/counter check for active arms, actual Adam tensor-version checks between paired views, explicit failure for unsupported attention representations, and per-case work completion checks before admission. None changes the loss, radius, optimizer, panels or budgets.

## Numerical and scientific contract

- Native K+ aggregate numerator, weights and selected denominator remain unchanged.
- Clean selection, first-error priorities and denominator are fixed across both views of one presentation.
- ADV and RANDOM perturb only eligible artificial mask embeddings, with per-block nominal radius 0.01 times the eligible embedding norm.
- Native BF16 logits/probabilities/model gradients/Adam remain; perturbation buffers and norms use FP32.
- Clean backward supplies the perturbation gradient and its parameter contribution together. Two half-weight views and two separately normalized responses precede one clip/Adam step.
- CLEAN2 takes two real clean views. Its numerical difference from ordinary KPLUS is measured, not required to vanish.
- Disabled and zero-radius attacks take the direct original path and must qualify exactly.
- Temporary qualification states never initialize the pilot. Both datasets and all four arms resume their own exact original PR32 KPLUS4 state.

The user explicitly granted a scoped exception to the author's-reference-reproduction prerequisite for this adapted pilot. FreeLB's pinned paper/code were inspected; no FreeLB author reproduction is claimed. See the experiment README for source attribution and the local first-order interpretation's limits.

## Bounded work

Main:512 shared feedback generations,2,048 Adam steps,4,096 ordinary presentations,7,168 response forwards/backwards,4,096 endpoint generations. Separately:12 disposable Adam steps,68 qualification response forwards/backwards and8 serving checks;32 main startup forwards/backwards;128 fixed post-update diagnostic forwards. The four qualification feedback packets are included in512. No extra canonical generations, learning-rate/radius tuning or automatic later continuation.

The preliminary estimate is **8–10 hours after a successful launch**, based on retained native costs; qualification will provide a better estimate. Network/staging delays are outside that estimate. Main admission requires139,502,315,166 bytes of additional peak capacity under the declared rolling-retention policy, including an8GiB reserve; qualification has its own40GiB admission.

Completed FSG intermediate diagnostic weight tensors were retired under existing retention authority after compact evidence was verified. All selected final checkpoints, original KPLUS4 states, raw result records and validity evidence remain. No active dependency was substituted.

## Decision rule

Compare ADV with KPLUS, CLEAN2 and RANDOM separately on each development panel, using summed paired verification counts and per-request mean latency across repeats. Preserve harmful/null results, all failure accounting and actual extra training costs. Better perturbed loss or source fitting alone is insufficient. Mixed/unresolved results do not establish equivalence or a generally useful adversarial direction.
